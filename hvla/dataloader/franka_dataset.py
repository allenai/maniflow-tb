"""Franka robot dataloader for ManiFlow multimodal training.

Reads LeRobot-format data (parquet + mp4 videos) with 2 cameras (front + gripper),
17-dim state (joint_pos[7] + joint_vel[7] + ee_xyz[3]) and variable-dim action,
and produces batches matching the multimodal collate format.
"""
import json
import os
import re
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
from accelerate.logging import get_logger

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

from hvla.dataloader.data_loading_utils import (
    clear_episode_group_memo,
    group_episode_tables,
    load_parquet_dataset,
    pad_repeat_last,
    read_first_row,
    table_cell,
    column_to_numpy,
    table_f32_row,
    table_f32_slice,
    table_str,
    valid_row_min_max,
)
from hvla.dataloader.episode_filtering.filter_creation_utils import complementary_dataset_filters_from_cfg
from hvla.dataloader.episode_filtering.dataset_filters import DatasetFilter
from hvla.training.trainer_utils.depth_noise import DEFAULT_DEPTH_NOISE_KWARGS

logger = get_logger(__name__)

FRONT_CAM_KEY = "observation.images.front_camera"
GRIPPER_CAM_KEY = "observation.images.gripper_camera"

DEPTH_FRONT_KEY = "observation.depth.front_camera"
DEPTH_GRIPPER_KEY = "observation.depth.gripper_camera"

DEFAULT_COLOR_JITTER = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}


def _build_depth_to_ptmap(front_cam_key: str, gripper_cam_key: str) -> dict:
    """Depth video key -> (RGB camera key, cam_name)."""
    front_depth_key = front_cam_key.replace("observation.images.", "observation.depth.")
    gripper_depth_key = gripper_cam_key.replace("observation.images.", "observation.depth.")
    return {
        front_depth_key: (front_cam_key, "front_camera"),
        gripper_depth_key: (gripper_cam_key, "gripper_camera"),
    }

MAX_DEPTH_M = 2.5

# Stored intrinsics are for 1280x720; native depth/RGB are half-res (640x360, 16:9).
RAW_INTRINSICS_REF_W = 1280
RAW_INTRINSICS_REF_H = 720


def _assert_native_16_9(native_w: int, native_h: int, cam_name: str) -> None:
    expected = RAW_INTRINSICS_REF_W / RAW_INTRINSICS_REF_H
    aspect = native_w / native_h
    if abs(aspect - expected) > 1e-3:
        raise ValueError(
            f"Expected 16:9 native resolution for {cam_name} (stored K is "
            f"{RAW_INTRINSICS_REF_W}x{RAW_INTRINSICS_REF_H}), got {native_w}x{native_h} "
            f"(aspect {aspect:.4f})"
        )

def _parse_lookback_weights(weights) -> list[tuple[float, float, float]]:
    """Parse comma-separated ``low-high_weight`` lookback ranges."""
    if weights is None or str(weights).strip().lower() in ("", "none", "null"):
        return [(0.0, 0.0, 0.2), (0.0, 1.0, 0.8)]

    if isinstance(weights, str):
        entries = [entry.strip() for entry in weights.split(",") if entry.strip()]
    else:
        entries = list(weights)
    parsed = []
    for entry in entries:
        if isinstance(entry, str):
            range_text, separator, weight_text = entry.rpartition("_")
            if not separator or "_" in range_text or "_" in weight_text:
                raise ValueError(f"Invalid lookback weight {entry!r}; expected 'low-high_weight'")
            bounds = range_text.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Invalid lookback range {range_text!r}; expected 'low-high_weight'")
            low_text, high_text = bounds
            low, high, weight = float(low_text), float(high_text), float(weight_text)
        else:
            if len(entry) != 3:
                raise ValueError("Each lookback weight must contain low, high, and weight")
            low, high, weight = (float(value) for value in entry)
        if low < 0 or high < low or weight < 0:
            raise ValueError(f"Invalid lookback range/weight: {entry!r}")
        parsed.append((low, high, weight))

    total_weight = sum(weight for _, _, weight in parsed)
    if not parsed or total_weight <= 0:
        raise ValueError("Lookback weights must contain at least one positive weight")
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Lookback weights must sum to 1.0, got {total_weight}")
    return parsed


def _sample_lookback(weights: list[tuple[float, float, float]]) -> float:
    """Sample a lookback value from parsed ``(low, high, weight)`` ranges."""
    low, high, _ = random.choices(weights, weights=[weight for _, _, weight in weights])[0]
    return random.uniform(low, high)


def _allowed_run_starts(labels: list[Optional[str]], allowed: set[str]) -> np.ndarray:
    """``out[i]`` = first index of the unbroken run of allowed labels ending at ``i``.

    ``out[i] == i`` when ``labels[i]`` is not allowed.
    """
    out = np.arange(len(labels), dtype=np.int32)
    for i in range(1, len(labels)):
        if labels[i] in allowed and labels[i - 1] in allowed:
            out[i] = out[i - 1]
    return out


def _format_lookback_weights(weights: list[tuple[float, float, float]]) -> str:
    return ",".join(f"{low:g}-{high:g}_{weight:g}" for low, high, weight in weights)


# Pixel slack when deciding whether two target_xy cells point at the same goal.
# Measured on aug11_stack2_v2: drift under an unchanged goal tops out at 5 px, while the
# nearest real goal switch is 16 px away, so 10 px sits in the empty band between them.
TARGET_MATCH_ATOL_PX = 10.0
TARGET_X_MIN_PX = 10.0
TARGET_X_MAX_PX = 630.0
TARGET_Y_MIN_PX = 0.0
TARGET_Y_MAX_PX = 330.0


def _has_out_of_range_target_coordinate(value) -> bool:
    """Whether a target cell has a finite x or y value outside its valid range."""
    if value is None:
        return False
    try:
        target_xy = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    if target_xy.size < 2:
        return False
    x, y = target_xy[:2]
    return bool(
        (np.isfinite(x) and not TARGET_X_MIN_PX <= x <= TARGET_X_MAX_PX)
        or (np.isfinite(y) and not TARGET_Y_MIN_PX <= y <= TARGET_Y_MAX_PX)
    )


def _redundant_table_move_mask(episode, timestamp, is_place, is_pick, position):
    """Rows of a redundant back-to-back table-to-table move.

    Merged ``pnp_table_table`` retries share one "place on table" pose, so a place run, a pick
    run, and another place run with a bit-identical position is a failed retry. Drop that first
    place and the pick before it. The last attempt in a chain stays. Runs are matched in
    (episode, timestamp) order. Returns (mask, number of redundant attempts).
    """
    n = len(episode)
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    order = np.lexsort((timestamp, episode))
    ep = episode[order]
    kind = np.where(is_place[order], 1, np.where(is_pick[order], 2, 0))
    starts = np.flatnonzero(np.r_[True, (ep[1:] != ep[:-1]) | (kind[1:] != kind[:-1])])
    run_ep, run_kind, run_pos = ep[starts], kind[starts], position[order][starts]

    k = np.arange(len(starts) - 2)
    redundant = k[(run_kind[k] == 1) & (run_kind[k + 1] == 2) & (run_kind[k + 2] == 1)
                  & (run_ep[k] == run_ep[k + 2]) & np.all(run_pos[k] == run_pos[k + 2], axis=1)]
    drop_run = np.zeros(len(starts), dtype=bool)
    drop_run[redundant] = True
    pick_before = redundant[redundant > 0] - 1
    drop_run[pick_before[(run_kind[pick_before] == 2) & (run_ep[pick_before] == run_ep[pick_before + 1])]] = True

    mask = np.empty(n, dtype=bool)
    mask[order] = np.repeat(drop_run, np.diff(np.r_[starts, n]))
    return mask, len(redundant)


def _drop_redundant_table_moves(episode_groups):
    """Drop redundant table-to-table rows in each episode. Returns (rows dropped, attempts)."""
    needed = ("goal.action", "goal.position", "episode_index", "timestamp")
    n_dropped = n_attempts = 0
    empty = []
    for ep, table in episode_groups.items():
        if len(table) == 0 or any(name not in table.column_names for name in needed):
            continue
        goals = table.column("goal.action")

        def is_goal(value):
            flags = pc.fill_null(pc.equal(goals, value), False)
            return np.asarray(flags.to_numpy(zero_copy_only=False), dtype=bool)

        position = column_to_numpy(table.column("goal.position"), dtype=np.float64)
        if position.ndim == 1:
            position = position.reshape(-1, 1)
        drop, n_att = _redundant_table_move_mask(
            column_to_numpy(table.column("episode_index")),
            column_to_numpy(table.column("timestamp")),
            is_goal("place on table"), is_goal("pick"), position,
        )
        n_drop = int(np.count_nonzero(drop))
        n_dropped += n_drop
        n_attempts += int(n_att)
        if n_drop == len(table):
            empty.append(ep)
        elif n_drop:
            episode_groups[ep] = table.filter(pa.array(~drop))
    for ep in empty:
        del episode_groups[ep]
    return n_dropped, n_attempts


def _targets_match(current, previous, atol: float = TARGET_MATCH_ATOL_PX) -> bool:
    """Whether two ``target_xy`` cells refer to the same goal.
    Exact equality is unusable: Bounds are TARGET_MATCH_ATOL_PX.
    """
    if current is None or previous is None:
        return current is None and previous is None
    curr = np.asarray(current, dtype=np.float64).reshape(-1)
    prev = np.asarray(previous, dtype=np.float64).reshape(-1)
    if curr.shape != prev.shape:
        return False
    return bool(np.allclose(curr, prev, rtol=0.0, atol=atol, equal_nan=True))


# Relative sampling rate per `action.low_level_action` phase. the post-release settle carries no useful signal.
DEFAULT_PHASE_SAMPLE_WEIGHTS = {
    "reorient_traj": 1.0,
    "reorient_start_traj": 0.1,
    "reorient_retract_traj": 0.1,
    "init_gripper_open_traj": 0.01,
    "noop_traj": 0.01,
    "grasp_traj": 0.01,
    "place_traj": 0.01,
    "gripper_close_traj": 0.01,
    "gripper_open_traj": 0.01,
    "lift_traj": 0.01,
    "pre_isolate_traj": 0.01,
    "isolate_traj": 0.01,
    "": 0.01
}


def _parse_phase_sample_weights(spec) -> Optional[dict[str, float]]:
    """Parse the ``phase_sample_weights`` flag into ``{phase: weight}``, or None if off.
    Accepts a dictionary or the string ``"default"`` (use defaults).
    """
    if spec is None:
        return None

    if isinstance(spec, dict):
        return spec
    if hasattr(spec, "to_dict"):
        return spec.to_dict(resolve=True)
    if isinstance(spec, str) and spec.strip().lower() == "default":
        return dict(DEFAULT_PHASE_SAMPLE_WEIGHTS)

    raise ValueError("phase_sample_weights must be a dictionary, the string 'default', or None")


def parse_target_front_crop(value) -> int:
    """Bottom-share percent of the target-centered square crop, or -1 to disable.

    The crop is always ``img_size x img_size``. Horizontally the target stays
    centered. Vertically ``value`` is the percent of the crop *below* the
    target: ``0`` is 100% above the target, ``50`` is zero-centered, ``33``
    is 67% top / 33% bottom, ``25`` is 75% top / 25% bottom. ``-1`` / ``None``
    disables the crop.
    """
    if value is None:
        return -1
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none", "null"):
            return -1
        value = int(float(text))
    else:
        value = int(value)
    if value == -1:
        return -1
    if value < 0 or value >= 100:
        raise ValueError(
            f"use_target_front_crop must be -1 (off) or 0-99 (bottom-share percent), got {value}"
        )
    return value


_DEPTH_RG8_VARIANT_RE = re.compile(r"^rg8(_[a-z0-9]+)*$")


def normalize_depth_rg8(value: str | bool | None) -> Optional[str]:
    # Accepts a variant suffix (rg8, rg8_224, rg8_55, rg8_55_224, ...), plus the
    # legacy boolean form. "55" in the name selects the 10-bit 5+5 unpack.
    if value is None or value is False:
        return None
    if value is True:
        return "rg8"
    text = str(value).strip().lower().lstrip("_")
    if text in ("", "false", "none", "null"):
        return None
    if text == "true":
        return "rg8"
    if not _DEPTH_RG8_VARIANT_RE.match(text):
        raise ValueError(
            f"Unknown depth_rg8 variant {value!r}; expected a suffix such as "
            "rg8, rg8_224, rg8_55, or rg8_55_224"
        )
    return text


def _rg8_pack55(variant: Optional[str]) -> bool:
    return variant is not None and "55" in variant.split("_")


# ---------------------------------------------------------------------------
# g12 depth videos: lossless HEVC gray12le, short GOP (8 front / 4 gripper), no B-frames,
# 224x224 (scripts/depth_conversion/reencode_to_g12_batch.py). Exact 12-bit depth like RG8 at
# ~1/3 of the bytes; a random frame decodes with PyAV in ~2.8 ms warm (RG8/torchcodec: 3.3-4 ms)
# and 3-80 ms from a cold file (RG8: 0.1-6 s, its mp4 index sits at the end of the file).
# ---------------------------------------------------------------------------
_DEPTH_G12_VARIANT_RE = re.compile(r"^g1[02](_[a-z0-9]+)*$")


def normalize_depth_g12(value: str | bool | None) -> Optional[str]:
    """Variant suffix of the g12 depth videos (e.g. ``g12_224`` -> ``<depth_key>_g12_224``), or None.

    Accepts None/False/""/"null"/"none" (off), True (``g12_224``), or a suffix string that starts
    with ``g12`` (``g12_224``, ``g12_224_ejbl``, ...).
    """
    if value is None or value is False:
        return None
    if value is True:
        return "g12_224"
    text = str(value).strip().lower().lstrip("_")
    if text in ("", "false", "none", "null"):
        return None
    if text == "true":
        return "g12_224"
    if not _DEPTH_G12_VARIANT_RE.match(text):
        raise ValueError(
            f"Unknown depth_g12 variant {value!r}; expected a suffix such as "
            "g12_224, g10_224, or g10_224_crf035"
        )
    return text


def _gray12_plane_to_uint16(frame) -> np.ndarray:
    """Exact uint16 (H, W) array of a decoded gray12le/gray16le PyAV frame, any PyAV version.

    PyAV < 14 raises on ``to_ndarray()`` for gray12le (the training image ships PyAV 12.3), so
    plane 0 is read through the buffer protocol: little-endian 16-bit words that hold the 12-bit
    value, ``line_size`` bytes per row (rows may be padded).
    """
    fmt = frame.format.name
    if not (fmt.startswith("gray12") or fmt.startswith("gray16") or fmt.startswith("gray10")
            or fmt.startswith("gray14") or fmt.startswith("gray9")):
        raise ValueError(f"expected a 16-bit-word gray frame, got pixel format {fmt!r}")
    plane = frame.planes[0]
    h, w, ls = int(frame.height), int(frame.width), int(plane.line_size)
    buf = np.frombuffer(plane, dtype=np.uint8)
    rows = buf[: h * ls].reshape(h, ls)[:, : 2 * w]
    return np.ascontiguousarray(rows).view("<u2").reshape(h, w).astype(np.uint16, copy=False)


def _decode_g12_frame_pyav(video_path, timestamp: float, fps: float | None = None) -> Optional[np.ndarray]:
    """Exact uint16 depth (mm) of the frame the dataloader would read at ``timestamp`` from a g12 video.

    Frame choice mirrors the RG8 path (`_decode_frame`: torchcodec ``seek_mode="approximate"``,
    ``get_frames_played_at([ts])``): clamp ``ts`` to ``[0, duration - 1e-6]`` and take frame
    ``floor(ts * fps)``. The parquet ``timestamp`` column is float32 and lands up to ~4e-6 s on either
    side of ``k / fps``, so this rule (not "closest frame") is what keeps RG8 and g12 reads identical.

    Made for videos with no B-frames and a short GOP: seek to the keyframe at or before the target and
    decode forward (at most GOP frames). Cost on 224x224: container open ~0.5 ms + 0.3-0.7 ms per
    decoded frame. Returns None when the file cannot be decoded.
    """
    try:
        import av  # PyAV
    except ImportError:  # pragma: no cover
        logger.warning("PyAV not available; cannot decode g12 depth")
        return None
    try:
        with av.open(str(video_path), format="mp4") as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            tb = stream.time_base
            rate = float(stream.average_rate) if stream.average_rate else float(fps or 25.0)
            ts = max(0.0, float(timestamp))
            if stream.duration is not None and tb is not None:
                dur = float(stream.duration * tb)
                if dur > 0:
                    ts = min(ts, dur - 1e-6)
            idx = int(math.floor(ts * rate))  # torchcodec approximate-mode index for a timestamp
            if tb is not None and idx > 0:
                container.seek(int((idx / rate) / tb), stream=stream, backward=True, any_frame=False)
            last = None
            for frame in container.decode(stream):
                t = frame.time
                if t is None:
                    continue
                last = frame
                if int(round(t * rate)) >= idx:  # reached the target (or the next frame if it is missing)
                    break
            if last is None:
                return None
            return _gray12_plane_to_uint16(last)
    except Exception as e:  # noqa: BLE001
        logger.warning("g12 depth decode failed: %s @ t=%.3f: %s", video_path, timestamp, e)
        return None


def _is_valid_action_flag(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(value[0]) if len(value) else False
    return bool(value)


def _row_is_zero_padded_action(table, row_idx: int, action_cols: list[str]) -> bool:
    """True when all present numeric action fields for this row are ~0."""
    checked = False
    for col in action_cols:
        if col not in table.column_names:
            continue
        value = table.column(col)[row_idx].as_py()
        if value is None:
            continue
        checked = True
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size and np.any(np.abs(arr) > 1e-8):
            return False
    return checked


def action_row_limit(
    table,
    n_rows: int,
    action_cols: Optional[list[str]] = None,
    action_mode: str = "abs",
) -> int:
    if "action.is_valid" in table.column_names:
        is_valid = table.column("action.is_valid").to_pylist()
        for i in range(n_rows - 1, -1, -1):
            if _is_valid_action_flag(is_valid[i]):
                return i + 1
        return 0
    # Absolute joint targets are never all-zero (self-collision), so an all-zero last row
    # is unambiguous terminal padding on datasets without action.is_valid.
    if action_mode != "abs":
        return n_rows
    cols = action_cols or [
        c for c in (
            "action.joint_target",
            "action.gripper_position",
            "action.pd_mode",
        )
        if c in table.column_names
    ]
    if cols and _row_is_zero_padded_action(table, n_rows - 1, cols):
        return n_rows - 1
    return n_rows


def _fold_short_runs(mask: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Fold runs shorter than ``min_run`` into the previous value."""
    out = mask.copy()
    start = 0
    for i in range(1, len(out) + 1):
        if i == len(out) or out[i] != out[start]:
            if i - start < min_run and start > 0:
                out[start:i] = out[start - 1]
            start = i
    return out


def _flip_pull_spans(ee: np.ndarray, grip: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of every flip pull in one reorient_traj clip.
    None when ``ee`` is empty, so callers can tell a missing clip from no pulls.
    """
    n = len(ee)
    if n == 0:
        return None
    step = np.diff(ee, axis=0, prepend=ee[:1])
    step[0] = 0
    dxy = np.linalg.norm(step[:, :2], axis=1)
    dz, z = step[:, 2], ee[:, 2]
    dist = np.linalg.norm(ee - ee[0], axis=1)
    close_end = 0
    if n > 3 and grip[min(n - 1, 40)] - grip[0] > 0.15:
        moved = np.flatnonzero(dist > 0.004)
        close_end = int(moved[0]) if len(moved) else n

    bursts = []
    i = close_end
    while i < n:
        if dxy[i] < 0.0025:
            i += 1
            continue
        j = i + 1
        while j < n and dxy[j] >= 0.0015:
            j += 1
        if j - i >= 4 and dxy[i:j].max() >= 0.003:
            bursts.append(j)
        i = j

    retracts = []
    i = close_end
    while i < n:
        if dz[i] <= 0.001 or dxy[i] >= 0.0025:
            i += 1
            continue
        j = i + 1
        while j < n and ((dz[j] > 0.0008 and dxy[j] < 0.003) or (z[j] - z[i] > 0.02 and dxy[j] < 0.003 and dz[j] > -0.0005)):
            j += 1
        while j > i + 1 and dz[j - 1] <= 0.0008:
            j -= 1
        if j - i >= 8 and z[min(j, n - 1)] - z[i] > 0.05 and any(0 <= i - pe <= 25 for pe in bursts):
            retracts.append((i, j))
            i = j
        else:
            i += 1

    pull = np.zeros(n, dtype=bool)
    cursor = close_end
    for a, b in retracts:
        if a - cursor >= 3:
            region = dxy[cursor:a]
            peak = float(region.max()) if len(region) else 0.0
            if peak > 0.002:
                hot = np.flatnonzero(region >= max(0.002, 0.40 * peak))
                if len(hot):
                    pull[cursor + int(hot[0]):a] = True
        cursor = b
    pull = _fold_short_runs(pull)
    hits = np.flatnonzero(pull)
    if len(hits) == 0:
        return []
    spans = []
    run_start = run_end = int(hits[0])
    for hit in hits[1:]:
        hit = int(hit)
        if hit == run_end + 1:
            run_end = hit
        else:
            spans.append((run_start, run_end))
            run_start = run_end = hit
    spans.append((run_start, run_end))
    return spans


def _flip_pull_ends(ee: np.ndarray, grip: np.ndarray) -> list[int]:
    """Last frame of every flip pull in one reorient_traj clip."""
    spans = _flip_pull_spans(ee, grip)
    if spans is None:
        return None
    return [end for _, end in spans]


def _low_level_ends(lows: list, name: str) -> list[int]:
    """Last frame of every contiguous run of ``name``."""
    ends = []
    i = 0
    while i < len(lows):
        if lows[i] == name:
            while i + 1 < len(lows) and lows[i + 1] == name:
                i += 1
            ends.append(i)
        i += 1
    return ends


def _rotate_cycle_states(lows: list, grip: np.ndarray) -> np.ndarray:
    """Execute labels for one rotate goal: 1 moving, 0 task complete, -1 stop.

    0 starts when the gripper command opens after it has closed on the block.
    -1 is only the last frame of each ``rotate_retract_traj``.
    """
    n = len(lows)
    state = np.ones(n, dtype=np.int8)
    ends = _low_level_ends(lows, "rotate_retract_traj")
    for ci, end in enumerate(ends):
        prev = 0 if ci == 0 else ends[ci - 1] + 1
        onset = None
        closed = False
        for k in range(prev, end + 1):
            if grip[k] >= 0.5:
                closed = True
            elif closed:
                onset = k
                break
        if onset is not None and onset < end:
            state[onset:end] = 0
        state[end] = -1
    return state


def _mark_until_resume(state: np.ndarray, lows: list, ends: list[int], resume_names: tuple[str, ...], value: int) -> None:
    """Set ``value`` from each completion frame until the next attempt starts."""
    n = len(lows)
    for ci, end in enumerate(ends):
        nxt = ends[ci + 1] if ci + 1 < len(ends) else n
        resume = nxt
        same = lows[end]
        for k in range(end + 1, nxt):
            if lows[k] == same:
                continue
            if lows[k] in resume_names:
                resume = k
                break
        state[end:resume] = value


def _flip_runs(lows: list) -> list[tuple[int, int, Optional[int]]]:
    """(start, end, retract_end) of every ``reorient_traj`` run, ends inclusive.

    ``retract_end`` is the last frame of the ``reorient_retract_traj`` run right after
    it, or None when the flip has no retract stage (older datasets).
    """
    n = len(lows)
    runs = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and lows[j] == lows[i]:
            j += 1
        if lows[i] == "reorient_traj":
            retract_end = None
            if j < n and lows[j] == "reorient_retract_traj":
                k = j + 1
                while k < n and lows[k] == "reorient_retract_traj":
                    k += 1
                retract_end = k - 1
            runs.append((i, j - 1, retract_end))
        i = j
    return runs


def _flip_attempt_ends(lows: list) -> list[int]:
    """Last frame of every flip attempt: end of the retract, else end of ``reorient_traj``."""
    return [end if retract_end is None else retract_end for _, end, retract_end in _flip_runs(lows)]


def _place_runs(lows: list) -> list[tuple[int, int, Optional[int]]]:
    """(start, end, retract_end) of every ``gripper_open_traj`` run, ends inclusive.

    ``retract_end`` is the last frame of the ``lift_traj`` run right after it, or None
    when the place has no retract stage (older datasets).
    """
    n = len(lows)
    runs = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and lows[j] == lows[i]:
            j += 1
        if lows[i] == "gripper_open_traj":
            retract_end = None
            if j < n and lows[j] == "lift_traj":
                k = j + 1
                while k < n and lows[k] == "lift_traj":
                    k += 1
                retract_end = k - 1
            runs.append((i, j - 1, retract_end))
        i = j
    return runs


def _place_attempt_ends(lows: list) -> list[int]:
    """Last frame of every place attempt: end of the retract, else end of ``gripper_open_traj``."""
    return [end if retract_end is None else retract_end for _, end, retract_end in _place_runs(lows)]


def _place_execute_states(lows: list) -> np.ndarray:
    """1 while placing, 0 from gripper open through the retract, -1 when the attempt ends."""
    state = np.ones(len(lows), dtype=np.int8)
    ends = _place_attempt_ends(lows)
    _mark_until_resume(state, lows, ends, ("place_traj", "place_delta_traj"), -1)
    for (start, _, _), end in zip(_place_runs(lows), ends):
        state[start:end] = 0
    return state


def _flip_execute_states(lows: list, ee: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """1 while moving, 0 after the pull, -1 when the flip attempt ends.

    With a ``reorient_retract_traj`` stage the retract is 0 and its last frame is -1.
    Without one, the pull comes from
    end-effector motion and -1 is the last ``reorient_traj`` frame.
    """
    state = np.ones(len(lows), dtype=np.int8)
    _mark_until_resume(
        state, lows, _flip_attempt_ends(lows),
        ("reorient_start_traj", "reorient_traj"), -1,
    )
    for i, end, retract_end in _flip_runs(lows):
        if retract_end is not None:
            state[end + 1:retract_end] = 0
            continue
        spans = _flip_pull_spans(ee[i:end + 1], grip[i:end + 1]) or []
        clip_end = end - i
        for si, (_, pe) in enumerate(spans):
            nxt = spans[si + 1][0] if si + 1 < len(spans) else clip_end
            if pe < nxt:
                state[i + pe:i + nxt] = 0
    return state


def _skill_execute_states(table) -> Optional[np.ndarray]:
    """Per-frame execute for pick, place, flip, and rotate.

    Other frames are 2. Every skill ends at -1. Flip is 0 after the pull until
    the attempt ends (retract stage when present). Rotate is 0 once the gripper opens, then -1 after retract.
    Place is 0 from ``gripper_open_traj`` until -1 on the last ``lift_traj`` frame (or gripper open frame).
    """
    needed = (
        "goal.action", "action.low_level_action", "action.gripper_position",
        "observation.state.ee_position", "observation.state.gripper_position",
    )
    if any(name not in table.column_names for name in needed):
        return None
    goals = table.column("goal.action").to_pylist()
    lows = table.column("action.low_level_action").to_pylist()
    grip_cmd = np.asarray(column_to_numpy(table.column("action.gripper_position"), dtype=np.float32))
    ee = np.asarray(column_to_numpy(table.column("observation.state.ee_position"), dtype=np.float64))
    obs_grip = np.asarray(column_to_numpy(table.column("observation.state.gripper_position"), dtype=np.float64))
    n = len(goals)
    grip_cmd = grip_cmd.reshape(n, -1)[:, 0] if grip_cmd.size else np.zeros(n, dtype=np.float32)
    ee = (ee.reshape(-1, 1) if ee.ndim == 1 else ee)[:, :3]
    obs_grip = obs_grip.reshape(n, -1)[:, 0] if obs_grip.size else np.zeros(n)
    state = np.full(n, 2, dtype=np.int8)
    any_skill = False
    i = 0
    while i < n:
        j = i + 1
        while j < n and goals[j] == goals[i]:
            j += 1
        g = (goals[i] or "").strip()
        if g.startswith("rotate"):
            any_skill = True
            state[i:j] = _rotate_cycle_states(lows[i:j], grip_cmd[i:j])
        elif g.startswith("flip"):
            any_skill = True
            state[i:j] = _flip_execute_states(lows[i:j], ee[i:j], obs_grip[i:j])
        elif g == "pick":
            any_skill = True
            seg = np.ones(j - i, dtype=np.int8)
            _mark_until_resume(
                seg, lows[i:j], _low_level_ends(lows[i:j], "grasp_delta_traj"),
                ("init_gripper_open_traj", "grasp_traj", "grasp_delta_traj"), -1,
            )
            state[i:j] = seg
        elif g.startswith("place"):
            any_skill = True
            state[i:j] = _place_execute_states(lows[i:j])
        i = j
    if not any_skill:
        return None
    return state


def _pause_indices_in_skill(goal: str, lows: list) -> list[int]:
    """Frames where each pick, place, flip, or rotate attempt stops (-1).

    Flip stops on the last frame of each ``reorient_retract_traj``, or of ``reorient_traj``
    when there is no retract stage. The pull itself is execute 0 and is not a stop.
    """
    g = (goal or "").strip()
    if g.startswith("flip"):
        return _flip_attempt_ends(lows)
    if g.startswith("rotate"):
        return _low_level_ends(lows, "rotate_retract_traj")
    if g == "pick":
        return _low_level_ends(lows, "grasp_delta_traj")
    if g.startswith("place"):
        return _place_attempt_ends(lows)
    return []


def _attempt_switch_frames(table) -> Optional[np.ndarray]:
    """Row indices where pick, place, flip, or rotate completes. None if columns are missing."""
    goals = table.column("goal.action").to_pylist()
    lows = table.column("action.low_level_action").to_pylist()
    n = len(goals)
    switches = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and goals[j] == goals[i]:
            j += 1
        for pause in _pause_indices_in_skill(goals[i], lows[i:j]):
            switches.append(i + pause)
        i = j
    if not switches:
        return None
    return np.asarray(switches, dtype=np.int32)


class FrankaMultiModalDataset(Dataset):
    """LeRobot-format dataset for Franka: parquet metadata + mp4 videos."""

    def __init__(self, root,
                 front_cam_key: str,
                 gripper_cam_key: str,
                 split: str = "train", val_ratio: float = 0.05,
                 horizon: int = 16, n_obs_steps: int = 1,
                 n_action_steps: int = 16,
                 camera_keys: Optional[list] = None,
                 use_normalization: bool = True, action_mode: str = "relative",
                 target_conditioned: bool = False,
                 target_sigma: float = 1.5, img_size: int = 224,
                 language_conditioned: bool = False,
                 language_mode: str = "action_goal",
                 use_unaltered_lang_action_goal: bool = True,
                 dataset_id: int = 0,
                 use_depth: bool = False,
                 depth_rg8: str | bool | None = None,
                 use_rgb_224: bool = False,
                 depth_g12: str | bool | None = None,
                 use_ft: bool = False,
                 use_previous_front_frame: bool = False,
                 previous_front_frame_lookback_weights=None,
                 freeze_on_goal_change: bool = False,
                 pause_on_attempt: bool = False,
                 use_target_front_crop: int = -1,
                 max_episodes: Optional[list] = None,
                 max_frames_per_episode: Optional[int] = None,
                 skip_first_n_steps_per_episode: Optional[int] = None,
                 joint_only_state: bool = True,
                 zero_state: bool = False,
                 dataset_filter: DatasetFilter | list[DatasetFilter] | None = None,
                 complementary_val: bool = True,
                 action_keys: Optional[list] = None,
                 dataset_ratio: Optional[list] = None,
                 phase_sample_weights=None,
                 goal_action_filter: Optional[list] = None,
                 skip_actions=None,
                 drop_redundant_table_moves: bool = False,
                 add_depth_noise: bool = False,
                 depth_noise_sigma_color_rgb: float = DEFAULT_DEPTH_NOISE_KWARGS["sigma_color_rgb"],
                 depth_noise_sigma_space: float = DEFAULT_DEPTH_NOISE_KWARGS["sigma_space"],
                 depth_noise_guide_bits: int = DEFAULT_DEPTH_NOISE_KWARGS["guide_bits"],
                 depth_noise_diameter: int = DEFAULT_DEPTH_NOISE_KWARGS["diameter"],
                 depth_noise_ideal_mix: float = DEFAULT_DEPTH_NOISE_KWARGS["ideal_mix"],
                 action_norm_band: str = "q01_q99"):
        # root can be a single path (str/Path) or a list of paths for multi-dataset
        if isinstance(root, (list, tuple)):
            self._roots = [Path(r) for r in root]
        else:
            self._roots = [Path(root)]
        self.root = self._roots[0]
        self._episode_root: dict[int, Path] = {}

        self.front_cam_key = front_cam_key
        self.gripper_cam_key = gripper_cam_key
        self._depth_to_ptmap = _build_depth_to_ptmap(front_cam_key, gripper_cam_key)
        self.split = split
        self.dataset_id = int(dataset_id)
        if dataset_filter is None:
            dataset_filter = []
        elif isinstance(dataset_filter, DatasetFilter):
            dataset_filter = [dataset_filter]
        self.dataset_filters: list[DatasetFilter] = list(dataset_filter)
        self.drop_redundant_table_moves = bool(drop_redundant_table_moves)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.use_normalization = use_normalization
        self.target_conditioned = target_conditioned
        self.language_conditioned = language_conditioned
        self.language_mode = language_mode
        self.use_unaltered_lang_action_goal = use_unaltered_lang_action_goal
        self.img_size = img_size
        self.target_sigma = target_sigma
        self.add_depth_noise = add_depth_noise and use_depth
        self.depth_noise_kwargs = {
            "sigma_color_rgb": float(depth_noise_sigma_color_rgb),
            "sigma_space": float(depth_noise_sigma_space),
            "guide_bits": int(depth_noise_guide_bits),
            "diameter": int(depth_noise_diameter),
            "ideal_mix": float(depth_noise_ideal_mix),
        }

        info_path = self.root / "meta" / "info.json"
        with open(info_path) as f:
            self.info = json.load(f)

        stats_path = self.root / "meta" / "stats.json"
        with open(stats_path) as f:
            self.stats = json.load(f)

        # Merge stats across roots: take min of q01, max of q99 for proper normalization
        for extra_root in self._roots[1:]:
            with open(extra_root / "meta" / "stats.json") as f:
                extra_stats = json.load(f)
            for key in extra_stats:
                if key not in self.stats:
                    self.stats[key] = extra_stats[key]
                    continue
                for q_key, op in [("q01", np.minimum), ("q99", np.maximum),
                                  ("min", np.minimum), ("max", np.maximum)]:
                    if q_key in self.stats[key] and q_key in extra_stats[key]:
                        a = np.array(self.stats[key][q_key], dtype=np.float64)
                        b = np.array(extra_stats[key][q_key], dtype=np.float64)
                        self.stats[key][q_key] = op(a, b).tolist()

        self.fps = self.info.get("fps", 30)
        self.total_episodes = self.info.get("total_episodes", 0)
        self.features = self.info.get("features", {})

        self.camera_keys = camera_keys or [
            k for k, v in self.features.items()
            if isinstance(v, dict) and v.get("dtype") == "video"
            and not v.get("info", {}).get("video.is_depth_map", False)
        ]

        self.use_depth = use_depth
        self.depth_rg8 = normalize_depth_rg8(depth_rg8)
        self._rg8_pack55 = _rg8_pack55(self.depth_rg8)
        # g12: lossless HEVC gray12le depth videos in <depth_key>_<variant>, decoded exactly with PyAV.
        self.depth_g12 = normalize_depth_g12(depth_g12)
        if self.depth_rg8 and self.depth_g12:
            raise ValueError(f"depth_rg8={self.depth_rg8!r} and depth_g12={self.depth_g12!r} are mutually exclusive; set one")
        self.use_rgb_224 = bool(use_rgb_224)
        if self.use_rgb_224:
            missing = []
            for root in self._roots:
                for cam_key in self.camera_keys:
                    vdir = root / "videos" / f"{cam_key}_224"
                    if not vdir.is_dir():
                        missing.append(str(vdir))
            if missing:
                raise FileNotFoundError(
                    f"use_rgb_224=True but 224 video dirs missing: {missing}. "
                    "Generate them with scripts/rgb_conversion/reencode_to_224_batch.py"
                )
            print("[FrankaDataset] RGB videos: <camera>_224")
        self._depth_keys: dict[str, str] = {}  # depth_video_key -> cam_name
        self._intrinsics: dict[str, tuple[float, float, float, float]] = {}  # cam_name -> (fx, fy, cx, cy)
        # Raw stored K (1280x720 sim render); used only for depth-noise disparity scaling.
        self._raw_intrinsics: dict[str, tuple[float, float, float, float]] = {}
        self.max_frames_per_episode = max_frames_per_episode
        self.skip_first_n_steps_per_episode = skip_first_n_steps_per_episode
        self.joint_only_state = joint_only_state
        self.zero_state = zero_state

        self._decoder_cache_size = 2
        self._decoder_cache: OrderedDict = OrderedDict()
        self._decoder_cache_pid: Optional[int] = None

        self._av_cache_size = 1
        self._av_cache: OrderedDict = OrderedDict()
        self._av_cache_pid: Optional[int] = None

        if self.use_depth:
            for depth_key, (rgb_key, cam_name) in self._depth_to_ptmap.items():
                if camera_keys is not None and not any(cam_name in k for k in camera_keys):
                    continue
                feat = self.features.get(depth_key, {})
                if isinstance(feat, dict) and feat.get("dtype") == "video":
                    self._depth_keys[depth_key] = cam_name
            if not self._depth_keys:
                raise ValueError(
                    f"use_depth=True but no depth video keys found in {info_path}. "
                    f"Available features: {list(self.features.keys())}"
                )
            depth_npy_dir = self.root / "depth_frames"
            # Raw depth: load float32 NPYs directly from source directory
            raw_depth_dir = self.info.get("raw_depth_dir")
            self._raw_depth_paths: list[Path] | None = None
            if raw_depth_dir:
                rd = Path(raw_depth_dir)
                if rd.exists():
                    self._raw_depth_paths = sorted(rd.glob("*_depth.npy"))
                    print(f"[FrankaDataset] Depth enabled (raw float32 NPYs from {rd}, {len(self._raw_depth_paths)} files) — keys: {list(self._depth_keys.keys())}")
                else:
                    print(f"[FrankaDataset] WARNING: raw_depth_dir={rd} does not exist, falling back")
            if self._raw_depth_paths is None:
                if depth_npy_dir.exists():
                    print(f"[FrankaDataset] Depth enabled (npy depth_frames + GPU backproj) — keys: {list(self._depth_keys.keys())}")
                else:
                    print(f"[FrankaDataset] Depth enabled (video decode + GPU backproj) — keys: {list(self._depth_keys.keys())}")
            if self.depth_g12:
                missing = []
                for root in self._roots:
                    for depth_key in self._depth_keys:
                        vdir = root / "videos" / f"{depth_key}_{self.depth_g12}"
                        if not vdir.is_dir():
                            missing.append(str(vdir))
                if missing:
                    raise FileNotFoundError(
                        f"depth_g12={self.depth_g12!r} but g12 depth video dirs missing: {missing}. "
                        "Generate them with scripts/depth_conversion/reencode_to_g12_batch.py"
                    )
                print(f"[FrankaDataset] Depth videos: <depth_key>_{self.depth_g12} (lossless HEVC gray12le, PyAV exact)")

        self.use_ft = use_ft
        self.single_default_task = "" if self.language_conditioned else "task"

        # Load parquet — merge across all roots with offset episode IDs
        self._episode_orig_id: dict[int, int] = {}
        self._multi_root_ds: list[tuple] | None = None
        if len(self._roots) == 1:
            dataset = load_parquet_dataset(self.root / "data")
        else:
            self._multi_root_ds = []
            ep_offset = 0
            all_col_names: set[str] = set()
            for ri, r in enumerate(self._roots):
                lazy_ds = load_parquet_dataset(r / "data")
                ep_col = lazy_ds.to_table(columns=["episode_index"]).column("episode_index")
                orig_eps = sorted(set(ep_col.to_pylist()))
                for orig_id in orig_eps:
                    new_id = orig_id + ep_offset
                    self._episode_root[new_id] = r
                    self._episode_orig_id[new_id] = orig_id
                self._multi_root_ds.append((lazy_ds, ep_offset))
                all_col_names |= set(lazy_ds.schema.names)
                ep_offset += max(orig_eps) + 1

            # Lazy dataset — don't materialize tables yet.
            # Use first root for schema probing; actual loading deferred to
            # group_episode_tables where only keep_cols are read.
            dataset = self._multi_root_ds[0][0]
            self._multi_root_all_cols = all_col_names
            print(f"[FrankaDataset] Will merge {len(self._roots)} datasets: "
                  f"{[str(r) for r in self._roots]} → {ep_offset} total episodes")

        self.custom_action_keys = action_keys
        if self.custom_action_keys:
            self.action_mode = "custom"
            self.action_identifier = None
            print(f"USING custom action keys: {self.custom_action_keys}")
        else:
            self.action_mode = action_mode
            if action_mode == "relative":
                self.action_identifier = "relative_joint_target"
            elif action_mode == "abs":
                self.action_identifier = "joint_target"
            elif action_mode == "ee":
                self.action_identifier = "ee_position"
            elif action_mode == "ee_target_7d":
                self.action_identifier = "ee_target"
            elif action_mode == "rel_ee_target_7d":
                self.action_identifier = "relative_ee_target"
            else:
                raise ValueError(f"Unsupported action mode in data loading: {action_mode}")
            print(f"USING action.{self.action_identifier} ACTIONS")

        self.freeze_on_goal_change = freeze_on_goal_change
        self.pause_on_attempt = pause_on_attempt
        self.use_previous_front_frame = use_previous_front_frame
        self.previous_front_frame_lookback_weights = _parse_lookback_weights(previous_front_frame_lookback_weights)
        if self.pause_on_attempt and self.use_previous_front_frame:
            raise ValueError("pause_on_attempt cannot be combined with use_previous_front_frame")
        if self.use_previous_front_frame:
            assert not self.custom_action_keys
            assert self.language_conditioned is True
            assert self.action_mode == "abs", "Previous front frame only verified for absolute action mode"
            logger.info("[FrankaDataset] lookback weights: %s",
                        ",".join(f"{low:g}-{high:g}_{weight:g}" for low, high, weight in self.previous_front_frame_lookback_weights))
            logger.info("[FrankaDataset] freeze_on_goal_change: %s", self.freeze_on_goal_change)
        elif self.freeze_on_goal_change:
            raise ValueError("freeze_on_goal_change requires use_previous_front_frame")

        self.use_target_front_crop = parse_target_front_crop(use_target_front_crop)
        if self.use_target_front_crop >= 0:
            assert not self.use_rgb_224, "use_target_front_crop requires native 640x360 RGB (use_rgb_224 must be off)"
            assert self.use_previous_front_frame is False, "Unclear which front frame is target, Conditioning or Action. Hence, not supported yet"
            bottom = self.use_target_front_crop
            print(f"[FrankaDataset] Target-centered {self.img_size}x{self.img_size} front-camera crop enabled (top:bottom = {100 - bottom}:{bottom})")

        # Probe schema for optional columns — use union of all column names for multi-root
        probe_schema = dataset.schema
        if self._multi_root_ds is not None:
            class _UnionSchema:
                def __init__(self, names):
                    self.names = list(names)
                    self.schema_names = self.names
            probe_schema = _UnionSchema(self._multi_root_all_cols)

        keep_cols = ["episode_index", "timestamp",
                     "observation.state.joint_position", "observation.state.joint_velocity",
                     "observation.state.ee_position", "observation.state.gripper_position",
                     "action.pd_mode"]
        if self.custom_action_keys:
            keep_cols.extend(self.custom_action_keys)
        else:
            keep_cols.extend([f"action.{self.action_identifier}", "action.gripper_position"])

        if "action.is_valid" in probe_schema.names:
            keep_cols.append("action.is_valid")
        if self.custom_action_keys:
            self._action_check_cols = list(self.custom_action_keys) + ["action.gripper_position", "action.pd_mode"]
        else:
            self._action_check_cols = [f"action.{self.action_identifier}", "action.gripper_position", "action.pd_mode"]

        _WRENCH_COL = "observation.state.wrench"
        if self.use_ft:
            if _WRENCH_COL not in probe_schema.names:
                raise ValueError(
                    f"use_ft=True but '{_WRENCH_COL}' not found in parquet. "
                    f"Available: {probe_schema.names}"
                )
            keep_cols.append(_WRENCH_COL)
        self._wrench_col = _WRENCH_COL

        self._lang_col = None
        if self.language_conditioned:
            if "goal.action" in probe_schema.names:
                keep_cols += ["goal.action"]
                self._lang_col = "goal.action"
            elif "action_goal_str" in probe_schema.names:
                keep_cols += ["action_goal_str"]
                self._lang_col = "action_goal_str"
            elif "action.low_level_action" in probe_schema.names:
                keep_cols += ["action.low_level_action"]
                self._lang_col = "action.low_level_action"
            if self._lang_col:
                print(f"[FrankaDataset] Language mode: '{self.language_mode}' — reading '{self._lang_col}' from parquet")
            else:
                print(f"[FrankaDataset] WARNING: language_conditioned but no language column found, falling back to 'task'")

        if "action.low_level_action" in probe_schema.names and "action.low_level_action" not in keep_cols:
            keep_cols += ["action.low_level_action"]
        if "goal.action" in probe_schema.names and "goal.action" not in keep_cols:
            keep_cols.append("goal.action")

        self.has_target_xy = False
        self._target_xy_col = None
        self._goal_target_xy = None
        need_target_xy = self.target_conditioned or self.language_mode == "action_goal" or self.use_target_front_crop >= 0
        if need_target_xy:
            if "goal.position" in probe_schema.names:
                self.has_target_xy = True
                self._target_xy_col = "goal.position"
                keep_cols += ["goal.position"]
                print("[FrankaDataset] Found 'goal.position' column in parquet data")
                if "goal.position_target" in probe_schema.names:
                    self._goal_target_xy = "goal.position_target"
                    keep_cols += ["goal.position_target"]
                    print("[FrankaDataset] Found 'goal.position_target' column in parquet data")

            elif "target_xy" in probe_schema.names:
                self.has_target_xy = True
                self._target_xy_col = "target_xy"
                keep_cols += ["target_xy"]
                print("[FrankaDataset] Found 'target_xy' column in parquet data")
            else:
                print("[FrankaDataset] WARNING: target_xy requested but column not found in data")
            if self.target_conditioned:
                front_cam_info = self.features.get(self.front_cam_key, {})
                cam_shape = front_cam_info.get("shape", [360, 640, 3])
                self.orig_cam_h = cam_shape[0]
                self.orig_cam_w = cam_shape[1]

        # Load place labels for the retry filter, then drop them if nothing else reads them.
        drop_goal_position_after = (
            self.drop_redundant_table_moves and "goal.position" in probe_schema.names
            and "goal.position" not in keep_cols
        )
        if drop_goal_position_after:
            keep_cols.append("goal.position")

        self._intrinsics_from_info = False
        if self.use_depth:
            info_intrinsics = self.info.get("camera_intrinsics", {})
            for depth_key, cam_name in self._depth_keys.items():
                cam_suffix = depth_key.replace("observation.depth.", "")
                intrinsics_col = f"observation.camera_intrinsics.{cam_suffix}"
                if intrinsics_col in probe_schema.names:
                    keep_cols.append(intrinsics_col)
                elif cam_suffix in info_intrinsics or cam_name in info_intrinsics:
                    self._intrinsics_from_info = True
                else:
                    raise ValueError(
                        f"use_depth=True but intrinsics not found in parquet or info.json for '{cam_suffix}'. "
                        f"Available parquet cols: {probe_schema.names}"
                    )

        cols_added_for_filter = set()
        for dataset_filter in self.dataset_filters:
            for col_name in dataset_filter.required_cols:
                if col_name not in probe_schema.names:
                    for schema_name in probe_schema.names:
                        if schema_name.lower() in col_name.lower():
                            col_name = schema_name
                            break
                    else:
                        continue

                if col_name not in keep_cols:
                    keep_cols.append(col_name)
                    cols_added_for_filter.add(col_name)

        if self.use_depth:
            if self._intrinsics_from_info:
                for depth_key, cam_name in self._depth_keys.items():
                    cam_suffix = depth_key.replace("observation.depth.", "")
                    info_key = cam_suffix if cam_suffix in info_intrinsics else cam_name
                    K = np.array(info_intrinsics[info_key], dtype=np.float32).reshape(3, 3)
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    self._raw_intrinsics[cam_name] = (fx, fy, cx, cy)
                    feat = self.features.get(depth_key, {})
                    native_shape = feat.get("shape", [])
                    if isinstance(native_shape, list) and len(native_shape) >= 2:
                        native_h, native_w = native_shape[0], native_shape[1]
                    else:
                        rgb_key = self._depth_to_ptmap[depth_key][0]
                        rgb_shape = self.features.get(rgb_key, {}).get("shape", [360, 640, 3])
                        native_h, native_w = rgb_shape[0], rgb_shape[1]
                    _assert_native_16_9(native_w, native_h, cam_name)
                    sx = self.img_size / native_w
                    sy = self.img_size / native_h
                    self._intrinsics[cam_name] = (fx * sx, fy * sy, cx * sx, cy * sy)
                    print(f"[FrankaDataset] Intrinsics {cam_name} (from info.json, rescaled to {self.img_size}x{self.img_size}): "
                          f"fx={fx*sx:.1f} fy={fy*sy:.1f} cx={cx*sx:.1f} cy={cy*sy:.1f}")
            else:
                intrinsics_cols = [f"observation.camera_intrinsics.{dk.replace('observation.depth.', '')}"
                                   for dk in self._depth_keys.keys()]
                first_row = read_first_row(dataset, intrinsics_cols)

                for depth_key, cam_name in self._depth_keys.items():
                    cam_suffix = depth_key.replace("observation.depth.", "")
                    intrinsics_col = f"observation.camera_intrinsics.{cam_suffix}"
                    K = np.array(first_row[intrinsics_col][0].as_py(), dtype=np.float32).reshape(3, 3)
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    self._raw_intrinsics[cam_name] = (fx, fy, cx, cy)
                    feat = self.features.get(depth_key, {})
                    native_shape = feat.get("shape", feat.get("info", {}).get("video.height", []))
                    depth_enc = feat.get("info", {}).get("video.depth_encoding", "dual_row_uint16_mm")
                    if isinstance(native_shape, list) and len(native_shape) >= 2:
                        native_h = native_shape[0] // 2 if depth_enc == "dual_row_uint16_mm" else native_shape[0]
                        native_w = native_shape[1]
                    else:
                        rgb_key = self._depth_to_ptmap[depth_key][0]
                        rgb_shape = self.features.get(rgb_key, {}).get("shape", [360, 640, 3])
                        native_h, native_w = rgb_shape[0], rgb_shape[1]
                    _assert_native_16_9(native_w, native_h, cam_name)
                    sx = self.img_size / native_w
                    sy = self.img_size / native_h
                    self._intrinsics[cam_name] = (fx * sx, fy * sy, cx * sx, cy * sy)
                    print(f"[FrankaDataset] Intrinsics {cam_name} (rescaled to {self.img_size}x{self.img_size}): "
                          f"fx={fx*sx:.1f} fy={fy*sy:.1f} cx={cx*sx:.1f} cy={cy*sy:.1f}")

        self.train_data_stats = {}

        # "valid_min_max" swaps the arm-action q01/q99 band for the min/max over
        # valid rows, so the clip to [-1, 1] never truncates a training label
        # (q01/q99 cut off the far ends of the reach). The swap is written into
        # self.stats, which train_hvla saves as the run's stats.json, so
        # serving denormalizes with the same band.
        if action_norm_band not in ("q01_q99", "valid_min_max"):
            raise ValueError(f"action_norm_band must be 'q01_q99' or 'valid_min_max', got {action_norm_band!r}")
        if action_norm_band == "valid_min_max":
            band_keys = list(self.custom_action_keys or [f"action.{self.action_identifier}"])
            for k in band_keys:
                bounds = [valid_row_min_max(load_parquet_dataset(r / "data"), k) for r in self._roots]
                lo = np.min(np.stack([b[0] for b in bounds]), axis=0)
                hi = np.max(np.stack([b[1] for b in bounds]), axis=0)
                s = self.stats[k]
                s["q01_quantile"], s["q99_quantile"] = s["q01"], s["q99"]
                s["q01"], s["q99"] = lo.tolist(), hi.tolist()
                s["norm_band"] = "valid_min_max"
                print(f"[FrankaDataset] {k} norm band valid_min_max: "
                      f"lo={np.round(lo, 3).tolist()} hi={np.round(hi, 3).tolist()} "
                      f"(was q01={np.round(s['q01_quantile'], 3).tolist()} q99={np.round(s['q99_quantile'], 3).tolist()})")

        if self.custom_action_keys:
            all_min, all_max = [], []
            for k in self.custom_action_keys:
                s = self.stats[k]
                all_min.append(np.array(s['q01']).astype(np.float32).reshape(-1))
                all_max.append(np.array(s['q99']).astype(np.float32).reshape(-1))
            self.train_data_stats["actions_min"] = np.concatenate(all_min)
            self.train_data_stats["actions_denominator"] = np.maximum(
                np.concatenate(all_max) - self.train_data_stats["actions_min"], 1e-6
            ).astype(np.float32)
        else:
            actions_max = np.array(self.stats[f"action.{self.action_identifier}"]['q99']).astype(np.float32)
            self.train_data_stats["actions_min"] = np.array(self.stats[f"action.{self.action_identifier}"]['q01']).astype(np.float32)
            self.train_data_stats["actions_denominator"] = np.maximum(actions_max - self.train_data_stats["actions_min"], 1e-6).astype(np.float32)

            action_gripper_max = np.float32(self.stats['action.gripper_position']['q99'])
            self.train_data_stats["actions_gripper_min"] = np.float32(self.stats['action.gripper_position']['q01'])
            self.train_data_stats["actions_gripper_denominator"] = np.maximum(action_gripper_max - self.train_data_stats["actions_gripper_min"], 1e-6).astype(np.float32)

            self.train_data_stats["actions_pd_mode_min"] = -1.0
            self.train_data_stats["actions_pd_mode_denominator"] = 1.0

        state_max = np.array(self.stats['observation.state.joint_position']['q99']).astype(np.float32)
        self.train_data_stats["state_min"] = np.array(self.stats['observation.state.joint_position']['q01']).astype(np.float32)
        self.train_data_stats["state_denominator"] = np.maximum(state_max - self.train_data_stats["state_min"], 1e-6).astype(np.float32)

        jvel_max = np.array(self.stats['observation.state.joint_velocity']['q99']).astype(np.float32)
        self.train_data_stats["jvel_min"] = np.array(self.stats['observation.state.joint_velocity']['q01']).astype(np.float32)
        self.train_data_stats["jvel_denominator"] = np.maximum(jvel_max - self.train_data_stats["jvel_min"], 1e-6).astype(np.float32)

        ee_max = np.array(self.stats['observation.state.ee_position']['q99']).astype(np.float32)
        self.train_data_stats["ee_min"] = np.array(self.stats['observation.state.ee_position']['q01']).astype(np.float32)
        self.train_data_stats["ee_denominator"] = np.maximum(ee_max - self.train_data_stats["ee_min"], 1e-6).astype(np.float32)

        gripper_max = np.float32(self.stats["observation.state.gripper_position"]['q99'])
        self.train_data_stats["gripper_min"] = np.float32(self.stats["observation.state.gripper_position"]['q01'])
        self.train_data_stats["gripper_denominator"] = np.maximum(gripper_max - self.train_data_stats["gripper_min"], 1e-6).astype(np.float32)

        if self.use_ft:
            wrench_max = np.array(self.stats[_WRENCH_COL]['q99']).astype(np.float32)
            self.train_data_stats["wrench_min"] = np.array(self.stats[_WRENCH_COL]['q01']).astype(np.float32)
            self.train_data_stats["wrench_denominator"] = np.maximum(wrench_max - self.train_data_stats["wrench_min"], 1e-6).astype(np.float32)

        if skip_actions is None or str(skip_actions).strip().lower() in (""):
            skip_actions = set()
        elif isinstance(skip_actions, str):
            skip_actions = {x.strip() for x in skip_actions.split(",") if x.strip()}
        else:
            skip_actions = set(skip_actions)
        empty_goal_action_filter = None
        skipped_empty_goal_actions = 0
        skipped_action_frames = 0
        if self._lang_col == "goal.action":
            goal_action = ds.field("goal.action")
            empty_goal_action_filter = goal_action == ""

        def goal_action_filter_for(data):
            if empty_goal_action_filter is None or "goal.action" not in data.schema.names:
                return None
            # Keep nulls; only the actual empty string is a boundary frame.
            def keep_nonempty_goal_actions(table):
                goal_action = table["goal.action"]
                return pc.or_(pc.is_null(goal_action), pc.not_equal(goal_action, ""))

            return keep_nonempty_goal_actions

        _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        _hide_progress = _rank != 0
        if self._multi_root_ds is not None:
            import pyarrow as pa
            all_groups: dict[int, pa.Table] = {}
            for lazy_ds, offset in self._multi_root_ds:
                root_keep = [c for c in keep_cols if c in lazy_ds.schema.names]
                root_filter = goal_action_filter_for(lazy_ds)
                root_groups, n_skip_act, n_skip_goal = group_episode_tables(
                    data=lazy_ds, keep_cols=root_keep,
                    skip_actions=skip_actions, additional_filter=root_filter,
                    disable_progress=_hide_progress,
                )
                skipped_action_frames += n_skip_act
                skipped_empty_goal_actions += n_skip_goal
                if offset > 0:
                    for orig_ep, tbl in root_groups.items():
                        new_ep = orig_ep + offset
                        old_col = tbl.column("episode_index")
                        new_col = pc.add(old_col, offset)
                        idx = tbl.schema.get_field_index("episode_index")
                        tbl = tbl.set_column(idx, tbl.schema.field(idx), new_col)
                        all_groups[new_ep] = tbl
                else:
                    all_groups.update(root_groups)
            self._episode_groups = all_groups
            print(f"[FrankaDataset] Merged {len(self._roots)} datasets → {len(all_groups)} episodes")
        else:
            data_filter = goal_action_filter_for(dataset)
            tables, skipped_action_frames, skipped_empty_goal_actions = group_episode_tables(
                data=dataset,
                keep_cols=keep_cols,
                skip_actions=skip_actions,
                additional_filter=data_filter,
                disable_progress=_hide_progress,
            )
            self._episode_groups = dict(tables)

        if skipped_empty_goal_actions:
            logger.warning("SKIPPED %d frames with EMPTY (!!!) goal.action labels. Ideally, this should be 0. Check Datagen", skipped_empty_goal_actions)
        logger.info("Skipped %d frames for actions: %s", skipped_action_frames, sorted(skip_actions))

        # Before episode and sample filters, so place/pick runs are still intact.
        if self.drop_redundant_table_moves:
            sample = next(iter(self._episode_groups.values()), None)
            has_goals = sample is not None and {"goal.action", "goal.position"} <= set(sample.column_names)
            if has_goals:
                before = sum(len(table) for table in self._episode_groups.values())
                n_dropped, n_attempts = _drop_redundant_table_moves(self._episode_groups)
                print(f"[FrankaDataset] dropped {n_dropped} rows from {n_attempts} "
                      f"redundant table-to-table moves ({before - n_dropped} remaining)")
            else:
                print("[FrankaDataset] drop_redundant_table_moves set but goal.action/goal.position are missing; skipped")
        if drop_goal_position_after:
            self._episode_groups = {ep: table.drop(["goal.position"]) for ep, table in self._episode_groups.items()}

        # the total number of val is defined proportional to the total number of episodes in the dataset
        # when using a filter, we want to use inlier vs outlier to define train and val
        total_eps = len(self._episode_groups)

        # Episode filters (spatial box partitioning, joint jumps, small rotates). Each drops its non-inliers.
        if self._multi_root_ds is not None:
            filter_sources = [(list(lazy_ds.files), offset) for lazy_ds, offset in self._multi_root_ds]
        else:
            filter_sources = [(list(dataset.files), 0)]
        for dataset_filter in self.dataset_filters:
            dataset_filter.bind_sources(filter_sources)
            # Scope filter to specific datasets if apply_to_datasets is set
            _atd = dataset_filter.apply_to_datasets
            if _atd is not None and len(self._roots) > 1:
                if isinstance(_atd, str):
                    _atd = [_atd]
                target_names = set(str(n) for n in _atd)
                # Match by dataset dir name or 0-based index
                target_roots = set()
                for i, r in enumerate(self._roots):
                    if r.name in target_names or str(i) in target_names:
                        target_roots.add(r)
                # Only filter episodes from targeted datasets
                eps_to_filter = {ep: tbl for ep, tbl in self._episode_groups.items()
                                 if self._episode_root.get(ep, self.root) in target_roots}
                eps_unfiltered = {ep for ep in self._episode_groups if ep not in eps_to_filter}
                print(f"[FrankaDataset] Filter scoped to datasets: {sorted(target_names)} "
                      f"({len(eps_to_filter)} episodes filtered, {len(eps_unfiltered)} kept unfiltered)")
            else:
                eps_to_filter = self._episode_groups

            inlier_eps, outlier_eps, straddler_eps = dataset_filter.apply(eps_to_filter)
            dropped = {ep for ep in eps_to_filter if ep not in inlier_eps}
            for ep in dropped:
                del self._episode_groups[ep]
            print(
                f"[FrankaDataset] {split} {dataset_filter}: "
                f"inliers={len(inlier_eps)} (kept), outliers={len(outlier_eps)} (dropped), "
                f"straddlers={len(straddler_eps)} (dropped)"
            )
        if cols_added_for_filter:
            keep_cols = set(keep_cols) - cols_added_for_filter
            for ep, table in self._episode_groups.items():
                if cols_added_for_filter.issubset(set(table.column_names)):
                    self._episode_groups[ep] = table.select(keep_cols)

        # Split episodes. Always a per-root list: 10000 or 10000,5000,2000.
        all_eps = sorted(self._episode_groups.keys())
        self.max_episodes = max_episodes

        self.split_seed = 42
        self.total_episodes = total_eps

        if complementary_val and any(f.val_is_complement for f in self.dataset_filters):
            # Train and val use complementary filters (e.g. inside vs outside),
            # so the populations are already disjoint. Train takes all inliers.
            n_val = int(total_eps * val_ratio) if val_ratio > 0 else 0
            n_train = total_eps - n_val
            if split == "val":
                self.episodes = all_eps[:n_val]
            else:
                self.episodes = all_eps[:n_train]
        else:
            # Per-root holdout: also works if there is only 1 root.
            per_root_all = {r: [] for r in self._roots}
            for ep in all_eps:
                r = self._episode_root.get(ep, self.root)
                per_root_all[r].append(ep)

            val_eps, train_eps = [], []
            for r in self._roots:
                eps = per_root_all[r]
                n_val_for_root = int(len(eps) * val_ratio) if val_ratio > 0 else 0
                val_eps.extend(eps[:n_val_for_root])
                train_eps.extend(eps[n_val_for_root:])
            self.episodes = val_eps if split == "val" else train_eps

        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(self.episodes)

        if max_episodes is not None:
            # Each cap is the combined train+validation budget for that root, so split
            # it according to val_ratio (e.g. 10,000 -> 500 val + 9,500 train).
            caps = [int(c) for c in max_episodes]
            if len(caps) != len(self._roots):
                raise ValueError(f"max_episodes has {len(caps)} caps but {len(self._roots)} dataset roots")
            per_root = {r: [] for r in self._roots}
            for ep in self.episodes:
                per_root[self._episode_root.get(ep, self.root)].append(ep)
            kept = []
            for r, cap in zip(self._roots, caps):
                eps = per_root[r]
                n_val_cap = int(cap * val_ratio) if val_ratio > 0 else 0
                split_cap = n_val_cap if split == "val" else cap - n_val_cap
                n_before = len(eps)
                if split_cap < n_before:
                    eps = eps[:split_cap]
                kept.extend(eps)
                print(f"[FrankaDataset] Subset {r.name}: {n_before}→{len(eps)}")
            self.episodes = kept

        # Balance episodes across datasets according to dataset_ratio
        if dataset_ratio is not None and len(self._roots) > 1:
            ratio = [float(r) for r in dataset_ratio]
            if len(ratio) != len(self._roots):
                raise ValueError(f"dataset_ratio length {len(ratio)} != num datasets {len(self._roots)}")
            # Group current episodes by root
            per_root_eps = {r: [] for r in self._roots}
            for ep in self.episodes:
                r = self._episode_root.get(ep, self.root)
                per_root_eps[r].append(ep)
            # Find the target count: scale so the smallest ratio group uses all its episodes
            counts = [len(per_root_eps[r]) for r in self._roots]
            max_per_unit = min(c / w for c, w in zip(counts, ratio) if w > 0)
            targets = [int(max_per_unit * w) for w in ratio]
            balanced = []
            for r, target in zip(self._roots, targets):
                eps = per_root_eps[r]
                if len(eps) > target:
                    eps = eps[:target]
                balanced.extend(eps)
            print(f"[FrankaDataset] Dataset ratio {ratio}: "
                  + ", ".join(f"{r.name}={len(per_root_eps[r])}→{t}" for r, t in zip(self._roots, targets))
                  + f" = {len(balanced)} total")
            self.episodes = balanced

        print(f"[FrankaDataset] {split} final episode IDs (up to 5): {self.episodes[:5]}")
        self._episode_groups = {ep: self._episode_groups[ep] for ep in self.episodes}

        self._attempt_switches: dict[int, np.ndarray] = {}
        self._execute_state: dict[int, np.ndarray] = {}
        if self.pause_on_attempt and not self.custom_action_keys:
            n_switches = 0
            for ep in self.episodes:
                table = self._episode_groups[ep]
                switches = _attempt_switch_frames(table)
                if switches is not None:
                    self._attempt_switches[ep] = switches
                    n_switches += len(switches)

                execute_state = _skill_execute_states(table)
                if execute_state is not None:
                    self._execute_state[ep] = execute_state
            if self._attempt_switches:
                print(f"[FrankaDataset] attempt switches on {n_switches} frames "
                      f"across {len(self._attempt_switches)} episodes")
            if self._execute_state:
                print(f"[FrankaDataset] execute state on {len(self._execute_state)} episodes")

        # Build index: (episode_id, row_idx) — allow all starting points,
        # padding with the last action when the window extends past the episode end
        self.phase_sample_weights = _parse_phase_sample_weights(phase_sample_weights)
        if self.phase_sample_weights is not None and "action.low_level_action" not in probe_schema.names:
            raise ValueError("phase_sample_weights needs the 'action.low_level_action' "
                             "column, which this dataset does not have")

        # Restrict sample *starts* to a subset of goals on ``self._lang_col``; out-of-list
        # rows stay in the table so a neighbor's action horizon can pad through them.
        # ``goal_action_filter`` accepts (a) None, (b) ``list[str]`` — one allow-list for
        # every root, or (c) ``list[list[str] | None]`` — one group per root (empty/None
        # group = no filter for that root).
        self._goal_action_filter_per_root: Optional[dict[Path, Optional[set[str]]]] = None
        if goal_action_filter:
            raw = goal_action_filter
            if not (isinstance(raw, (list, tuple)) and raw
                    and all(g is None or isinstance(g, (list, tuple, set)) for g in raw)):
                raw = [raw]  # flat list -> single group
            groups = [set(g) if g else None for g in raw]
            if len(groups) == 1:
                groups = groups * len(self._roots)
            elif len(groups) != len(self._roots):
                raise ValueError(
                    f"goal_action_filter has {len(groups)} groups but {len(self._roots)} "
                    f"dataset roots; pass 1 group (applied to all) or one per root."
                )
            if any(g is not None for g in groups):
                if self._lang_col is None:
                    raise ValueError("goal_action_filter requires a language column "
                                     "(goal.action / action_goal_str / action.low_level_action)")
                assert self.phase_sample_weights is None, \
                    "goal_action_filter and phase_sample_weights are mutually exclusive"
                self._goal_action_filter_per_root = dict(zip(self._roots, groups))

        self._samples = []
        # Parallel to _samples; consumed by WeightedRandomSampler. Empty when weighting is off.
        self.sample_weights: list[float] = []
        kept: dict[str, int] = {}
        dropped: dict[str, int] = {}
        dropped_terminal_rows = 0
        dropped_task_rows = 0
        dropped_target_rows = 0
        # Per filtered episode: earliest index previous_index may reach back to without leaving
        # the allowed goals. Only populated when use_previous_front_frame is on.
        self._allowed_run_start: dict[int, np.ndarray] = {}
        max_back = (round(self.n_action_steps * max(high for _, high, _ in self.previous_front_frame_lookback_weights))
                    if self.use_previous_front_frame else 0)
        clamped_lookback_starts = 0
        for ep in self.episodes:
            table = self._episode_groups[ep]
            n = len(table)
            start = self.skip_first_n_steps_per_episode or 0
            if self.max_frames_per_episode is not None:
                n = min(n, start + self.max_frames_per_episode)
            sample_end = action_row_limit(table, n, self._action_check_cols, self.action_mode,)
            dropped_terminal_rows += n - sample_end
            candidate_indices = range(start, sample_end)
            if self._target_xy_col is not None:
                if self._target_xy_col in table.column_names:
                    target_values = table.column(self._target_xy_col).to_pylist()
                    kept_target_indices = [
                        i for i in candidate_indices
                        if not _has_out_of_range_target_coordinate(target_values[i])
                    ]
                    dropped_target_rows += len(candidate_indices) - len(kept_target_indices)
                    candidate_indices = kept_target_indices

            ep_task_filter = (self._goal_action_filter_per_root
                              or {}).get(self._episode_root.get(ep, self.root))

            if ep_task_filter is not None:
                task_vals = table.column(self._lang_col).to_pylist()
                run_start = None
                if self.use_previous_front_frame:
                    run_start = _allowed_run_starts(task_vals, ep_task_filter)
                    self._allowed_run_start[ep] = run_start
                for i in candidate_indices:
                    if task_vals[i] in ep_task_filter:
                        self._samples.append((ep, i))
                        if run_start is not None and run_start[i] > 0 and i - run_start[i] < max_back:
                            clamped_lookback_starts += 1
                    else:
                        dropped_task_rows += 1
            elif self.phase_sample_weights is None:
                self._samples.extend((ep, i) for i in candidate_indices)
            else:
                # Zero-weight phases are dropped as window *starts* only; their rows stay in
                # the table so they can still fill the tail of a retained action horizon.
                if "action.low_level_action" not in table.column_names:
                    raise ValueError(f"phase_sample_weights is set but episode {ep} ({self._root_for_episode(ep)})"
                                     f" has no  'action.low_level_action' column")
                phases = table.column("action.low_level_action").to_pylist()
                for i in candidate_indices:
                    phase = phases[i]
                    weight = self.phase_sample_weights.get(phase, 1.0)
                    if weight > 0:
                        self._samples.append((ep, i))
                        self.sample_weights.append(weight)
                        kept[phase] = kept.get(phase, 0) + 1
                    else:
                        dropped[phase] = dropped.get(phase, 0) + 1

        # One weighted epoch draws this many samples, so a weight-1.0 phase is seen about
        # once and a weight-0.1 phase about a tenth as often as it appears in the data.
        self.effective_num_samples = (max(1, round(sum(self.sample_weights))) if self.sample_weights else None)
        if self.phase_sample_weights is not None:
            print(f"[FrankaDataset] {split}: phase sampling "
                f"{', '.join(f'{p}x{self.phase_sample_weights.get(p, 1.0):g}={c}' for p, c in sorted(kept.items()))}"
                f"{' | dropped ' + ', '.join(f'{p}={c}' for p, c in sorted(dropped.items())) if dropped else ''}"
                f" | weighted epoch = {self.effective_num_samples} draws")

        if self._goal_action_filter_per_root is not None:
            desc = "; ".join(f"{r.name}={sorted(g) if g else '*'}"
                             for r, g in self._goal_action_filter_per_root.items())
            clamp_desc = (f" | lookback clamped for up to {clamped_lookback_starts} starts"
                          if self.use_previous_front_frame else "")
            print(f"[FrankaDataset] {split}: goal_action_filter [{desc}] "
                  f"| kept {len(self._samples)} sample starts, dropped {dropped_task_rows}{clamp_desc}")

        print(f"[FrankaDataset] Excluding {dropped_terminal_rows} zero-padded terminal rows (action.is_valid when present; abs-only zero-action check otherwise)")
        if self._target_xy_col is not None:
            print(f"[FrankaDataset] Excluding {dropped_target_rows} rows with finite "
                f"{self._target_xy_col} coordinates outside "
                f"x=[{TARGET_X_MIN_PX:g}, {TARGET_X_MAX_PX:g}], "
                f"y=[{TARGET_Y_MIN_PX:g}, {TARGET_Y_MAX_PX:g}]")
        print(f"[FrankaDataset] {split}: {len(self.episodes)} episodes, {len(self._samples)} samples"
              f"{f' (skip first {self.skip_first_n_steps_per_episode})' if self.skip_first_n_steps_per_episode else ''}"
              f"{f' (truncated to {self.max_frames_per_episode} frames/ep)' if self.max_frames_per_episode else ''}")

    def __len__(self):
        return len(self._samples)

    def _root_for_episode(self, episode_index: int) -> Path:
        return self._episode_root.get(episode_index, self.root)

    def _orig_episode_index(self, episode_index: int) -> int:
        """Map global episode_index back to the original index within its dataset root."""
        return self._episode_orig_id.get(episode_index, episode_index)

    def _get_video_path(self, camera_key: str, episode_index: int) -> Path:
        orig_ep = self._orig_episode_index(episode_index)
        chunk = f"chunk-{orig_ep // 1000:03d}"
        episode_mod = orig_ep % 1000
        # RGB 224 sidecars live in <camera>_224; depth keys are unchanged.
        if self.use_rgb_224 and camera_key in self.camera_keys:
            camera_key = f"{camera_key}_224"
        return self._root_for_episode(episode_index) / "videos" / camera_key / chunk / f"file-{episode_mod:03d}.mp4"

    def _crop_centered(self, img: np.ndarray, cx: float, cy: float,
                       size: Optional[int] = None,
                       bottom_pct: Optional[int] = None) -> np.ndarray:
        """Crop a square ``size x size`` window from an HWC image.

        Horizontally the target ``(cx, cy)`` is always centered. Vertically
        ``bottom_pct`` is the percent of the crop below the target (``0`` =
        100% above, ``50`` = zero-centered). Pixel coords are in the image's
        native resolution. Regions that fall outside the frame are zero-padded.
        """
        size = size or self.img_size
        if bottom_pct is None:
            bottom_pct = self.use_target_front_crop if self.use_target_front_crop >= 0 else 50
        H, W = img.shape[:2]
        top_px = int(round(size * (100 - bottom_pct) / 100.0))
        x0 = int(round(float(cx))) - size // 2
        y0 = int(round(float(cy))) - top_px

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x0 + size - W)
        pad_bottom = max(0, y0 + size - H)

        crop = img[max(0, y0):min(H, y0 + size), max(0, x0):min(W, x0 + size)]
        if pad_left or pad_top or pad_right or pad_bottom:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            if img.ndim == 3:
                pad_width = pad_width + ((0, 0),)
            crop = np.pad(crop, pad_width, mode="constant", constant_values=0)
        if crop.shape[0] != size or crop.shape[1] != size:
            out_shape = (size, size, img.shape[2]) if img.ndim == 3 else (size, size)
            out = np.zeros(out_shape, dtype=img.dtype)
            h, w = min(size, crop.shape[0]), min(size, crop.shape[1])
            out[:h, :w] = crop[:h, :w]
            return out
        return crop

    def _build_heatmap_image(self, target_xy: np.ndarray) -> np.ndarray:
        """Build (3, img_size, img_size) Gaussian heatmap from pixel coords.

        target_xy is in original camera resolution. Returns float32 in [0, 1].
        """
        S = self.img_size
        heatmap = np.zeros((S, S), dtype=np.float32)
        if not np.isnan(target_xy[0]):
            x = target_xy[0] * S / self.orig_cam_w
            y = target_xy[1] * S / self.orig_cam_h
            sigma_px = self.target_sigma * (S / 14.0)
            yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px ** 2))
        return np.stack([heatmap, heatmap, heatmap], axis=0)

    def _get_decoder(self, video_path: Path):
        """Return a cached VideoDecoder for this path, opening it once per worker.

        ``seek_mode="approximate"`` uses header fps/duration instead of scanning
        the file to build a seek index. On these CFR dumps that mapping matches
        exact mode for timestamp lookup (see ``_decode_frame``); opening is ~50x
        faster. Cache is a per-worker LRU keyed by process id (DataLoader workers
        are separate processes, so each gets its own cache; spawned/forked
        decoders are never shared).
        """
        from torchcodec.decoders import VideoDecoder
        pid = os.getpid()
        if getattr(self, "_decoder_cache_pid", None) != pid:
            self._decoder_cache = OrderedDict()
            self._decoder_cache_pid = pid
        key = str(video_path)
        dec = self._decoder_cache.get(key)
        if dec is None:
            dec = VideoDecoder(video_path, seek_mode="approximate", num_ffmpeg_threads=1)
            self._decoder_cache[key] = dec
            if len(self._decoder_cache) > self._decoder_cache_size:
                # evict least-recently-used
                _, old_dec = self._decoder_cache.popitem(last=False)
                del old_dec
        else:
            self._decoder_cache.move_to_end(key)
        return dec

    def _decode_frame(self, video_path: Path, timestamp: float) -> Optional[np.ndarray]:
        try:
            dec = self._get_decoder(video_path)
            # begin/end_stream_seconds fall back to 0 / duration when there was
            # no exact scan (approximate mode). Keep timestamp lookup — torchcodec
            # maps pts with floor(ts * fps), matching the previous exact
            # get_frames_played_at behavior. Do not use round(ts * fps): float32
            # parquet timestamps sit just below integers and round() is 1 frame late.
            min_ts = dec.metadata.begin_stream_seconds
            max_ts = dec.metadata.end_stream_seconds
            ts = float(timestamp)
            if min_ts is not None:
                ts = max(min_ts, ts)
            if max_ts is not None:
                ts = min(max_ts - 1e-6, ts)
            frame = dec.get_frames_played_at([ts])
            data = frame.data.numpy().transpose(0, 2, 3, 1)[0]  # HWC uint8
            del frame
            return data
        except (RuntimeError, IndexError):
            logger.warning("Corrupt frame: %s @ t=%.3f — skipping", video_path, timestamp)
            return None

    def _get_av_container(self, video_path: Path):
        """Return a cached PyAV container for this path (per-worker LRU).

        torchcodec only emits RGB24, which quantizes 12-bit depth down to 8 bits.
        PyAV wraps libav directly and can emit 16-bit grayscale, so gray12le depth
        is decoded losslessly here. Cache is keyed by pid like _get_decoder so each
        DataLoader worker process owns its own containers (never shared).
        """
        import av
        pid = os.getpid()
        if getattr(self, "_av_cache_pid", None) != pid:
            self._av_cache = OrderedDict()
            self._av_cache_pid = pid
        key = str(video_path)
        container = self._av_cache.get(key)
        if container is None:
            container = av.open(key)
            self._av_cache[key] = container
            if len(self._av_cache) > self._av_cache_size:
                _, old = self._av_cache.popitem(last=False)  # evict least-recently-used
                try:
                    old.close()
                except Exception:
                    pass
        else:
            self._av_cache.move_to_end(key)
        return container

    def _decode_depth_frame_16bit(self, video_path: Path, timestamp: float) -> Optional[np.ndarray]:
        """Decode a gray12le depth frame to uint16 millimeters without precision loss.

        HEVC gray12le dumps use B-frames (GOP=2, decode-order I/I/B). Keyframe
        seek can overshoot to the *next* I-frame, so taking the first decoded
        frame with ``time >= ts`` skipped every other (B) frame. Seek a couple
        of frames early, decode forward, and keep the closest presentation
        timestamp (stop only after passing the target). Pixel data is copied
        out immediately so later PyAV frames cannot clobber the candidate.
        gray16le is scaled up from gray12le by <<4, so we shift back to real mm.
        """
        try:
            container = self._get_av_container(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else float(self.fps)
            frame_dt = 1.0 / max(fps, 1e-6)
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            ts = max(0.0, timestamp)
            if duration is not None:
                ts = min(ts, duration - 1e-6)
            # Two frames covers GOP=2 B-frames whose next I-frame is stored first.
            seek_ts = max(0.0, ts - 2.0 * frame_dt)
            offset = int(seek_ts / stream.time_base)
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            best_diff = float("inf")
            depth_mm = None
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                diff = abs(frame.time - ts)
                if diff < best_diff:
                    best_diff = diff
                    depth_mm = frame.reformat(format="gray16le").to_ndarray().astype(np.uint16) >> 4
                if frame.time > ts:
                    break
            return depth_mm  # (H, W) uint16, millimeters
        except Exception:
            logger.warning("Corrupt depth frame: %s @ t=%.3f — skipping", video_path, timestamp)
            return None

    def _decode_depth_rg8(self, video_path: Path, timestamp: float) -> Optional[np.ndarray]:
        """Decode an RG8-split depth frame.

        8+4 (default): Depth = R + G * 256
        5+5 (variant contains '55'): Depth = (R << 5) | G, 10-bit / 1023 mm
        """
        try:
            raw_rgb = self._decode_frame(video_path, timestamp)
            if raw_rgb is None:
                return None
            r = raw_rgb[:, :, 0].astype(np.uint16)
            g = raw_rgb[:, :, 1].astype(np.uint16)
            if self._rg8_pack55:
                return (r << 5) | g
            return r + g * 256
        except Exception:
            logger.warning("Corrupt RG8 depth frame: %s @ t=%.3f — skipping", video_path, timestamp)
            return None

    def _get_depth_npy_path(self, depth_key: str, episode_index: int) -> Path:
        """Return path to pre-extracted depth .npy for this episode."""
        orig_ep = self._orig_episode_index(episode_index)
        chunk = f"chunk-{orig_ep // 1000:03d}"
        episode_mod = orig_ep % 1000
        return self._root_for_episode(episode_index) / "depth_frames" / depth_key / chunk / f"file-{episode_mod:03d}.npy"

    # npy v1 header is 128 bytes; each depth frame is H*W*2 bytes (uint16).
    _NPY_HEADER_BYTES = 128
    _DEPTH_FRAME_BYTES = 224 * 224 * 2  # 100352

    def _load_depth_from_npy(self, depth_key: str, episode_index: int, frame_idx: int) -> Optional[np.ndarray]:
        """Load a single depth frame from pre-extracted NPY or raw source NPY.

        Tries pre-extracted uint16 depth_frames first, then falls back to
        raw float32 NPYs referenced by raw_depth_dir in info.json.
        """
        npy_path = self._get_depth_npy_path(depth_key, episode_index)
        if npy_path.exists():
            try:
                offset = self._NPY_HEADER_BYTES + frame_idx * self._DEPTH_FRAME_BYTES
                with open(npy_path, "rb") as f:
                    f.seek(offset)
                    raw = f.read(self._DEPTH_FRAME_BYTES)
                if len(raw) < self._DEPTH_FRAME_BYTES:
                    return None
                depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape(224, 224)
                if depth_mm.max() == 0:
                    return None
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth_m[depth_mm == 0] = MAX_DEPTH_M
                depth_m[depth_m > MAX_DEPTH_M] = MAX_DEPTH_M
                return depth_m
            except Exception:
                return None

        return self._load_raw_depth_npy(episode_index, self._depth_keys[depth_key])

    def _fx_for_depth_noise(self, cam_name: str) -> Optional[float]:
        """fx at img_size for depth noise: raw K (1280x720) scaled to img_size."""
        raw = self._raw_intrinsics.get(cam_name)
        if raw is None:
            return None
        raw_fx, _, _, _ = raw
        return raw_fx * (self.img_size / RAW_INTRINSICS_REF_W)

    def _rgb_to_gray_for_depth_noise(
        self, rgb: np.ndarray | None, depth_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if rgb is None:
            return None
        gray = np.array(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
        if gray.shape != depth_shape:
            gray = np.array(
                Image.fromarray(gray).resize(
                    (depth_shape[1], depth_shape[0]), Image.BILINEAR,
                ),
                dtype=np.uint8,
            )
        return gray

    def _maybe_augment_depth(
        self,
        depth_m: np.ndarray,
        cam_name: str,
        rgb_gray: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.add_depth_noise:
            return depth_m
        fx = self._fx_for_depth_noise(cam_name)
        if fx is None:
            return depth_m
        from hvla.training.trainer_utils.depth_noise import add_noise_to_raw_depth

        return add_noise_to_raw_depth(
            depth_m,
            fx,
            gray=rgb_gray,
            max_depth_m=MAX_DEPTH_M,
            **self.depth_noise_kwargs,
        )

    def _load_raw_depth_npy(self, episode_index: int, cam_name: str) -> Optional[np.ndarray]:
        """Load depth from raw float32 source NPY (meters, native resolution)."""
        if self._raw_depth_paths is None:
            return None
        orig_ep = self._orig_episode_index(episode_index)
        if orig_ep >= len(self._raw_depth_paths):
            return None
        try:
            depth_m = np.load(self._raw_depth_paths[orig_ep]).astype(np.float32)
            if depth_m.ndim == 3:
                depth_m = depth_m[:, :, 0]
            depth_m[~np.isfinite(depth_m)] = 0.0
            depth_m[depth_m > MAX_DEPTH_M] = 0.0
            S = self.img_size
            if depth_m.shape != (S, S):
                depth_m = np.array(
                    Image.fromarray(depth_m).resize((S, S), Image.NEAREST)
                )
            return depth_m
        except Exception:
            return None

    def _get_depth_encoding(self, depth_key: str) -> str:
        """Read depth encoding from info.json feature metadata.

        Returns 'gray12le_mm' or 'dual_row_uint16_mm' (default for legacy data).
        """
        feat = self.features.get(depth_key, {})
        info = feat.get("info", {})
        return info.get("video.depth_encoding", "dual_row_uint16_mm")

    def _decode_depth_raw(
        self, video_path: Path, timestamp: float,
        depth_key: str | None = None,
        cam_name: str | None = None,
    ) -> Optional[np.ndarray]:
        """Decode depth video frame to raw depth in meters (img_size, img_size).

        Supports gray12le (H, W) uint16 via PyAV (default), RG8-split (fast, opt-in
        via a depth_rg8 variant), g12 (lossless HEVC gray12le with a short GOP and no
        B-frames, opt-in via a depth_g12 variant; exact and fast with PyAV), and
        legacy dual-row (2H, W, 3) uint8.
        Back-projection to XYZ is deferred to the GPU obs_encoder.
        """
        encoding = self._get_depth_encoding(depth_key) if depth_key else "dual_row_uint16_mm"

        if self.depth_g12:
            depth_mm = _decode_g12_frame_pyav(video_path, timestamp, fps=self.fps)
        elif self.depth_rg8:
            depth_mm = self._decode_depth_rg8(video_path, timestamp)
        elif encoding == "gray12le_mm":
            depth_mm = self._decode_depth_frame_16bit(video_path, timestamp)
        else:
            raw = self._decode_frame(video_path, timestamp)
            if raw is None:
                return None
            H2 = raw.shape[0] // 2
            lo = raw[:H2, :, 0].astype(np.uint16)
            hi = raw[H2:, :, 0].astype(np.uint16)
            depth_mm = (hi << 8) | lo

        # Moved from above. Ideally, fail the training - why do we support None after loading?
        if depth_mm is None:
            return None

        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_m[depth_mm == 0] = 0.0
        depth_m[depth_m > MAX_DEPTH_M] = 0.0

        S = self.img_size
        if depth_m.shape != (S, S):
            depth_m = np.array(Image.fromarray(depth_m).resize((S, S), Image.NEAREST))
        if cam_name is None and depth_key is not None:
            cam_name = self._depth_keys.get(depth_key)
        return depth_m

    def __getitem__(self, idx):
        # Keep sampling another example until a decodable video is found.  This avoids
        # recursive __getitem__ calls when a video contains a corrupt frame.
        sample_count = len(self)
        candidate_idx = idx % sample_count
        while True:
            candidate_ep, candidate_start = self._samples[candidate_idx]
            candidate_table = self._episode_groups[candidate_ep]
            candidate_timestamp = float(table_cell(candidate_table, "timestamp", candidate_start))
            candidate_ok = True
            candidate_images = {}
            for candidate_cam_key in self.camera_keys:
                candidate_vpath = self._get_video_path(candidate_cam_key, candidate_ep)
                if not candidate_vpath.exists():
                    if self.use_rgb_224:
                        raise FileNotFoundError(f"RGB 224 video missing: {candidate_vpath}")
                    continue
                candidate_frame = self._decode_frame(candidate_vpath, candidate_timestamp)
                if candidate_frame is None:
                    candidate_ok = False
                    break
                candidate_images[candidate_cam_key] = candidate_frame
            if candidate_ok:
                selected_idx = candidate_idx
                images = candidate_images
                break
            candidate_idx = random.randrange(sample_count)

        ep, start = self._samples[selected_idx]
        table = self._episode_groups[ep]
        n = len(table)
        timestamp = float(table_cell(table, "timestamp", start))

        arm_state = table_f32_row(table, "observation.state.joint_position", start)
        jvel_state = table_f32_row(table, "observation.state.joint_velocity", start)
        ee_state = table_f32_row(table, "observation.state.ee_position", start)[:3]  # xyz only
        gripper_state = table_f32_row(table, "observation.state.gripper_position", start).reshape(-1)
        play_mode_state = np.ones(1, dtype=np.float32)
        if self.pause_on_attempt:
            exec_state = self._execute_state.get(ep)
            # Input is 1 or 0. -1 is a full stop and is predicted, not fed back in.
            if exec_state is not None and int(exec_state[start]) == 0:
                play_mode_state[0] = 0.0

        # Action horizon (pad by repeating last row if window extends past episode end)
        end = min(start + self.horizon, action_row_limit(table, n, self._action_check_cols, self.action_mode,))
        pad_len = self.horizon - (end - start)

        if self.custom_action_keys:
            parts = []
            for k in self.custom_action_keys:
                part = table_f32_slice(table, k, start, end)
                if part.ndim == 1:
                    part = part[:, None]
                parts.append(part)
            action = np.concatenate(parts, axis=-1)
            action = pad_repeat_last(action, pad_len)
        else:
            action_col = f"action.{self.action_identifier}"
            action = table_f32_slice(table, action_col, start, end)
            action_gripper = table_f32_slice(table, "action.gripper_position", start, end)
            if action_gripper.ndim > 1:
                action_gripper = action_gripper.reshape(action_gripper.shape[0], -1)[:, 0]
            action = pad_repeat_last(action, pad_len)
            action_gripper = pad_repeat_last(action_gripper, pad_len)
            action_play_mode = np.ones(self.horizon, dtype=np.float32)
            switches = self._attempt_switches.get(ep) if self.pause_on_attempt else None
            n_real = end - start
            if switches is not None and n_real > 0:
                idx = int(np.searchsorted(switches, start, side="left"))
                if idx < len(switches) and int(switches[idx]) < end:
                    src = int(switches[idx])
                    local = src - start
                    row = table_f32_slice(table, action_col, src, src + 1).reshape(-1)
                    grip = table_f32_slice(table, "action.gripper_position", src, src + 1).reshape(-1)[:1]
                    action = np.array(action, copy=True)
                    action_gripper = np.array(action_gripper, copy=True)
                    action[local:] = row
                    action_gripper[local:] = grip
            # Per-frame labels, then keep -1 through the rest of the chunk once it appears.
            skill = self._execute_state.get(ep) if self.pause_on_attempt else None
            if skill is not None and n_real > 0:
                sl = skill[start:end]
                action_play_mode[:n_real] = np.where(sl != 2, sl, action_play_mode[:n_real])
                if n_real < self.horizon and sl[-1] != 2:
                    action_play_mode[n_real:] = action_play_mode[n_real - 1]
                stop = np.flatnonzero(action_play_mode == -1.0)
                if len(stop):
                    action_play_mode[int(stop[0]):] = -1.0
                # The four actions before a flip's -1 open the gripper (0 = open). Input state is unchanged.
                for local in np.flatnonzero(sl == -1):
                    if table_str(table, "goal.action", start + int(local)).startswith("flip"):
                        action_gripper = np.array(action_gripper, copy=True)
                        action_gripper[max(0, int(local) - 4):int(local)] = 0.0

        if self.use_normalization:
            arm_state =  (arm_state - self.train_data_stats["state_min"]) / self.train_data_stats["state_denominator"]
            arm_state = 2 * arm_state - 1.0

            jvel_state = (jvel_state - self.train_data_stats["jvel_min"]) / self.train_data_stats["jvel_denominator"]
            jvel_state = 2 * jvel_state - 1.0

            ee_state = (ee_state - self.train_data_stats["ee_min"][:3]) / self.train_data_stats["ee_denominator"][:3]
            ee_state = 2 * ee_state - 1.0

            action = (action - self.train_data_stats["actions_min"]) / self.train_data_stats["actions_denominator"]
            action = 2 * action - 1.0
            action = np.clip(action, -1.0, 1.0)

        if self.use_previous_front_frame:
            lookback = _sample_lookback(self.previous_front_frame_lookback_weights)
            previous_index = max(0, start - round(self.n_action_steps * lookback))
            run_start = self._allowed_run_start.get(ep)
            if run_start is not None:
                previous_index = max(previous_index, int(run_start[start]))
            assert self.front_cam_key in self.camera_keys
            vpath = self._get_video_path(self.front_cam_key, ep)
            if self.use_rgb_224 and not vpath.exists():
                raise FileNotFoundError(f"RGB 224 video missing: {vpath}")
            if vpath.exists():
                previous_timestamp = float(table_cell(table, "timestamp", previous_index))
                prev_frame = self._decode_frame(vpath, previous_timestamp)
                if prev_frame is not None:
                    images["previous_front_frame"] = prev_frame
                else:
                    images["previous_front_frame"] = images.get(self.front_cam_key)

            # play_mode: 1 = execute, 0 = stop. Stop when the lookback action differs from the current one. Freeze does not gate this.
            action_changed = False
            if self.language_conditioned:
                action_changed = table_str(table, self._lang_col, start) != table_str(table, self._lang_col, previous_index)
                if action_changed:
                    action_play_mode = np.zeros(self.horizon, dtype=np.float32)

                if self.freeze_on_goal_change:
                    goal_changed = action_changed
                    if not goal_changed and self.target_conditioned and self._target_xy_col:
                        goal_changed = not _targets_match(
                            table_cell(table, self._target_xy_col, start),
                            table_cell(table, self._target_xy_col, previous_index),
                        )

                    if goal_changed:
                        hold_idx = max(0, start - 1)
                        hold_row = table_f32_slice(table, f"action.{self.action_identifier}", hold_idx, hold_idx + 1)
                        hold_gripper = table_f32_slice(table, "action.gripper_position", hold_idx, hold_idx + 1).reshape(-1)[:1]
                        if self.use_normalization:
                            hold_row = (hold_row - self.train_data_stats["actions_min"]) / self.train_data_stats["actions_denominator"]
                            hold_row = np.clip(2 * hold_row - 1.0, -1.0, 1.0)

                        action = np.repeat(hold_row, self.horizon, axis=0)
                        action_gripper = np.repeat(hold_gripper, self.horizon, axis=0)

            task_row_idx = previous_index
        else:
            task_row_idx = start

        state_parts = [arm_state, jvel_state, ee_state, gripper_state, play_mode_state]
        if self.joint_only_state:
            state_parts = [arm_state, gripper_state, play_mode_state]
        if self.use_ft:
            wrench = table_f32_row(table, self._wrench_col, start)
            if self.use_normalization:
                wrench = (wrench - self.train_data_stats["wrench_min"]) / self.train_data_stats["wrench_denominator"]
                wrench = 2 * wrench - 1.0
            state_parts.append(wrench)
        state = np.concatenate(state_parts, axis=0)
        if self.zero_state:
            state = np.zeros_like(state)
        if not self.custom_action_keys:
            action = np.concatenate([action, np.expand_dims(action_gripper, axis=1), np.expand_dims(action_play_mode, axis=1)], axis=1)

        # Language
        if self.single_default_task:
            lang = "task"
        else:
            action_str = table_str(table, self._lang_col, task_row_idx)
            if self.language_mode == "action_goal" and self._target_xy_col:
                pos = table_cell(table, self._target_xy_col, task_row_idx)
                if pos is not None:
                    pos = np.array(pos, dtype=np.float32)
                    if np.isfinite(pos).all():
                        point_strings = [action_str]
                        target_pos = table_cell(table, self._goal_target_xy, task_row_idx) if self._goal_target_xy else None

                        if self.use_unaltered_lang_action_goal:
                            point_strings.append(f"{pos[0]:.0f},{pos[1]:.0f}")
                            if target_pos is not None:
                                target_pos = np.array(target_pos, dtype=np.float32)
                                if target_pos.size >= 2 and np.isfinite(target_pos[:2]).all():
                                    point_strings.append(f"{target_pos[0]:.0f},{target_pos[1]:.0f}")
                        else:
                            # Get native resolution from front camera
                            front_cam_info = self.features.get(self.front_cam_key, {})
                            cam_shape = front_cam_info.get("shape", [360, 640, 3])
                            native_h = cam_shape[0]
                            native_w = cam_shape[1]
                            # Scale pos to img_size
                            scaled_x = pos[0] * self.img_size / native_w
                            scaled_y = pos[1] * self.img_size / native_h
                            point_strings.append(f"{scaled_x:.0f},{scaled_y:.0f}")

                            if target_pos is not None:
                                target_pos = np.array(target_pos, dtype=np.float32)
                                if target_pos.size >= 2 and np.isfinite(target_pos[:2]).all():
                                    target_x = target_pos[0] * self.img_size / native_w
                                    target_y = target_pos[1] * self.img_size / native_h
                                    point_strings.append(f"{target_x:.0f},{target_y:.0f}")

                        lang = ' '.join(point_strings)
                    else:
                        lang = action_str
                else:
                    lang = action_str
            else:
                lang = action_str
            if not lang:
                lang = "task"

        # Spatial target
        target_xy = None
        target_heatmap = None
        if self.has_target_xy and self.target_conditioned:
            target_xy = np.array(table_cell(table, self._target_xy_col, task_row_idx), dtype=np.float32)
            target_heatmap = self._build_heatmap_image(target_xy)

        if self.use_target_front_crop >= 0:
            front = images.get(self.front_cam_key)
            assert front is not None
            crop_xy = table_cell(table, self._target_xy_col, start)
            crop_xy = np.asarray(crop_xy, dtype=np.float32).reshape(-1) if crop_xy is not None else None
            if crop_xy is not None and crop_xy.size >= 2 and np.isfinite(crop_xy[:2]).all():
                images["target_front_crop"] = self._crop_centered(front, float(crop_xy[0]), float(crop_xy[1]))
            else:
                images["target_front_crop"] = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Depth loading
        depth_maps = {}
        if self.use_depth:
            frame_idx = round(timestamp * self.fps)
            for depth_key, cam_name in self._depth_keys.items():
                depth_m = self._load_depth_from_npy(depth_key, ep, frame_idx)
                if depth_m is None:
                    if self.depth_rg8:
                        depth_vpath = self._get_video_path(f"{depth_key}_{self.depth_rg8}", ep)
                    elif self.depth_g12:
                        depth_vpath = self._get_video_path(f"{depth_key}_{self.depth_g12}", ep)
                    else:
                        depth_vpath = self._get_video_path(depth_key, ep)
                    if depth_vpath.exists():
                        depth_m = self._decode_depth_raw(
                            depth_vpath, timestamp, depth_key=depth_key, cam_name=cam_name,
                        )
                if depth_m is not None:
                    if self.add_depth_noise:
                        rgb_key = self._depth_to_ptmap[depth_key][0]
                        rgb_gray = self._rgb_to_gray_for_depth_noise(
                            images.get(rgb_key), depth_m.shape,
                        )
                        depth_m = self._maybe_augment_depth(depth_m, cam_name, rgb_gray)
                    depth_maps[cam_name] = depth_m

        return {
            "images": images,
            "state": state,
            "action": action,
            "lang": lang,
            "target_xy": target_xy,
            "target_heatmap": target_heatmap,
            "depth_maps": depth_maps,
            "intrinsics": self._intrinsics if self.use_depth else {},
        }


def _pil_to_tensor(img: np.ndarray, size: int = 224) -> torch.Tensor:
    """HWC uint8 numpy -> [3, size, size] float32 tensor in [0,1].
    Skip the bilinear resize when the decoded frame is already ``size x size``
    """
    from torchvision.transforms.functional import to_tensor
    pil = Image.fromarray(img)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BILINEAR)
    return to_tensor(pil)  # [3, H, W] float32 [0,1]


class JointImageAugment:
    """ ``T.RandomCrop`` / ``T.RandomRotation`` redraw their parameters on every call,
    so they cannot express "the same crop and rotation on three tensors". Instead
    :meth:`sample` draws the geometry once per sample and :meth:`_geometric`
    replays it through ``transforms.functional``, keeping RGB pixel (u, v),
    heatmap pixel (u, v) and depth pixel (u, v) on the same scene point — which is
    what lets heatmap/pointmap token *i* correspond to RGB token *i*.
    """

    def __init__(self, img_size: int = 224, min_crop_ratio: float = 0.95,
                 rotation_degrees: float = 5.0,
                 color_jitter: Optional[dict] = None) -> None:
        self.img_size = int(img_size)
        self.min_crop_ratio = float(min_crop_ratio)
        self.rotation_degrees = float(rotation_degrees)
        self.color_jitter = T.ColorJitter(**(color_jitter or DEFAULT_COLOR_JITTER))

    def sample(self) -> tuple[int, int, int, float]:
        """Draw (top, left, crop_size, angle) once for one sample."""
        crop_ratio = random.uniform(self.min_crop_ratio, 1.0)
        crop_size = max(1, min(self.img_size, int(self.img_size * crop_ratio)))
        margin = self.img_size - crop_size
        return (random.randint(0, margin),
                random.randint(0, margin),
                crop_size,
                random.uniform(-self.rotation_degrees, self.rotation_degrees))

    def _geometric(self, img: torch.Tensor, params: tuple[int, int, int, float],
                   nearest: bool = False) -> torch.Tensor:
        top, left, crop_size, angle = params
        interp = TF.InterpolationMode.NEAREST if nearest else TF.InterpolationMode.BILINEAR
        img = TF.crop(img, top, left, crop_size, crop_size)
        img = TF.resize(img, [self.img_size, self.img_size],
                        interpolation=interp, antialias=not nearest)
        return TF.rotate(img, angle, interpolation=interp, expand=False, fill=0.0)

    def apply_rgb(self, img: torch.Tensor, params: tuple[int, int, int, float]) -> torch.Tensor:
        """[3, H, W] float32 in [0, 1] -> augmented [3, img_size, img_size]."""
        return self.color_jitter(self._geometric(img, params))

    def apply_heatmap(self, hm: torch.Tensor, params: tuple[int, int, int, float]) -> torch.Tensor:
        """[3, H, W] float32 Gaussian heatmap -> geometrically matched to RGB."""
        return self._geometric(hm, params)

    def apply_depth(self, depth: torch.Tensor, params: tuple[int, int, int, float]) -> torch.Tensor:
        """[H, W] float32 metres -> geometrically matched to RGB.

        NEAREST resampling avoids inventing depths across object boundaries, and
        the zero fill from the rotation reuses the existing "invalid" sentinel
        (downstream code already treats ``z <= 0`` as no measurement).
        """
        return self._geometric(depth.unsqueeze(0), params, nearest=True).squeeze(0)

    def obtain_warping_matrix(self, params: tuple[int, int, int, float]) -> torch.Tensor:
        """[3, 3] inverse of the augmentation matrix H (augmented pixel = H @ source pixel).

        The back-projection keeps the original K and un-projects each augmented pixel along
        the ray of its source pixel (K' = H K; 4 values cannot hold the rotation when fx != fy).
        """
        top, left, crop_size, angle = params
        s, c, t = self.img_size / crop_size, (self.img_size - 1) / 2.0, math.radians(angle)
        cos_t, sin_t = math.cos(t), math.sin(t)
        crop = torch.tensor([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=torch.float64)
        resize = torch.tensor([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1]], dtype=torch.float64)
        rotate = torch.tensor([[cos_t, sin_t, c - c * (cos_t + sin_t)],  # TF.rotate about the centre
                               [-sin_t, cos_t, c - c * (cos_t - sin_t)], [0, 0, 1]], dtype=torch.float64)
        return torch.linalg.inv(rotate @ resize @ crop).float()


def franka_multimodal_collate_fn(batch: list[dict],
                                front_cam_key: str,
                                gripper_cam_key: str,
                                include_ptmap: bool = True,
                                language_conditioned: bool = False,
                                rgb_cameras: Optional[list[str]] = None,
                                depth_cameras: Optional[list[str]] = None,
                                use_previous_front_frame: bool = False,
                                use_target_front_crop: int = -1,
                                augment: Optional[JointImageAugment] = None) -> dict:
    """Collate Franka samples into the multimodal obs format.

    ``rgb_cameras`` / ``depth_cameras`` list short camera names to include
    (e.g. ``["front_camera"]``).  When *None* both cameras are included for
    backward compatibility.

    ``augment`` (train split only) applies one geometry draw per sample across
    RGB, heatmap and depth so the modalities stay pixel-aligned.

    Returns dict with:
        obs: {<cam>_rgb, [<cam>_depth, camera_intrinsics], agent_pos, [task_name]}
        action: Tensor [B, T_a, action_dim]
        lang: list[str]
    """
    _RGB_KEY_MAP = {"front_camera": front_cam_key, "gripper_camera": gripper_cam_key}
    if rgb_cameras is None:
        rgb_cameras = ["front_camera", "gripper_camera"]
    if depth_cameras is None:
        depth_cameras = ["front_camera", "gripper_camera"]

    # One geometry draw per SAMPLE (not per batch), replayed on every modality below.
    aug_params = [augment.sample() for _ in batch] if augment is not None else None

    obs: dict = {}
    for cam_name in rgb_cameras:
        cam_key = _RGB_KEY_MAP[cam_name]
        short = cam_name.replace("_camera", "")  # front / gripper
        rgb_list = []
        for i, s in enumerate(batch):
            img = s["images"].get(cam_key)
            rgb = _pil_to_tensor(img) if img is not None else torch.zeros(3, 224, 224)
            rgb_list.append(rgb if aug_params is None
                            else augment.apply_rgb(rgb, aug_params[i]))
        obs[f"{short}_rgb"] = torch.stack(rgb_list).unsqueeze(1)  # [B,1,3,224,224]

    if use_previous_front_frame:
        prev_rgb_list = []
        for i, s in enumerate(batch):
            img = s["images"].get("previous_front_frame")
            prev = _pil_to_tensor(img) if img is not None else torch.zeros(3, 224, 224)
            prev_rgb_list.append(prev if aug_params is None
                                 else augment.apply_rgb(prev, aug_params[i]))
        obs["prev_front_rgb"] = torch.stack(prev_rgb_list).unsqueeze(1)

    if use_target_front_crop >= 0:
        crop_list = []
        for i, s in enumerate(batch):
            crop = _pil_to_tensor(s["images"].get("target_front_crop"))
            crop_list.append(crop if aug_params is None
                             else augment.apply_rgb(crop, aug_params[i]))
        obs["target_front_crop_rgb"] = torch.stack(crop_list).unsqueeze(1)

    if include_ptmap:
        B = len(batch)
        has_depth = any(s.get("depth_maps") for s in batch)

        if has_depth:
            for cam_name in depth_cameras:
                short = cam_name.replace("_camera", "")
                depth_list = []
                for i, s in enumerate(batch):
                    dm = s.get("depth_maps", {})
                    depth = (torch.from_numpy(dm[cam_name]) if cam_name in dm
                             else torch.zeros(224, 224))
                    depth_list.append(depth if aug_params is None
                                      else augment.apply_depth(depth, aug_params[i]))
                obs[f"{short}_depth"] = torch.stack(depth_list).unsqueeze(1).unsqueeze(2)  # [B,1,1,H,W]

            intrinsics = batch[0].get("intrinsics", {})
            K_list = []
            for cam in depth_cameras:
                if cam in intrinsics:
                    K_list.append(torch.tensor(intrinsics[cam], dtype=torch.float32))
                else:
                    K_list.append(torch.zeros(4, dtype=torch.float32))
            K = torch.stack(K_list)  # [n_cams, 4]
            K_batch = K.unsqueeze(0).expand(B, -1, -1).clone()  # original K
            if aug_params is not None:  # per-sample H^-1: un-project each pixel along its source pixel's ray
                H_inv = torch.stack([augment.obtain_warping_matrix(p) for p in aug_params])  # [B, 3, 3]
                obs["aug_H_inv"] = H_inv.unsqueeze(1)  # [B, 1, 3, 3]
            obs["camera_intrinsics"] = K_batch.unsqueeze(1)  # [B, 1, n_cams, 4]
        else:
            n_cams = len(depth_cameras)
            for cam_name in depth_cameras:
                short = cam_name.replace("_camera", "")
                obs[f"{short}_depth"] = torch.zeros(B, 1, 1, 224, 224)
            obs["camera_intrinsics"] = torch.zeros(B, 1, n_cams, 4)

    # State
    states = [s["state"] for s in batch]
    if states[0] is not None:
        obs["agent_pos"] = torch.from_numpy(np.stack(states)).unsqueeze(1)  # [B,1,19]

    # Spatial target
    if batch[0].get("target_xy") is not None:
        obs["target_xy"] = torch.from_numpy(
            np.stack([s["target_xy"] for s in batch])).unsqueeze(1)  # [B, 1, 2]

    # Target heatmap image — must follow the RGB geometry exactly, otherwise the
    # Gaussian peak points at a different patch than the one it marks in the image.
    if batch[0].get("target_heatmap") is not None:
        hm_list = []
        for i, s in enumerate(batch):
            hm = torch.from_numpy(s["target_heatmap"])
            hm_list.append(hm if aug_params is None
                           else augment.apply_heatmap(hm, aug_params[i]))
        obs["target_heatmap"] = torch.stack(hm_list).unsqueeze(1)  # [B, 1, 3, 224, 224]

    action = torch.from_numpy(np.stack([s["action"] for s in batch]))  # [B,T,9]
    lang = [s["lang"] for s in batch]

    if language_conditioned:
        obs["task_name"] = lang

    return {"obs": obs, "action": action, "lang": lang}


def build_franka_multimodal_dataloader(cfg):
    """Build train + val dataloaders for Franka multimodal data."""
    from functools import partial

    dcfg = cfg.datasets.vla_data
    _raw_root = dcfg.data_root_dir
    if isinstance(_raw_root, (list, tuple)):
        root = [str(r) for r in _raw_root]
    elif isinstance(_raw_root, str) and "," in _raw_root:
        root = [r.strip() for r in _raw_root.split(",") if r.strip()]
    else:
        root = str(_raw_root)
    horizon = getattr(cfg.framework, "horizon", 16)
    n_action_steps = getattr(cfg.framework, "n_action_steps", 16)
    n_obs_steps = getattr(cfg.framework, "n_obs_steps", 1)
    val_ratio = float(getattr(dcfg, "val_ratio", 0.01))
    use_depth = bool(getattr(dcfg, "use_depth", False))
    add_depth_noise = bool(getattr(dcfg, "add_depth_noise", False))
    depth_noise_sigma_color_rgb = float(
        getattr(dcfg, "depth_noise_sigma_color_rgb", DEFAULT_DEPTH_NOISE_KWARGS["sigma_color_rgb"])
    )
    depth_noise_sigma_space = float(
        getattr(dcfg, "depth_noise_sigma_space", DEFAULT_DEPTH_NOISE_KWARGS["sigma_space"])
    )
    depth_noise_guide_bits = int(
        getattr(dcfg, "depth_noise_guide_bits", DEFAULT_DEPTH_NOISE_KWARGS["guide_bits"])
    )
    depth_noise_diameter = int(
        getattr(dcfg, "depth_noise_diameter", DEFAULT_DEPTH_NOISE_KWARGS["diameter"])
    )
    depth_noise_ideal_mix = float(
        getattr(dcfg, "depth_noise_ideal_mix", DEFAULT_DEPTH_NOISE_KWARGS["ideal_mix"])
    )
    # Opt in to the RG8-split depth decoder by naming a variant
    # ("rg8", "rg8_224", "rg8_55", "rg8_55_224"); default is PyAV gray12le.
    depth_rg8 = normalize_depth_rg8(getattr(dcfg, "depth_rg8", None))
    # Opt in to the g12 depth videos (lossless HEVC gray12le, exact PyAV decode) by naming
    # a variant suffix ("g12_224"); mutually exclusive with depth_rg8.
    depth_g12 = normalize_depth_g12(getattr(dcfg, "depth_g12", None))
    use_rgb_224 = bool(getattr(dcfg, "use_rgb_224", False))
    use_ft = bool(getattr(dcfg, "use_ft", False))
    use_target_front_crop = parse_target_front_crop(getattr(cfg.framework, "use_target_front_crop", -1))
    use_previous_front_frame = bool(getattr(cfg.framework, "use_previous_front_frame", False))
    previous_front_frame_lookback_weights = getattr(dcfg, "previous_front_frame_lookback_weights", None)
    lookback_weights_spec = _format_lookback_weights(_parse_lookback_weights(previous_front_frame_lookback_weights))
    freeze_on_goal_change = bool(getattr(dcfg, "freeze_on_goal_change", False))
    pause_on_attempt = bool(getattr(dcfg, "pause_on_attempt", False))
    include_ptmap = bool(getattr(dcfg, "include_ptmap", True))
    if use_depth:
        include_ptmap = True
    num_workers = int(getattr(dcfg, "num_workers", 4))
    prefetch = int(getattr(dcfg, "prefetch_factor", 2))
    action_mode = getattr(dcfg, "action_mode", "relative")
    _tim = getattr(cfg.framework, "target_injection_mode", None)
    target_conditioned = _tim is not None and str(_tim) != "null"
    target_sigma = float(getattr(cfg.framework, "target_sigma", 1.5))
    language_conditioned = bool(getattr(cfg.framework, "language_conditioned", False))
    language_mode = str(getattr(cfg.framework, "language_mode", "action_goal"))
    use_unaltered_lang_action_goal = bool(getattr(dcfg, "use_unaltered_lang_action_goal", True))
    _max_ep = getattr(dcfg, "max_episodes", None)
    max_episodes = None if _max_ep is None or str(_max_ep).strip().lower() in ("", "null") else \
        [int(x) for x in str(_max_ep).split(",") if x.strip()]
    _max_frames = getattr(dcfg, "max_frames_per_episode", None)
    max_frames_per_episode = int(_max_frames) if _max_frames is not None and str(_max_frames) != "null" else None
    _skip_n = getattr(dcfg, "skip_first_n_steps_per_episode", None)
    skip_first_n_steps_per_episode = int(_skip_n) if _skip_n is not None and str(_skip_n) != "null" else None
    joint_only_state = bool(getattr(dcfg, "joint_only_state", True))
    zero_state = bool(getattr(dcfg, "zero_state", False))
    phase_sample_weights = getattr(dcfg, "phase_sample_weights", None)
    _skip_actions = getattr(dcfg, "skip_actions", None)
    if _skip_actions is None or str(_skip_actions).strip().lower() == "":
        skip_actions = []
    elif isinstance(_skip_actions, str):
        skip_actions = [x.strip() for x in _skip_actions.split(",") if x.strip()]
    else:
        skip_actions = list(_skip_actions)

    drop_redundant_table_moves = bool(getattr(dcfg, "drop_redundant_table_moves", False))
    _goal_action_filter = getattr(dcfg, "goal_action_filter", None)
    if _goal_action_filter is None or str(_goal_action_filter).strip().lower() in ("", "null"):
        goal_action_filter = None
    elif isinstance(_goal_action_filter, str):
        # ';' separates per-dataset groups, ',' separates tasks within a group.
        # E.g. "pick,place" (all datasets), "pick,place;stack;lift" (one per --data_root_dir),
        #      "pick,place;;lift" (middle dataset unfiltered).
        goal_action_filter = [[x.strip() for x in g.split(",") if x.strip()]
                              for g in _goal_action_filter.split(";")]
    else:
        goal_action_filter = list(_goal_action_filter)
    action_norm_band = str(getattr(dcfg, "action_norm_band", "q01_q99"))

    _action_keys_raw = getattr(dcfg, "action_keys", None)
    if _action_keys_raw is not None and str(_action_keys_raw) != "null":
        if isinstance(_action_keys_raw, str):
            action_keys = str(_action_keys_raw).split(",")
        else:
            action_keys = list(_action_keys_raw)
    else:
        action_keys = None

    _ratio_raw = getattr(dcfg, "dataset_ratio", None)
    dataset_ratio = None if _ratio_raw is None or str(_ratio_raw).strip().lower() in ("", "null") else \
        [float(x) for x in str(_ratio_raw).split(",") if x.strip()]

    if num_workers == 0:
        prefetch = None
    bs = dcfg.per_device_batch_size

    train_filters, val_filters = complementary_dataset_filters_from_cfg(dcfg)

    shape_obs = cfg.framework.shape_meta.obs
    _special_rgb = {"prev_front_rgb", "target_front_crop_rgb"}
    rgb_cameras = [k.replace("_rgb", "_camera") for k in shape_obs
                   if str(k).endswith("_rgb") and shape_obs[k] is not None
                   and k not in _special_rgb]
    depth_cameras = [k.replace("_depth", "_camera") for k in shape_obs if str(k).endswith("_depth") and shape_obs[k] is not None]
    ptmap_cameras = [k.replace("_ptmap", "_camera") for k in shape_obs if str(k).endswith("_ptmap") and shape_obs[k] is not None]
    if not rgb_cameras:
        rgb_cameras = ["front_camera", "gripper_camera"]
    if not depth_cameras and not ptmap_cameras:
        depth_cameras = rgb_cameras
    elif not depth_cameras:
        depth_cameras = ptmap_cameras

    front_cam_key = str(getattr(dcfg, "front_cam_key", FRONT_CAM_KEY))
    gripper_cam_key = str(getattr(dcfg, "gripper_cam_key", GRIPPER_CAM_KEY))
    _KEY_MAP = {"front_camera": front_cam_key, "gripper_camera": gripper_cam_key}
    explicit_cam_keys = [_KEY_MAP[c] for c in rgb_cameras]

    train_ds = FrankaMultiModalDataset(root, split="train",
                                       val_ratio=val_ratio,
                                       horizon=horizon, n_obs_steps=n_obs_steps,
                                       n_action_steps=n_action_steps,
                                       camera_keys=explicit_cam_keys,
                                       action_mode=action_mode,
                                       target_conditioned=target_conditioned,
                                       target_sigma=target_sigma,
                                       language_conditioned=language_conditioned,
                                       language_mode=language_mode,
                                       use_unaltered_lang_action_goal=use_unaltered_lang_action_goal,
                                       use_depth=use_depth,
                                       depth_rg8=depth_rg8,
                                       depth_g12=depth_g12,
                                       use_rgb_224=use_rgb_224,
                                       add_depth_noise=add_depth_noise,
                                       depth_noise_sigma_color_rgb=depth_noise_sigma_color_rgb,
                                       depth_noise_sigma_space=depth_noise_sigma_space,
                                       depth_noise_guide_bits=depth_noise_guide_bits,
                                       depth_noise_diameter=depth_noise_diameter,
                                       depth_noise_ideal_mix=depth_noise_ideal_mix,
                                       use_ft=use_ft,
                                       use_previous_front_frame=use_previous_front_frame,
                                       previous_front_frame_lookback_weights=(previous_front_frame_lookback_weights),
                                       freeze_on_goal_change=freeze_on_goal_change,
                                       pause_on_attempt=pause_on_attempt,
                                       use_target_front_crop=use_target_front_crop,
                                       max_episodes=max_episodes,
                                       dataset_filter=train_filters,
                                       max_frames_per_episode=max_frames_per_episode,
                                       skip_first_n_steps_per_episode=skip_first_n_steps_per_episode,
                                       joint_only_state=joint_only_state,
                                       zero_state=zero_state,
                                       action_keys=action_keys,
                                       dataset_ratio=dataset_ratio,
                                       phase_sample_weights=phase_sample_weights,
                                       goal_action_filter=goal_action_filter,
                                       skip_actions=skip_actions,
                                       drop_redundant_table_moves=drop_redundant_table_moves,
                                       front_cam_key=front_cam_key,
                                       gripper_cam_key=gripper_cam_key,
                                       action_norm_band=action_norm_band,
    )


    val_ds = FrankaMultiModalDataset(root,
                                     split="val",
                                     val_ratio=val_ratio,
                                     horizon=horizon, n_obs_steps=n_obs_steps,
                                     n_action_steps=n_action_steps,
                                     camera_keys=explicit_cam_keys,
                                     action_mode=action_mode,
                                     target_conditioned=target_conditioned,
                                     target_sigma=target_sigma,
                                     language_conditioned=language_conditioned,
                                     language_mode=language_mode,
                                     use_unaltered_lang_action_goal=use_unaltered_lang_action_goal,
                                     use_depth=use_depth,
                                     depth_rg8=depth_rg8,
                                     depth_g12=depth_g12,
                                     use_rgb_224=use_rgb_224,
                                     add_depth_noise=False,
                                     use_ft=use_ft,
                                     use_previous_front_frame=use_previous_front_frame,
                                     previous_front_frame_lookback_weights=(previous_front_frame_lookback_weights),
                                     freeze_on_goal_change=freeze_on_goal_change,
                                     pause_on_attempt=pause_on_attempt,
                                     use_target_front_crop=use_target_front_crop,
                                     max_episodes=max_episodes,
                                     dataset_filter=val_filters,
                                     max_frames_per_episode=max_frames_per_episode,
                                     skip_first_n_steps_per_episode=skip_first_n_steps_per_episode,
                                     joint_only_state=joint_only_state,
                                     zero_state=zero_state,
                                     action_keys=action_keys,
                                     dataset_ratio=dataset_ratio,
                                     phase_sample_weights=phase_sample_weights,
                                     goal_action_filter=goal_action_filter,
                                     skip_actions=skip_actions,
                                     drop_redundant_table_moves=drop_redundant_table_moves,
                                     front_cam_key=front_cam_key,
                                     gripper_cam_key=gripper_cam_key,
                                     action_norm_band=action_norm_band,
    )

    collate = partial(franka_multimodal_collate_fn, include_ptmap=include_ptmap,
                      language_conditioned=language_conditioned,
                      rgb_cameras=rgb_cameras, depth_cameras=depth_cameras,
                      front_cam_key=front_cam_key, gripper_cam_key=gripper_cam_key,
                      use_previous_front_frame=use_previous_front_frame,
                      use_target_front_crop=use_target_front_crop)

    # Train-split only; val never augments.
    _cj = getattr(cfg.framework, "color_jitter", None)
    train_augment = JointImageAugment(
        img_size=train_ds.img_size,
        min_crop_ratio=float(getattr(cfg.framework, "random_crop_ratio", 0.95)),
        rotation_degrees=float(getattr(cfg.framework, "random_rotation_degrees", 5.0)),
        color_jitter={k: float(v) for k, v in _cj.items()} if _cj is not None else None,
    )

    # Persist dataset metadata into config so saved runs can be reconstructed.
    cfg.datasets.vla_data.previous_front_frame_lookback_weights = lookback_weights_spec
    cfg.datasets.vla_data.freeze_on_goal_change = freeze_on_goal_change
    cfg.datasets.vla_data.pause_on_attempt = pause_on_attempt
    # Persist split metadata into config so inference scripts can reconstruct val episodes
    from omegaconf import OmegaConf
    if hasattr(cfg, "framework"):
        try:
            OmegaConf.update(cfg, "framework.use_target_front_crop", int(use_target_front_crop), force_add=True)
        except Exception:
            pass
    if hasattr(cfg, "datasets") and hasattr(cfg.datasets, "vla_data"):
        try:
            OmegaConf.update(cfg, "datasets.vla_data.split_seed", int(train_ds.split_seed), force_add=True)
            OmegaConf.update(cfg, "datasets.vla_data.total_episodes", int(train_ds.total_episodes), force_add=True)
        except Exception:
            pass

    persistent = num_workers > 0
    # Weighted sampling replaces plain shuffling on train only; val stays a deterministic order.
    # Either way the train sampler gets an explicit generator so the trainer can reseed it per
    # epoch; without one a resumed run replays the same shuffle and retrains on its head.
    train_generator = torch.Generator()
    train_generator.manual_seed(int(getattr(cfg, "seed", 3047)))
    if train_ds.sample_weights:
        train_sampler = WeightedRandomSampler(weights=train_ds.sample_weights,
                                              num_samples=train_ds.effective_num_samples, replacement=True,
                                              generator=train_generator)
    else:
        train_sampler = RandomSampler(train_ds, generator=train_generator)
    train_loader = DataLoader(train_ds, batch_size=bs,
                              collate_fn=partial(collate, augment=train_augment),
                              num_workers=num_workers, prefetch_factor=prefetch,
                              pin_memory=True,
                              sampler=train_sampler,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate,
                            num_workers=num_workers, prefetch_factor=prefetch,
                            pin_memory=True, shuffle=False,
                            persistent_workers=persistent)
    clear_episode_group_memo()

    return train_loader, val_loader
