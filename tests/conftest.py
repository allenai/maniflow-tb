"""Shared fixtures for the unit tests.

Everything runs on CPU against tiny synthetic LeRobot datasets written to ``tmp_path``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional

# Must be set before hvla is imported: keeps group_episode_tables off /tmp/franka_episode_groups.
os.environ["FRANKA_EPISODE_CACHE"] = "0"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hvla.dataloader.data_loading_utils import clear_episode_group_memo
from hvla.dataloader.franka_dataset import FRONT_CAM_KEY, GRIPPER_CAM_KEY, FrankaMultiModalDataset

FPS = 10
VIDEO_W, VIDEO_H = 64, 36

# One row = (goal.action, action.low_level_action).
Row = tuple[Optional[str], str]
Episode = Sequence[Row]


def row_base(episode: int, t: int) -> float:
    """First joint_target / joint_position value of row ``t`` in ``episode`` (never 0)."""
    return float(1000 * episode + t + 1)


def joint_vector(episode: int, t: int) -> list[float]:
    base = row_base(episode, t)
    return [base + 0.01 * k for k in range(7)]


def _write_video(path: Path, n_frames: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = VIDEO_W, VIDEO_H, "yuv420p"
    for t in range(n_frames):
        img = np.full((VIDEO_H, VIDEO_W, 3), (10 * t) % 256, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def write_franka_dataset(
    root: Path,
    episodes: Sequence[Episode],
    *,
    is_valid: Optional[Sequence[Sequence[bool]]] = None,
    with_goal_action: bool = True,
    with_low_level_action: bool = True,
    with_is_valid: bool = True,
    with_videos: bool = False,
) -> Path:
    """Write the minimum LeRobot layout ``FrankaMultiModalDataset`` reads.

    Row ``t`` of episode ``e`` carries ``joint_target == joint_position == row_base(e, t) + 0.01 * k``,
    so tests can tell which row an output came from. Invalid rows are zero-padded like real datagen.
    """
    columns: dict[str, list[Any]] = {
        "episode_index": [], "frame_index": [], "timestamp": [],
        "observation.state.joint_position": [], "observation.state.joint_velocity": [],
        "observation.state.ee_position": [], "observation.state.gripper_position": [],
        "action.joint_target": [], "action.gripper_position": [], "action.pd_mode": [],
        "action.is_valid": [], "goal.action": [], "action.low_level_action": [],
    }
    for e, episode in enumerate(episodes):
        valid = list(is_valid[e]) if is_valid is not None else [True] * len(episode)
        assert len(valid) == len(episode)
        for t, (goal, low_level) in enumerate(episode):
            columns["episode_index"].append(e)
            columns["frame_index"].append(t)
            columns["timestamp"].append(t / FPS)
            columns["observation.state.joint_position"].append(joint_vector(e, t))
            columns["observation.state.joint_velocity"].append([0.0] * 7)
            columns["observation.state.ee_position"].append([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
            columns["observation.state.gripper_position"].append(float(t))
            columns["action.joint_target"].append(joint_vector(e, t) if valid[t] else [0.0] * 7)
            columns["action.gripper_position"].append(float(t) if valid[t] else 0.0)
            columns["action.pd_mode"].append(1.0 if valid[t] else 0.0)
            columns["action.is_valid"].append(valid[t])
            columns["goal.action"].append(goal)
            columns["action.low_level_action"].append(low_level)
    if not with_goal_action:
        del columns["goal.action"]
    if not with_low_level_action:
        del columns["action.low_level_action"]
    if not with_is_valid:
        del columns["action.is_valid"]

    schema_overrides = {
        "timestamp": pa.float32(),
        "observation.state.gripper_position": pa.float32(),
        "action.gripper_position": pa.float32(),
        "action.pd_mode": pa.float32(),
    }
    table = pa.table({k: pa.array(v, type=schema_overrides.get(k)) for k, v in columns.items()})
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_dir / "file-000.parquet")

    features: dict[str, Any] = {}
    if with_videos:
        for cam in (FRONT_CAM_KEY, GRIPPER_CAM_KEY):
            features[cam] = {"dtype": "video", "shape": [VIDEO_H, VIDEO_W, 3], "info": {"video.is_depth_map": False}}
            for e, episode in enumerate(episodes):
                _write_video(root / "videos" / cam / "chunk-000" / f"file-{e:03d}.mp4", len(episode))

    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"fps": FPS, "total_episodes": len(episodes), "features": features}))

    def band(dim: Optional[int]) -> dict[str, Any]:
        return {"q01": [-1.0] * dim, "q99": [1.0] * dim} if dim else {"q01": -1.0, "q99": 1.0}

    stats = {
        "action.joint_target": band(7),
        "action.gripper_position": band(None),
        "action.pd_mode": band(None),
        "observation.state.joint_position": band(7),
        "observation.state.joint_velocity": band(7),
        "observation.state.ee_position": band(6),
        "observation.state.gripper_position": band(None),
    }
    (meta / "stats.json").write_text(json.dumps(stats))
    return root


MakeDataset = Callable[..., Path]
BuildDataset = Callable[..., FrankaMultiModalDataset]


@pytest.fixture(scope="session", autouse=True)
def accelerate_state() -> None:
    """hvla logs through accelerate, which raises unless a PartialState exists."""
    from accelerate import PartialState

    PartialState()


@pytest.fixture(autouse=True)
def reset_episode_memo() -> Iterator[None]:
    clear_episode_group_memo()
    yield
    clear_episode_group_memo()


@pytest.fixture
def make_franka_dataset(tmp_path: Path) -> MakeDataset:
    """``make_franka_dataset(episodes, name="ds", **kwargs) -> root`` under ``tmp_path``."""

    def _make(episodes: Sequence[Episode], name: str = "ds", **kwargs: Any) -> Path:
        return write_franka_dataset(tmp_path / name, episodes, **kwargs)

    return _make


@pytest.fixture
def build_dataset() -> BuildDataset:
    """``build_dataset(root, **overrides)`` with minimal constructor defaults."""

    def _build(root: Path | Sequence[Path], **overrides: Any) -> FrankaMultiModalDataset:
        kwargs: dict[str, Any] = {
            "front_cam_key": FRONT_CAM_KEY,
            "gripper_cam_key": GRIPPER_CAM_KEY,
            "split": "train",
            "val_ratio": 0.0,
            "horizon": 4,
            "n_action_steps": 4,
            "action_mode": "abs",
            "use_normalization": False,
            "language_conditioned": True,
            "language_mode": "action_only",
        }
        kwargs.update(overrides)
        root_arg = [Path(r) for r in root] if isinstance(root, (list, tuple)) else Path(root)
        return FrankaMultiModalDataset(root_arg, **kwargs)

    return _build
