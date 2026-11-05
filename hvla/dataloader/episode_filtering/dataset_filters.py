import json
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Sequence, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hvla.dataloader.data_loading_utils import column_to_numpy, _cache_prefix, _exclusive, _job_id
from hvla.dataloader.episode_filtering.stl_lite import always, eventually, conj

DEFAULT_SPATIAL_BOUNDS = (0.45, 0.65, -0.1, 0.1)  # (x_lo, x_hi, y_lo, y_hi)
DEFAULT_SPATIAL_OBJECT_COL = "observation.state.object_pose.block_r"
DEFAULT_SPATIAL_BOUNDS_MARGIN = 0.03

DEFAULT_FILTER_MODULE = __name__


class DatasetFilter:
    # True when train and val use opposite settings of this filter, so their episode populations are disjoint.
    val_is_complement: bool = False

    def __init__(self, **kwargs):
        self.apply_to_datasets = kwargs.pop("apply_to_datasets", None)
        self.kwargs = kwargs

    @property
    def required_cols(self) -> list[str]:
        raise NotImplementedError

    @property
    def binary_parameters(self) -> dict[str, tuple]:
        raise NotImplementedError

    def bind_sources(self, sources: list[tuple[list[str], int]]) -> None:
        """Parquet files and episode-index offset per dataset root, for filters that read parquet directly."""

    def apply(self, episode_groups: dict) -> tuple[set[int], set[int], set[int]]:
        """
        Returns (inlier_eps, outlier_eps, straddler_eps):
          - inlier_eps: object(s) always within filter constraints (all frames satisfy bounds)
          - outlier_eps: object(s) never within filter constraints (no frame satisfies bounds)
          - straddler_eps: object(s) cross the boundary (dropped for clean split)
        """
        raise NotImplementedError

    def __repr__(self):
        kw = getattr(self, "kwargs", None) or {}
        return f"{self.__class__.__name__}(**{kw})"

    @staticmethod
    def prepare_kwargs_for_super(locals_dict):
        return {k: v for k, v in locals_dict.items() if k not in ["self", "kwargs"] and not k.startswith("__")}

    def get_complement(self, param: str):
        if param not in self.binary_parameters:
            raise ValueError(f"Given {param!r} is not defined as a binary setting")

        cur_value = getattr(self, param)
        possible_values = self.binary_parameters[param]

        if cur_value == possible_values[0]:
            return possible_values[1]

        if cur_value == possible_values[1]:
            return possible_values[0]

        raise ValueError(f"Found {param}={cur_value!r} instead of one of {possible_values!r}")


class ClassifyEpisodesSpatial(DatasetFilter):
    _valid_spatial_regions = ("outside", "inside")

    def __init__(self,
        spatial_region: Literal["inside", "outside"],
        object_col: str = DEFAULT_SPATIAL_OBJECT_COL,
        bounds: Sequence[float] = DEFAULT_SPATIAL_BOUNDS,
        inside_outside_margin : float = DEFAULT_SPATIAL_BOUNDS_MARGIN,
        **kwargs: Any,
    ):
        """

        Args:
            spatial_region: inside or outside
            object_col: must exist in dataset entries
            bounds: (x_lo, x_hi, y_lo, y_hi)
            inside_outside_margin: shift bounds by half this margin in a direction according to spatial_region
        """
        super().__init__(**self.prepare_kwargs_for_super(locals()))

        self._required_cols = [object_col]

        if inside_outside_margin < 0:
            raise ValueError(f"inside_outside_margin must be >= 0 (got {inside_outside_margin})")
        self.margin = inside_outside_margin / 2

        if spatial_region not in self._valid_spatial_regions:
            raise ValueError(f"Given {spatial_region!r} is not among valid options {self._valid_spatial_regions}")
        self.spatial_region = spatial_region

        if len(bounds) != 4:
            raise ValueError(f"Got bounds of length {len(bounds)} instead of expected 4 (x_lo, x_hi, y_lo, y_hi)")
        self.bounds = (
            bounds[0] + self.margin, bounds[1] - self.margin, bounds[2] + self.margin, bounds[3] - self.margin
        ) if spatial_region == "inside" else (
            bounds[0] - self.margin, bounds[1] + self.margin, bounds[2] - self.margin, bounds[3] + self.margin
        )

        self._binary_parameters = {"spatial_region": self._valid_spatial_regions}

    @property
    def required_cols(self) -> list[str]:
        return self._required_cols

    @property
    def binary_parameters(self) -> dict[str, Any]:
        return self._binary_parameters

    def apply(self,
        episode_groups: dict,
    ) -> tuple[set[int], set[int], set[int]]:
        """Classify episodes by whether the tracked object stays inside/outside a box.

        Extracts XY from ``object_col`` (first 2 elements of the pose vector) and checks
        against ``bounds`` = (x_lo, x_hi, y_lo, y_hi).
        """
        x_lo, x_hi, y_lo, y_hi = self.bounds
        inside_eps: set[int] = set()
        outside_eps: set[int] = set()
        straddler_eps: set[int] = set()

        object_col = self.required_cols[0]

        for ep, table in episode_groups.items():
            poses = column_to_numpy(table[object_col])
            x, y = poses[:, 0], poses[:, 1]
            rho_box = conj(x - x_lo, x_hi - x, y - y_lo, y_hi - y)

            if always(rho_box)[0] >= 0:
                inside_eps.add(ep)
            elif eventually(rho_box)[0] < 0:
                outside_eps.add(ep)
            else:
                straddler_eps.add(ep)

        if self.spatial_region == "inside":
            return inside_eps, outside_eps, straddler_eps
        else:
            return outside_eps, inside_eps, straddler_eps


class DropJointJumpEpisodes(DatasetFilter):
    """Drop episodes where any joint changes by more than ``max_joint_delta`` rad between consecutive frames.

    These are resets/teleports without a recorded trajectory (mostly between rotate retries). The commanded
    actions jump too, so any action chunk spanning one would teach a teleport.
    """

    def __init__(self,
        max_joint_delta: float = 0.5,
        joint_col: str = "observation.state.joint_position",
        **kwargs: Any,
    ):
        super().__init__(**self.prepare_kwargs_for_super(locals()))
        self.max_joint_delta = float(max_joint_delta)
        self.joint_col = joint_col

    @property
    def required_cols(self) -> list[str]:
        return [self.joint_col, "frame_index"]

    @property
    def binary_parameters(self) -> dict[str, tuple]:
        return {}

    def apply(self, episode_groups: dict) -> tuple[set[int], set[int], set[int]]:
        inliers: set[int] = set()
        outliers: set[int] = set()
        for ep, table in episode_groups.items():
            joints = column_to_numpy(table[self.joint_col], np.float32)
            if "frame_index" in table.column_names:
                frames = column_to_numpy(table["frame_index"])
            else:
                frames = np.arange(len(joints))
            order = np.argsort(frames, kind="stable")
            joints, frames = joints[order], frames[order]
            # Rows dropped earlier (skip_actions, empty goals) leave gaps that are not jumps.
            consecutive = np.diff(frames) == 1
            step = np.abs(np.diff(joints, axis=0)).max(axis=1) if len(joints) > 1 else np.zeros(0)
            if np.any(step[consecutive] > self.max_joint_delta):
                outliers.add(ep)
            else:
                inliers.add(ep)
        return inliers, outliers, set()


_OBJECT_POSE_PREFIX = "observation.state.object_pose."
_ROTATE_SCAN_COLS = ["episode_index", "frame_index", "goal.action", "action.high_level_action",
                     "action.high_level_args", "action.low_level_action"]
_ROTATE_CYCLE_MEMO: dict = {}


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def z_rotation_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Signed rotation about world z (CCW positive) from quaternions ``q0`` to ``q1``, both (w, x, y, z)."""
    r = _quat_wxyz_to_mat(q1) @ np.swapaxes(_quat_wxyz_to_mat(q0), -1, -2)
    return np.degrees(np.arctan2(r[..., 1, 0] - r[..., 0, 1], r[..., 0, 0] + r[..., 1, 1]))


def _rotated_block(high_level_args: str) -> str | None:
    try:
        return json.loads(high_level_args)[1]
    except (ValueError, TypeError, IndexError):
        return None


def _stable_hash(strings: list[str]) -> np.ndarray:
    return np.array([zlib.crc32(s.encode()) for s in strings], dtype=np.int64)


def _rotate_rows(fpath: str) -> dict[str, np.ndarray] | None:
    """Rows of one parquet inside a high-level rotate, with the rotated block's (w, x, y, z) quaternion."""
    names = pq.read_schema(fpath).names
    required = ("episode_index", "frame_index", "action.high_level_action",
                "action.high_level_args", "action.low_level_action")
    if any(c not in names for c in required):
        return None
    cols = [c for c in _ROTATE_SCAN_COLS if c in names]
    table = pq.read_table(fpath, columns=cols, use_threads=False, memory_map=True)
    is_rotate = column_to_numpy(pc.fill_null(pc.equal(table["action.high_level_action"], "rotate"), False))
    rows = np.flatnonzero(is_rotate)
    if not len(rows):
        return None
    table = table.take(pa.array(rows))

    args = pc.fill_null(table["action.high_level_args"], "")
    seg_key = args
    if "goal.action" in cols:
        seg_key = pc.binary_join_element_wise(pc.fill_null(table["goal.action"], ""), args, "|")
    key_enc = seg_key.combine_chunks().dictionary_encode()
    args_enc = args.combine_chunks().dictionary_encode()
    blocks = np.array([_rotated_block(s) for s in args_enc.dictionary.to_pylist()], dtype=object)
    row_block = blocks[args_enc.indices.to_numpy()]

    quat = np.full((len(rows), 4), np.nan)
    pose_cols = {b: f"{_OBJECT_POSE_PREFIX}{b}" for b in set(blocks.tolist())
                 if b and f"{_OBJECT_POSE_PREFIX}{b}" in names}
    if pose_cols:
        poses = pq.read_table(fpath, columns=list(pose_cols.values()), use_threads=False, memory_map=True)
        for block, col_name in pose_cols.items():
            sel = np.flatnonzero(row_block == block)
            col = poses[col_name].take(pa.array(rows[sel]))
            valid = column_to_numpy(pc.is_valid(col))
            if valid.any():
                quat[sel[valid]] = column_to_numpy(col.filter(pa.array(valid)), np.float64)[:, 3:7]

    return {
        "episode": column_to_numpy(table["episode_index"]).astype(np.int64),
        "frame": column_to_numpy(table["frame_index"]).astype(np.int64),
        "key": _stable_hash(key_enc.dictionary.to_pylist())[key_enc.indices.to_numpy()],
        "start": column_to_numpy(pc.fill_null(
            pc.equal(table["action.low_level_action"], "rotate_start_traj"), False)).astype(bool),
        "quat": quat,
    }


def rotate_cycles_from_rows(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-cycle block yaw from rotate rows of any number of parquets.

    A rotate segment is a run of consecutive frames with the same (goal.action, high_level_args). Within it a
    cycle starts at each switch into ``rotate_start_traj`` and ends the frame before the next one (or at the
    segment end); frames before the first start are not part of a cycle. A segment with no start is one cycle.
    """
    empty = {"episode": np.zeros(0, np.int64), "start_frame": np.zeros(0, np.int64),
             "end_frame": np.zeros(0, np.int64), "yaw_deg": np.zeros(0)}
    if not len(rows["episode"]):
        return empty
    order = np.lexsort((rows["frame"], rows["episode"]))
    ep, frame, key, start, quat = (rows[k][order] for k in ("episode", "frame", "key", "start", "quat"))

    new_seg = np.ones(len(ep), dtype=bool)
    new_seg[1:] = (ep[1:] != ep[:-1]) | (key[1:] != key[:-1]) | (frame[1:] != frame[:-1] + 1)
    rise = start & ~(np.r_[False, start[:-1]] & ~new_seg)
    seg = np.cumsum(new_seg) - 1
    cum = np.cumsum(rise)
    seg_first = np.flatnonzero(new_seg)
    cycle_in_seg = cum - (cum[seg_first] - rise[seg_first])[seg]
    seg_has_start = np.zeros(len(seg_first), dtype=bool)
    seg_has_start[seg[rise]] = True

    idx = np.flatnonzero((cycle_in_seg > 0) | ~seg_has_start[seg])
    s, c = seg[idx], cycle_in_seg[idx]
    new_cycle = np.r_[True, (s[1:] != s[:-1]) | (c[1:] != c[:-1])]
    first = idx[new_cycle]
    last = idx[np.r_[new_cycle[1:], True]]
    return {"episode": ep[first], "start_frame": frame[first], "end_frame": frame[last],
            "yaw_deg": z_rotation_deg(quat[first], quat[last])}


def scan_rotate_cycles(sources: list[tuple[list[str], int]]) -> dict[str, np.ndarray]:
    """Rotate cycles over all parquet sources. Episode ids are offset per source like the dataset's.

    Shares the episode loader's /tmp cache and lock, so one rank scans and the rest read the result.
    """
    memo_key = tuple((tuple(files), int(offset)) for files, offset in sources)
    if memo_key in _ROTATE_CYCLE_MEMO:
        return _ROTATE_CYCLE_MEMO[memo_key]
    t0 = time.time()
    n_files = sum(len(files) for files, _ in sources)
    n_workers = max(1, min(int(os.environ.get("FRANKA_EPISODE_READ_WORKERS", "32")), n_files or 1))

    def scan():
        parts = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for files, offset in sources:
                for part in pool.map(_rotate_rows, files):
                    if part is not None:
                        part["episode"] = part["episode"] + offset
                        parts.append(part)
        if not parts:
            return rotate_cycles_from_rows({"episode": np.zeros(0, np.int64)})
        return rotate_cycles_from_rows({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})

    use_cache = os.environ.get("FRANKA_EPISODE_CACHE", "1") not in ("0", "false", "False") and n_files > 0
    if not use_cache:
        cycles, src = scan(), f"{n_files} files"
    else:
        cache_dir = Path(os.environ.get("FRANKA_EPISODE_CACHE_DIR", "/tmp/franka_episode_groups"))
        all_files = [f for files, _ in sources for f in files]
        prefix = _cache_prefix(all_files, ["rotate_cycles", [o for _, o in sources]], set(), "rotate_cycles")
        cache_path = cache_dir / f"{prefix}.npz"
        with _exclusive(cache_dir / f"{prefix}.lock"):
            exists = cache_path.exists() and cache_path.stat().st_size
            if exists and (_job_id() or cache_path.stat().st_mtime >= t0):
                with np.load(cache_path) as cached:
                    cycles, src = {k: cached[k] for k in cached.files}, f"cache {cache_path.name}"
            else:
                cycles, src = scan(), f"{n_files} files"
                tmp = f"{cache_path}.tmp"
                with open(tmp, "wb") as fh:
                    np.savez(fh, **cycles)
                os.replace(tmp, cache_path)

    print(f"[FrankaDataset] rotate scan: {len(cycles['episode'])} cycles in "
          f"{len(np.unique(cycles['episode']))} episodes from {src} in {time.time() - t0:.1f}s")
    _ROTATE_CYCLE_MEMO[memo_key] = cycles
    return cycles


class DropSmallRotateEpisodes(DatasetFilter):
    """Drop episodes with any rotate cycle that turns the rotated block less than ``min_rotate_deg`` about world z.

    These are failed attempts (gripper goes through the motion, block does not turn) followed by retries.
    Reads parquet directly rather than through ``required_cols``: only the rotated block's pose is needed per
    rotate, and loading every ``object_pose`` column into the episode tables would cost tens of GB.
    """

    def __init__(self, min_rotate_deg: float = 10.0, **kwargs: Any):
        super().__init__(**self.prepare_kwargs_for_super(locals()))
        self.min_rotate_deg = float(min_rotate_deg)
        self._sources: list[tuple[list[str], int]] | None = None

    @property
    def required_cols(self) -> list[str]:
        return []

    @property
    def binary_parameters(self) -> dict[str, tuple]:
        return {}

    def bind_sources(self, sources: list[tuple[list[str], int]]) -> None:
        self._sources = sources

    def apply(self, episode_groups: dict) -> tuple[set[int], set[int], set[int]]:
        if self._sources is None:
            raise RuntimeError(f"{self!r}: bind_sources() must be called before apply()")
        cycles = scan_rotate_cycles(self._sources)
        small = np.abs(cycles["yaw_deg"]) < self.min_rotate_deg
        bad = set(cycles["episode"][small].tolist())
        outliers = {ep for ep in episode_groups if ep in bad}
        return set(episode_groups) - outliers, outliers, set()


if __name__ == "__main__":
    import os
    from contextlib import redirect_stdout, redirect_stderr

    import rtamt

    def run_silently(fn, *args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with (redirect_stdout(devnull), redirect_stderr(devnull)):
                return fn(*args, **kwargs)

    def classify_episodes_xy_box_over_time(
            episode_groups: dict,
            object_col: str,
            bounds: tuple[float, float, float, float],
    ) -> tuple[set[int], set[int], set[int]]:
        """Classify episodes by whether the tracked object stays inside/outside a box.

        Extracts XY from ``object_col`` (first 2 elements of the pose vector) and checks
        against ``bounds`` = (x_lo, x_hi, y_lo, y_hi).

        Returns (inside_eps, outside_eps, straddler_eps):
          - inside_eps: object always within box (all frames satisfy bounds)
          - outside_eps: object never within box (no frame satisfies bounds)
          - straddler_eps: object crosses the boundary (dropped for clean split)
        """
        x_lo, x_hi, y_lo, y_hi = bounds
        inside_eps: set[int] = set()
        outside_eps: set[int] = set()
        straddler_eps: set[int] = set()

        for ep, ep_df in episode_groups.items():
            x = ep_df[f"{object_col}.x"].to_numpy(dtype='float32')
            y = ep_df[f"{object_col}.y"].to_numpy(dtype='float32')
            rho_box = conj(x - x_lo, x_hi - x, y - y_lo, y_hi - y)

            if always(rho_box)[0] >= 0:
                inside_eps.add(ep)
            elif eventually(rho_box)[0] < 0:
                outside_eps.add(ep)
            else:
                straddler_eps.add(ep)

        return inside_eps, outside_eps, straddler_eps

    def classify_episodes_xy_box_over_time_rtamt(
            episode_groups: dict,
            object_col: str,
            bounds: tuple[float, float, float, float],
    ) -> tuple[set[int], set[int], set[int]]:
        x_lo, x_hi, y_lo, y_hi = bounds
        inside_eps: set[int] = set()
        outside_eps: set[int] = set()
        straddler_eps: set[int] = set()

        spec = rtamt.STLDiscreteTimeSpecification()
        spec.declare_var('x', 'float')
        spec.declare_var('y', 'float')
        spec.declare_const('x_lo', 'float', x_lo)
        spec.declare_const('x_hi', 'float', x_hi)
        spec.declare_const('y_lo', 'float', y_lo)
        spec.declare_const('y_hi', 'float', y_hi)
        spec.spec = '(x >= x_lo) and (x <= x_hi) and (y >= y_lo) and (y <= y_hi)'
        spec.parse()

        for ep, ep_df in episode_groups.items():
            x = ep_df[f"{object_col}.x"].to_numpy(dtype='float32')
            y = ep_df[f"{object_col}.y"].to_numpy(dtype='float32')

            spec.reset()
            in_box = np.array(spec.evaluate({"time": list(range(len(x))), "x": x, "y": y}))[:, 1] >= 0

            if in_box.all():
                inside_eps.add(ep)
            elif not in_box.any():
                outside_eps.add(ep)
            else:
                straddler_eps.add(ep)

        return inside_eps, outside_eps, straddler_eps


    def compare(bounds=(-1.,1.,-.5,.5)):
        import time
        import cProfile
        import pstats

        import pandas as pd
        import pyarrow as pa

        profiler = cProfile.Profile()

        groups = {}
        for r in np.linspace(min(bounds[1], bounds[2]) / 2.0, max(bounds[1], bounds[2]) * 2.0, 100, endpoint=True):
            x = r * np.cos(np.linspace(0, 2 * np.pi, 360), dtype=np.float32)
            y = r / 2 * np.sin(np.linspace(0, 2 * np.pi, 360), dtype=np.float32)
            z = np.array([0.1] * 360, dtype=np.float32)
            groups[r] = pd.DataFrame({f"block.{x}": ar for x, ar in zip("xyz", [x, y, z])})

        object_col = "block"

        table_groups = {ep: pa.Table.from_pandas(group) for ep, group in groups.items()}
        for ep, table in table_groups.items():
            xy = np.stack([table[f"{object_col}.x"].to_numpy(), table[f"{object_col}.y"].to_numpy()], axis=-1)
            table_groups[ep] = table.add_column(0, object_col, pa.FixedShapeTensorArray.from_numpy_ndarray(xy))

        dfilter = ClassifyEpisodesSpatial(
            spatial_region="inside", object_col=object_col, bounds=bounds, inside_outside_margin=0.0
        )

        def wrap_filter(episode_groups, *args, **kwargs):
            return dfilter.apply(episode_groups)

        for method_name, method in zip(["simple", "filter", "rtamt"], [classify_episodes_xy_box_over_time, wrap_filter, classify_episodes_xy_box_over_time_rtamt]):
            print(f"Benchmarking {method_name}...")

            stime = time.time()
            inside, outside, straddler = None, None, None

            def to_silence():
                nonlocal stime, inside, outside, straddler
                # print(f"Warming up {method_name}...")
                cur_groups = groups if method_name not in ["filter"] else table_groups
                inside, outside, straddler = method(cur_groups, object_col=object_col, bounds=bounds)
                stime = time.time()
                if method_name == "filter":
                    profiler.enable()
                for _ in range(5):
                    inside, outside, straddler = method(cur_groups, object_col=object_col, bounds=bounds)
                if method_name == "filter":
                    profiler.disable()

            run_silently(to_silence)

            print(f"{method_name} completed in {time.time() - stime:.2} s")
            for name, bucket in zip(["inside", "outside", "straddler"], [inside, outside, straddler]):
                print(f"  {name}: {len(bucket)} [{min(bucket):.2}, {max(bucket):.2}]")

        print("\nPROFILER:")
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(50)

    compare()
