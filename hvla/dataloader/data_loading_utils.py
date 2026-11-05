from pathlib import Path
from collections import defaultdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import os
import time

from pyarrow import dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

# Train + val in one process share one load. Cleared after both splits are built.
_MEMO: dict = {}


def clear_episode_group_memo() -> None:
    _MEMO.clear()


def column_to_numpy(chunked: pa.ChunkedArray, dtype=None) -> np.ndarray:
    """Convert a pyarrow column to numpy without pandas.

    List / fixed-size-list / tensor columns become 2D ``(N, D)``. Scalar
    numeric or string columns become 1D ``(N,)``.
    """
    col = chunked.combine_chunks()
    n = len(col)
    if pa.types.is_fixed_size_list(col.type):
        arr = np.asarray(col.values.to_numpy()).reshape(n, col.type.list_size)
    elif pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        arr = np.asarray(pc.list_flatten(col).to_numpy()).reshape(n, -1)
    elif isinstance(col.type, pa.lib.FixedShapeTensorType):
        arr = col.to_numpy_ndarray()
    else:
        arr = np.asarray(col.to_numpy(zero_copy_only=False))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def table_cell(table: pa.Table, name: str, i: int, default=None):
    """Python value at row ``i`` of column ``name``, or ``default`` if missing/null."""
    if name not in table.column_names:
        return default
    value = table.column(name)[i].as_py()
    return default if value is None else value


def table_str(table: pa.Table, name: Optional[str], i: int) -> str:
    if not name:
        return ""
    value = table_cell(table, name, i, "")
    return "" if value is None else str(value)


def table_f32_row(table: pa.Table, name: str, i: int) -> np.ndarray:
    return np.asarray(table_cell(table, name, i), dtype=np.float32)


def table_f32_slice(table: pa.Table, name: str, start: int, stop: int) -> np.ndarray:
    return column_to_numpy(table.slice(start, stop - start).column(name), dtype=np.float32)


def pad_repeat_last(arr: np.ndarray, pad_len: int) -> np.ndarray:
    if pad_len <= 0:
        return arr
    return np.concatenate([arr, np.repeat(arr[-1:], pad_len, axis=0)], axis=0)


def load_parquet_dataset(data_dir: Path) -> ds.Dataset:
    # Explicit file list so backups such as ``*.parquet.orig_bak`` are ignored.
    parquet_files = sorted(data_dir.resolve().rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return ds.dataset([str(path) for path in parquet_files], format="parquet")


def read_first_row(data: ds.Dataset, columns: list[str]) -> pa.Table:
    """One-row probe. Avoids Dataset.head, which SIGSEGVs on pyarrow 14."""
    for fpath in data.files:
        parquet_file = pq.ParquetFile(fpath)
        for rg in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg, columns=columns)
            if len(table):
                return table.slice(0, 1)
    raise ValueError(f"No rows found while reading columns {columns}")


def _tables_by_episode(table: pa.Table) -> dict[int, pa.Table]:
    """Split a table on episode_index runs. Episodes may span files / chunks."""
    grouped: dict[int, list[pa.RecordBatch]] = defaultdict(list)
    for chunk in table.to_batches(max_chunksize=40_000):
        ep = chunk["episode_index"].to_numpy()
        cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
        bounds = np.concatenate([[0], cuts, [len(ep)]])
        for s, e in zip(bounds[:-1], bounds[1:]):
            grouped[int(ep[s])].append(chunk.slice(s, e - s))
    return {i: pa.Table.from_batches(slices) for i, slices in grouped.items()}


def _load_one(fpath, cols, extra_filter):
    try:
        table = pq.read_table(fpath, columns=cols, use_threads=False, memory_map=True)
    except Exception as exc:
        raise RuntimeError(f"Failed reading parquet {fpath}: {exc}") from exc
    n_goal = 0
    if extra_filter is not None:
        keep = extra_filter(table)
        n_goal = len(table) - int(pc.sum(pc.cast(keep, pa.int64())).as_py() or 0)
        table = table.filter(keep)
    return table, n_goal


def _job_id() -> str:
    for key in ("FRANKA_EPISODE_CACHE_RUN", "BEAKER_JOB_ID", "BEAKER_WORKLOAD_ID",
                "BEAKER_ASSIGNMENT_ID", "SLURM_JOB_ID", "TORCHELASTIC_RUN_ID"):
        if os.environ.get(key):
            return os.environ[key]
    return ""


@contextmanager
def _exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _cache_prefix(files, cols, skip, filter_id) -> str:
    # "leading_skip": caches written when skip_actions dropped every matching row must not be reused.
    digest = hashlib.sha1(repr((cols, sorted(skip), filter_id, "leading_skip")).encode())
    for fpath in files:
        st = os.stat(fpath)
        digest.update(fpath.encode())
        digest.update(f"{st.st_mtime_ns}{st.st_size}".encode())
    job = _job_id()
    tag = hashlib.sha1(job.encode()).hexdigest()[:10] if job else "anon"
    return f"{tag}_{digest.hexdigest()[:12]}"


def _read_cache(path: Path) -> dict[int, pa.Table]:
    with pa.memory_map(str(path), "r") as src:
        return _tables_by_episode(pa.ipc.open_file(src).read_all())


def _write_cache(path: Path, tables: dict[int, pa.Table]) -> None:
    if not tables:
        return
    tmp = str(path) + ".tmp"
    first = next(iter(tables.values()))
    with pa.OSFile(tmp, "wb") as sink:
        with pa.ipc.new_file(sink, first.schema) as writer:
            for tbl in tables.values():
                for batch in tbl.to_batches(max_chunksize=40_000):
                    writer.write_batch(batch)
    os.replace(tmp, path)


_LL_ACTION_COL = "action.low_level_action"


def _trim_leading_skip_rows(table: pa.Table, skip_actions: set[str]) -> tuple[pa.Table, int]:
    """Drop the *leading* run of rows whose low-level action is in ``skip_actions``.

    Only the prefix is removed: a skip action that appears mid-episode (e.g. an
    ``init_gripper_open_traj`` segment after a retract) is kept, so the action
    windows built on top of this table stay contiguous in time. Returns the
    trimmed table and the number of rows dropped.
    """
    if not skip_actions or _LL_ACTION_COL not in table.column_names:
        return table, 0
    ll = table.column(_LL_ACTION_COL).to_numpy(zero_copy_only=False)
    keep = ~np.isin(ll, list(skip_actions))
    if keep.all():
        return table, 0
    first_keep = int(np.argmax(keep)) if keep.any() else len(table)
    return table.slice(first_keep), first_keep


def group_episode_tables(
    data: ds.Dataset,
    keep_cols: list[str],
    skip_actions: set[str] | None = None,
    disable_progress: bool = True,
    additional_filter=None,
) -> tuple[dict, int, int]:
    """``episode_index → table`` for ``keep_cols``. Also skip-action / empty-goal counts.

    ``skip_actions`` only trims the leading run of matching rows per episode
    (see ``_trim_leading_skip_rows``).

    Rank 0 reads parquet in a thread pool and writes a job-scoped /tmp Arrow
    file. Other ranks mmap it. Train/val in one process share ``_MEMO``.
    """
    skip = set(skip_actions or ())
    cols = [c for c in keep_cols if c in data.schema.names]
    files = list(data.files)
    filter_id = getattr(additional_filter, "__name__", "none") if additional_filter else "none"
    key = (tuple(files), tuple(cols), tuple(sorted(skip)), filter_id)
    if key in _MEMO:
        return _MEMO[key]

    t0 = time.time()
    n_workers = max(1, min(int(os.environ.get("FRANKA_EPISODE_READ_WORKERS", "32")), len(files) or 1))
    use_cache = os.environ.get("FRANKA_EPISODE_CACHE", "1") not in ("0", "false", "False") and bool(files)
    cache_dir = Path(os.environ.get("FRANKA_EPISODE_CACHE_DIR", "/tmp/franka_episode_groups"))
    prefix = _cache_prefix(files, cols, skip, filter_id) if use_cache else ""

    def finish(tables, n_act=0, n_goal=0, src=""):
        _MEMO[key] = (tables, n_act, n_goal)
        how = f" cache {src}" if src else f" {len(files)} files, {n_workers} workers"
        print(f"[FrankaDataset] parquet load: {len(tables)} episodes,{how} in {time.time() - t0:.1f}s")
        return _MEMO[key]

    def load_parquet():
        parts, n_act, n_goal = [], 0, 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_load_one, f, cols, additional_filter) for f in files]
            for fut in tqdm(futs, desc=f"slicing episodes ({n_workers} workers)",
                            disable=disable_progress):
                table, n_g = fut.result()
                n_goal += n_g
                if len(table):
                    parts.append(table)
        if not parts:
            return {}, n_act, n_goal
        # Trim per episode, not per file: episodes may span files.
        tables: dict[int, pa.Table] = {}
        for ep_idx, table in _tables_by_episode(pa.concat_tables(parts)).items():
            table, n_a = _trim_leading_skip_rows(table, skip)
            n_act += n_a
            if len(table):
                tables[ep_idx] = table
        return tables, n_act, n_goal

    if not files:
        return finish({})
    if not use_cache:
        return finish(*load_parquet())

    cache_path = cache_dir / f"{prefix}.arrow"
    with _exclusive(cache_dir / f"{prefix}.lock"):
        exists = cache_path.exists() and cache_path.stat().st_size
        # Job-id prefix already isolates launches. Anon: only a sibling write.
        usable = exists and (_job_id() or cache_path.stat().st_mtime >= t0)
        if usable:
            try:
                return finish(_read_cache(cache_path), src=cache_path.name)
            except Exception as exc:
                print(f"[FrankaDataset] parquet cache unreadable ({exc}); rebuilding")
        elif exists:
            print(f"[FrankaDataset] ignoring leftover parquet cache {cache_path.name}")

        tables, n_act, n_goal = load_parquet()
        try:
            _write_cache(cache_path, tables)
        except Exception as exc:
            print(f"[FrankaDataset] parquet cache write failed ({exc})")
        return finish(tables, n_act, n_goal)


_ACTION_VALID_COL = "action.is_valid"
_IS_LAST_COL = "is_last"


def valid_row_min_max(data: ds.Dataset, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim min / max of ``col`` over the rows training actually reads.

    ``meta/stats.json`` min / max include the zeroed terminal rows
    (``action.is_valid == False``), so they cannot serve as a normalization
    band. Datasets without the column fall back to dropping ``is_last`` rows.
    """
    schema_names = set(data.schema.names)
    if _ACTION_VALID_COL in schema_names:
        filter_expr = ds.field(_ACTION_VALID_COL) == True  # noqa: E712
    elif _IS_LAST_COL in schema_names:
        filter_expr = ds.field(_IS_LAST_COL) == False  # noqa: E712
    else:
        filter_expr = None
    table = data.to_table(columns=[col], filter=filter_expr)
    if table.num_rows == 0:
        raise ValueError(f"No valid rows for {col!r} under {data.files[0]}")
    arr = column_to_numpy(table.column(col), dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)
