"""hvla/dataloader/data_loading_utils.py."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from hvla.dataloader import data_loading_utils as dlu
from hvla.dataloader.data_loading_utils import (
    _cache_prefix,
    _job_id,
    _tables_by_episode,
    _trim_leading_skip_rows,
    clear_episode_group_memo,
    column_to_numpy,
    group_episode_tables,
    load_parquet_dataset,
    pad_repeat_last,
    read_first_row,
    table_cell,
    table_f32_row,
    table_f32_slice,
    table_str,
    valid_row_min_max,
)

JOB_ID_KEYS = ("FRANKA_EPISODE_CACHE_RUN", "BEAKER_JOB_ID", "BEAKER_WORKLOAD_ID",
               "BEAKER_ASSIGNMENT_ID", "SLURM_JOB_ID", "TORCHELASTIC_RUN_ID")


def _write_parquet(path: Path, table: pa.Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


class TestTableHelpers:
    def test_column_to_numpy_list(self) -> None:
        col = pa.chunked_array([pa.array([[1.0, 2.0], [3.0, 4.0]])])
        np.testing.assert_array_equal(column_to_numpy(col), [[1.0, 2.0], [3.0, 4.0]])

    def test_column_to_numpy_fixed_size_list(self) -> None:
        arr = pa.FixedSizeListArray.from_arrays(pa.array([1, 2, 3, 4, 5, 6], type=pa.float32()), 3)
        out = column_to_numpy(pa.chunked_array([arr]))
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out[1], [4, 5, 6])

    def test_column_to_numpy_scalar_and_dtype(self) -> None:
        out = column_to_numpy(pa.chunked_array([pa.array([1, 2, 3])]), dtype=np.float32)
        assert out.shape == (3,) and out.dtype == np.float32

    def test_column_to_numpy_multiple_chunks(self) -> None:
        col = pa.chunked_array([pa.array([[1.0]]), pa.array([[2.0]])])
        np.testing.assert_array_equal(column_to_numpy(col), [[1.0], [2.0]])

    def test_table_cell_and_str(self) -> None:
        table = pa.table({"a": [1, None], "s": ["x", None]})
        assert table_cell(table, "a", 0) == 1
        assert table_cell(table, "a", 1, default=7) == 7
        assert table_cell(table, "missing", 0, default="d") == "d"
        assert table_str(table, "s", 0) == "x"
        assert table_str(table, "s", 1) == ""
        assert table_str(table, "a", 0) == "1"
        assert table_str(table, None, 0) == ""

    def test_table_f32(self) -> None:
        table = pa.table({"v": [[1, 2], [3, 4], [5, 6]]})
        row = table_f32_row(table, "v", 1)
        assert row.dtype == np.float32 and row.tolist() == [3.0, 4.0]
        sl = table_f32_slice(table, "v", 1, 3)
        assert sl.dtype == np.float32 and sl.tolist() == [[3.0, 4.0], [5.0, 6.0]]

    def test_pad_repeat_last(self) -> None:
        arr = np.array([[1, 1], [2, 2]])
        assert pad_repeat_last(arr, 0) is arr
        assert pad_repeat_last(arr, -3) is arr
        padded = pad_repeat_last(arr, 2)
        assert padded.tolist() == [[1, 1], [2, 2], [2, 2], [2, 2]]


def _ll_table(labels: list[str]) -> pa.Table:
    return pa.table({"action.low_level_action": labels, "t": list(range(len(labels)))})


class TestTrimLeadingSkipRows:
    def test_only_leading_run(self) -> None:
        table = _ll_table(["init", "init", "move", "init", "noop"])
        trimmed, dropped = _trim_leading_skip_rows(table, {"init", "noop"})
        assert dropped == 2
        assert trimmed.column("t").to_pylist() == [2, 3, 4]

    def test_all_rows_skipped(self) -> None:
        trimmed, dropped = _trim_leading_skip_rows(_ll_table(["init", "init"]), {"init"})
        assert (len(trimmed), dropped) == (0, 2)

    def test_nothing_to_trim(self) -> None:
        table = _ll_table(["move", "init"])
        assert _trim_leading_skip_rows(table, {"init"}) == (table, 0)

    def test_no_skip_set_or_column(self) -> None:
        table = _ll_table(["init"])
        assert _trim_leading_skip_rows(table, set()) == (table, 0)
        no_col = pa.table({"t": [0]})
        assert _trim_leading_skip_rows(no_col, {"init"}) == (no_col, 0)


def test_tables_by_episode_spanning_tables() -> None:
    first = pa.table({"episode_index": [0, 0, 1], "t": [0, 1, 0]})
    second = pa.table({"episode_index": [1, 2], "t": [1, 0]})
    groups = _tables_by_episode(pa.concat_tables([first, second]))
    assert sorted(groups) == [0, 1, 2]
    assert groups[1].column("t").to_pylist() == [0, 1]
    assert len(groups[0]) == 2 and len(groups[2]) == 1


class TestLoaders:
    def test_load_parquet_dataset_empty(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_parquet_dataset(tmp_path)

    def test_load_parquet_dataset_ignores_backups(self, tmp_path: Path) -> None:
        good = _write_parquet(tmp_path / "chunk-000" / "file-000.parquet", pa.table({"a": [1]}))
        _write_parquet(tmp_path / "chunk-000" / "file-001.parquet.orig_bak", pa.table({"a": [2]}))
        assert load_parquet_dataset(tmp_path).files == [str(good.resolve())]

    def test_read_first_row(self, tmp_path: Path) -> None:
        _write_parquet(tmp_path / "file-000.parquet", pa.table({"a": [5, 6], "b": [7, 8]}))
        row = read_first_row(load_parquet_dataset(tmp_path), ["a"])
        assert row.to_pydict() == {"a": [5]}

    def test_read_first_row_empty(self, tmp_path: Path) -> None:
        _write_parquet(tmp_path / "file-000.parquet", pa.table({"a": pa.array([], type=pa.int64())}))
        with pytest.raises(ValueError):
            read_first_row(load_parquet_dataset(tmp_path), ["a"])


def _episode_dataset(tmp_path: Path) -> Path:
    table = pa.table({
        "episode_index": [0, 0, 0, 0, 1, 1, 1],
        "action.low_level_action": ["init", "init", "move", "init", "move", "move", "init"],
        "goal.action": ["g", "g", "g", "", "g", "", "g"],
        "extra": [0, 1, 2, 3, 4, 5, 6],
    })
    _write_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet", table)
    return tmp_path / "data"


def drop_empty_goals(table: pa.Table) -> pa.ChunkedArray:
    return pc.not_equal(table["goal.action"], "")


class TestGroupEpisodeTables:
    COLS: ClassVar[list[str]] = ["episode_index", "action.low_level_action", "goal.action", "not_in_schema"]

    def test_leading_skip_counts(self, tmp_path: Path) -> None:
        data = load_parquet_dataset(_episode_dataset(tmp_path))
        tables, n_act, n_goal = group_episode_tables(data, self.COLS, skip_actions={"init"})
        assert (n_act, n_goal) == (2, 0)
        assert tables[0].column("action.low_level_action").to_pylist() == ["move", "init"]
        assert len(tables[1]) == 3
        assert "not_in_schema" not in tables[0].column_names

    def test_additional_filter(self, tmp_path: Path) -> None:
        data = load_parquet_dataset(_episode_dataset(tmp_path))
        tables, n_act, n_goal = group_episode_tables(data, self.COLS, additional_filter=drop_empty_goals)
        assert (n_act, n_goal) == (0, 2)
        assert [len(tables[0]), len(tables[1])] == [3, 2]

    def test_memo(self, tmp_path: Path) -> None:
        data = load_parquet_dataset(_episode_dataset(tmp_path))
        first = group_episode_tables(data, self.COLS)
        assert group_episode_tables(data, self.COLS) is first
        clear_episode_group_memo()
        assert group_episode_tables(data, self.COLS) is not first

    def test_disk_cache_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("FRANKA_EPISODE_CACHE", "1")
        monkeypatch.setenv("FRANKA_EPISODE_CACHE_DIR", str(cache_dir))
        monkeypatch.setenv("FRANKA_EPISODE_CACHE_RUN", "unit-test-run")
        data = load_parquet_dataset(_episode_dataset(tmp_path))

        built, _, _ = group_episode_tables(data, self.COLS, skip_actions={"init"})
        assert len(list(cache_dir.glob("*.arrow"))) == 1
        clear_episode_group_memo()
        capsys.readouterr()
        cached, _, _ = group_episode_tables(data, self.COLS, skip_actions={"init"})
        assert " cache " in capsys.readouterr().out
        assert {ep: t.to_pydict() for ep, t in cached.items()} == {ep: t.to_pydict() for ep, t in built.items()}


class TestValidRowMinMax:
    def test_prefers_is_valid(self, tmp_path: Path) -> None:
        _write_parquet(tmp_path / "f.parquet", pa.table({
            "a": [[0.0, 0.0], [1.0, 5.0], [3.0, 2.0]],
            "action.is_valid": [False, True, True],
            "is_last": [False, False, False],
        }))
        lo, hi = valid_row_min_max(load_parquet_dataset(tmp_path), "a")
        assert lo.tolist() == [1.0, 2.0] and hi.tolist() == [3.0, 5.0]

    def test_falls_back_to_is_last(self, tmp_path: Path) -> None:
        _write_parquet(tmp_path / "f.parquet", pa.table({"a": [[1.0], [2.0], [0.0]], "is_last": [False, False, True]}))
        lo, hi = valid_row_min_max(load_parquet_dataset(tmp_path), "a")
        assert (lo.tolist(), hi.tolist()) == ([1.0], [2.0])

    def test_falls_back_to_all_rows(self, tmp_path: Path) -> None:
        _write_parquet(tmp_path / "f.parquet", pa.table({"a": [[1.0], [-2.0]]}))
        lo, hi = valid_row_min_max(load_parquet_dataset(tmp_path), "a")
        assert (lo.tolist(), hi.tolist()) == ([-2.0], [1.0])

    def test_no_valid_rows(self, tmp_path: Path) -> None:
        _write_parquet(tmp_path / "f.parquet", pa.table({"a": [[1.0]], "action.is_valid": [False]}))
        with pytest.raises(ValueError):
            valid_row_min_max(load_parquet_dataset(tmp_path), "a")


class TestCacheKey:
    @pytest.fixture(autouse=True)
    def _clear_job_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in JOB_ID_KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_job_id_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _job_id() == ""
        monkeypatch.setenv("SLURM_JOB_ID", "slurm")
        assert _job_id() == "slurm"
        monkeypatch.setenv("BEAKER_JOB_ID", "beaker")
        assert _job_id() == "beaker"
        monkeypatch.setenv("FRANKA_EPISODE_CACHE_RUN", "explicit")
        assert _job_id() == "explicit"

    def test_cache_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = str(_write_parquet(tmp_path / "f.parquet", pa.table({"a": [1]})))
        base = _cache_prefix([f], ["a"], set(), "none")
        assert base.startswith("anon_")
        assert _cache_prefix([f], ["a"], set(), "none") == base
        assert _cache_prefix([f], ["a"], {"init"}, "none") != base
        assert _cache_prefix([f], ["a"], set(), "keep_nonempty_goal_actions") != base
        assert _cache_prefix([f], ["a", "b"], set(), "none") != base
        monkeypatch.setenv("BEAKER_JOB_ID", "job")
        assert not _cache_prefix([f], ["a"], set(), "none").startswith("anon_")


def test_clear_memo() -> None:
    dlu._MEMO["sentinel"] = object()
    clear_episode_group_memo()
    assert dlu._MEMO == {}
