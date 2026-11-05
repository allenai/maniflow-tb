"""Pure helpers in hvla/dataloader/franka_dataset.py."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from hvla.dataloader.franka_dataset import (
    DEFAULT_PHASE_SAMPLE_WEIGHTS,
    FRONT_CAM_KEY,
    GRIPPER_CAM_KEY,
    _allowed_run_starts,
    _assert_native_16_9,
    _build_depth_to_ptmap,
    _format_lookback_weights,
    _has_out_of_range_target_coordinate,
    _is_valid_action_flag,
    _parse_lookback_weights,
    _parse_phase_sample_weights,
    _rg8_pack55,
    _row_is_zero_padded_action,
    _sample_lookback,
    _targets_match,
    action_row_limit,
    normalize_depth_g12,
    normalize_depth_rg8,
    parse_target_front_crop,
)

DEFAULT_LOOKBACK = [(0.0, 0.0, 0.2), (0.0, 1.0, 0.8)]


class TestLookbackWeights:
    @pytest.mark.parametrize("value", [None, "", "none", "null", "NULL"])
    def test_default(self, value: Any) -> None:
        assert _parse_lookback_weights(value) == DEFAULT_LOOKBACK

    def test_string(self) -> None:
        assert _parse_lookback_weights("0-0.5_0.25, 0.5-2_0.75") == [(0.0, 0.5, 0.25), (0.5, 2.0, 0.75)]

    def test_tuples(self) -> None:
        assert _parse_lookback_weights([(0, 1, 1)]) == [(0.0, 1.0, 1.0)]

    @pytest.mark.parametrize("value", [
        "0-1",          # no weight
        "0_1_1",        # underscore inside the range
        "0-1-2_1",      # three bounds
        "2-1_1",        # high < low
        "0-1_0.5",      # does not sum to 1
        "0-1_0",        # no positive weight
        [(-1, 1, 1)],   # low < 0
        [(0, 1, -1)],   # negative weight
        [(0, 1)],       # wrong arity
    ])
    def test_invalid(self, value: Any) -> None:
        with pytest.raises(ValueError):
            _parse_lookback_weights(value)

    def test_format_round_trip(self) -> None:
        spec = "0-0_0.2,0-1_0.8"
        assert _format_lookback_weights(_parse_lookback_weights(spec)) == spec

    def test_sample_within_range(self) -> None:
        random.seed(0)
        draws = [_sample_lookback(DEFAULT_LOOKBACK) for _ in range(500)]
        assert all(0.0 <= d <= 1.0 for d in draws)
        assert any(d == 0.0 for d in draws) and any(d > 0.0 for d in draws)

    def test_sample_zero_width(self) -> None:
        assert _sample_lookback([(0.5, 0.5, 1.0)]) == 0.5


class TestAllowedRunStarts:
    LABELS = ["pick"] * 5 + ["place"] * 5

    @pytest.mark.parametrize(("allowed", "expected"), [
        ({"place"}, [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]),
        ({"pick", "place"}, [0] * 10),
        ({"pick"}, [0, 0, 0, 0, 0, 5, 6, 7, 8, 9]),
        (set(), list(range(10))),
    ])
    def test_runs(self, allowed: set[str], expected: list[int]) -> None:
        assert _allowed_run_starts(self.LABELS, allowed).tolist() == expected

    def test_none_breaks_run(self) -> None:
        assert _allowed_run_starts(["a", None, "a", "a"], {"a"}).tolist() == [0, 1, 2, 2]

    def test_empty_and_dtype(self) -> None:
        assert _allowed_run_starts([], {"a"}).tolist() == []
        assert _allowed_run_starts(["a"], {"a"}).dtype == np.int32


class TestTargetChecks:
    @pytest.mark.parametrize(("value", "expected"), [
        (None, False),
        ([5.0], False),
        ("abc", False),
        ([10.0, 0.0], False),           # inclusive lower edges
        ([630.0, 330.0], False),        # inclusive upper edges
        ([9.9, 100.0], True),
        ([100.0, 330.1], True),
        ([float("nan"), 100.0], False),  # non-finite axis never counts
        ([float("inf"), 100.0], False),
        ([float("nan"), 400.0], True),   # but the other axis still does
    ])
    def test_out_of_range(self, value: Any, expected: bool) -> None:
        assert _has_out_of_range_target_coordinate(value) is expected

    @pytest.mark.parametrize(("current", "previous", "expected"), [
        (None, None, True),
        (None, [1.0, 2.0], False),
        ([1.0, 2.0], [1.0, 2.0, 3.0], False),
        ([0.0, 0.0], [10.0, 0.0], True),    # atol is inclusive
        ([0.0, 0.0], [10.5, 0.0], False),
        ([float("nan"), 1.0], [float("nan"), 1.0], True),
    ])
    def test_targets_match(self, current: Any, previous: Any, expected: bool) -> None:
        assert _targets_match(current, previous) is expected


class TestSmallParsers:
    def test_phase_weights(self) -> None:
        assert _parse_phase_sample_weights(None) is None
        weights = {"a": 1.0}
        assert _parse_phase_sample_weights(weights) is weights
        default = _parse_phase_sample_weights(" Default ")
        assert default == DEFAULT_PHASE_SAMPLE_WEIGHTS and default is not DEFAULT_PHASE_SAMPLE_WEIGHTS

    @pytest.mark.parametrize("value", ["foo", 3, ["a"]])
    def test_phase_weights_invalid(self, value: Any) -> None:
        with pytest.raises(ValueError):
            _parse_phase_sample_weights(value)

    @pytest.mark.parametrize(("value", "expected"), [
        (None, -1), ("", -1), ("null", -1), ("None", -1), (-1, -1),
        (0, 0), (99, 99), ("33", 33), ("50.0", 50), (33.7, 33),
    ])
    def test_target_front_crop(self, value: Any, expected: int) -> None:
        assert parse_target_front_crop(value) == expected

    @pytest.mark.parametrize("value", [100, -2, "150"])
    def test_target_front_crop_invalid(self, value: Any) -> None:
        with pytest.raises(ValueError):
            parse_target_front_crop(value)

    @pytest.mark.parametrize(("value", "expected"), [
        (None, None), (False, None), ("", None), ("false", None), ("none", None), ("null", None),
        (True, "rg8"), ("true", "rg8"), ("RG8_224", "rg8_224"), ("_rg8_55", "rg8_55"),
    ])
    def test_depth_rg8(self, value: Any, expected: Any) -> None:
        assert normalize_depth_rg8(value) == expected

    @pytest.mark.parametrize(("value", "expected"), [
        (None, None), (False, None), ("null", None),
        (True, "g12_224"), ("true", "g12_224"), ("g12", "g12"), ("G10_224_crf035", "g10_224_crf035"),
    ])
    def test_depth_g12(self, value: Any, expected: Any) -> None:
        assert normalize_depth_g12(value) == expected

    @pytest.mark.parametrize(("fn", "value"), [
        (normalize_depth_rg8, "rgb8"), (normalize_depth_rg8, "rg8-224"),
        (normalize_depth_g12, "g11"), (normalize_depth_g12, "rg8"),
    ])
    def test_depth_invalid(self, fn: Any, value: str) -> None:
        with pytest.raises(ValueError):
            fn(value)

    def test_rg8_pack55(self) -> None:
        assert _rg8_pack55("rg8_55") and _rg8_pack55("rg8_55_224")
        assert not _rg8_pack55("rg8_552") and not _rg8_pack55("rg8") and not _rg8_pack55(None)

    def test_native_16_9(self) -> None:
        _assert_native_16_9(640, 360, "front_camera")
        _assert_native_16_9(1280, 720, "front_camera")
        with pytest.raises(ValueError):
            _assert_native_16_9(640, 480, "front_camera")

    def test_depth_to_ptmap(self) -> None:
        assert _build_depth_to_ptmap(FRONT_CAM_KEY, GRIPPER_CAM_KEY) == {
            "observation.depth.front_camera": (FRONT_CAM_KEY, "front_camera"),
            "observation.depth.gripper_camera": (GRIPPER_CAM_KEY, "gripper_camera"),
        }


@pytest.mark.parametrize(("value", "expected"), [
    (None, False), (True, True), (0, False), (1, True),
    ([1], True), ([0], False), ([], False), ((True,), True), (np.array([0]), False),
])
def test_is_valid_action_flag(value: Any, expected: bool) -> None:
    assert _is_valid_action_flag(value) is expected


ACTION_COLS = ["action.joint_target", "action.gripper_position", "action.pd_mode"]


def _action_table(rows: list[tuple[list[float], float, float]], is_valid: list[Any] | None = None) -> pa.Table:
    data: dict[str, Any] = {
        "action.joint_target": [r[0] for r in rows],
        "action.gripper_position": [r[1] for r in rows],
        "action.pd_mode": [r[2] for r in rows],
    }
    if is_valid is not None:
        data["action.is_valid"] = is_valid
    return pa.table(data)


NONZERO = ([1.0] * 7, 0.5, 1.0)
ZERO = ([0.0] * 7, 0.0, 0.0)


class TestActionRowLimit:
    def test_zero_padded_row(self) -> None:
        table = _action_table([NONZERO, ZERO])
        assert not _row_is_zero_padded_action(table, 0, ACTION_COLS)
        assert _row_is_zero_padded_action(table, 1, ACTION_COLS)

    def test_zero_padded_needs_a_checked_column(self) -> None:
        table = _action_table([ZERO])
        assert not _row_is_zero_padded_action(table, 0, ["action.missing"])
        nulls = pa.table({"action.pd_mode": pa.array([None], type=pa.float32())})
        assert not _row_is_zero_padded_action(nulls, 0, ["action.pd_mode"])

    def test_is_valid_path(self) -> None:
        assert action_row_limit(_action_table([NONZERO] * 4, [True, True, False, False]), 4) == 2
        assert action_row_limit(_action_table([NONZERO] * 2, [False, False]), 2) == 0
        assert action_row_limit(_action_table([NONZERO] * 2, [[1], [0]]), 2) == 1

    def test_is_valid_wins_over_zero_check(self) -> None:
        assert action_row_limit(_action_table([NONZERO, ZERO], [True, True]), 2) == 2

    def test_abs_zero_padded_last_row(self) -> None:
        assert action_row_limit(_action_table([NONZERO, NONZERO, ZERO]), 3) == 2
        assert action_row_limit(_action_table([NONZERO, NONZERO]), 2) == 2

    def test_non_abs_keeps_all_rows(self) -> None:
        assert action_row_limit(_action_table([NONZERO, ZERO]), 2, action_mode="relative") == 2

    def test_explicit_action_cols(self) -> None:
        table = _action_table([NONZERO, ([5.0] * 7, 0.0, 0.0)])
        assert action_row_limit(table, 2, ["action.gripper_position", "action.pd_mode"]) == 1
