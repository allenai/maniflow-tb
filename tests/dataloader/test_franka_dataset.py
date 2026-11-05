"""FrankaMultiModalDataset end-to-end on tiny synthetic LeRobot datasets (see conftest.py)."""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pytest
from conftest import VIDEO_H, VIDEO_W, BuildDataset, MakeDataset, joint_vector, row_base

from hvla.dataloader.franka_dataset import FRONT_CAM_KEY, GRIPPER_CAM_KEY, FrankaMultiModalDataset

PICK = ("pick", "pick_traj")
FLIP_THEN_DONE = [("flip forward", "reorient_traj")] * 6 + [("done", "noop_traj")] * 6


def sample_at(dataset: FrankaMultiModalDataset, episode: int, t: int) -> dict[str, Any]:
    return dataset[dataset._samples.index((episode, t))]


def arm_rows(sample: dict[str, Any]) -> list[float]:
    """First joint of each action row, i.e. ``row_base`` of the row it came from."""
    return sample["action"][:, 0].tolist()


class TestBasics:
    def test_sample_shapes_and_values(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        dataset = build_dataset(make_franka_dataset([[PICK] * 6] * 2))
        assert len(dataset) == 12

        sample = sample_at(dataset, 1, 1)
        assert sample["state"].shape == (9,)
        np.testing.assert_allclose(sample["state"][:7], joint_vector(1, 1))
        assert sample["state"][7:].tolist() == [1.0, 1.0]  # gripper position, pd_mode
        assert sample["action"].shape == (4, 9)
        assert arm_rows(sample) == [row_base(1, t) for t in (1, 2, 3, 4)]
        assert sample["action"][:, 7].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert sample["lang"] == "pick"
        assert sample["images"] == {}

    def test_action_window_repeats_last_row(self, make_franka_dataset: MakeDataset,
                                            build_dataset: BuildDataset) -> None:
        dataset = build_dataset(make_franka_dataset([[PICK] * 6]))
        assert arm_rows(sample_at(dataset, 0, 4)) == [row_base(0, t) for t in (4, 5, 5, 5)]

    def test_val_split_by_episode(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        root = make_franka_dataset([[PICK] * 3] * 10)
        train = build_dataset(root, split="train", val_ratio=0.2)
        val = build_dataset(root, split="val", val_ratio=0.2)
        assert sorted(val.episodes) == [0, 1]
        assert sorted(train.episodes) == list(range(2, 10))
        assert {ep for ep, _ in train._samples}.isdisjoint({ep for ep, _ in val._samples})

    def test_invalid_trailing_rows_are_not_starts(self, make_franka_dataset: MakeDataset,
                                                  build_dataset: BuildDataset) -> None:
        root = make_franka_dataset([[PICK] * 6], is_valid=[[True] * 4 + [False] * 2])
        dataset = build_dataset(root)
        assert sorted(dataset._samples) == [(0, t) for t in range(4)]
        assert arm_rows(sample_at(dataset, 0, 3)) == [row_base(0, 3)] * 4

    def test_empty_goal_rows_dropped(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        episode = [PICK, PICK, PICK, ("", "noop_traj")]
        dataset = build_dataset(make_franka_dataset([episode]))
        assert dataset._episode_groups[0].column("goal.action").to_pylist() == ["pick"] * 3

    @pytest.mark.xfail(strict=True, reason=(
        "keep_nonempty_goal_actions uses pc.or_, which yields null for null goal.action, so Table.filter "
        "drops those rows despite the 'Keep nulls' comment (pc.or_kleene would keep them)"))
    def test_null_goal_rows_kept(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        episode = [PICK, PICK, (None, "pick_traj"), PICK, ("", "noop_traj")]
        dataset = build_dataset(make_franka_dataset([episode]))
        assert len(dataset._episode_groups[0]) == 4


class TestSkipActions:
    EPISODE = ([("pick", "init_gripper_open_traj")] * 2 + [PICK] * 3
               + [("pick", "init_gripper_open_traj")] * 2 + [("pick", "noop_traj")])

    @pytest.mark.parametrize("skip", ["init_gripper_open_traj", ["init_gripper_open_traj"]])
    def test_trims_only_leading_rows(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset,
                                     skip: Any) -> None:
        dataset = build_dataset(make_franka_dataset([self.EPISODE]), skip_actions=skip)
        table = dataset._episode_groups[0]
        assert len(table) == 6
        assert table.column("action.low_level_action").to_pylist()[3:5] == ["init_gripper_open_traj"] * 2
        # The first window starts at original row 2 and pads through the kept mid-episode skip rows.
        assert arm_rows(sample_at(dataset, 0, 0)) == [row_base(0, t) for t in (2, 3, 4, 5)]


class TestGoalActionFilter:
    PICK_PLACE = [PICK] * 3 + [("place", "place_traj")] * 3

    def test_only_allowed_labels_start_samples(self, make_franka_dataset: MakeDataset,
                                               build_dataset: BuildDataset) -> None:
        dataset = build_dataset(make_franka_dataset([self.PICK_PLACE]), goal_action_filter=["place"])
        assert sorted(dataset._samples) == [(0, 3), (0, 4), (0, 5)]
        # Out-of-list rows stay in the table, so windows are unaffected.
        assert len(dataset._episode_groups[0]) == 6

    def test_per_root_groups(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        root_a = make_franka_dataset([self.PICK_PLACE], name="a")
        root_b = make_franka_dataset([[("lift", "lift_traj")] * 4], name="b")
        dataset = build_dataset([root_a, root_b], goal_action_filter=[["place"], []])
        # Root b's episode 0 becomes global episode 1; its empty group means "no filter".
        assert sorted(dataset._samples) == [(0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2), (1, 3)]

    def test_group_count_mismatch(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        with pytest.raises(ValueError, match="groups"):
            build_dataset(make_franka_dataset([self.PICK_PLACE]), goal_action_filter=[["pick"], ["place"]])

    def test_requires_language_column(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        with pytest.raises(ValueError, match="language column"):
            build_dataset(make_franka_dataset([self.PICK_PLACE]), language_conditioned=False,
                          goal_action_filter=["place"])

    def test_exclusive_with_phase_sample_weights(self, make_franka_dataset: MakeDataset,
                                                 build_dataset: BuildDataset) -> None:
        with pytest.raises(AssertionError, match="mutually exclusive"):
            build_dataset(make_franka_dataset([self.PICK_PLACE]), goal_action_filter=["place"],
                          phase_sample_weights={"place_traj": 1.0})


class TestPreviousFrameClamp:
    """use_previous_front_frame + goal_action_filter must never read a filtered-out goal."""

    def _build(self, build_dataset: BuildDataset, make_franka_dataset: MakeDataset, **kwargs: Any
               ) -> FrankaMultiModalDataset:
        return build_dataset(make_franka_dataset([FLIP_THEN_DONE]), use_previous_front_frame=True,
                             camera_keys=[FRONT_CAM_KEY], **kwargs)

    def test_prompts_stay_in_allow_list(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset,
                                        capsys: pytest.CaptureFixture[str]) -> None:
        dataset = self._build(build_dataset, make_franka_dataset, goal_action_filter=["done"])
        # Starts 6..9 sit within n_action_steps (4) of the flip -> done boundary.
        assert "lookback clamped for up to 4 starts" in capsys.readouterr().out

        def prompts() -> Counter[str]:
            random.seed(0)
            return Counter(dataset[i]["lang"] for _ in range(25) for i in range(len(dataset)))

        assert set(prompts()) == {"done"}
        dataset._allowed_run_start = {}  # without the clamp the earlier goal leaks in
        assert "flip forward" in prompts()

    def test_no_clamp_count_when_boundary_is_episode_start(self, make_franka_dataset: MakeDataset,
                                                           build_dataset: BuildDataset,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
        self._build(build_dataset, make_franka_dataset, goal_action_filter=["flip forward"])
        assert "lookback clamped for up to 0 starts" in capsys.readouterr().out

    def test_freeze_between_allowed_goals_is_preserved(self, make_franka_dataset: MakeDataset,
                                                       build_dataset: BuildDataset) -> None:
        dataset = self._build(build_dataset, make_franka_dataset, goal_action_filter=["flip forward", "done"],
                              freeze_on_goal_change=True, previous_front_frame_lookback_weights="1-1_1")
        sample = sample_at(dataset, 0, 7)  # previous_index = 7 - 4 = 3 ("flip forward")
        assert sample["lang"] == "flip forward"
        assert arm_rows(sample) == [row_base(0, 6)] * 4  # hold the last pre-start action

    def test_clamp_prevents_freeze_from_excluded_goal(self, make_franka_dataset: MakeDataset,
                                                      build_dataset: BuildDataset) -> None:
        dataset = self._build(build_dataset, make_franka_dataset, goal_action_filter=["done"],
                              freeze_on_goal_change=True, previous_front_frame_lookback_weights="1-1_1")
        sample = sample_at(dataset, 0, 7)  # previous_index clamped from 3 to 6
        assert sample["lang"] == "done"
        assert arm_rows(sample) == [row_base(0, t) for t in (7, 8, 9, 10)]


class TestActionNormBand:
    def test_valid_min_max_uses_valid_rows(self, make_franka_dataset: MakeDataset,
                                           build_dataset: BuildDataset) -> None:
        root = make_franka_dataset([[PICK] * 5, [PICK] * 5], is_valid=[[True] * 3 + [False] * 2] * 2)
        dataset = build_dataset(root, action_norm_band="valid_min_max")
        lo, hi = np.array(joint_vector(0, 0)), np.array(joint_vector(1, 2))
        np.testing.assert_allclose(dataset.train_data_stats["actions_min"], lo)
        np.testing.assert_allclose(dataset.train_data_stats["actions_denominator"], hi - lo, rtol=1e-6)
        assert dataset.stats["action.joint_target"]["norm_band"] == "valid_min_max"
        assert dataset.stats["action.joint_target"]["q01_quantile"] == [-1.0] * 7

    def test_default_band_uses_quantiles(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        dataset = build_dataset(make_franka_dataset([[PICK] * 3]))
        np.testing.assert_allclose(dataset.train_data_stats["actions_min"], [-1.0] * 7)
        np.testing.assert_allclose(dataset.train_data_stats["actions_denominator"], [2.0] * 7)

    def test_unknown_band(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        with pytest.raises(ValueError, match="action_norm_band"):
            build_dataset(make_franka_dataset([[PICK] * 3]), action_norm_band="minmax")


class TestVideos:
    def test_frames_decode(self, make_franka_dataset: MakeDataset, build_dataset: BuildDataset) -> None:
        root = make_franka_dataset([[PICK] * 6], with_videos=True)
        dataset = build_dataset(root, camera_keys=[FRONT_CAM_KEY], use_previous_front_frame=True)
        images = sample_at(dataset, 0, 3)["images"]
        for key in (FRONT_CAM_KEY, "previous_front_frame"):
            assert images[key].shape == (VIDEO_H, VIDEO_W, 3)
            assert images[key].dtype == np.uint8

    def test_cameras_discovered_from_info(self, make_franka_dataset: MakeDataset,
                                          build_dataset: BuildDataset) -> None:
        dataset = build_dataset(make_franka_dataset([[PICK] * 3], with_videos=True))
        assert set(dataset.camera_keys) == {FRONT_CAM_KEY, GRIPPER_CAM_KEY}
        assert set(dataset[0]["images"]) == {FRONT_CAM_KEY, GRIPPER_CAM_KEY}
