"""hvla/dataloader/episode_filtering/dataset_filters.py."""
from __future__ import annotations

import pyarrow as pa
import pytest

from hvla.dataloader.episode_filtering.dataset_filters import ClassifyEpisodesSpatial

COL = "block_pose"
UNIT_BOX = (0.0, 1.0, 0.0, 1.0)


def poses(*xy: tuple[float, float]) -> pa.Table:
    return pa.table({COL: [[x, y, 0.1] for x, y in xy]})


EPISODES = {
    0: poses((0.5, 0.5), (0.2, 0.8)),   # always inside
    1: poses((2.0, 2.0), (-1.0, 0.5)),  # never inside
    2: poses((0.5, 0.5), (2.0, 0.5)),   # crosses the boundary
}


def spatial(region: str, margin: float = 0.0, **kwargs: object) -> ClassifyEpisodesSpatial:
    return ClassifyEpisodesSpatial(region, object_col=COL, bounds=UNIT_BOX, inside_outside_margin=margin, **kwargs)


class TestClassifyEpisodesSpatial:
    def test_inside_partition(self) -> None:
        inliers, outliers, straddlers = spatial("inside").apply(EPISODES)
        assert (inliers, outliers, straddlers) == ({0}, {1}, {2})
        assert inliers | outliers | straddlers == set(EPISODES)

    def test_outside_swaps_inliers_and_outliers(self) -> None:
        assert spatial("outside").apply(EPISODES) == ({1}, {0}, {2})

    @pytest.mark.parametrize(("region", "expected"), [
        ("inside", (0.1, 0.9, 0.1, 0.9)),
        ("outside", (-0.1, 1.1, -0.1, 1.1)),
    ])
    def test_margin_moves_bounds(self, region: str, expected: tuple[float, ...]) -> None:
        assert spatial(region, margin=0.2).bounds == pytest.approx(expected)

    def test_margin_changes_classification(self) -> None:
        near_edge = {0: poses((0.05, 0.5))}
        assert spatial("inside").apply(near_edge)[0] == {0}
        assert spatial("inside", margin=0.2).apply(near_edge)[1] == {0}

    def test_boundary_counts_as_inside(self) -> None:
        assert spatial("inside").apply({0: poses((0.0, 0.0), (1.0, 1.0))})[0] == {0}

    def test_single_frame_and_empty(self) -> None:
        assert spatial("inside").apply({7: poses((0.5, 0.5))}) == ({7}, set(), set())
        assert spatial("inside").apply({}) == (set(), set(), set())

    def test_required_cols(self) -> None:
        assert spatial("inside").required_cols == [COL]

    @pytest.mark.parametrize("kwargs", [
        {"spatial_region": "middle"},
        {"spatial_region": "inside", "bounds": (0.0, 1.0, 0.0)},
        {"spatial_region": "inside", "inside_outside_margin": -0.1},
    ])
    def test_invalid_arguments(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            ClassifyEpisodesSpatial(**kwargs)

    def test_get_complement(self) -> None:
        assert spatial("inside").get_complement("spatial_region") == "outside"
        assert spatial("outside").get_complement("spatial_region") == "inside"
        with pytest.raises(ValueError):
            spatial("inside").get_complement("bounds")

    @pytest.mark.xfail(strict=True, reason=(
        "ClassifyEpisodesSpatial builds super() kwargs with prepare_kwargs_for_super(locals()), which drops the "
        "'kwargs' entry, so apply_to_datasets from the config never reaches DatasetFilter and stays None"))
    def test_apply_to_datasets_is_kept(self) -> None:
        assert spatial("inside", apply_to_datasets=["ds_a"]).apply_to_datasets == ["ds_a"]
