from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from omegaconf import OmegaConf

from hvla.dataloader.episode_filtering.dataset_filters import (
    DatasetFilter, DropJointJumpEpisodes, DropSmallRotateEpisodes, DEFAULT_FILTER_MODULE,
)

DEFAULT_FILTER_CONFIGS_DIR = (
    Path(__file__).resolve().parents[2] / "config" / "episode_filtering"
)


def resolve_filter_config_path(config_arg: str, *, configs_dir: Path | None = None) -> Path:
    """Resolve a filter config path or preset name to an existing file."""
    configs_dir = DEFAULT_FILTER_CONFIGS_DIR if configs_dir is None else configs_dir
    raw = Path(config_arg)
    candidates = [
        raw,
        raw.with_suffix(".yaml") if raw.suffix != ".yaml" else raw,
        configs_dir / raw.name,
        configs_dir / f"{raw.name}.yaml",
        configs_dir / raw,
        configs_dir / f"{raw}.yaml",
    ]
    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"filter config not found for {config_arg!r}; tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def load_dataset_filter_cfg(
    config_arg: str,
    overrides: list[str] | None = None,
    *,
    configs_dir: Path | None = None,
) -> dict:
    path = resolve_filter_config_path(config_arg, configs_dir=configs_dir)
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(cfg, resolve=True)


def dataset_filter_from_cfg(cfg: dict) -> DatasetFilter | None:
    qualname: str | None = cfg.get("type", None)
    if qualname is None:
        return None

    module_path, _, cls_name = qualname.rpartition(".")
    if not module_path:
        module_path = DEFAULT_FILTER_MODULE

    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name, None)

    if cls is None:
        raise ValueError(f"Class {cls_name!r} not found in {module_path=!r} for {qualname=!r}")

    if not isinstance(cls, type):
        raise ValueError(f"{cls_name!r} in {module_path=!r} is not a type")

    if not issubclass(cls, DatasetFilter):
        raise TypeError(f"{qualname} is not a DatasetFilter")

    return cls(**{k: v for k, v in cfg.items() if k != "type"})


def to_dotlist(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    def walk(x, prefix=""):
        out = []
        if isinstance(x, dict):
            for k, v in x.items():
                key = f"{prefix}.{k}" if prefix else k
                out.extend(walk(v, key))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                key = f"{prefix}.{i}"
                out.extend(walk(v, key))
        else:
            if x is None:
                x = "null"
            out.append(f"{prefix}={x}")
        return out

    return walk(cfg)


def _optional_float(value) -> float | None:
    if value is None or str(value).strip().lower() in ("", "null", "none"):
        return None
    return float(value)


def quality_filters_from_cfg(cfg) -> list[DatasetFilter]:
    """Episode-quality filters, applied identically to train and val."""
    filters: list[DatasetFilter] = []
    max_joint_delta = _optional_float(getattr(cfg, "filter_joint_jumps", None))
    if max_joint_delta is not None:
        filters.append(DropJointJumpEpisodes(max_joint_delta=max_joint_delta))
    min_rotate_deg = _optional_float(getattr(cfg, "filter_small_rotates", None))
    if min_rotate_deg is not None:
        filters.append(DropSmallRotateEpisodes(min_rotate_deg=min_rotate_deg))
    return filters


def spatial_dataset_filters_from_cfg(cfg) -> tuple[DatasetFilter | None, DatasetFilter | None]:
    filter_config = getattr(cfg, "filter_config", None)
    if filter_config is None:
        return None, None

    overrides = getattr(cfg, "filter_override", None)
    if overrides is not None:
        overrides = to_dotlist(overrides)
    cfg_dict = load_dataset_filter_cfg(str(filter_config), overrides)

    val_complements = cfg_dict.pop("val_complements", None) or []
    if isinstance(val_complements, str):  # in case yaml format doesn't ensure a list
        val_complements = [val_complements]

    if cfg_dict.get("spatial_region", None) is None:
        return None, None

    train_filter = dataset_filter_from_cfg(cfg_dict)

    if val_complements:
        val_overrides = {complement: train_filter.get_complement(complement) for complement in val_complements}
        val_filter = dataset_filter_from_cfg({**cfg_dict, **val_overrides})
        train_filter.val_is_complement = val_filter.val_is_complement = True
        return train_filter, val_filter

    return train_filter, train_filter


def complementary_dataset_filters_from_cfg(cfg) -> tuple[list[DatasetFilter], list[DatasetFilter]]:
    """(train_filters, val_filters). An episode is kept only if every filter in the list keeps it."""
    train_spatial, val_spatial = spatial_dataset_filters_from_cfg(cfg)
    quality = quality_filters_from_cfg(cfg)
    train_filters = ([train_spatial] if train_spatial is not None else []) + quality
    val_filters = ([val_spatial] if val_spatial is not None else []) + quality
    return train_filters, val_filters


def complementary_dataset_filters_from_args(args: argparse.Namespace) -> tuple[list[DatasetFilter], list[DatasetFilter]]:
    return complementary_dataset_filters_from_cfg(OmegaConf.create({
        "filter_config": getattr(args, "filter_config", None),
        "filter_override": getattr(args, "filter_override", None) or [],
        "filter_joint_jumps": getattr(args, "filter_joint_jumps", None),
        "filter_small_rotates": getattr(args, "filter_small_rotates", None),
    }))
