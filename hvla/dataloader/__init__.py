import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from hvla.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe", accelerator=None): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from hvla.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        num_workers = int(getattr(vla_dataset_cfg, "num_workers", 4))
        prefetch_factor = int(getattr(vla_dataset_cfg, "prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            # shuffle=True
        )        
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:

            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader, None
    elif dataset_py == "lerobot_datasets_yam":
        from functools import partial
        from hvla.dataloader.lerobot_datasets import (
            get_vla_dataset,
            yamrobot_collate_fn,
            _load_camera_info,
        )
        from hvla.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

        vla_dataset_cfg = cfg.datasets.vla_data
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        # Load camera intrinsics once (lru_cache avoids redundant I/O).
        # Path: {data_root_dir}/{first_dataset_name}/meta/head_camera_info.json
        # Written by build_lerobot_dataset_yam.py.
        data_root = Path(vla_dataset_cfg.data_root_dir)
        first_ds_name = DATASET_NAMED_MIXTURES[vla_dataset_cfg.data_mix][0][0]
        camera_info_path = data_root / first_ds_name / "meta" / "head_camera_info.json"
        camera_info = _load_camera_info(str(camera_info_path))

        max_points = int(getattr(vla_dataset_cfg, "max_points", 10_000))
        num_workers = int(getattr(vla_dataset_cfg, "num_workers", 4))
        prefetch_factor = int(getattr(vla_dataset_cfg, "prefetch_factor", 2))
        yam_collate = partial(yamrobot_collate_fn, camera_info=camera_info, max_points=max_points)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=yam_collate,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

        return vla_train_dataloader, None

    elif dataset_py == "lerobot_datasets_yam_multimodal":
        from functools import partial
        import torch.utils.data as tud
        from hvla.dataloader.lerobot_datasets import (
            get_vla_dataset,
            yamrobot_multimodal_collate_fn,
            _load_camera_info,
        )
        from hvla.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

        vla_dataset_cfg = cfg.datasets.vla_data
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        data_root     = Path(vla_dataset_cfg.data_root_dir)
        first_ds_name = DATASET_NAMED_MIXTURES[vla_dataset_cfg.data_mix][0][0]
        meta_dir      = data_root / first_ds_name / "meta"

        head_cam_info    = _load_camera_info(str(meta_dir / "head_camera_info.json"))
        wrist_r_cam_info = _load_camera_info(str(meta_dir / "wrist_r_camera_info.json"))
        wrist_l_cam_info = _load_camera_info(str(meta_dir / "wrist_l_camera_info.json"))

        ptmap_size = int(getattr(vla_dataset_cfg, "ptmap_size", 224))
        include_ptmap = bool(getattr(vla_dataset_cfg, "include_ptmap", True))
        prefetch_factor = int(getattr(vla_dataset_cfg, "prefetch_factor", 2))
        yam_mm_collate = partial(
            yamrobot_multimodal_collate_fn,
            head_camera_info=head_cam_info,
            wrist_r_camera_info=wrist_r_cam_info,
            wrist_l_camera_info=wrist_l_cam_info,
            ptmap_size=ptmap_size,
            augment=True,
            include_ptmap=include_ptmap,
        )
        yam_mm_collate_val = partial(
            yamrobot_multimodal_collate_fn,
            head_camera_info=head_cam_info,
            wrist_r_camera_info=wrist_r_cam_info,
            wrist_l_camera_info=wrist_l_cam_info,
            ptmap_size=ptmap_size,
            augment=False,
            include_ptmap=include_ptmap,
        )

        # --- Validation split (per sub-dataset, fixed seed=42) ---
        # val_split_mode: "episodes" = hold out whole episodes (good for testing); "steps" = random 1% of steps (legacy).
        import copy
        from collections import defaultdict
        val_split_mode = getattr(vla_dataset_cfg, "val_split_mode", "episodes")
        val_ratio = float(getattr(vla_dataset_cfg, "val_ratio", 0.01))
        val_rng = np.random.default_rng(42)
        val_subsets = []
        for sub_ds in vla_dataset.datasets:
            all_steps = sub_ds._all_steps  # list of (trajectory_id, base_index)
            if val_split_mode == "episodes":
                # Group steps by trajectory_id; shuffle episode indices; hold out val_ratio of episodes entirely.
                by_traj = defaultdict(list)
                for (tid, bidx) in all_steps:
                    by_traj[tid].append((tid, bidx))
                traj_ids = np.array(list(by_traj.keys()))
                val_rng.shuffle(traj_ids)
                n_val_ep = max(1, int(len(traj_ids) * val_ratio))
                val_trajs = set(traj_ids[:n_val_ep])
                val_steps = [s for s in all_steps if s[0] in val_trajs]
                train_steps = [s for s in all_steps if s[0] not in val_trajs]
            else:
                # Legacy: random val_ratio of steps (no full episodes).
                n = len(all_steps)
                idx = np.arange(n)
                val_rng.shuffle(idx)
                n_val = max(1, int(n * val_ratio))
                val_steps = [all_steps[j] for j in idx[:n_val]]
                train_steps = [all_steps[j] for j in idx[n_val:]]
            val_ds = copy.copy(sub_ds)
            val_ds._all_steps = val_steps
            val_subsets.append(val_ds)
            sub_ds._all_steps = train_steps
        # Rebuild mixture's step_order/step_pos/lengths after modifying _all_steps
        if vla_dataset._step_order:  # only populated when _sequential_step_sampling=True
            for i, sub_ds in enumerate(vla_dataset.datasets):
                n_train = len(sub_ds._all_steps)
                order = np.arange(n_train)
                np.random.default_rng(vla_dataset.seed).shuffle(order)
                vla_dataset._step_order[i] = order
                vla_dataset._step_pos[i] = 0
        vla_dataset._dataset_lengths = np.array([len(ds) for ds in vla_dataset.datasets])
        val_dataset = tud.ConcatDataset(val_subsets)
        logger.info(
            f"Val split: {val_split_mode}, train steps: {sum(len(ds) for ds in vla_dataset.datasets)}, "
            f"val steps: {len(val_dataset)}"
        )
        # ---

        def _worker_init_fn(worker_id):
            """Offset each worker's _step_pos so workers cover non-overlapping
            segments of the shuffled step_order, preventing 8x duplication."""
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            num_workers = worker_info.num_workers
            for i in range(len(dataset.datasets)):
                n = len(dataset.datasets[i].all_steps)
                dataset._step_pos[i] = worker_id * (n // num_workers)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=yam_mm_collate,
            num_workers=int(getattr(vla_dataset_cfg, "num_workers", 4)),
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
        )

        num_workers_val = int(getattr(vla_dataset_cfg, "num_workers", 4))
        vla_val_dataloader = DataLoader(
            val_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=yam_mm_collate_val,
            num_workers=num_workers_val,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            shuffle=False,
        )

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

        return vla_train_dataloader, vla_val_dataloader

    elif dataset_py == "franka_multimodal":
        from hvla.dataloader.franka_dataset import build_franka_multimodal_dataloader
        return build_franka_multimodal_dataloader(cfg)

    elif dataset_py == "lerobot_datasets_yam_sharded":
        from hvla.dataloader.sharded_yam_dataset import build_yam_sharded_dataloader
        return build_yam_sharded_dataloader(cfg, accelerator=accelerator)

    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]

        return vlm_train_dataloader, None
