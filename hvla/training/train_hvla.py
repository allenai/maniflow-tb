"""
The trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re
from PIL import Image

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from hvla.training.trainer_utils.trainer_tools import normalize_dotlist_args
from hvla.model.framework import build_framework
from hvla.training.trainer_utils.trainer_tools import TrainerUtils
from hvla.training.trainer_utils.trainer_tools import build_param_lr_groups, save_first_batch_data
from hvla.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from hvla.training.trainer_utils.arch_diagram import generate_arch_diagram
from hvla.training.muon_optim import build_muon_optimizer_and_scheduler

def _use_muon(cfg) -> bool:
    """True when the Muon+AdamW optimizer (DDP, no DeepSpeed) is selected via config."""
    optimizer_cfg = getattr(cfg.trainer, "optimizer", None)
    name = getattr(optimizer_cfg, "name", "") if optimizer_cfg is not None else ""
    return str(name or "").lower() == "muon"


def build_accelerator(cfg) -> Accelerator:
    """Create the Accelerator for this run.

    Default: DeepSpeed ZeRO-2 (unchanged behavior). When the Muon optimizer is selected we
    fall back to plain DDP, because distributed Muon runs its own parameter ``all_gather``
    that is incompatible with ZeRO gradient / optimizer-state partitioning.
    """
    if _use_muon(cfg):
        from accelerate.utils import DistributedDataParallelKwargs

        grad_accum = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1) or 1)
        # find_unused_parameters: the policy registers conditionally-used params (e.g.
        # heatmap / target modules that only receive gradients when the batch carries the
        # corresponding inputs), which would otherwise trip DDP's all-params-get-grad check.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        return Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=grad_accum,
            kwargs_handlers=[ddp_kwargs],
        )

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_clipping=1.0,
        gradient_accumulation_steps=1,
        offload_optimizer_device="none",
        offload_param_device="none",
        zero3_init_flag=False,
    )
    return Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="bf16")


def _untie_shared_tensors(state_dict: dict) -> dict:
    """Clone tensors that share storage so safetensors can serialize them (e.g. T5's tied
    embed_tokens/shared weights). Keeps every key; only the duplicate is cloned."""
    seen: dict[int, str] = {}
    out: dict = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        ptr = v.untyped_storage().data_ptr()
        out[k] = v.detach().clone() if ptr in seen else v
        seen.setdefault(ptr, k)
    return out

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)

# Written last, once every payload of a checkpoint is durably in place.  Resume only
# considers marked checkpoints, so a job killed mid-save cannot poison the next attempt.
CHECKPOINT_MARKER_SUFFIX = ".done"

# Substrings that torch.load / safetensors emit for truncated or damaged files.  Matched on
# the message because load_pretrained_backbones re-raises everything as RuntimeError.
_CORRUPTION_HINTS = (
    "PytorchStreamReader",
    "central directory",
    "Error while deserializing header",
    "HeaderTooLarge",
    "MetadataIncompleteBuffer",
    "InvalidHeaderDeserialization",
    "Ran out of input",
    "pickle data was truncated",
    "invalid load key",
    "unexpected EOF",
)


def _looks_like_corruption(error: BaseException) -> bool:
    """True only with positive evidence of a damaged file.

    Type alone is not enough: OutOfMemoryError subclasses RuntimeError, and a key/shape
    mismatch is a config problem, so neither may get a checkpoint quarantined.
    """
    if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
        return False
    if isinstance(error, EOFError):
        return True
    message = f"{error} {error.__cause__ or ''}".lower()
    return any(hint.lower() in message for hint in _CORRUPTION_HINTS)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # stats.json is saved after dataloader construction (see prepare_data)
        # to ensure merged stats are used when co-training on multiple datasets

    return output_dir



def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from hvla.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader, vla_val_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, accelerator=accelerator)

    # Save (potentially merged) stats.json into checkpoint dir for inference
    if not dist.is_initialized() or dist.get_rank() == 0:
        train_ds = vla_train_dataloader.dataset
        stats = getattr(train_ds, "stats", None)
        if stats is not None:
            stats_dst = output_dir / "stats.json"
            with open(stats_dst, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved dataset stats.json to {stats_dst}")

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader, vla_val_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler

class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator, vla_val_dataloader=None):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        # DeepSpeed clips grads and shards optimizer state itself; the DDP/Muon path must
        # clip explicitly and can restore optimizer state normally on resume.
        self._use_deepspeed = self.accelerator.distributed_type == DistributedType.DEEPSPEED

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # Multi-node WandB guard: only node-0 / rank-0 should touch WandB.
        # In Beaker multi-node jobs each replica runs the full script; without
        # this guard every replica's rank-0 GPU creates its own WandB run.
        _node_rank = int(os.environ.get("BEAKER_REPLICA_RANK",
                                        os.environ.get("NODE_RANK", "0")))
        self._wandb_leader = (self.accelerator.is_main_process and _node_rank == 0)
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights (optimizer/scheduler/EMA deferred to after accelerator.prepare)
        self._init_checkpointing() # TODO merge with load pretrained weights

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # accelerate wraps the loader in a BatchSamplerShard whose `.sampler` is a placeholder,
        # but the shard still iterates this original sampler, so reseeding it controls the shuffle.
        self._train_sampler = getattr(self.vla_train_dataloader, "sampler", None)

        # initialize distributed training components
        if self.vla_val_dataloader is not None:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_val_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vla_val_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        # Restore optimizer/scheduler/EMA now that the optimizer is DeepSpeed-wrapped
        self._restore_pending_training_state()

        self._init_wandb()

    def _fast_forward_lr_scheduler(self):
        """Advance the scheduler to completed_steps in closed form instead of replaying every step."""
        if self.completed_steps <= 0:
            return
        try:
            self.lr_scheduler.last_epoch = self.completed_steps
            if hasattr(self.lr_scheduler, "_step_count"):
                self.lr_scheduler._step_count = self.completed_steps + 1
            values = self.lr_scheduler.get_lr()
            for group, lr in zip(self.lr_scheduler.optimizer.param_groups, values):
                group["lr"] = lr
            self.lr_scheduler._last_lr = [g["lr"] for g in self.lr_scheduler.optimizer.param_groups]
        except Exception as e:
            logger.warning(f"Closed-form LR fast-forward failed ({e}); replaying scheduler steps.")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
        logger.info(
            f"LR scheduler advanced to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
        )

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        _node = os.environ.get("BEAKER_REPLICA_RANK", os.environ.get("NODE_RANK", "0"))
        logger.info(
            f"[WandB guard] node_rank={_node}  "
            f"is_main_process={self.accelerator.is_main_process}  "
            f"_wandb_leader={self._wandb_leader}  "
            f"global_rank={dist.get_rank() if dist.is_initialized() else '?'}  "
            f"world_size={dist.get_world_size() if dist.is_initialized() else '?'}"
        )
        if self._wandb_leader:
            wandb_id_file = os.path.join(self.config.output_dir, "wandb_run_id.txt")
            previous_run_id = None
            if self._resume_meta is not None and os.path.exists(wandb_id_file):
                with open(wandb_id_file) as f:
                    previous_run_id = f.read().strip() or None

            # Each resume gets a fresh wandb run rather than appending to (or rewinding) the old one;
            # the previous id is kept in the config so the attempts can be linked.
            wandb_config, wandb_tags = self._build_wandb_config()
            if previous_run_id:
                wandb_config["resumed_from_wandb_id"] = previous_run_id
                wandb_config["resumed_from_step"] = self.completed_steps
            run = wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
                config=wandb_config,
                tags=wandb_tags,
            )
            with open(wandb_id_file, "w") as f:
                f.write(run.id)

    def _build_wandb_config(self):
        """Extract structured config and tags for wandb grouping/filtering."""
        fw = self.config.framework
        ds = self.config.datasets.vla_data
        shape_obs = fw.shape_meta.obs

        _is_active = lambda k: getattr(shape_obs, k, None) is not None

        has_front_rgb = _is_active("front_rgb")
        has_gripper_rgb = _is_active("gripper_rgb")
        has_front_depth = _is_active("front_depth")
        has_gripper_depth = _is_active("gripper_depth")
        use_rgb = getattr(fw, "use_rgb", True)
        use_ptmap = getattr(fw, "use_ptmap", False)
        pcn = getattr(fw, "per_camera_ptmap_bounds", None) is not None

        depth_encoder = getattr(fw, "ptmap_encoder_type", "dense")
        ptmap_backbone = getattr(fw, "ptmap_backbone", "convnextv2_tiny.fcmae_ft_in22k_in1k")
        rgb_backbone = getattr(fw, "rgb_backbone", "vit_base_patch16_clip_224.openai")

        target_mode = getattr(fw, "target_injection_mode", None) or "none"
        lang_conditioned = getattr(fw, "language_conditioned", False)
        lang_mode = getattr(fw, "language_mode", "action_only")

        action_mode = getattr(ds, "action_mode", "abs")
        horizon = getattr(fw, "horizon", 50)
        n_action_steps = getattr(fw, "n_action_steps", 25)

        # Short backbone name for tags
        bb_short = ptmap_backbone
        if "dinov3" in ptmap_backbone:
            bb_short = "dinov3s" if "small" in ptmap_backbone else "dinov3b"
        elif "dinov2" in ptmap_backbone:
            bb_short = "dinov2s" if "small" in ptmap_backbone else "dinov2b"
        elif "convnext" in ptmap_backbone:
            bb_short = "convnextv2"

        # Dataset roots
        _raw_root = getattr(ds, "data_root_dir", "")
        if isinstance(_raw_root, (list, tuple)):
            dataset_roots = [str(r) for r in _raw_root]
        elif isinstance(_raw_root, str) and "," in _raw_root:
            dataset_roots = [r.strip() for r in _raw_root.split(",")]
        else:
            dataset_roots = [str(_raw_root)]
        dataset_names = [r.rstrip("/").split("/")[-1] for r in dataset_roots]

        # Filter info
        filter_config = getattr(ds, "filter_config", None)
        filter_enabled = filter_config is not None and str(filter_config) != "null"
        filter_override = getattr(ds, "filter_override", None)
        filter_region = None
        filter_apply_to = None
        if filter_enabled and filter_override is not None:
            filter_region = getattr(filter_override, "spatial_region", None)
            _atd = getattr(filter_override, "apply_to_datasets", None)
            if _atd is not None:
                filter_apply_to = list(_atd) if not isinstance(_atd, str) else [_atd]

        def _optional_float(value):
            return None if value is None or str(value).lower() in ("", "null", "none") else float(value)

        filter_joint_jumps = _optional_float(getattr(ds, "filter_joint_jumps", None))
        filter_small_rotates = _optional_float(getattr(ds, "filter_small_rotates", None))

        wandb_config = {
            "rgb/front": has_front_rgb,
            "rgb/gripper": has_gripper_rgb,
            "rgb/enabled": use_rgb,
            "rgb/backbone": rgb_backbone,
            "depth/front": has_front_depth,
            "depth/gripper": has_gripper_depth,
            "depth/enabled": use_ptmap,
            "depth/encoder": depth_encoder,
            "depth/backbone": ptmap_backbone,
            "depth/per_cam_norm": pcn,
            "target/mode": target_mode,
            "language/enabled": lang_conditioned,
            "language/mode": lang_mode,
            "action/mode": action_mode,
            "action/horizon": horizon,
            "action/n_action_steps": n_action_steps,
            "training/batch_size": getattr(ds, "per_device_batch_size", 128),
            "training/num_gpus": self.accelerator.num_processes,
            "training/global_batch": (
                getattr(ds, "per_device_batch_size", 128)
                * self.accelerator.num_processes
            ),
            "dataset/names": dataset_names,
            "dataset/num_datasets": len(dataset_roots),
            "dataset/max_episodes": getattr(ds, "max_episodes", None),
            "filter/enabled": filter_enabled,
            "filter/config": str(filter_config) if filter_enabled else None,
            "filter/spatial_region": str(filter_region) if filter_region else None,
            "filter/apply_to_datasets": filter_apply_to,
            "filter/joint_jumps_max_rad": filter_joint_jumps,
            "filter/small_rotates_min_deg": filter_small_rotates,
            "dataset/ratio": str(getattr(ds, "dataset_ratio", None)),
        }

        tags = []
        if has_front_rgb:
            tags.append("front_rgb")
        if has_gripper_rgb:
            tags.append("gripper_rgb")
        if has_front_depth:
            tags.append("front_depth")
        if has_gripper_depth:
            tags.append("gripper_depth")
        if not use_rgb:
            tags.append("no_rgb")
        if use_ptmap:
            tags.append(f"depth_{depth_encoder}")
            tags.append(bb_short)
        if pcn:
            tags.append("per_cam_norm")
        if lang_conditioned:
            tags.append(f"lang_{lang_mode}")
        tags.append(f"h{horizon}_a{n_action_steps}")
        tags.append(action_mode)
        if len(dataset_roots) > 1:
            tags.append(f"{len(dataset_roots)}_datasets")
        for dn in dataset_names:
            tags.append(dn)
        if filter_enabled:
            tags.append(f"filter_{filter_config}")
            if filter_region:
                tags.append(f"spatial_{filter_region}")
        if filter_joint_jumps is not None:
            tags.append(f"nojump{filter_joint_jumps:g}")
        if filter_small_rotates is not None:
            tags.append(f"minrot{filter_small_rotates:g}")

        # wandb rejects the whole run (HTTP 400) if any tag exceeds 64 characters.
        tags = list(dict.fromkeys(t[:64] for t in tags if t))

        return wandb_config, tags

    def _resume_mode(self):
        """auto (resume if a finished checkpoint exists), force (require one), or never."""
        mode = getattr(self.config.trainer, "resume_mode", None)
        if mode is None:
            # Configs that predate resume_mode keep their old behavior.
            mode = "force" if getattr(self.config.trainer, "is_resume", False) else "never"
        mode = str(mode).strip().lower()
        if mode not in ("auto", "force", "never"):
            raise ValueError(f"trainer.resume_mode must be one of auto/force/never, got {mode!r}")
        return mode

    def _list_resume_candidates(self):
        """Newest-first checkpoint descriptors.

        Only checkpoints with a `.done` marker are trusted.  The marker is written after every
        rank's payload is in place, so a job killed mid-save leaves an unmarked partial that is
        skipped instead of being picked up as "latest" and crashing every retry.
        """
        markers = []
        for name in os.listdir(self.checkpoint_dir):
            if not re.match(rf"^steps_\d+{re.escape(CHECKPOINT_MARKER_SUFFIX)}$", name):
                continue
            try:
                with open(os.path.join(self.checkpoint_dir, name)) as f:
                    meta = json.load(f)
                meta["marker"] = name
                meta["step"] = int(meta["step"])
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Ignoring unreadable checkpoint marker {name}: {e}")
                continue
            if os.path.exists(os.path.join(self.checkpoint_dir, meta.get("model_file", ""))):
                markers.append(meta)
        if markers:
            markers.sort(key=lambda m: m["step"], reverse=True)
            return markers
        return self._legacy_resume_candidates()

    def _legacy_resume_candidates(self):
        """Checkpoints written before markers existed; they have no per-rank optimizer state."""
        candidates = []
        pattern = re.compile(r"^steps_(\d+)_(pytorch_model\.pt|model\.safetensors)$")
        for name in os.listdir(self.checkpoint_dir):
            m = pattern.match(name)
            if not m or not os.path.isfile(os.path.join(self.checkpoint_dir, name)):
                continue
            step = int(m.group(1))
            candidates.append({
                "step": step,
                "model_file": name,
                "training_state": f"steps_{step}_training_state.pt",
                "state_dir": None,
                "world_size": None,
                "marker": None,
            })
        candidates.sort(key=lambda c: c["step"], reverse=True)
        if candidates:
            logger.info(f"No checkpoint markers found; falling back to {len(candidates)} legacy checkpoint(s).")
        return candidates

    def _broadcast_from_main(self, value):
        """Share a rank-0 decision with every rank so they cannot diverge."""
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return value
        box = [value if self.accelerator.is_main_process else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _agreed(self, ok: bool) -> bool:
        """True only when every rank reports success."""
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return ok
        t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=self.accelerator.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def _quarantine_checkpoint(self, meta):
        """Rename a checkpoint that failed to load to *.corrupt so it leaves the candidate set."""
        if self.accelerator.is_main_process:
            for key in ("marker", "model_file", "training_state", "state_dir"):
                name = meta.get(key)
                if not name:
                    continue
                path = os.path.join(self.checkpoint_dir, name)
                if os.path.exists(path):
                    try:
                        os.replace(path, path + ".corrupt")
                    except OSError as e:
                        logger.warning(f"Could not quarantine {path}: {e}")
        self.accelerator.wait_for_everyone()

    def _init_checkpointing(self):
        """Pick the checkpoint to resume from and load its weights and scheduler/EMA file.

        Runs BEFORE accelerator.prepare(); optimizer state needs the wrapped optimizer and is
        restored in _restore_pending_training_state() afterwards.
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        resume_mode = self._resume_mode()
        self.resume_from_checkpoint = pretrained_checkpoint
        self.completed_steps = 0
        self._resume_meta = None
        self._pending_training_state = None

        candidates = []
        if resume_mode != "never":
            candidates = self._broadcast_from_main(
                self._list_resume_candidates() if self.accelerator.is_main_process else None
            ) or []

        for meta in candidates:
            model_file = os.path.join(self.checkpoint_dir, meta["model_file"])
            training_state_path = os.path.join(self.checkpoint_dir, meta["training_state"])
            error, training_state = None, None
            try:
                self.model = self.load_pretrained_backbones(self.model, model_file, reload_modules=None)
                if os.path.exists(training_state_path):
                    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
            except Exception as e:
                error = e
            if not self._agreed(error is None):
                # Quarantining is irreversible, so require every failing rank to show positive
                # evidence of a damaged file; anything else (OOM, key mismatch) must surface.
                if not self._agreed(error is None or _looks_like_corruption(error)):
                    raise RuntimeError(
                        f"Checkpoint {meta['model_file']} failed to load for a reason that does not look "
                        f"like file corruption ({error}); leaving checkpoints untouched."
                    ) from error
                logger.warning(
                    f"Checkpoint {meta['model_file']} is corrupt ({error}); renaming it to *.corrupt "
                    "and trying the previous one."
                )
                self._quarantine_checkpoint(meta)
                continue

            self._resume_meta = meta
            self.completed_steps = meta["step"]
            self.resume_from_checkpoint = model_file
            self._pending_training_state = training_state
            if training_state is None:
                logger.warning(f"No {meta['training_state']} found; scheduler/EMA will start fresh.")
            logger.info(f"Resuming training from {model_file} at step {self.completed_steps}")
            return

        if resume_mode == "force":
            raise RuntimeError(
                f"trainer.resume_mode=force but no loadable checkpoint exists in {self.checkpoint_dir}. "
                "Check the run id passed to --resume."
            )
        if candidates:
            logger.warning("Every checkpoint was corrupt; starting training from scratch.")

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")


    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _restore_pending_training_state(self):
        """Apply optimizer/scheduler/EMA state for the checkpoint picked in _init_checkpointing().

        Must be called AFTER accelerator.prepare() so self.optimizer is already wrapped.
        Optimizer state comes from the per-rank state dir; the `optimizer` entry in older
        _training_state.pt files is rank 0's shard only, so it is used solely for
        single-process runs.
        """
        if self._resume_meta is None:
            return
        training_state = self._pending_training_state or {}
        self._pending_training_state = None  # free memory

        # --- optimizer ---
        if not self._load_optimizer_state():
            if (
                not self._use_deepspeed
                and self.accelerator.num_processes == 1
                and training_state.get("optimizer")
            ):
                self.optimizer.load_state_dict(training_state["optimizer"])
                logger.info("Optimizer state restored from legacy training state.")
            else:
                logger.warning("Optimizer state unavailable for this checkpoint; Adam/Muon moments restart from zero.")
        # --- lr_scheduler ---
        if training_state.get("lr_scheduler"):
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
            logger.info("LR scheduler state restored.")
        else:
            self._fast_forward_lr_scheduler()
        # --- EMA ---
        framework = self.accelerator.unwrap_model(self.model)
        if hasattr(framework, "load_ema_state") and training_state.get("ema"):
            framework.load_ema_state(training_state["ema"])
            logger.info("EMA state restored.")

    def _save_optimizer_state(self, state_dir):
        """Save optimizer state from every rank.  Must be called on all ranks.

        Optimizer state is rank-sharded on both paths (ZeRO partitions; distributed Muon keeps
        momentum only for the matrices each rank updates), so a rank-0-only save cannot be restored.
        """
        import shutil

        # Clear a partial dir left by an attempt killed mid-save at this same step.
        if self.accelerator.is_main_process and os.path.isdir(state_dir):
            shutil.rmtree(state_dir, ignore_errors=True)
        self.accelerator.wait_for_everyone()

        if self._use_deepspeed:
            self.accelerator.save_state(state_dir)
        else:
            os.makedirs(state_dir, exist_ok=True)
            self._atomic_save(
                lambda p: torch.save(self.optimizer.state_dict(), p),
                os.path.join(state_dir, f"optimizer_rank{self.accelerator.process_index}.pt"),
            )

    def _load_optimizer_state(self):
        """Restore per-rank optimizer state saved by _save_optimizer_state(). Returns True on success."""
        state_dirname = self._resume_meta.get("state_dir")
        state_dir = os.path.join(self.checkpoint_dir, state_dirname) if state_dirname else None
        if not state_dir or not os.path.isdir(state_dir):
            logger.warning(
                f"No optimizer state saved with checkpoint step {self.completed_steps} "
                "(pre-marker checkpoint, or pruned because a newer checkpoint existed)."
            )
            return False

        saved_world_size = self._resume_meta.get("world_size")
        if saved_world_size != self.accelerator.num_processes:
            raise RuntimeError(
                f"Cannot resume: checkpoint {state_dir} was saved with {saved_world_size} GPUs but this job "
                f"runs {self.accelerator.num_processes}. Optimizer state is sharded per GPU and cannot be "
                f"remapped; resubmit with the same num_nodes x num_gpus_per_node as the original run."
            )

        error = None
        if self._use_deepspeed:
            try:
                self.accelerator.load_state(state_dir)
            except Exception as e:
                error = e
            if not self._agreed(error is None):
                # load_state may have partially applied on some ranks, so a silent fallback is unsafe.
                raise RuntimeError(
                    f"Optimizer state in {state_dir} failed to load ({error}). Delete that directory to "
                    "resume from the same step with a fresh optimizer."
                ) from error
        else:
            state = None
            try:
                path = os.path.join(state_dir, f"optimizer_rank{self.accelerator.process_index}.pt")
                state = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                error = e
            # Fall back on every rank together: mixing restored and fresh AdamW state across
            # ranks would make the replicated parameters drift apart.
            if not self._agreed(error is None):
                logger.warning(f"Optimizer state in {state_dir} failed to load ({error}); restarting it fresh.")
                return False
            self.optimizer.load_state_dict(state)
        logger.info(f"Optimizer state restored from {state_dir}")
        return True

    @staticmethod
    def _atomic_save(write_fn, path):
        """Write via a temp file and rename, so readers never observe a partial file."""
        tmp = f"{path}.tmp"
        write_fn(tmp)
        os.replace(tmp, path)

    def _save_checkpoint(self):
        """Save weights, scheduler/EMA, and per-rank optimizer state.  Must be called on all ranks.

        Ordering matters: every payload is renamed into place, all ranks sync, and only then
        does rank 0 write the `steps_N.done` marker that _list_resume_candidates() trusts.
        """
        step = self.completed_steps
        save_format = getattr(self.config.trainer, "save_format", "pt")
        if save_format == "safetensors":
            model_name = f"steps_{step}_model.safetensors"
        elif save_format == "pt":
            model_name = f"steps_{step}_pytorch_model.pt"
        else:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
        training_state_name = f"steps_{step}_training_state.pt"
        state_dirname = f"state_{step}"

        self._save_optimizer_state(os.path.join(self.checkpoint_dir, state_dirname))

        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{step}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            model_path = os.path.join(self.checkpoint_dir, model_name)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                self._atomic_save(lambda p: save_file(_untie_shared_tensors(state_dict), p), model_path)
            else:
                self._atomic_save(lambda p: torch.save(state_dict, p), model_path)

            # save training metadata
            summary_data = {
                "steps": step,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            # Scheduler + EMA (EMA weights are also read at inference time).  Optimizer state
            # lives in the per-rank state dir instead.
            framework = self.accelerator.unwrap_model(self.model)
            training_state = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "ema": framework.get_ema_state() if hasattr(framework, "get_ema_state") else None,
                "completed_steps": step,
            }
            self._atomic_save(
                lambda p: torch.save(training_state, p),
                os.path.join(self.checkpoint_dir, training_state_name),
            )

            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                config = self.config.unwrap()
            else:
                config = self.config

            logger.info("📊 Saving configuration...")
            output_dir = Path(self.config.output_dir)
            # save full config
            self._atomic_save(lambda p: OmegaConf.save(config, p), str(output_dir / "config.yaml"))
            logger.info("✅ Configuration files saved")

        # The marker must not appear until every rank's optimizer shard is durable.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            marker = {
                "step": int(step),
                "model_file": model_name,
                "training_state": training_state_name,
                "state_dir": state_dirname,
                "world_size": int(self.accelerator.num_processes),
            }
            self._atomic_save(
                lambda p: Path(p).write_text(json.dumps(marker)),
                os.path.join(self.checkpoint_dir, f"steps_{step}{CHECKPOINT_MARKER_SUFFIX}"),
            )

            # Each state dir holds fp32 optimizer moments (several x model size), so keep only the newest.
            import shutil
            for name in os.listdir(self.checkpoint_dir):
                m = re.match(r"^state_(\d+)$", name)
                if m and int(m.group(1)) != step:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, name), ignore_errors=True)

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if self._wandb_leader:
                # add learning rate 
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _log_images(self, batch):
        """Log depth (turbo colormap), RGB, and pointmap visualizations to wandb."""
        if not self._wandb_leader:
            return

        import matplotlib.cm as cm

        obs = batch["obs"]
        images_dict = {}

        # RGB cameras
        for cam in ["front", "gripper"]:
            key = f"{cam}_rgb"
            if key in obs:
                # [B, 1, 3, H, W] -> first sample, first obs step
                rgb = obs[key][0, 0].detach().cpu().float().numpy()  # [3, H, W] in [0,1]
                rgb_hwc = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]
                rgb_uint8 = (np.clip(rgb_hwc, 0, 1) * 255).astype(np.uint8)
                images_dict[f"viz/{cam}_rgb"] = wandb.Image(rgb_uint8)

        # Depth cameras (turbo colormap)
        for cam in ["front", "gripper"]:
            key = f"{cam}_depth"
            if key in obs:
                # [B, 1, 1, H, W] -> first sample, in meters
                depth = obs[key][0, 0, 0].detach().cpu().float().numpy()  # [H, W]
                valid = depth > 0
                if valid.any():
                    lo = np.percentile(depth[valid], 2)
                    hi = np.percentile(depth[valid], 98)
                    if hi > lo:
                        norm = np.clip((depth - lo) / (hi - lo), 0, 1)
                    else:
                        norm = np.zeros_like(depth)
                    norm[~valid] = 0
                    colored = (cm.turbo(norm)[:, :, :3] * 255).astype(np.uint8)
                    colored[~valid] = 0
                    images_dict[f"viz/{cam}_depth_turbo"] = wandb.Image(colored)

        # Pointmap visualization (back-project depth -> XYZ, normalize per-channel)
        if "camera_intrinsics" in obs:
            depth_cams = [cam for cam in ["front", "gripper"] if f"{cam}_depth" in obs]
            for cam_idx, cam in enumerate(depth_cams):
                key = f"{cam}_depth"
                depth = obs[key][0, 0].detach().cpu().float()  # [1, H, W]
                n_intr = obs["camera_intrinsics"].shape[2]
                if cam_idx >= n_intr:
                    continue
                intrinsics = obs["camera_intrinsics"][0, 0, cam_idx].detach().cpu().float()  # [4]
                fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
                H, W = depth.shape[1], depth.shape[2]
                u = torch.arange(W, dtype=torch.float32)
                v = torch.arange(H, dtype=torch.float32)
                uu, vv = torch.meshgrid(u, v, indexing='xy')
                X = (uu - cx) * depth[0] / fx.clamp(min=1e-6)
                Y = (vv - cy) * depth[0] / fy.clamp(min=1e-6)
                Z = depth[0]
                ptmap = torch.stack([X, Y, Z], dim=0).numpy()  # [3, H, W]
                valid = ptmap[2] > 0
                # Normalize each channel independently to [0, 1] for visualization
                ptmap_viz = np.zeros((H, W, 3), dtype=np.uint8)
                for c in range(3):
                    ch = ptmap[c]
                    ch_valid = ch[valid]
                    if len(ch_valid) > 0:
                        lo = np.percentile(ch_valid, 2)
                        hi = np.percentile(ch_valid, 98)
                        if hi > lo:
                            norm_ch = np.clip((ch - lo) / (hi - lo), 0, 1)
                        else:
                            norm_ch = np.zeros_like(ch)
                        norm_ch[~valid] = 0
                        ptmap_viz[:, :, c] = (norm_ch * 255).astype(np.uint8)
                images_dict[f"viz/{cam}_pointmap"] = wandb.Image(ptmap_viz)

        if images_dict:
            wandb.log(images_dict, step=self.completed_steps)

    def _set_data_epoch(self, epoch: int):
        """Reseed the train sampler so each epoch (and each resume) draws a different order.

        RandomSampler/WeightedRandomSampler have no set_epoch, so without reseeding every
        attempt would replay the shuffle produced by set_seed() from the top.
        """
        sampler = getattr(self, "_train_sampler", None)
        if sampler is None:
            return
        generator = getattr(sampler, "generator", None)
        if generator is not None:
            generator.manual_seed(self._data_seed + epoch)
        elif callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)

    def _create_data_iterators(self):
        """Create the train iterator, fast-forwarded to the resumed position."""
        self._data_seed = int(getattr(self.config, "seed", 3047))
        steps_per_epoch = len(self.vla_train_dataloader)
        grad_accum = max(1, int(self.accelerator.gradient_accumulation_steps))
        consumed = self.completed_steps * grad_accum
        self.vla_epoch_count = consumed // steps_per_epoch if steps_per_epoch else 0
        offset = consumed % steps_per_epoch if steps_per_epoch else 0

        self._set_data_epoch(self.vla_epoch_count)

        loader = self.vla_train_dataloader
        if offset:
            # Skips at the batch-sampler level, so skipped samples are never fetched or decoded.
            from accelerate import skip_first_batches

            loader = skip_first_batches(self.vla_train_dataloader, offset)
            logger.info(f"Resuming data at epoch {self.vla_epoch_count}, batch {offset}/{steps_per_epoch}")
        self.vla_iter = iter(loader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            self._set_data_epoch(self.vla_epoch_count)
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # cProfile: profile first N steps on rank 0 and dump to weka
        import cProfile
        _profile_steps = int(os.environ.get("PROFILE_STEPS", "0"))
        _profiler = None
        if _profile_steps > 0 and dist.get_rank() == 0:
            _profiler = cProfile.Profile()
            _profiler.enable()
            logger.info(f"cProfile enabled for first {_profile_steps} steps")

        # create progress bar (initial=completed_steps so bar shows correct position on resume)
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # main training loop
        _save_first_batch = getattr(self.config, "save_first_batch", False)
        if isinstance(_save_first_batch, str):
            _save_first_batch = _save_first_batch.lower() == "true"

        _t_data_sum, _t_count = 0.0, 0
        _t_window_start = time.perf_counter()
        _logging_frequency = int(self.config.trainer.logging_frequency)
        if _logging_frequency <= 0:
            raise ValueError("trainer.logging_frequency must be greater than zero")

        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            _t_data = time.perf_counter() - t_start_data
            _t_data_sum += _t_data
            _t_count += 1

            # Save first batch for diagnostic inspection
            if _save_first_batch and self.completed_steps == 0:
                logger.info("📊 Saving first batch data for diagnostics...")
                save_first_batch_data(self.config.output_dir, batch_vla, self.accelerator.is_main_process)
                _save_first_batch = False

            step_metrics = self._train_step(batch_vla)
            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_wait": f"{_t_data:.3f}",
                        }
                    )

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # Cross-rank timing once per logging window (collective — the gate is
            # identical on all ranks, so every rank enters together).
            if (
                self.accelerator.sync_gradients
                and self.completed_steps % _logging_frequency == 0
                and _t_count > 0
            ):
                t = torch.tensor([_t_data_sum], device=self.accelerator.device)
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                wall_s = (time.perf_counter() - _t_window_start) / _t_count
                data_s = min(t.item() / _t_count, wall_s)
                model_s = wall_s - data_s

                step_metrics["time/wall_per_step"] = wall_s
                step_metrics["time/data_time"] = data_s
                step_metrics["time/model_time"] = model_s
                step_metrics["time/data_pct"] = 100.0 * data_s / max(wall_s, 1e-9)

                if self.accelerator.is_main_process:
                    logger.info(
                        f"[timing] step {self.completed_steps} (last {_t_count} steps): "
                        f"wall {wall_s:.3f}s/step = data {data_s:.3f}s "
                        f"({100.0 * data_s / max(wall_s, 1e-9):.1f}%) + model {model_s:.3f}s"
                    )

                _t_data_sum, _t_count = 0.0, 0
                _t_window_start = time.perf_counter()

            self._log_metrics(step_metrics)

            # cProfile: dump after N steps
            if _profiler is not None and self.completed_steps >= _profile_steps:
                _profiler.disable()
                profile_dir = Path(self.config.output_dir) / "profiles"
                profile_dir.mkdir(parents=True, exist_ok=True)
                run_id = getattr(self.config, "run_id", "unknown")
                prof_path = profile_dir / f"{run_id}.prof"
                _profiler.dump_stats(str(prof_path))
                import pstats, io
                s = io.StringIO()
                ps = pstats.Stats(_profiler, stream=s).sort_stats("cumulative")
                ps.print_stats(80)
                s.write("\n\n--- BY TOTAL TIME ---\n\n")
                ps2 = pstats.Stats(_profiler, stream=s).sort_stats("tottime")
                ps2.print_stats(80)
                txt_path = profile_dir / f"{run_id}.txt"
                txt_path.write_text(s.getvalue())
                logger.info(f"cProfile dumped to {prof_path} and {txt_path}")
                _profiler = None

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
                self._log_images(batch_vla)

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        # Use val dataloader if available, otherwise fall back to a train batch
        if self.vla_val_dataloader is not None:
            if not hasattr(self, "_val_iter") or self._val_iter is None:
                self._val_iter = iter(self.vla_val_dataloader)
            try:
                examples = next(self._val_iter)
            except StopIteration:
                self._val_iter = iter(self.vla_val_dataloader)
                examples = next(self._val_iter)
        else:
            examples = self._get_next_batch()
        score = 0.0
        actions = examples["action"].cpu().numpy()  # ground truth [B, T, D]
        # Predict actions using the model
        framework = self.accelerator.unwrap_model(self.model)
        framework.eval()
        with torch.no_grad():
            output_dict = framework.predict_action(
                examples=examples, use_ddim=True, num_ddim_steps=20
            )
        framework.train()

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T_pred, D
            # Truncate ground truth to match prediction length if needed
            T_pred = normalized_actions.shape[1]
            actions = actions[:, :T_pred, :]
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["loss/mse_score"] = average_score

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # autocast is applied inside model.forward() to avoid DeepSpeed stripping it
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            total_loss = action_loss

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # Gradient clipping: DeepSpeed clips internally (plugin gradient_clipping);
            # the DDP/Muon path must clip explicitly, and only once grads are synced.
            if not self._use_deepspeed and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    getattr(self.config.trainer, "gradient_clipping", 1.0),
                )

            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

        # update EMA after optimizer step (only when gradients actually synced)
        if self.accelerator.sync_gradients:
            framework = self.accelerator.unwrap_model(self.model)
            if hasattr(framework, "ema"):
                framework.ema.step(framework.policy)

        step_metrics = {"loss/action_dit": action_loss.item()}

        framework_name = getattr(self.config.framework, "name", "")
        if framework_name.startswith("ManiFlow"):
            _mf_key_map = {
                "mf_loss_flow":             "loss/flow_matching",
                "mf_loss_ct":               "loss/consistency_training",
                "mf_v_flow_pred_magnitude": "debug/flow_pred_magnitude",
                "mf_v_ct_pred_magnitude":   "debug/consistency_training_pred_magnitude",
                "mf_bc_loss":               "loss/total",
                "mf_rgb_cond_token_std":    "debug/rgb_token_std",
                "mf_ptmap_cond_token_std":  "debug/ptmap_token_std",
            }
            step_metrics.update({
                _mf_key_map[k]: v
                for k, v in output_dict.items()
                if k in _mf_key_map
            })
            step_metrics.update({
                k.replace("mf_", ""): v
                for k, v in output_dict.items()
                if k.startswith("mf_time/")
            })

        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(_untie_shared_tensors(state_dict), os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

            # save full in-memory config to final_model metadata
            import json 
            import yaml
            cfg_obj = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
            OmegaConf.save(cfg_obj, os.path.join(final_checkpoint, "config.yaml"))
            with open(os.path.join(final_checkpoint, "config.yaml"), "r") as f_yaml, \
                 open(os.path.join(final_checkpoint, "config.json"), "w") as f_json:
                json.dump(yaml.safe_load(f_yaml), f_json, indent=2)
            logger.info(f"Config saved to {final_checkpoint}/config.yaml and config.json")


        # close W&B
        if self._wandb_leader:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg, accelerator) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)

    # Save architecture diagram if requested
    _save_arch = getattr(cfg, "save_arch_diagram", False)
    if isinstance(_save_arch, str):
        _save_arch = _save_arch.lower() == "true"
    if _save_arch and (not dist.is_initialized() or dist.get_rank() == 0):
        try:
            diagram = generate_arch_diagram(cfg, vla)
            arch_path = output_dir / "architecture.txt"
            arch_path.write_text(diagram)
            logger.info(f"Architecture diagram saved to {arch_path}")
        except Exception as e:
            logger.warning(f"Failed to generate architecture diagram: {e}")

    # Since we know use omegaconf to save this toucing code is no longer required and can be removed
    # Touch framework keys so they appear in saved config.yaml
    # (OmegaConf.merge in the framework bypasses AccessTrackedConfig)
    _FRAMEWORK_KEYS_FOR_INFERENCE = (
        # Architecture
        "n_action_steps", "num_inference_steps", "horizon", "n_obs_steps",
        "n_layer", "n_head", "n_emb", "shape_meta",
        "encoder_type", "encoder_output_dim", "obs_as_global_cond", "qkv_bias", "qk_norm",
        # Backbones
        "rgb_backbone", "ptmap_backbone", "use_rgb", "use_ptmap",
        # Training ratios (needed to reconstruct policy)
        "flow_batch_ratio", "consistency_batch_ratio",
        "sample_t_mode_flow", "sample_t_mode_consistency",
        "sample_dt_mode_consistency", "sample_target_t_mode",
        "denoise_timesteps", "diffusion_timestep_embed_dim",
        "diffusion_target_t_embed_dim",
        # State injection
        "state_injection_mode", "propr_mask_prob",
        # Language conditioning
        "language_conditioned", "language_model", "lang_injection_mode", "language_mode",
        # Target conditioning
        "target_injection_mode", "target_sigma",
        "heatmap_encoder_type", "heatmap_backbone",
        # Normalization
        "rgb_mean", "rgb_std", "use_pc_color",
        "ptmap_min", "ptmap_max",
        "per_camera_ptmap_bounds",
        "pre_norm_modality",
        # Pointmap encoder
        "ptmap_encoder_type",
        "sparse_num_points", "sparse_out_channels", "sparse_hidden_dim",
        "use_previous_front_frame",
        "use_target_front_crop",
    )
    for _k in _FRAMEWORK_KEYS_FOR_INFERENCE:
        getattr(cfg.framework, _k, None)
    # touch action such that it appears in saved config.yaml (needed for inference)
    _shape_meta = getattr(cfg.framework, "shape_meta", None)
    if _shape_meta is not None:
        getattr(_shape_meta, "action", None)
    getattr(cfg.datasets.vla_data, "action_mode", None)
    # shape_meta.obs.* gets touched key-by-key elsewhere (wandb config, dataloader),
    # which makes bare "shape_meta" a non-leaf and excludes untouched siblings from
    # the export. shape_meta.action is never read elsewhere (model code consumes it
    # via OmegaConf.to_container in one shot, bypassing access tracking), so touch it
    # explicitly here to ensure action shape survives into the saved config.yaml.
    getattr(cfg.framework.shape_meta, "action", None)

    # prepare data
    vla_train_dataloader, vla_val_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    if _use_muon(cfg):
        optimizer, lr_scheduler = build_muon_optimizer_and_scheduler(model=vla, cfg=cfg)
    else:
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="examples/Franka/train_maniflow_multimodal.yaml", help="Path to YAML config")
    parser.add_argument("--resume_config_yaml", type=str, default=None,
                         help="Path to the resumed run's saved config.yaml (written by _save_checkpoint). "
                              "When set, its values win over the base --config_yaml template, but this "
                              "invocation's own CLI flags still win over both — see the merge order below.")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)

    # Merge order is template < resume_cfg < cli_cfg: the resumed run's saved config
    # (architecture, dataset schema, ...) overrides the base template, but anything
    # explicitly passed on this invocation's command line always wins last. This
    # relies on the submitting script (submit_maniflow_train.py) never emitting
    # architecture-defining flags when resuming, so cli_cfg only ever contains
    # submission-time knobs (or explicit, intentional overrides) on a resume run.
    if args.resume_config_yaml:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.resume_config_yaml))
    cfg = OmegaConf.merge(cfg, cli_cfg)

    accelerator = build_accelerator(cfg)
    accelerator.print(accelerator.state)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg, accelerator)
