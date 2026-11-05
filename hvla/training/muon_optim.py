"""Muon (orthogonalized momentum) for weight matrices, AdamW for everything else.

Muon (https://kellerjordan.github.io/posts/muon/) takes the momentum buffer of a matrix
parameter and replaces it with its closest semi-orthogonal matrix, approximated by a
quintic Newton-Schulz iteration. Only parameters that are actually weight matrices
benefit from this, so everything else (biases, norm weights, embeddings, the output
projection) is handed to AdamW.

`muon_adamw_param_groups` splits a module's trainable parameters into the `use_muon`-tagged
groups that `muon.MuonWithAuxAdam` (the *distributed* variant from Keller Jordan's
`muon-optimizer` package, pinned in the repo `pyproject.toml`) consumes directly.
`MuonWithAuxAdam` is a single, real `torch.optim.Optimizer` that runs Muon on the
`use_muon=True` groups and an internal AdamW on the rest, so it composes with
`accelerator.prepare`, torch/HF LR schedulers, and standard `state_dict`/`load_state_dict`
checkpointing.

`build_muon_optimizer_and_scheduler` is the maniflow-specific entry point: it does the
routing (head + positional/target embeddings forced onto AdamW), builds the optimizer, and
returns a `LambdaLR` that schedules **only** the AdamW groups (Muon runs at a fixed LR).
It is imported by both `train_hvla.py` and the local debug script, so the two never drift.
This module deliberately avoids importing `accelerate`, so it can be exercised standalone.

Parameter routing (see `muon_adamw_param_groups`) is primarily by shape:
    ndim >= 2  weight matrices / conv filters (flattened) -> Muon
    ndim <= 1  biases, norm gains/scales, learned scalars -> AdamW
Callers may additionally force specific modules/params onto AdamW regardless of shape via
`adamw_modules`/`adamw_params` -- pass the output head and any positional/target
embeddings there, since Muon is only meant for hidden weight matrices.

Distributed constraint: the distributed Muon `step()` shards the Newton-Schulz work across
ranks and `all_gather`s the updated matrices, which assumes plain DDP with all-reduced
gradients. It is INCOMPATIBLE with DeepSpeed ZeRO (which partitions gradients / optimizer
state *within* a parameter), so `train_hvla.py` disables DeepSpeed and runs DDP when this
optimizer is selected. `step()` also requires an initialized process group, which is always
true under an `accelerate`/`torchrun` launch (world size 1 included).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def muon_adamw_param_groups(
    module: nn.Module,
    *,
    muon_lr: float,
    muon_momentum: float,
    muon_weight_decay: float,
    adamw_lr: float,
    adamw_betas: tuple[float, float],
    adamw_eps: float,
    adamw_weight_decay: float,
    adamw_modules: Iterable[nn.Module] = (),
    adamw_params: Iterable[torch.Tensor] = (),
) -> list[dict]:
    """Split ``module``'s trainable parameters into an AdamW group and a Muon group.

    Args:
        module: module whose (trainable) parameters are routed. Frozen parameters
            (``requires_grad=False``) are skipped entirely.
        muon_lr / muon_momentum / muon_weight_decay: Muon-branch hyperparameters.
        adamw_lr / adamw_betas / adamw_eps / adamw_weight_decay: AdamW-branch
            hyperparameters.
        adamw_modules: modules kept on AdamW regardless of shape (e.g. the output head).
        adamw_params: individual parameters kept on AdamW regardless of shape (e.g.
            positional / target embeddings).

    Returns:
        Param groups in the exact schema ``MuonWithAuxAdam`` requires. The AdamW group is
        listed first so ``lr_scheduler.get_last_lr()[0]`` reports the scheduled AdamW LR.
    """
    adamw_ids = {id(p) for m in adamw_modules for p in m.parameters()}
    adamw_ids |= {id(p) for p in adamw_params}

    muon: list[torch.Tensor] = []
    adam: list[torch.Tensor] = []
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and id(param) not in adamw_ids:
            muon.append(param)
        else:
            adam.append(param)

    groups: list[dict] = []
    if adam:
        groups.append(
            dict(
                params=adam,
                lr=adamw_lr,
                betas=tuple(adamw_betas),
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
                use_muon=False,
            )
        )
    if muon:
        groups.append(
            dict(
                params=muon,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                use_muon=True,
            )
        )
    assert groups, f"{type(module).__name__} has no trainable parameters"
    return groups


def _build_muon_lr_lambda(num_warmup_steps: int):
    """Warm up to the Muon LR over num_warmup_steps, then hold constant (no decay).

    Warmup avoids slamming the from-scratch matrices with full-strength orthogonalized
    updates at step 0 (a common source of early divergence / NaN).
    """
    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0
    return fn


def _build_adamw_lr_lambda(name: str, num_warmup_steps: int, num_training_steps: int):
    """Return a LambdaLR multiplier replicating HF ``get_scheduler`` for the AdamW groups.

    Kept in sync with the non-Muon path (``setup_optimizer_and_scheduler``) so scheduled
    AdamW groups behave identically. Applied per-group; Muon groups use a constant 1.0.
    """
    name = str(name).lower()

    def _warmup(step: int) -> float:
        return float(step) / float(max(1, num_warmup_steps))

    if name == "cosine":
        def fn(step: int) -> float:
            if step < num_warmup_steps:
                return _warmup(step)
            progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return fn
    if name == "linear":
        def fn(step: int) -> float:
            if step < num_warmup_steps:
                return _warmup(step)
            return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
        return fn
    if name in ("constant", "constant_with_warmup"):
        def fn(step: int) -> float:
            if name == "constant":
                return 1.0
            return min(1.0, _warmup(step))
        return fn
    raise ValueError(f"Unsupported trainer.lr_scheduler_type for the Muon path: {name!r}")


def build_muon_optimizer_and_scheduler(model, cfg):
    """Muon for from-scratch weight matrices; AdamW for everything else.

    Muon: from-scratch DiT matrices + sparse pointmap encoder. AdamW: biases/norms,
    embeddings, output head, and the pretrained CLIP backbone. Only AdamW groups are
    scheduled; Muon runs at a fixed LR.

    Args:
        model: the maniflow framework (``model.policy.model`` is the DiTX backbone).
        cfg: the training config (reads ``cfg.trainer.optimizer`` / ``learning_rate`` /
            ``lr_scheduler_type`` / ``num_warmup_steps`` / ``max_train_steps``).

    Returns:
        ``(MuonWithAuxAdam, torch.optim.lr_scheduler.LambdaLR)``.
    """
    optimizer_cfg = cfg.trainer.optimizer
    betas = tuple(optimizer_cfg.betas)
    muon_lr = float(getattr(optimizer_cfg, "muon_lr", 2e-2))
    muon_momentum = float(getattr(optimizer_cfg, "muon_momentum", 0.95))
    muon_weight_decay = float(getattr(optimizer_cfg, "muon_weight_decay", 0.0))

    # AdamW (not Muon) for the head, input embedding, pretrained CLIP backbone, and DiT
    # positional/target embeddings; the from-scratch sparse ptmap encoder stays on Muon.
    dit = model.policy.model
    adamw_modules = [dit.final_layer, dit.input_emb]
    obs_encoder = model.policy.obs_encoder
    if getattr(obs_encoder, "use_rgb", False) and hasattr(obs_encoder, "rgb_encoder"):
        adamw_modules.append(obs_encoder.rgb_encoder)
    adamw_params = [
        p for n, p in dit.named_parameters()
        if p.requires_grad and p.ndim >= 2 and ("pos_emb" in n or "target_embed" in n)
    ]

    param_groups = muon_adamw_param_groups(
        model,
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
        muon_weight_decay=muon_weight_decay,
        adamw_lr=cfg.trainer.learning_rate.base,
        adamw_betas=betas,
        adamw_eps=optimizer_cfg.eps,
        adamw_weight_decay=optimizer_cfg.weight_decay,
        adamw_modules=adamw_modules,
        adamw_params=adamw_params,
    )
    from muon import MuonWithAuxAdam

    optimizer = MuonWithAuxAdam(param_groups)

    if not dist.is_initialized() or dist.get_rank() == 0:
        for group in optimizer.param_groups:
            kind = "muon" if group.get("use_muon") else "adamw"
            logger.info(f"[muon] {kind} group: lr={group['lr']}, num_params={len(group['params'])}")

    # Both groups warm up; AdamW then follows lr_scheduler_type (e.g. cosine decay), Muon
    # holds constant after warmup (its update is already spectral-norm-normalized).
    adamw_lambda = _build_adamw_lr_lambda(
        cfg.trainer.lr_scheduler_type,
        cfg.trainer.num_warmup_steps,
        cfg.trainer.max_train_steps,
    )
    muon_lambda = _build_muon_lr_lambda(cfg.trainer.num_warmup_steps)
    lr_lambdas = [
        (muon_lambda if group.get("use_muon") else adamw_lambda)
        for group in optimizer.param_groups
    ]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    return optimizer, lr_scheduler
