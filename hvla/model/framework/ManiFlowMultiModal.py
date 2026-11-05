"""
ManiFlow MultiModal Framework (v2 — Pointmap)
=============================================
hvla framework wrapper for ManiFlowMultiModalPolicy (standalone, no maniflow dependency).

Batch format expected (from collate_fn matching shape_meta):
    {
        "obs": {
            "<cam>_rgb":          Tensor [B, 1, 3, H, W]   RGB in [0, 1]
            "<cam>_depth":        Tensor [B, 1, 1, H, W]   raw depth in meters (GPU backproj)
            "camera_intrinsics":  Tensor [B, N_cams, 4]    [fx, fy, cx, cy] per cam
            "agent_pos":          Tensor [B, 1, state_dim]
        },
        "action": Tensor [B, T_a, action_dim],
        "lang":   List[str]
    }

Camera keys and image resolution are derived from shape_meta.obs:
  - Keys ending with "_rgb" → RGB cameras
  - Keys ending with "_depth" → GPU back-projection + pointmap encoder
  - (legacy: "_ptmap" keys still supported for pre-computed XYZ pointmaps)
  - Resolution from the shape [C, H, W]

Forward output:
    {"action_loss": scalar Tensor}  compatible with VLATrainer._train_step
"""

import copy

import torch
from omegaconf import OmegaConf

from hvla.dataloader.franka_dataset import parse_target_front_crop
from hvla.model.framework.base_framework import baseframework
from hvla.model.tools import FRAMEWORK_REGISTRY
from hvla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Defaults for optional framework config fields.  Merged under config.framework
# so every field can be accessed as ``fw.<key>`` without repeated .get() calls.
_MANIFLOW_MM_DEFAULTS = OmegaConf.create(
    {
        "encoder_type": "spatial",
        "state_injection_mode": "token",
        "lang_injection_mode": "token",
        "pre_norm_modality": False,
        "use_rgb": True,
        "use_ptmap": True,
        "num_inference_steps": 10,
        "obs_as_global_cond": True,
        "qkv_bias": False,
        "qk_norm": False,
        "rgb_backbone": "vit_base_patch16_clip_224.openai",
        "ptmap_backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "propr_mask_prob": 0.0,
        "language_conditioned": False,
        "language_model": "t5-small",
        "flow_batch_ratio": 0.75,
        "consistency_batch_ratio": 0.25,
        "sample_t_mode_flow": "beta",
        "sample_t_mode_consistency": "discrete",
        "sample_dt_mode_consistency": "uniform",
        "sample_target_t_mode": "relative",
        "denoise_timesteps": 10,
        "diffusion_timestep_embed_dim": 128,
        "diffusion_target_t_embed_dim": 128,
        "use_pc_color": True,
        "ptmap_min": [-1.0, -1.0, 0.0],
        "ptmap_max": [1.0, 1.0, 2.5],
        "rgb_mean": None,  # auto-detect from backbone; falls back to CLIP
        "rgb_std": None,
        "freeze_rgb_backbone": False,
        "random_crop_ratio": 0.95,
        "random_rotation_degrees": 5.0,
        "color_jitter": {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        },
        "ema": {
            "update_after_step": 0,
            "inv_gamma": 1.0,
            "power": 0.75,
            "min_value": 0.0,
            "max_value": 0.9999,
        },
        "target_injection_mode": None,
        "target_sigma": 1.5,
        "target_mask_prob": 0.0,
        "heatmap_encoder_type": "shared_rgb",
        "heatmap_backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "language_mode": "action_goal",
        "ptmap_encoder_type": "dense",
        "sparse_num_points": 64,
        "sparse_out_channels": 256,
        "sparse_hidden_dim": 128,
        "per_camera_ptmap_bounds": None,
        "use_previous_front_frame": False,
        "use_target_front_crop": -1,
    }
)


@FRAMEWORK_REGISTRY.register("ManiFlowMultiModal")
class ManiFlowMultiModalFramework(baseframework):
    """
    Consistency Flow training (75 % flow + 25 % consistency) with EMA.

    Data is assumed to be pre-normalized by the dataloader (q99 normalization).
    ImageNet normalisation for the backbone is handled inside MultiModalObsEncoder.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config

        from hvla.model.modules.maniflow.flow_policy import ManiFlowMultiModalPolicy
        from hvla.model.modules.maniflow.ema_model import EMAModel

        fw = OmegaConf.merge(_MANIFLOW_MM_DEFAULTS, config.framework)
        shape_meta = OmegaConf.to_container(fw.shape_meta, resolve=True)

        # ------------------------------------------------------------------ #
        # Build policy                                                        #
        # ------------------------------------------------------------------ #
        self.policy = ManiFlowMultiModalPolicy(
            shape_meta=shape_meta,
            horizon=fw.horizon,
            n_action_steps=fw.n_action_steps,
            n_obs_steps=fw.n_obs_steps,
            num_inference_steps=fw.num_inference_steps,
            obs_as_global_cond=fw.obs_as_global_cond,
            n_layer=fw.n_layer,
            n_head=fw.n_head,
            n_emb=fw.n_emb,
            qkv_bias=fw.qkv_bias,
            qk_norm=fw.qk_norm,
            rgb_backbone=fw.rgb_backbone,
            ptmap_backbone=fw.ptmap_backbone,
            propr_mask_prob=fw.propr_mask_prob,
            encoder_type=fw.encoder_type,
            state_injection_mode=fw.state_injection_mode,
            lang_injection_mode=fw.lang_injection_mode,
            language_conditioned=fw.language_conditioned,
            language_model=fw.language_model,
            flow_batch_ratio=fw.flow_batch_ratio,
            consistency_batch_ratio=fw.consistency_batch_ratio,
            sample_t_mode_flow=fw.sample_t_mode_flow,
            sample_t_mode_consistency=fw.sample_t_mode_consistency,
            sample_dt_mode_consistency=fw.sample_dt_mode_consistency,
            sample_target_t_mode=fw.sample_target_t_mode,
            denoise_timesteps=fw.denoise_timesteps,
            diffusion_timestep_embed_dim=fw.diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=fw.diffusion_target_t_embed_dim,
            use_pc_color=fw.use_pc_color,
            use_rgb=fw.use_rgb,
            use_ptmap=fw.use_ptmap,
            ptmap_min=list(fw.ptmap_min),
            ptmap_max=list(fw.ptmap_max),
            rgb_mean=list(fw.rgb_mean) if fw.rgb_mean is not None else None,
            rgb_std=list(fw.rgb_std) if fw.rgb_std is not None else None,
            freeze_rgb_backbone=fw.freeze_rgb_backbone,
            target_injection_mode=fw.target_injection_mode,
            target_sigma=fw.target_sigma,
            target_mask_prob=fw.target_mask_prob,
            heatmap_encoder_type=fw.heatmap_encoder_type,
            heatmap_backbone=fw.heatmap_backbone,
            ptmap_encoder_type=fw.ptmap_encoder_type,
            sparse_num_points=fw.sparse_num_points,
            sparse_out_channels=fw.sparse_out_channels,
            sparse_hidden_dim=fw.sparse_hidden_dim,
            per_camera_ptmap_bounds=(
                OmegaConf.to_container(fw.per_camera_ptmap_bounds, resolve=True)
                if fw.per_camera_ptmap_bounds is not None else None
            ),
            use_previous_front_frame=fw.use_previous_front_frame,
            use_target_front_crop=parse_target_front_crop(fw.use_target_front_crop),
        )

        # ------------------------------------------------------------------ #
        # EMA shadow model                                                    #
        # ------------------------------------------------------------------ #
        ema_cfg = OmegaConf.to_container(fw.ema, resolve=True)
        self.ema = EMAModel(
            model=copy.deepcopy(self.policy),
            update_after_step=ema_cfg["update_after_step"],
            inv_gamma=ema_cfg["inv_gamma"],
            power=ema_cfg["power"],
            min_value=ema_cfg["min_value"],
            max_value=ema_cfg["max_value"],
        )

    def forward(self, batch: dict) -> dict:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = self.policy.compute_loss(batch, ema_model=self.ema.averaged_model)

        # loss_dict values are already .item()'d scalars; no extra GPU sync needed

        return {"action_loss": loss, **{f"mf_{k}": v for k, v in loss_dict.items()}}

    @torch.inference_mode()
    def predict_action(
        self,
        examples,
        use_ddim: bool = True,
        num_ddim_steps: int = 20,
        **kwargs,
    ) -> dict:
        """
        Inference wrapper used by eval_action_model().
        Delegates to self.policy.predict_action(obs_dict) and returns
        {"normalized_actions": np.ndarray [B, T, D]} as expected by the trainer.
        """
        import numpy as np

        obs_dict = examples["obs"]
        device = next(self.parameters()).device
        obs_dict = {
            k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray)
                else v.to(device) if hasattr(v, 'to') else v)
            for k, v in obs_dict.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.policy.predict_action(obs_dict)
        action_pred = result["action"].cpu().float().numpy()  # [B, n_action_steps, Da]
        return {
            "normalized_actions": action_pred,
            "action_pred": result.get("action_pred", None),
        }

    def get_ema_state(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "averaged_model": self.ema.averaged_model.state_dict(),
            "optimization_step": self.ema.optimization_step,
            "decay": self.ema.decay,
        }

    def load_ema_state(self, state: dict) -> None:
        """Restore EMA state from checkpoint."""
        self.ema.averaged_model.load_state_dict(state["averaged_model"])
        self.ema.optimization_step = state["optimization_step"]
        self.ema.decay = state["decay"]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.ema.averaged_model = self.ema.averaged_model.to(*args, **kwargs)
        return self
