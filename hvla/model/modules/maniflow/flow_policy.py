"""
ManiFlow Multi-Modal Policy (standalone, no maniflow dependency).

Flattened from the maniflow inheritance chain:
  BasePolicy -> ManiFlowTransformerPointcloudPolicy -> ManiFlowTransformerMultiModalPolicy

Core algorithms (flow matching, consistency training, ODE sampling) are absorbed
from the pointcloud parent class. Normalizer is removed (was identity / no-op).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint
from torch.distributions import Beta

from hvla.model.modules.maniflow.obs_encoder import MultiModalObsEncoder
from hvla.model.modules.maniflow.ditx import DiTX


# ---------------------------------------------------------------------------
# Inlined utilities (from maniflow.common.pytorch_util)
# ---------------------------------------------------------------------------

def dict_apply(x: Dict[str, torch.Tensor], func) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            result[key] = [func(item) if hasattr(item, 'to') else item for item in value]
        else:
            result[key] = func(value)
    return result


# ---------------------------------------------------------------------------
# Inlined sampling utilities (from maniflow.model.common.sample_util)
# ---------------------------------------------------------------------------

def f_mode(u, s):
    return 1 - u - s * (torch.cos(torch.pi / 2 * u) ** 2 - 1 + u)


def sample_logit_normal(batch_size, m=0.0, s=1.0, device='cuda'):
    u = torch.normal(mean=m, std=s, size=(batch_size, 1, 1), device=device)
    t = torch.sigmoid(u)
    return t


def sample_mode(batch_size, s=1.29, device='cuda'):
    u = torch.rand(batch_size, 1, 1, device=device)
    t = f_mode(u, s)
    t = torch.clamp(t, 0, 1)
    return t


def sample_cosmap(batch_size, device='cuda'):
    u = torch.rand(batch_size, 1, 1, device=device)
    t = 1 - 1 / (torch.tan(torch.pi / 2 * u) + 1)
    t = torch.clamp(t, 0, 1)
    return t


def sample_beta(batch_size, s=0.999, alpha=1.0, beta=1.5, device='cuda'):
    beta_dist = Beta(torch.tensor([alpha], device=device),
                     torch.tensor([beta], device=device))
    raw_samples = beta_dist.sample((batch_size, 1, 1))
    t = s * raw_samples
    return t


# ---------------------------------------------------------------------------
# Inlined debug helper (from maniflow.common.model_util)
# ---------------------------------------------------------------------------

def print_params(model):
    params_dict = {}
    all_num_param = sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()
    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class ManiFlowMultiModalPolicy(nn.Module):
    """
    Multi-camera RGB + pointmap policy with configurable camera keys, image
    resolution, state/language injection modes, and dynamic visual token counts.

    Consistency Flow training (75% flow + 25% consistency) with EMA teacher.
    No normalizer — data is assumed to be pre-normalized by the dataloader.
    """

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps=None,
        obs_as_global_cond: bool = True,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_target_t_embed_dim: int = 256,
        n_layer: int = 12,
        n_head: int = 8,
        n_emb: int = 768,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        block_type: str = "DiTX",
        rgb_backbone: str = "vit_base_patch16_clip_224.openai",
        ptmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        propr_mask_prob: float = 0.0,
        encoder_type: str = "spatial",
        state_injection_mode: str = "token",
        lang_injection_mode: str = "token",
        pre_norm_modality: bool = False,
        language_conditioned: bool = False,
        language_model: str = "t5-small",
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        denoise_timesteps: int = 10,
        sample_t_mode_flow: str = "beta",
        sample_t_mode_consistency: str = "discrete",
        sample_dt_mode_consistency: str = "uniform",
        sample_target_t_mode: str = "relative",
        use_pc_color: bool = True,
        use_rgb: bool = True,
        use_ptmap: bool = True,
        ptmap_min: Optional[List[float]] = None,
        ptmap_max: Optional[List[float]] = None,
        rgb_mean: Optional[List[float]] = None,
        rgb_std: Optional[List[float]] = None,
        freeze_rgb_backbone: bool = False,
        target_injection_mode: Optional[str] = None,
        target_sigma: float = 1.5,
        target_mask_prob: float = 0.0,
        heatmap_encoder_type: str = "shared_rgb",
        heatmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        # Sparse point encoder config
        ptmap_encoder_type: str = "dense",
        sparse_num_points: int = 64,
        sparse_out_channels: int = 256,
        sparse_hidden_dim: int = 128,
        # Per-camera pointmap bounds: dict mapping depth/ptmap key -> {"min": [x,y,z], "max": [x,y,z]}
        per_camera_ptmap_bounds: Optional[Dict[str, Dict[str, List[float]]]] = None,
        use_previous_front_frame: bool = False,
        use_target_front_crop: int = -1,
        # Deprecated aliases — ignored, kept for checkpoint compat
        target_conditioned: bool = False,
        use_heatmap: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_pc_color = use_pc_color
        if target_injection_mode == "legacy":
            target_injection_mode = "target_embedding"
        if target_conditioned and target_injection_mode is None:
            target_injection_mode = "target_embedding"
        self.target_injection_mode = target_injection_mode
        self.target_conditioned = target_injection_mode is not None

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        state_dim = shape_meta["obs"]["agent_pos"]["shape"][0]

        _valid = lambda k: shape_meta["obs"][k] is not None
        # Always use sorted keys for consistency, but ensure prev_front_rgb is at the end if present
        rgb_keys = sorted(k for k in shape_meta["obs"] if k.endswith("_rgb") and _valid(k)) if use_rgb else []
        if use_previous_front_frame:
            if "prev_front_rgb" not in rgb_keys:
                rgb_keys.append("prev_front_rgb")
            else:
                # Ensure it's at the end
                rgb_keys.remove("prev_front_rgb")
                rgb_keys.append("prev_front_rgb")
        if use_target_front_crop >= 0:
            if "target_front_crop_rgb" not in rgb_keys:
                rgb_keys.append("target_front_crop_rgb")
            else:
                rgb_keys.remove("target_front_crop_rgb")
                rgb_keys.append("target_front_crop_rgb")
        ptmap_keys = sorted(k for k in shape_meta["obs"] if k.endswith("_ptmap") and _valid(k))
        depth_keys = sorted(k for k in shape_meta["obs"] if k.endswith("_depth") and _valid(k))
        if use_rgb and not rgb_keys:
            raise ValueError("use_rgb=True but shape_meta.obs contains no '_rgb' keys")
        if use_ptmap and not ptmap_keys and not depth_keys:
            raise ValueError("use_ptmap=True but shape_meta.obs contains no '_ptmap' or '_depth' keys")

        # Derive img_size from whichever modality is enabled
        if use_rgb:
            img_size = shape_meta["obs"][rgb_keys[0]]["shape"][-1]
        elif ptmap_keys:
            img_size = shape_meta["obs"][ptmap_keys[0]]["shape"][-1]
        else:
            img_size = shape_meta["obs"][depth_keys[0]]["shape"][-1]

        _uses_heatmap_encoder = target_injection_mode in ("token", "timestep_concat")
        obs_encoder = MultiModalObsEncoder(
            rgb_backbone=rgb_backbone,
            ptmap_backbone=ptmap_backbone,
            state_dim=state_dim,
            propr_mask_prob=propr_mask_prob,
            encoder_type=encoder_type,
            use_rgb=use_rgb,
            use_ptmap=use_ptmap,
            rgb_keys=rgb_keys,
            ptmap_keys=ptmap_keys,
            depth_keys=depth_keys,
            img_size=img_size,
            ptmap_min=ptmap_min,
            ptmap_max=ptmap_max,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            freeze_rgb_backbone=freeze_rgb_backbone,
            use_heatmap=_uses_heatmap_encoder,
            heatmap_encoder_type=heatmap_encoder_type,
            heatmap_backbone=heatmap_backbone,
            ptmap_encoder_type=ptmap_encoder_type,
            sparse_num_points=sparse_num_points,
            sparse_out_channels=sparse_out_channels,
            sparse_hidden_dim=sparse_hidden_dim,
            per_camera_ptmap_bounds=per_camera_ptmap_bounds,
        )
        rgb_cond_dim = obs_encoder.rgb_feature_dim if use_rgb else 0
        ptmap_cond_dim = obs_encoder.ptmap_feature_dim if use_ptmap else 0
        heatmap_cond_dim = obs_encoder.heatmap_feature_dim if _uses_heatmap_encoder else 0

        n_rgb_tokens = obs_encoder.n_rgb_cams * obs_encoder.tokens_per_rgb_cam if use_rgb else 0
        n_ptmap_tokens = obs_encoder.n_ptmap_cams * obs_encoder.tokens_per_ptmap_cam if use_ptmap else 0
        n_heatmap_tokens = obs_encoder.tokens_per_heatmap if _uses_heatmap_encoder else 0

        model = DiTX(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            rgb_cond_dim=rgb_cond_dim,
            ptmap_cond_dim=ptmap_cond_dim,
            state_dim=state_dim,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            pre_norm_modality=pre_norm_modality,
            language_conditioned=language_conditioned,
            language_model=language_model,
            state_injection_mode=state_injection_mode,
            lang_injection_mode=lang_injection_mode,
            use_rgb=use_rgb,
            use_ptmap=use_ptmap,
            n_rgb_tokens=n_rgb_tokens,
            n_ptmap_tokens=n_ptmap_tokens,
            target_injection_mode=target_injection_mode,
            target_sigma=target_sigma,
            target_mask_prob=target_mask_prob,
            heatmap_cond_dim=heatmap_cond_dim,
            n_heatmap_tokens=n_heatmap_tokens,
            tokens_per_rgb_cam=obs_encoder.tokens_per_rgb_cam if use_rgb else 0,
            img_size=img_size,
            heatmap_share_rgb_pos_embed=(heatmap_encoder_type == "cnn" and use_rgb),
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.horizon = horizon
        self.rgb_cond_dim = rgb_cond_dim
        self.ptmap_cond_dim = ptmap_cond_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.state_injection_mode = state_injection_mode
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "Sum of batch ratios must equal 1.0"
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in ("absolute", "relative"), \
            "sample_target_t_mode must be 'absolute' or 'relative'"

        cprint("[ManiFlowMultiModalPolicy] Initialized", "yellow")
        cprint(f"  rgb_cond_dim        : {rgb_cond_dim}", "yellow")
        cprint(f"  ptmap_cond_dim      : {ptmap_cond_dim}", "yellow")
        cprint(f"  n_rgb_tokens        : {n_rgb_tokens}", "yellow")
        cprint(f"  n_ptmap_tokens      : {n_ptmap_tokens}", "yellow")
        cprint(f"  state_injection_mode: {state_injection_mode}", "yellow")
        cprint(f"  lang_injection_mode : {lang_injection_mode}", "yellow")
        cprint(f"  action_dim          : {action_dim}", "yellow")
        cprint(f"  horizon             : {horizon}", "yellow")
        cprint(f"  rgb_keys            : {rgb_keys}", "yellow")
        cprint(f"  ptmap_keys          : {ptmap_keys}", "yellow")
        cprint(f"  img_size            : {img_size}", "yellow")
        print_params(self)

    # ========= inference ============

    def conditional_sample(self, condition_data, **kwargs):
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)

        ode_traj = self.sample_ode(
            x0=noise,
            N=self.num_inference_steps,
            **kwargs)

        return ode_traj[-1]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = obs_dict
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get("task_name", None)
            assert lang_cond is not None, "Language goal is required"

        target_xy = None
        if self.target_injection_mode in ("target_embedding", "target_pe") and "target_xy" in nobs:
            target_xy = nobs["target_xy"][:, 0, :].to(device)  # (B, 2)

        nobs_filtered = {k: v for k, v in nobs.items() if k != "target_xy"}
        this_nobs = dict_apply(
            nobs_filtered,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]).to(device),
        )
        enc_out = self.obs_encoder(this_nobs)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        nsample = self.conditional_sample(
            cond_data,
            rgb_cond=enc_out["rgb_tokens"],
            ptmap_cond=enc_out["ptmap_tokens"],
            state_emb=enc_out["state_emb"],
            lang_cond=lang_cond,
            target_xy=target_xy,
            heatmap_cond=enc_out.get("heatmap_tokens"),
            **self.kwargs,
        )
        naction_pred = nsample[..., :Da]
        action_pred = naction_pred
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    # ========= training ============

    def sample_t(self, batch_size, mode="uniform"):
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f"Unsupported sample_t_mode {mode}.")
        return t

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")
        return dt

    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        interpolated_data_point = noise_coeff * noise + timestep * target
        return interpolated_data_point

    def _slice_cond(self, cond, start, end):
        """Batch-slice a per-modality condition dict."""
        return {k: (v[start:end] if v is not None and (torch.is_tensor(v) or isinstance(v, (list, tuple))) else v)
                for k, v in cond.items()}

    def get_flow_velocity(self, actions, **model_kwargs):
        target_dict = {}
        flow_batchsize = actions.shape[0]
        device = actions.device

        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)

        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow

        x_0_flow = torch.randn_like(actions, device=device)
        x_1_flow = actions.to(device)
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow

        return target_dict

    def get_consistency_velocity(self, actions, cond, ema_model=None):
        target_dict = {}
        consistency_batchsize = actions.shape[0]
        device = actions.device

        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone()

        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0)
        t_next = t_next.view(-1, 1, 1)

        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        x0_ct = torch.randn_like(actions, device=device)
        x1_ct = actions.to(device)
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next,
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(),
                **cond,
            )
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()

        target_dict['x_t'] = x_t_ct
        target_dict['t'] = t_ct
        target_dict['target_t'] = target_t_ct
        target_dict['v_target'] = v_ct

        return target_dict

    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        if N is None:
            N = self.num_inference_steps
        dt = 1. / N
        traj = []
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def compute_loss(self, batch, ema_model=None, **kwargs):
        nobs = batch["obs"]
        nactions = batch["action"].to(self.device)

        batch_size = nactions.shape[0]
        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get("task_name", None)
            assert lang_cond is not None, "Language goal is required"

        target_xy = None
        if self.target_injection_mode in ("target_embedding", "target_pe") and "target_xy" in nobs:
            target_xy = nobs["target_xy"][:, 0, :].to(self.device)  # (B, 2)

        nobs_filtered = {k: v for k, v in nobs.items() if k != "target_xy"}
        this_nobs = dict_apply(
            nobs_filtered,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]).to(self.device),
        )

        # --- profiling: obs encoder ---
        _t0 = torch.cuda.Event(enable_timing=True)
        _t1 = torch.cuda.Event(enable_timing=True)
        _t0.record()
        enc_out = self.obs_encoder(this_nobs)
        _t1.record()

        # reshape from (B*n_obs_steps, ...) to (B, ...) -- merge obs steps
        def _reshape_enc(t):
            if t is None:
                return None
            if t.ndim == 3:  # (B*n_obs, N_tok, C) -> (B, n_obs*N_tok, C)
                return t.reshape(batch_size, -1, t.shape[-1])
            else:  # (B*n_obs, D) -> (B, D) using first obs step
                return t.reshape(batch_size, -1, t.shape[-1])[:, 0, :]

        cond = {
            "rgb_cond": _reshape_enc(enc_out["rgb_tokens"]),
            "ptmap_cond": _reshape_enc(enc_out["ptmap_tokens"]),
            "state_emb": _reshape_enc(enc_out["state_emb"]),
            "lang_cond": lang_cond,
            "target_xy": target_xy,
            "heatmap_cond": _reshape_enc(enc_out.get("heatmap_tokens")),
        }

        # ---- per-modality token diversity (debug) ----
        feature_std = {}
        for key in ("rgb_cond", "ptmap_cond"):
            t = cond[key]
            if t is not None:
                t_norm = F.normalize(t.float(), dim=-1, p=2)
                per_channel_std = t_norm.std(dim=[0, 1])  # (D,)
                feature_std[key] = per_channel_std.mean()

        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)

        flow_cond = self._slice_cond(cond, 0, flow_batchsize)
        flow_target_dict = self.get_flow_velocity(nactions[:flow_batchsize])
        v_flow_pred = self.model(
            sample=flow_target_dict["x_t"],
            timestep=flow_target_dict["t"].squeeze(),
            target_t=flow_target_dict["target_t"].squeeze(),
            **flow_cond,
        )
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2))

        ct_cond = self._slice_cond(cond, flow_batchsize, flow_batchsize + consistency_batchsize)
        consistency_target_dict = self.get_consistency_velocity(
            nactions[flow_batchsize:flow_batchsize + consistency_batchsize],
            cond=ct_cond,
            ema_model=ema_model,
        )
        v_ct_pred = self.model(
            sample=consistency_target_dict["x_t"],
            timestep=consistency_target_dict["t"].squeeze(),
            target_t=consistency_target_dict["target_t"].squeeze(),
            **ct_cond,
        )
        v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2))

        # ---- combine losses ----
        loss = 0.0
        v_flow_target = flow_target_dict["v_target"]
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction="none")
        loss_flow = reduce(loss_flow, "b ... -> b (...)", "mean")
        loss += loss_flow.mean()

        v_ct_target = consistency_target_dict["v_target"]
        loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction="none")
        loss_ct = reduce(loss_ct, "b ... -> b (...)", "mean")
        loss += loss_ct.mean()

        loss = loss.mean()

        loss_dict = {
            "loss_flow": loss_flow.mean().item(),
            "loss_ct": loss_ct.mean().item(),
            "v_flow_pred_magnitude": v_flow_pred_magnitude.item(),
            "v_ct_pred_magnitude": v_ct_pred_magnitude.item(),
            "bc_loss": loss.item(),
            **{f"{k}_token_std": v.item() for k, v in feature_std.items()},
        }
        return loss, loss_dict
