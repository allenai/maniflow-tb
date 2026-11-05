"""
Multi-modal observation encoder (standalone, no maniflow dependency).

Siamese 2D encoder for N-camera RGB + N-camera XYZ pointmap + robot state.
Extracted from maniflow.policy.maniflow_multimodal_policy.MultiModalObsEncoder.

Supports pointmap encoder modes:
  - "dense": ConvNeXtV2 on full-resolution XYZ pointmaps (high quality, slow)
  - "sparse": Lightweight PointNet on FPS/random-sampled 3D points (fast)
  - "defm": Frozen DeFM ViT-S/14 on raw depth with metric-aware normalization
"""

import math
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint


# ---------------------------------------------------------------------------
# DeFM metric-aware depth normalization (vectorized GPU version)
# Ported from https://github.com/leggedrobotics/defm/blob/main/defm/utils/utils.py
# ---------------------------------------------------------------------------
DEFM_MEAN = torch.tensor([0.248880, 0.495620, 0.492858]).view(1, 3, 1, 1)
DEFM_STD = torch.tensor([0.139357, 0.271314, 0.297177]).view(1, 3, 1, 1)


def defm_preprocess_depth_batch(depth: torch.Tensor) -> torch.Tensor:
    """Convert raw metric depth [N, 1, H, W] to DeFM 3-channel input [N, 3, H, W].

    Three channels encode depth at different scales:
      C1: log1p(d) / log1p(100)  -- global (up to 100m)
      C2: clip(log1p(d) / log1p(9), 0, 1)  -- mid-range
      C3: per-image min-max of log depth  -- local relative
    Then normalized with DeFM's pretrained mean/std.
    """
    MAX_DEPTH_C1 = 100.0
    MAX_DEPTH_C2 = 9.0

    d = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d = torch.clamp(d, min=0.0, max=MAX_DEPTH_C1)
    log_d = torch.log1p(d)

    c1 = log_d / math.log1p(MAX_DEPTH_C1)
    c2 = torch.clamp(log_d / math.log1p(MAX_DEPTH_C2), min=0.0, max=1.0)

    N = d.shape[0]
    log_flat = log_d.view(N, -1)
    min_log = log_flat.min(dim=1)[0].view(N, 1, 1, 1)
    max_log = log_flat.max(dim=1)[0].view(N, 1, 1, 1)
    denom = max_log - min_log
    denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
    c3 = (log_d - min_log) / denom_safe
    c3 = torch.where(denom > 0, c3, torch.zeros_like(c3))

    x = torch.cat([c1, c2, c3], dim=1)  # [N, 3, H, W]

    mean = DEFM_MEAN.to(x.device, x.dtype)
    std = DEFM_STD.to(x.device, x.dtype)
    x = (x - mean) / std
    return x


def _load_defm_vit_s14() -> nn.Module:
    """Load DeFM ViT-S/14 with distributed-safe serialization.

    In multi-GPU (DDP) training, only rank 0 downloads the hub repo first,
    then all other ranks load from cache after a barrier.
    """
    import torch.distributed as dist

    def _do_load():
        return torch.hub.load(
            'leggedrobotics/defm:main', 'defm_vit_s14',
            pretrained=True, trust_repo=True,
        )

    if dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_rank() == 0:
            model = _do_load()
        dist.barrier()
        if dist.get_rank() != 0:
            model = _do_load()
    else:
        model = _do_load()
    return model


class SparsePointEncoder(nn.Module):
    """Lightweight PointNet-style encoder for sparse 3D point sets.

    Processes N sampled XYZ points through shared MLPs. Each point becomes
    one output token — no global pooling, preserving spatial structure for
    the downstream DiTX transformer.

    Architecture: Linear(3→h) → LN → GELU → Linear(h→h) → LN → GELU → Linear(h→out)
    """

    def __init__(self, out_channels: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, 3] normalized XYZ coordinates in [-1, 1]
        Returns:
            [B, N, out_channels] per-point features (each point = one token)
        """
        return self.net(points)


class HeatmapCNNEncoder(nn.Module):
    """From-scratch CNN encoder for the target heatmap image.

    Maps [B, 3, H, W] -> [B, C, grid_size, grid_size] via four stride-2 conv blocks. The
    input is resampled to grid_size * 16 so the /16 stack lands exactly on the RGB
    backbone's token grid (14 for patch-16 CLIP/SigLIP, 16 for patch-14 so400m). With the
    shared RGB positional embedding in DiTX, heatmap token i is then spatially aligned with
    RGB token i. Trained from scratch, so it is in-distribution for the synthetic Gaussian
    heatmaps (unlike a pretrained CLIP ViT) and does not touch the RGB backbone's weights.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256, grid_size: int = 14):
        super().__init__()
        self.input_size = grid_size * 16
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),    # H/2
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),             # H/4
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),            # H/8
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),  # H/16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        return self.net(x)  # [B, out_channels, grid_size, grid_size]


class MultiModalObsEncoder(nn.Module):
    """
    Siamese 2D encoder for N-camera RGB + N-camera XYZ pointmap.

    Camera keys, image resolution, pointmap bounds, and state MLP hidden dim
    are all configurable.  Token counts per camera are determined dynamically
    via probe forward passes during __init__.

    Two encoder_type modes:
      "spatial": features_only=True -> spatial tokens per camera
      "pooled":  num_classes=0      -> 1 global token per camera

    Returns dict: {"rgb_tokens", "ptmap_tokens", "state_emb"}.
    """

    def __init__(
        self,
        rgb_backbone: str = "vit_base_patch16_clip_224.openai",
        ptmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        state_dim: int = 28,
        propr_mask_prob: float = 0.0,
        encoder_type: str = "spatial",
        use_rgb: bool = True,
        use_ptmap: bool = True,
        rgb_keys: Optional[List[str]] = None,
        ptmap_keys: Optional[List[str]] = None,
        depth_keys: Optional[List[str]] = None,
        img_size: int = 224,
        ptmap_min: Optional[List[float]] = None,
        ptmap_max: Optional[List[float]] = None,
        rgb_mean: Optional[List[float]] = None,
        rgb_std: Optional[List[float]] = None,
        freeze_rgb_backbone: bool = False,
        use_heatmap: bool = False,
        heatmap_encoder_type: str = "shared_rgb",
        heatmap_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        # Sparse point encoder config
        ptmap_encoder_type: str = "dense",  # "dense" (ConvNeXtV2) or "sparse" (PointNet)
        sparse_num_points: int = 64,
        sparse_out_channels: int = 256,
        sparse_hidden_dim: int = 128,
        # Per-camera pointmap bounds: dict mapping depth/ptmap key -> {"min": [x,y,z], "max": [x,y,z]}
        per_camera_ptmap_bounds: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ):
        super().__init__()

        assert encoder_type in ("spatial", "pooled"), f"encoder_type must be 'spatial' or 'pooled', got '{encoder_type}'"
        assert use_rgb or use_ptmap, "At least one of use_rgb or use_ptmap must be True"

        self.propr_mask_prob = propr_mask_prob
        self.encoder_type = encoder_type
        self.use_rgb = use_rgb
        self.use_ptmap = use_ptmap
        self.freeze_rgb_backbone = freeze_rgb_backbone
        self.use_heatmap = use_heatmap
        self.heatmap_encoder_type = heatmap_encoder_type
        self.img_size = img_size

        # Camera keys (derived from shape_meta by the policy, or defaults)
        self.rgb_keys = (rgb_keys or ["head_rgb", "wrist_r_rgb", "wrist_l_rgb"]) if use_rgb else []
        self.ptmap_keys = ptmap_keys or ["head_ptmap", "wrist_r_ptmap", "wrist_l_ptmap"]
        self.depth_keys = depth_keys or []
        self.use_gpu_backproj = len(self.depth_keys) > 0
        self.n_rgb_cams = len(self.rgb_keys)
        self.n_ptmap_cams = len(self.depth_keys) if self.use_gpu_backproj else len(self.ptmap_keys)
        self.ptmap_encoder_type = ptmap_encoder_type
        self.sparse_num_points = sparse_num_points

        # Pointmap normalization bounds (global fallback)
        _ptmap_min = ptmap_min or [-1.0, -1.0, 0.0]
        _ptmap_max = ptmap_max or [1.0, 1.0, 2.0]

        # Per-camera bounds override
        self.per_camera_ptmap_bounds = per_camera_ptmap_bounds
        self._use_per_cam_norm = per_camera_ptmap_bounds is not None and len(per_camera_ptmap_bounds) > 0

        # ---- RGB backbone ----
        if use_rgb:
            if encoder_type == "spatial":
                self.rgb_encoder = timm.create_model(
                    f"hf_hub:timm/{rgb_backbone}",
                    pretrained=True,
                    features_only=True,
                    out_indices=[-1],
                )
                with torch.no_grad():
                    probe = self.rgb_encoder(torch.zeros(1, 3, img_size, img_size))[0]  # [1, C, H, W]
                    C_rgb = probe.shape[1]
                    self.tokens_per_rgb_cam = probe.shape[2] * probe.shape[3]
            else:
                self.rgb_encoder = timm.create_model(
                    f"hf_hub:timm/{rgb_backbone}",
                    pretrained=True,
                    num_classes=0,
                )
                with torch.no_grad():
                    probe = self.rgb_encoder(torch.zeros(1, 3, img_size, img_size))  # [1, C]
                    C_rgb = probe.shape[1]
                    self.tokens_per_rgb_cam = 1
            self.rgb_feature_dim = C_rgb
            if freeze_rgb_backbone:
                self.rgb_encoder.requires_grad_(False)
                self.rgb_encoder.eval()
                cprint(f"[MultiModalObsEncoder] RGB backbone FROZEN ({rgb_backbone})", "yellow")
        else:
            self.tokens_per_rgb_cam = 0
            self.rgb_feature_dim = 0

        # ---- Pointmap backbone ----
        if use_ptmap:
            if ptmap_encoder_type == "defm":
                self.defm_encoder = _load_defm_vit_s14()
                for p in self.defm_encoder.parameters():
                    p.requires_grad = False
                self.defm_encoder.eval()
                self.ptmap_feature_dim = 384  # ViT-S hidden dim
                self.tokens_per_ptmap_cam = (img_size // 14) ** 2  # 16x16 = 256 for 224
                cprint(f"[DeFM] Loaded frozen ViT-S/14 depth encoder "
                       f"({self.tokens_per_ptmap_cam} tokens/cam, dim={self.ptmap_feature_dim})", "cyan")
            else:
                # Pointmap normalization buffers (used by dense, sparse, shared_rgb)
                self.register_buffer(
                    "ptmap_min",
                    torch.tensor(_ptmap_min, dtype=torch.float32).view(1, 3, 1, 1),
                )
                self.register_buffer(
                    "ptmap_max",
                    torch.tensor(_ptmap_max, dtype=torch.float32).view(1, 3, 1, 1),
                )

                # Per-camera bounds: stacked [n_ptmap_cams, 3, 1, 1] for vectorized norm
                if self._use_per_cam_norm:
                    cam_keys = sorted(self.depth_keys) if self.use_gpu_backproj else sorted(self.ptmap_keys)
                    mins, maxs = [], []
                    for ck in cam_keys:
                        if ck in per_camera_ptmap_bounds:
                            b = per_camera_ptmap_bounds[ck]
                        else:
                            b = {"min": _ptmap_min, "max": _ptmap_max}
                            cprint(f"[MultiModalObsEncoder] No per-cam bounds for '{ck}', using global", "yellow")
                        mins.append(torch.tensor(b["min"], dtype=torch.float32).view(1, 3, 1, 1))
                        maxs.append(torch.tensor(b["max"], dtype=torch.float32).view(1, 3, 1, 1))
                    self.register_buffer("_per_cam_min", torch.cat(mins, dim=0))  # [N_cams, 3, 1, 1]
                    self.register_buffer("_per_cam_max", torch.cat(maxs, dim=0))
                    cprint(f"[MultiModalObsEncoder] Per-camera ptmap bounds for {cam_keys}", "cyan")
                    for i, ck in enumerate(cam_keys):
                        cprint(f"  {ck}: min={mins[i].squeeze().tolist()} max={maxs[i].squeeze().tolist()}", "cyan")

                # Pixel coordinate grids for GPU back-projection
                if self.use_gpu_backproj:
                    u = torch.arange(img_size, dtype=torch.float32)
                    v = torch.arange(img_size, dtype=torch.float32)
                    uu, vv = torch.meshgrid(u, v, indexing='xy')
                    self.register_buffer("_uu", uu, persistent=False)
                    self.register_buffer("_vv", vv, persistent=False)

                if ptmap_encoder_type == "sparse":
                    self.ptmap_encoder = SparsePointEncoder(
                        out_channels=sparse_out_channels,
                        hidden_dim=sparse_hidden_dim,
                    )
                    self.ptmap_feature_dim = sparse_out_channels
                    self.tokens_per_ptmap_cam = sparse_num_points
                elif ptmap_encoder_type == "shared_rgb":
                    assert use_rgb, "ptmap_encoder_type='shared_rgb' requires use_rgb=True"
                    self.ptmap_feature_dim = C_rgb
                    self.tokens_per_ptmap_cam = self.tokens_per_rgb_cam
                else:
                    _is_vit_backbone = "vit" in ptmap_backbone or "dino" in ptmap_backbone
                    _ptmap_kwargs = {"img_size": img_size} if _is_vit_backbone else {}
                    if encoder_type == "spatial":
                        self.ptmap_encoder = timm.create_model(
                            ptmap_backbone,
                            pretrained=True,
                            features_only=True,
                            out_indices=[-1],
                            **_ptmap_kwargs,
                        )
                        with torch.no_grad():
                            probe = self.ptmap_encoder(torch.zeros(1, 3, img_size, img_size))[0]
                            C_pt = probe.shape[1]
                            self.tokens_per_ptmap_cam = probe.shape[2] * probe.shape[3]
                    else:
                        self.ptmap_encoder = timm.create_model(
                            ptmap_backbone,
                            pretrained=True,
                            num_classes=0,
                            **_ptmap_kwargs,
                        )
                        with torch.no_grad():
                            probe = self.ptmap_encoder(torch.zeros(1, 3, img_size, img_size))
                            C_pt = probe.shape[1]
                            self.tokens_per_ptmap_cam = 1
                    self.ptmap_feature_dim = C_pt
        else:
            self.tokens_per_ptmap_cam = 0

        # ---- RGB normalization ----
        # Augmentation lives in the dataloader (JointImageAugment), which draws it
        # per sample and replays the same geometry on RGB, heatmap and depth.
        if use_rgb:
            _clip_mean = [0.48145466, 0.4578275, 0.40821073]
            _clip_std = [0.26862954, 0.26130258, 0.27577711]
            if rgb_mean is not None:
                _rgb_mean = rgb_mean
            elif hasattr(self.rgb_encoder, "pretrained_cfg"):
                _rgb_mean = list(self.rgb_encoder.pretrained_cfg.get("mean", _clip_mean))
            else:
                _rgb_mean = _clip_mean
            if rgb_std is not None:
                _rgb_std = rgb_std
            elif hasattr(self.rgb_encoder, "pretrained_cfg"):
                _rgb_std = list(self.rgb_encoder.pretrained_cfg.get("std", _clip_std))
            else:
                _rgb_std = _clip_std
            self.register_buffer("img_mean", torch.tensor(_rgb_mean).view(1, 3, 1, 1))
            self.register_buffer("img_std", torch.tensor(_rgb_std).view(1, 3, 1, 1))
            cprint(f"[MultiModalObsEncoder] RGB norm mean={_rgb_mean} std={_rgb_std}", "cyan")

        # ---- Heatmap backbone (configurable: shared_rgb / convnextv2 / disabled) ----
        if use_heatmap:
            if heatmap_encoder_type == "shared_rgb":
                assert use_rgb, "heatmap_encoder_type='shared_rgb' requires use_rgb=True"
                self.heatmap_feature_dim = self.rgb_feature_dim
                self.tokens_per_heatmap = self.tokens_per_rgb_cam
            elif heatmap_encoder_type == "convnextv2":
                if encoder_type == "spatial":
                    self.heatmap_encoder = timm.create_model(
                        heatmap_backbone, pretrained=True,
                        features_only=True, out_indices=[-1])
                    with torch.no_grad():
                        probe = self.heatmap_encoder(torch.zeros(1, 3, img_size, img_size))[0]
                        self.heatmap_feature_dim = probe.shape[1]
                        self.tokens_per_heatmap = probe.shape[2] * probe.shape[3]
                else:
                    self.heatmap_encoder = timm.create_model(
                        heatmap_backbone, pretrained=True, num_classes=0)
                    with torch.no_grad():
                        probe = self.heatmap_encoder(torch.zeros(1, 3, img_size, img_size))
                        self.heatmap_feature_dim = probe.shape[1]
                        self.tokens_per_heatmap = 1
            elif heatmap_encoder_type == "cnn":
                assert use_rgb, "heatmap_encoder_type='cnn' shares the RGB positional grid; requires use_rgb=True"
                assert encoder_type == "spatial", "heatmap_encoder_type='cnn' requires encoder_type='spatial'"
                self.heatmap_encoder = HeatmapCNNEncoder(
                    in_channels=3, out_channels=256,
                    grid_size=int(round(math.sqrt(self.tokens_per_rgb_cam))))
                with torch.no_grad():
                    probe = self.heatmap_encoder(torch.zeros(1, 3, img_size, img_size))  # [1, C, H/16, W/16]
                    self.heatmap_feature_dim = probe.shape[1]
                    self.tokens_per_heatmap = probe.shape[2] * probe.shape[3]
                assert self.tokens_per_heatmap == self.tokens_per_rgb_cam, (
                    f"cnn heatmap grid ({self.tokens_per_heatmap} tokens) must match the RGB grid "
                    f"({self.tokens_per_rgb_cam} tokens) so heatmap token i aligns with RGB token i")
            else:
                raise ValueError(f"Unknown heatmap_encoder_type: {heatmap_encoder_type}")
            cprint(f"[MultiModalObsEncoder] heatmap encoder: type={heatmap_encoder_type} "
                   f"({self.tokens_per_heatmap} tokens, dim={self.heatmap_feature_dim})", "cyan")
        else:
            self.heatmap_feature_dim = 0
            self.tokens_per_heatmap = 0

        self.state_dim = state_dim
        # NOTE: state adaptor and per-modality LayerNorm are applied inside DiTX.

        _ptmap_desc = "disabled"
        if use_ptmap:
            if ptmap_encoder_type == "defm":
                _ptmap_desc = f"defm_vit_s14(frozen, {self.tokens_per_ptmap_cam} tok/cam)"
            elif ptmap_encoder_type == "sparse":
                _ptmap_desc = f"sparse_pointnet(num_pts={sparse_num_points}, dim={sparse_out_channels})"
            elif ptmap_encoder_type == "shared_rgb":
                _ptmap_desc = f"shared_rgb({rgb_backbone})"
            else:
                _ptmap_desc = ptmap_backbone
        cprint(
            f"[MultiModalObsEncoder] encoder_type={encoder_type} use_rgb={use_rgb} use_ptmap={use_ptmap} "
            f"rgb={rgb_backbone if use_rgb else 'disabled'}"
            f"{f' ({self.n_rgb_cams} cams, {self.tokens_per_rgb_cam} tok/cam)' if use_rgb else ''} "
            f"ptmap={_ptmap_desc}"
            f"{f' ({self.n_ptmap_cams} cams, {self.tokens_per_ptmap_cam} tok/cam)' if use_ptmap else ''} "
            f"img_size={img_size} state_dim={state_dim}",
            "cyan",
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_ptmap and self.ptmap_encoder_type == "defm":
            self.defm_encoder.eval()
        if self.use_rgb and self.freeze_rgb_backbone:
            self.rgb_encoder.eval()
        return self

    @property
    def visual_token_count(self) -> int:
        """Total visual tokens (without state token)."""
        count = self.n_rgb_cams * self.tokens_per_rgb_cam
        if self.use_ptmap:
            count += self.n_ptmap_cams * self.tokens_per_ptmap_cam
        return count

    def _depth_to_pointmap_gpu(self, depth: torch.Tensor, intrinsics: torch.Tensor,
                               H_inv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """GPU pinhole back-projection: depth + intrinsics -> XYZ pointmap.

        Args:
            depth: [B*N, 1, H, W] float32 depth in meters
            intrinsics: [B*N, 4] float32 [fx, fy, cx, cy] rescaled to (H, W)
            H_inv: optional [B, 3, 3] augmented pixel -> source pixel (train-time augmentation,
                shared by the N cameras); each pixel is un-projected along its source pixel's ray

        Returns:
            [B*N, 3, H, W] float32 XYZ pointmap
        """
        fx = intrinsics[:, 0:1, None, None]
        fy = intrinsics[:, 1:2, None, None]
        cx = intrinsics[:, 2:3, None, None]
        cy = intrinsics[:, 3:4, None, None]
        u, v = self._uu[None, None], self._vv[None, None]  # [1, 1, H, W]
        if H_inv is not None:
            uv1 = torch.stack([self._uu, self._vv, torch.ones_like(self._uu)])  # [3, H, W]
            H_inv = H_inv.repeat_interleave(depth.shape[0] // H_inv.shape[0], dim=0)  # [B*N, 3, 3]
            u, v = torch.einsum("nij,jhw->nihw", H_inv[:, :2], uv1).split(1, dim=1)  # [B*N, 1, H, W]
        X = (u - cx) * depth / fx.clamp(min=1e-6)
        Y = (v - cy) * depth / fy.clamp(min=1e-6)
        return torch.cat([X, Y, depth], dim=1)

    def _sparse_encode(self, ptmaps_raw: torch.Tensor, B: int) -> torch.Tensor:
        """Sparse point sampling + PointNet encoding.

        Args:
            ptmaps_raw: [B*N_cams, 3, H, W] XYZ pointmap from back-projection
            B: batch size

        Returns:
            [B, N_cams * num_points, feature_dim] point tokens
        """
        BN = ptmaps_raw.shape[0]
        N_pts = self.sparse_num_points
        HW = self.img_size * self.img_size

        # Normalize first (handles per-camera bounds), then flatten
        ptmaps_norm = self._normalize_ptmaps(ptmaps_raw, B)  # [BN, 3, H, W]
        points_norm = ptmaps_norm.permute(0, 2, 3, 1).reshape(BN, HW, 3)

        # Vectorized random sampling: assign random priorities, mask invalid to -inf,
        # then topk selects N_pts highest-priority valid points per batch element.
        valid_mask = ptmaps_raw[:, 2, :, :].reshape(BN, HW) > 0  # [BN, H*W]
        rand_priority = torch.rand(BN, HW, device=ptmaps_raw.device)
        rand_priority[~valid_mask] = -1.0
        _, top_idx = rand_priority.topk(N_pts, dim=1)  # [BN, N_pts]

        # Gather sampled points: [BN, N_pts, 3]
        sampled = torch.gather(points_norm, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))

        # PointNet encoding: [BN, N_pts, 3] → [BN, N_pts, out_channels]
        point_feats = self.ptmap_encoder(sampled)
        return point_feats.reshape(B, self.n_ptmap_cams * N_pts, self.ptmap_feature_dim)

    def _normalize_ptmaps(self, ptmaps_raw: torch.Tensor, B: int) -> torch.Tensor:
        """Clamp + normalize pointmaps to [-1, 1].

        Uses per-camera bounds if available, otherwise global bounds.
        Args:
            ptmaps_raw: [B*N_cams, 3, H, W]
            B: batch size
        Returns:
            [B*N_cams, 3, H, W] normalized to [-1, 1]
        """
        if self._use_per_cam_norm:
            BN = ptmaps_raw.shape[0]
            N = self.n_ptmap_cams
            # Reshape to [B, N, 3, H, W] so we can apply per-camera bounds
            ptmaps = ptmaps_raw.view(B, N, 3, self.img_size, self.img_size)
            # _per_cam_min/max are [N, 3, 1, 1] — broadcast over B, H, W
            lo = self._per_cam_min.unsqueeze(0)  # [1, N, 3, 1, 1]
            hi = self._per_cam_max.unsqueeze(0)
            ptmaps = torch.clamp(ptmaps, lo, hi)
            ptmaps = 2.0 * (ptmaps - lo) / (hi - lo).clamp(min=1e-6) - 1.0
            return ptmaps.view(BN, 3, self.img_size, self.img_size)
        else:
            ptmaps = torch.clamp(ptmaps_raw, self.ptmap_min, self.ptmap_max)
            return 2.0 * (ptmaps - self.ptmap_min) / (self.ptmap_max - self.ptmap_min) - 1.0

    def _aggregate_spatial(self, feat_4d: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, H*W, C]"""
        _B, _C, _H, _W = feat_4d.shape
        return feat_4d.flatten(2).transpose(1, 2)

    def forward(self, obs: dict) -> dict:
        """
        Returns dict with per-modality raw features:
          "rgb_tokens":  (B, n_rgb_cams * L_rgb, C_rgb)   or None
          "ptmap_tokens": (B, n_ptmap_cams * L_pt, C_pt)  or None
          "state_emb":   (B, state_dim)
        """
        B = obs["agent_pos"].shape[0]

        # ---- 1. RGB path (backbone only, no projection) ----
        rgb_tokens = None
        if self.use_rgb:
            rgb_imgs = torch.stack([obs[k] for k in self.rgb_keys], dim=1)
            rgb_flat = rgb_imgs.reshape(B * self.n_rgb_cams, 3, self.img_size, self.img_size)
            rgb_flat = (rgb_flat - self.img_mean) / self.img_std
            rgb_feat_raw = self.rgb_encoder(rgb_flat)
            if self.encoder_type == "spatial":
                rgb_spatial = self._aggregate_spatial(rgb_feat_raw[0])
            else:
                rgb_spatial = rgb_feat_raw.unsqueeze(1)
            L_rgb = rgb_spatial.shape[1]
            rgb_tokens = rgb_spatial.reshape(B, self.n_rgb_cams * L_rgb, self.rgb_feature_dim)

        # ---- 2. Pointmap/Depth path ----
        ptmap_tokens = None
        if self.use_ptmap:
            if self.ptmap_encoder_type == "defm":
                # DeFM path: raw depth -> 3-channel metric norm -> frozen ViT-S/14
                with torch.no_grad():
                    depth_stack = torch.stack([obs[k] for k in self.depth_keys], dim=1)
                    depth_flat = depth_stack.reshape(B * self.n_ptmap_cams, 1, self.img_size, self.img_size)
                    defm_input = defm_preprocess_depth_batch(depth_flat)
                    out = self.defm_encoder.get_intermediate_layers(
                        defm_input, n=1, reshape=True, return_class_token=False
                    )
                    spatial_feat = out[0]  # [BN, C, H', W']
                ptmap_spatial = self._aggregate_spatial(spatial_feat)  # [BN, H'*W', C]
                L_pt = ptmap_spatial.shape[1]
                ptmap_tokens = ptmap_spatial.reshape(B, self.n_ptmap_cams * L_pt, self.ptmap_feature_dim)
            else:
                if self.use_gpu_backproj:
                    with torch.no_grad():
                        depth_stack = torch.stack([obs[k] for k in self.depth_keys], dim=1)
                        depth_flat = depth_stack.reshape(B * self.n_ptmap_cams, 1, self.img_size, self.img_size)
                        intrinsics = obs["camera_intrinsics"]
                        intrinsics_flat = intrinsics.reshape(B * self.n_ptmap_cams, 4)
                        ptmaps_raw = self._depth_to_pointmap_gpu(depth_flat, intrinsics_flat, obs.get("aug_H_inv"))
                else:
                    # Possibly Not implemented for Franka. Add an assert if confident
                    ptmaps_raw = torch.stack([obs[k] for k in self.ptmap_keys], dim=1)
                    ptmaps_raw = ptmaps_raw.reshape(B * self.n_ptmap_cams, 3, self.img_size, self.img_size)

                if self.ptmap_encoder_type == "sparse":
                    ptmap_tokens = self._sparse_encode(ptmaps_raw, B)
                elif self.ptmap_encoder_type == "shared_rgb":
                    ptmaps_norm = self._normalize_ptmaps(ptmaps_raw, B)
                    ptmap_feat_raw = self.rgb_encoder(ptmaps_norm)
                    if self.encoder_type == "spatial":
                        ptmap_spatial = self._aggregate_spatial(ptmap_feat_raw[0])
                    else:
                        ptmap_spatial = ptmap_feat_raw.unsqueeze(1)
                    L_pt = ptmap_spatial.shape[1]
                    ptmap_tokens = ptmap_spatial.reshape(B, self.n_ptmap_cams * L_pt, self.ptmap_feature_dim)
                else:
                    ptmaps_norm = self._normalize_ptmaps(ptmaps_raw, B)
                    ptmap_feat_raw = self.ptmap_encoder(ptmaps_norm)
                    if self.encoder_type == "spatial":
                        ptmap_spatial = self._aggregate_spatial(ptmap_feat_raw[0])
                    else:
                        ptmap_spatial = ptmap_feat_raw.unsqueeze(1)
                    L_pt = ptmap_spatial.shape[1]
                    ptmap_tokens = ptmap_spatial.reshape(B, self.n_ptmap_cams * L_pt, self.ptmap_feature_dim)

        # ---- 3. Heatmap path ----
        heatmap_tokens = None
        if self.use_heatmap and "target_heatmap" in obs:
            hm_img = obs["target_heatmap"]  # (B, 3, H, W)
            if self.heatmap_encoder_type == "shared_rgb":
                hm_normed = (hm_img - self.img_mean) / self.img_std
                hm_feat = self.rgb_encoder(hm_normed)
                if self.encoder_type == "spatial":
                    heatmap_tokens = self._aggregate_spatial(hm_feat[0])
                else:
                    heatmap_tokens = hm_feat.unsqueeze(1)
            elif self.heatmap_encoder_type == "convnextv2":
                hm_feat = self.heatmap_encoder(hm_img)
                if self.encoder_type == "spatial":
                    heatmap_tokens = self._aggregate_spatial(hm_feat[0])
                else:
                    heatmap_tokens = hm_feat.unsqueeze(1)
            elif self.heatmap_encoder_type == "cnn":
                # From-scratch CNN on the raw [0,1] heatmap -> [B, C, H/16, W/16] on the
                # same grid as the RGB ViT tokens.
                hm_feat = self.heatmap_encoder(hm_img)
                heatmap_tokens = self._aggregate_spatial(hm_feat)

        # ---- 4. State path (masking only, adaptor is in DiTX) ----
        agent_pos = obs["agent_pos"]
        if self.training and self.propr_mask_prob > 0:
            mask = (torch.rand(B, 1, device=agent_pos.device) > self.propr_mask_prob).float()
            agent_pos = agent_pos * mask

        return {
            "rgb_tokens": rgb_tokens,
            "ptmap_tokens": ptmap_tokens,
            "heatmap_tokens": heatmap_tokens,
            "state_emb": agent_pos,
        }
