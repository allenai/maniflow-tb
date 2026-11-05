"""
DiTX: Consistency Flow Training model with a Diffusion Transformer backbone.
(standalone, no maniflow dependency)

Merges:
  - maniflow.model.diffusion.positional_embedding (SinusoidalPosEmb)
  - maniflow.model.diffusion.ditx_block (modulate, AdaptiveLayerNorm, CrossAttention, DiTXBlock)
  - maniflow.model.diffusion.ditx (FinalLayer, DiTX)

References:
  DiT: https://github.com/facebookresearch/DiT
  RDT: https://github.com/thu-ml/RoboticsDiffusionTransformer
"""

import re
import math
import logging
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, RmsNorm, use_fused_attn
from termcolor import cprint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------------------------------------------------------
# DiTX block components
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_linear = nn.Linear(dim_cond, dim * 2)
        self.cond_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            self.cond_linear
        )
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[:dim], 1.)
        nn.init.zeros_(self.cond_linear.bias[dim:])

    def forward(self, x, cond=None):
        x = self.ln(x)
        gamma, beta = self.cond_modulation(cond).chunk(2, dim=-1)
        x = x * gamma + beta
        return x


class CrossAttention(nn.Module):
    """A cross-attention layer with flash attention."""
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q, key=k, value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


class DiTXBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, p_drop_attn=0.,
                 qkv_bias=False, qk_norm=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=p_drop_attn)

        # Cross-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            norm_layer=nn.LayerNorm, **block_kwargs)

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu, drop=0.0)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # AdaLN modulation
        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )

    def forward(self, x, time_c, context_c, attn_mask=None, context_mask=None):
        modulation = self.adaLN_modulation(time_c)
        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # Self-Attention
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # Cross-Attention
        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=context_mask)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output

        # MLP
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


# ---------------------------------------------------------------------------
# DiTX main model
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """The final layer of DIT-X, adopted from RDT."""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class DiTX(nn.Module):
    """Consistency Flow Training model with a Diffusion Transformer backbone."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        rgb_cond_dim: int = 768,
        ptmap_cond_dim: int = 768,
        state_dim: int = 28,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_target_t_embed_dim: int = 256,
        block_type: str = "DiTX",
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        pre_norm_modality: bool = False,
        language_conditioned: bool = False,
        language_model: str = "t5-small",
        state_injection_mode: Optional[str] = None,
        lang_injection_mode: str = "token",
        use_rgb: bool = True,
        use_ptmap: bool = True,
        n_rgb_tokens: int = 0,
        n_ptmap_tokens: int = 0,
        ptmap_injection_mode: str = "token",
        target_injection_mode: Optional[str] = None,
        target_sigma: float = 1.5,
        target_mask_prob: float = 0.0,
        heatmap_cond_dim: int = 0,
        n_heatmap_tokens: int = 0,
        tokens_per_rgb_cam: int = 196,
        img_size: int = 224,
        heatmap_share_rgb_pos_embed: bool = False,
        # Deprecated aliases — ignored, kept for checkpoint compat
        target_conditioned: bool = False,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.language_conditioned = language_conditioned
        self.pre_norm_modality = pre_norm_modality
        self.state_injection_mode = state_injection_mode or "token"
        self.lang_injection_mode = lang_injection_mode

        T = horizon
        self.T = T
        self.horizon = horizon

        # input embedding stem
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))

        # per-modality adaptors (source_dim -> n_emb), following lang_adaptor pattern
        self.use_rgb = use_rgb
        self.use_ptmap = use_ptmap
        self.ptmap_injection_mode = ptmap_injection_mode if use_ptmap else "token"
        self.n_rgb_tokens = n_rgb_tokens if use_rgb else 0
        self.n_ptmap_tokens = n_ptmap_tokens if use_ptmap else 0
        if use_rgb:
            self.rgb_adaptor = self.build_condition_adapter("mlp2x_gelu", rgb_cond_dim, n_emb)
        if use_ptmap:
            self.ptmap_adaptor = self.build_condition_adapter("mlp2x_gelu", ptmap_cond_dim, n_emb)
        self.state_adaptor = self.build_condition_adapter("mlp2x_gelu", state_dim, n_emb)

        # per-modality positional embedding + LayerNorm
        if use_rgb:
            self.rgb_pos_embed = nn.Parameter(
                torch.zeros(1, n_rgb_tokens * n_obs_steps, n_emb))
            self.rgb_norm = nn.LayerNorm(n_emb)
        if use_ptmap:
            self.ptmap_pos_embed = nn.Parameter(
                torch.zeros(1, n_ptmap_tokens * n_obs_steps, n_emb))
            self.ptmap_norm = nn.LayerNorm(n_emb)
        self.state_pos_embed = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.state_norm = nn.LayerNorm(n_emb)

        # timestep and target_t cond encoder
        self.flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        self.flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_target_t_embed_dim),
            nn.Linear(diffusion_target_t_embed_dim, diffusion_target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_target_t_embed_dim * 4, self.hidden_dim),
        )

        # Dynamic time adaptor input dim
        n_time_inputs = 2
        if self.state_injection_mode == "timestep_concat":
            n_time_inputs += 1
        if self.language_conditioned and self.lang_injection_mode == "timestep_concat":
            n_time_inputs += 1
        if target_injection_mode == "timestep_concat":
            n_time_inputs += 1
        if use_ptmap and self.ptmap_injection_mode == "timestep_concat":
            n_time_inputs += 1
        time_adaptor_in_dim = self.hidden_dim * n_time_inputs
        self.timestep_target_t_adaptor = nn.Linear(time_adaptor_in_dim, self.hidden_dim)

        # Language conditioning
        if self.language_conditioned:
            self.load_T5_encoder(model_name=language_model, freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu",
                in_features=self.language_encoder_out_dim,
                out_features=n_emb
            )
            if self.lang_injection_mode == "token":
                self.lang_pos_embed = nn.Parameter(torch.zeros(1, 64, n_emb))
                self.lang_norm = nn.LayerNorm(n_emb)

        # Spatial target conditioning
        if target_injection_mode == "legacy":
            target_injection_mode = "target_embedding"
        if target_conditioned and target_injection_mode is None:
            target_injection_mode = "target_embedding"
        self.target_injection_mode = target_injection_mode
        self.target_conditioned = target_injection_mode is not None
        # When True, heatmap tokens reuse the RGB positional embedding (front-cam slice) so
        # heatmap token i is aligned with RGB token i instead of a separate learned PE.
        self.heatmap_share_rgb_pos_embed = bool(heatmap_share_rgb_pos_embed) and use_rgb
        self.target_sigma = target_sigma
        self.target_mask_prob = target_mask_prob
        self.tokens_per_rgb_cam = tokens_per_rgb_cam
        self.img_size = img_size
        # Derived from the backbone's actual token count so patch-14 backbones
        # (e.g. so400m SigLIP: 16x16 tokens) map target_xy to the right patch.
        self.grid_size = int(round(math.sqrt(tokens_per_rgb_cam))) if tokens_per_rgb_cam else img_size // 16
        self.patch_size = img_size / self.grid_size
        if self.target_conditioned:
            if target_injection_mode == "target_embedding":
                self.target_embed = nn.Parameter(torch.zeros(1, 1, n_emb))
                cprint(f"[DiTX] Target embedding conditioning "
                       f"(sigma={target_sigma}, mask_prob={target_mask_prob})", "cyan")
            elif target_injection_mode == "target_pe":
                self.target_pos_embed = nn.Parameter(
                    torch.zeros(1, tokens_per_rgb_cam, n_emb))
                cprint(f"[DiTX] Target PE conditioning "
                       f"(sigma={target_sigma}, mask_prob={target_mask_prob})", "cyan")
            else:
                self.heatmap_adaptor = self.build_condition_adapter(
                    "mlp2x_gelu", heatmap_cond_dim, n_emb)
                self.heatmap_norm = nn.LayerNorm(n_emb)
                if target_injection_mode == "token" and not self.heatmap_share_rgb_pos_embed:
                    self.heatmap_pos_embed = nn.Parameter(
                        torch.zeros(1, n_heatmap_tokens * n_obs_steps, n_emb))
                cprint(f"[DiTX] Heatmap target conditioning: mode={target_injection_mode}, "
                       f"n_tokens={n_heatmap_tokens}, cond_dim={heatmap_cond_dim}, "
                       f"mask_prob={target_mask_prob}", "cyan")

        # Transformer blocks
        self.block_type = block_type
        if block_type == "DiTX":
            self.blocks = nn.ModuleList([
                DiTXBlock(n_emb, n_head, mlp_ratio=mlp_ratio, p_drop_attn=p_drop_attn,
                    qkv_bias=qkv_bias, qk_norm=qk_norm) for _ in range(n_layer)
            ])
            cprint(f"[DiTX Transformer] Initialized {n_layer} DiTX blocks with hidden size {n_emb}, "
                    f"num heads {n_head}, mlp ratio {mlp_ratio}, dropout {p_drop_attn}, "
                    f"qkv_bias {qkv_bias}, qk_norm {qk_norm}", "cyan")

        # Final Layer
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.initialize_weights()
        cprint(f"[DiTX Transformer] Initialized weights for DiTX", "green")

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')
        return projector

    def load_T5_encoder(self, model_name, freeze=True):
        import time as _time
        from transformers import T5Config, T5EncoderModel, AutoTokenizer
        import torch.distributed as dist
        t5_model_name = ["t5-small", "t5-base", "t5-large"]
        assert model_name in t5_model_name, f"Model name {model_name} not in {t5_model_name}"
        pretrained_model_id = f"google-t5/{model_name}"
        if model_name == "t5-small":
            encoder_cfg = T5Config(
                d_model=512,
                d_kv=64,
                d_ff=2048,
                num_layers=6,
                num_heads=6,
            )
        elif model_name == "t5-base":
            encoder_cfg = T5Config(
                d_model=768,
                d_kv=64,
                d_ff=3072,
                num_layers=12,
                num_heads=12,
            )
        else:  # t5-large
            encoder_cfg = T5Config(
                d_model=1024,
                d_kv=64,
                d_ff=4096,
                num_layers=24,
                num_heads=16,
            )
        self.language_encoder_out_dim = encoder_cfg.d_model
        is_distributed = dist.is_initialized()
        is_main = (not is_distributed) or (dist.get_rank() == 0)
        if is_distributed and not is_main:
            dist.barrier()

        for attempt in range(5):
            try:
                self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(pretrained_model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
                break
            except (OSError, Exception) as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt * 10
                cprint(f"[T5] Attempt {attempt+1} failed ({e}), retrying in {wait}s...", "yellow")
                _time.sleep(wait)

        if is_distributed and is_main:
            dist.barrier()
        if freeze:
            self.language_encoder.eval()
            for param in self.language_encoder.parameters():
                param.requires_grad = False
        cprint(f"Loaded T5 encoder: {model_name}", "green")

    def encode_text_input_T5(self, lang_cond, norm_lang_embedding=False,
                             output_type="sentence",
                             device="cuda" if torch.cuda.is_available() else "cpu"):
        language_inputs = self.tokenizer(
            lang_cond, return_tensors="pt", padding=True, truncation=True)
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings, attention_mask
        # Mask-weighted mean
        m = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        sentence_embedding = (token_embeddings * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)
        return sentence_embedding, attention_mask

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None
        nn.init.normal_(self.pos_emb, std=0.02)

        # per-modality positional embeddings (+ target_embed / heatmap_pos_embed)
        if self.target_injection_mode == "target_embedding":
            nn.init.normal_(self.target_embed, std=0.02)
        for pe_name in ["rgb_pos_embed", "ptmap_pos_embed", "state_pos_embed",
                        "lang_pos_embed", "heatmap_pos_embed", "target_pos_embed"]:
             if hasattr(self, pe_name):
                nn.init.normal_(getattr(self, pe_name), std=0.02)

        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.flow_target_t_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for mlp_name in ["rgb_adaptor", "ptmap_adaptor", "state_adaptor", "heatmap_adaptor"]:
            if hasattr(self, mlp_name):
                mlp = getattr(self, mlp_name)
                nn.init.kaiming_normal_(mlp[0].weight, nonlinearity='leaky_relu')
                nn.init.constant_(mlp[0].bias, 0) if mlp[0].bias is not None else None
                nn.init.kaiming_normal_(mlp[-1].weight, nonlinearity='linear')
                nn.init.constant_(mlp[-1].bias, 0) if mlp[-1].bias is not None else None
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)

        if self.language_conditioned:
            nn.init.normal_(self.lang_adaptor[0].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[0].bias, 0) if self.lang_adaptor[0].bias is not None else None
            nn.init.normal_(self.lang_adaptor[-1].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[-1].bias, 0) if self.lang_adaptor[-1].bias is not None else None

        # per-modality nn.LayerNorm: default init (weight=1, bias=0) is correct

        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def _build_target_heatmap(self, target_xy: torch.Tensor, n_total_rgb_tokens: int,
                              device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build a (B, n_total_rgb_tokens) Gaussian heatmap from pixel coords.

        target_xy: (B, 2) pixel coordinates (x, y) in the resized img_size space (e.g. 224x224).
                   The caller is responsible for scaling from original resolution.
        Returns:   (B, n_total_rgb_tokens) with Gaussian on first-camera tokens, zeros elsewhere.
        """
        B = target_xy.shape[0]
        G = self.grid_size  # 14

        patch_col = target_xy[:, 0].float() / self.patch_size
        patch_row = target_xy[:, 1].float() / self.patch_size

        # Samples with NaN target_xy get a zero heatmap (no target signal)
        nan_mask = torch.isnan(patch_col) | torch.isnan(patch_row)
        patch_col = torch.where(nan_mask, torch.zeros_like(patch_col), patch_col)
        patch_row = torch.where(nan_mask, torch.zeros_like(patch_row), patch_row)

        row_coords = torch.arange(G, device=device, dtype=dtype)
        col_coords = torch.arange(G, device=device, dtype=dtype)
        grid_r, grid_c = torch.meshgrid(row_coords, col_coords, indexing="ij")  # (G, G)
        grid_r = grid_r.reshape(1, -1)  # (1, G*G)
        grid_c = grid_c.reshape(1, -1)  # (1, G*G)

        dr = grid_r - patch_row.unsqueeze(1)  # (B, G*G)
        dc = grid_c - patch_col.unsqueeze(1)  # (B, G*G)
        heatmap = torch.exp(-(dr ** 2 + dc ** 2) / (2.0 * self.target_sigma ** 2))  # (B, G*G=196)

        # Zero out heatmap for samples that had NaN target_xy
        heatmap = heatmap * (~nan_mask).unsqueeze(1).to(heatmap.dtype)

        # Pad with zeros for other cameras
        if n_total_rgb_tokens > self.tokens_per_rgb_cam:
            pad = torch.zeros(B, n_total_rgb_tokens - self.tokens_per_rgb_cam,
                              device=device, dtype=dtype)
            heatmap = torch.cat([heatmap, pad], dim=1)  # (B, n_total_rgb_tokens)

        return heatmap

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                target_t: Union[torch.Tensor, float, int],
                rgb_cond: Optional[torch.Tensor] = None,
                ptmap_cond: Optional[torch.Tensor] = None,
                state_emb: Optional[torch.Tensor] = None,
                lang_cond: Union[torch.Tensor, list, str] = None,
                target_xy: Optional[torch.Tensor] = None,
                heatmap_cond: Optional[torch.Tensor] = None,
                **kwargs):
        # process input
        input_emb = self.input_emb(sample)
        x = input_emb + self.pos_emb

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.flow_timestep_encoder(timesteps)

        # 2. target_t
        target_ts = target_t
        if not torch.is_tensor(target_ts):
            target_ts = torch.tensor([target_ts], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(target_ts) and len(target_ts.shape) == 0:
            target_ts = target_ts[None].to(sample.device)
        target_ts = target_ts.expand(sample.shape[0])
        target_t_embed = self.flow_target_t_encoder(target_ts)

        # Time conditioning
        time_parts = [timestep_embed, target_t_embed]
        if self.state_injection_mode == "timestep_concat":
            if state_emb is None:
                raise ValueError(
                    "state_injection_mode is 'timestep_concat' but state_emb is None.")
            state_adapted = self.state_adaptor(state_emb)
            state_adapted = self.state_norm(state_adapted)
            time_parts.append(state_adapted)
        if self.language_conditioned and self.lang_injection_mode == "timestep_concat":
            assert lang_cond is not None, "language_conditioned=True but lang_cond is None"
            lang_sent, _ = self.encode_text_input_T5(lang_cond, output_type="sentence", device=sample.device)
            lang_sent = self.lang_adaptor(lang_sent)
            time_parts.append(lang_sent)
        if self.target_injection_mode == "timestep_concat" and heatmap_cond is not None:
            hm_pooled = heatmap_cond.mean(dim=1)  # (B, C)
            hm_adapted = self.heatmap_adaptor(hm_pooled)
            if self.training and self.target_mask_prob > 0:
                mask = (torch.rand(hm_adapted.shape[0], 1, device=hm_adapted.device)
                        > self.target_mask_prob).to(hm_adapted.dtype)
                hm_adapted = hm_adapted * mask
            hm_adapted = self.heatmap_norm(hm_adapted)
            time_parts.append(hm_adapted)
        if self.use_ptmap and self.ptmap_injection_mode == "timestep_concat" and ptmap_cond is not None:
            ptmap_pooled = ptmap_cond.mean(dim=1)  # (B, C)
            ptmap_adapted = self.ptmap_adaptor(ptmap_pooled)
            ptmap_adapted = self.ptmap_norm(ptmap_adapted)
            time_parts.append(ptmap_adapted)
        time_input = torch.cat(time_parts, dim=-1)
        time_c = self.timestep_target_t_adaptor(time_input)

        # 3. context tokens (per-modality: adaptor -> pos_embed -> LayerNorm)
        parts = []
        lang_mask, lang_start = None, 0
        if rgb_cond is not None:
            rgb_emb = self.rgb_adaptor(rgb_cond)
            rgb_emb = rgb_emb + self.rgb_pos_embed[:, :rgb_emb.shape[1]]
            if self.target_injection_mode == "target_embedding" and target_xy is not None:
                heatmap = self._build_target_heatmap(
                    target_xy, rgb_emb.shape[1], rgb_emb.device, rgb_emb.dtype)
                if self.training and self.target_mask_prob > 0:
                    mask = (torch.rand(rgb_emb.shape[0], 1, device=rgb_emb.device)
                            > self.target_mask_prob).to(rgb_emb.dtype)
                    heatmap = heatmap * mask
                rgb_emb = rgb_emb + heatmap.unsqueeze(-1) * self.target_embed
            elif self.target_injection_mode == "target_pe" and target_xy is not None:
                heatmap = self._build_target_heatmap(
                    target_xy, self.tokens_per_rgb_cam, rgb_emb.device, rgb_emb.dtype)
                if self.training and self.target_mask_prob > 0:
                    mask = (torch.rand(rgb_emb.shape[0], 1, device=rgb_emb.device)
                            > self.target_mask_prob).to(rgb_emb.dtype)
                    heatmap = heatmap * mask
                target_enc = heatmap.unsqueeze(-1) * self.target_pos_embed
                rgb_emb[:, :self.tokens_per_rgb_cam] = (
                    rgb_emb[:, :self.tokens_per_rgb_cam] + target_enc)
            rgb_emb = self.rgb_norm(rgb_emb)
            parts.append(rgb_emb)
        if ptmap_cond is not None and self.ptmap_injection_mode == "token":
            ptmap_emb = self.ptmap_adaptor(ptmap_cond)
            ptmap_emb = ptmap_emb + self.ptmap_pos_embed[:, :ptmap_emb.shape[1]]
            ptmap_emb = self.ptmap_norm(ptmap_emb)
            parts.append(ptmap_emb)
        if self.state_injection_mode == "token":
            state_tok = self.state_adaptor(state_emb).unsqueeze(1)  # (B, 1, n_emb)
            state_tok = state_tok + self.state_pos_embed
            state_tok = self.state_norm(state_tok)
            parts.append(state_tok)
        if self.language_conditioned and self.lang_injection_mode == "token":
            assert lang_cond is not None
            lang_start = sum(p.shape[1] for p in parts)
            lang_c, lang_mask = self.encode_text_input_T5(lang_cond, output_type="token", device=sample.device)
            lang_c = self.lang_adaptor(lang_c)
            lang_c = lang_c + self.lang_pos_embed[:, :lang_c.shape[1]]
            lang_c = self.lang_norm(lang_c)
            parts.append(lang_c)
        if self.target_injection_mode == "token" and heatmap_cond is not None:
            hm_emb = self.heatmap_adaptor(heatmap_cond)
            # Share the RGB positional embedding (front-cam slice) so heatmap token i lines
            # up with RGB token i; otherwise use the heatmap's own learned PE.
            hm_pos = self.rgb_pos_embed if self.heatmap_share_rgb_pos_embed else self.heatmap_pos_embed
            hm_emb = hm_emb + hm_pos[:, :hm_emb.shape[1]]
            if self.training and self.target_mask_prob > 0:
                mask = (torch.rand(hm_emb.shape[0], 1, 1, device=hm_emb.device)
                        > self.target_mask_prob).to(hm_emb.dtype)
                hm_emb = hm_emb * mask
            hm_emb = self.heatmap_norm(hm_emb)
            parts.append(hm_emb)
        context_c = torch.cat(parts, dim=1)

        # Every context token is real except T5 padding, which must not be attended to.
        context_mask = None
        if lang_mask is not None:
            context_mask = torch.ones(context_c.shape[:2], dtype=torch.bool, device=context_c.device)
            context_mask[:, lang_start:lang_start + lang_mask.shape[1]] = lang_mask.bool()

        # 5. transformer blocks
        for block in self.blocks:
            x = block(x, time_c, context_c, context_mask=context_mask)

        # 6. head
        x = self.final_layer(x)
        x = x[:, -self.horizon:]

        return x
