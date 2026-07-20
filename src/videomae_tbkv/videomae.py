"""
VideoMAE ViT (fine-tune variant) with integrated TBKV caching/matching and
src.core.counting FLOP instrumentation.

Architecture follows the official VideoMAE repo (modeling_finetune.py):
- 3D tubelet patch embedding (tubelet_size=2, 16x16 patches):
  16 frames @ 224x224 -> 8*14*14 = 1568 tokens.
- Joint space-time attention, qkv fused weight with separate q_bias/v_bias
  (k bias fixed at zero).
- Fixed sinusoidal position embedding, mean-pooling head with fc_norm.

TBKV operates at clip granularity: a caching clip runs at full compute and
stores per-block background tokens with their (aligned) K/V projections;
subsequent clips run in matching mode, where background tokens that match
the cache reuse the cached K/V instead of recomputing the projections.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.core.base import ExtendedModule
from src.core.counting import (
    CountedAdd,
    CountedBias,
    CountedConv,
    CountedLinear,
    CountedMatmul,
)

from .cache import Cache
from .patch import extract_bg_fg_tokens
from .tbkv_utils import mps_gather_workaround


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table (from the official VideoMAE repo)."""
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    hid = torch.arange(d_hid, dtype=torch.float32).unsqueeze(0)
    angle = position / torch.pow(10000.0, 2 * (hid // 2) / d_hid)
    table = torch.zeros(n_position, d_hid)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.unsqueeze(0)  # [1, N, C]


class PatchEmbed3D(nn.Module):
    """Tubelet embedding: Conv3d with kernel/stride (tubelet, 16, 16)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=16, tubelet_size=2):
        super().__init__()
        self.num_patches = (
            (num_frames // tubelet_size) * (img_size // patch_size) ** 2
        )
        self.conv = CountedConv(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.bias = CountedBias(embed_dim, spatial_dims=3)

    def forward(self, x):
        # x: [B, C, T, H, W] -> [B, N, embed_dim]
        x = self.bias(self.conv(x))
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = CountedLinear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = CountedLinear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TBKVAttention(nn.Module):
    """
    VideoMAE attention with TBKV caching/matching. Same matching algorithm
    as the ported per-frame patch (see patch.py), applied to clip tokens.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = CountedLinear(dim, dim)
        self.k = CountedLinear(dim, dim)  # k bias stays zero (VideoMAE)
        self.v = CountedLinear(dim, dim)
        self.proj = CountedLinear(dim, dim)
        self.matmul = CountedMatmul()

    def forward(
        self, x: torch.Tensor, cache: Optional[Cache], x_unnorm: torch.Tensor,
        state: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # Returns (x_attn, k, v, new_tokens, attn_map)
        B, N, C = x.shape

        old_attn = None if state["caching"] else state.get("prev_attn_map")

        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

        new_tokens = None

        if cache is not None and old_attn is not None:
            x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, old_attn)
            should_match = x_bg.shape[1] > 0
            if should_match:
                x_bg_unnorm, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn)
        else:
            should_match = False

        if should_match:
            k_matched, v_matched, matched_cache_tokens, idx_matched, idx_unmatched = \
                cache.match_tokens(x_bg, r_match=state["r_match"], matmul=self.matmul)

            if state.get("verbose"):
                num_matched = (
                    idx_matched.shape[1] if idx_matched.dim() > 1 else idx_matched.shape[0]
                )
                print(
                    f"  Layer stats: Total={N}, BG={x_bg.shape[1]}, "
                    f"Matched={num_matched}, Cache={cache.tokens.shape[1]}"
                )

            unm_x_bg = gather(
                x_bg, dim=1, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C)
            )
            tokens_forward = torch.cat([matched_cache_tokens, unm_x_bg, x_fg], dim=1)

            unm_x_bg_unnorm = gather(
                x_bg_unnorm, dim=1,
                index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C),
            )
            new_tokens = torch.cat(
                [matched_cache_tokens, unm_x_bg_unnorm, x_fg_unnorm], dim=1
            )

            # Q over all forwarded tokens; K/V only for unmatched + fg tokens.
            q = self.q(tokens_forward).reshape(
                1, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)
            tokens_for_kv = torch.cat([unm_x_bg, x_fg], dim=1)
            k_unm = self.k(tokens_for_kv).reshape(
                1, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)
            v_unm = self.v(tokens_for_kv).reshape(
                1, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)

            k = torch.cat([k_matched, k_unm], dim=2)
            v = torch.cat([v_matched, v_unm], dim=2)

            # K/V projection FLOPs avoided via cache reuse (reported
            # separately, excluded from totals -- src/tbkv convention).
            if self.matmul.count_mode:
                num_matched_bg = (
                    idx_matched.shape[1] if idx_matched.dim() > 1 else idx_matched.shape[0]
                )
                self.matmul.counts["saved_kv_linear_flops"] += (
                    2 * B * num_matched_bg * C * C
                )
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self.matmul(q * self.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn_map = attn

        N_q = q.shape[2]
        x = self.matmul(attn, v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)

        return x, k, v, new_tokens, attn_map


class TBKVBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, state):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TBKVAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.add = CountedAdd()
        self.cache = None
        # Shared (not registered) mutable run state owned by the model.
        object.__setattr__(self, "state", state)

    def forward(self, x):
        state = self.state
        norm_x = self.norm1(x)
        x_attn, k, v, new_tokens, attn_map = self.attn(
            norm_x, self.cache, x_unnorm=x, state=state
        )

        if new_tokens is not None and new_tokens.shape[1] != x.shape[1]:
            # Tokens were reduced via matching; residual over the reduced set.
            x = self.add(new_tokens, x_attn)
        else:
            x = self.add(x, x_attn)

        if state["caching"]:
            # Cache background tokens with K/V rows aligned to them.
            x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, attn_map)
            self.cache = Cache(
                k[:, :, idx_bg, :],
                v[:, :, idx_bg, :],
                x_bg,
                layer=self,
                verbose=state.get("verbose", False),
            )
        else:
            state["prev_attn_map"] = attn_map

        x = self.add(x, self.mlp(self.norm2(x)))
        return x


class VideoMAETBKV(nn.Module):
    """
    VideoMAE ViT-B/16 (Kinetics-400 fine-tune) with TBKV.

    Modes (set via set_caching / used per clip):
    - caching=True: full compute; each block stores its background-token cache.
    - caching=False with caches present: matching mode (TBKV acceleration).
    - caching=False with caches cleared: plain baseline forward.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=400,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_frames=16,
        tubelet_size=2,
        r_match=0.6,
        verbose=False,
    ):
        super().__init__()
        self.state = {
            "caching": False,
            "r_match": r_match,
            "verbose": verbose,
            "prev_attn_map": None,
        }
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_chans, embed_dim, num_frames, tubelet_size
        )
        self.register_buffer(
            "pos_embed",
            get_sinusoid_encoding_table(self.patch_embed.num_patches, embed_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [TBKVBlock(embed_dim, num_heads, mlp_ratio, self.state) for _ in range(depth)]
        )
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = CountedLinear(embed_dim, num_classes)

    # ---- TBKV control ----

    def set_caching(self, caching: bool):
        self.state["caching"] = caching

    def reset_caches(self):
        for block in self.blocks:
            block.cache = None
        self.state["prev_attn_map"] = None

    # ---- Counting API (mirrors src.core.base.ExtendedModule) ----

    def extended_modules(self):
        return [m for m in self.modules() if isinstance(m, ExtendedModule)]

    def counting(self, mode=True):
        for module in self.extended_modules():
            module.count_mode = mode

    def no_counting(self):
        self.counting(mode=False)

    def clear_counts(self):
        for module in self.extended_modules():
            module.counts.clear()

    def total_counts(self):
        return sum(m.counts for m in self.extended_modules())

    # ---- Forward ----

    def forward(self, x):
        # x: [B, 3, T, H, W]
        self.state["prev_attn_map"] = None
        x = self.patch_embed(x)
        # Token order is temporal-major, so a clip shorter than num_frames
        # (e.g. a few caching frames) takes the leading positions.
        x = x + self.pos_embed[:, : x.shape[1]].type_as(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_norm(x.mean(dim=1))
        return self.head(x)


def load_videomae_checkpoint(model: VideoMAETBKV, path):
    """
    Load an official VideoMAE fine-tuned checkpoint ({'module': state_dict},
    fused qkv weight + separate q_bias/v_bias) into VideoMAETBKV.
    """
    checkpoint = torch.load(str(path), map_location="cpu")
    sd = checkpoint.get("module", checkpoint.get("model", checkpoint))

    new_sd = {}
    dim = model.head.in_features
    for key, value in sd.items():
        value = value.float()
        if key == "patch_embed.proj.weight":
            new_sd["patch_embed.conv.weight"] = value
        elif key == "patch_embed.proj.bias":
            new_sd["patch_embed.bias.bias"] = value
        elif key.endswith("attn.qkv.weight"):
            prefix = key[: -len("qkv.weight")]
            w_q, w_k, w_v = value.chunk(3, dim=0)
            new_sd[prefix + "q.weight"] = w_q
            new_sd[prefix + "k.weight"] = w_k
            new_sd[prefix + "v.weight"] = w_v
            # k bias is fixed at zero in VideoMAE
            new_sd[prefix + "k.bias"] = torch.zeros(dim)
        elif key.endswith("attn.q_bias"):
            new_sd[key.replace("q_bias", "q.bias")] = value
        elif key.endswith("attn.v_bias"):
            new_sd[key.replace("v_bias", "v.bias")] = value
        else:
            new_sd[key] = value

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # pos_embed is a non-persistent buffer; nothing else should be missing.
    problems = [k for k in missing if k != "pos_embed"]
    if problems or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing: {problems}, unexpected: {unexpected}"
        )
    return model


def build_videomae_tbkv(
    weights_path=None, device="cuda", r_match=0.6, verbose=False,
    model_size="vit_b",
):
    from pathlib import Path

    from src.utils.config import load_config

    config = load_config(
        Path("configs", "models", f"videomae_{model_size}.yml")
    )
    model_config = config["model"]
    if weights_path is None:
        weights_path = Path(config["weights"])
    model = VideoMAETBKV(
        img_size=model_config["input_shape"][2],
        patch_size=model_config["tubelet_shape"][1],
        num_classes=model_config["classes"],
        embed_dim=model_config["embed_dim"],
        depth=model_config["depth"],
        num_heads=model_config["num_heads"],
        mlp_ratio=model_config["mlp_ratio"],
        num_frames=model_config["input_shape"][0],
        tubelet_size=model_config["tubelet_shape"][0],
        r_match=r_match,
        verbose=verbose,
    )
    load_videomae_checkpoint(model, weights_path)
    model = model.to(device)
    model.eval()
    return model
