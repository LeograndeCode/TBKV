"""OCTABlock — a Transformer block with object-centric temporal K/V caching.

Inherits the base :class:`~src.core.blocks.Block` and, when
``enable_object_cache`` is set, replaces the dense attention with OCTA's
cluster + track + cache path.  When the flag is off the block falls back to the
exact base (dense) forward, so ``enable_object_cache=False`` is bit-for-bit the
standard model.
"""

import torch
import torch.nn as nn

from src.core.blocks import Block
from src.core.counting import CountedLinear
from src.octa.object_cache import (
    ObjectCacheConfig,
    TrackCache,
    perform_object_matching,
)


class OCTABlock(Block):
    def __init__(
        self,
        has_class_token: bool = False,
        enable_object_cache: bool = False,
        cluster_similarity_threshold: float = 0.6,
        offset_window_size: float = 2.0,
        object_match_threshold: float = 0.55,
        object_merge_ratio: float = 0.5,
        track_eviction_patience: int = 4,
        merging_iterations: int = 2,
        caching: bool = False,
        block_idx: int = 0,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)

        self.has_class_token = has_class_token
        self.block_idx = int(block_idx)
        # ``caching`` mirrors the TBKV eval harness (caching pass builds tracks,
        # matching pass reuses them).  Both passes run the same forward; the
        # flag only labels the recorded per-frame stats.
        self.caching = bool(caching)
        self._initial_caching = bool(caching)

        self.config = ObjectCacheConfig(
            cluster_similarity_threshold=cluster_similarity_threshold,
            offset_window_size=offset_window_size,
            object_match_threshold=object_match_threshold,
            object_merge_ratio=object_merge_ratio,
            track_eviction_patience=track_eviction_patience,
            merging_iterations=merging_iterations,
            enable_object_cache=enable_object_cache,
        )

        self.cache = None            # TrackCache (per block)
        self.prev_attn_map = None    # kept for harness-compatibility (unused)
        self._frame_stats = []

        # Split the fused qkv into separate q/k/v so matched clusters can skip
        # the K/V projections (self.qkv is retained for the dense fallback).
        qkv: nn.Linear = self.qkv
        dim = qkv.in_features
        all_head_dim = qkv.out_features // 3
        device = qkv.weight.device
        dtype = qkv.weight.dtype
        w_q, w_k, w_v = qkv.weight.data.chunk(3, dim=0)

        def build_linear(w):
            lin = CountedLinear(dim, all_head_dim, device=device, dtype=dtype)
            lin.weight.data.copy_(w)
            return lin

        self.q = build_linear(w_q)
        self.k = build_linear(w_k)
        self.v = build_linear(w_v)
        self._dim = dim
        self._head_dim = all_head_dim

    # -- housekeeping -------------------------------------------------------
    def reset_self(self):
        super().reset_self()
        self.caching = self._initial_caching
        self.cache = None
        self.prev_attn_map = None
        self._frame_stats = []

    def _record_frame_stats(self, stats):
        self._frame_stats.append(stats)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        qkv_w = prefix + "qkv.weight"
        qkv_b = prefix + "qkv.bias"
        q_w = prefix + "q.weight"
        q_b = prefix + "q.bias"

        # Checkpoint has fused qkv -> split into q/k/v for this block.
        if qkv_w in state_dict and q_w not in state_dict:
            w_q, w_k, w_v = state_dict[qkv_w].chunk(3, dim=0)
            state_dict[prefix + "q.weight"] = w_q
            state_dict[prefix + "k.weight"] = w_k
            state_dict[prefix + "v.weight"] = w_v
        if qkv_b in state_dict and q_b not in state_dict:
            b_q, b_k, b_v = state_dict[qkv_b].chunk(3, dim=0)
            state_dict[prefix + "q.bias"] = b_q
            state_dict[prefix + "k.bias"] = b_k
            state_dict[prefix + "v.bias"] = b_v
        # Checkpoint has split q/k/v -> reconstruct fused qkv for dense path.
        if q_w in state_dict and qkv_w not in state_dict:
            state_dict[qkv_w] = torch.cat(
                [state_dict[prefix + "q.weight"], state_dict[prefix + "k.weight"],
                 state_dict[prefix + "v.weight"]], dim=0)
        if q_b in state_dict and qkv_b not in state_dict:
            state_dict[qkv_b] = torch.cat(
                [state_dict[prefix + "q.bias"], state_dict[prefix + "k.bias"],
                 state_dict[prefix + "v.bias"]], dim=0)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    # -- forward ------------------------------------------------------------
    def forward(self, x, prev_attnmap=None, positions=None):
        if not self.config.enable_object_cache:
            # Exact dense forward (bit-for-bit the base block).
            return super().forward(x), None, positions
        return self._forward_object(x, positions)

    def _forward_object(self, x, positions):
        skip_1 = x
        x_norm = self.input_layer_norm(x)

        if self.has_class_token:
            cls_norm = x_norm[:, :1, :]
            cls_unnorm = skip_1[:, :1, :]
            patch_norm = x_norm[:, 1:, :]
            patch_unnorm = skip_1[:, 1:, :]
        else:
            cls_norm = cls_unnorm = None
            patch_norm = x_norm
            patch_unnorm = skip_1

        B, N, C = patch_norm.shape
        if positions is None:
            idx = torch.arange(N, device=x.device, dtype=patch_norm.dtype)
            positions = torch.stack([idx, torch.zeros_like(idx)], dim=-1)[None].expand(B, -1, -1)

        if self.cache is None or self.cache.batch_size != B:
            self.cache = TrackCache(batch_size=B)

        # Pass the layer-normed tokens (correct scale for the q/k/v projections).
        # Clustering (object_bipartite_matching) and track matching L2-normalize
        # internally for the *similarity* computation, so representatives keep
        # the layer-norm scale the projections expect.
        q, k, v, new_tokens, new_positions, info = perform_object_matching(
            patch_norm, patch_unnorm, positions, None,
            self.cache, self.q, self.k, self.v, self.heads, self.config,
        )

        b = B
        saved = int(info["saved_kv_flops"])
        self._record_frame_stats({
            "phase": "caching" if self.caching else "matching",
            "n_bg": int(info["n_tokens"]),
            "n_fg": 0,
            "merged_tokens": int(info["n_clusters"]),
            "cache_size": int(info["n_tracks_active"]),
            "n_bg_matched": int(info["n_clusters_matched"]),
            "n_bg_unmatched": int(info["n_clusters_fresh"]),
            "n_unique_cache_used": int(info["n_clusters_matched"]),
            "saved_kv_linear_flops": saved,
            "matching_overhead_flops": int(info["match_overhead_flops"]),
            "n_tracks_active": int(info["n_tracks_active"]),
            "n_tracks_evicted": int(info["n_tracks_evicted"]),
            "cache_kv_size_mb": self._cache_kv_mb(),
        })

        if self.has_class_token:
            q, k, v, skip_1 = self._prepend_class_token(
                q, k, v, new_tokens, cls_norm, cls_unnorm)
        else:
            skip_1 = new_tokens

        b_size = q.shape[0]
        n_q = q.shape[2]
        attn = self.matmul(q / self.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = self.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b_size, n_q, -1)
        out = self.projection(out)
        x = self.add(self.drop_path(out), skip_1)

        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)

        return x, None, new_positions

    def _prepend_class_token(self, q, k, v, new_tokens, cls_norm, cls_unnorm):
        b_size = q.shape[0]
        head_dim = cls_norm.shape[-1] // self.heads
        cls_q = self.q(cls_norm).reshape(b_size, 1, self.heads, head_dim).permute(0, 2, 1, 3)
        cls_k = self.k(cls_norm).reshape(b_size, 1, self.heads, head_dim).permute(0, 2, 1, 3)
        cls_v = self.v(cls_norm).reshape(b_size, 1, self.heads, head_dim).permute(0, 2, 1, 3)
        q = torch.cat([cls_q, q], dim=2)
        k = torch.cat([cls_k, k], dim=2)
        v = torch.cat([cls_v, v], dim=2)
        skip_1 = torch.cat([cls_unnorm, new_tokens], dim=1)
        return q, k, v, skip_1

    def _cache_kv_mb(self):
        if self.cache is None:
            return 0.0
        total = 0
        for b in range(self.cache.batch_size):
            for tr in self.cache.tracks[b].values():
                total += tr.K.numel() * tr.K.element_size() * 2
        return total / 1e6
