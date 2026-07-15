import torch
import torch.nn as nn

from src.tbkv.tbkv_utils import compute_merge, extract_bg_fg_tokens
from src.tbkv.cache import Cache
from src.tbkv.match import perform_tbkv_matching, perform_tbkv_matching_with_tome
from src.tbkv.secondary import make_secondary_policy
from src.core.blocks import Block
from src.core.counting import CountedLinear
from src.reduction.base import ReductionContext


def merging(x, attn_map, k, v, merging_iterations, local_merge_ratio=0.5,
            split_tokens=True, bg_ratio=0.5):
    """Kept for backward compatibility / external callers.

    NOTE: TBKVBlock no longer calls this per-frame during caching. Instead it
    accumulates raw background K/V across the whole caching window and calls
    compute_merge() once at finalize time (see TBKVBlock._finalize_cache).
    This function still performs a single-shot merge over whatever x/k/v it's
    given, so it remains valid as a standalone utility.
    """
    B, N, C = x.shape

    # Either split tokens by saliency, or treat all tokens as cacheable.
    if split_tokens:
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, attn_map, bg_ratio)
    else:
        x_bg = x
        x_fg = x[:, :0, :]
        idx_bg = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

    # x_bg: [B, max_bg_tokens, C], idx_bg: [B, max_bg_tokens]
    max_bg_tokens = x_bg.shape[1]

    # Extract K and V for background tokens
    # k, v have shape: [B, num_heads, N, head_dim]
    num_heads = k.shape[1]
    head_dim = k.shape[-1]

    # Expand idx_bg for gathering: [B, max_bg_tokens] -> [B, num_heads, max_bg_tokens, head_dim]
    idx_expanded = idx_bg.unsqueeze(1).unsqueeze(-1).expand(B, num_heads, max_bg_tokens, head_dim)

    # Gather k and v for background tokens
    k_bg = torch.gather(k, dim=2, index=idx_expanded)  # [B, num_heads, max_bg_tokens, head_dim]
    v_bg = torch.gather(v, dim=2, index=idx_expanded)  # [B, num_heads, max_bg_tokens, head_dim]

    # Reshape for merging: [B, num_heads, max_bg_tokens, head_dim] -> [B, max_bg_tokens, num_heads*head_dim]
    k_bg_flat = k_bg.transpose(1, 2).reshape(B, max_bg_tokens, num_heads * head_dim)
    v_bg_flat = v_bg.transpose(1, 2).reshape(B, max_bg_tokens, num_heads * head_dim)

    # Merge background tokens
    m, u, merged_x_bg = compute_merge(x_bg, merging_iterations, local_merge_ratio)
    k_merged_flat = m(k_bg_flat)  # [B, merged_tokens, num_heads*head_dim]
    v_merged_flat = m(v_bg_flat)  # [B, merged_tokens, num_heads*head_dim]

    # Reshape back: [B, merged_tokens, num_heads*head_dim] -> [B, num_heads, merged_tokens, head_dim]
    merged_tokens = k_merged_flat.shape[1]
    k_merged = k_merged_flat.reshape(B, merged_tokens, num_heads, head_dim).transpose(1, 2)
    v_merged = v_merged_flat.reshape(B, merged_tokens, num_heads, head_dim).transpose(1, 2)

    n_bg_orig = int(x_bg.shape[1])
    n_fg = int(x_fg.shape[1])
    return k_merged, v_merged, merged_x_bg, n_bg_orig, n_fg


class TBKVBlock(Block):

    def __init__(self, local_merge_ratio: float = 0.5, merging_iterations: int = 1, r_match: float = 0.75, bg_ratio: float = 0.5, caching: bool = False, has_class_token: bool = False, raw: bool = False, use_tome: bool = False, tome_r: int = 0, split_tokens: bool = True,
                 kv_reuse_only: bool = False, matching_start_block: int = 0,
                 token_skip: bool = False, secondary: str = "none",
                 secondary_keep: float = 0.5, tbkv_all_blocks: bool = False,
                 block_idx: int = 0,
                 reducer=None, block_index: int = 0, depth: int = 12, **super_kwargs):

        super().__init__(**super_kwargs)

        self.r_match = r_match
        self.bg_ratio = bg_ratio
        self.local_merge_ratio = local_merge_ratio
        self.merging_iterations = merging_iterations
        self.caching = caching
        self._initial_caching = caching
        self.has_class_token = has_class_token
        self.raw = raw
        self.use_tome = use_tome
        self.tome_r = tome_r
        self.split_tokens = split_tokens

        # ViTDet-only TBKV variants. The detection path drives blocks directly
        # (see src/models/tbkv_vitdet.py) and reads these off the block, so they
        # must exist even though the ViViT path leaves them at their defaults.
        self.kv_reuse_only = bool(kv_reuse_only)
        self.matching_start_block = int(matching_start_block)
        self.block_idx = int(block_idx)
        self.token_skip = bool(token_skip)
        self.tbkv_all_blocks = bool(tbkv_all_blocks)
        self.secondary_name = str(secondary)
        self.secondary_keep = float(secondary_keep)
        self.secondary_policy = make_secondary_policy(secondary, keep_ratio=secondary_keep)

        # Shared across the whole stack (AViT accumulates state across depth),
        # so assign rather than register -- registering it in every block would
        # duplicate it in the state dict and break checkpoint loading.
        object.__setattr__(self, "reducer", reducer)
        self.block_index = block_index
        self.depth = depth
        self.cache = None
        self.prev_attn_map = None
        self._kf_tok = None
        self._kf_k = None
        self._kf_v = None
        self._mlp_delta = None

        # Accumulation buffers for cross-frame merging. During the caching
        # window, raw (unmerged) background K/V/tokens for every frame are
        # appended here. Once the block transitions out of caching mode,
        # _finalize_cache() runs compute_merge() ONCE over everything that
        # was accumulated, producing a single bounded-size cache instead of
        # a cache that grows linearly with the number of caching frames.
        self._pending_x_bg = []
        self._pending_x_bg_norm = []
        self._pending_k_bg = []
        self._pending_v_bg = []

        qkv: nn.Linear = self.qkv
        dim = qkv.in_features
        all_head_dim = qkv.out_features // 3
        device = qkv.weight.device
        dtype = qkv.weight.dtype

        w_q, w_k, w_v = qkv.weight.data.chunk(3, dim=0)

        def build_linear(w, b=None):
            lin = CountedLinear(dim, all_head_dim, device=device, dtype=dtype)
            lin.weight.data.copy_(w)
            if b is not None:
                lin.bias.data.copy_(b)
            return lin

        self.q = build_linear(w_q, None)
        self.k = build_linear(w_k, None)
        self.v = build_linear(w_v, None)
        # dim saved for FLOPs accounting
        self._dim = dim
        self._head_dim = all_head_dim

    def _record_frame_stats(self, stats):
        if not hasattr(self, '_frame_stats'):
            self._frame_stats = []
        self._frame_stats.append(stats)

    def _append_cache(self, k_new, v_new, tok_new, tok_new_norm):
        if self.cache is None:
            self.cache = Cache(k_new, v_new, tok_new, tok_new_norm)
        else:
            self.cache = Cache(
                torch.cat([self.cache.K, k_new], dim=2),
                torch.cat([self.cache.V, v_new], dim=2),
                torch.cat([self.cache.tokens, tok_new], dim=1),
                torch.cat([self.cache.tokens_norm, tok_new_norm], dim=1),
            )

    def _cache_size_mb(self):
        return (self.cache.K.numel() * self.cache.K.element_size() * 2) / 1e6

    def _finalize_cache(self):
        """Merge everything accumulated during the caching window into a
        single bounded-size cache. Called exactly once, lazily, on the
        first forward() call after self.caching has become False.
        """
        if not self._pending_x_bg:
            return

        x_bg_all = torch.cat(self._pending_x_bg, dim=1)         # [B, total_bg_tokens, C]
        x_bg_norm_all = torch.cat(self._pending_x_bg_norm, dim=1)
        k_bg_all = torch.cat(self._pending_k_bg, dim=2)   # [B, heads, total_bg_tokens, head_dim]
        v_bg_all = torch.cat(self._pending_v_bg, dim=2)   # [B, heads, total_bg_tokens, head_dim]

        B, total_tokens, C = x_bg_all.shape
        num_heads, head_dim = k_bg_all.shape[1], k_bg_all.shape[-1]

        k_bg_flat = k_bg_all.transpose(1, 2).reshape(B, total_tokens, num_heads * head_dim)
        v_bg_flat = v_bg_all.transpose(1, 2).reshape(B, total_tokens, num_heads * head_dim)

        # Single merge pass over the FULL accumulated set of background
        # tokens across all caching frames -- this is what bounds cache
        # size regardless of how many frames were spent in caching mode.
        # The similarity metric is computed on the normed view; the resulting
        # merge is then applied identically to the unnormed view and to K/V.
        m, u, merged_x_norm = compute_merge(
            x_bg_norm_all, self.merging_iterations, self.local_merge_ratio
        )
        merged_x = m(x_bg_all)
        k_merged_flat = m(k_bg_flat)
        v_merged_flat = m(v_bg_flat)

        merged_tokens = k_merged_flat.shape[1]
        k_merged = k_merged_flat.reshape(B, merged_tokens, num_heads, head_dim).transpose(1, 2)
        v_merged = v_merged_flat.reshape(B, merged_tokens, num_heads, head_dim).transpose(1, 2)

        # Overwrite (not append) -- this replaces any prior cache state.
        self.cache = Cache(k_merged, v_merged, merged_x, merged_x_norm)

        self._record_frame_stats({
            'phase': 'cache_finalize',
            'caching_frames_accumulated': len(self._pending_x_bg),
            'total_bg_tokens_accumulated': int(total_tokens),
            'merged_tokens': int(merged_tokens),
            'cache_kv_size_mb': self._cache_size_mb(),
        })

        self._pending_x_bg = []
        self._pending_x_bg_norm = []
        self._pending_k_bg = []
        self._pending_v_bg = []

    def _split_matching_inputs(self, x, skip_1, prev_attnmap):
        if self.has_class_token:
            cls_x_norm = x[:, :1, :]
            cls_x_unnorm = skip_1[:, :1, :]
            patch_x_norm = x[:, 1:, :]
            patch_x_unnorm = skip_1[:, 1:, :]
            patch_attnmap = prev_attnmap[:, :, 1:, 1:]
        else:
            cls_x_norm = cls_x_unnorm = None
            patch_x_norm = x
            patch_x_unnorm = skip_1
            patch_attnmap = prev_attnmap

        return cls_x_norm, cls_x_unnorm, patch_x_norm, patch_x_unnorm, patch_attnmap

    def _prepend_class_token(self, q, k, v, new_tokens, cls_x_norm, cls_x_unnorm):
        b_size = q.shape[0]
        head_dim = cls_x_norm.shape[-1] // self.heads
        cls_q = self.q(cls_x_norm).reshape(b_size, 1, self.heads, head_dim).permute(0, 2, 1, 3)
        cls_k = self.k(cls_x_norm).reshape(b_size, 1, self.heads, head_dim).permute(0, 2, 1, 3)
        cls_v = self.v(cls_x_norm).reshape(b_size, 1, self.heads, head_dim).permute(0, 2, 1, 3)
        q = torch.cat([cls_q, q], dim=2)
        k = torch.cat([cls_k, k], dim=2)
        v = torch.cat([cls_v, v], dim=2)
        skip_1 = torch.cat([cls_x_unnorm, new_tokens], dim=1)
        return q, k, v, skip_1

    def _forward_matching(self, x, skip_1, prev_attnmap):
        self._match_calls = getattr(self, '_match_calls', 0) + 1
        prev_attnmap = prev_attnmap.to(x.device)

        cls_x_norm, cls_x_unnorm, patch_x_norm, patch_x_unnorm, patch_attnmap = self._split_matching_inputs(
            x, skip_1, prev_attnmap
        )

        if self.split_tokens:
            x_bg, x_fg, _, _ = extract_bg_fg_tokens(patch_x_norm, patch_attnmap, self.bg_ratio)
        else:
            x_bg = patch_x_norm
            x_fg = patch_x_norm[:, :0, :]

        if self.use_tome:
            tome_info = {
                "r": [self.tome_r],
                "class_token": self.has_class_token,
                "distill_token": False,
                "trace_source": False,
                "source": None,
            }
            q, k, v, new_tokens, flop_info = perform_tbkv_matching_with_tome(
                x_unnorm=patch_x_unnorm,
                x=patch_x_norm,
                old_attn=patch_attnmap,
                cache=self.cache,
                q_proj=self.q,
                k_proj=self.k,
                v_proj=self.v,
                num_heads=self.heads,
                r_match=self.r_match,
                bg_ratio=self.bg_ratio,
                split_tokens=self.split_tokens,
                device=x.device,
                _tome_info=tome_info,
            )
        else:
            q, k, v, new_tokens, flop_info = perform_tbkv_matching(
                x_unnorm=patch_x_unnorm,
                x=patch_x_norm,
                old_attn=patch_attnmap,
                cache=self.cache,
                q_proj=self.q,
                k_proj=self.k,
                v_proj=self.v,
                num_heads=self.heads,
                r_match=self.r_match,
                bg_ratio=self.bg_ratio,
                split_tokens=self.split_tokens,
                device=x.device,
            )

        n_unique_cache = int(flop_info['n_matched_tokens'])
        n_bg_matched = int(x_bg.shape[1]) - int(flop_info['n_unmatched_tokens'])
        cache_size = int(self.cache.tokens.shape[1])
        saved_kv_flops = int(n_bg_matched * self._dim * self._head_dim * 2)
        match_overhead_flops = int(x_bg.shape[1] * cache_size * self._dim)
        cache_kv_read_bytes = int(
            n_unique_cache
            * self.heads
            * (self._head_dim // self.heads)
            * self.cache.K.element_size()
            * 2
        )
        self._record_frame_stats({
            'phase': 'matching',
            'n_bg': int(x_bg.shape[1]),
            'n_fg': int(x_fg.shape[1]),
            'n_bg_matched': n_bg_matched,
            'n_bg_unmatched': int(flop_info['n_unmatched_tokens']),
            'n_unique_cache_used': n_unique_cache,
            'cache_size': cache_size,
            'n_q': int(flop_info['n_q_tokens']),
            'n_kv': int(flop_info['n_kv_tokens']),
            'saved_kv_linear_flops': saved_kv_flops,
            'matching_overhead_flops': match_overhead_flops,
            'cache_kv_read_bytes': cache_kv_read_bytes,
        })

        if self.has_class_token:
            q, k, v, skip_1 = self._prepend_class_token(q, k, v, new_tokens, cls_x_norm, cls_x_unnorm)
        else:
            skip_1 = new_tokens

        b_size = q.shape[0]
        n_q = q.shape[2]
        attn = self.matmul(q / self.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        self.prev_attn_map = attn.detach().cpu()

        x = self.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(b_size, n_q, -1)
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # Hand the foreground tokens to the SOTA reducer, if one is configured.
        # TBKV already handles the background via its K/V cache; this composes
        # the two, so each token is reduced by exactly one mechanism.
        n_fg = int(flop_info['n_fg_tokens'])
        op = self._plan_foreground_reduction(x, attn, k, n_fg)
        if op is not None:
            x = torch.cat([x[:, :-n_fg], op(x[:, -n_fg:])], dim=1)
            # The attention map is handed to the next block as prev_attnmap and
            # is what drives its foreground/background split. If the tokens
            # shrink but the map does not, that split indexes a longer map into
            # a shorter token tensor. Reduce the query axis to match.
            attn = self._reduce_attn_rows(attn, op, n_fg)
            self.prev_attn_map = attn.detach().cpu()

        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)

        return x, self.prev_attn_map

    @staticmethod
    def _reduce_attn_rows(attn, op, n_fg):
        """Apply a token operator to the foreground rows of the attention map.

        The operator maps [B, N, C] -> [B, N', C], so the head and key axes are
        folded into the channel axis, reduced, and unfolded. Only the query axis
        changes; key columns are untouched (nothing removes keys here).
        """
        B, H, _, Nk = attn.shape
        fg = attn[:, :, -n_fg:, :].permute(0, 2, 1, 3).reshape(B, n_fg, H * Nk)
        fg = op(fg)
        n_new = fg.shape[1]
        fg = fg.reshape(B, n_new, H, Nk).permute(0, 2, 1, 3)
        return torch.cat([attn[:, :, :-n_fg, :], fg], dim=2)

    def _plan_foreground_reduction(self, x, attn, k, n_fg):
        """Plan the configured TokenReducer over the foreground tokens only.

        Layout matters here. perform_tbkv_matching builds the sequence as
        [matched_cache | unmatched_bg | foreground], and _prepend_class_token
        puts the class token in front of both the queries and the keys. So the
        foreground is exactly the last n_fg entries of the query axis *and* of
        the key axis, which is what lets us hand the reducer a self-contained
        sub-problem: fg tokens, fg-to-fg attention, fg keys.
        """
        if self.reducer is None or n_fg <= 1:
            return None

        cls_attn = (
            attn[:, :, 0, -n_fg:].mean(dim=1) if self.has_class_token else None
        )
        ctx = ReductionContext(
            attn=attn[:, :, -n_fg:, -n_fg:],
            keys=k[:, :, -n_fg:, :],
            cls_attn=cls_attn,
            # The slice we hand over contains no class token, even though the
            # full sequence does; the class token is preserved untouched below.
            has_class_token=False,
            block_index=self.block_index,
            depth=self.depth,
        )

        return self.reducer.plan(x[:, -n_fg:], ctx)

    def reset_self(self):
        super().reset_self()
        if self.reducer is not None and self.block_index == 0:
            self.reducer.reset()
        self.caching = self._initial_caching
        self.cache = None
        self.prev_attn_map = None
        self._kf_tok = None
        self._kf_k = None
        self._kf_v = None
        self._mlp_delta = None
        self._match_calls = 0
        self._frame_stats = []   # list of dicts, one per forward call
        self._pending_x_bg = []
        self._pending_x_bg_norm = []
        self._pending_k_bg = []
        self._pending_v_bg = []

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        qkv_w = prefix + "qkv.weight"
        qkv_b = prefix + "qkv.bias"
        q_w   = prefix + "q.weight"
        q_b   = prefix + "q.bias"

        # Case 1: checkpoint has combined qkv → split into q/k/v for our block
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

        # Case 2: checkpoint has split q/k/v but no qkv → reconstruct qkv
        # (self.qkv is still used in the standard/caching forward path)
        if q_w in state_dict and qkv_w not in state_dict:
            state_dict[qkv_w] = torch.cat(
                [state_dict[prefix + "q.weight"],
                 state_dict[prefix + "k.weight"],
                 state_dict[prefix + "v.weight"]], dim=0)
        if q_b in state_dict and qkv_b not in state_dict:
            state_dict[qkv_b] = torch.cat(
                [state_dict[prefix + "q.bias"],
                 state_dict[prefix + "k.bias"],
                 state_dict[prefix + "v.bias"]], dim=0)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, prev_attnmap=None):
        skip_1 = x
        x = self.input_layer_norm(x)

        # Lazily finalize the accumulated cache exactly once, the first
        # time we're no longer in caching mode but still have pending
        # (unmerged) frames sitting in the accumulation buffers.
        if not self.caching and self.cache is None and self._pending_x_bg:
            self._finalize_cache()

        if not self.caching and prev_attnmap is not None and self.cache is not None:
            return self._forward_matching(x, skip_1, prev_attnmap)

        # Standard path (caching mode or first frame with no prev_attnmap)
        tokens_norm = x
        x = self.qkv(x)
        x, ats_indices = self._forward_attention(
            x, skip_1, prev_attnmap=prev_attnmap, tokens_norm=tokens_norm
        )
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)

        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)

        return x, self.prev_attn_map

    def forward_tbkv_all(self, x):
        """
        TBKV on *all* blocks, including windowed ones.

        Two decoupled reuse mechanisms:
          * **KV-reuse** (TBKV) — per-position temporal matching against the
            keyframe.  A token whose (layer-normed) feature barely changed
            reuses the keyframe's K/V, skipping its K/V projection.  Attention
            still runs over the FULL window token set so window partitioning and
            relative-position embeddings stay exact.
          * **Secondary token skip** (Eventful / MaskVD / STGT) — a pointwise,
            window-independent MLP skip: temporally stable tokens reuse the
            previous frame's MLP delta instead of recomputing it.

        ``self.caching`` (or an empty keyframe cache) marks the keyframe, which
        computes everything and stores the reuse state.  Matching frames reuse.
        """
        skip_1 = x
        x_norm = self.input_layer_norm(x)
        B, N, C = x_norm.shape
        H = self.heads
        hd = self.qkv.out_features // (3 * H)

        xw = self._partition_windows(x_norm, in_qkv_domain=False)   # [BW, T, C]
        BW, T, _ = xw.shape

        q = self.q(xw).view(BW, T, H, hd).permute(0, 2, 1, 3)       # [BW, H, T, hd]

        is_keyframe = self.caching or self._kf_tok is None
        if is_keyframe:
            k = self.k(xw).view(BW, T, H, hd).permute(0, 2, 1, 3)
            v = self.v(xw).view(BW, T, H, hd).permute(0, 2, 1, 3)
            self._kf_tok = xw.detach()
            self._kf_k = k.detach()
            self._kf_v = v.detach()
        else:
            xn = xw / (xw.norm(dim=-1, keepdim=True) + 1e-6)
            cn = self._kf_tok / (self._kf_tok.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (xn * cn).sum(dim=-1)                              # [BW, T]
            n_match = max(0, min(T, int(T * float(self.r_match))))
            order = sim.argsort(dim=-1, descending=True)
            idx_unm = order[:, n_match:]
            k = self._kf_k.clone()
            v = self._kf_v.clone()
            n_unm = idx_unm.shape[1]
            if n_unm > 0:
                xu = xw.gather(1, idx_unm.unsqueeze(-1).expand(-1, -1, C))
                ku = self.k(xu).view(BW, n_unm, H, hd).permute(0, 2, 1, 3)
                vu = self.v(xu).view(BW, n_unm, H, hd).permute(0, 2, 1, 3)
                su = idx_unm.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, hd)
                k = k.scatter(2, su, ku)
                v = v.scatter(2, su, vu)
            self._record_frame_stats({
                'phase': 'matching', 'n_total': int(N),
                'n_bg_matched': int(T - n_unm) * (BW // max(1, B)),
                'n_bg_unmatched': int(n_unm) * (BW // max(1, B)),
                'kvreuse_windowed': True,
            })

        # ── Windowed attention over the full token set (exact rel-pos) ─────────
        attn = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            attn = self.relative_position(attn, q)
        attn = attn.softmax(dim=-1)
        out = self.matmul(attn, v)                                  # [BW, H, T, hd]
        out = self._recombine_heads(out)                           # [BW, T, C]
        out = self._recombine_windows(out)                         # [B, N, C]
        out = self.projection(out)
        s2 = self.add(self.drop_path(out), skip_1)                 # [B, N, C]

        # ── MLP with secondary token skip (pointwise -> window independent) ────
        policy = self.secondary_policy
        can_defer = (not is_keyframe) and policy.active and self._mlp_delta is not None
        if can_defer:
            active_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            sal = torch.zeros((B, N), device=x.device)
            keep_idx, _ = policy.select(
                id(self), s2, sal, active_idx=active_idx, n_grid=N)
            gk = keep_idx.unsqueeze(-1).expand(-1, -1, C)
            s2_keep = s2.gather(1, gk)
            delta_keep = self._forward_mlp(self.mlp_layer_norm(s2_keep))
            new_delta = self._mlp_delta.scatter(1, gk, delta_keep)
            if getattr(policy, "name", "none") == "eventful":
                policy.update(id(self), s2)
        else:
            new_delta = self._forward_mlp(self.mlp_layer_norm(s2))
            if policy.active and getattr(policy, "name", "none") == "eventful":
                policy.update(id(self), s2)

        self._mlp_delta = new_delta.detach()
        x_out = self.add(self.drop_path(new_delta), s2)
        self._prev_output = x_out.detach()
        return x_out, None

    def _forward_attention(self, x, tokens, prev_attnmap=None, tokens_norm=None):
        # (batch, token, dim)
        B, N, C = tokens.shape
        if tokens_norm is None:
            tokens_norm = tokens

        # Partition the windows and attention heads. _window_partition
        # is a noop if self.window_size is None. Windows are arranged
        # along the batch dimension.
        x = self._partition_windows(x, in_qkv_domain=True)
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Token pooling is a noop if self.pool_size is None.
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            x = self.relative_position(x, q)
        x = x.softmax(dim=-1)

        H = self.heads
        head_dim = self.qkv.out_features // (3 * H)

        if self.caching:
            if self.raw:
                # Raw mode never merges -- it caches exact tokens verbatim,
                # so there's nothing to accumulate-then-merge. Kept as an
                # incremental append, same as before.
                if self.has_class_token:
                    k_new = k[:, :, 1:, :]
                    v_new = v[:, :, 1:, :]
                    tok_new = tokens[:, 1:, :]
                    tok_new_norm = tokens_norm[:, 1:, :]
                else:
                    k_new, v_new, tok_new = k, v, tokens
                    tok_new_norm = tokens_norm

                self._append_cache(k_new, v_new, tok_new, tok_new_norm)
                n_cached = int(self.cache.tokens.shape[1])
                n_patch = int(k_new.shape[2])
                self._record_frame_stats({
                    'phase': 'caching',
                    'original_tokens': int(N),
                    'n_bg': n_patch,
                    'n_fg': 0,
                    'merged_tokens': n_patch,
                    'cache_size': n_cached,
                    'cache_kv_size_mb': self._cache_size_mb(),
                })

            else:
                # Non-raw caching: extract this frame's background K/V and
                # STASH them (no merge yet). The actual compute_merge() call
                # happens once, across ALL accumulated caching frames, in
                # _finalize_cache() when the block transitions to matching
                # mode. This is what bounds final cache size instead of
                # letting it grow linearly with the number of caching frames.
                if self.has_class_token:
                    k_patch = k[:, :, 1:, :]
                    v_patch = v[:, :, 1:, :]
                    attn_patch = x[:, :, 1:, 1:]
                    patch_tokens = tokens[:, 1:, :]
                    patch_tokens_norm = tokens_norm[:, 1:, :]
                else:
                    k_patch, v_patch, attn_patch, patch_tokens = k, v, x, tokens
                    patch_tokens_norm = tokens_norm

                if self.split_tokens:
                    x_bg, x_fg, idx_bg, _ = extract_bg_fg_tokens(patch_tokens, attn_patch, self.bg_ratio)
                else:
                    x_bg = patch_tokens
                    x_fg = patch_tokens[:, :0, :]
                    idx_bg = torch.arange(
                        patch_tokens.shape[1], device=x.device
                    ).unsqueeze(0).expand(patch_tokens.shape[0], -1)

                # Same background positions, post-LayerNorm view.
                x_bg_norm = torch.gather(
                    patch_tokens_norm, dim=1,
                    index=idx_bg.unsqueeze(-1).expand(-1, -1, patch_tokens_norm.shape[-1]),
                )

                num_heads_local = k_patch.shape[1]
                head_dim_local = k_patch.shape[-1]
                idx_expanded = idx_bg.unsqueeze(1).unsqueeze(-1).expand(
                    -1, num_heads_local, -1, head_dim_local
                )
                k_bg = torch.gather(k_patch, dim=2, index=idx_expanded)
                v_bg = torch.gather(v_patch, dim=2, index=idx_expanded)

                self._pending_x_bg.append(x_bg)
                self._pending_x_bg_norm.append(x_bg_norm)
                self._pending_k_bg.append(k_bg)
                self._pending_v_bg.append(v_bg)

                self._record_frame_stats({
                    'phase': 'caching',
                    'original_tokens': int(N),
                    'n_bg': int(x_bg.shape[1]),
                    'n_fg': int(x_fg.shape[1]),
                    'pending_frames_accumulated': len(self._pending_x_bg),
                })

        else:
            self.prev_attn_map = x.detach().cpu()

        # Adaptive token sampling is a noop if self.ats_fraction is None.
        x, ats_indices = self._adaptive_token_sampling(x, v)

        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)
        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        # (batch, token, dim)

        return x, ats_indices