
import torch
import torch.nn as nn

from src.tbkv.tbkv_utils import compute_merge, extract_bg_fg_tokens
from src.tbkv.cache import Cache
from src.tbkv.match import perform_tbkv_matching, perform_tbkv_matching_with_tome
from src.tbkv.secondary import make_secondary_policy
from src.core.blocks import Block
from src.core.counting import CountedLinear


def merging(
    x,
    attn_map,
    k,
    v,
    merging_iterations,
    local_merge_ratio=0.5,
    split_tokens=True,
    bg_ratio=0.5,
):
    B, N, C = x.shape
    
    # Either split tokens by saliency, or treat all tokens as cacheable.
    if split_tokens:
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, attn_map, bg_ratio=bg_ratio)
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
    m, u, merged_x_bg = compute_merge(
        x_bg,
        merging_iterations=merging_iterations,
        local_merge_ratio=local_merge_ratio,
    )
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

    def __init__(self, local_merge_ratio: float = 0.5, merging_iterations: int = 1, r_match: float = 0.75, bg_ratio: float = 0.5, caching: bool = False, has_class_token: bool = False, raw: bool = False, use_tome: bool = False, tome_r: int = 0, split_tokens: bool = True, kv_reuse_only: bool = False, matching_start_block: int = 0, token_skip: bool = False, secondary: str = "none", secondary_keep: float = 0.5, tbkv_all_blocks: bool = False, block_idx: int = 0, **super_kwargs):

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
        self.kv_reuse_only = kv_reuse_only
        self.matching_start_block = int(matching_start_block)
        self.block_idx = int(block_idx)
        self.token_skip = bool(token_skip)
        # Secondary token-reduction policy applied AFTER TBKV matching
        # (Eventful / MaskVD / STGT).  "none" == pure TBKV.
        self.secondary_name = str(secondary)
        self.secondary_keep = float(secondary_keep)
        self.secondary_policy = make_secondary_policy(secondary, keep_ratio=secondary_keep)
        # TBKV-on-all-blocks (incl. windowed): per-position KV-reuse keyframe
        # state + secondary MLP-skip delta cache.
        self.tbkv_all_blocks = bool(tbkv_all_blocks)
        self._kf_tok = None
        self._kf_k = None
        self._kf_v = None
        self._mlp_delta = None
        self._prev_output = None
        self.cache = None  
        self.prev_attn_map = None

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

    def _append_cache(self, k_new, v_new, tok_new):
        if self.cache is None:
            self.cache = Cache(k_new, v_new, tok_new)
        else:
            self.cache = Cache(
                torch.cat([self.cache.K, k_new], dim=2),
                torch.cat([self.cache.V, v_new], dim=2),
                torch.cat([self.cache.tokens, tok_new], dim=1),
            )

    def _cache_size_mb(self):
        return (self.cache.K.numel() * self.cache.K.element_size() * 2) / 1e6

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
                split_tokens=self.split_tokens,
                device=x.device,
                _tome_info=tome_info,
                bg_ratio=self.bg_ratio,
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
                split_tokens=self.split_tokens,
                device=x.device,
                bg_ratio=self.bg_ratio,
                secondary_policy=self.secondary_policy,
                block_key=id(self),
            )

        # Batch size for correct FLOPs accounting:
        # CountedLinear counts x.numel() * out_features which includes B, so saved
        # FLOPs must also be multiplied by the batch size.
        b = patch_x_norm.shape[0]
        n_bg = flop_info['n_bg_tokens']
        n_fg = flop_info['n_fg_tokens']
        n_unique_cache = int(flop_info['n_matched_tokens'])
        n_bg_matched = n_bg - int(flop_info['n_unmatched_tokens'])
        cache_size = int(self.cache.tokens.shape[1])
        saved_kv_flops = int(b * n_bg_matched * self._dim * self._head_dim * 2)
        match_overhead_flops = int(b * n_bg * cache_size * self._dim)
        cache_kv_read_bytes = int(
            n_unique_cache
            * self.heads
            * (self._head_dim // self.heads)
            * self.cache.K.element_size()
            * 2
        )
        self._record_frame_stats({
            'phase': 'matching',
            'n_bg': n_bg,
            'n_fg': n_fg,
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

        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)

        return x, self.prev_attn_map

    def reset_self(self):
        super().reset_self()
        self.caching = self._initial_caching
        self.cache = None
        self.prev_attn_map = None
        self._kf_tok = None
        self._kf_k = None
        self._kf_v = None
        self._mlp_delta = None
        self._prev_output = None
        self._match_calls = 0
        self._frame_stats = []   # list of dicts, one per forward call
        if getattr(self, "secondary_policy", None) is not None:
            self.secondary_policy.reset()

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
        if self.tbkv_all_blocks:
            return self.forward_tbkv_all(x)

        skip_1 = x
        x = self.input_layer_norm(x)
        x_norm = x  # save normalized tokens for cache (used in _forward_attention)

        # For block 0 (or whenever cross-block attn is unavailable), use this
        # block's previous-frame attention as the saliency source.
        attn_source = prev_attnmap
        if attn_source is None:
            attn_source = self.prev_attn_map

        start_ok = self.block_idx >= self.matching_start_block

        can_match = (
            not self.caching
            and start_ok
            and attn_source is not None
            and self.cache is not None
            and attn_source.ndim == 4
            and attn_source.shape[0] == x.shape[0]
            and attn_source.shape[-1] == x.shape[1]
            and attn_source.shape[-2] == x.shape[1]
        )
        if can_match:
            return self._forward_matching(x, skip_1, attn_source)

        # Standard path (caching mode or first frame with no prev_attnmap)
        x = self.qkv(x)
        # Pass x_norm so _forward_attention stores correct feature vectors in the
        # cache (not V values), making cosine-similarity matching meaningful.
        x, ats_indices = self._forward_attention(x, x_norm, prev_attnmap=attn_source)
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

    def _forward_attention(self, x, tokens, prev_attnmap=None):
        # (batch, token, dim)
        B, N, C = tokens.shape


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
                if self.has_class_token:
                    k_new = k[:, :, 1:, :]
                    v_new = v[:, :, 1:, :]
                    tok_new = tokens[:, 1:, :]
                else:
                    k_new, v_new, tok_new = k, v, tokens

                self._append_cache(k_new, v_new, tok_new)
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
                if self.has_class_token:
                    B_merge = v.shape[0]
                    N_patch = v.shape[2] - 1
                    k_patch = k[:, :, 1:, :]
                    v_patch = v[:, :, 1:, :]
                    attn_patch = x[:, :, 1:, 1:]
                    x_for_merge = v_patch.transpose(1, 2).reshape(B_merge, N_patch, H * head_dim)
                    k_new, v_new, tok_new, n_bg, n_fg = merging(
                        x=x_for_merge,
                        attn_map=attn_patch,
                        k=k_patch,
                        v=v_patch,
                        merging_iterations=self.merging_iterations,
                        local_merge_ratio=self.local_merge_ratio,
                        split_tokens=self.split_tokens,
                    )
                else:
                    B_merge = v.shape[0]
                    n_tokens = v.shape[2]
                    # FIX: use actual input features (tokens) not V values.
                    # tokens is now x_norm passed from forward(); for windowed
                    # blocks partition it the same way as the qkv output.
                    if self.window_size is not None:
                        tokens_for_merge = self._partition_windows(
                            tokens, in_qkv_domain=False
                        )
                    else:
                        tokens_for_merge = tokens
                    k_new, v_new, tok_new, n_bg, n_fg = merging(
                        x=tokens_for_merge,
                        attn_map=x,
                        k=k,
                        v=v,
                        merging_iterations=self.merging_iterations,
                        local_merge_ratio=self.local_merge_ratio,
                        split_tokens=self.split_tokens,
                    )

                self._append_cache(k_new, v_new, tok_new)
                n_cached = int(self.cache.tokens.shape[1])
                self._record_frame_stats({
                    'phase': 'caching',
                    'original_tokens': int(N),
                    'n_bg': n_bg,
                    'n_fg': n_fg,
                    'merged_tokens': int(tok_new.shape[1]),
                    'cache_size': n_cached,
                    'cache_kv_size_mb': self._cache_size_mb(),
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

