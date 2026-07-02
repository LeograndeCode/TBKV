
import torch
import torch.nn as nn

from src.tbkv.tbkv_utils import compute_merge, extract_bg_fg_tokens
from src.tbkv.cache import Cache
from src.tbkv.match import perform_tbkv_matching, perform_tbkv_matching_with_tome
from src.core.blocks import Block
from src.core.counting import CountedLinear


def merging(x, attn_map, k, v, merging_iterations, split_tokens=True):
    B, N, C = x.shape
    
    # Either split tokens by saliency, or treat all tokens as cacheable.
    if split_tokens:
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, attn_map)
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
    m, u, merged_x_bg = compute_merge(x_bg, merging_iterations)
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

    def __init__(self, local_merge_ratio: float = 0.5, merging_iterations: int = 1, r_match: float = 0.75, caching: bool = False, has_class_token: bool = False, raw: bool = False, use_tome: bool = False, tome_r: int = 0, split_tokens: bool = True, **super_kwargs):

        super().__init__(**super_kwargs)

        self.r_match = r_match
        self.local_merge_ratio = local_merge_ratio
        self.merging_iterations = merging_iterations
        self.caching = caching
        self._initial_caching = caching
        self.has_class_token = has_class_token
        self.raw = raw
        self.use_tome = use_tome
        self.tome_r = tome_r
        self.split_tokens = split_tokens
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

        if self.split_tokens:
            x_bg, x_fg, _, _ = extract_bg_fg_tokens(patch_x_norm, patch_attnmap)
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
        self._match_calls = 0
        self._frame_stats = []   # list of dicts, one per forward call

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

        if not self.caching and prev_attnmap is not None and self.cache is not None:
            return self._forward_matching(x, skip_1, prev_attnmap)

        # Standard path (caching mode or first frame with no prev_attnmap)
        x = self.qkv(x)
        x, ats_indices = self._forward_attention(x, skip_1, prev_attnmap=prev_attnmap)
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)

        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)

        return x, self.prev_attn_map

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
                        split_tokens=self.split_tokens,
                    )
                else:
                    B_merge = v.shape[0]
                    n_tokens = v.shape[2]
                    k_new, v_new, tok_new, n_bg, n_fg = merging(
                        x=v.transpose(1, 2).reshape(B_merge, n_tokens, H * head_dim),
                        attn_map=x,
                        k=k,
                        v=v,
                        merging_iterations=self.merging_iterations,
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

