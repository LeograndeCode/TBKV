"""
Token matching logic for TBKV (Token-Based Key-Value caching).
This module contains the matching algorithm that reuses cached K/V tokens.
"""

import torch
from typing import Tuple, Optional
from src.tbkv.tbkv_utils import extract_bg_fg_tokens, get_objective_score
from src.tbkv.merge import bipartite_soft_matching
from src.tbkv.cache import Cache

try:
    from tome.merge import merge_source, merge_wavg
    from tome.utils import parse_r
    _TOME_AVAILABLE = True
except ImportError:
    _TOME_AVAILABLE = False


def mps_gather_workaround(input, dim, index):
    # MPS gather workaround: move to CPU, gather, move back
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim=dim,
            index=index.unsqueeze(-1)
        ).squeeze(-1)
    return torch.gather(input, dim=dim, index=index)


def perform_tbkv_matching(
    x: torch.Tensor,
    x_unnorm: Optional[torch.Tensor],
    old_attn: torch.Tensor,
    cache,
    q_proj,
    k_proj,
    v_proj,
    num_heads: int,
    r_match: float,
    split_tokens: bool,
    device: torch.device,
    bg_ratio: float = 0.5,
    secondary_policy=None,
    block_key=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """
    Perform TBKV matching to reuse cached K/V tokens.
    
    Args:
        x: Normalized input tokens [B, N, C]
        x_unnorm: Unnormalized input tokens for residual [B, N, C]
        old_attn: Attention map from previous layer [B, num_heads, N, N]
        cache: Cache object containing cached K/V
        q_proj: Query projection layer
        k_proj: Key projection layer
        v_proj: Value projection layer
        num_heads: Number of attention heads
        r_match: Matching ratio
        device: Device for computation
        
    Returns:
        q: Query tensor [B, num_heads, N_reduced, head_dim]
        k: Key tensor [B, num_heads, N_kv, head_dim]
        v: Value tensor [B, num_heads, N_kv, head_dim]
        new_tokens: Reduced tokens for residual connection [B, N_reduced, C]
        flop_info: Dictionary with FLOP counting information
    """
    B, N, C = x.shape
    
    gather = mps_gather_workaround if device.type == "mps" else torch.gather
    
    # Either split tokens by saliency, or run TBKV on all tokens.
    if split_tokens:
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, old_attn, bg_ratio=bg_ratio)  # [B, N_bg, C], [B, N_fg, C]
    else:
        x_bg = x
        x_fg = x[:, :0, :]

    # Apply matching algorithm on background tokens only. Eventful temporal
    # gating needs a stable index space across frames, so disable dedup there.
    policy_name = getattr(secondary_policy, "name", "none") if secondary_policy is not None else "none"
    dedup_matches = policy_name != "eventful"
    k_matched, v_matched, matched_cache_tokens, idx_matched, idx_unmatched = cache.match_tokens(
        x_bg, r_match=r_match, deduplicate=dedup_matches) 
    
    # Get unmatched background tokens using indices into x_bg
    unm_bg_tokens = gather(x_bg, dim=-2, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C))

    # Fresh tokens (these get NEW K/V) = unmatched background + foreground.
    fresh_norm = torch.cat([unm_bg_tokens, x_fg], dim=-2)

    # Unnormalized fresh tokens for the residual connection.
    if x_unnorm is not None:
        if split_tokens:
            x_bg_unnorm, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn, bg_ratio=bg_ratio)
        else:
            x_bg_unnorm = x_unnorm
            x_fg_unnorm = x_unnorm[:, :0, :]
        unm_bg_tokens_unnorm = gather(x_bg_unnorm, dim=-2, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C))
        fresh_unnorm = torch.cat([unm_bg_tokens_unnorm, x_fg_unnorm], dim=-2)
    else:
        fresh_unnorm = fresh_norm

    # ── Secondary technique applied on the fresh tokens + their new K/V ────────
    # For spatial/mask policies we do hard pruning (drop deferred tokens).
    # For Eventful we keep token count stable and reuse previous-frame K/V for
    # deferred tokens when available (Eventful-like temporal accumulator).
    n_kept_fresh = fresh_norm.shape[1]
    keep_local = defer_local = fresh_orig = None
    if (secondary_policy is not None and getattr(secondary_policy, "active", False)
            and fresh_norm.shape[1] > 1):
        if split_tokens:
            unm_orig = torch.gather(idx_bg, 1, idx_unmatched)
            fresh_orig = torch.cat([unm_orig, idx_fg], dim=1)
        else:
            fresh_orig = idx_unmatched
        sal = get_objective_score(old_attn).squeeze(-1)  # [B, N] spatial saliency
        fresh_sal = torch.gather(sal, 1, fresh_orig)
        keep_local, defer_local = secondary_policy.select(
            block_key, fresh_norm, fresh_sal, active_idx=fresh_orig, n_grid=N,
        )
        policy_name = getattr(secondary_policy, "name", "none")
        if policy_name != "eventful":
            gather_keep = keep_local.unsqueeze(-1).expand(-1, -1, C)
            fresh_norm = torch.gather(fresh_norm, 1, gather_keep)
            fresh_unnorm = torch.gather(fresh_unnorm, 1, gather_keep)
            n_kept_fresh = fresh_norm.shape[1]
        if policy_name == "eventful":
            secondary_policy.update(block_key, x)

    # Tokens for forward pass: matched cache + kept fresh.
    tokens_forward = torch.cat([matched_cache_tokens, fresh_norm], dim=-2)
    new_tokens = torch.cat([matched_cache_tokens, fresh_unnorm], dim=-2)

    # Compute K/V for fresh tokens. For Eventful, reuse previous-frame K/V on
    # deferred tokens when valid; recompute only keep tokens.
    policy_name = getattr(secondary_policy, "name", "none") if secondary_policy is not None else "none"
    if (policy_name == "eventful" and keep_local is not None and defer_local is not None
            and fresh_orig is not None):
        a = fresh_norm.shape[1]
        head_dim = C // num_heads
        k_new = fresh_norm.new_zeros((B, num_heads, a, head_dim))
        v_new = fresh_norm.new_zeros((B, num_heads, a, head_dim))

        # Recompute K/V for keep tokens.
        n_keep = int(keep_local.shape[1])
        if n_keep > 0:
            gk = keep_local.unsqueeze(-1).expand(-1, -1, C)
            fresh_keep = torch.gather(fresh_norm, 1, gk)
            k_keep = k_proj(fresh_keep).reshape(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            v_keep = v_proj(fresh_keep).reshape(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            sk = keep_local.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
            k_new = k_new.scatter(2, sk, k_keep)
            v_new = v_new.scatter(2, sk, v_keep)

        # Deferred tokens reuse previous-frame K/V if all indices are valid,
        # otherwise fallback to fresh recompute for those deferred tokens.
        n_defer = int(defer_local.shape[1])
        n_fallback = 0
        if n_defer > 0:
            gd = defer_local.unsqueeze(-1).expand(-1, -1, C)
            fresh_defer = torch.gather(fresh_norm, 1, gd)
            defer_orig = torch.gather(fresh_orig, 1, defer_local)
            prev = secondary_policy.get_prev_kv(block_key)
            can_reuse = False
            if prev is not None:
                prev_k, prev_v, prev_valid = prev
                if prev_k.shape[2] == N:
                    valid_defer = torch.gather(prev_valid, 1, defer_orig)
                    can_reuse = bool(valid_defer.all().item())
            if can_reuse:
                idx_prev = defer_orig.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
                k_def = torch.gather(prev_k, 2, idx_prev)
                v_def = torch.gather(prev_v, 2, idx_prev)
            else:
                k_def = k_proj(fresh_defer).reshape(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)
                v_def = v_proj(fresh_defer).reshape(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)
                n_fallback = n_defer
            sd = defer_local.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
            k_new = k_new.scatter(2, sd, k_def)
            v_new = v_new.scatter(2, sd, v_def)

        # Save current-frame fresh K/V in full-grid form for next frame reuse.
        secondary_policy.update_kv(block_key, fresh_orig, k_new, v_new, n_grid=N)
        n_kept_fresh = n_keep + n_fallback
    else:
        tokens_for_kv = fresh_norm
        k_new = k_proj(tokens_for_kv).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)
        v_new = v_proj(tokens_for_kv).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)

    # Compute Q on ALL forward tokens (matched cache + kept fresh)
    q = q_proj(tokens_forward).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)

    # Concatenate cached K/V with newly computed K/V
    k = torch.cat([k_matched, k_new], dim=-2)
    v = torch.cat([v_matched, v_new], dim=-2)
    
    # Collect FLOP counting information
    flop_info = {
        'n_bg_tokens': x_bg.shape[1],
        'n_matched_tokens': matched_cache_tokens.shape[1],
        'n_unmatched_tokens': idx_unmatched.shape[1],
        'n_fg_tokens': x_fg.shape[1],
        'n_q_tokens': q.shape[2],
        'n_kv_tokens': n_kept_fresh,  # kept fresh tokens (unmatched bg + fg, gated)
        'n_kept_fresh': n_kept_fresh,
    }
    
    return q, k, v, new_tokens, flop_info


def perform_tbkv_matching_with_tome(
    x: torch.Tensor,
    x_unnorm: Optional[torch.Tensor],
    old_attn: torch.Tensor,
    cache,
    q_proj,
    k_proj,
    v_proj,
    num_heads: int,
    r_match: float,
    split_tokens: bool,
    device: torch.device,
    _tome_info: dict,
    bg_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """
    Perform TBKV matching with ToMe on foreground tokens.
    
    Args:
        x: Normalized input tokens [B, N, C]
        x_unnorm: Unnormalized input tokens for residual [B, N, C]
        old_attn: Attention map from previous layer [B, num_heads, N, N]
        cache: Cache object containing cached K/V
        q_proj: Query projection layer
        k_proj: Key projection layer
        v_proj: Value projection layer
        num_heads: Number of attention heads
        r_match: Matching ratio
        device: Device for computation
        _tome_info: Dictionary containing ToMe configuration
        
    Returns:
        q: Query tensor [B, num_heads, N_reduced, head_dim]
        k: Key tensor [B, num_heads, N_kv, head_dim]
        v: Value tensor [B, num_heads, N_kv, head_dim]
        new_tokens: Reduced tokens for residual connection [B, N_reduced, C]
        flop_info: Dictionary with FLOP counting information
    """
    B, N, C = x.shape
    
    gather = mps_gather_workaround if device.type == "mps" else torch.gather
    
    # Either split tokens by saliency, or run TBKV on all tokens.
    if split_tokens:
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, old_attn, bg_ratio=bg_ratio)  # [B, N_bg, C], [B, N_fg, C]
    else:
        x_bg = x
        x_fg = x[:, :0, :]

    # Apply ToMe on foreground tokens to further reduce them
    # Compute K projection for foreground tokens to use as metric (similar to tome/patch/timm.py)
    r = _tome_info["r"].pop(0)
    x_fg_reduced = x_fg
    x_fg_unnorm_reduced = None
    
    if r > 0:
        # Compute K for foreground tokens and use mean over heads as metric
        k_fg_for_metric = k_proj(x_fg).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)
        metric = k_fg_for_metric.mean(1)  # [B, N_fg, head_dim]
        
        # Apply ToMe using the metric (similar to ToMeAttention in tome/patch/timm.py)
        merge, _ = bipartite_soft_matching(
            metric,
            r,
            _tome_info["class_token"],
            _tome_info["distill_token"],
        )
        if _tome_info["trace_source"]:
            _tome_info["source"] = merge_source(
                merge, x_fg, _tome_info["source"]
            )
        # Use local size tracking for foreground tokens only (not shared across blocks)
        x_fg_reduced, _ = merge_wavg(merge, x_fg, size=None)
        
        # Also apply the same merging to unnormalized foreground tokens if available
        if x_unnorm is not None:
            if split_tokens:
                _, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn, bg_ratio=bg_ratio)
            else:
                x_fg_unnorm = x_unnorm[:, :0, :]
            x_fg_unnorm_reduced, _ = merge_wavg(merge, x_fg_unnorm, size=None)
        else:
            x_fg_unnorm_reduced = x_fg_reduced
    else:
        # No ToMe reduction - use original foreground tokens
        if x_unnorm is not None:
            if split_tokens:
                _, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn, bg_ratio=bg_ratio)
            else:
                x_fg_unnorm = x_unnorm[:, :0, :]
            x_fg_unnorm_reduced = x_fg_unnorm
        else:
            x_fg_unnorm_reduced = x_fg_reduced

    # Apply matching algorithm on background tokens only
    k_matched, v_matched, matched_cache_tokens, idx_matched, idx_unmatched = cache.match_tokens(
        x_bg, r_match=r_match) 
    
    # Get unmatched background tokens using indices into x_bg
    unm_bg_tokens = gather(x_bg, dim=-2, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C))

    # Tokens for forward pass: matched cache + unmatched bg + reduced fg (after ToMe)
    tokens_forward = torch.cat([matched_cache_tokens, unm_bg_tokens, x_fg_reduced], dim=-2)



    # For residual connection, we need unnormalized version
    # Apply same merging to unnormalized tokens to maintain consistency
    if x_unnorm is not None:
        if split_tokens:
            x_bg_unnorm, _, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn, bg_ratio=bg_ratio)
        else:
            x_bg_unnorm = x_unnorm
        unm_bg_tokens_unnorm = gather(x_bg_unnorm, dim=-2, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C))
        # Use the reduced unnormalized foreground tokens (same merging applied)
        new_tokens = torch.cat([matched_cache_tokens, unm_bg_tokens_unnorm, x_fg_unnorm_reduced], dim=-2)
    else:
        new_tokens = tokens_forward

    # Compute K and V only on newly computed tokens (unmatched bg + reduced fg), NOT on matched cache
    tokens_for_kv = torch.cat([unm_bg_tokens, x_fg_reduced], dim=-2)
    k_new = k_proj(tokens_for_kv).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)
    v_new = v_proj(tokens_for_kv).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)

    # Compute Q on ALL tokens (matched cache + unmatched bg + reduced fg)
    q = q_proj(tokens_forward).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)

    # Concatenate cached K/V with newly computed K/V
    k = torch.cat([k_matched, k_new], dim=-2)
    v = torch.cat([v_matched, v_new], dim=-2)
    
    # Collect FLOP counting information
    flop_info = {
        'n_bg_tokens': x_bg.shape[1],
        'n_matched_tokens': matched_cache_tokens.shape[1],
        'n_unmatched_tokens': idx_unmatched.shape[1],
        'n_fg_tokens': x_fg_reduced.shape[1],  # Use reduced count after ToMe
        'n_q_tokens': q.shape[2],
        'n_kv_tokens': idx_unmatched.shape[1] + x_fg_reduced.shape[1]  # unmatched bg + reduced fg
    }
    
    return q, k, v, new_tokens, flop_info
