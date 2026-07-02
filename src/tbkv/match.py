"""
Token matching logic for TBKV (Token-Based Key-Value caching).
This module contains the matching algorithm that reuses cached K/V tokens.
"""

import torch
from typing import Tuple, Optional
from src.tbkv.tbkv_utils import extract_bg_fg_tokens
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
    device: torch.device
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
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, old_attn)  # [B, N_bg, C], [B, N_fg, C]
    else:
        x_bg = x
        x_fg = x[:, :0, :]

    # Apply matching algorithm on background tokens only
    k_matched, v_matched, matched_cache_tokens, idx_matched, idx_unmatched = cache.match_tokens(
        x_bg, r_match=r_match) 
    
    # Get unmatched background tokens using indices into x_bg
    unm_bg_tokens = gather(x_bg, dim=-2, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C))

    # Tokens for forward pass: matched cache + unmatched bg + fg
    tokens_forward = torch.cat([matched_cache_tokens, unm_bg_tokens, x_fg], dim=-2)

    # For residual connection, we need unnormalized version
    if x_unnorm is not None:
        if split_tokens:
            x_bg_unnorm, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn)
        else:
            x_bg_unnorm = x_unnorm
            x_fg_unnorm = x_unnorm[:, :0, :]
        unm_bg_tokens_unnorm = gather(x_bg_unnorm, dim=-2, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C))
        new_tokens = torch.cat([matched_cache_tokens, unm_bg_tokens_unnorm, x_fg_unnorm], dim=-2)
    else:
        new_tokens = tokens_forward

    # Compute K and V only on newly computed tokens (unmatched bg + fg), NOT on matched cache
    tokens_for_kv = torch.cat([unm_bg_tokens, x_fg], dim=-2)
    k_new = k_proj(tokens_for_kv).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)
    v_new = v_proj(tokens_for_kv).reshape(B, -1, num_heads, C // num_heads).permute(0, 2, 1, 3)

    # Compute Q on ALL tokens (matched cache + unmatched bg + fg)
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
        'n_kv_tokens': idx_unmatched.shape[1] + x_fg.shape[1]  # unmatched bg + fg
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
        x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, old_attn)  # [B, N_bg, C], [B, N_fg, C]
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
                _, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn)
            else:
                x_fg_unnorm = x_unnorm[:, :0, :]
            x_fg_unnorm_reduced, _ = merge_wavg(merge, x_fg_unnorm, size=None)
        else:
            x_fg_unnorm_reduced = x_fg_reduced
    else:
        # No ToMe reduction - use original foreground tokens
        if x_unnorm is not None:
            if split_tokens:
                _, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn)
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
            x_bg_unnorm, _, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn)
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
