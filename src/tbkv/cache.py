import torch

class Cache():
    """Holds merged background K/V plus the tokens that produced them.

    Two views of the same merged tokens are kept:
      - ``tokens``      : pre-LayerNorm, used for the residual stream.
      - ``tokens_norm`` : post-LayerNorm, used for cosine matching and for the
                          Q projection, both of which operate on normed tokens
                          everywhere else in the block.
    Mixing the two makes the matching metric compare vectors from different
    distributions and feeds q_proj activations it never sees at train time.
    """

    def __init__(self, K, V, tokens, tokens_norm=None):
        self.K = K
        self.V = V
        self.tokens = tokens
        self.tokens_norm = tokens_norm if tokens_norm is not None else tokens
        self.old_attn = None
        # Post-block output tokens (after attention + MLP), aligned 1:1 with
        # ``tokens`` on the full N grid.  Populated only in token-skip mode so
        # that a matched background token can reuse the entire block output
        # instead of just its cached K/V projection.
        self.outputs = None
        # Previous *matching*-frame post-block output on the full N grid.
        # Used by a secondary policy (Eventful/MaskVD/STGT) to reuse the last
        # frame's result for deferred active tokens.  Updated every matching
        # frame (unlike ``outputs`` which is fixed at the keyframe).
        self.prev_output = None

    def set_old_attn(self, attn):
        self.old_attn = attn


    def match_tokens(self, x: torch.Tensor, r_match: float, x_prenorm: torch.Tensor = None):

        B, N, C = x.shape

        # x is post-LayerNorm, so match against the post-LayerNorm cache view.
        cache_tokens = self.tokens_norm  # [B, N_cache, C]

        a = x / x.norm(dim=-1, keepdim=True)  # [B, N, C]
        b = cache_tokens / cache_tokens.norm(dim=-1, keepdim=True)  # [B, N_cache, C]

        scores = a @ b.transpose(-1, -2)  # [B, N, N_cache]

        r = min(a.shape[1], int(a.shape[1] * r_match))  # number of tokens to match

        node_max, node_idx = scores.max(dim=-1)  # [B, N], [B, N]

        # Sort by score to find top r tokens to match
        edge_idx = node_max.argsort(dim=-1, descending=True)  # [B, N]

        # Split into unmatched (lower score) and matched (higher score)
        unm_idx = edge_idx[:, r:]  # [B, N-r] - indices of unmatched tokens
        mt_idx = edge_idx[:, :r]   # [B, r] - indices of matched tokens

        # Safety check: ensure mt_idx is within bounds of node_idx
        max_node_idx = node_idx.shape[-1] - 1
        if mt_idx.numel() > 0 and mt_idx.max() > max_node_idx:
            print(f"  WARNING: mt_idx {mt_idx.max().item()} exceeds node_idx size {node_idx.shape[-1]}, clamping...")
            mt_idx = torch.clamp(mt_idx, 0, max_node_idx)

        # Get unique cache indices that were matched
        mc_idx = torch.gather(node_idx, dim=-1, index=mt_idx)  # [B, r]

        # Deduplicate per batch item, then pad to uniform length
        unique_per_batch = [torch.unique(mc_idx[b], sorted=False) for b in range(B)]
        max_unique = max(u.shape[0] for u in unique_per_batch)
        # Pad by repeating the last index so gather stays in-bounds
        mc_idx_unique = torch.stack([
            torch.cat([u, u[-1:].expand(max_unique - u.shape[0])]) for u in unique_per_batch
        ], dim=0)  # [B, max_unique]

        # K and V are [B, H, N_cache, head_dim]
        k_matched = torch.gather(self.K, dim=2, index=mc_idx_unique.unsqueeze(1).unsqueeze(-1).expand(-1, self.K.shape[1], -1, self.K.shape[-1]))  # [B, H, max_unique, head_dim]
        v_matched = torch.gather(self.V, dim=2, index=mc_idx_unique.unsqueeze(1).unsqueeze(-1).expand(-1, self.V.shape[1], -1, self.V.shape[-1]))  # [B, H, max_unique, head_dim]

        gather_idx = mc_idx_unique.unsqueeze(-1).expand(-1, -1, C)
        mc = torch.gather(self.tokens, dim=1, index=gather_idx)            # [B, max_unique, C]
        mc_norm = torch.gather(self.tokens_norm, dim=1, index=gather_idx)  # [B, max_unique, C]

        return k_matched, v_matched, mc, mc_norm, mt_idx, unm_idx
