import torch

from .tbkv_utils import mps_gather_workaround


class Cache:
    """
    Per-block TBKV cache holding the (background) tokens of the last caching
    frame together with their key/value projections. K/V rows are stored
    aligned with self.tokens, i.e. row i of K/V belongs to token i.
    """

    def __init__(self, K, V, tokens, layer=None, verbose=False):
        self.layer = layer
        self.K = K  # [1, H, N_cache, head_dim]
        self.V = V  # [1, H, N_cache, head_dim]
        self.tokens = tokens  # [1, N_cache, C]
        self.old_attn = None
        self.verbose = verbose

    def set_old_attn(self, attn):
        self.old_attn = attn

    def match_tokens(self, x: torch.Tensor, r_match: float, matmul=None):
        """
        Match input (background) tokens against the cached tokens by cosine
        similarity. The top r_match fraction of input tokens is considered
        "matched" and their K/V are reused from the cache.

        :param matmul: Optional CountedMatmul module; when given, the
            similarity matmul is executed through it so the matching
            overhead is FLOP-counted.

        Returns:
            k_matched: [1, H, n_unique, head_dim] cached keys of matched tokens
            v_matched: [1, H, n_unique, head_dim] cached values of matched tokens
            matched_cache_tokens: [1, n_unique, C] cached tokens that were hit
            mt_idx: indices (into x) of matched tokens
            unm_idx: indices (into x) of unmatched tokens
        """
        B, N, C = x.shape

        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

        cache_tokens = self.tokens  # [B, N_cache, C]

        a = x / x.norm(dim=-1, keepdim=True)  # [B, N, C]
        b = cache_tokens / cache_tokens.norm(dim=-1, keepdim=True)  # [B, N_cache, C]

        if matmul is not None:
            scores = matmul(a, b.transpose(-1, -2))  # [B, N, N_cache]
        else:
            scores = a @ b.transpose(-1, -2)  # [B, N, N_cache]

        r = min(a.shape[1], int(a.shape[1] * r_match))  # number of tokens to match

        node_max, node_idx = scores.max(dim=-1)  # [B, N], [B, N]

        # Sort by score to find top r tokens to match
        edge_idx = node_max.argsort(dim=-1, descending=True)  # [B, N]

        # Split into unmatched (lower score) and matched (higher score)
        unm_idx = edge_idx[:, r:]  # [B, N-r] - indices of unmatched tokens
        mt_idx = edge_idx[:, :r]  # [B, r] - indices of matched tokens

        # Cache indices hit by the matched tokens
        mc_idx = gather(node_idx, dim=-1, index=mt_idx)  # [B, r]

        # Deduplicate cache indices
        mc_idx_unique = torch.unique(mc_idx, sorted=False)  # [num_unique]
        mc_idx_unique = mc_idx_unique.unsqueeze(0)  # [1, num_unique]

        if self.verbose:
            print(
                f"  Layer matching: {r}/{N} bg tokens matched to "
                f"{mc_idx_unique.shape[1]} unique cache tokens "
                f"(cache size: {cache_tokens.shape[1]})"
            )

        # K and V are [1, H, N_cache, head_dim], aligned with cache_tokens
        k_matched = gather(
            self.K,
            dim=2,
            index=mc_idx_unique.unsqueeze(1)
            .unsqueeze(-1)
            .expand(-1, self.K.shape[1], -1, self.K.shape[-1]),
        )
        v_matched = gather(
            self.V,
            dim=2,
            index=mc_idx_unique.unsqueeze(1)
            .unsqueeze(-1)
            .expand(-1, self.V.shape[1], -1, self.V.shape[-1]),
        )

        mc = gather(
            cache_tokens, dim=1, index=mc_idx_unique.unsqueeze(-1).expand(-1, -1, C)
        )  # [B, num_unique, C]

        return k_matched, v_matched, mc, mt_idx, unm_idx
