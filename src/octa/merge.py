"""Self-contained bipartite soft matching for OCTA clustering.

This reimplements the ToMe / TBKV bipartite soft matching so the OCTA package
has no dependency on ``src/tbkv``.  The key addition is a *similarity threshold*:
only token pairs whose best cosine similarity exceeds the threshold are merged
(up to a hard 50% cap).  Threshold gating makes clustering self-limiting — once
tokens are distinct objects, few pairs merge, so the token count stays stable
when the operation is stacked across many blocks.
"""

from typing import Callable, Tuple

import torch


def do_nothing(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    return x


def object_bipartite_matching(
    metric: torch.Tensor,
    sim_threshold: float,
    max_ratio: float = 0.5,
) -> Tuple[Callable, int]:
    """Threshold-gated bipartite soft matching.

    Splits tokens into even (a) and odd (b) sets, matches each a-token to its
    most similar b-token, and merges only a-tokens whose best cosine similarity
    is >= ``sim_threshold``, capped at ``max_ratio * T`` merges.  The number of
    merges ``r`` is made uniform across the batch (the per-item minimum count
    above threshold, capped) so the result is a single dense tensor.

    Args:
        metric: [B, T, C] tokens used to measure similarity.
        sim_threshold: cosine similarity in [-1, 1] required to merge a pair.
        max_ratio: hard cap on the merged fraction (<= 0.5 per bipartite rule).

    Returns:
        (merge_fn, r) where merge_fn(x, mode="mean") -> [B, T - r, C].
    """
    B, T, _ = metric.shape
    r_cap = int(max(0.0, min(0.5, max_ratio)) * T)
    if r_cap <= 0 or T < 2:
        return do_nothing, 0

    with torch.no_grad():
        m = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
        a, b = m[..., ::2, :], m[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)              # [B, na, nb]
        node_max, node_idx = scores.max(dim=-1)       # [B, na]

        # Uniform r across the batch: minimum #pairs above threshold, capped.
        above = (node_max >= sim_threshold).sum(dim=-1)   # [B]
        r = int(min(int(above.min().item()), r_cap))
        if r <= 0:
            return do_nothing, 0

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # [B, na, 1]
        unm_idx = edge_idx[:, r:, :]                   # kept a-tokens
        src_idx = edge_idx[:, :r, :]                   # merged a-tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # their b targets

    def merge_fn(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        srcs = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), srcs, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge_fn, r
