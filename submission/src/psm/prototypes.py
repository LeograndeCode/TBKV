"""Memory construction: merge accumulated tokens into prototypes.

``compute_merge`` is the paper's memory-construction step: ``T`` successive
bipartite soft-matching passes over the tokens gathered during the caching
window, each merging the most similar half-fraction, leaving a compact
prototype set the retrieval step (src/psm/filter.py) matches against.
"""
from typing import Callable, Tuple

import torch

from src.psm.merge import bipartite_soft_matching


def compute_merge(x: torch.Tensor, merging_iterations: int,
                  local_merge_ratio: float = 0.5) -> Tuple[Callable, Callable, torch.Tensor]:
    """Merge tokens into prototypes with successive soft-matching passes.

    :param x: Tokens to be merged [B, N, C].
    :param merging_iterations: Number of successive bipartite matching passes
        (the paper's T).
    :param local_merge_ratio: Fraction of tokens removed per iteration. The
        token count decays as (1 - local_merge_ratio) ** merging_iterations,
        so this must stay small when merging_iterations is large.
    :return: merge function, unmerge function, and the merged tokens.
    """
    merged_tokens = x
    m_seq = []
    u_seq = []
    for _ in range(merging_iterations):
        m_i, u_i = bipartite_soft_matching(
            merged_tokens,
            local_merge_ratio,
            class_token=False,
            distill_token=False,
        )
        m_seq.append(m_i)
        u_seq.append(u_i)
        merged_tokens = m_i(merged_tokens)

    def m(tokens: torch.Tensor) -> torch.Tensor:
        out = tokens
        for m_i in m_seq:
            out = m_i(out)
        return out

    def u(tokens: torch.Tensor) -> torch.Tensor:
        out = tokens
        for u_i in reversed(u_seq):
            out = u_i(out)
        return out

    return m, u, merged_tokens
