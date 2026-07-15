"""ToMe -- "Token Merging: Your ViT But Faster" (Bolya et al., ICLR 2023).

Bipartite soft matching: split tokens into alternating src/dst sets, match each
src token to its most similar dst token by cosine similarity on the attention
keys, and merge the r most-similar pairs by averaging. Reference behaviour:
r tokens are removed per block, and merged tokens carry a size for weighted
averaging so a merged token keeps the influence of everything inside it.
"""

from typing import Optional

import torch

from src.reduction.base import ReduceOp, ReductionContext, TokenReducer
from src.tbkv.merge import bipartite_soft_matching


class ToMeReducer(TokenReducer):
    """
    :param r: tokens removed per block (the paper's r). A constant schedule.
    """

    def __init__(self, r: int = 8):
        super().__init__()
        self.r = int(r)
        self._size = None

    def reset(self) -> None:
        self._size = None

    def plan(self, x: torch.Tensor, ctx: ReductionContext) -> Optional[ReduceOp]:
        if ctx.block_index == 0:
            self.reset()

        B, N, _ = x.shape
        protected = 1 if ctx.has_class_token else 0
        # Can never remove more than half the mergeable tokens.
        r = min(self.r, (N - protected) // 2)
        if r <= 0 or ctx.keys is None:
            return None

        # ToMe's metric is the key, averaged over heads.
        metric = ctx.keys.mean(dim=1)  # [B, N, head_dim]

        # bipartite_soft_matching takes r as a *fraction* of the token count,
        # so convert the paper's absolute r into the ratio it expects.
        merge, _ = bipartite_soft_matching(
            metric, r / N, class_token=ctx.has_class_token, distill_token=False
        )

        # Proportional attention / weighted merging: a token that already
        # represents k originals must count k times when averaged again.
        size = self._size
        # In the TBKV foreground path the candidate set is re-selected each
        # block, so a size vector carried over from the previous block need not
        # match the current token count. Fall back to uniform sizes when it
        # doesn't line up.
        if size is None or size.shape[1] != N:
            size = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
        new_size = merge(size, mode="sum")

        def op(t: torch.Tensor) -> torch.Tensor:
            return merge(t * size, mode="sum") / new_size

        self._size = new_size
        return op
