"""Modular token-reduction interface.

Every reduction algorithm (ToMe, EViT, DynamicViT, AViT) is expressed as a
single thing: given a set of tokens and the attention state that produced
them, return an **operator** that maps [B, N, C] -> [B, N', C].

That one abstraction is what lets each algorithm be written once and used in
two places:

  * ReducerBlock  -- applies the operator to the whole token sequence, which
                     reproduces the algorithm as published (the baseline).
  * TBKVBlock     -- applies the same operator to only the foreground slice,
                     leaving the cached/background tokens to TBKV.

Because an operator is just a tensor map, it can be applied identically to the
normed view, the unnormed residual view, and any per-token bookkeeping, which
is what keeps the two call sites from drifting apart.

Reduction is applied to the post-attention residual stream (after the
attention output is added to the skip, before the MLP). This is where ToMe and
EViT operate in their reference implementations, and it is a valid hook for
DynamicViT and AViT as well.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

# An operator maps a token tensor [B, N, C] to a reduced one [B, N', C].
ReduceOp = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ReductionContext:
    """Everything a reducer may need, gathered at the reduction hook.

    attn:      post-softmax attention, [B, heads, N_q, N_k]. Row i is the
               attention of token i; column j is attention *received* by j.
    cls_attn:  attention the class token pays each candidate token, [B, N].
               Supplied explicitly because in the TBKV foreground path the
               candidates are a slice of the sequence, so a reducer cannot
               recover this from attn by indexing row 0. EViT and DynamicViT
               prefer it when present.
    keys:      per-head keys, [B, heads, N_k, head_dim]. ToMe's metric.
    has_class_token: whether index 0 of the token sequence is the class token.
    block_index / depth: position in the stack, for schedule-driven reducers.

    In the TBKV foreground path these are already sliced down to the
    foreground tokens, so a reducer never needs to know which call site it is
    serving.
    """

    attn: Optional[torch.Tensor] = None
    keys: Optional[torch.Tensor] = None
    cls_attn: Optional[torch.Tensor] = None
    has_class_token: bool = False
    block_index: int = 0
    depth: int = 12


class TokenReducer(nn.Module):
    """Base class. Subclasses implement plan().

    A reducer is a single object shared by every block in the stack; blocks
    identify themselves via ctx.block_index. Sharing matters for AViT, whose
    halting scores accumulate across depth.
    """

    def plan(self, x: torch.Tensor, ctx: ReductionContext) -> Optional[ReduceOp]:
        """Return an operator reducing x's token axis, or None for a no-op."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any state carried across blocks. Called at block_index 0."""
        pass
