"""DynamicViT -- "DynamicViT: Efficient Vision Transformers with Dynamic Token
Sparsification" (Rao et al., NeurIPS 2021).

DynamicViT prunes tokens at a few fixed depths rather than every block: at each
pruning stage it keeps a keep_ratio fraction of the tokens still alive, so the
survival rate compounds (keep_ratio ** stages_passed).

IMPORTANT -- what is and is not faithful here.
The paper scores tokens with a learned prediction head that consumes a local
feature (the token) and a global feature (the mean of all tokens), and trains
it end-to-end with a Gumbel-Softmax relaxation and a sparsity loss. That head
is meaningless without training. Since this runs inference-only on frozen ViViT
weights, we keep DynamicViT's *architecture* -- staged pruning at fixed depths,
compounding keep ratio -- but replace the untrained head with an attention-based
importance score (attention received, modulated by the token's own magnitude,
which is the same local+global pairing the head is given).

So this is a faithful DynamicViT *schedule* with a training-free scorer. It is
a fair efficiency baseline, but it is not the published accuracy of DynamicViT,
and it should be labelled as such in any comparison table.
"""

from typing import Optional, Sequence

import torch

from src.reduction.base import ReduceOp, ReductionContext, TokenReducer


class DynamicViTReducer(TokenReducer):
    """
    :param keep_ratio: fraction of surviving tokens kept at each pruning stage.
    :param prune_layers: block indices at which pruning happens (paper: 3/6/9
        for a 12-block ViT).
    """

    def __init__(
        self,
        keep_ratio: float = 0.7,
        prune_layers: Sequence[int] = (3, 6, 9),
    ):
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.prune_layers = tuple(int(i) for i in prune_layers)

    def plan(self, x: torch.Tensor, ctx: ReductionContext) -> Optional[ReduceOp]:
        if ctx.block_index not in self.prune_layers or ctx.attn is None:
            return None
        if self.keep_rate_invalid():
            return None

        # Global term: how much attention each token receives from the rest.
        if ctx.cls_attn is not None:
            received = ctx.cls_attn
        else:
            attn = ctx.attn.mean(dim=1)          # [B, N_q, N_k], heads averaged
            received = attn.mean(dim=1)          # [B, N_k]
            if ctx.has_class_token:
                received = received[:, 1:]

        n_patch = received.shape[1]
        n_keep = max(1, int(round(self.keep_ratio * n_patch)))
        if n_keep >= n_patch:
            return None

        # Local term: the token's own magnitude.
        patches = x[:, 1:] if ctx.has_class_token else x
        local = patches.norm(dim=-1)             # [B, n_patch]
        local = local / (local.amax(dim=-1, keepdim=True) + 1e-6)

        scores = received * local

        keep_idx = scores.topk(n_keep, dim=-1).indices
        has_cls = ctx.has_class_token

        def op(t: torch.Tensor) -> torch.Tensor:
            C = t.shape[-1]
            patches_t = t[:, 1:] if has_cls else t
            kept = patches_t.gather(
                dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, C)
            )
            if has_cls:
                return torch.cat([t[:, :1], kept], dim=1)
            return kept

        return op

    def keep_rate_invalid(self) -> bool:
        return not (0.0 < self.keep_ratio < 1.0)
