"""EViT -- "Not All Patches are What You Need: Expediting Vision Transformers
via Token Reorganizations" (Liang et al., ICLR 2022).

Rank patch tokens by the attention the class token pays them. Keep the top
keep_rate as "attentive" tokens; do not simply discard the rest -- fuse all
inattentive tokens into a single token, weighted by their class attention, and
append it. The fused token is what keeps EViT from losing the information in
the pruned tail, and it is the part most reimplementations drop.
"""

from typing import Optional, Sequence

import torch

from src.reduction.base import ReduceOp, ReductionContext, TokenReducer


class EViTReducer(TokenReducer):
    """
    :param keep_rate: fraction of patch tokens kept as attentive.
    :param fuse: append the weighted fusion of inattentive tokens (EViT's
        contribution). Setting this False degrades EViT to plain top-k pruning.
    :param prune_layers: block indices at which reorganization happens. EViT
        applies it at a few depths, not every block -- applying keep_rate at all
        12 blocks would compound to keep_rate ** 12 and delete the sequence.
    """

    def __init__(self, keep_rate: float = 0.7, fuse: bool = True,
                 prune_layers: Sequence[int] = (3, 6, 9)):
        super().__init__()
        self.keep_rate = float(keep_rate)
        self.fuse = bool(fuse)
        self.prune_layers = tuple(int(i) for i in prune_layers)

    def plan(self, x: torch.Tensor, ctx: ReductionContext) -> Optional[ReduceOp]:
        if self.keep_rate >= 1.0 or ctx.attn is None:
            return None
        if ctx.block_index not in self.prune_layers:
            return None

        if ctx.cls_attn is not None:
            # TBKV foreground path: class attention over the candidates only.
            scores = ctx.cls_attn
        elif ctx.has_class_token:
            # Class attention: row 0 (the class token's query) over patches.
            scores = ctx.attn[:, :, 0, 1:].mean(dim=1)
        else:
            # No class token to rank by. Fall back to attention *received*: the
            # mean over queries of each token's column, i.e. "how much does the
            # rest of the sequence look at me".
            scores = ctx.attn.mean(dim=1).mean(dim=1)

        n_patch = scores.shape[1]
        n_keep = max(1, int(round(self.keep_rate * n_patch)))
        if n_keep >= n_patch:
            return None

        idx = scores.argsort(dim=-1, descending=True)
        keep_idx = idx[:, :n_keep]
        drop_idx = idx[:, n_keep:]

        # Attention mass of the inattentive tokens, renormalised into fusion
        # weights over just those tokens.
        drop_scores = scores.gather(dim=1, index=drop_idx)  # [B, n_drop]
        weights = drop_scores / (drop_scores.sum(dim=1, keepdim=True) + 1e-6)

        fuse = self.fuse
        has_cls = ctx.has_class_token

        def op(t: torch.Tensor) -> torch.Tensor:
            C = t.shape[-1]
            cls_t = t[:, :1] if has_cls else None
            patches = t[:, 1:] if has_cls else t

            kept = patches.gather(dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, C))
            parts = [kept]
            if fuse:
                dropped = patches.gather(
                    dim=1, index=drop_idx.unsqueeze(-1).expand(-1, -1, C)
                )
                fused = (dropped * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
                parts.append(fused)

            if cls_t is not None:
                parts.insert(0, cls_t)
            return torch.cat(parts, dim=1)

        return op
