"""AViT -- "Adaptive Token Sampler / A-ViT: Adaptive Tokens for Efficient Vision
Transformer" (Yin et al., CVPR 2022).

A-ViT's mechanism is deliberately parameter-free: it reuses a *single existing
dimension* of each token's embedding as that token's halting score, rather than
adding a head. Per block, h_k = sigmoid(gamma * (x_k[0] - beta)). Halting scores
accumulate across depth, and a token stops once its cumulative score crosses
1 - eps. Stopped tokens are not computed in later blocks.

This part is genuinely training-free, so the mechanism transfers to frozen
ViViT weights as published. What does *not* transfer is calibration: the paper
trains with a ponder loss that shapes dimension 0 into a useful halting signal.
On frozen weights, dimension 0 was never trained to mean anything, so the
halting pattern is essentially arbitrary and this baseline should be read as
"A-ViT's adaptive-depth mechanism, uncalibrated", not as A-ViT's accuracy.

Because tokens halt at different depths and we prune rather than mask, the
cumulative halting state is carried on the *surviving* token set and is
reindexed by the same operator that reduces the tokens.
"""

from typing import Optional

import torch

from src.reduction.base import ReduceOp, ReductionContext, TokenReducer


class AViTReducer(TokenReducer):
    """
    :param gamma: halting-score slope.
    :param beta: halting-score shift. The paper uses a fixed beta=10, which
        only works because training shapes dimension 0 to be large. On frozen
        weights dimension 0 is roughly zero-mean, so sigmoid(gamma*(x0 - 10))
        underflows to 0, nothing ever halts, and the baseline silently becomes
        a no-op. Pass beta=None (the default here) to instead centre the shift
        on each sample's own dimension-0 mean, which restores a live halting
        signal. This is a calibration we add, not something the paper does.
    :param eps: a token halts once cumulative halting exceeds 1 - eps.
    :param max_drop_rate: cap on the fraction of tokens droppable per block.
        Guards against the whole sequence halting at once on uncalibrated
        weights, which would otherwise make the baseline meaningless.
    """

    def __init__(
        self,
        gamma: float = 5.0,
        beta: Optional[float] = None,
        eps: float = 0.01,
        max_drop_rate: float = 0.2,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.beta = None if beta is None else float(beta)
        self.eps = float(eps)
        self.max_drop_rate = float(max_drop_rate)
        self._cumulative = None

    def reset(self) -> None:
        self._cumulative = None

    def plan(self, x: torch.Tensor, ctx: ReductionContext) -> Optional[ReduceOp]:
        if ctx.block_index == 0:
            self.reset()

        B, N, _ = x.shape
        has_cls = ctx.has_class_token
        patches = x[:, 1:] if has_cls else x
        n_patch = patches.shape[1]
        if n_patch <= 1:
            return None

        # A-ViT's parameter-free halting score: dimension 0 of the token.
        x0 = patches[..., 0]                                    # [B, n_patch]
        beta = (
            x0.mean(dim=1, keepdim=True) if self.beta is None
            else torch.as_tensor(self.beta, device=x0.device, dtype=x0.dtype)
        )
        h = torch.sigmoid(self.gamma * (x0 - beta))             # [B, n_patch]

        if self._cumulative is None or self._cumulative.shape[1] != n_patch:
            self._cumulative = torch.zeros_like(h)
        self._cumulative = self._cumulative + h

        halted = self._cumulative >= (1.0 - self.eps)

        # Drop the most-halted tokens, up to the per-block cap. Using a fixed
        # count (rather than the raw mask) keeps tensor shapes uniform across
        # the batch, which the rest of the stack requires.
        n_drop = int(halted.float().sum(dim=1).min().item())
        n_drop = min(n_drop, int(self.max_drop_rate * n_patch), n_patch - 1)
        if n_drop <= 0:
            return None

        # Keep the tokens furthest from halting.
        keep_idx = self._cumulative.topk(
            n_patch - n_drop, dim=-1, largest=False
        ).indices

        # The surviving tokens' halting state must follow them.
        self._cumulative = self._cumulative.gather(dim=1, index=keep_idx)

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
