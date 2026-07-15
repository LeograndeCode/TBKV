"""TBKV rebuilt on top of Eventful, as motion compensation rather than token reduction.

Why this exists
---------------
Measured on ImageNet-VID / ViTDet, TBKV used as a *token reducer* contributes
negative value: it costs ~16 mAP@50 to save ~25% FLOPs, and stacking it under
Eventful is strictly worse than Eventful alone. So the reducer framing is dead.

But TBKV's underlying idea -- match tokens by *content* against a cache, rather
than by position -- is aimed at a real weakness in Eventful that Eventful cannot
fix from the inside:

    Eventful's TokenGate keeps a reference state p, where p[i] is the value of
    grid position i on the last frame it was updated. It recomputes the top-k
    tokens by |c - p| and leaves the rest stale.

    Under camera motion the content that was at position i is now at position j.
    So p is wrong at BOTH positions: the gate burns update budget on both, and
    every token it cannot afford to update keeps a value belonging to different
    content. Eventful degrades exactly here, and no choice of k repairs it,
    because the reference itself is mis-addressed.

This block keeps Eventful's gating untouched and repairs only the *reference*.
For each token the gate left stale, we search a small neighbourhood of the
previous frame for the position whose content best matches, and reuse that
position's output. Content-addressed reuse -- TBKV's actual idea -- applied to
the value Eventful reuses, instead of to deleting tokens.

Nothing is removed, so this cannot destroy accuracy the way the reducer did; the
worst case is that it declines to fire and the block is exactly Eventful.

Cost
----
A full N x N content match would cost ~5.7e10 FLOPs across the stack -- three
times Eventful's entire budget, which would defeat the point. Two things keep it
negligible (~1e8 FLOPs/block):

  * match on a low-dimensional slice of the token, not all 768 channels;
  * search only a local window, which is the right prior anyway, since camera
    motion is local displacement.
"""

import torch
import torch.nn.functional as F

from src.core.blocks import EventfulBlock, EventfulTokenwiseBlock


class TBKVEventfulBlock(EventfulBlock):
    """Eventful, with a motion-compensated reference for stale tokens.

    :param tbkv_radius: half-width of the local search window, in tokens. The
        maximum displacement that can be compensated. 0 disables the block's
        TBKV behaviour entirely (it becomes plain Eventful).
    :param tbkv_dim: channels of the matching descriptor. Matching runs on a
        slice of the token rather than all of them, which is what makes the
        search cheap enough to be worth doing.
    :param tbkv_tau: cosine-similarity threshold. A stale token only adopts a
        matched value if the match is at least this good; otherwise it keeps
        Eventful's behaviour. This is the safety valve -- a bad match is worse
        than a stale value, so we would rather do nothing.
    """

    def __init__(self, tbkv_radius=3, tbkv_dim=64, tbkv_tau=0.9, **super_kwargs):
        super().__init__(**super_kwargs)
        self.tbkv_radius = int(tbkv_radius)
        self.tbkv_dim = int(tbkv_dim)
        self.tbkv_tau = float(tbkv_tau)
        self._prev_key = None   # [B, N, D] descriptors of the previous frame
        self._prev_out = None   # [B, N, C] block outputs of the previous frame
        self._last_index = None  # positions the gate refreshed this frame

    def reset_self(self):
        super().reset_self()
        self._prev_key = None
        self._prev_out = None
        self._last_index = None

    # The MLP gate is the last gate before the block output, so the positions it
    # updated are exactly the positions whose output is fresh. Everything else is
    # stale and is what we try to repair.
    def _forward_post_attention(self, x, skip_1):
        x, index = self.projection_gate(x)
        x = self.projection(x)
        x = self.projection_accumulator(x, index)

        x = self.add(self.drop_path(x), skip_1)
        skip_2 = x

        if self.gate_before_ln:
            x, index = self.mlp_gate(x)
            x = self.mlp_layer_norm(x)
        else:
            x = self.mlp_layer_norm(x)
            x, index = self.mlp_gate(x)
        self._last_index = index
        x = self._forward_mlp(x)
        x = self.mlp_accumulator(x, index)
        x = self.add(self.drop_path(x), skip_2)
        return x

    def forward(self, x):
        key = x[..., : self.tbkv_dim].detach()  # content descriptor: block input
        out = super().forward(x)
        out = self._motion_refresh(key, out)
        self._prev_key = key
        self._prev_out = out.detach()
        return out

    def _motion_refresh(self, key, out):
        """Replace stale token outputs with the best local content match."""
        if self.tbkv_radius <= 0 or self._prev_out is None or self._last_index is None:
            return out
        # Windowed blocks arrange windows along the batch axis, so the token axis
        # is not a plain HxW grid and the local search has no meaning there.
        if self.window_size is not None or self.input_size is None:
            return out

        B, N, C = out.shape
        H, W = self.input_size
        if H * W != N:
            return out

        r = self.tbkv_radius
        k = 2 * r + 1
        D = key.shape[-1]

        q = F.normalize(key.float(), dim=-1)                       # [B, N, D]
        p = F.normalize(self._prev_key.float(), dim=-1)            # [B, N, D]

        # Gather each position's local neighbourhood from the previous frame.
        p_grid = p.transpose(1, 2).reshape(B, D, H, W)
        cand = F.unfold(p_grid, kernel_size=k, padding=r)          # [B, D*k*k, N]
        cand = cand.view(B, D, k * k, N).permute(0, 3, 2, 1)       # [B, N, k*k, D]

        sim = (cand * q.unsqueeze(2)).sum(dim=-1)                  # [B, N, k*k]
        best_sim, best_off = sim.max(dim=-1)                       # [B, N]

        # Turn the winning window offset into an absolute source position.
        pos = torch.arange(N, device=out.device)
        row, col = pos // W, pos % W
        d_row = best_off // k - r
        d_col = best_off % k - r
        src_r = (row.unsqueeze(0) + d_row).clamp(0, H - 1)
        src_c = (col.unsqueeze(0) + d_col).clamp(0, W - 1)
        src = (src_r * W + src_c)                                  # [B, N]

        matched = self._prev_out.gather(
            1, src.unsqueeze(-1).expand(-1, -1, C)
        ).to(out.dtype)

        # Only stale positions are eligible: a token the gate just recomputed is
        # exact, and must never be overwritten by a cached approximation.
        stale = torch.ones(B, N, dtype=torch.bool, device=out.device)
        if self._last_index is not None:
            stale.scatter_(1, self._last_index, False)

        # And only adopt a match we actually trust. Below tau we keep Eventful's
        # stale value, so this can only help relative to plain Eventful.
        take = stale & (best_sim >= self.tbkv_tau)
        # A zero displacement is what Eventful already does -- no point rewriting.
        take &= (src != pos.unsqueeze(0))

        return torch.where(take.unsqueeze(-1), matched, out)


class TBKVEventfulTokenwiseBlock(EventfulTokenwiseBlock):
    """Windowed counterpart. Accepts the TBKV kwargs and ignores them.

    Windowed blocks arrange windows along the batch axis, so their token axis is
    not an HxW grid and a local spatial search is meaningless there. This class
    exists purely so the shared block_config can carry tbkv_* keys; its
    behaviour is identical to EventfulTokenwiseBlock, which keeps the comparison
    against plain Eventful exact -- only the global blocks differ.
    """

    def __init__(self, tbkv_radius=0, tbkv_dim=0, tbkv_tau=0.0, **super_kwargs):
        super().__init__(**super_kwargs)
