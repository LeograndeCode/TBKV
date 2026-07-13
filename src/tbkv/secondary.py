"""
Modular *secondary* token-reduction policies applied **after** TBKV.

TBKV performs cross-frame K/V matching: background tokens that match the cache
reuse cached K/V (ViViT) or skip the whole block (ViTDet).  Everything that TBKV
did *not* reuse — the "active" / "fresh" tokens (unmatched background +
foreground) — is then handed to one of these policies, which applies a *second*
technique on those tokens and their new K/V:

    * ``stgt``      spatial top-k: keep the ``keep_ratio`` most salient active
                    tokens, drop / defer the rest.
    * ``eventful``  temporal gating: keep the ``keep_ratio`` active tokens whose
                    features changed the most since the previous frame; the
                    stable ones reuse their previous result.
    * ``maskvd``    mask gating: keep active tokens whose saliency passes an
                    EMA-smoothed mask threshold (a heat-map style mask), i.e. a
                    soft spatial mask that is temporally smoothed.
    * ``none``      identity (pure TBKV).

Each policy is a light, stateful object.  It exposes a single ``select`` method
that returns *local* indices into the active-token axis:

    keep_local, defer_local = policy.select(block_key, x_active, saliency_active)

``keep_local``  → tokens to fully recompute this frame.
``defer_local`` → tokens to drop (ViViT) or reuse-from-previous-output (ViTDet).

The policies are deliberately model-agnostic: they only look at the active
tokens' features and saliency, so the exact reuse/drop mechanics live in the
model forward (which knows whether it must keep a fixed token grid).
"""

from __future__ import annotations

import torch


def _split_keep_defer(score: torch.Tensor, keep_ratio: float):
    """
    Given a per-token ``score`` [B, A] (higher == keep), return
    ``(keep_local, defer_local)`` index tensors that partition the A active
    tokens.  ``keep_ratio`` is clamped so at least one token is always kept when
    A > 0.
    """
    b, a = score.shape
    if a == 0:
        empty = score.new_zeros((b, 0), dtype=torch.long)
        return empty, empty
    n_keep = int(round(a * float(keep_ratio)))
    n_keep = max(1, min(a, n_keep))
    order = score.argsort(dim=-1, descending=True)
    return order[:, :n_keep], order[:, n_keep:]


class SecondaryPolicy:
    """Base class.  ``none`` behaves as the identity (keep everything)."""

    name = "none"

    def __init__(self, keep_ratio: float = 1.0, **kwargs):
        self.keep_ratio = float(keep_ratio)

    def reset(self):
        """Clear per-video temporal state (called between videos)."""

    def select(self, block_key, x_active: torch.Tensor, saliency_active: torch.Tensor,
               active_idx=None, n_grid=None):
        b, a, _ = x_active.shape
        keep = torch.arange(a, device=x_active.device).unsqueeze(0).expand(b, -1)
        defer = x_active.new_zeros((b, 0), dtype=torch.long)
        return keep, defer

    @property
    def active(self) -> bool:
        return False


class STGTPolicy(SecondaryPolicy):
    """Spatial top-k: keep the most salient active tokens."""

    name = "stgt"

    def __init__(self, keep_ratio: float = 0.5, **kwargs):
        super().__init__(keep_ratio=keep_ratio)

    def select(self, block_key, x_active, saliency_active, active_idx=None,
               n_grid=None):
        return _split_keep_defer(saliency_active, self.keep_ratio)

    @property
    def active(self) -> bool:
        return self.keep_ratio < 1.0


class EventfulPolicy(SecondaryPolicy):
    """
    Temporal gating.  Keeps the active tokens whose feature vectors changed the
    most (largest L2 delta) relative to the previous frame; low-change tokens
    are deferred (reuse previous result).  State is keyed per block and indexed
    on the full token grid so deltas stay position-aligned across frames.
    """

    name = "eventful"

    def __init__(self, keep_ratio: float = 0.5, **kwargs):
        super().__init__(keep_ratio=keep_ratio)
        self._prev = {}  # block_key -> [B, N, C] previous features (full grid)
        self._prev_kv = {}  # block_key -> (k_full [B,H,N,D], v_full [B,H,N,D], valid [B,N])

    def reset(self):
        self._prev = {}
        self._prev_kv = {}

    def select(self, block_key, x_active, saliency_active, active_idx=None,
               n_grid=None):
        b, a, c = x_active.shape
        if a == 0:
            empty = x_active.new_zeros((b, 0), dtype=torch.long)
            return empty, empty
        prev = self._prev.get(block_key)
        if (prev is None or active_idx is None or n_grid is None
                or prev.shape[1] != n_grid):
            # No usable history (first frame, or the token grid changed size
            # between frames -- ViViT's grid is not position-stable): keep all.
            score = torch.full((b, a), float("inf"), device=x_active.device)
        else:
            prev_active = prev.gather(
                1, active_idx.unsqueeze(-1).expand(-1, -1, c)
            )
            score = (x_active - prev_active).pow(2).sum(dim=-1)  # [B, A] L2^2
        keep, defer = _split_keep_defer(score, self.keep_ratio)
        return keep, defer

    def update(self, block_key, x_full):
        """Store the current full-grid features for next-frame delta."""
        self._prev[block_key] = x_full.detach()

    def get_prev_kv(self, block_key):
        """Return previous-frame full-grid K/V + validity mask, if available."""
        return self._prev_kv.get(block_key)

    def update_kv(self, block_key, active_idx, k_active, v_active, n_grid: int):
        """
        Store the current-frame K/V at full-grid positions ``active_idx``.
        ``k_active`` / ``v_active`` are [B, H, A, D] and ``active_idx`` is [B, A].
        """
        b, h, a, d = k_active.shape
        device = k_active.device
        k_full = torch.zeros((b, h, n_grid, d), device=device, dtype=k_active.dtype)
        v_full = torch.zeros((b, h, n_grid, d), device=device, dtype=v_active.dtype)
        valid = torch.zeros((b, n_grid), device=device, dtype=torch.bool)

        scatter_idx = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, h, -1, d)
        k_full = k_full.scatter(2, scatter_idx, k_active)
        v_full = v_full.scatter(2, scatter_idx, v_active)
        valid = valid.scatter(1, active_idx, True)

        self._prev_kv[block_key] = (k_full.detach(), v_full.detach(), valid.detach())

    @property
    def active(self) -> bool:
        return self.keep_ratio < 1.0


class MaskVDPolicy(SecondaryPolicy):
    """
    Mask gating.  Maintains an EMA-smoothed saliency "heat map" over the token
    grid and keeps active tokens whose smoothed saliency is in the top
    ``keep_ratio`` — a temporally-consistent soft spatial mask, in the spirit of
    MaskVD's predicted object mask.
    """

    name = "maskvd"

    def __init__(self, keep_ratio: float = 0.5, ema: float = 0.5, **kwargs):
        super().__init__(keep_ratio=keep_ratio)
        self.ema = float(ema)
        self._heat = {}  # block_key -> [B, N] smoothed saliency

    def reset(self):
        self._heat = {}

    def select(self, block_key, x_active, saliency_active, active_idx=None,
               n_grid=None):
        b, a, _ = x_active.shape
        if a == 0:
            empty = x_active.new_zeros((b, 0), dtype=torch.long)
            return empty, empty
        score = saliency_active
        if active_idx is not None and n_grid is not None:
            heat = self._heat.get(block_key)
            if heat is None or heat.shape[1] != n_grid:
                heat = torch.zeros((b, n_grid), device=x_active.device)
            heat_active = heat.gather(1, active_idx)
            score = self.ema * heat_active + (1.0 - self.ema) * saliency_active
            # Write the blended score back into the heat map at active positions.
            heat = heat.scatter(1, active_idx, score)
            self._heat[block_key] = heat
        return _split_keep_defer(score, self.keep_ratio)

    @property
    def active(self) -> bool:
        return self.keep_ratio < 1.0


class SpatialGatePolicy(SecondaryPolicy):
    """
    Spatial Gate (CST-ViT style).  Intra-frame spatial redundancy elimination:
    among the active / fresh tokens that TBKV must recompute, tokens that are
    highly similar to another active token in the *same* frame are spatially
    redundant.  We keep the least-redundant ``keep_ratio`` fraction (the
    representatives) and defer the rest, which reuse a neighbour's result.

    This is the third stage of CST-ViT's cascade (DTG -> OTG -> SG).  TBKV's
    global cross-frame cosine match already subsumes the temporal gates
    (DTG / OTG); this policy adds the orthogonal *spatial* gate on top.

    A token's ``redundancy`` is its maximum cosine similarity to any other
    active token; the keep-score is the negative of that redundancy so the most
    distinctive tokens are recomputed and near-duplicates are dropped/reused.
    """

    name = "spatial"

    def __init__(self, keep_ratio: float = 0.5, **kwargs):
        super().__init__(keep_ratio=keep_ratio)

    def select(self, block_key, x_active, saliency_active, active_idx=None,
               n_grid=None):
        b, a, c = x_active.shape
        if a <= 1:
            keep = torch.arange(a, device=x_active.device).unsqueeze(0).expand(b, -1)
            defer = x_active.new_zeros((b, 0), dtype=torch.long)
            return keep, defer
        xn = x_active / x_active.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim = xn @ xn.transpose(-1, -2)  # [B, A, A] pairwise cosine similarity
        eye = torch.eye(a, device=x_active.device, dtype=torch.bool)
        sim = sim.masked_fill(eye.unsqueeze(0), float("-inf"))
        redundancy = sim.max(dim=-1).values  # [B, A] max similarity to any peer
        score = -redundancy  # higher == more distinctive == keep
        return _split_keep_defer(score, self.keep_ratio)

    @property
    def active(self) -> bool:
        return self.keep_ratio < 1.0


_POLICIES = {
    "none": SecondaryPolicy,
    "stgt": STGTPolicy,
    "eventful": EventfulPolicy,
    "maskvd": MaskVDPolicy,
    "spatial": SpatialGatePolicy,
}


def make_secondary_policy(name, keep_ratio: float = 0.5, **kwargs):
    """Factory.  ``name`` in {none, stgt, eventful, maskvd, spatial} (case-insensitive)."""
    if name is None:
        return SecondaryPolicy(keep_ratio=1.0)
    key = str(name).lower()
    if key not in _POLICIES:
        raise ValueError(
            f"Unknown secondary policy '{name}'. "
            f"Options: {sorted(_POLICIES)}"
        )
    cls = _POLICIES[key]
    if key == "none":
        return cls(keep_ratio=1.0)
    return cls(keep_ratio=keep_ratio, **kwargs)
