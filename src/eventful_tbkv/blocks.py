"""EventfulTBKV: the REAL Eventful block, with TBKV as a recompute-set filter.

The TBKV logic lives in a mixin so it can sit on top of BOTH Eventful block
variants unchanged:

  * EventfulTBKVBlock          <- EventfulBlock          (global-attention / ViViT)
  * EventfulTBKVTokenwiseBlock <- EventfulTokenwiseBlock (windowed / ViTDet)

With TBKV disabled (cache_reuse=0) each is bit-for-bit its base Eventful block.
TBKV then only ever *subtracts* work: Eventful's qkv gate proposes the tokens to
recompute this frame, and TBKV drops the fraction that best match a content cache
(built from the eventful tokens of the warm-up / caching frames), forcing the
reduced set onto all three gates. So per-frame FLOPs are <= Eventful's by
construction. The token gate sits before window partitioning, so the same filter
applies to windowed and global blocks identically.

The content-matching policy itself is the shared tbkv_keep_mask (src/tbkv_filter),
the same one MaskVD+TBKV uses.
"""

import torch
import torch.nn.functional as F

from src.core.blocks import EventfulBlock, EventfulTokenwiseBlock
from src.tbkv.tbkv_utils import compute_merge
from src.tbkv_filter import tbkv_keep_mask


class _Cache:
    def __init__(self, tok):
        self.tok = tok   # [B, M, C] merged eventful-token features (match metric)


class _EventfulTBKVMixin:
    """Adds the TBKV recompute-set filter to an Eventful block. Mix in FIRST."""

    def __init__(self, cache_reuse=0.5, merge_iterations=4, merge_ratio=0.5,
                 caching=False, substitute=False, **super_kwargs):
        super().__init__(**super_kwargs)
        self.cache_reuse = float(cache_reuse)
        self.merge_iterations = int(merge_iterations)
        self.merge_ratio = float(merge_ratio)
        self.caching = bool(caching)
        self._initial_caching = bool(caching)
        # substitute=True (KV-space substitution): matched tokens are pushed
        # BACK into the qkv gate (so their K/V refresh and the class token can
        # attend to them), but with their qkv-gate input replaced by the cached
        # merged token they matched. The projection/MLP gates still process only
        # the kept set, so part of the FLOP saving is retained. Without this the
        # matched tokens' K/V stay stale and, for a class-token readout (ViViT),
        # the substitution never reaches the output.
        self.substitute = bool(substitute)
        self.cache = None
        self._acc = []          # LN-space eventful tokens (match metric + subst src)
        self._forced = None      # index for the projection/MLP gates (kept set)
        self._forced_qkv = None  # index for the qkv gate (kept + matched, when substituting)
        self._subst = None       # (positions [B, m], cache tokens [B, m, C]) in gate-input space
        self.dbg_proposed = 0
        self.dbg_recomputed = 0

    def reset_self(self):
        super().reset_self()
        self.caching = self._initial_caching
        self.cache = None
        self._acc = []
        self._forced = None
        self._forced_qkv = None
        self._subst = None

    def _finalize_cache(self):
        if not self._acc:
            return
        tok = torch.cat(self._acc, dim=1)
        _m, _u, tok_m = compute_merge(tok, self.merge_iterations, self.merge_ratio)
        self.cache = _Cache(tok_m)
        self._acc = []

    def _compute_forced_index(self, gin):
        self._subst = None
        self._forced_qkv = None
        gate = self.qkv_gate
        if gate.first or self.caching or self.cache is None or self.cache_reuse <= 0:
            return None
        with torch.no_grad():
            cand = gate.policy(gin - gate.p, dim=-1)           # [B, k] proposed tokens
            B, k = cand.shape[0], cand.shape[1]
            C = gin.shape[-1]
            feat = gin.gather(1, cand.unsqueeze(-1).expand(B, k, C))
            protect = (cand == 0)                              # never drop the class token
            keep = tbkv_keep_mask(feat, self.cache.tok, self.cache_reuse, protect=protect)
            self.dbg_proposed += int(B * k)
            self.dbg_recomputed += int(keep.sum())
            # tbkv_keep_mask keeps a fixed count per row, so the kept set is
            # uniform-width and can be packed back into an index tensor.
            n_keep = int(keep[0].sum())
            keep_idx = cand[keep].reshape(B, n_keep)
            if self.substitute:
                self._prepare_substitution(cand, feat, keep, n_keep)
            return keep_idx

    def _prepare_substitution(self, cand, feat, keep, n_keep):
        """Push matched tokens back into the qkv gate with cache-token input.

        The qkv gate recomputes the kept AND matched tokens (its natural
        proposal ``cand``), but the matched tokens' gate input is swapped for
        the cached merged token they best match -- so their K/V reflect the
        cache. The projection/MLP gates keep using ``keep`` only.
        """
        B, k = cand.shape
        n_match = k - n_keep
        if n_match <= 0:
            return
        matched_pos = cand[~keep].reshape(B, n_match)          # [B, m] seq indices
        # Best-matching cache token per candidate (same metric as tbkv_keep_mask).
        sim = F.normalize(feat, dim=-1) @ F.normalize(self.cache.tok, dim=-1).transpose(1, 2)
        best = sim.argmax(dim=-1)                              # [B, k]
        matched_cache = best[~keep].reshape(B, n_match)        # [B, m]
        C = self.cache.tok.shape[-1]
        src = self.cache.tok.gather(1, matched_cache.unsqueeze(-1).expand(B, n_match, C))
        self._subst = (matched_pos, src)
        self._forced_qkv = cand                                # qkv recomputes kept + matched

    def forward(self, x):
        if (not self.caching) and self.cache is None and self._acc:
            self._finalize_cache()
        gin = x if self.gate_before_ln else self.input_layer_norm(x)
        self._forced = self._compute_forced_index(gin)
        return super().forward(x)

    def _substitute_gate_input(self, x):
        """Overwrite matched positions' gate-input features with cache tokens."""
        if self._subst is None:
            return x
        pos, src = self._subst
        C = x.shape[-1]
        return x.scatter(-2, pos.unsqueeze(-1).expand(-1, -1, C), src)

    def _forward_pre_attention(self, x):
        skip_1 = x
        # qkv gate gets kept+matched (when substituting); other gates get kept.
        forced_qkv = self._forced if self._forced_qkv is None else self._forced_qkv
        if self.gate_before_ln:
            x = self._substitute_gate_input(x)
            x, index = self.qkv_gate(x, forced_index=forced_qkv)
            x = self.input_layer_norm(x)
        else:
            x = self.input_layer_norm(x)
            x = self._substitute_gate_input(x)
            x, index = self.qkv_gate(x, forced_index=forced_qkv)
        if self.caching:
            self._acc.append(x.detach())
        x = self.qkv(x)
        return skip_1, x, index

    def _forward_post_attention(self, x, skip_1):
        x, index = self.projection_gate(x, forced_index=self._forced)
        x = self.projection(x)
        x = self.projection_accumulator(x, index)

        x = self.add(self.drop_path(x), skip_1)
        skip_2 = x

        if self.gate_before_ln:
            x, index = self.mlp_gate(x, forced_index=self._forced)
            x = self.mlp_layer_norm(x)
        else:
            x = self.mlp_layer_norm(x)
            x, index = self.mlp_gate(x, forced_index=self._forced)
        x = self._forward_mlp(x)
        x = self.mlp_accumulator(x, index)
        x = self.add(self.drop_path(x), skip_2)
        return x


class EventfulTBKVBlock(_EventfulTBKVMixin, EventfulBlock):
    """Global-attention Eventful + TBKV (ViViT spatial, ViTDet global layers)."""


class EventfulTBKVTokenwiseBlock(_EventfulTBKVMixin, EventfulTokenwiseBlock):
    """Windowed Eventful + TBKV (ViTDet windowed layers)."""
