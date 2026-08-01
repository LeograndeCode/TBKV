"""Shared PSM filter: the one idea reused across base methods.

Every "<method> + PSM" integration reduces to the same operation: the base
method (Eventful, MaskVD, ...) proposes a set of tokens to compute this frame;
PSM drops the fraction of them that best match a content cache, because a token
that looks like something already stable in the cache does not need recomputing.

Keeping this in one function means each integration only has to answer two
method-specific questions -- "what are the candidate tokens' features?" and "how
do I skip a token?" -- while the content-matching policy lives here, once.
"""

import torch
import torch.nn.functional as F


def psm_keep_mask(cand_feat, cache_feat, cache_reuse, protect=None):
    """Decide which candidate tokens to keep (recompute) vs drop (reuse).

    :param cand_feat: [B, k, D] features of the base method's proposed tokens.
    :param cache_feat: [B, M, D] cache features to match against.
    :param cache_reuse: fraction of candidates to drop (0 -> keep all).
    :param protect: optional bool [B, k]; True entries are never dropped.
    :returns: bool [B, k] keep-mask (True = recompute, False = reuse from cache).
    """
    B, k = cand_feat.shape[0], cand_feat.shape[1]
    keep_all = torch.ones(B, k, dtype=torch.bool, device=cand_feat.device)
    if cache_reuse <= 0 or k == 0 or cache_feat is None or cache_feat.shape[1] == 0:
        return keep_all

    sim = F.normalize(cand_feat, dim=-1) @ F.normalize(cache_feat, dim=-1).transpose(1, 2)
    best = sim.max(dim=-1).values                       # [B, k] best cache match
    if protect is not None:
        best = best.masked_fill(protect, float("-inf"))  # protected -> most novel -> kept

    n_keep = max(1, k - int(round(cache_reuse * k)))
    keep_pos = best.argsort(dim=-1)[:, :n_keep]          # least cache-similar kept
    mask = torch.zeros(B, k, dtype=torch.bool, device=cand_feat.device)
    mask.scatter_(1, keep_pos, True)
    return mask
