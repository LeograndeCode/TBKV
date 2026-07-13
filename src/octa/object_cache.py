"""
OCTA — Object-Centric Temporal caching with Adaptive tracks.

Training-free inference cache for factorized ViViT that replaces TBKV's
saliency-based background/foreground split with object-centric clustering plus
per-object tracking.

Per spatial frame, for each block:
  1. cluster_tokens         — group patch tokens into clusters using threshold
     bipartite soft matching (each merge group is a cluster; the merged token
     is its representative).  No saliency gating — every cluster is eligible.
  2. match_clusters_to_tracks — offset-aware LOCAL matching of clusters to
     existing tracks (search only tracks whose constant-velocity predicted
     centroid lies within ``offset_window_size`` grid units).  Never global.
  3. matched clusters reuse the track's cached K/V (skip QKV); unmatched
     clusters compute fresh K/V and spawn a NEW track.
  4. TrackCache holds track-indexed state (representative, K/V, centroid
     history, staleness) with create/update/predict/evict.

Token positions flow block-to-block: the first block uses grid coordinates and
each block emits its clusters' centroids as the next block's positions, so the
offset window stays meaningful even after the token set is reduced.

Set ``enable_object_cache=False`` to bypass OCTA entirely (exact dense path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.octa.merge import object_bipartite_matching


# ---------------------------------------------------------------------------
# Configuration (Step 7)
# ---------------------------------------------------------------------------
@dataclass
class ObjectCacheConfig:
    """Hyperparameters for the object-centric cache.

    Cosine thresholds are in ``[-1, 1]``; ``offset_window_size`` is in
    token-grid units (on a 14x14 grid, 2.0 means within 2 cells).
    """

    cluster_similarity_threshold: float = 0.6   # merge tokens above this sim
    offset_window_size: float = 2.0             # grid-unit search radius
    object_match_threshold: float = 0.55        # cosine sim to accept a match
    object_merge_ratio: float = 0.5             # per-pass merge cap (<= 0.5)
    track_eviction_patience: int = 4            # frames unmatched before evict
    merging_iterations: int = 2                 # bipartite merge passes
    enable_object_cache: bool = False           # False => pure dense bypass

    def __post_init__(self) -> None:
        if not (0.0 <= self.object_merge_ratio <= 0.5):
            raise ValueError(
                f"object_merge_ratio must be in [0, 0.5], got {self.object_merge_ratio}"
            )


# ---------------------------------------------------------------------------
# Step 1 — Clustering via threshold bipartite merge
# ---------------------------------------------------------------------------
@dataclass
class TokenClusters:
    """Clustering result for a single frame (B batch, N tokens, K clusters)."""

    representatives: torch.Tensor          # [B, K, C] merged (mean) cluster token
    membership: torch.Tensor               # [B, K, N] binary: token n in cluster k
    centroids: torch.Tensor                # [B, K, D] position-space centroids
    sizes: torch.Tensor                    # [B, K] tokens per cluster
    mean_saliency: Optional[torch.Tensor]  # [B, K] diagnostics only (or None)

    @property
    def num_clusters(self) -> int:
        return int(self.representatives.shape[1])


def cluster_tokens(
    x: torch.Tensor,
    positions: torch.Tensor,
    config: ObjectCacheConfig,
    saliency: Optional[torch.Tensor] = None,
) -> TokenClusters:
    """Cluster tokens using threshold-gated bipartite soft matching.

    Each merge group becomes a cluster; the merged (mean) token is its
    representative.  Membership over the *original* N tokens is tracked by
    pushing a one-hot matrix through the same merge op with an ``amax``
    reduction (keeps it binary).  Centroids are the mean of member positions.

    Args:
        x: [B, N, C] tokens (typically L2-normalized for stable matching).
        positions: [B, N, D] per-token coordinates (grid space).
        config: OCTA hyperparameters.
        saliency: optional [B, N] per-token saliency for diagnostics only.
    """
    B, N, C = x.shape
    D = positions.shape[-1]

    membership = torch.eye(N, device=x.device, dtype=x.dtype)[None].expand(B, N, N).clone()
    reps = x
    for _ in range(max(1, int(config.merging_iterations))):
        merge_fn, r = object_bipartite_matching(
            reps, config.cluster_similarity_threshold, config.object_merge_ratio
        )
        if r == 0:
            break
        reps = merge_fn(reps, mode="mean")            # [B, K, C]
        membership = merge_fn(membership, mode="amax")  # [B, K, N] stays binary

    sizes = membership.sum(dim=-1)                      # [B, K]
    centroids = torch.bmm(membership, positions) / sizes.clamp(min=1.0).unsqueeze(-1)  # [B, K, D]

    mean_sal = None
    if saliency is not None:
        mean_sal = (membership * saliency.unsqueeze(1)).sum(-1) / sizes.clamp(min=1.0)

    assert reps.shape[0] == B and reps.shape[-1] == C
    assert membership.shape[-1] == N
    assert centroids.shape[-1] == D
    return TokenClusters(reps, membership, centroids, sizes, mean_sal)


# ---------------------------------------------------------------------------
# Steps 2 & 4 — Tracks and the track-indexed cache
# ---------------------------------------------------------------------------
@dataclass
class Track:
    """State for a single tracked object (single batch element)."""

    track_id: int
    representative: torch.Tensor           # [C]      merged cluster token
    K: torch.Tensor                        # [H, hd]  cached key for the rep
    V: torch.Tensor                        # [H, hd]  cached value for the rep
    centroid_history: List[torch.Tensor] = field(default_factory=list)  # [D] each
    frames_since_matched: int = 0
    last_updated_frame: int = 0


def predict_position(track: Track) -> torch.Tensor:
    """Constant-velocity centroid prediction from the last two observations.

    Zero motion if only one observation exists.
    """
    hist = track.centroid_history
    if len(hist) == 0:
        return torch.zeros(2)
    if len(hist) == 1:
        return hist[-1]
    return hist[-1] + (hist[-1] - hist[-2])


class TrackCache:
    """Per-block, track-indexed cache (Step 4).

    Tracks are stored per batch element so dynamic create/evict needs no ragged
    batched tensors.  Token / K / V tensors live on-device; the small
    bookkeeping (ids, centroids, counters) is plain Python.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.tracks: List[Dict[int, Track]] = [dict() for _ in range(batch_size)]
        self._next_id: List[int] = [0] * batch_size
        self.frame_idx: int = 0

    def reset(self) -> None:
        self.tracks = [dict() for _ in range(self.batch_size)]
        self._next_id = [0] * self.batch_size
        self.frame_idx = 0

    def create_track(self, b, representative, k, v, centroid) -> int:
        tid = self._next_id[b]
        self._next_id[b] += 1
        self.tracks[b][tid] = Track(
            track_id=tid,
            representative=representative.detach(),
            K=k.detach(),
            V=v.detach(),
            centroid_history=[centroid.detach()],
            frames_since_matched=0,
            last_updated_frame=self.frame_idx,
        )
        return tid

    def update_track(self, b, track_id, centroid,
                     representative=None, k=None, v=None) -> None:
        tr = self.tracks[b][track_id]
        tr.centroid_history.append(centroid.detach())
        if len(tr.centroid_history) > 2:
            tr.centroid_history = tr.centroid_history[-2:]
        tr.frames_since_matched = 0
        tr.last_updated_frame = self.frame_idx
        # Matched clusters reuse cached K/V (Step 5); refresh only if provided.
        if representative is not None:
            tr.representative = representative.detach()
        if k is not None:
            tr.K = k.detach()
        if v is not None:
            tr.V = v.detach()

    def predict_position(self, b, track_id) -> torch.Tensor:
        return predict_position(self.tracks[b][track_id])

    def evict_stale_tracks(self, max_frames_unmatched: int) -> int:
        """Increment staleness and drop tracks unmatched beyond patience."""
        evicted = 0
        for b in range(self.batch_size):
            dead = []
            for tid, tr in self.tracks[b].items():
                tr.frames_since_matched += 1
                if tr.frames_since_matched > max_frames_unmatched:
                    dead.append(tid)
            for tid in dead:
                del self.tracks[b][tid]
                evicted += 1
        return evicted

    def num_tracks(self, b: int) -> int:
        return len(self.tracks[b])


# ---------------------------------------------------------------------------
# Step 2 — Local, offset-aware cluster -> track matching
# ---------------------------------------------------------------------------
def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
    return (a * b).sum(-1)


def match_clusters_to_tracks(
    clusters: TokenClusters,
    cache: TrackCache,
    config: ObjectCacheConfig,
) -> Tuple[List[Dict[int, int]], List[List[int]]]:
    """Match this frame's clusters to existing tracks using LOCAL search.

    For each cluster, only tracks whose constant-velocity predicted centroid is
    within ``offset_window_size`` grid units are candidates (never global).
    Among those, the highest cosine similarity above ``object_match_threshold``
    wins.  Clusters with no in-window candidate above threshold are returned as
    unmatched (they become NEW tracks).

    Returns (matches, unmatched): per batch element, {cluster_idx -> track_id}
    and list of unmatched cluster indices.
    """
    B = clusters.representatives.shape[0]
    K = clusters.num_clusters
    win = float(config.offset_window_size)
    thresh = float(config.object_match_threshold)

    matches: List[Dict[int, int]] = [dict() for _ in range(B)]
    unmatched: List[List[int]] = [list() for _ in range(B)]

    for b in range(B):
        track_ids = list(cache.tracks[b].keys())
        if len(track_ids) == 0:
            unmatched[b] = list(range(K))
            continue

        rep_all = clusters.representatives[b]     # [K, C]
        cen_all = clusters.centroids[b]           # [K, D]

        # Stack track state into dense tensors (one build per block, not per pair).
        trep = torch.stack([cache.tracks[b][t].representative.to(rep_all.device)
                            for t in track_ids], dim=0)                  # [T, C]
        tpred = torch.stack([cache.predict_position(b, t).to(cen_all.device)
                             for t in track_ids], dim=0)                 # [T, D]

        # Vectorized distance + cosine-similarity matrices [K, T].
        dist = torch.cdist(cen_all, tpred)                               # [K, T]
        cn = rep_all / (rep_all.norm(dim=-1, keepdim=True) + 1e-6)
        tn = trep / (trep.norm(dim=-1, keepdim=True) + 1e-6)
        sim = cn @ tn.transpose(0, 1)                                    # [K, T]

        valid = (dist <= win) & (sim >= thresh)
        sim_masked = sim.masked_fill(~valid, float("-inf"))
        if not torch.isfinite(sim_masked).any():
            unmatched[b] = list(range(K))
            continue

        # Greedy 1:1 assignment by descending similarity, resolved on CPU lists
        # (bulk transfer once, no per-pair .item() syncs).
        finite = valid.nonzero(as_tuple=False)                          # [P, 2] (k, t)
        if finite.numel() == 0:
            unmatched[b] = list(range(K))
            continue
        ks = finite[:, 0].tolist()
        ts = finite[:, 1].tolist()
        ss = sim[finite[:, 0], finite[:, 1]].tolist()
        order = sorted(range(len(ss)), key=lambda i: ss[i], reverse=True)

        matched_clusters: set = set()
        claimed: set = set()
        for i in order:
            j = ks[i]
            t_local = ts[i]
            if j in matched_clusters or t_local in claimed:
                continue
            matches[b][j] = track_ids[t_local]
            matched_clusters.add(j)
            claimed.add(t_local)

        unmatched[b] = [j for j in range(K) if j not in matched_clusters]

    return matches, unmatched


# ---------------------------------------------------------------------------
# Steps 3, 5, 6 — Forward integration + FLOP accounting
# ---------------------------------------------------------------------------
def perform_object_matching(
    x: torch.Tensor,
    x_unnorm: Optional[torch.Tensor],
    positions: torch.Tensor,
    saliency: Optional[torch.Tensor],
    cache: TrackCache,
    q_proj,
    k_proj,
    v_proj,
    num_heads: int,
    config: ObjectCacheConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Object-centric replacement for TBKV's per-block matching.

    Clusters the frame, matches clusters to tracks locally, reuses cached K/V
    for matched clusters, computes fresh K/V for unmatched clusters, then
    spawns / updates tracks.  Everything operates at cluster-representative
    granularity, mirroring TBKV's reduced-token output.

    Returns (q, k, v, new_tokens, new_positions, flop_info) with q/k/v of shape
    [B, H, K, hd], new_tokens [B, K, C] (unnormalized, for the residual), and
    new_positions [B, K, D] (cluster centroids for the next block).
    """
    B, N, C = x.shape
    head_dim = C // num_heads
    device = x.device
    if x_unnorm is None:
        x_unnorm = x

    # -- Step 1: cluster ----------------------------------------------------
    clusters = cluster_tokens(x, positions, config, saliency=saliency)
    K = clusters.num_clusters
    reps = clusters.representatives                       # [B, K, C]
    sizes = clusters.sizes.clamp(min=1.0).unsqueeze(-1)   # [B, K, 1]
    reps_unnorm = torch.bmm(clusters.membership, x_unnorm) / sizes  # [B, K, C]

    # -- Step 2: match clusters to tracks (local) --------------------------
    matches, unmatched = match_clusters_to_tracks(clusters, cache, config)

    # -- Steps 3 & 5: assemble K/V (reuse matched, compute fresh) -----------
    k_out = reps.new_zeros((B, num_heads, K, head_dim))
    v_out = reps.new_zeros((B, num_heads, K, head_dim))

    n_matched_total = 0
    n_fresh_total = 0
    for b in range(B):
        for j, tid in matches[b].items():
            tr = cache.tracks[b][tid]
            k_out[b, :, j, :] = tr.K.to(device)
            v_out[b, :, j, :] = tr.V.to(device)
            cache.update_track(b, tid, clusters.centroids[b, j])
            n_matched_total += 1

        fresh_idx = unmatched[b]
        if len(fresh_idx) > 0:
            idx = torch.as_tensor(fresh_idx, device=device, dtype=torch.long)
            fresh_reps = reps[b].index_select(0, idx)              # [F, C]
            k_fresh = k_proj(fresh_reps).reshape(-1, num_heads, head_dim).permute(1, 0, 2)
            v_fresh = v_proj(fresh_reps).reshape(-1, num_heads, head_dim).permute(1, 0, 2)
            for local, j in enumerate(fresh_idx):
                k_out[b, :, j, :] = k_fresh[:, local, :]
                v_out[b, :, j, :] = v_fresh[:, local, :]
                cache.create_track(
                    b, reps[b, j], k_fresh[:, local, :], v_fresh[:, local, :],
                    clusters.centroids[b, j],
                )
            n_fresh_total += len(fresh_idx)

    # Q is computed on ALL representatives.
    q_out = q_proj(reps).reshape(B, K, num_heads, head_dim).permute(0, 2, 1, 3)

    # Shape assertions (Step 5).
    assert q_out.shape == (B, num_heads, K, head_dim)
    assert k_out.shape == v_out.shape == (B, num_heads, K, head_dim)
    assert reps_unnorm.shape == (B, K, C)

    # Evict stale tracks after matching (Step 4).
    cache.frame_idx += 1
    n_evicted = cache.evict_stale_tracks(config.track_eviction_patience)

    # -- Step 6: FLOP accounting -------------------------------------------
    kv_flops_per_token = 2 * C * C                          # k_proj + v_proj
    saved_kv_flops = int(B * n_matched_total * kv_flops_per_token)
    # Local matching cost: cosine over (clusters x in-window tracks) ~ K*C each.
    n_active_tracks = int(sum(cache.num_tracks(b) for b in range(B)))
    match_overhead = int(B * K * C)
    flop_info = {
        "n_tokens": int(N),
        "n_clusters": int(K),
        "n_clusters_matched": int(n_matched_total),
        "n_clusters_fresh": int(n_fresh_total),
        "n_q_tokens": int(K),
        "n_kv_tokens": int(n_fresh_total),
        "saved_kv_flops": saved_kv_flops,
        "match_overhead_flops": match_overhead,
        "n_tracks_active": n_active_tracks,
        "n_tracks_evicted": int(n_evicted),
    }

    return q_out, k_out, v_out, reps_unnorm, clusters.centroids, flop_info
