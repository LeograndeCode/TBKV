"""
Diagnostics + correctness checks for the OCTA object-centric cache.

Runs on synthetic token streams (no model weights required) and:
  * asserts tensor shapes / partitioning through clustering + matching + forward,
  * verifies constant-velocity prediction,
  * prints per-frame diagnostics: #clusters, #tracks, matched vs fresh,
    centroid prediction error, and token-reduction ratio.

Usage:
    python scripts/misc/test_object_cache.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.octa import (  # noqa: E402
    ObjectCacheConfig,
    Track,
    TrackCache,
    cluster_tokens,
    match_clusters_to_tracks,
    perform_object_matching,
    predict_position,
)


def _grid_positions(grid_hw, device, dtype):
    h, w = grid_hw
    rows = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w)
    cols = torch.arange(w, device=device, dtype=dtype).repeat(h)
    return torch.stack([rows, cols], dim=-1)  # [N, 2]


def _make_frame(grid_hw, C, n_objects, t, device):
    h, w = grid_hw
    N = h * w
    torch.manual_seed(0)
    centers = torch.rand(n_objects, 2) * torch.tensor([h, w]).float()
    protos = torch.randn(n_objects, C)
    drift = torch.tensor([0.6, 0.4])
    x = torch.randn(N, C) * 0.05
    pos = _grid_positions(grid_hw, device, torch.float32)
    for o in range(n_objects):
        c = centers[o] + drift * t
        d = torch.linalg.norm(pos - c, dim=-1)
        w_o = torch.exp(-(d ** 2) / (2 * 2.0 ** 2))
        x = x + w_o.unsqueeze(-1) * protos[o]
    return x.unsqueeze(0).to(device)  # [1, N, C]


def main():
    device = torch.device("cpu")
    grid_hw = (14, 14)
    N = grid_hw[0] * grid_hw[1]
    C = 768
    num_heads = 12

    cfg = ObjectCacheConfig(
        cluster_similarity_threshold=0.5,
        offset_window_size=3.0,
        object_match_threshold=0.4,
        object_merge_ratio=0.5,
        track_eviction_patience=3,
        merging_iterations=2,
        enable_object_cache=True,
    )

    q = torch.nn.Linear(C, C)
    k = torch.nn.Linear(C, C)
    v = torch.nn.Linear(C, C)
    positions = _grid_positions(grid_hw, device, torch.float32)[None]  # [1, N, 2]

    # -- clustering unit checks -------------------------------------------
    x0 = torch.nn.functional.normalize(_make_frame(grid_hw, C, 4, 0, device), dim=-1)
    clusters = cluster_tokens(x0, positions, cfg)
    assert clusters.membership.shape[-1] == N
    counts = clusters.membership.sum(dim=1)  # [B, N]
    assert torch.allclose(counts, torch.ones_like(counts)), "tokens must be partitioned 1:1"
    assert int(clusters.sizes.sum().item()) == N
    print(f"[cluster] N={N} -> K={clusters.num_clusters} clusters, "
          f"sizes sum={int(clusters.sizes.sum())} (ok)")

    # -- predict_position --------------------------------------------------
    tr = Track(0, x0[0, 0], torch.zeros(num_heads, C // num_heads),
               torch.zeros(num_heads, C // num_heads),
               centroid_history=[torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0])])
    pred = predict_position(tr)
    assert torch.allclose(pred, torch.tensor([3.0, 5.0])), pred
    print(f"[predict] CV prediction ok -> {pred.tolist()}")

    # -- multi-frame forward + diagnostics --------------------------------
    cache = TrackCache(batch_size=1)
    print("\nframe |  K  | tracks | matched | fresh | evict | reduction | cent.err")
    print("------+-----+--------+---------+-------+-------+-----------+---------")
    for t in range(8):
        xf = torch.nn.functional.normalize(_make_frame(grid_hw, C, 4, t, device), dim=-1)
        cl = cluster_tokens(xf, positions, cfg)
        m, _ = match_clusters_to_tracks(cl, cache, cfg)
        cent_err, n_pred = 0.0, 0
        for j, tid in m[0].items():
            pc = cache.predict_position(0, tid)
            cent_err += torch.linalg.norm(cl.centroids[0, j] - pc).item()
            n_pred += 1
        cent_err = cent_err / max(1, n_pred)

        qh, kh, vh, new_tokens, new_pos, info = perform_object_matching(
            xf, xf, positions, None, cache, q, k, v, num_heads, cfg,
        )
        assert qh.shape == (1, num_heads, info["n_clusters"], C // num_heads)
        assert new_tokens.shape == (1, info["n_clusters"], C)
        assert new_pos.shape == (1, info["n_clusters"], 2)
        reduction = 1.0 - info["n_clusters"] / N
        print(f"  {t:^3} | {info['n_clusters']:^3} | {info['n_tracks_active']:^6} | "
              f"{info['n_clusters_matched']:^7} | {info['n_clusters_fresh']:^5} | "
              f"{info['n_tracks_evicted']:^5} | {reduction:^9.2f} | {cent_err:^7.2f}")

    print("\nAll shape/partition/prediction checks passed.")


if __name__ == "__main__":
    main()
