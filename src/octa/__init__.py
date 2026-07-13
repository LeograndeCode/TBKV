"""OCTA — Object-Centric Temporal caching with Adaptive tracks.

A self-contained, training-free inference cache for factorized ViViT.  This
package does not modify or depend on the ``src/tbkv`` implementation; the token
merging is reimplemented here (``merge.py``) so OCTA is fully isolated.
"""

from src.octa.object_cache import (
    ObjectCacheConfig,
    TokenClusters,
    Track,
    TrackCache,
    cluster_tokens,
    match_clusters_to_tracks,
    perform_object_matching,
    predict_position,
)

__all__ = [
    "ObjectCacheConfig",
    "TokenClusters",
    "Track",
    "TrackCache",
    "cluster_tokens",
    "match_clusters_to_tracks",
    "perform_object_matching",
    "predict_position",
]
