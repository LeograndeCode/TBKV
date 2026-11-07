"""Persistent Semantic Memory (PSM).

A training-free, inference-time layer that adds content-addressable reuse to an
existing video-transformer accelerator. The modules map onto the paper as
follows:

    blocks.py       Algorithm 1 -- the layered filter, plus representation
                    reuse (KV-space substitution)
    filter.py       Memory Retrieval -- cosine matching of candidate tokens
                    against the prototype cache, and the keep/reuse decision
    prototypes.py   Memory Construction -- compute_merge(), which summarises
                    accumulated tokens into prototypes with T merge passes
    merge.py        the bipartite soft-matching primitive one merge pass uses
"""

from src.psm.filter import psm_keep_mask
from src.psm.prototypes import compute_merge
from src.psm.blocks import EventfulPSMBlock, EventfulPSMTokenwiseBlock

__all__ = [
    "psm_keep_mask",
    "compute_merge",
    "EventfulPSMBlock",
    "EventfulPSMTokenwiseBlock",
]
