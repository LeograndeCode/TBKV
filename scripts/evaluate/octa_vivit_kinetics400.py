#!/usr/bin/env python3
"""Evaluate OCTA (object-centric temporal caching) ViViT on Kinetics-400.

Usage:
    python scripts/evaluate/octa_vivit_kinetics400.py octa \
        n_items=10 frame_split=4

Top-level CLI overrides (e.g. object_merge_ratio=0.4 offset_window_size=3.0)
are routed into the spatial block config where OCTA runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.octa_vivit import OCTAFactorizedViViT
from src.utils.config import initialize_run
from utils.evaluate import run_evaluations, evaluate_vivit_metrics


_OCTA_OVERRIDE_KEYS = (
    "enable_object_cache",
    "cluster_similarity_threshold",
    "offset_window_size",
    "object_match_threshold",
    "object_merge_ratio",
    "track_eviction_patience",
    "merging_iterations",
    "caching",
)


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    spatial_cfg = config["model"].get("spatial_config", {}).get("block_config")
    if spatial_cfg is not None:
        for key in _OCTA_OVERRIDE_KEYS:
            if key in config:
                spatial_cfg[key] = config[key]
    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    run_evaluations(config, OCTAFactorizedViViT, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main()
