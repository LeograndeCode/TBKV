#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.tbkv_vivit import TBKVFactorizedViViT
from src.utils.config import initialize_run
from utils.evaluate import run_evaluations, evaluate_vivit_metrics


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    # Route top-level CLI overrides (e.g. r_match=0.5 bg_ratio=0.4) into both
    # block configs, so a swept value means the same thing in each stack.
    _routed = (
        "r_match", "merging_iterations", "caching",
        "secondary", "secondary_keep", "split_tokens", "bg_ratio",
        "local_merge_ratio",
    )
    for stack in ("spatial_config", "temporal_config"):
        block_cfg = config["model"].get(stack, {}).get("block_config")
        if block_cfg is None:
            continue
        for key in _routed:
            if key in config:
                block_cfg[key] = config[key]
    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    run_evaluations(config, TBKVFactorizedViViT, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main()
