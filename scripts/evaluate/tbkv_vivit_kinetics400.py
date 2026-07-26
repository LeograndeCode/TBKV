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
    # Layered-style hyperparameter aliases, so the standalone accepts the same
    # names as the layered EventfulTBKV runs:
    #   cr / cache_reuse   -> bg_ratio          (fraction treated as background,
    #                                            i.e. eligible for cache reuse)
    #   mi / merge_iterations -> merging_iterations
    # NOTE: bg_ratio only sizes the reusable (background) set; how much of it is
    # actually reused is bg_ratio * r_match. So cr=0.95 with the default
    # r_match=0.5 reuses ~47% of tokens, not 95%. Pass r_match=1.0 alongside cr
    # to make cr behave like the layered "fraction reused" knob.
    _aliases = {"cr": "bg_ratio", "cache_reuse": "bg_ratio",
                "mi": "merging_iterations", "merge_iterations": "merging_iterations"}
    for alias, target in _aliases.items():
        if alias in config and target not in config:
            config[target] = config[alias]

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
