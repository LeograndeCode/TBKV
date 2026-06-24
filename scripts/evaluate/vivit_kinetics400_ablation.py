#!/usr/bin/env python3
"""
Ablation sweep: local_merge_ratio × r_match for TBKV ViViT on Kinetics-400.

Usage:
    python scripts/evaluate/vivit_kinetics400_ablation.py tbkv n_items=25
    python scripts/evaluate/vivit_kinetics400_ablation.py tbkv n_items=1000

Any config key can be overridden on the command line as KEY=VALUE, e.g.:
    python scripts/evaluate/vivit_kinetics400_ablation.py tbkv \\
        n_items=50 \\
        merge_values=[0.5,0.75] \\
        r_match_values=[0.75,0.95,1.0]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.tbkv_vivit import TBKVFactorizedViViT
from src.utils.config import initialize_run
from utils.ablation import run_evaluations, evaluate_vivit_metrics


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    run_evaluations(config, TBKVFactorizedViViT, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main()
