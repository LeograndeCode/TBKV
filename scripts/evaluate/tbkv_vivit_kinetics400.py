#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.tbkv_vivit import TBKVFactorizedViViT
from src.utils.config import initialize_run
from src.utils.evaluate_tbkv import run_evaluations, evaluate_vivit_metrics


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
