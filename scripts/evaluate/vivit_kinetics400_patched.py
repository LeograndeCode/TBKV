#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.vivit_patched import (
    AViTPatchedViViT,
    DynamicViTPatchedViViT,
    ToMePatchedViViT,
    EViTPatchedViViT,
)
from src.utils.config import initialize_run
from src.utils.evaluate import evaluate_vivit_metrics, run_evaluations


MODEL_MAP = {
    "avit": AViTPatchedViViT,
    "dynamicvit": DynamicViTPatchedViViT,
    "tome": ToMePatchedViViT,
    "evit": EViTPatchedViViT,
}


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    impl = str(config.get("model_impl", "avit")).lower()
    if impl not in MODEL_MAP:
        raise ValueError(f"Unknown model_impl '{impl}'. Options: {sorted(MODEL_MAP)}")

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    run_evaluations(config, MODEL_MAP[impl], data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main()
