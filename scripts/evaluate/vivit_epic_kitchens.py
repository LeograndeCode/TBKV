#!/usr/bin/env python3

from pathlib import Path

from src.datasets.epic_kitchens import EPICKitchens
from src.models.vivit import FactorizedViViT
from src.utils.config import initialize_run
from src.utils.evaluate import evaluate_vivit_metrics, run_evaluations


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_epic_kitchens")
    )
    data = EPICKitchens(Path("data", "epic_kitchens"), split="validation")
    run_evaluations(config, FactorizedViViT, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main()
