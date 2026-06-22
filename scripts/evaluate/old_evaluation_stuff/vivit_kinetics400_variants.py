#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.evit_vivit import EVITFactorizedViViT
from src.models.tbkv_vivit import TBKVFactorizedViViT
from src.utils.config import load_config
from src.utils.evaluate_tbkv import evaluate_vivit_metrics, run_evaluations


def build_config(config_name, variant, n_items, weights, output, overrides):
    config_dir = Path("configs", "evaluate", "vivit_kinetics400")
    config_path = config_dir / f"{config_name}.yml"
    cfg = load_config(config_path, to_container=False)

    if "_name" not in cfg:
        cfg["_name"] = f"{config_name}_{variant}"

    if n_items is not None:
        cfg["n_items"] = n_items

    if weights is not None:
        cfg["weights"] = str(weights)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    resolved = OmegaConf.to_container(cfg, resolve=True)

    if output is not None:
        resolved["_output"] = str(output)

    output_dir = Path(resolved["_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(resolved), output_dir / "config.yml", resolve=True)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ViViT variants (TBKV or EVIT) on Kinetics-400.",
    )
    parser.add_argument(
        "--variant",
        choices=["tbkv", "evit"],
        required=True,
        help="Which model variant to evaluate.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config name under configs/evaluate/vivit_kinetics400 (without .yml). Defaults to tbkv for TBKV and base for EVIT.",
    )
    parser.add_argument("--n-items", type=int, default=None, help="Limit evaluated videos.")
    parser.add_argument("--weights", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--output", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--split", default="val", choices=["train", "test", "val"], help="Kinetics split.")
    parser.add_argument("--decode-size", type=int, default=224)
    parser.add_argument("--decode-fps", type=int, default=25)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override in dotlist format, e.g. model.batch_views=false",
    )
    args = parser.parse_args()

    config_name = args.config or ("tbkv" if args.variant == "tbkv" else "base")
    config = build_config(
        config_name=config_name,
        variant=args.variant,
        n_items=args.n_items,
        weights=args.weights,
        output=args.output,
        overrides=args.overrides,
    )

    model_class = TBKVFactorizedViViT if args.variant == "tbkv" else EVITFactorizedViViT

    data = Kinetics400(
        Path("data", "kinetics400"),
        split=args.split,
        decode_size=args.decode_size,
        decode_fps=args.decode_fps,
    )

    run_evaluations(config, model_class, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main()
