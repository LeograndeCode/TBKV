#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.models.tbkv_vivit import TBKVFactorizedViViT
from src.utils.config import load_config
from utils.evaluate import evaluate_vivit_metrics, run_evaluations


class FilteredDataset(Dataset):
	def __init__(self, base_dataset, indices, target_frames=None):
		self.base_dataset = base_dataset
		self.indices = indices
		self.target_frames = target_frames

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		video, label = self.base_dataset[self.indices[index]]
		if self.target_frames is not None:
			video = video[: self.target_frames]
		return video, label


def _recommended_min_frames(config, temporal_views):
	input_t = int(config["model"]["input_shape"][0])
	temporal_stride = int(config["model"]["temporal_stride"])
	view_size = input_t * temporal_stride
	return view_size + max(0, temporal_views - 1)


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate TBKV ViViT on longer clips by increasing temporal views "
			"and filtering clips with enough decoded frames."
		)
	)
	parser.add_argument(
		"config_name",
		help="Config name under configs/evaluate/vivit_kinetics400 (e.g., tbkv)",
	)
	parser.add_argument("--n-items", type=int, default=25)
	parser.add_argument("--decode-size", type=int, default=224)
	parser.add_argument("--decode-fps", type=int, default=25)
	parser.add_argument("--temporal-views", type=int, default=8)
	parser.add_argument("--min-frames", type=int, default=None)
	parser.add_argument("--target-frames", type=int, default=None)
	parser.add_argument("--overrides", nargs="*", default=[])
	args = parser.parse_args()

	config_path = Path("configs", "evaluate", "vivit_kinetics400", f"{args.config_name}.yml")
	cfg = load_config(config_path, to_container=False)

	dotlist = [
		f"model.temporal_views={args.temporal_views}",
		f"n_items={args.n_items}",
	] + args.overrides
	cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
	cfg["_name"] = args.config_name
	config = OmegaConf.to_container(cfg, resolve=True)

	recommended_min_frames = _recommended_min_frames(config, args.temporal_views)
	min_frames = args.min_frames if args.min_frames is not None else recommended_min_frames

	data = Kinetics400(
		Path("data", "kinetics400"),
		split="val",
		decode_size=args.decode_size,
		decode_fps=args.decode_fps,
		shuffle=False,
	)

	valid_indices = [
		i for i, info in enumerate(data.videos_info) if len(info["frames"]) >= min_frames
	]
	if not valid_indices:
		raise RuntimeError(
			f"No videos with >= {min_frames} frames at decode_fps={args.decode_fps}."
		)

	filtered = FilteredDataset(data, valid_indices, target_frames=args.target_frames)
	config["n_items"] = min(args.n_items, len(filtered))

	target_frames = args.target_frames if args.target_frames is not None else "all"
	suffix = (
		f"{args.config_name}-moreframes-tv{args.temporal_views}-minf{min_frames}"
		f"-tf{target_frames}-fps{args.decode_fps}-n{config['n_items']}"
	)
	config["_name"] = suffix
	config["_output"] = f"results/evaluate/vivit_kinetics400/{suffix}/"
	Path(config["_output"]).mkdir(parents=True, exist_ok=True)

	print("More-frames TBKV evaluation setup")
	print(f"  config: {args.config_name}")
	print(f"  temporal_views: {args.temporal_views}")
	print(f"  target_frames: {target_frames}")
	print(f"  decode_fps: {args.decode_fps}")
	print(f"  min_frames: {min_frames} (recommended: {recommended_min_frames})")
	print(f"  eligible_videos: {len(valid_indices)}")
	print(f"  n_items used: {config['n_items']}")
	print(f"  output: {config['_output']}")

	run_evaluations(config, TBKVFactorizedViViT, filtered, evaluate_vivit_metrics)


if __name__ == "__main__":
	main()
