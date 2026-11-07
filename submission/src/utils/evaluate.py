import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.base import dict_csv_header, dict_csv_line, dict_string
from src.utils.misc import (
    TopKAccuracy,
    get_device_description,
    get_pytorch_device,
    set_policies,
    tee_print,
)


def evaluate_vivit_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    top_1 = TopKAccuracy(k=1)
    top_5 = TopKAccuracy(k=5)
    batch_size = config.get("batch_size", 1)
    num_workers = config.get("num_workers", 2)
    if batch_size > 1:
        # Videos have variable spatial dimensions; pad to max size in each mini-batch
        def collate_pad(batch):
            videos, labels = zip(*batch)
            # Pad spatial dims to the max in this batch
            max_h = max(v.shape[2] for v in videos)
            max_w = max(v.shape[3] for v in videos)
            max_t = max(v.shape[0] for v in videos)
            padded = []
            for v in videos:
                t, c, h, w = v.shape
                pad = torch.zeros(max_t, c, max_h, max_w, dtype=v.dtype)
                pad[:t, :, :h, :w] = v
                padded.append(pad)
            return torch.stack(padded), torch.tensor(labels)
        data_loader = DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=collate_pad)
    else:
        data_loader = DataLoader(data, batch_size=1, num_workers=num_workers)
    n_items = config.get("n_items", len(data_loader))
    n_evaluated = 0
    for idx, (video, label) in tqdm(enumerate(data_loader), total=n_items, ncols=0, file=sys.stdout):
        if idx >= n_items:
            break

        model.reset()

        with torch.inference_mode():
            output = model(video.to(device))
        label = label.to(device)
        top_1.update(output, label)
        top_5.update(output, label)
        n_evaluated += video.shape[0]  # actual items in batch

        if n_evaluated > 0 and n_evaluated % 10 == 0:
            cur_top1 = top_1.compute() * 100
            cur_top5 = top_5.compute() * 100
            counts = model.total_counts() / n_evaluated
            print(
                f"[{n_evaluated:>4}/{n_items * batch_size}]  "
                f"Top-1: {cur_top1:.1f}%  Top-5: {cur_top5:.1f}%  "
                f"linear_flops: {counts.get('linear_flops', 0):.3e}",
                flush=True,
            )

    metrics = {"top_1": top_1.compute(), "top_5": top_5.compute()}
    counts = model.total_counts() / max(n_evaluated, 1)
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def run_evaluations(config, model_class, data, evaluate_function):
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    model = model_class(**(config["model"]))
    msg = model.load_state_dict(
        torch.load(config["weights"], map_location="cpu"), strict=False
    )
    print(f"Weight loading: {msg}")
    model = model.to(device)

    completed = []
    output_dir = Path(config["_output"])

    def do_evaluation(title):
        with open(output_dir / "output.txt", "a") as tee_file:
            # Run the evaluation.
            model.eval()
            results = evaluate_function(device, model, data, config)

            # Print and save results.
            tee_print(title, tee_file)
            tee_print(get_device_description(device), tee_file)
            if isinstance(results, dict):
                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
                for key, val in results.items():
                    tee_print(key.capitalize(), tee_file)
                    tee_print(dict_string(val), tee_file)
            else:
                tee_print(results, tee_file)
            tee_print("", tee_file)
            completed.append(title)

    # Evaluate the model.
    if config.get("vanilla", False):
        do_evaluation("Vanilla")
    for k in config.get("token_top_k", []):
        from src.core.policies import TokenNormTopK
        set_policies(model, TokenNormTopK, k=k)
        do_evaluation(f"Token top k={k}")
    for fraction in config.get("token_top_fraction", []):
        from src.core.policies import TokenNormTopFraction
        set_policies(model, TokenNormTopFraction, fraction=fraction)
        do_evaluation(f"Token top {fraction * 100:.1f}%")
    for threshold in config.get("token_thresholds", []):
        from src.core.policies import TokenNormThreshold
        set_policies(model, TokenNormThreshold, threshold=threshold)
        do_evaluation(f"Token threshold {threshold}")


def save_csv_results(results, output_dir, first_run=False):
    for key, val in results.items():
        with open(output_dir / f"{key}.csv", "a") as csv_file:
            if first_run:
                print(dict_csv_header(val), file=csv_file)
            print(dict_csv_line(val), file=csv_file)
