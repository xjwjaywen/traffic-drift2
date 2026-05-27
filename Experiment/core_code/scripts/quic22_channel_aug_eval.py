"""
QUIC22 targeted channel augmentation evaluation.

This is a lightweight test-time augmentation diagnostic. It does not update
model parameters. For each target period and augmentation setting, it averages
the frozen model's predictions over the original input plus K augmented views.

The goal is to distinguish targeted channel augmentation from broad MVFC-style
augmentation/adaptation.
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.models import TTATCModel
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


CHANNELS = {
    "size": 0,
    "direction": 1,
    "ipt": 2,
}


def load_source_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["num_classes"], cfg


def parse_positions(region):
    if region == "all":
        return list(range(30))
    if region == "front":
        return list(range(0, 10))
    if region == "middle":
        return list(range(10, 20))
    if region == "tail":
        return list(range(20, 30))
    match = re.fullmatch(r"p(\d+)_(\d+)", region)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return list(range(start, end + 1))
    raise ValueError(f"Unknown region: {region}")


def parse_setting(setting):
    if setting == "raw":
        return {"name": setting, "kind": "raw"}

    parts = setting.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid setting: {setting}")

    channel = parts[0]
    if channel not in CHANNELS and channel != "packet":
        raise ValueError(f"Unknown channel in setting: {setting}")

    if channel == "packet":
        if parts[1] != "mask":
            raise ValueError(f"Unsupported packet setting: {setting}")
        return {
            "name": setting,
            "kind": "packet_mask",
            "prob": float(parts[2]),
            "positions": list(range(30)),
        }

    if parts[1] == "noise":
        return {
            "name": setting,
            "kind": "mul_noise",
            "channel": CHANNELS[channel],
            "positions": list(range(30)),
            "std": float(parts[2]),
        }
    if parts[1] in ("front", "middle", "tail") and len(parts) == 4 and parts[2] == "dropout":
        return {
            "name": setting,
            "kind": "dropout",
            "channel": CHANNELS[channel],
            "positions": parse_positions(parts[1]),
            "prob": float(parts[3]),
        }
    if parts[1] == "dropout":
        return {
            "name": setting,
            "kind": "dropout",
            "channel": CHANNELS[channel],
            "positions": list(range(30)),
            "prob": float(parts[2]),
        }
    raise ValueError(f"Unsupported setting: {setting}")


def augment_ppi(ppi, spec):
    if spec["kind"] == "raw":
        return ppi

    view = ppi.clone()
    if spec["kind"] == "packet_mask":
        keep = (torch.rand(ppi.size(0), 1, ppi.size(2), device=ppi.device) > spec["prob"]).float()
        return view * keep

    channel = spec["channel"]
    positions = torch.tensor(spec["positions"], device=ppi.device, dtype=torch.long)
    if spec["kind"] == "mul_noise":
        noise = 1.0 + torch.randn(ppi.size(0), 1, len(spec["positions"]), device=ppi.device) * spec["std"]
        view[:, channel : channel + 1, positions] = view[:, channel : channel + 1, positions] * noise
    elif spec["kind"] == "dropout":
        keep = (torch.rand(ppi.size(0), 1, len(spec["positions"]), device=ppi.device) > spec["prob"]).float()
        view[:, channel : channel + 1, positions] = view[:, channel : channel + 1, positions] * keep
    else:
        raise ValueError(f"Unsupported augmentation kind: {spec['kind']}")
    return view


@torch.no_grad()
def evaluate_setting(model, loader, device, spec, num_views, include_raw):
    all_labels = []
    all_preds = []

    for batch in tqdm(loader, desc=spec["name"]):
        ppi = batch["ppi"].to(device)
        labels = batch["label"].to(device)
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)

        probs_sum = torch.zeros(ppi.size(0), model.num_classes, device=device)
        n = 0
        if include_raw or spec["kind"] == "raw":
            probs_sum += F.softmax(model(ppi, flow_stats), dim=1)
            n += 1

        if spec["kind"] != "raw":
            for _ in range(num_views):
                view = augment_ppi(ppi, spec)
                probs_sum += F.softmax(model(view, flow_stats), dim=1)
                n += 1

        probs = probs_sum / max(n, 1)
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_labels, all_preds)


def write_report(rows, output_path):
    by_period = {}
    for row in rows:
        by_period.setdefault(row["period"], []).append(row)

    lines = [
        "# QUIC22 Targeted Channel Augmentation Summary",
        "",
        "This is frozen-model test-time augmentation averaging; no model parameters are updated.",
        "",
        "## Best Setting By Period",
        "",
        "| period | raw macro-F1 | best setting | best macro-F1 | delta |",
        "|---|---:|---|---:|---:|",
    ]
    for period in sorted(by_period):
        period_rows = by_period[period]
        raw = next(row for row in period_rows if row["setting"] == "raw")
        best = max((row for row in period_rows if row["setting"] != "raw"), key=lambda r: r["macro_f1"])
        lines.append(
            f"| {period} | {raw['macro_f1']:.4f} | {best['setting']} | "
            f"{best['macro_f1']:.4f} | {best['delta_macro_f1_vs_raw']:+.4f} |"
        )

    lines += [
        "",
        "## All Settings",
        "",
        "| period | setting | accuracy | macro-F1 | delta macro-F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["period"], r["setting"])):
        lines.append(
            f"| {row['period']} | {row['setting']} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['delta_macro_f1_vs_raw']:+.4f} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "- Positive deltas indicate that robustness to that channel perturbation helps target-period prediction.",
        "- If broad packet/channel perturbations hurt but one targeted channel helps, QUIC drift should be handled with channel-specific augmentation rather than generic TTA.",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_quic22.yaml")
    parser.add_argument("--checkpoint", default="outputs/quic22_cnn/best_model.pt")
    parser.add_argument("--periods", nargs="*", default=["W-2022-46", "W-2022-47"])
    parser.add_argument("--settings", nargs="*", default=[
        "raw",
        "size_noise_0.02",
        "size_noise_0.05",
        "ipt_noise_0.05",
        "ipt_noise_0.10",
        "direction_dropout_0.02",
        "direction_front_dropout_0.02",
        "direction_front_dropout_0.05",
        "packet_mask_0.02",
    ])
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--include-raw", action="store_true", default=True)
    parser.add_argument("--output-dir", default="outputs/quic22_channel_augmentation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    selected_periods = set(args.periods)
    specs = [parse_setting(setting) for setting in args.settings]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model, num_classes, _ = load_source_model(args.checkpoint, device)
    model.num_classes = num_classes
    print(f"Num classes: {num_classes}")

    loaders, _ = build_sequential_test_loaders(cfg["data"])
    loaders = [(period, loader) for period, loader in loaders if period in selected_periods]
    if not loaders:
        raise RuntimeError(f"No selected periods found: {args.periods}")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for period, loader in loaders:
        print(f"\n=== Period: {period} ===")
        raw_metric = None
        for spec in specs:
            metrics = evaluate_setting(
                model=model,
                loader=loader,
                device=device,
                spec=spec,
                num_views=args.num_views,
                include_raw=args.include_raw,
            )
            if spec["name"] == "raw":
                raw_metric = metrics
            if raw_metric is None:
                delta_f1 = 0.0
                delta_acc = 0.0
            else:
                delta_f1 = metrics["macro_f1"] - raw_metric["macro_f1"]
                delta_acc = metrics["accuracy"] - raw_metric["accuracy"]
            row = {
                "period": period,
                "setting": spec["name"],
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "delta_macro_f1_vs_raw": float(delta_f1),
                "delta_accuracy_vs_raw": float(delta_acc),
                "num_views": args.num_views,
                "include_raw": int(args.include_raw),
            }
            rows.append(row)
            print(
                f"{spec['name']:<30} acc={row['accuracy']:.4f} "
                f"macro={row['macro_f1']:.4f} delta={row['delta_macro_f1_vs_raw']:+.4f}"
            )

    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": args.config, "checkpoint": args.checkpoint, "rows": rows}, f, indent=2)

    report_path = os.path.join(args.output_dir, "quic22_channel_augmentation_report.md")
    write_report(rows, report_path)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    import multiprocessing as _mp

    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()

