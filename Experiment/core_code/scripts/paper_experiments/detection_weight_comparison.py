"""
Detection weight comparison (Paper experiment 3).

Compares manual weights vs equal weights vs adaptive weights for
the unsupervised collapse detection module.

Adaptive weights: optimize on M7+M9 (early drift periods), test on M12.

Usage from Experiment/core_code/:
    python scripts/paper_experiments/detection_weight_comparison.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/paper_experiments/detection_weights
"""
import argparse
import csv
import json
import os
import sys
from itertools import product

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(PARENT_DIR))

import prototype_recalibration_tls22 as proto
from unsupervised_collapse_detection import (
    compute_per_class_signals,
    detect_collapse_candidates,
)


def get_actual_collapsed(labels, preds, num_classes, recall_threshold=0.1, min_support=50):
    collapsed = []
    for c in range(num_classes):
        mask = labels == c
        support = int(mask.sum())
        if support >= min_support:
            recall = float((preds[mask] == c).sum()) / support
            if recall < recall_threshold:
                collapsed.append(c)
    return set(collapsed)


def evaluate_detection(collapse_cands, actual_set, top_k=20):
    detected = set(r["class"] for r in collapse_cands[:top_k])
    tp = len(detected & actual_set)
    precision = tp / max(len(detected), 1)
    recall = tp / max(len(actual_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_detected": len(detected), "n_actual": len(actual_set), "tp": tp}


def run_detection(model, eval_cfg, ref_period, tgt_period, device, num_classes, weights, top_k=20):
    """Run detection pipeline for one target period with given weights."""
    ref_loader, _ = proto.make_test_loader(eval_cfg, ref_period)
    ref_out = proto.collect_outputs(model, ref_loader, device, desc=f"Ref {ref_period}")
    ref_preds = ref_out["logits"].argmax(dim=1).numpy()

    tgt_loader, _ = proto.make_test_loader(eval_cfg, tgt_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device, desc=f"Tgt {tgt_period}")
    tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()
    tgt_labels = tgt_out["labels"]

    signals = compute_per_class_signals(
        ref_out["features"], ref_preds, ref_out["logits"],
        tgt_out["features"], tgt_preds, tgt_out["logits"],
        num_classes,
    )
    collapse_cands, _ = detect_collapse_candidates(signals, top_k=top_k, weights=weights)
    actual = get_actual_collapsed(tgt_labels, tgt_preds, num_classes)

    return evaluate_detection(collapse_cands, actual, top_k), signals, actual


def optimize_weights_grid(model, eval_cfg, ref_period, train_periods, device, num_classes, top_k=20):
    """Grid search for best weights on training periods (M7, M9)."""
    # Precompute signals for training periods
    ref_loader, _ = proto.make_test_loader(eval_cfg, ref_period)
    ref_out = proto.collect_outputs(model, ref_loader, device, desc=f"Ref {ref_period}")
    ref_preds = ref_out["logits"].argmax(dim=1).numpy()

    period_data = []
    for period in train_periods:
        tgt_loader, _ = proto.make_test_loader(eval_cfg, period)
        tgt_out = proto.collect_outputs(model, tgt_loader, device, desc=f"Tgt {period}")
        tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()
        tgt_labels = tgt_out["labels"]

        signals = compute_per_class_signals(
            ref_out["features"], ref_preds, ref_out["logits"],
            tgt_out["features"], tgt_preds, tgt_out["logits"],
            num_classes,
        )
        actual = get_actual_collapsed(tgt_labels, tgt_preds, num_classes)
        period_data.append((signals, actual))

    # Grid search over weights (coarse: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_f1 = -1
    best_weights = None

    candidates = []
    for w1, w2, w3, w4 in product(grid, repeat=4):
        w5 = 1.0 - w1 - w2 - w3 - w4
        if w5 < -0.01 or w5 > 0.51:
            continue
        w5 = max(0, w5)
        weights = [w1, w2, w3, w4, w5]
        total = sum(weights)
        if total < 0.01:
            continue
        weights = [w / total for w in weights]
        candidates.append(weights)

    print(f"  Grid search over {len(candidates)} weight combinations...")
    for weights in candidates:
        total_f1 = 0
        for signals, actual in period_data:
            cands, _ = detect_collapse_candidates(signals, top_k=top_k, weights=weights)
            ev = evaluate_detection(cands, actual, top_k)
            total_f1 += ev["f1"]
        avg_f1 = total_f1 / len(period_data)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_weights = weights

    return best_weights, best_f1


def main():
    parser = argparse.ArgumentParser(description="Detection weight comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    weight_configs = {
        "manual": [0.40, 0.20, 0.15, 0.15, 0.10],
        "equal": [0.20, 0.20, 0.20, 0.20, 0.20],
    }

    # Step 1: Find adaptive weights on M7+M9
    print("=== Finding adaptive weights on M7+M9 ===")
    train_periods = ["M-2022-7", "M-2022-9"]
    adaptive_weights, adaptive_train_f1 = optimize_weights_grid(
        model, eval_cfg, args.reference_period, train_periods, device, num_classes, args.top_k)
    weight_configs["adaptive"] = adaptive_weights
    print(f"  Adaptive weights: {[f'{w:.2f}' for w in adaptive_weights]} (train F1={adaptive_train_f1:.3f})")

    # Step 2: Evaluate all weight configs on M7, M9, M11, M12
    test_periods = ["M-2022-7", "M-2022-9", "M-2022-11", "M-2022-12"]
    results = []

    print(f"\n=== Detection results (top-{args.top_k}) ===")
    print(f"{'Config':<12} {'Weights':<35} {'Period':<12} {'P':>6} {'R':>6} {'F1':>6} {'GT':>4}")
    print("-" * 90)

    for name, weights in weight_configs.items():
        for period in test_periods:
            ev, _, _ = run_detection(
                model, eval_cfg, args.reference_period, period, device, num_classes,
                weights, args.top_k)
            row = {
                "weight_config": name,
                "weights": str([f"{w:.2f}" for w in weights]),
                "period": period,
                **ev,
            }
            results.append(row)
            w_str = ",".join(f"{w:.2f}" for w in weights)
            print(f"  {name:<10} [{w_str}] {period:<10} "
                  f"{ev['precision']:>5.3f} {ev['recall']:>5.3f} {ev['f1']:>5.3f} {ev['n_actual']:>3d}")

    write_csv = lambda path, rows: None  # use our own
    fieldnames = list(results[0].keys())
    with open(os.path.join(args.output_dir, "weight_comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "weight_configs": {k: v for k, v in weight_configs.items()},
        "adaptive_train_periods": train_periods,
        "adaptive_train_f1": adaptive_train_f1,
        "test_periods": test_periods,
        "top_k": args.top_k,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
