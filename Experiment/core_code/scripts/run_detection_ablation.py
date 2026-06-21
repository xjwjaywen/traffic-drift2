"""
Detection signal ablation: reproduce Table 4 (unsupervised collapse detection).

Runs the same detection pipeline with different signal combinations and
saves per-config results as JSON artifacts for reproducibility.

Usage from Experiment/core_code/:
    python scripts/run_detection_ablation.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --output-dir outputs/detection_ablation
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto
from unsupervised_collapse_detection import (
    compute_per_class_signals,
    detect_collapse_candidates,
)

ABLATION_CONFIGS = [
    {
        "name": "fd_only",
        "label": "FD only",
        "weights": [0.0, 1.0, 0.0, 0.0, 0.0],
    },
    {
        "name": "count_fd",
        "label": "Count + FD",
        "weights": [0.50, 0.50, 0.0, 0.0, 0.0],
    },
    {
        "name": "4signal",
        "label": "4-signal composite (count+FD+margin+conf)",
        "weights": [0.40, 0.25, 0.20, 0.15, 0.0],
    },
    {
        "name": "5signal_full",
        "label": "+ entropy (5-signal, default)",
        "weights": [0.40, 0.20, 0.15, 0.15, 0.10],
    },
]


def evaluate_detection(collapse_cands, actual_collapsed, num_classes):
    detected_classes = set(r["class"] for r in collapse_cands)
    actual_set = set(actual_collapsed)
    tp = detected_classes & actual_set
    precision = len(tp) / max(len(detected_classes), 1)
    recall = len(tp) / max(len(actual_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "n_detected": len(detected_classes),
        "n_actual": len(actual_set),
        "n_tp": len(tp),
        "detected": sorted(detected_classes),
        "true_positives": sorted(tp),
        "false_positives": sorted(detected_classes - actual_set),
        "missed": sorted(actual_set - detected_classes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.12)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_out = proto.collect_outputs(model, ref_loader, device,
                                    desc=f"Ref {args.reference_period}")
    ref_preds = ref_out["logits"].argmax(dim=1).numpy()

    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device,
                                    desc=f"Tgt {args.target_period}")
    tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()
    tgt_labels = tgt_out["labels"]

    fd_results = compute_per_class_signals(
        ref_out["features"], ref_preds, ref_out["logits"],
        tgt_out["features"], tgt_preds, tgt_out["logits"],
        num_classes, args.min_samples,
    )

    # Ground truth: recall < 0.1 AND support >= 50 (matches paper's 12-class set)
    actual_collapsed = []
    for c in range(num_classes):
        mask = tgt_labels == c
        support = int(mask.sum())
        if support >= 50:
            recall = float((tgt_preds[mask] == c).sum()) / float(support)
            if recall < 0.1:
                actual_collapsed.append(c)

    print(f"Reference: {args.reference_period}, Target: {args.target_period}")
    print(f"Actual collapsed ({len(actual_collapsed)}): {actual_collapsed}")
    print()

    all_results = []
    for cfg in ABLATION_CONFIGS:
        collapse_cands, _ = detect_collapse_candidates(
            fd_results, top_k=args.top_k,
            score_threshold=args.score_threshold,
            weights=cfg["weights"],
        )
        metrics = evaluate_detection(collapse_cands, actual_collapsed, num_classes)
        result = {
            "config_name": cfg["name"],
            "label": cfg["label"],
            "weights": cfg["weights"],
            "score_threshold": args.score_threshold,
            **metrics,
        }
        all_results.append(result)

        print(f"{cfg['label']:<45}  P={metrics['precision']:.2f}  "
              f"R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  "
              f"({metrics['n_tp']}/{metrics['n_actual']} detected, "
              f"{metrics['n_detected']-metrics['n_tp']} FP)")

        per_cfg_path = os.path.join(args.output_dir, f"{cfg['name']}.json")
        with open(per_cfg_path, "w") as f:
            json.dump(result, f, indent=2)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "min_samples": args.min_samples,
        "actual_collapsed": actual_collapsed,
        "top_k": args.top_k,
        "score_threshold": args.score_threshold,
        "ablation_results": all_results,
    }
    with open(os.path.join(args.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(all_results)} configs to {args.output_dir}/")
    print("\nTable for paper:")
    print(f"{'Signal Combination':<35} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['label']:<35} {r['precision']:>10.2f} {r['recall']:>8.2f} {r['f1']:>6.2f}")


if __name__ == "__main__":
    main()
