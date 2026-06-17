"""
Weight sensitivity analysis for unsupervised 5-signal collapse detection.

Tests robustness of the default weight vector w = (0.40, 0.20, 0.15, 0.15, 0.10)
by perturbing each weight +/-50% and sampling 20 random Dirichlet weight vectors.
For each configuration, computes detection precision/recall/F1 against
ground-truth collapse classes (recall < 0.1).

Output: CSV with columns [weight_config, w1, w2, w3, w4, w5, precision, recall, f1]

Usage from Experiment/core_code/:
    python scripts/run_weight_sensitivity.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/weight_sensitivity
"""
import argparse
import csv
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
    DEFAULT_WEIGHTS,
)

SIGNAL_NAMES = ["count_drop", "fd", "margin_drop", "conf_drop", "entropy_rise"]


def normalize_weights(w):
    """Normalize weight vector to sum to 1."""
    total = sum(w)
    if total <= 0:
        return [1.0 / len(w)] * len(w)
    return [x / total for x in w]


def generate_perturbation_configs(base_weights, perturbation=0.5):
    """Generate configs by perturbing each weight +/-50%."""
    configs = []
    for i in range(len(base_weights)):
        for direction, label in [(1.0, "up"), (-1.0, "down")]:
            w = list(base_weights)
            w[i] = max(0.0, w[i] * (1.0 + direction * perturbation))
            w = normalize_weights(w)
            name = f"w{i+1}_{SIGNAL_NAMES[i]}_{label}{int(perturbation*100)}pct"
            configs.append({"name": name, "weights": w})
    return configs


def generate_dirichlet_configs(n_samples, seed, alpha_scale=5.0):
    """Generate random Dirichlet-sampled weight vectors."""
    rng = np.random.default_rng(seed)
    base = np.array(DEFAULT_WEIGHTS)
    alpha = base * alpha_scale / base.sum()
    alpha = np.maximum(alpha, 0.1)
    configs = []
    for i in range(n_samples):
        w = rng.dirichlet(alpha).tolist()
        configs.append({"name": f"dirichlet_{i:02d}", "weights": w})
    return configs


def evaluate_detection(collapse_cands, actual_collapsed):
    """Compute precision, recall, F1 for detected vs actual collapse."""
    detected = set(r["class"] for r in collapse_cands)
    actual = set(actual_collapsed)
    tp = detected & actual
    precision = len(tp) / max(len(detected), 1)
    recall = len(tp) / max(len(actual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_detected": len(detected),
        "n_tp": len(tp),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Weight sensitivity analysis for unsupervised collapse detection."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.12)
    parser.add_argument("--n-dirichlet", type=int, default=20,
                        help="Number of random Dirichlet weight samples")
    parser.add_argument("--perturbation", type=float, default=0.5,
                        help="Perturbation magnitude (fraction, default 0.50 = +/-50%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    # Collect reference and target features (done once)
    print(f"Loading reference period: {args.reference_period}")
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_out = proto.collect_outputs(model, ref_loader, device,
                                    desc=f"Ref {args.reference_period}")
    ref_preds = ref_out["logits"].argmax(dim=1).numpy()

    print(f"Loading target period: {args.target_period}")
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device,
                                    desc=f"Tgt {args.target_period}")
    tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()
    tgt_labels = tgt_out["labels"]

    print(f"Reference: {len(ref_preds)} samples, Target: {len(tgt_preds)} samples")

    # Compute per-class signals (weight-independent, done once)
    print("Computing per-class collapse signals...")
    fd_results = compute_per_class_signals(
        ref_out["features"], ref_preds, ref_out["logits"],
        tgt_out["features"], tgt_preds, tgt_out["logits"],
        num_classes, args.min_samples,
    )

    # Ground truth: actual collapsed classes
    actual_collapsed = []
    for c in range(num_classes):
        mask = tgt_labels == c
        if mask.sum() > 0:
            recall_c = float((tgt_preds[mask] == c).sum()) / float(mask.sum())
            if recall_c < 0.1:
                actual_collapsed.append(c)

    print(f"Actual collapsed classes ({len(actual_collapsed)}): {sorted(actual_collapsed)}")

    # Build weight configurations
    all_configs = []

    # 1. Default weights
    all_configs.append({
        "name": "default",
        "weights": list(DEFAULT_WEIGHTS),
    })

    # 2. Perturbation configs (+/-50% each weight)
    all_configs.extend(generate_perturbation_configs(DEFAULT_WEIGHTS, args.perturbation))

    # 3. Dirichlet-sampled configs
    all_configs.extend(generate_dirichlet_configs(args.n_dirichlet, args.seed))

    print(f"\nEvaluating {len(all_configs)} weight configurations...")
    print(f"{'Config':<40} {'w1':>5} {'w2':>5} {'w3':>5} {'w4':>5} {'w5':>5} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 95)

    csv_rows = []
    for cfg in all_configs:
        w = cfg["weights"]
        collapse_cands, _ = detect_collapse_candidates(
            fd_results,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            weights=w,
        )
        metrics = evaluate_detection(collapse_cands, actual_collapsed)

        row = {
            "weight_config": cfg["name"],
            "w1": round(w[0], 4),
            "w2": round(w[1], 4),
            "w3": round(w[2], 4),
            "w4": round(w[3], 4),
            "w5": round(w[4], 4),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "n_detected": metrics["n_detected"],
            "n_tp": metrics["n_tp"],
        }
        csv_rows.append(row)

        print(f"{cfg['name']:<40} {w[0]:>5.2f} {w[1]:>5.2f} {w[2]:>5.2f} "
              f"{w[3]:>5.2f} {w[4]:>5.2f} {metrics['precision']:>6.3f} "
              f"{metrics['recall']:>6.3f} {metrics['f1']:>6.3f}")

    # Write CSV
    csv_path = os.path.join(args.output_dir, "weight_sensitivity.csv")
    fieldnames = ["weight_config", "w1", "w2", "w3", "w4", "w5",
                  "precision", "recall", "f1", "n_detected", "n_tp"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary statistics
    f1_values = [r["f1"] for r in csv_rows]
    default_f1 = csv_rows[0]["f1"]
    perturbation_f1 = [r["f1"] for r in csv_rows[1:11]]
    dirichlet_f1 = [r["f1"] for r in csv_rows[11:]]

    summary = {
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "n_configs": len(all_configs),
        "n_actual_collapsed": len(actual_collapsed),
        "actual_collapsed": sorted(actual_collapsed),
        "default_weights": list(DEFAULT_WEIGHTS),
        "default_f1": default_f1,
        "perturbation_f1_mean": round(float(np.mean(perturbation_f1)), 4),
        "perturbation_f1_std": round(float(np.std(perturbation_f1)), 4),
        "perturbation_f1_min": round(float(np.min(perturbation_f1)), 4),
        "perturbation_f1_max": round(float(np.max(perturbation_f1)), 4),
        "dirichlet_f1_mean": round(float(np.mean(dirichlet_f1)), 4) if dirichlet_f1 else None,
        "dirichlet_f1_std": round(float(np.std(dirichlet_f1)), 4) if dirichlet_f1 else None,
        "dirichlet_f1_min": round(float(np.min(dirichlet_f1)), 4) if dirichlet_f1 else None,
        "dirichlet_f1_max": round(float(np.max(dirichlet_f1)), 4) if dirichlet_f1 else None,
        "overall_f1_mean": round(float(np.mean(f1_values)), 4),
        "overall_f1_std": round(float(np.std(f1_values)), 4),
    }
    summary_path = os.path.join(args.output_dir, "weight_sensitivity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Weight Sensitivity Summary ===")
    print(f"Default F1:            {default_f1:.3f}")
    print(f"Perturbation (+/-50%): {np.mean(perturbation_f1):.3f} +/- {np.std(perturbation_f1):.3f} "
          f"[{np.min(perturbation_f1):.3f}, {np.max(perturbation_f1):.3f}]")
    if dirichlet_f1:
        print(f"Dirichlet (n={len(dirichlet_f1)}):      {np.mean(dirichlet_f1):.3f} +/- {np.std(dirichlet_f1):.3f} "
              f"[{np.min(dirichlet_f1):.3f}, {np.max(dirichlet_f1):.3f}]")
    print(f"\nSaved: {csv_path}")
    print(f"       {summary_path}")


if __name__ == "__main__":
    main()
