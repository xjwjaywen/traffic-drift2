"""
Temperature scaling / Platt scaling calibration baseline.

Learns an optimal temperature T on a validation period (M3 or M4) by
minimizing negative log-likelihood, then applies temperature scaling to
target-period (M12) predictions.  Reports per-class recall, macro-F1,
and collapse-class F1 before and after calibration.

This tests whether simple calibration can address concept drift -- the
hypothesis is that it cannot, because the issue is distributional shift
rather than miscalibrated softmax outputs.

Usage from Experiment/core_code/:
    python scripts/run_calibration_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/calibration_baseline
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto

DEFAULT_COLLAPSE_CLASSES = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
DEFAULT_STABLE_CLASSES = [
    8, 15, 44, 57, 59, 62, 64, 76, 94, 98,
    99, 107, 113, 119, 128, 130, 131, 132, 144, 145,
]


def parse_int_list(value, default=None):
    if value is None:
        return list(default or [])
    return [int(x) for x in str(value).replace(",", " ").split()]


def nll_with_temperature(T, logits, labels):
    """Compute negative log-likelihood after temperature scaling."""
    T = max(T, 0.01)
    scaled_logits = logits / T
    log_probs = F.log_softmax(scaled_logits, dim=1)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    nll = F.nll_loss(log_probs, labels_t).item()
    return nll


def learn_temperature(logits, labels, T_range=(0.1, 10.0)):
    """Find optimal temperature by minimizing NLL on validation set.

    Uses scipy scalar optimizer (bounded Brent) for a clean 1-D search.
    """
    def objective(T):
        return nll_with_temperature(T, logits, labels)

    result = minimize_scalar(objective, bounds=T_range, method="bounded",
                             options={"maxiter": 200, "xatol": 1e-4})
    return result.x, result.fun


def compute_per_class_recall(labels, preds, num_classes):
    """Compute recall per class."""
    recalls = {}
    for c in range(num_classes):
        mask = labels == c
        support = int(mask.sum())
        if support == 0:
            continue
        correct = int((preds[mask] == c).sum())
        recalls[c] = {"recall": correct / support, "support": support}
    return recalls


def compute_group_metrics(labels, preds, num_classes, class_group, group_name):
    """Compute macro-F1 for a subset of classes."""
    from sklearn.metrics import f1_score

    # Restrict to samples whose true label is in class_group
    group_set = set(class_group)
    mask = np.array([int(y) in group_set for y in labels], dtype=bool)
    if mask.sum() == 0:
        return {f"{group_name}_macro_f1": None, f"{group_name}_support": 0}

    group_labels = labels[mask]
    group_preds = preds[mask]

    present_classes = sorted(group_set & set(np.unique(group_labels)))
    if not present_classes:
        return {f"{group_name}_macro_f1": None, f"{group_name}_support": int(mask.sum())}

    f1 = f1_score(group_labels, group_preds, labels=present_classes,
                  average="macro", zero_division=0)
    return {
        f"{group_name}_macro_f1": round(float(f1), 4),
        f"{group_name}_support": int(mask.sum()),
    }


def compute_all_metrics(labels, preds, num_classes, collapse_classes, stable_classes):
    """Compute full set of metrics."""
    from sklearn.metrics import f1_score

    overall_macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    overall_accuracy = float((labels == preds).mean())

    collapse_metrics = compute_group_metrics(
        labels, preds, num_classes, collapse_classes, "collapse")
    stable_metrics = compute_group_metrics(
        labels, preds, num_classes, stable_classes, "stable")

    per_class = compute_per_class_recall(labels, preds, num_classes)
    collapsed_count = sum(
        1 for c in collapse_classes
        if c in per_class and per_class[c]["recall"] < 0.1
    )

    return {
        "overall_accuracy": round(float(overall_accuracy), 4),
        "overall_macro_f1": round(float(overall_macro_f1), 4),
        **collapse_metrics,
        **stable_metrics,
        "collapsed_count": collapsed_count,
        "total_collapse_classes": len(collapse_classes),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Temperature scaling calibration baseline for concept drift."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-period", default="M-2022-4",
                        help="Validation period for learning temperature (default: M-2022-4)")
    parser.add_argument("--target-period", default="M-2022-12",
                        help="Target period to evaluate (default: M-2022-12)")
    parser.add_argument("--collapse-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    collapse_classes = parse_int_list(args.collapse_classes, DEFAULT_COLLAPSE_CLASSES)
    stable_classes = parse_int_list(args.stable_classes, DEFAULT_STABLE_CLASSES)

    # Step 1: Load validation period and learn temperature
    print(f"=== Step 1: Learn temperature on {args.val_period} ===")
    val_loader, _ = proto.make_test_loader(eval_cfg, args.val_period)
    val_out = proto.collect_outputs(model, val_loader, device,
                                    desc=f"Val {args.val_period}")
    val_logits = val_out["logits"]
    val_labels = val_out["labels"]

    T_star, nll_star = learn_temperature(val_logits, val_labels)
    nll_T1 = nll_with_temperature(1.0, val_logits, val_labels)
    print(f"Optimal temperature: T* = {T_star:.4f}")
    print(f"Val NLL: T=1.0 -> {nll_T1:.4f},  T*={T_star:.4f} -> {nll_star:.4f}")

    # Step 2: Evaluate on target period
    print(f"\n=== Step 2: Evaluate on {args.target_period} ===")
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device,
                                    desc=f"Tgt {args.target_period}")
    tgt_logits = tgt_out["logits"]
    tgt_labels = tgt_out["labels"]

    # Before calibration (T=1)
    preds_before = tgt_logits.argmax(dim=1).numpy()
    metrics_before = compute_all_metrics(
        tgt_labels, preds_before, num_classes, collapse_classes, stable_classes
    )

    # After temperature scaling
    scaled_logits = tgt_logits / T_star
    preds_after = scaled_logits.argmax(dim=1).numpy()
    metrics_after = compute_all_metrics(
        tgt_labels, preds_after, num_classes, collapse_classes, stable_classes
    )

    # Per-class recall comparison for collapse classes
    recall_before = compute_per_class_recall(tgt_labels, preds_before, num_classes)
    recall_after = compute_per_class_recall(tgt_labels, preds_after, num_classes)

    collapse_recall_comparison = {}
    for c in collapse_classes:
        rb = recall_before.get(c, {}).get("recall", 0.0)
        ra = recall_after.get(c, {}).get("recall", 0.0)
        sup = recall_before.get(c, {}).get("support", 0)
        collapse_recall_comparison[str(c)] = {
            "before": round(rb, 4),
            "after": round(ra, 4),
            "delta": round(ra - rb, 4),
            "support": sup,
        }

    # Note: for temperature scaling, argmax is invariant to T (monotone transform)
    # so predictions only change at decision boundaries due to numerical effects.
    # The key insight is that calibration DOES NOT change predictions.
    preds_changed = int((preds_before != preds_after).sum())

    print(f"\n=== Results ===")
    print(f"Temperature: T* = {T_star:.4f}")
    print(f"Predictions changed: {preds_changed}/{len(preds_before)}")
    print(f"\n{'Metric':<25} {'Before (T=1)':>15} {'After (T*)':>15} {'Delta':>10}")
    print("-" * 70)
    for key in ["overall_accuracy", "overall_macro_f1", "collapse_macro_f1",
                "stable_macro_f1", "collapsed_count"]:
        before = metrics_before.get(key)
        after = metrics_after.get(key)
        if before is not None and after is not None:
            delta = after - before
            print(f"{key:<25} {before:>15.4f} {after:>15.4f} {delta:>+10.4f}")

    print(f"\nPer-class recall (collapse classes):")
    print(f"{'Class':>6} {'Before':>8} {'After':>8} {'Delta':>8} {'Support':>8}")
    for c in collapse_classes:
        info = collapse_recall_comparison.get(str(c), {})
        print(f"{c:>6} {info.get('before', 0):>8.3f} {info.get('after', 0):>8.3f} "
              f"{info.get('delta', 0):>+8.3f} {info.get('support', 0):>8}")

    # Save results
    results = {
        "val_period": args.val_period,
        "target_period": args.target_period,
        "num_classes": num_classes,
        "optimal_temperature": round(T_star, 6),
        "val_nll_T1": round(nll_T1, 6),
        "val_nll_Tstar": round(nll_star, 6),
        "predictions_changed": preds_changed,
        "total_samples": len(preds_before),
        "before": metrics_before,
        "after": metrics_after,
        "collapse_recall_comparison": collapse_recall_comparison,
        "note": (
            "Temperature scaling preserves argmax ordering (monotone transform), "
            "so predictions are nearly identical. This confirms calibration alone "
            "cannot address concept drift / class collapse."
        ),
    }
    out_path = os.path.join(args.output_dir, "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
