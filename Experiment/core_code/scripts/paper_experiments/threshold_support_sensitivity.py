"""
Threshold & support sensitivity analysis (Paper experiments 1 & 2).

Runs CARE once, then re-evaluates with different collapse definitions:
  - τ ∈ {0.05, 0.10, 0.15}: recall threshold for "collapsed"
  - support ∈ {50, 100, 200}: minimum samples to be considered

This avoids re-running the repair pipeline for each configuration.

Usage from Experiment/core_code/:
    python scripts/paper_experiments/threshold_support_sensitivity.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/paper_experiments/threshold_sensitivity
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(PARENT_DIR))

import prototype_recalibration_tls22 as proto
from collapse_active_maintenance_tls22 import (
    DEFAULT_ABSORBER_CLASSES,
    DEFAULT_COLLAPSE_CLASSES,
    DEFAULT_STABLE_CLASSES,
    build_head_training_set,
    fit_head,
    parse_int_list,
    predict_head,
    prototype_distance_signals,
    sample_replay_indices,
    select_indices,
    write_csv,
)


def find_collapse_classes(labels, preds, num_classes, recall_threshold, min_support):
    """Find classes with recall < threshold and support >= min_support."""
    collapsed = []
    class_info = []
    for c in range(num_classes):
        mask = labels == c
        support = int(mask.sum())
        if support < min_support:
            continue
        recall = float((preds[mask] == c).sum()) / support if support > 0 else 0.0
        class_info.append({"class": c, "recall": recall, "support": support})
        if recall < recall_threshold:
            collapsed.append(c)
    return collapsed, class_info


def evaluate_with_classes(labels, preds, collapse_classes, stable_classes, recall_threshold=0.1):
    """Compute macro-F1 for collapse/stable class subsets.

    Uses the same method as the main pipeline: compute classification_report
    on ALL samples (full confusion matrix), then extract per-class F1 for the
    target class group and average. This correctly accounts for cross-class
    false positives.
    """
    report = proto.compute_metrics(labels, preds)["classification_report"]

    result = {}
    result["overall_macro_f1"] = float(report.get("macro avg", {}).get("f1-score", 0))

    if collapse_classes:
        f1_vals = []
        collapsed = 0
        for c in collapse_classes:
            item = report.get(str(c), {})
            f1_vals.append(float(item.get("f1-score", 0.0)))
            recall = float(item.get("recall", 0.0))
            if recall < recall_threshold:
                collapsed += 1
        result["collapse_macro_f1"] = float(np.mean(f1_vals)) if f1_vals else 0.0
        result["collapse_count"] = collapsed
        result["num_collapse_classes"] = len(collapse_classes)

    if stable_classes:
        f1_vals = []
        for c in stable_classes:
            item = report.get(str(c), {})
            f1_vals.append(float(item.get("f1-score", 0.0)))
        result["stable_macro_f1"] = float(np.mean(f1_vals)) if f1_vals else 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="Threshold & support sensitivity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-per-class", type=int, default=5)
    parser.add_argument("--target-repeat", type=int, default=2)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--recall-thresholds", default="0.05,0.10,0.15")
    parser.add_argument("--support-thresholds", default="50,100,200")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    recall_thresholds = [float(x) for x in args.recall_thresholds.split(",")]
    support_thresholds = [int(x) for x in args.support_thresholds.split(",")]
    stable_classes = parse_int_list(None, DEFAULT_STABLE_CLASSES)

    print(f"=== Threshold & Support Sensitivity ===")
    print(f"Recall thresholds: {recall_thresholds}")
    print(f"Support thresholds: {support_thresholds}")

    # Collect reference
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(model, ref_loader, device, desc=f"Ref {args.reference_period}")
    prototypes, _, valid_mask = proto.build_prototypes(
        ref_outputs["features"], ref_outputs["labels"], num_classes, 1)

    # Collect target
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_outputs = proto.collect_outputs(model, tgt_loader, device, desc=f"Tgt {args.target_period}")
    features = tgt_outputs["features"]
    logits = tgt_outputs["logits"]
    labels = tgt_outputs["labels"]
    static_preds = logits.argmax(dim=1).numpy()

    # Step 1: Find collapse classes for each (τ, support) combination
    print(f"\n--- Collapse class discovery ---")
    configs = {}
    for tau in recall_thresholds:
        for min_sup in support_thresholds:
            collapsed, info = find_collapse_classes(labels, static_preds, num_classes, tau, min_sup)
            key = f"tau{tau}_sup{min_sup}"
            configs[key] = {
                "tau": tau, "min_support": min_sup,
                "collapse_classes": collapsed, "n_collapsed": len(collapsed),
            }
            print(f"  τ={tau:.2f}, support≥{min_sup}: {len(collapsed)} collapsed classes → {collapsed}")

    # Step 2: Run CARE once with margin selection
    nearest_distance, nearest_proto = prototype_distance_signals(features, prototypes, valid_mask)
    ref_preds = ref_outputs["logits"].argmax(dim=1)
    ref_pred_counts = torch.zeros(num_classes)
    for c in range(num_classes):
        ref_pred_counts[c] = (ref_preds == c).sum()

    replay_classes = list(range(num_classes))
    replay_idx = sample_replay_indices(
        ref_outputs["labels"], replay_classes, args.replay_per_class, args.seed + 10007)

    idx = select_indices(
        "margin", logits, labels, args.budget, num_classes,
        DEFAULT_COLLAPSE_CLASSES, DEFAULT_ABSORBER_CLASSES, args.seed,
        nearest_distance=nearest_distance, nearest_proto=nearest_proto,
        features=features, prototypes=prototypes, ref_pred_counts=ref_pred_counts,
    )
    selected_labels = labels[idx.numpy()]
    train_features, train_labels_t = build_head_training_set(
        features[idx], selected_labels,
        ref_outputs["features"], ref_outputs["labels"],
        replay_idx, args.target_repeat,
    )

    replay_features = ref_outputs["features"][replay_idx] if replay_idx.numel() > 0 else None
    replay_logits = None
    if replay_features is not None and args.distill_weight > 0:
        with torch.no_grad():
            model.cls_head.eval()
            replay_logits = model.cls_head(replay_features.to(device)).cpu()

    head = fit_head(
        model, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device,
        distill_features=replay_features, distill_logits=replay_logits,
        distill_weight=args.distill_weight, distill_temperature=args.distill_temperature,
        seed=args.seed,
    )
    care_preds = predict_head(head, features, device)

    # Step 3: Evaluate with each collapse definition
    eval_mask = np.ones(len(labels), dtype=bool)
    eval_mask[idx.numpy()] = False
    strict_labels = labels[eval_mask]
    strict_static = static_preds[eval_mask]
    strict_care = care_preds[eval_mask]

    results = []
    print(f"\n--- Results ---")
    print(f"{'Config':<20} {'#Col':>5} {'Static Col-F1':>14} {'CARE Col-F1':>13} {'Δ':>8} {'Static Macro':>13} {'CARE Macro':>11}")
    print("-" * 95)

    for key, cfg in sorted(configs.items()):
        cc = cfg["collapse_classes"]
        tau = cfg["tau"]
        static_eval = evaluate_with_classes(strict_labels, strict_static, cc, stable_classes, recall_threshold=tau)
        care_eval = evaluate_with_classes(strict_labels, strict_care, cc, stable_classes, recall_threshold=tau)
        delta_col = care_eval.get("collapse_macro_f1", 0) - static_eval.get("collapse_macro_f1", 0)

        row = {
            "config": key,
            "recall_threshold": cfg["tau"],
            "support_threshold": cfg["min_support"],
            "n_collapse_classes": cfg["n_collapsed"],
            "collapse_classes": str(cc),
            "static_overall_macro_f1": static_eval["overall_macro_f1"],
            "care_overall_macro_f1": care_eval["overall_macro_f1"],
            "static_collapse_f1": static_eval.get("collapse_macro_f1", ""),
            "care_collapse_f1": care_eval.get("collapse_macro_f1", ""),
            "delta_collapse_f1": delta_col,
            "static_stable_f1": static_eval.get("stable_macro_f1", ""),
            "care_stable_f1": care_eval.get("stable_macro_f1", ""),
            "care_collapsed_count": care_eval.get("collapse_count", ""),
        }
        results.append(row)

        sc = static_eval.get("collapse_macro_f1", 0)
        cc_f1 = care_eval.get("collapse_macro_f1", 0)
        print(f"  τ={cfg['tau']:.2f} sup≥{cfg['min_support']:<3d}  "
              f"{cfg['n_collapsed']:>3d}   "
              f"{sc:>13.4f}  {cc_f1:>12.4f}  {delta_col:>+7.4f}  "
              f"{static_eval['overall_macro_f1']:>12.4f}  {care_eval['overall_macro_f1']:>10.4f}")

    write_csv(os.path.join(args.output_dir, "sensitivity_results.csv"), results)

    # Save per-class static recalls for reference
    _, all_info = find_collapse_classes(labels, static_preds, num_classes, 1.0, 0)
    write_csv(os.path.join(args.output_dir, "per_class_static_recalls.csv"), all_info)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
