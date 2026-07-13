"""
Self-Evolving baseline: pseudo-label fine-tuning without human annotations.

Adapts the core idea from Chen et al. 2025 (arXiv:2501.04246) to our setting:
  1. Run frozen source model on target-period data
  2. Select high-confidence predictions as "silver samples" (pseudo-labels)
  3. Fine-tune classification head on pseudo-labeled samples only
  4. Evaluate on all target samples

Key differences from CARE:
  - No human labels at all (zero annotation cost)
  - Selection by model confidence, not active learning
  - No source replay or knowledge distillation (vanilla version)
  - Optional: +replay and +KD variants for ablation

Usage from Experiment/core_code/:
    python scripts/baselines/self_evolving_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/baselines/self_evolving/seed_0 \
      --seed 0
"""
import argparse
import copy
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(PARENT_DIR))

import prototype_recalibration_tls22 as proto
from collapse_active_maintenance_tls22 import (
    DEFAULT_ABSORBER_CLASSES,
    DEFAULT_COLLAPSE_CLASSES,
    DEFAULT_STABLE_CLASSES,
    fit_head,
    predict_head,
    sample_replay_indices,
    summarize,
    write_csv,
)


def select_silver_samples(logits, threshold):
    """Select samples whose max softmax probability exceeds threshold.

    Returns indices and their pseudo-labels (argmax predictions).
    """
    probs = F.softmax(logits, dim=1)
    max_probs, pred_classes = probs.max(dim=1)
    mask = max_probs >= threshold
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    pseudo_labels = pred_classes[indices].numpy()
    return indices, pseudo_labels, max_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.90,0.95,0.99,0.997",
        help="Confidence thresholds to sweep (comma-separated).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--replay-mode",
        choices=["none", "all"],
        default="none",
        help="Optional source replay: 'none' (pure self-evolving) or 'all'.",
    )
    parser.add_argument("--replay-per-class", type=int, default=0)
    parser.add_argument(
        "--replay-distill-weight",
        type=float,
        default=0.0,
        help="KL distillation weight on replay samples.",
    )
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    collapse_classes = DEFAULT_COLLAPSE_CLASSES
    stable_classes = DEFAULT_STABLE_CLASSES
    thresholds_list = [float(t) for t in args.thresholds.split(",")]
    thresholds = {
        "collapse": args.collapse_recall_threshold,
        "severe": args.severe_recall_threshold,
    }

    print(f"Device: {device}")
    print(f"Target period: {args.target_period}")
    print(f"Num classes: {num_classes}")
    print(f"Confidence thresholds: {thresholds_list}")
    print(f"Replay mode: {args.replay_mode}, per_class: {args.replay_per_class}")

    # --- Prepare replay if requested ---
    replay_features = None
    replay_logits = None
    replay_labels_np = None
    replay_count = 0
    if args.replay_mode == "all" and args.replay_per_class > 0:
        ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
        ref_outputs = proto.collect_outputs(
            model, ref_loader, device, desc=f"Reference {args.reference_period}"
        )
        all_classes = list(range(num_classes))
        replay_idx = sample_replay_indices(
            ref_outputs["labels"], all_classes,
            args.replay_per_class, args.seed + 10007,
        )
        if replay_idx.numel() > 0:
            replay_features = ref_outputs["features"][replay_idx]
            replay_labels_np = ref_outputs["labels"][replay_idx.numpy()]
            replay_count = int(replay_idx.numel())
            if args.replay_distill_weight > 0:
                with torch.no_grad():
                    model.cls_head.eval()
                    replay_logits = model.cls_head(
                        replay_features.to(device)
                    ).cpu()
            print(f"Prepared {replay_count} replay samples from {args.reference_period}")

    # --- Collect target-period outputs ---
    loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    outputs = proto.collect_outputs(
        model, loader, device, desc=f"Collect {args.target_period}"
    )
    features = outputs["features"]
    logits = outputs["logits"]
    labels = outputs["labels"]
    static_preds = logits.argmax(dim=1).numpy()

    # --- Static baseline ---
    rows = []
    per_class_rows = []

    static_summary, static_report = summarize(
        labels, static_preds, collapse_classes, stable_classes, thresholds
    )
    rows.append({
        "method": "static",
        "strategy": "",
        "budget": 0,
        "confidence_threshold": "",
        "silver_samples": 0,
        "silver_collapse_count": 0,
        "silver_correct_rate": "",
        "replay_samples": replay_count,
        **{f"full_{k}": v for k, v in static_summary.items()},
    })
    print(
        f"Static: macro_f1={static_summary['overall_macro_f1']:.4f} "
        f"collapse_f1={static_summary['bad_macro_f1']:.4f}"
    )

    # --- Sweep confidence thresholds ---
    for conf_thresh in thresholds_list:
        silver_idx, pseudo_labels, max_probs = select_silver_samples(
            logits, conf_thresh
        )
        n_silver = len(silver_idx)
        if n_silver == 0:
            print(f"  threshold={conf_thresh}: 0 silver samples, skipping")
            continue

        true_labels_of_silver = labels[silver_idx.numpy()]
        correct_rate = float((pseudo_labels == true_labels_of_silver).mean())
        silver_collapse = int(np.isin(true_labels_of_silver, collapse_classes).sum())

        # Per-class breakdown of silver samples
        silver_class_counts = {}
        for c in range(num_classes):
            cnt = int((pseudo_labels == c).sum())
            if cnt > 0:
                silver_class_counts[c] = cnt

        print(
            f"  threshold={conf_thresh}: {n_silver} silver samples, "
            f"accuracy={correct_rate:.4f}, "
            f"collapse_in_silver={silver_collapse}, "
            f"classes_represented={len(silver_class_counts)}"
        )

        # Build training set: silver samples + optional replay
        train_features_parts = [features[silver_idx]]
        train_labels_parts = [
            torch.as_tensor(pseudo_labels, dtype=torch.long)
        ]
        distill_feat = None
        distill_log = None

        if replay_features is not None and replay_count > 0:
            train_features_parts.append(replay_features)
            train_labels_parts.append(
                torch.as_tensor(replay_labels_np, dtype=torch.long)
            )
            if args.replay_distill_weight > 0 and replay_logits is not None:
                distill_feat = replay_features
                distill_log = replay_logits

        train_features = torch.cat(train_features_parts, dim=0)
        train_labels_cat = torch.cat(train_labels_parts, dim=0)

        head = fit_head(
            model,
            train_features,
            train_labels_cat,
            args.ft_lr,
            args.ft_epochs,
            args.ft_batch_size,
            args.ft_weight_decay,
            device,
            distill_features=distill_feat,
            distill_logits=distill_log,
            distill_weight=args.replay_distill_weight,
            distill_temperature=args.distill_temperature,
            seed=args.seed,
        )
        preds = predict_head(head, features, device)

        full_summary, full_report = summarize(
            labels, preds, collapse_classes, stable_classes, thresholds
        )

        method_label = "self_evolving"
        if replay_count > 0:
            method_label = "self_evolving+replay"
            if args.replay_distill_weight > 0:
                method_label = "self_evolving+replay+kd"

        rows.append({
            "method": method_label,
            "strategy": f"confidence>={conf_thresh}",
            "budget": 0,
            "confidence_threshold": conf_thresh,
            "silver_samples": n_silver,
            "silver_collapse_count": silver_collapse,
            "silver_correct_rate": f"{correct_rate:.4f}",
            "replay_samples": replay_count,
            **{f"full_{k}": v for k, v in full_summary.items()},
        })

        for c in collapse_classes:
            item = full_report.get(str(c), {})
            per_class_rows.append({
                "method": method_label,
                "confidence_threshold": conf_thresh,
                "class_id": c,
                "full_support": int(item.get("support", 0)),
                "full_recall": float(item.get("recall", 0.0)),
                "full_f1": float(item.get("f1-score", 0.0)),
                "silver_count": silver_class_counts.get(c, 0),
            })

        print(
            f"    -> macro_f1={full_summary['overall_macro_f1']:.4f} "
            f"collapse_f1={full_summary['bad_macro_f1']:.4f} "
            f"collapsed_count={full_summary.get('collapsed_count', '?')}"
        )

    # --- Save results ---
    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), rows)
    write_csv(
        os.path.join(args.output_dir, "per_collapse_class_m12.csv"),
        per_class_rows,
    )

    summary = {
        "target_period": args.target_period,
        "reference_period": args.reference_period,
        "num_classes": num_classes,
        "seed": args.seed,
        "thresholds": thresholds_list,
        "replay_mode": args.replay_mode,
        "replay_per_class": args.replay_per_class,
        "replay_distill_weight": args.replay_distill_weight,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
