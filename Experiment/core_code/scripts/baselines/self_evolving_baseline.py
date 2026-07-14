"""
Self-Evolving baseline: pseudo-label fine-tuning without human annotations.

Adapts the core idea from Chen et al. 2025 (arXiv:2501.04246) to our setting:
  1. Run frozen source model on target-period data (train split)
  2. Select high-confidence predictions as "silver samples" (pseudo-labels)
  3. Fine-tune model on pseudo-labeled samples (head-only or full FFT)
  4. Evaluate on held-out test split (never used for pseudo-label selection)

Key differences from CARE:
  - No human labels at all (zero annotation cost)
  - Selection by model confidence, not active learning
  - No source replay or knowledge distillation (vanilla version)
  - Optional: +replay and +KD variants for ablation

Default hyperparameters follow Chen et al. 2025:
  epochs=50, batch_size=500, lr=0.0025, holdout_ratio=0.2

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
    fit_full_model,
    fit_head,
    predict_full_model,
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


def split_train_holdout(n_samples, holdout_ratio, seed):
    """Split indices into train and holdout sets deterministically."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_holdout = max(1, int(n_samples * holdout_ratio))
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    return np.sort(train_idx), np.sort(holdout_idx)


def per_class_silver_stats(pseudo_labels, true_labels, num_classes):
    """Compute per-class silver sample statistics.

    Returns dict: class_id -> {pseudo_count, true_count, accuracy}
    - pseudo_count: how many samples were predicted as this class
    - true_count: how many selected samples truly belong to this class
    - accuracy: among samples predicted as this class, fraction truly correct
    """
    stats = {}
    for c in range(num_classes):
        pseudo_mask = pseudo_labels == c
        true_mask = true_labels == c
        pseudo_count = int(pseudo_mask.sum())
        true_count = int(true_mask.sum())
        if pseudo_count > 0:
            accuracy = float((pseudo_labels[pseudo_mask] == true_labels[pseudo_mask]).mean())
        else:
            accuracy = None
        if pseudo_count > 0 or true_count > 0:
            stats[c] = {
                "pseudo_count": pseudo_count,
                "true_count": true_count,
                "accuracy": accuracy,
            }
    return stats


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
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of target data reserved for evaluation (Chen et al. use 0.2).",
    )
    parser.add_argument("--ft-lr", type=float, default=0.0025,
                        help="Learning rate (Chen et al. default: 0.0025).")
    parser.add_argument("--ft-epochs", type=int, default=50,
                        help="Training epochs (Chen et al. default: 50).")
    parser.add_argument("--ft-batch-size", type=int, default=500,
                        help="Batch size (Chen et al. default: 500).")
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--ft-depth",
        choices=["head", "full"],
        default="full",
        help="Fine-tuning depth: 'head' (cls head only) or 'full' (encoder+head, Chen et al. default).",
    )
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
    print(f"FT depth: {args.ft_depth}, lr: {args.ft_lr}, epochs: {args.ft_epochs}, "
          f"batch_size: {args.ft_batch_size}")
    print(f"Holdout ratio: {args.holdout_ratio}")
    print(f"Replay mode: {args.replay_mode}, per_class: {args.replay_per_class}")

    # --- Prepare replay if requested ---
    replay_features = None
    replay_logits = None
    replay_labels_np = None
    replay_ppi = None
    replay_count = 0
    if args.replay_mode == "all" and args.replay_per_class > 0:
        ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
        ref_outputs = proto.collect_outputs(
            model, ref_loader, device, desc=f"Reference {args.reference_period}",
            keep_ppi=(args.ft_depth == "full"),
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
            if args.ft_depth == "full" and "ppi" in ref_outputs:
                replay_ppi = ref_outputs["ppi"][replay_idx]
            if args.replay_distill_weight > 0:
                with torch.no_grad():
                    model.cls_head.eval()
                    replay_logits = model.cls_head(
                        replay_features.to(device)
                    ).cpu()
            print(f"Prepared {replay_count} replay samples from {args.reference_period}")

    # --- Collect target-period outputs ---
    keep_ppi = (args.ft_depth == "full")
    loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    outputs = proto.collect_outputs(
        model, loader, device, desc=f"Collect {args.target_period}",
        keep_ppi=keep_ppi,
    )
    features = outputs["features"]
    logits = outputs["logits"]
    labels = outputs["labels"]
    all_ppi = outputs.get("ppi")
    static_preds = logits.argmax(dim=1).numpy()
    n_total = len(labels)

    # --- Split into train / holdout ---
    train_split, holdout_split = split_train_holdout(
        n_total, args.holdout_ratio, args.seed + 42
    )
    n_train = len(train_split)
    n_holdout = len(holdout_split)
    print(f"M12 split: {n_train} train, {n_holdout} holdout "
          f"(ratio={args.holdout_ratio})")

    holdout_labels = labels[holdout_split]
    holdout_features = features[holdout_split]
    holdout_ppi = all_ppi[holdout_split] if all_ppi is not None else None

    train_features = features[train_split]
    train_logits = logits[train_split]
    train_labels = labels[train_split]
    train_ppi = all_ppi[train_split] if all_ppi is not None else None

    # --- Static baseline (evaluated on holdout only) ---
    rows = []
    per_class_rows = []

    static_holdout_preds = static_preds[holdout_split]
    static_holdout_summary, static_holdout_report = summarize(
        holdout_labels, static_holdout_preds, collapse_classes, stable_classes, thresholds
    )
    static_full_summary, static_full_report = summarize(
        labels, static_preds, collapse_classes, stable_classes, thresholds
    )
    rows.append({
        "method": "static",
        "strategy": "",
        "budget": 0,
        "confidence_threshold": "",
        "silver_samples": 0,
        "silver_collapse_true_count": 0,
        "silver_collapse_pseudo_count": 0,
        "silver_correct_rate": "",
        "replay_samples": replay_count,
        "ft_depth": "",
        **{f"holdout_{k}": v for k, v in static_holdout_summary.items()},
        **{f"strict_{k}": v for k, v in static_holdout_summary.items()},
        **{f"full_{k}": v for k, v in static_full_summary.items()},
    })
    print(
        f"Static: holdout_f1={static_holdout_summary['overall_macro_f1']:.4f} "
        f"collapse_f1={static_holdout_summary['bad_macro_f1']:.4f}"
    )

    # --- Sweep confidence thresholds ---
    for conf_thresh in thresholds_list:
        silver_idx, pseudo_labels, max_probs = select_silver_samples(
            train_logits, conf_thresh
        )
        n_silver = len(silver_idx)
        if n_silver == 0:
            print(f"  threshold={conf_thresh}: 0 silver samples, skipping")
            continue

        true_labels_of_silver = train_labels[silver_idx.numpy()]
        correct_rate = float((pseudo_labels == true_labels_of_silver).mean())

        class_stats = per_class_silver_stats(pseudo_labels, true_labels_of_silver, num_classes)

        silver_collapse_true = sum(
            class_stats.get(c, {}).get("true_count", 0) for c in collapse_classes
        )
        silver_collapse_pseudo = sum(
            class_stats.get(c, {}).get("pseudo_count", 0) for c in collapse_classes
        )

        print(
            f"  threshold={conf_thresh}: {n_silver} silver samples, "
            f"accuracy={correct_rate:.4f}, "
            f"collapse_true={silver_collapse_true}, "
            f"collapse_pseudo={silver_collapse_pseudo}, "
            f"classes_represented={len(class_stats)}"
        )

        # Build training set: silver samples + optional replay
        silver_features = train_features[silver_idx]
        silver_ppi = train_ppi[silver_idx] if train_ppi is not None else None

        if args.ft_depth == "full":
            train_ppi_parts = [silver_ppi]
            train_label_parts = [torch.as_tensor(pseudo_labels, dtype=torch.long)]
            replay_mask_parts = [torch.zeros(n_silver, dtype=torch.bool)]

            if replay_ppi is not None and replay_count > 0:
                train_ppi_parts.append(replay_ppi)
                train_label_parts.append(
                    torch.as_tensor(replay_labels_np, dtype=torch.long)
                )
                replay_mask_parts.append(torch.ones(replay_count, dtype=torch.bool))

            full_train_ppi = torch.cat(train_ppi_parts, dim=0)
            full_train_labels = torch.cat(train_label_parts, dim=0)
            full_replay_mask = torch.cat(replay_mask_parts, dim=0)

            ft_model = fit_full_model(
                model,
                full_train_ppi,
                full_train_labels,
                args.ft_lr,
                args.ft_epochs,
                args.ft_batch_size,
                args.ft_weight_decay,
                device,
                distill_model=model if args.replay_distill_weight > 0 else None,
                distill_weight=args.replay_distill_weight,
                distill_temperature=args.distill_temperature,
                seed=args.seed,
                is_replay=full_replay_mask,
            )
            holdout_preds = predict_full_model(ft_model, holdout_ppi, device)
            full_preds = predict_full_model(ft_model, all_ppi, device)
            del ft_model
        else:
            ft_feat_parts = [silver_features]
            ft_label_parts = [torch.as_tensor(pseudo_labels, dtype=torch.long)]
            distill_feat = None
            distill_log = None

            if replay_features is not None and replay_count > 0:
                ft_feat_parts.append(replay_features)
                ft_label_parts.append(
                    torch.as_tensor(replay_labels_np, dtype=torch.long)
                )
                if args.replay_distill_weight > 0 and replay_logits is not None:
                    distill_feat = replay_features
                    distill_log = replay_logits

            ft_features = torch.cat(ft_feat_parts, dim=0)
            ft_labels = torch.cat(ft_label_parts, dim=0)

            head = fit_head(
                model,
                ft_features,
                ft_labels,
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
            holdout_preds = predict_head(head, holdout_features, device)
            full_preds = predict_head(head, features, device)

        holdout_summary, holdout_report = summarize(
            holdout_labels, holdout_preds, collapse_classes, stable_classes, thresholds
        )

        full_summary, full_report = summarize(
            labels, full_preds, collapse_classes, stable_classes, thresholds
        )

        # Strict: exclude silver samples from full evaluation
        silver_global_idx = train_split[silver_idx.numpy()]
        strict_mask = np.ones(n_total, dtype=bool)
        strict_mask[silver_global_idx] = False
        strict_labels = labels[strict_mask]
        strict_preds = full_preds[strict_mask]
        strict_summary, strict_report = summarize(
            strict_labels, strict_preds, collapse_classes, stable_classes, thresholds
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
            "silver_collapse_true_count": silver_collapse_true,
            "silver_collapse_pseudo_count": silver_collapse_pseudo,
            "silver_correct_rate": f"{correct_rate:.4f}",
            "replay_samples": replay_count,
            "ft_depth": args.ft_depth,
            **{f"holdout_{k}": v for k, v in holdout_summary.items()},
            **{f"strict_{k}": v for k, v in strict_summary.items()},
            **{f"full_{k}": v for k, v in full_summary.items()},
        })

        for c in collapse_classes:
            h_item = holdout_report.get(str(c), {})
            s_item = strict_report.get(str(c), {})
            f_item = full_report.get(str(c), {})
            cs = class_stats.get(c, {})
            per_class_rows.append({
                "method": method_label,
                "confidence_threshold": conf_thresh,
                "class_id": c,
                "silver_true_count": cs.get("true_count", 0),
                "silver_pseudo_count": cs.get("pseudo_count", 0),
                "silver_class_accuracy": cs.get("accuracy", ""),
                "holdout_support": int(h_item.get("support", 0)),
                "holdout_recall": float(h_item.get("recall", 0.0)),
                "holdout_f1": float(h_item.get("f1-score", 0.0)),
                "strict_support": int(s_item.get("support", 0)),
                "strict_recall": float(s_item.get("recall", 0.0)),
                "strict_f1": float(s_item.get("f1-score", 0.0)),
                "full_support": int(f_item.get("support", 0)),
                "full_recall": float(f_item.get("recall", 0.0)),
                "full_f1": float(f_item.get("f1-score", 0.0)),
            })

        print(
            f"    -> holdout: macro_f1={holdout_summary['overall_macro_f1']:.4f} "
            f"collapse_f1={holdout_summary['bad_macro_f1']:.4f} "
            f"collapsed={holdout_summary.get('collapsed_count', '?')}"
        )
        print(
            f"       strict:  macro_f1={strict_summary['overall_macro_f1']:.4f} "
            f"collapse_f1={strict_summary['bad_macro_f1']:.4f}"
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
        "holdout_ratio": args.holdout_ratio,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "thresholds": thresholds_list,
        "ft_depth": args.ft_depth,
        "ft_lr": args.ft_lr,
        "ft_epochs": args.ft_epochs,
        "ft_batch_size": args.ft_batch_size,
        "ft_weight_decay": args.ft_weight_decay,
        "replay_mode": args.replay_mode,
        "replay_per_class": args.replay_per_class,
        "replay_distill_weight": args.replay_distill_weight,
        "distill_temperature": args.distill_temperature,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
