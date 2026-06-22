"""
MEMENTO baseline: Full-model update + memory replay + output rectification.

Adapts the MEMENTO approach (Cerasuolo et al., Computer Networks 2024) to our
collapse-repair setting:
  1. Full model fine-tuning (encoder + head) on target labels + memory replay
  2. Knowledge distillation on replay samples (same as CARE)
  3. Output rectification: bias correction on the classification head to
     compensate for class imbalance between old (replay) and new (target) data

Key differences from CARE:
  - Full model update instead of head-only
  - Adds output rectification (bias correction) after training
  - No active selection strategy awareness — uses same margin selector for
    fair comparison

Usage from Experiment/core_code/:
    python scripts/baselines/memento_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/baselines/memento
"""
import argparse
import copy
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
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
    build_head_training_set,
    collapse_counts,
    parse_int_list,
    predict_full_model,
    predict_head,
    prototype_distance_signals,
    replay_class_set,
    sample_replay_indices,
    select_indices,
    summarize,
    write_csv,
)


def fit_full_model_memento(
    source_model,
    train_ppi,
    train_labels,
    lr,
    epochs,
    batch_size,
    weight_decay,
    device,
    distill_model=None,
    distill_weight=0.5,
    distill_temperature=2.0,
    seed=0,
    is_replay=None,
):
    """Full model fine-tuning (MEMENTO-style).

    Same as CARE's fit_full_model but returns the model for post-hoc
    rectification rather than immediately predicting.
    """
    ft_model = copy.deepcopy(source_model)
    ft_model.train()
    if distill_model is not None:
        distill_model.eval()

    n = train_ppi.shape[0]
    dataset = torch.utils.data.TensorDataset(
        train_ppi,
        train_labels,
        torch.arange(n),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False,
    )

    opt = torch.optim.AdamW(ft_model.parameters(), lr=lr, weight_decay=weight_decay)
    temp = distill_temperature

    for epoch in range(epochs):
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
            logits = ft_model(x)
            loss = F.cross_entropy(logits, y)

            if distill_model is not None and distill_weight > 0 and is_replay is not None:
                batch_replay = is_replay[idx]
                replay_idx = idx[batch_replay] if batch_replay.any() else None
                if replay_idx is not None and replay_idx.numel() > 0:
                    replay_logits = ft_model(x[batch_replay])
                    with torch.no_grad():
                        teacher_logits = distill_model(x[batch_replay])
                    student_log_probs = F.log_softmax(replay_logits / temp, dim=1)
                    teacher_probs = F.softmax(teacher_logits / temp, dim=1)
                    distill_loss = F.kl_div(
                        student_log_probs, teacher_probs, reduction="batchmean"
                    ) * (temp ** 2)
                    loss = loss + distill_weight * distill_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    return ft_model


def apply_output_rectification(model, n_old, n_new, num_classes, old_classes, device):
    """MEMENTO-style output rectification: adjust classification head bias.

    When training on imbalanced old/new data, the model's bias terms shift
    toward classes with more training samples. This corrects for that by
    adjusting the bias of the final linear layer.

    bias_correction[old_classes] -= log(n_old / n_new)

    where n_old = number of replay (old) samples, n_new = number of target
    (new) samples used in training.
    """
    if n_old <= 0 or n_new <= 0:
        return model

    correction = math.log(n_old / n_new)

    head = model.cls_head if hasattr(model, "cls_head") else model
    last_linear = None
    for module in reversed(list(head.modules())):
        if isinstance(module, nn.Linear):
            last_linear = module
            break

    if last_linear is None or last_linear.bias is None:
        return model

    with torch.no_grad():
        for c in old_classes:
            if 0 <= c < last_linear.bias.shape[0]:
                last_linear.bias[c] -= correction

    return model


def main():
    parser = argparse.ArgumentParser(description="MEMENTO baseline for collapse repair")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--strategy", default="margin")
    parser.add_argument("--collapse-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--absorber-classes", default=None)
    parser.add_argument("--eval-collapse-classes", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft-lr", type=float, default=1e-4,
                        help="Full model LR (lower than head-only)")
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-per-class", type=int, default=5)
    parser.add_argument("--target-repeat", type=int, default=2)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--rectification", action="store_true", default=True,
                        help="Apply output rectification (default: True)")
    parser.add_argument("--no-rectification", dest="rectification", action="store_false")
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    collapse_classes = parse_int_list(args.collapse_classes, DEFAULT_COLLAPSE_CLASSES)
    eval_collapse_classes = (parse_int_list(args.eval_collapse_classes)
                             if args.eval_collapse_classes else collapse_classes)
    stable_classes = parse_int_list(args.stable_classes, DEFAULT_STABLE_CLASSES)
    absorber_classes = parse_int_list(args.absorber_classes, DEFAULT_ABSORBER_CLASSES)
    thresholds = {
        "collapse": args.collapse_recall_threshold,
        "severe": args.severe_recall_threshold,
    }

    print(f"=== MEMENTO Baseline ===")
    print(f"Device: {device}")
    print(f"Budget: {args.budget}, Strategy: {args.strategy}")
    print(f"Rectification: {args.rectification}")

    # Collect reference period outputs (need PPI for full-model training)
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(
        model, ref_loader, device,
        desc=f"Reference {args.reference_period}",
        keep_ppi=True,
    )

    # Build prototypes for selection strategies
    prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref_outputs["features"], ref_outputs["labels"], num_classes, 1,
    )

    # Replay indices (all-class, same as CARE full config)
    replay_classes = list(range(num_classes))
    replay_idx = sample_replay_indices(
        ref_outputs["labels"], replay_classes, args.replay_per_class, args.seed + 10007,
    )
    n_replay = int(replay_idx.numel())
    print(f"Replay samples: {n_replay} from {len(replay_classes)} classes")

    # Collect target period
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_outputs = proto.collect_outputs(
        model, tgt_loader, device,
        desc=f"Target {args.target_period}",
        keep_ppi=True,
    )
    features = tgt_outputs["features"]
    logits = tgt_outputs["logits"]
    labels = tgt_outputs["labels"]
    all_ppi = tgt_outputs["ppi"]
    static_preds = logits.argmax(dim=1).numpy()

    nearest_distance, nearest_proto = prototype_distance_signals(features, prototypes, valid_mask)
    ref_preds = ref_outputs["logits"].argmax(dim=1)
    ref_pred_counts = torch.zeros(num_classes)
    for c in range(num_classes):
        ref_pred_counts[c] = (ref_preds == c).sum()

    # Static baseline
    static_summary, _ = summarize(labels, static_preds, eval_collapse_classes, stable_classes, thresholds)

    # Active selection
    idx = select_indices(
        args.strategy, logits, labels, args.budget, num_classes,
        collapse_classes, absorber_classes, args.seed,
        nearest_distance=nearest_distance, nearest_proto=nearest_proto,
        features=features, prototypes=prototypes, ref_pred_counts=ref_pred_counts,
    )
    selected_labels = labels[idx.numpy()]
    n_target = len(selected_labels) * max(1, args.target_repeat)

    # Build full-model training set: target PPI (repeated) + replay PPI
    target_repeat = max(1, args.target_repeat)
    ppi_parts = [all_ppi[idx]] * target_repeat
    lbl_parts = [torch.as_tensor(selected_labels, dtype=torch.long)] * target_repeat
    replay_mask_parts = [torch.zeros(len(idx), dtype=torch.bool)] * target_repeat

    if replay_idx.numel() > 0 and "ppi" in ref_outputs:
        ppi_parts.append(ref_outputs["ppi"][replay_idx])
        lbl_parts.append(torch.as_tensor(
            ref_outputs["labels"][replay_idx.numpy()], dtype=torch.long))
        replay_mask_parts.append(torch.ones(replay_idx.numel(), dtype=torch.bool))

    train_ppi = torch.cat(ppi_parts, dim=0)
    train_labels_t = torch.cat(lbl_parts, dim=0)
    replay_mask = torch.cat(replay_mask_parts, dim=0)

    print(f"Training set: {train_ppi.shape[0]} samples "
          f"({n_target} target + {n_replay} replay)")

    # Full model fine-tuning with KD on replay
    ft_model = fit_full_model_memento(
        model, train_ppi, train_labels_t,
        lr=args.ft_lr, epochs=args.ft_epochs,
        batch_size=args.ft_batch_size, weight_decay=args.ft_weight_decay,
        device=device,
        distill_model=model if args.distill_weight > 0 else None,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        seed=args.seed,
        is_replay=replay_mask,
    )

    eval_mask = np.ones(len(labels), dtype=bool)
    eval_mask[idx.numpy()] = False

    # Evaluate WITHOUT rectification first (before mutating the model)
    preds_norect = predict_full_model(ft_model, all_ppi, device)
    norect_strict, _ = summarize(
        labels[eval_mask], preds_norect[eval_mask], eval_collapse_classes, stable_classes, thresholds
    )
    norect_full, _ = summarize(
        labels, preds_norect, eval_collapse_classes, stable_classes, thresholds
    )

    # Apply output rectification, then evaluate again
    if args.rectification and n_replay > 0 and n_target > 0:
        print(f"Applying output rectification (n_old={n_replay}, n_new={n_target})")
        apply_output_rectification(
            ft_model, n_replay, n_target, num_classes, replay_classes, device,
        )

    preds = predict_full_model(ft_model, all_ppi, device)
    strict_summary, strict_report = summarize(
        labels[eval_mask], preds[eval_mask], eval_collapse_classes, stable_classes, thresholds
    )
    full_summary, full_report = summarize(
        labels, preds, eval_collapse_classes, stable_classes, thresholds
    )

    # Save results
    rows = [
        {
            "method": "static", "budget": 0, "strategy": args.strategy,
            **{f"strict_{k}": v for k, v in static_summary.items()},
        },
        {
            "method": "memento", "budget": args.budget,
            "strategy": args.strategy,
            "rectification": args.rectification,
            "replay_samples": n_replay,
            **{f"strict_{k}": v for k, v in strict_summary.items()},
            **{f"full_{k}": v for k, v in full_summary.items()},
        },
        {
            "method": "memento_no_rect", "budget": args.budget,
            "strategy": args.strategy,
            "rectification": False,
            "replay_samples": n_replay,
            **{f"strict_{k}": v for k, v in norect_strict.items()},
            **{f"full_{k}": v for k, v in norect_full.items()},
        },
    ]

    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), rows)

    summary = {
        "method": "memento",
        "budget": args.budget,
        "strategy": args.strategy,
        "seed": args.seed,
        "rectification": args.rectification,
        "replay_per_class": args.replay_per_class,
        "replay_samples": n_replay,
        "target_repeat": args.target_repeat,
        "distill_weight": args.distill_weight,
        "ft_lr": args.ft_lr,
        "ft_epochs": args.ft_epochs,
        "strict_macro_f1": strict_summary.get("overall_macro_f1"),
        "strict_collapse_f1": strict_summary.get("bad_macro_f1"),
        "strict_stable_f1": strict_summary.get("stable_macro_f1"),
        "full_macro_f1": full_summary.get("overall_macro_f1"),
        "full_collapse_f1": full_summary.get("bad_macro_f1"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults:")
    print(f"  Static:  macro={static_summary.get('overall_macro_f1', 0):.4f}")
    print(f"  MEMENTO: macro={strict_summary.get('overall_macro_f1', 0):.4f} "
          f"collapse={strict_summary.get('bad_macro_f1', 0):.4f} "
          f"stable={strict_summary.get('stable_macro_f1', 0):.4f}")
    print(f"  No-rect: macro={norect_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={norect_strict.get('bad_macro_f1', 0):.4f}")
    print(f"Results saved to {args.output_dir}")

    del ft_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
