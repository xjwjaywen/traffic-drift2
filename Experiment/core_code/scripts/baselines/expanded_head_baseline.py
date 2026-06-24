"""
Expanded Head baseline: wider classification head with more capacity.

Tests whether the original single-layer head (Linear(256, C)) lacks
capacity to separate drifted classes, by inserting a hidden layer.

Approach:
  1. Build a 2-layer head: Linear(feat_dim, hidden_dim) -> ReLU ->
     Linear(hidden_dim, num_classes)
  2. Initialize the output layer from the original head weights
  3. Fine-tune on target labels + replay (same pipeline as CARE)

This is NOT a full PRIME implementation — PRIME involves plasticity
diagnostics, Net2Net width expansion of existing layers, and
progressive expansion triggers. This simpler test isolates one
question: does adding head capacity help collapse repair?

Usage from Experiment/core_code/:
    python scripts/baselines/expanded_head_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/baselines/expanded_head
"""
import argparse
import copy
import csv
import json
import os
import subprocess
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
    fit_head,
    parse_int_list,
    predict_head,
    prototype_distance_signals,
    sample_replay_indices,
    select_indices,
    summarize,
    write_csv,
)


class ExpandedHead(nn.Module):
    """Two-layer classification head with hidden layer.

    Architecture: Linear(feat_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, C)

    The output layer is initialized from the original head's weights
    (original head: Linear(feat_dim, C)). The first layer projects to the
    hidden dim; the second layer is initialized so that the expanded head
    starts near the original head's decision boundary.
    """

    def __init__(self, feat_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def init_expanded_from_original(expanded_head, original_fc, seed=0):
    """Initialize expanded head to approximate the original linear head.

    Original head: y = W_orig @ x + b_orig  (shape: [C, feat_dim])

    We decompose this into two layers:
      fc1: h = ReLU(W1 @ x)     (shape: [hidden_dim, feat_dim])
      fc2: y = W2 @ h + b_orig  (shape: [C, hidden_dim])

    Strategy: fc1 is initialized with small random weights (Kaiming),
    fc2's first feat_dim columns approximate W_orig via identity-like
    mapping through ReLU, remaining columns are zero.

    This is approximate because ReLU clips negatives, but provides a
    much better starting point than random initialization.
    """
    feat_dim = original_fc.in_features
    hidden_dim = expanded_head.fc1.out_features

    gen = torch.Generator()
    gen.manual_seed(seed)

    with torch.no_grad():
        nn.init.kaiming_normal_(expanded_head.fc1.weight)
        nn.init.zeros_(expanded_head.fc1.bias)

        nn.init.zeros_(expanded_head.fc2.weight)
        copy_dim = min(hidden_dim, feat_dim)
        expanded_head.fc2.weight[:, :copy_dim] = original_fc.weight[:, :copy_dim]
        if original_fc.bias is not None:
            expanded_head.fc2.bias.copy_(original_fc.bias)
        else:
            nn.init.zeros_(expanded_head.fc2.bias)

        expanded_head.fc1.weight[:copy_dim, :] = torch.eye(copy_dim, feat_dim)
        expanded_head.fc1.bias[:copy_dim] = 0.0


def fit_expanded_head(
    model,
    expanded_head,
    train_features,
    train_labels,
    lr,
    epochs,
    batch_size,
    weight_decay,
    device,
    distill_features=None,
    distill_logits=None,
    distill_weight=0.0,
    distill_temperature=2.0,
    seed=0,
):
    """Train the expanded head on features."""
    expanded_head.train()
    n = train_features.shape[0]

    dataset = torch.utils.data.TensorDataset(
        train_features, train_labels, torch.arange(n),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False,
    )
    opt = torch.optim.AdamW(expanded_head.parameters(), lr=lr, weight_decay=weight_decay)

    distill_start = None
    if distill_features is not None and distill_logits is not None and distill_weight > 0:
        distill_start = n - distill_features.shape[0]
        distill_set = set(range(distill_start, n))

    temp = distill_temperature

    for epoch in range(epochs):
        for feat, lbl, idx in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            logits = expanded_head(feat)
            loss = F.cross_entropy(logits, lbl)

            if distill_start is not None and distill_weight > 0:
                replay_mask = idx >= distill_start
                if replay_mask.any():
                    replay_local = idx[replay_mask] - distill_start
                    replay_local = replay_local.clamp(0, distill_logits.shape[0] - 1)
                    teacher_logits = distill_logits[replay_local].to(device)
                    student_logits = logits[replay_mask]
                    student_log = F.log_softmax(student_logits / temp, dim=1)
                    teacher_prob = F.softmax(teacher_logits / temp, dim=1)
                    kd_loss = F.kl_div(student_log, teacher_prob, reduction="batchmean") * (temp ** 2)
                    loss = loss + distill_weight * kd_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    expanded_head.eval()
    return expanded_head


@torch.no_grad()
def predict_expanded_head(head, features, device, chunk_size=8192):
    preds = []
    head.eval()
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start:start + chunk_size].to(device)
        preds.append(head(chunk).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser(description="Expanded head baseline (capacity test)")
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
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-per-class", type=int, default=5)
    parser.add_argument("--target-repeat", type=int, default=2)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    args = parser.parse_args()

    # Fix all random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    thresholds_dict = {
        "collapse": args.collapse_recall_threshold,
        "severe": args.severe_recall_threshold,
    }

    # Get script commit hash for reproducibility
    script_hash = "unknown"
    try:
        script_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    print(f"=== Expanded Head Baseline (commit: {script_hash}) ===")
    print(f"Device: {device}")
    print(f"Budget: {args.budget}, Strategy: {args.strategy}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Original head: Linear({model.cls_head.fc.in_features}, {model.cls_head.fc.out_features})")

    # Collect reference
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(
        model, ref_loader, device, desc=f"Reference {args.reference_period}",
    )
    feat_dim = ref_outputs["features"].shape[1]

    # Build prototypes
    prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref_outputs["features"], ref_outputs["labels"], num_classes, 1,
    )

    # Prepare replay
    replay_classes = list(range(num_classes))
    replay_idx = sample_replay_indices(
        ref_outputs["labels"], replay_classes, args.replay_per_class, args.seed + 10007,
    )

    # Collect target
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_outputs = proto.collect_outputs(
        model, tgt_loader, device, desc=f"Target {args.target_period}",
    )
    features = tgt_outputs["features"]
    logits = tgt_outputs["logits"]
    labels = tgt_outputs["labels"]
    static_preds = logits.argmax(dim=1).numpy()

    static_summary, _ = summarize(labels, static_preds, eval_collapse_classes, stable_classes, thresholds_dict)

    nearest_distance, nearest_proto = prototype_distance_signals(features, prototypes, valid_mask)
    ref_preds = ref_outputs["logits"].argmax(dim=1)
    ref_pred_counts = torch.zeros(num_classes)
    for c in range(num_classes):
        ref_pred_counts[c] = (ref_preds == c).sum()

    # Active selection
    idx = select_indices(
        args.strategy, logits, labels, args.budget, num_classes,
        collapse_classes, absorber_classes, args.seed,
        nearest_distance=nearest_distance, nearest_proto=nearest_proto,
        features=features, prototypes=prototypes, ref_pred_counts=ref_pred_counts,
    )
    selected_labels = labels[idx.numpy()]

    # Build training set
    train_features, train_labels_t = build_head_training_set(
        features[idx], selected_labels,
        ref_outputs["features"], ref_outputs["labels"],
        replay_idx, args.target_repeat,
    )

    # Prepare distillation targets
    replay_features = ref_outputs["features"][replay_idx] if replay_idx.numel() > 0 else None
    replay_logits = None
    if replay_features is not None and args.distill_weight > 0:
        with torch.no_grad():
            model.cls_head.eval()
            replay_logits = model.cls_head(replay_features.to(device)).cpu()

    print(f"Training set: {train_features.shape[0]} samples")
    eval_mask = np.ones(len(labels), dtype=bool)
    eval_mask[idx.numpy()] = False

    results_rows = [
        {"method": "static", "strategy": args.strategy, "budget": 0,
         **{f"strict_{k}": v for k, v in static_summary.items()}},
    ]

    # 1. Original head (CARE-style baseline for comparison)
    head_orig = fit_head(
        model, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device,
        distill_features=replay_features,
        distill_logits=replay_logits,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        seed=args.seed,
    )
    preds_orig = predict_head(head_orig, features, device)
    orig_strict, _ = summarize(
        labels[eval_mask], preds_orig[eval_mask], eval_collapse_classes, stable_classes, thresholds_dict
    )
    results_rows.append({
        "method": "original_head", "budget": args.budget,
        "strategy": args.strategy, "head_type": "original",
        **{f"strict_{k}": v for k, v in orig_strict.items()},
    })

    # 2. Expanded head initialized from original
    expanded = ExpandedHead(feat_dim, num_classes, hidden_dim=args.hidden_dim).to(device)
    init_expanded_from_original(expanded, model.cls_head.fc, seed=args.seed)
    expanded = fit_expanded_head(
        model, expanded, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device,
        distill_features=replay_features,
        distill_logits=replay_logits,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        seed=args.seed,
    )
    preds_exp = predict_expanded_head(expanded, features, device)
    exp_strict, _ = summarize(
        labels[eval_mask], preds_exp[eval_mask], eval_collapse_classes, stable_classes, thresholds_dict
    )
    results_rows.append({
        "method": "expanded_head", "budget": args.budget,
        "strategy": args.strategy, "head_type": "expanded",
        "hidden_dim": args.hidden_dim,
        **{f"strict_{k}": v for k, v in exp_strict.items()},
    })

    # 3. Expanded head without KD
    expanded_nokd = ExpandedHead(feat_dim, num_classes, hidden_dim=args.hidden_dim).to(device)
    init_expanded_from_original(expanded_nokd, model.cls_head.fc, seed=args.seed)
    expanded_nokd = fit_expanded_head(
        model, expanded_nokd, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device, seed=args.seed,
    )
    preds_nokd = predict_expanded_head(expanded_nokd, features, device)
    nokd_strict, _ = summarize(
        labels[eval_mask], preds_nokd[eval_mask], eval_collapse_classes, stable_classes, thresholds_dict
    )
    results_rows.append({
        "method": "expanded_head_nokd", "budget": args.budget,
        "strategy": args.strategy, "head_type": "expanded_nokd",
        "hidden_dim": args.hidden_dim,
        **{f"strict_{k}": v for k, v in nokd_strict.items()},
    })

    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), results_rows)

    summary = {
        "method": "expanded_head",
        "budget": args.budget,
        "strategy": args.strategy,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "feat_dim": feat_dim,
        "original_head_arch": f"Linear({feat_dim}, {num_classes})",
        "expanded_head_arch": f"Linear({feat_dim}, {args.hidden_dim}) -> ReLU -> Linear({args.hidden_dim}, {num_classes})",
        "script_commit": script_hash,
        "original_head_macro_f1": orig_strict.get("overall_macro_f1"),
        "expanded_head_macro_f1": exp_strict.get("overall_macro_f1"),
        "expanded_nokd_macro_f1": nokd_strict.get("overall_macro_f1"),
        "original_head_collapse_f1": orig_strict.get("bad_macro_f1"),
        "expanded_head_collapse_f1": exp_strict.get("bad_macro_f1"),
        "expanded_nokd_collapse_f1": nokd_strict.get("bad_macro_f1"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults:")
    print(f"  Static:         macro={static_summary.get('overall_macro_f1', 0):.4f}")
    print(f"  Original head:  macro={orig_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={orig_strict.get('bad_macro_f1', 0):.4f}")
    print(f"  Expanded head:  macro={exp_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={exp_strict.get('bad_macro_f1', 0):.4f}")
    print(f"  Expanded no-KD: macro={nokd_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={nokd_strict.get('bad_macro_f1', 0):.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
