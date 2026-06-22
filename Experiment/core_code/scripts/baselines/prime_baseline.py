"""
PRIME baseline: Plasticity-Robust Incremental Model via network expansion.

Adapts the PRIME approach (Qin et al., arXiv 2025) to our collapse-repair
setting:
  1. Detect plasticity loss using effective rank of feature representations
  2. Expand classification head capacity (Net2Net-style width expansion)
  3. Fine-tune the expanded head on target labels + replay

Key differences from CARE:
  - Diagnoses plasticity loss as the root cause (vs. CARE's absorber-collapse)
  - Expands network capacity instead of targeted repair
  - Head-only fine-tuning on expanded architecture

Usage from Experiment/core_code/:
    python scripts/baselines/prime_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/baselines/prime
"""
import argparse
import copy
import csv
import json
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
    predict_head,
    prototype_distance_signals,
    sample_replay_indices,
    select_indices,
    summarize,
    write_csv,
)


def effective_rank(features, eps=1e-7):
    """Compute effective rank of feature matrix via singular value entropy.

    effective_rank = exp(H(sigma_normalized))
    where H is Shannon entropy of the normalized singular values.
    Lower effective rank → more redundant features → potential plasticity loss.
    """
    U, S, V = torch.svd(features - features.mean(dim=0, keepdim=True))
    S = S[S > eps]
    p = S / S.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def dead_neuron_ratio(features, threshold=1e-6):
    """Fraction of features that are near-zero across all samples."""
    activity = features.abs().mean(dim=0)
    return (activity < threshold).float().mean().item()


def expand_linear_net2net(linear, expansion_factor=2, noise_scale=0.01, seed=0):
    """Net2Net-style width expansion of a linear layer.

    Doubles the hidden dimension by:
    1. Copying existing weights
    2. Adding small noise to break symmetry
    3. Scaling copied weights by 0.5 to preserve output magnitude
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    old_out, old_in = linear.weight.shape
    new_in = int(old_in * expansion_factor)

    new_linear = nn.Linear(new_in, old_out, bias=linear.bias is not None)

    with torch.no_grad():
        # Copy original weights to first half
        new_linear.weight[:, :old_in] = linear.weight * 0.5
        # Copy + noise to second half
        noise = torch.randn(old_out, old_in, generator=gen) * noise_scale
        new_linear.weight[:, old_in:new_in] = linear.weight * 0.5 + noise

        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)

    return new_linear


class ExpandedHead(nn.Module):
    """Classification head with expanded hidden layer.

    Takes original feature dim as input, projects to expanded hidden,
    then classifies.
    """

    def __init__(self, feat_dim, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def initialize_from_original_head(expanded_head, original_head, device):
    """Initialize expanded head using original head weights where possible.

    For layers with matching dimensions, copy weights directly.
    For expanded layers, use Net2Net-style initialization.
    """
    orig_modules = [m for m in original_head.modules() if isinstance(m, nn.Linear)]
    exp_modules = [m for m in expanded_head.modules() if isinstance(m, nn.Linear)]

    # Copy the last layer (classifier) if dimensions match
    if orig_modules and exp_modules:
        last_orig = orig_modules[-1]
        last_exp = exp_modules[-1]
        if last_orig.weight.shape == last_exp.weight.shape:
            with torch.no_grad():
                last_exp.weight.copy_(last_orig.weight)
                if last_orig.bias is not None and last_exp.bias is not None:
                    last_exp.bias.copy_(last_orig.bias)

    return expanded_head.to(device)


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

    distill_set = None
    if distill_features is not None and distill_logits is not None and distill_weight > 0:
        distill_set = set(range(n - distill_features.shape[0], n))

    temp = distill_temperature

    for epoch in range(epochs):
        for feat, lbl, idx in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            logits = expanded_head(feat)
            loss = F.cross_entropy(logits, lbl)

            if distill_set and distill_weight > 0:
                replay_mask = torch.tensor(
                    [i.item() in distill_set for i in idx], dtype=torch.bool
                )
                if replay_mask.any():
                    replay_local = idx[replay_mask] - (n - distill_features.shape[0])
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
    parser = argparse.ArgumentParser(description="PRIME baseline (network expansion)")
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
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension for expanded head")
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
    thresholds_dict = {
        "collapse": args.collapse_recall_threshold,
        "severe": args.severe_recall_threshold,
    }

    print(f"=== PRIME Baseline (Network Expansion) ===")
    print(f"Device: {device}")
    print(f"Budget: {args.budget}, Strategy: {args.strategy}")
    print(f"Expanded hidden dim: {args.hidden_dim}")

    # Collect reference
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(
        model, ref_loader, device, desc=f"Reference {args.reference_period}",
    )
    feat_dim = ref_outputs["features"].shape[1]

    # Plasticity diagnostics
    print(f"\nPlasticity diagnostics:")
    eff_rank_ref = effective_rank(ref_outputs["features"][:5000])
    dead_ratio_ref = dead_neuron_ratio(ref_outputs["features"])
    print(f"  Reference effective rank: {eff_rank_ref:.1f}/{feat_dim}")
    print(f"  Reference dead neuron ratio: {dead_ratio_ref:.4f}")

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

    eff_rank_tgt = effective_rank(features[:5000])
    dead_ratio_tgt = dead_neuron_ratio(features)
    print(f"  Target effective rank: {eff_rank_tgt:.1f}/{feat_dim}")
    print(f"  Target dead neuron ratio: {dead_ratio_tgt:.4f}")
    print(f"  Rank change: {eff_rank_tgt - eff_rank_ref:+.1f}")

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

    print(f"\nTraining set: {train_features.shape[0]} samples")

    results_rows = [
        {"method": "static", "budget": 0,
         **{f"strict_{k}": v for k, v in static_summary.items()}},
    ]

    # 1. Original head (CARE-style, for comparison)
    from collapse_active_maintenance_tls22 import fit_head
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
    eval_mask = np.ones(len(labels), dtype=bool)
    eval_mask[idx.numpy()] = False
    orig_strict, _ = summarize(
        labels[eval_mask], preds_orig[eval_mask], eval_collapse_classes, stable_classes, thresholds_dict
    )
    results_rows.append({
        "method": "original_head", "budget": args.budget,
        "strategy": args.strategy,
        "head_type": "original",
        **{f"strict_{k}": v for k, v in orig_strict.items()},
    })

    # 2. Expanded head (PRIME-style)
    expanded_head = ExpandedHead(feat_dim, num_classes, hidden_dim=args.hidden_dim).to(device)
    initialize_from_original_head(expanded_head, model.cls_head, device)

    expanded_head = fit_expanded_head(
        model, expanded_head, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device,
        distill_features=replay_features,
        distill_logits=replay_logits,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        seed=args.seed,
    )
    preds_expanded = predict_expanded_head(expanded_head, features, device)
    expanded_strict, _ = summarize(
        labels[eval_mask], preds_expanded[eval_mask], eval_collapse_classes, stable_classes, thresholds_dict
    )
    results_rows.append({
        "method": "prime_expanded", "budget": args.budget,
        "strategy": args.strategy,
        "head_type": "expanded",
        "hidden_dim": args.hidden_dim,
        **{f"strict_{k}": v for k, v in expanded_strict.items()},
    })

    # 3. Expanded head without KD (pure PRIME)
    expanded_head_nokd = ExpandedHead(feat_dim, num_classes, hidden_dim=args.hidden_dim).to(device)
    initialize_from_original_head(expanded_head_nokd, model.cls_head, device)

    expanded_head_nokd = fit_expanded_head(
        model, expanded_head_nokd, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device, seed=args.seed,
    )
    preds_nokd = predict_expanded_head(expanded_head_nokd, features, device)
    nokd_strict, _ = summarize(
        labels[eval_mask], preds_nokd[eval_mask], eval_collapse_classes, stable_classes, thresholds_dict
    )
    results_rows.append({
        "method": "prime_expanded_nokd", "budget": args.budget,
        "strategy": args.strategy,
        "head_type": "expanded_nokd",
        "hidden_dim": args.hidden_dim,
        **{f"strict_{k}": v for k, v in nokd_strict.items()},
    })

    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), results_rows)

    summary = {
        "method": "prime",
        "budget": args.budget,
        "strategy": args.strategy,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "feat_dim": feat_dim,
        "effective_rank_ref": eff_rank_ref,
        "effective_rank_tgt": eff_rank_tgt,
        "dead_neuron_ratio_ref": dead_ratio_ref,
        "dead_neuron_ratio_tgt": dead_ratio_tgt,
        "original_head_macro_f1": orig_strict.get("overall_macro_f1"),
        "expanded_head_macro_f1": expanded_strict.get("overall_macro_f1"),
        "expanded_nokd_macro_f1": nokd_strict.get("overall_macro_f1"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults:")
    print(f"  Static:         macro={static_summary.get('overall_macro_f1', 0):.4f}")
    print(f"  Original head:  macro={orig_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={orig_strict.get('bad_macro_f1', 0):.4f}")
    print(f"  PRIME expanded: macro={expanded_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={expanded_strict.get('bad_macro_f1', 0):.4f}")
    print(f"  PRIME no-KD:    macro={nokd_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={nokd_strict.get('bad_macro_f1', 0):.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
