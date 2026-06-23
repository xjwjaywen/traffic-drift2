"""
CADE baseline: Contrastive Autoencoder for drift detection.

Adapts the CADE approach (Yang et al., USENIX Security 2021) to our
collapse-repair setting. CADE is primarily a detection method, so we compare
its drift detection capability against our 5-signal composite detector.

Approach:
  1. Train a contrastive autoencoder on reference-period features
  2. Use reconstruction error + contrastive distance as drift score
  3. Threshold the score to flag drifted samples
  4. Evaluate detection P/R/F1 against ground-truth collapsed classes
  5. Also test: use CADE-detected samples for label querying, then repair
     with head-only fine-tuning (to compare detection → repair pipeline)

Usage from Experiment/core_code/:
    python scripts/baselines/cade_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/baselines/cade
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
    fit_head,
    parse_int_list,
    predict_head,
    prototype_distance_signals,
    sample_replay_indices,
    select_indices,
    summarize,
    write_csv,
)


class ContrastiveAutoencoder(nn.Module):
    """Contrastive autoencoder for drift detection (CADE-style).

    Encoder maps features to a latent space where same-class samples are
    pulled together and different-class samples are pushed apart.
    Decoder reconstructs the original features.
    """

    def __init__(self, feat_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


def contrastive_loss(z, labels, temperature=0.5):
    """Supervised contrastive loss in latent space."""
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.t() / temperature
    n = z.size(0)
    self_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & self_mask
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)) & self_mask

    if mask_pos.float().sum() == 0:
        return torch.tensor(0.0, device=z.device)

    exp_sim = torch.exp(sim) * self_mask.float()
    denom = (exp_sim * mask_neg.float()).sum(dim=1, keepdim=True) + 1e-12
    log_prob = sim - torch.log(denom + exp_sim * mask_pos.float())
    loss = -(log_prob * mask_pos.float()).sum() / mask_pos.float().sum()
    return loss


def train_cade(
    ref_features,
    ref_labels,
    feat_dim,
    latent_dim=64,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    contrastive_weight=1.0,
    device="cpu",
    seed=0,
):
    """Train contrastive autoencoder on reference features."""
    cae = ContrastiveAutoencoder(feat_dim, latent_dim).to(device)
    opt = torch.optim.Adam(cae.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(
        ref_features, torch.as_tensor(ref_labels, dtype=torch.long),
    )
    rng = torch.Generator()
    rng.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=rng, drop_last=True,
    )

    for epoch in range(epochs):
        total_loss = 0
        for feat, lbl in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            z, recon = cae(feat)
            recon_loss = F.mse_loss(recon, feat)
            cl = contrastive_loss(z, lbl)
            loss = recon_loss + contrastive_weight * cl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f"  CAE epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}")

    cae.eval()
    return cae


@torch.no_grad()
def compute_drift_scores(cae, ref_features, ref_labels, tgt_features, num_classes, device):
    """Compute per-sample drift score using reconstruction error + latent distance.

    For each target sample, the drift score combines:
    1. Reconstruction error (MSE between input and reconstructed feature)
    2. Latent distance to nearest class centroid from reference
    """
    cae.eval()

    # Compute reference class centroids in latent space
    ref_z, _ = cae(ref_features.to(device))
    ref_z = ref_z.cpu()
    centroids = torch.zeros(num_classes, ref_z.shape[1])
    centroid_valid = torch.zeros(num_classes, dtype=torch.bool)
    for c in range(num_classes):
        mask = ref_labels == c
        if mask.sum() > 0:
            centroids[c] = ref_z[mask].mean(dim=0)
            centroid_valid[c] = True

    # Score target samples
    chunk_size = 4096
    all_scores = []
    all_recon_errors = []
    all_latent_distances = []

    for start in range(0, tgt_features.shape[0], chunk_size):
        chunk = tgt_features[start:start + chunk_size].to(device)
        z, recon = cae(chunk)
        z = z.cpu()
        recon = recon.cpu()
        chunk_cpu = chunk.cpu()

        recon_error = ((recon - chunk_cpu) ** 2).mean(dim=1)
        z_norm = F.normalize(z, dim=1)
        cent_norm = F.normalize(centroids, dim=1)
        sims = z_norm @ cent_norm.t()
        sims[:, ~centroid_valid] = -1e9
        latent_dist = 1.0 - sims.max(dim=1).values

        score = recon_error + latent_dist
        all_scores.append(score)
        all_recon_errors.append(recon_error)
        all_latent_distances.append(latent_dist)

    return (
        torch.cat(all_scores),
        torch.cat(all_recon_errors),
        torch.cat(all_latent_distances),
    )


def evaluate_detection(scores, labels, collapse_classes, thresholds_list):
    """Evaluate drift detection at multiple thresholds.

    Ground truth: a sample is "drifted" if its true label is in collapse_classes.
    """
    gt_drift = np.isin(labels, collapse_classes)
    results = []

    for thresh in thresholds_list:
        pred_drift = scores.numpy() > thresh
        tp = (pred_drift & gt_drift).sum()
        fp = (pred_drift & ~gt_drift).sum()
        fn = (~pred_drift & gt_drift).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({
            "threshold": float(thresh),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_flagged": int(pred_drift.sum()),
            "n_drift": int(gt_drift.sum()),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="CADE baseline (contrastive drift detection)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--strategy", default="margin",
                        help="AL strategy for margin-based repair comparison")
    parser.add_argument("--collapse-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--absorber-classes", default=None)
    parser.add_argument("--eval-collapse-classes", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cae-epochs", type=int, default=100)
    parser.add_argument("--cae-lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-per-class", type=int, default=5)
    parser.add_argument("--target-repeat", type=int, default=2)
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

    print(f"=== CADE Baseline (Contrastive Drift Detection) ===")
    print(f"Device: {device}")

    # Collect reference
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(
        model, ref_loader, device, desc=f"Reference {args.reference_period}",
    )
    feat_dim = ref_outputs["features"].shape[1]

    # Train contrastive autoencoder
    print(f"\nTraining contrastive autoencoder ({args.cae_epochs} epochs)...")
    cae = train_cade(
        ref_outputs["features"], ref_outputs["labels"], feat_dim,
        latent_dim=args.latent_dim, epochs=args.cae_epochs,
        lr=args.cae_lr, contrastive_weight=args.contrastive_weight,
        device=device, seed=args.seed,
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
    static_summary, _ = summarize(labels, static_preds, eval_collapse_classes, stable_classes, thresholds)

    # Compute drift scores
    print("\nComputing drift scores...")
    drift_scores, recon_errors, latent_distances = compute_drift_scores(
        cae, ref_outputs["features"], ref_outputs["labels"],
        features, num_classes, device,
    )

    # Evaluate detection at multiple thresholds
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    thresh_values = [float(torch.quantile(drift_scores.float(), p/100.0)) for p in percentiles]
    detection_results = evaluate_detection(drift_scores, labels, collapse_classes, thresh_values)

    best_det = max(detection_results, key=lambda x: x["f1"])
    print(f"\nBest detection F1={best_det['f1']:.4f} "
          f"(P={best_det['precision']:.4f}, R={best_det['recall']:.4f}) "
          f"at threshold={best_det['threshold']:.4f}")

    # Detection-guided repair: use CADE scores to select samples for labeling
    # Select top-B samples by drift score
    top_idx = torch.argsort(drift_scores, descending=True)[:args.budget]
    selected_labels = labels[top_idx.numpy()]

    # Also run margin-based selection for comparison
    prototypes, _, valid_mask = proto.build_prototypes(
        ref_outputs["features"], ref_outputs["labels"], num_classes, 1,
    )
    nearest_distance, nearest_proto = prototype_distance_signals(features, prototypes, valid_mask)
    ref_preds = ref_outputs["logits"].argmax(dim=1)
    ref_pred_counts = torch.zeros(num_classes)
    for c in range(num_classes):
        ref_pred_counts[c] = (ref_preds == c).sum()

    margin_idx = select_indices(
        args.strategy, logits, labels, args.budget, num_classes,
        collapse_classes, absorber_classes, args.seed,
        nearest_distance=nearest_distance, nearest_proto=nearest_proto,
        features=features, prototypes=prototypes, ref_pred_counts=ref_pred_counts,
    )

    # Prepare replay
    replay_classes = list(range(num_classes))
    replay_idx = sample_replay_indices(
        ref_outputs["labels"], replay_classes, args.replay_per_class, args.seed + 10007,
    )

    results_rows = [
        {"method": "static", "strategy": "", "budget": 0,
         **{f"strict_{k}": v for k, v in static_summary.items()}},
    ]

    # Repair with CADE-selected samples
    for sel_name, sel_idx in [("cade_score", top_idx), (args.strategy, margin_idx)]:
        sel_labels = labels[sel_idx.numpy()]
        train_features, train_labels_t = build_head_training_set(
            features[sel_idx], sel_labels,
            ref_outputs["features"], ref_outputs["labels"],
            replay_idx, args.target_repeat,
        )

        replay_features = ref_outputs["features"][replay_idx] if replay_idx.numel() > 0 else None
        replay_logits = None
        if replay_features is not None:
            with torch.no_grad():
                model.cls_head.eval()
                replay_logits = model.cls_head(replay_features.to(device)).cpu()

        head = fit_head(
            model, train_features, train_labels_t,
            args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
            device,
            distill_features=replay_features,
            distill_logits=replay_logits,
            distill_weight=0.5, distill_temperature=2.0,
            seed=args.seed,
        )
        preds = predict_head(head, features, device)

        eval_mask = np.ones(len(labels), dtype=bool)
        eval_mask[sel_idx.numpy()] = False
        strict_summary_sel, _ = summarize(
            labels[eval_mask], preds[eval_mask], eval_collapse_classes, stable_classes, thresholds
        )
        full_summary_sel, _ = summarize(labels, preds, eval_collapse_classes, stable_classes, thresholds)

        results_rows.append({
            "method": f"cade_repair_{sel_name}", "budget": args.budget,
            "strategy": sel_name,
            "replay_samples": int(replay_idx.numel()),
            **{f"strict_{k}": v for k, v in strict_summary_sel.items()},
            **{f"full_{k}": v for k, v in full_summary_sel.items()},
        })
        print(f"  {sel_name}: macro={strict_summary_sel.get('overall_macro_f1', 0):.4f} "
              f"collapse={strict_summary_sel.get('bad_macro_f1', 0):.4f}")

    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), results_rows)

    # Save detection results
    det_rows = []
    for i, p in enumerate(percentiles):
        det_rows.append({"percentile": p, **detection_results[i]})
    write_csv(os.path.join(args.output_dir, "detection_results.csv"), det_rows)

    summary = {
        "method": "cade",
        "seed": args.seed,
        "cae_epochs": args.cae_epochs,
        "latent_dim": args.latent_dim,
        "contrastive_weight": args.contrastive_weight,
        "best_detection_f1": best_det["f1"],
        "best_detection_precision": best_det["precision"],
        "best_detection_recall": best_det["recall"],
        "best_detection_threshold": best_det["threshold"],
        "repair_cade_macro_f1": results_rows[1].get("strict_overall_macro_f1") if len(results_rows) > 1 else None,
        "repair_margin_macro_f1": results_rows[2].get("strict_overall_macro_f1") if len(results_rows) > 2 else None,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetection results saved to {args.output_dir}/detection_results.csv")
    print(f"Repair results saved to {args.output_dir}/results_by_budget.csv")


if __name__ == "__main__":
    main()
