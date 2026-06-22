"""
ILETC baseline: GAN-generated replay features for incremental learning.

Adapts the ILETC approach (Zhu et al., Computer Networks 2023) to our
collapse-repair setting:
  1. Train a conditional WGAN-GP on reference-period features (per-class)
  2. Generate synthetic replay features to replace real reference samples
  3. Fine-tune classification head on target labels + GAN-generated replay

Key differences from CARE:
  - Replay features are GAN-generated, not real reference samples
  - Head-only fine-tuning (same as CARE default)
  - No knowledge distillation (original ILETC doesn't use KD)

Usage from Experiment/core_code/:
    python scripts/baselines/iletc_baseline.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/baselines/iletc
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


class Generator(nn.Module):
    """Conditional generator: noise + class_embedding -> feature."""

    def __init__(self, noise_dim, num_classes, feat_dim, hidden_dim=256):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 64)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, noise, labels):
        emb = self.class_emb(labels)
        x = torch.cat([noise, emb], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional discriminator: feature + class_embedding -> score."""

    def __init__(self, num_classes, feat_dim, hidden_dim=256):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 64)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, labels):
        emb = self.class_emb(labels)
        x = torch.cat([features, emb], dim=1)
        return self.net(x)


def gradient_penalty(disc, real, fake, labels, device):
    """WGAN-GP gradient penalty."""
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = disc(interp, labels)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_wgan_gp(
    ref_features,
    ref_labels,
    num_classes,
    feat_dim,
    noise_dim=128,
    epochs=200,
    batch_size=128,
    lr=1e-4,
    n_critic=5,
    gp_weight=10.0,
    device="cpu",
    seed=0,
):
    """Train conditional WGAN-GP on reference features."""
    gen = Generator(noise_dim, num_classes, feat_dim).to(device)
    disc = Discriminator(num_classes, feat_dim).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.9))

    dataset = torch.utils.data.TensorDataset(
        ref_features, torch.as_tensor(ref_labels, dtype=torch.long),
    )
    rng = torch.Generator()
    rng.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=rng, drop_last=True,
    )

    for epoch in range(epochs):
        for i, (real_feat, real_lbl) in enumerate(loader):
            real_feat = real_feat.to(device)
            real_lbl = real_lbl.to(device)
            bs = real_feat.size(0)

            # Train discriminator
            noise = torch.randn(bs, noise_dim, device=device)
            fake_feat = gen(noise, real_lbl).detach()
            d_real = disc(real_feat, real_lbl).mean()
            d_fake = disc(fake_feat, real_lbl).mean()
            gp = gradient_penalty(disc, real_feat, fake_feat, real_lbl, device)
            d_loss = d_fake - d_real + gp_weight * gp

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train generator every n_critic steps
            if (i + 1) % n_critic == 0:
                noise = torch.randn(bs, noise_dim, device=device)
                fake_feat = gen(noise, real_lbl)
                g_loss = -disc(fake_feat, real_lbl).mean()

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

        if (epoch + 1) % 50 == 0:
            print(f"  GAN epoch {epoch+1}/{epochs}: D_loss={d_loss.item():.4f}")

    gen.eval()
    return gen


def generate_replay_features(gen, num_classes, per_class, noise_dim, device, seed=0):
    """Generate per_class synthetic features for each class."""
    gen.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 99999)

    all_features = []
    all_labels = []

    for c in range(num_classes):
        noise = torch.randn(per_class, noise_dim, device=device, generator=rng)
        labels = torch.full((per_class,), c, dtype=torch.long, device=device)
        with torch.no_grad():
            features = gen(noise, labels)
        all_features.append(features.cpu())
        all_labels.extend([c] * per_class)

    return torch.cat(all_features, dim=0), np.array(all_labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="ILETC baseline (GAN replay)")
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
    parser.add_argument("--gan-epochs", type=int, default=200)
    parser.add_argument("--gan-lr", type=float, default=1e-4)
    parser.add_argument("--noise-dim", type=int, default=128)
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

    print(f"=== ILETC Baseline (GAN Replay) ===")
    print(f"Device: {device}")
    print(f"Budget: {args.budget}, Strategy: {args.strategy}")

    # Collect reference period features for GAN training
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(
        model, ref_loader, device, desc=f"Reference {args.reference_period}",
    )
    feat_dim = ref_outputs["features"].shape[1]

    # Build prototypes (for selection strategies)
    prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref_outputs["features"], ref_outputs["labels"], num_classes, 1,
    )

    # Train conditional WGAN-GP on reference features
    print(f"\nTraining conditional WGAN-GP ({args.gan_epochs} epochs)...")
    gan_gen = train_wgan_gp(
        ref_outputs["features"],
        ref_outputs["labels"],
        num_classes,
        feat_dim,
        noise_dim=args.noise_dim,
        epochs=args.gan_epochs,
        device=device,
        seed=args.seed,
    )

    # Generate synthetic replay features
    gan_features, gan_labels = generate_replay_features(
        gan_gen, num_classes, args.replay_per_class, args.noise_dim, device, args.seed,
    )
    n_replay = gan_features.shape[0]
    print(f"Generated {n_replay} synthetic replay features")

    # Also prepare real replay for comparison
    real_replay_idx = sample_replay_indices(
        ref_outputs["labels"], list(range(num_classes)), args.replay_per_class, args.seed + 10007,
    )

    # Collect target period
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_outputs = proto.collect_outputs(
        model, tgt_loader, device, desc=f"Target {args.target_period}",
    )
    features = tgt_outputs["features"]
    logits = tgt_outputs["logits"]
    labels = tgt_outputs["labels"]
    static_preds = logits.argmax(dim=1).numpy()

    nearest_distance, nearest_proto = prototype_distance_signals(features, prototypes, valid_mask)
    ref_preds = ref_outputs["logits"].argmax(dim=1)
    ref_pred_counts = torch.zeros(num_classes)
    for c in range(num_classes):
        ref_pred_counts[c] = (ref_preds == c).sum()

    static_summary, _ = summarize(labels, static_preds, eval_collapse_classes, stable_classes, thresholds)

    # Active selection
    idx = select_indices(
        args.strategy, logits, labels, args.budget, num_classes,
        collapse_classes, absorber_classes, args.seed,
        nearest_distance=nearest_distance, nearest_proto=nearest_proto,
        features=features, prototypes=prototypes, ref_pred_counts=ref_pred_counts,
    )
    selected_labels = labels[idx.numpy()]

    # Build training set with GAN replay
    target_repeat = max(1, args.target_repeat)
    feat_parts = [features[idx]] * target_repeat
    lbl_parts = [torch.as_tensor(selected_labels, dtype=torch.long)] * target_repeat
    feat_parts.append(gan_features)
    lbl_parts.append(torch.as_tensor(gan_labels, dtype=torch.long))
    train_features = torch.cat(feat_parts, dim=0)
    train_labels_t = torch.cat(lbl_parts, dim=0)

    print(f"Training set: {train_features.shape[0]} samples "
          f"({len(selected_labels) * target_repeat} target + {n_replay} GAN replay)")

    # Fit head (no KD — original ILETC doesn't use distillation)
    head = fit_head(
        model, train_features, train_labels_t,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device,
        distill_features=None, distill_logits=None,
        distill_weight=0.0, distill_temperature=2.0,
        seed=args.seed,
    )
    preds = predict_head(head, features, device)

    eval_mask = np.ones(len(labels), dtype=bool)
    eval_mask[idx.numpy()] = False
    strict_summary, _ = summarize(
        labels[eval_mask], preds[eval_mask], eval_collapse_classes, stable_classes, thresholds
    )
    full_summary, _ = summarize(labels, preds, eval_collapse_classes, stable_classes, thresholds)

    # Also run with real replay for comparison
    real_feat_parts = [features[idx]] * target_repeat
    real_lbl_parts = [torch.as_tensor(selected_labels, dtype=torch.long)] * target_repeat
    if real_replay_idx.numel() > 0:
        real_feat_parts.append(ref_outputs["features"][real_replay_idx])
        real_lbl_parts.append(torch.as_tensor(
            ref_outputs["labels"][real_replay_idx.numpy()], dtype=torch.long))
    real_train_features = torch.cat(real_feat_parts, dim=0)
    real_train_labels = torch.cat(real_lbl_parts, dim=0)

    head_real = fit_head(
        model, real_train_features, real_train_labels,
        args.ft_lr, args.ft_epochs, args.ft_batch_size, args.ft_weight_decay,
        device,
        distill_features=None, distill_logits=None,
        distill_weight=0.0, distill_temperature=2.0,
        seed=args.seed,
    )
    preds_real = predict_head(head_real, features, device)
    real_strict, _ = summarize(
        labels[eval_mask], preds_real[eval_mask], eval_collapse_classes, stable_classes, thresholds
    )

    # Save results
    rows = [
        {
            "method": "static", "budget": 0,
            **{f"strict_{k}": v for k, v in static_summary.items()},
        },
        {
            "method": "iletc_gan_replay", "budget": args.budget,
            "strategy": args.strategy,
            "replay_type": "gan",
            "replay_samples": n_replay,
            "gan_epochs": args.gan_epochs,
            **{f"strict_{k}": v for k, v in strict_summary.items()},
            **{f"full_{k}": v for k, v in full_summary.items()},
        },
        {
            "method": "iletc_real_replay", "budget": args.budget,
            "strategy": args.strategy,
            "replay_type": "real",
            "replay_samples": int(real_replay_idx.numel()),
            **{f"strict_{k}": v for k, v in real_strict.items()},
        },
    ]
    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), rows)

    summary = {
        "method": "iletc",
        "budget": args.budget,
        "strategy": args.strategy,
        "seed": args.seed,
        "replay_per_class": args.replay_per_class,
        "gan_epochs": args.gan_epochs,
        "noise_dim": args.noise_dim,
        "gan_replay_samples": n_replay,
        "ft_lr": args.ft_lr,
        "ft_epochs": args.ft_epochs,
        "strict_macro_f1_gan": strict_summary.get("overall_macro_f1"),
        "strict_collapse_f1_gan": strict_summary.get("bad_macro_f1"),
        "strict_macro_f1_real": real_strict.get("overall_macro_f1"),
        "strict_collapse_f1_real": real_strict.get("bad_macro_f1"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults:")
    print(f"  Static:     macro={static_summary.get('overall_macro_f1', 0):.4f}")
    print(f"  GAN replay: macro={strict_summary.get('overall_macro_f1', 0):.4f} "
          f"collapse={strict_summary.get('bad_macro_f1', 0):.4f}")
    print(f"  Real replay: macro={real_strict.get('overall_macro_f1', 0):.4f} "
          f"collapse={real_strict.get('bad_macro_f1', 0):.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
