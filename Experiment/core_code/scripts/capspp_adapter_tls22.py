"""
CAPS++ minimal adapter MVP for TLS-Year22.

This script tests the next step beyond prototype-only CAPS: target-side
representation adaptation with a small residual adapter.

Frozen:
  - encoder
  - classifier

Learned online:
  - residual adapter A(h), initialized as zero residual
  - target prototype bank p_t by EMA

Causal online protocol per target batch:
  1. Predict with current adapter and target prototypes.
  2. Select high-confidence pseudo-labeled samples.
  3. Update target prototypes with adapted features.
  4. Update adapter for future batches using:
       L = lambda_proto * ||norm(h') - norm(p_t_y)||^2
         + lambda_anchor * ||h' - h||^2 for stable pseudo-labels

Usage:
    python scripts/capspp_adapter_tls22.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --reference-period M-2022-4 \
        --target-period M-2022-12 \
        --output-dir outputs/capspp_adapter_tls22_M-2022-12
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import caps_target_prototype_tls22 as caps
import prototype_recalibration_tls22 as proto


def parse_float_list(value):
    return [float(x) for x in value.replace(",", " ").split()]


class ResidualAdapter(nn.Module):
    """Small residual MLP, initialized to identity."""

    def __init__(self, dim, bottleneck=64, scale=1.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim),
        )
        # Start from exact identity: h' = h.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.scale * self.net(h)


def group_mask_from_pseudo(pseudo, classes):
    if not classes:
        return torch.zeros_like(pseudo, dtype=torch.bool)
    class_tensor = torch.tensor(sorted(classes), device=pseudo.device, dtype=pseudo.dtype)
    return (pseudo.unsqueeze(1) == class_tensor.unsqueeze(0)).any(dim=1)


def adapter_losses(
    adapter,
    feat_b,
    pseudo,
    mask,
    target_prototypes,
    stable_classes,
    lambda_proto,
    lambda_anchor,
):
    if mask.sum().item() == 0:
        return None, {"proto_loss": 0.0, "anchor_loss": 0.0}

    h_prime = adapter(feat_b)
    h_sel = h_prime[mask]
    pseudo_sel = pseudo[mask]
    proto_sel = target_prototypes[pseudo_sel].detach()

    proto_loss = (
        F.normalize(h_sel, dim=1) - F.normalize(proto_sel, dim=1)
    ).pow(2).sum(dim=1).mean()

    stable_mask = group_mask_from_pseudo(pseudo_sel, stable_classes)
    if stable_mask.sum().item() > 0:
        anchor_loss = (h_sel[stable_mask] - feat_b[mask][stable_mask]).pow(2).mean()
    else:
        anchor_loss = torch.tensor(0.0, device=feat_b.device)

    loss = lambda_proto * proto_loss + lambda_anchor * anchor_loss
    return loss, {
        "proto_loss": float(proto_loss.detach().item()),
        "anchor_loss": float(anchor_loss.detach().item()),
    }


def run_capspp_online(
    features,
    logits_static,
    labels,
    cls_head,
    source_prototypes,
    valid_mask,
    batch_size,
    alpha,
    tau_conf,
    tau_margin,
    tau_entropy,
    momentum,
    require_proto_agreement,
    adapter_dim,
    adapter_lr,
    adapter_weight_decay,
    adapter_steps,
    lambda_proto,
    lambda_anchor,
    stable_classes,
):
    tau_margin = 0.0 if tau_margin is None else tau_margin
    dim = features.shape[1]
    device = features.device
    adapter = ResidualAdapter(dim, bottleneck=adapter_dim).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=adapter_lr, weight_decay=adapter_weight_decay
    )
    cls_head = cls_head.to(device)
    cls_head.eval()
    for p in cls_head.parameters():
        p.requires_grad_(False)

    target_prototypes = source_prototypes.clone()
    all_preds = []
    accepted_total = 0
    accepted_by_class = defaultdict(int)
    loss_sums = defaultdict(float)
    update_steps = 0

    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_b = features[start:end]

        # Predict before updating on this batch.
        with torch.no_grad():
            h_prime = adapter(feat_b)
            logits_b = cls_head(h_prime)
            sims = caps.prototype_cosine(h_prime, target_prototypes, valid_mask)
            scores = logits_b + alpha * sims
            preds = scores.argmax(dim=1)
            all_preds.append(preds.cpu())
            mask, pseudo, _ = caps.select_update_samples(
                logits_b,
                h_prime,
                target_prototypes,
                valid_mask,
                tau_conf,
                tau_margin,
                tau_entropy,
                require_proto_agreement,
            )

        accepted_total += int(mask.sum().item())

        # Prototype update uses current adapted features, but affects future batches.
        updated = caps.update_prototypes_ema(
            target_prototypes, h_prime.detach(), pseudo, mask, momentum
        )
        for c, count in updated:
            accepted_by_class[c] += count

        if mask.sum().item() == 0:
            continue

        for _ in range(adapter_steps):
            optimizer.zero_grad()
            loss, loss_info = adapter_losses(
                adapter,
                feat_b,
                pseudo.detach(),
                mask.detach(),
                target_prototypes,
                stable_classes,
                lambda_proto,
                lambda_anchor,
            )
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            update_steps += 1
            loss_sums["loss"] += float(loss.detach().item())
            loss_sums["proto_loss"] += loss_info["proto_loss"]
            loss_sums["anchor_loss"] += loss_info["anchor_loss"]

    preds = torch.cat(all_preds).numpy()
    avg_losses = {
        key: value / max(update_steps, 1)
        for key, value in loss_sums.items()
    }
    return preds, {
        "accepted_total": accepted_total,
        "accepted_rate": accepted_total / max(n, 1),
        "accepted_by_class": dict(sorted(accepted_by_class.items())),
        "num_updated_classes": len(accepted_by_class),
        "adapter_update_steps": update_steps,
        **avg_losses,
    }


def main():
    parser = argparse.ArgumentParser(description="CAPS++ adapter MVP for TLS-Year22.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", default="outputs/capspp_adapter_tls22")
    parser.add_argument("--alphas", default="2,5")
    parser.add_argument("--tau-confs", default="0.8,0.9")
    parser.add_argument("--momentums", default="0.9")
    parser.add_argument("--adapter-lrs", default="0.0003,0.001")
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-weight-decay", type=float, default=0.0)
    parser.add_argument("--adapter-steps", type=int, default=1)
    parser.add_argument("--lambda-proto", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--tau-margin", type=float, default=0.0)
    parser.add_argument("--tau-entropy", type=float, default=None)
    parser.add_argument("--require-proto-agreement", type=caps.parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--grid-device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--bad-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--min-prototype-support", type=int, default=1)
    parser.add_argument("--top-k-confusions", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    alphas = parse_float_list(args.alphas)
    tau_confs = parse_float_list(args.tau_confs)
    momentums = parse_float_list(args.momentums)
    adapter_lrs = parse_float_list(args.adapter_lrs)
    bad_classes = proto.parse_class_list(args.bad_classes, proto.DEFAULT_BAD_CLASSES)
    stable_classes = proto.parse_class_list(
        args.stable_classes, proto.DEFAULT_STABLE_CLASSES
    )
    stable_set = set(stable_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    eval_cfg = proto.load_config(args.config)
    print("Loading frozen source model...")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes
    print(f"Num classes: {num_classes}")
    print(f"Online batch size: {args.batch_size}")

    print(f"Building reference loader: {args.reference_period}")
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    print(f"Building target loader: {args.target_period}")
    target_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)

    ref = proto.collect_outputs(
        model, ref_loader, device, f"Reference {args.reference_period}"
    )
    target = proto.collect_outputs(
        model, target_loader, device, f"Target {args.target_period}"
    )

    source_prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref["features"], ref["labels"], num_classes, args.min_prototype_support
    )
    print(
        f"Built source prototypes for {int(valid_mask.sum())}/{num_classes} classes "
        f"(min_support={args.min_prototype_support})."
    )

    labels = target["labels"]
    static_preds = target["logits"].argmax(dim=1).numpy()
    static_metrics = proto.summarize_predictions(
        labels, static_preds, bad_classes, stable_classes
    )

    grid_device = torch.device("cpu") if args.grid_device == "cpu" else device
    print(f"Running CAPS++ grid on: {grid_device}")
    try:
        features_grid = target["features"].to(grid_device)
        source_prototypes_grid = source_prototypes.to(grid_device)
        valid_mask_grid = valid_mask.to(grid_device)
    except torch.OutOfMemoryError:
        if grid_device.type == "cuda":
            print("CUDA OOM while staging CAPS++ tensors; falling back to CPU grid.")
            torch.cuda.empty_cache()
            grid_device = torch.device("cpu")
            features_grid = target["features"]
            source_prototypes_grid = source_prototypes
            valid_mask_grid = valid_mask
        else:
            raise

    cls_head = model.cls_head.to(grid_device)
    param_grid = [
        (alpha, tau_conf, momentum, adapter_lr)
        for alpha in alphas
        for tau_conf in tau_confs
        for momentum in momentums
        for adapter_lr in adapter_lrs
    ]

    summary_rows = [{
        "method": "static",
        "alpha": "",
        "tau_conf": "",
        "momentum": "",
        "adapter_lr": "",
        "accepted_rate": "",
        "num_updated_classes": "",
        "adapter_update_steps": "",
        **static_metrics,
    }]
    best = {
        "score": -float("inf"),
        "params": None,
        "preds": None,
        "metrics": None,
        "stats": None,
    }

    pbar = tqdm(param_grid, desc="CAPS++ grid")
    for alpha, tau_conf, momentum, adapter_lr in pbar:
        preds, stats = run_capspp_online(
            features_grid,
            target["logits"],
            labels,
            cls_head,
            source_prototypes_grid,
            valid_mask_grid,
            args.batch_size,
            alpha,
            tau_conf,
            args.tau_margin,
            args.tau_entropy,
            momentum,
            args.require_proto_agreement,
            args.adapter_dim,
            adapter_lr,
            args.adapter_weight_decay,
            args.adapter_steps,
            args.lambda_proto,
            args.lambda_anchor,
            stable_set,
        )
        metrics = proto.summarize_predictions(labels, preds, bad_classes, stable_classes)
        summary_rows.append({
            "method": "capspp_adapter",
            "alpha": alpha,
            "tau_conf": tau_conf,
            "momentum": momentum,
            "adapter_lr": adapter_lr,
            "accepted_rate": stats["accepted_rate"],
            "num_updated_classes": stats["num_updated_classes"],
            "adapter_update_steps": stats["adapter_update_steps"],
            **metrics,
        })
        pbar.set_postfix({
            "a": alpha,
            "tau": tau_conf,
            "lr": adapter_lr,
            "bad_f1": f"{metrics['bad_macro_f1']:.4f}",
            "accpt": f"{stats['accepted_rate']:.3f}",
        })

        bad_f1 = metrics["bad_macro_f1"] or -1.0
        stable_f1 = metrics["stable_macro_f1"] or -1.0
        score = bad_f1 + 1e-3 * stable_f1
        if score > best["score"]:
            best.update({
                "score": score,
                "params": {
                    "alpha": alpha,
                    "tau_conf": tau_conf,
                    "momentum": momentum,
                    "adapter_lr": adapter_lr,
                },
                "preds": preds,
                "metrics": metrics,
                "stats": stats,
            })

    proto.write_csv(
        os.path.join(args.output_dir, "results_by_params.csv"),
        summary_rows,
        [
            "method", "alpha", "tau_conf", "momentum", "adapter_lr",
            "accepted_rate", "num_updated_classes", "adapter_update_steps",
            "overall_accuracy", "overall_macro_f1",
            "bad_accuracy", "bad_macro_f1", "bad_support",
            "stable_accuracy", "stable_macro_f1", "stable_support",
        ],
    )

    static_by_class = proto.per_class_f1(labels, static_preds, num_classes)
    best_by_class = proto.per_class_f1(labels, best["preds"], num_classes)
    accepted_by_class = best["stats"]["accepted_by_class"]
    per_rows = []
    for s, b in zip(static_by_class, best_by_class):
        c = s["class"]
        group = "bad" if c in bad_classes else "stable" if c in stable_classes else "other"
        per_rows.append({
            "class": c,
            "group": group,
            "reference_support": int(proto_support[c]),
            "target_support": s["support"],
            "accepted_updates": int(accepted_by_class.get(c, 0)),
            "static_f1": s["f1"],
            "best_capspp_f1": b["f1"],
            "delta_f1": b["f1"] - s["f1"],
            "static_recall": s["recall"],
            "best_capspp_recall": b["recall"],
            "delta_recall": b["recall"] - s["recall"],
        })
    slug = caps.period_slug(args.target_period)
    proto.write_csv(
        os.path.join(args.output_dir, f"per_class_metrics_{slug}.csv"),
        per_rows,
        [
            "class", "group", "reference_support", "target_support",
            "accepted_updates",
            "static_f1", "best_capspp_f1", "delta_f1",
            "static_recall", "best_capspp_recall", "delta_recall",
        ],
    )

    before_conf = proto.top_bad_confusions(
        labels, static_preds, bad_classes, args.top_k_confusions
    )
    after_conf = proto.top_bad_confusions(
        labels, best["preds"], bad_classes, args.top_k_confusions
    )
    for row in before_conf:
        row["method"] = "static"
    for row in after_conf:
        row["method"] = "best_capspp_adapter"
    proto.write_csv(
        os.path.join(args.output_dir, f"bad_confusion_before_after_{slug}.csv"),
        before_conf + after_conf,
        [
            "method", "true_class", "pred_class", "rank_for_bad_class",
            "confusion_count", "confusion_rate", "support",
        ],
    )

    caps.write_update_stats(
        os.path.join(args.output_dir, "accepted_updates_by_class.csv"),
        accepted_by_class,
        proto_support,
        labels,
    )
    caps.write_pair_summary(
        os.path.join(args.output_dir, f"pair_summary_{slug}.csv"),
        labels,
        static_preds,
        best["preds"],
        before_conf,
    )

    meta = {
        "method": "CAPS++ adapter MVP",
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "adapter_dim": args.adapter_dim,
        "adapter_steps": args.adapter_steps,
        "lambda_proto": args.lambda_proto,
        "lambda_anchor": args.lambda_anchor,
        "best_params_by_bad_macro_f1": best["params"],
        "best_metrics": best["metrics"],
        "best_update_stats": {
            "accepted_total": best["stats"]["accepted_total"],
            "accepted_rate": best["stats"]["accepted_rate"],
            "num_updated_classes": best["stats"]["num_updated_classes"],
            "adapter_update_steps": best["stats"]["adapter_update_steps"],
            "loss": best["stats"].get("loss"),
            "proto_loss": best["stats"].get("proto_loss"),
            "anchor_loss": best["stats"].get("anchor_loss"),
        },
        "bad_classes": bad_classes,
        "stable_classes": stable_classes,
        "num_classes": num_classes,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    p = best["params"]
    print("\n=== CAPS++ adapter summary ===")
    print(
        f"{'static':<20} macro_f1={static_metrics['overall_macro_f1']:.4f} "
        f"bad_f1={static_metrics['bad_macro_f1']:.4f} "
        f"stable_f1={static_metrics['stable_macro_f1']:.4f}"
    )
    print(
        f"{'best_capspp':<20} alpha={p['alpha']} tau={p['tau_conf']} "
        f"m={p['momentum']} lr={p['adapter_lr']} "
        f"macro_f1={best['metrics']['overall_macro_f1']:.4f} "
        f"bad_f1={best['metrics']['bad_macro_f1']:.4f} "
        f"stable_f1={best['metrics']['stable_macro_f1']:.4f} "
        f"accepted={best['stats']['accepted_rate']:.3f}"
    )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
