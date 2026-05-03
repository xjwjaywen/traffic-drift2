"""
CAPS prototype-only MVP for TLS-Year22.

CAPS here means Collapse-Aware Prototype Stabilization. This first MVP does not
update the backbone. It maintains target-side EMA prototypes with conservative
confidence gates, then re-scores frozen logits with target prototype similarity.

Online protocol:
  For each target batch:
    1. Predict with current target prototypes.
    2. Select high-confidence pseudo-labeled samples from the frozen classifier.
    3. Update the corresponding target prototypes by EMA for future batches.

This tests whether source prototypes need target-side adaptation to help
class-conditional collapse, while avoiding direct same-batch label leakage.

Usage:
    python scripts/caps_target_prototype_tls22.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --reference-period M-2022-4 \
        --target-period M-2022-12 \
        --output-dir outputs/caps_target_prototype_tls22
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Allow running as: python scripts/caps_target_prototype_tls22.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


def parse_float_list(value):
    return [float(x) for x in value.replace(",", " ").split()]


def parse_bool(value):
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def period_slug(period):
    return period.lower().replace("m-2022-", "m").replace("-", "_")


def prototype_cosine(features, prototypes, valid_mask=None):
    feat_n = F.normalize(features, dim=1)
    proto_n = F.normalize(prototypes, dim=1)
    sims = feat_n @ proto_n.t()
    if valid_mask is not None:
        sims[:, ~valid_mask] = -1e9
    return sims


def prototype_margin(sims):
    top2 = torch.topk(sims, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def select_update_samples(
    logits,
    features,
    target_prototypes,
    valid_mask,
    tau_conf,
    tau_margin,
    tau_entropy,
    require_proto_agreement,
):
    probs = F.softmax(logits, dim=1)
    conf, pseudo = probs.max(dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    sims = prototype_cosine(features, target_prototypes, valid_mask)
    nearest = sims.argmax(dim=1)
    margin = prototype_margin(sims)

    mask = conf >= tau_conf
    if tau_entropy is not None:
        mask = mask & (entropy <= tau_entropy)
    if tau_margin > 0:
        mask = mask & (margin >= tau_margin)
    if require_proto_agreement:
        mask = mask & (pseudo == nearest)

    return mask, pseudo, {
        "mean_conf": float(conf.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "mean_proto_margin": float(margin.mean().item()),
    }


def update_prototypes_ema(prototypes, features, pseudo, mask, momentum):
    updated_classes = []
    if mask.sum().item() == 0:
        return updated_classes

    selected_features = features[mask]
    selected_pseudo = pseudo[mask]
    for c in torch.unique(selected_pseudo).tolist():
        class_mask = selected_pseudo == c
        mean_feature = selected_features[class_mask].mean(dim=0)
        prototypes[c] = momentum * prototypes[c] + (1.0 - momentum) * mean_feature
        updated_classes.append((int(c), int(class_mask.sum().item())))
    return updated_classes


def run_caps_online(
    features,
    logits,
    labels,
    source_prototypes,
    valid_mask,
    batch_size,
    alpha,
    tau_conf,
    tau_margin,
    tau_entropy,
    momentum,
    require_proto_agreement,
):
    target_prototypes = source_prototypes.clone()
    all_preds = []
    accepted_total = 0
    batch_stats = []
    accepted_by_class = defaultdict(int)

    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_b = features[start:end]
        logit_b = logits[start:end]

        # Causal online prediction: current prototypes affect this batch;
        # this batch updates prototypes only for future batches.
        sims = prototype_cosine(feat_b, target_prototypes, valid_mask)
        scores = logit_b + alpha * sims
        preds = scores.argmax(dim=1)
        all_preds.append(preds.cpu())

        mask, pseudo, raw_stats = select_update_samples(
            logit_b,
            feat_b,
            target_prototypes,
            valid_mask,
            tau_conf,
            tau_margin,
            tau_entropy,
            require_proto_agreement,
        )
        accepted_total += int(mask.sum().item())
        updated = update_prototypes_ema(
            target_prototypes, feat_b, pseudo, mask, momentum
        )
        for c, count in updated:
            accepted_by_class[c] += count
        batch_stats.append({
            "batch_start": start,
            "batch_end": end,
            "accepted": int(mask.sum().item()),
            "batch_size": end - start,
            **raw_stats,
        })

    preds = torch.cat(all_preds).numpy()
    return preds, {
        "accepted_total": accepted_total,
        "accepted_rate": accepted_total / max(n, 1),
        "accepted_by_class": dict(sorted(accepted_by_class.items())),
        "num_updated_classes": len(accepted_by_class),
        "batch_stats": batch_stats,
    }


def write_update_stats(path, accepted_by_class, proto_support, labels):
    rows = []
    for c, count in sorted(accepted_by_class.items()):
        target_support = int((labels == c).sum())
        rows.append({
            "class": c,
            "accepted": count,
            "reference_support": int(proto_support[c]),
            "target_support": target_support,
            "accepted_per_target_support": count / target_support if target_support else 0.0,
        })
    proto.write_csv(
        path,
        rows,
        [
            "class", "accepted", "reference_support",
            "target_support", "accepted_per_target_support",
        ],
    )


def pair_confusion_rate(labels, preds, true_class, pred_class):
    mask = labels == true_class
    support = int(mask.sum())
    if support == 0:
        return 0, 0.0, 0
    count = int((preds[mask] == pred_class).sum())
    return count, count / support, support


def write_pair_summary(path, labels, static_preds, caps_preds, static_confusion_rows):
    """Compare CAPS against static on static top bad-class confusion pairs."""
    rows = []
    seen = set()
    for row in static_confusion_rows:
        c = int(row["true_class"])
        j = int(row["pred_class"])
        key = (c, j)
        if key in seen:
            continue
        seen.add(key)
        s_count, s_rate, support = pair_confusion_rate(labels, static_preds, c, j)
        c_count, c_rate, _ = pair_confusion_rate(labels, caps_preds, c, j)
        rows.append({
            "true_class": c,
            "pred_class": j,
            "rank_for_bad_class": int(row["rank_for_bad_class"]),
            "support": support,
            "static_count": s_count,
            "static_rate": s_rate,
            "caps_count": c_count,
            "caps_rate": c_rate,
            "delta_count": c_count - s_count,
            "delta_rate": c_rate - s_rate,
        })
    proto.write_csv(
        path,
        rows,
        [
            "true_class", "pred_class", "rank_for_bad_class", "support",
            "static_count", "static_rate", "caps_count", "caps_rate",
            "delta_count", "delta_rate",
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="CAPS target-adaptive prototype MVP for TLS-Year22."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", default="outputs/caps_target_prototype_tls22")
    parser.add_argument("--alphas", default="0,0.5,1,2,5")
    parser.add_argument("--tau-confs", default="0.7,0.8,0.9,0.95")
    parser.add_argument("--momentums", default="0.9,0.99,0.999")
    parser.add_argument("--tau-margin", type=float, default=0.0)
    parser.add_argument("--tau-entropy", type=float, default=None)
    parser.add_argument("--require-proto-agreement", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--grid-device",
        choices=["auto", "cpu"],
        default="auto",
        help="Device for the parameter grid after feature extraction.",
    )
    parser.add_argument("--bad-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--min-prototype-support", type=int, default=1)
    parser.add_argument("--top-k-confusions", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    alphas = parse_float_list(args.alphas)
    tau_confs = parse_float_list(args.tau_confs)
    momentums = parse_float_list(args.momentums)
    bad_classes = proto.parse_class_list(args.bad_classes, proto.DEFAULT_BAD_CLASSES)
    stable_classes = proto.parse_class_list(
        args.stable_classes, proto.DEFAULT_STABLE_CLASSES
    )

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
    batch_size = args.batch_size or int(eval_cfg["data"].get("batch_size", 256))
    print(f"Num classes: {num_classes}")
    print(f"Online batch size: {batch_size}")

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

    features = target["features"]
    logits = target["logits"]
    labels = target["labels"]
    slug = period_slug(args.target_period)
    static_preds = logits.argmax(dim=1).numpy()
    static_metrics = proto.summarize_predictions(
        labels, static_preds, bad_classes, stable_classes
    )

    summary_rows = [{
        "method": "static",
        "alpha": "",
        "tau_conf": "",
        "momentum": "",
        "accepted_rate": "",
        "num_updated_classes": "",
        **static_metrics,
    }]

    best = {
        "score": -float("inf"),
        "alpha": None,
        "tau_conf": None,
        "momentum": None,
        "preds": None,
        "metrics": None,
        "stats": None,
    }

    grid_device = torch.device("cpu") if args.grid_device == "cpu" else device
    print(f"Running CAPS parameter grid on: {grid_device}")
    features_grid = features.to(grid_device)
    logits_grid = logits.to(grid_device)
    source_prototypes_grid = source_prototypes.to(grid_device)
    valid_mask_grid = valid_mask.to(grid_device)

    param_grid = [
        (alpha, tau_conf, momentum)
        for alpha in alphas
        for tau_conf in tau_confs
        for momentum in momentums
    ]
    pbar = tqdm(param_grid, desc="CAPS grid")
    for alpha, tau_conf, momentum in pbar:
        preds, stats = run_caps_online(
            features_grid,
            logits_grid,
            labels,
            source_prototypes_grid,
            valid_mask_grid,
            batch_size,
            alpha,
            tau_conf,
            args.tau_margin,
            args.tau_entropy,
            momentum,
            args.require_proto_agreement,
        )
        metrics = proto.summarize_predictions(
            labels, preds, bad_classes, stable_classes
        )
        summary_rows.append({
            "method": "caps_target_proto",
            "alpha": alpha,
            "tau_conf": tau_conf,
            "momentum": momentum,
            "accepted_rate": stats["accepted_rate"],
            "num_updated_classes": stats["num_updated_classes"],
            **metrics,
        })
        pbar.set_postfix({
            "alpha": alpha,
            "tau": tau_conf,
            "m": momentum,
            "bad_f1": f"{metrics['bad_macro_f1']:.4f}",
            "accpt": f"{stats['accepted_rate']:.3f}",
        })

        bad_f1 = metrics["bad_macro_f1"] or -1.0
        stable_f1 = metrics["stable_macro_f1"] or -1.0
        score = bad_f1 + 1e-3 * stable_f1
        if score > best["score"]:
            best.update({
                "score": score,
                "alpha": alpha,
                "tau_conf": tau_conf,
                "momentum": momentum,
                "preds": preds,
                "metrics": metrics,
                "stats": stats,
            })

    summary_fields = [
        "method", "alpha", "tau_conf", "momentum",
        "accepted_rate", "num_updated_classes",
        "overall_accuracy", "overall_macro_f1",
        "bad_accuracy", "bad_macro_f1", "bad_support",
        "stable_accuracy", "stable_macro_f1", "stable_support",
    ]
    proto.write_csv(
        os.path.join(args.output_dir, "results_by_params.csv"),
        summary_rows,
        summary_fields,
    )

    static_by_class = proto.per_class_f1(labels, static_preds, num_classes)
    best_by_class = proto.per_class_f1(labels, best["preds"], num_classes)
    per_rows = []
    accepted_by_class = best["stats"]["accepted_by_class"]
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
            "best_caps_f1": b["f1"],
            "delta_f1": b["f1"] - s["f1"],
            "static_recall": s["recall"],
            "best_caps_recall": b["recall"],
            "delta_recall": b["recall"] - s["recall"],
        })
    proto.write_csv(
        os.path.join(args.output_dir, f"per_class_metrics_{slug}.csv"),
        per_rows,
        [
            "class", "group", "reference_support", "target_support",
            "accepted_updates",
            "static_f1", "best_caps_f1", "delta_f1",
            "static_recall", "best_caps_recall", "delta_recall",
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
        row["alpha"] = ""
        row["tau_conf"] = ""
        row["momentum"] = ""
    for row in after_conf:
        row["method"] = "best_caps_target_proto"
        row["alpha"] = best["alpha"]
        row["tau_conf"] = best["tau_conf"]
        row["momentum"] = best["momentum"]
    proto.write_csv(
        os.path.join(args.output_dir, f"bad_confusion_before_after_{slug}.csv"),
        before_conf + after_conf,
        [
            "method", "alpha", "tau_conf", "momentum",
            "true_class", "pred_class", "rank_for_bad_class",
            "confusion_count", "confusion_rate", "support",
        ],
    )
    write_pair_summary(
        os.path.join(args.output_dir, f"pair_summary_{slug}.csv"),
        labels,
        static_preds,
        best["preds"],
        before_conf,
    )

    write_update_stats(
        os.path.join(args.output_dir, "accepted_updates_by_class.csv"),
        accepted_by_class,
        proto_support,
        labels,
    )

    meta = {
        "method": "CAPS prototype-only MVP",
        "online_protocol": "predict with current prototypes, then update for future batches",
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "alphas": alphas,
        "tau_confs": tau_confs,
        "momentums": momentums,
        "tau_margin": args.tau_margin,
        "tau_entropy": args.tau_entropy,
        "require_proto_agreement": args.require_proto_agreement,
        "batch_size": batch_size,
        "best_alpha_by_bad_macro_f1": best["alpha"],
        "best_tau_conf_by_bad_macro_f1": best["tau_conf"],
        "best_momentum_by_bad_macro_f1": best["momentum"],
        "best_metrics": best["metrics"],
        "best_update_stats": {
            "accepted_total": best["stats"]["accepted_total"],
            "accepted_rate": best["stats"]["accepted_rate"],
            "num_updated_classes": best["stats"]["num_updated_classes"],
        },
        "bad_classes": bad_classes,
        "stable_classes": stable_classes,
        "num_classes": num_classes,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== CAPS target prototype summary ===")
    print(
        f"{'static':<20} alpha={'':<4} tau={'':<4} m={'':<5} "
        f"macro_f1={static_metrics['overall_macro_f1']:.4f} "
        f"bad_f1={static_metrics['bad_macro_f1']:.4f} "
        f"stable_f1={static_metrics['stable_macro_f1']:.4f}"
    )
    print(
        f"{'best_caps':<20} alpha={best['alpha']:<4} "
        f"tau={best['tau_conf']:<4} m={best['momentum']:<5} "
        f"macro_f1={best['metrics']['overall_macro_f1']:.4f} "
        f"bad_f1={best['metrics']['bad_macro_f1']:.4f} "
        f"stable_f1={best['metrics']['stable_macro_f1']:.4f} "
        f"accepted={best['stats']['accepted_rate']:.3f}"
    )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
