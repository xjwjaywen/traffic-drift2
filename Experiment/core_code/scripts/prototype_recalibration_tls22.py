"""
Prototype-aware recalibration for TLS-Year22 class-conditional collapse.

This is a frozen-model diagnostic/MVP:
  1. Build class prototypes from a reference period using frozen encoder features.
  2. Re-score target-period logits with source/reference prototype similarity.
  3. Report overall, bad-class, stable-class, and top confusion-pair effects.

Usage:
    python scripts/prototype_recalibration_tls22.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --reference-period M-2022-4 \
        --target-period M-2022-12 \
        --output-dir outputs/prototype_recalibration_tls22
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

# Allow running as: python scripts/prototype_recalibration_tls22.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.models import TTATCModel
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


DEFAULT_BAD_CLASSES = [
    5, 6, 10, 11, 12, 14, 26, 28, 38, 47,
    48, 56, 66, 69, 104, 109, 118, 139, 164, 174,
]
DEFAULT_STABLE_CLASSES = [
    8, 15, 44, 57, 59, 62, 64, 76, 94, 98,
    99, 107, 113, 119, 128, 130, 131, 132, 144, 145,
]


def parse_class_list(value, default):
    """Parse comma/space-separated class ids."""
    if value is None or value.strip() == "":
        return list(default)
    parts = value.replace(",", " ").split()
    return [int(x) for x in parts]


def load_source_model(checkpoint_path, device):
    """Load trained source model from a train.py checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt["num_classes"]


def make_test_loader(eval_cfg, period):
    """Build the CESNET test loader for one period."""
    data_cfg = dict(eval_cfg["data"])
    data_cfg["test_period"] = period
    _, _, test_loader, num_classes = build_dataloaders(data_cfg)
    return test_loader, num_classes


@torch.no_grad()
def collect_outputs(model, loader, device, desc, keep_ppi=False):
    """Collect frozen features, logits, and labels for one period."""
    all_features = []
    all_logits = []
    all_labels = []
    all_ppi = []

    model.eval()
    for batch in tqdm(loader, desc=desc):
        ppi = batch["ppi"].to(device)
        labels = batch["label"]
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)

        logits, features = model(ppi, flow_stats, return_repr=True)
        all_features.append(features.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        if keep_ppi:
            all_ppi.append(batch["ppi"])

    result = {
        "features": torch.cat(all_features, dim=0),
        "logits": torch.cat(all_logits, dim=0),
        "labels": torch.cat(all_labels, dim=0).numpy(),
    }
    if keep_ppi:
        result["ppi"] = torch.cat(all_ppi, dim=0)
    return result


def build_prototypes(features, labels, num_classes, min_support):
    """
    Build one mean feature prototype per class from reference-period features.

    Returns:
        prototypes: Tensor(C, D), mean raw feature per class
        support: ndarray(C,), reference support
        valid_mask: Tensor(C,), true when support >= min_support
    """
    dim = features.shape[1]
    prototypes = torch.zeros(num_classes, dim, dtype=features.dtype)
    support = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        support[c] = len(idx)
        if len(idx) >= min_support:
            prototypes[c] = features[idx].mean(dim=0)

    valid_mask = torch.tensor(support >= min_support)
    return prototypes, support, valid_mask


def prototype_scores(features, prototypes, valid_mask, invalid_value=0.0):
    """Compute -squared L2 distance between L2-normalized features/prototypes."""
    feat_n = F.normalize(features, dim=1)
    proto_n = F.normalize(prototypes, dim=1)
    # For normalized vectors: -||h-p||^2 = 2*cos(h,p)-2.
    scores = 2.0 * feat_n @ proto_n.t() - 2.0
    if valid_mask is not None:
        scores[:, ~valid_mask] = invalid_value
    return scores


def subset_accuracy(labels, preds, classes):
    """Compute accuracy on samples whose true label is in classes."""
    classes = set(classes)
    mask = np.array([int(y) in classes for y in labels], dtype=bool)
    if mask.sum() == 0:
        return {"accuracy": None, "support": 0}
    return {
        "accuracy": float((labels[mask] == preds[mask]).mean()),
        "support": int(mask.sum()),
    }


def group_macro_f1_from_report(report, classes):
    """Average global per-class F1 over a class group."""
    values = []
    support = 0
    for c in classes:
        item = report.get(str(c), {})
        values.append(float(item.get("f1-score", 0.0)))
        support += int(item.get("support", 0))
    if not values:
        return None, 0
    return float(np.mean(values)), support


def summarize_predictions(labels, preds, bad_classes, stable_classes):
    """Return overall/bad/stable metric summary."""
    overall = compute_metrics(labels, preds)
    bad = subset_accuracy(labels, preds, bad_classes)
    stable = subset_accuracy(labels, preds, stable_classes)
    bad_macro_f1, bad_support = group_macro_f1_from_report(
        overall["classification_report"], bad_classes
    )
    stable_macro_f1, stable_support = group_macro_f1_from_report(
        overall["classification_report"], stable_classes
    )
    return {
        "overall_accuracy": overall["accuracy"],
        "overall_macro_f1": overall["macro_f1"],
        "bad_accuracy": bad["accuracy"],
        "bad_macro_f1": bad_macro_f1,
        "bad_support": bad_support,
        "stable_accuracy": stable["accuracy"],
        "stable_macro_f1": stable_macro_f1,
        "stable_support": stable_support,
    }


def per_class_f1(labels, preds, num_classes):
    """Return support and per-class F1 for all classes."""
    report = compute_metrics(labels, preds)["classification_report"]
    rows = []
    for c in range(num_classes):
        item = report.get(str(c), {})
        rows.append({
            "class": c,
            "support": int(item.get("support", 0)),
            "precision": float(item.get("precision", 0.0)),
            "recall": float(item.get("recall", 0.0)),
            "f1": float(item.get("f1-score", 0.0)),
        })
    return rows


def top_bad_confusions(labels, preds, bad_classes, top_k):
    """Top non-correct predictions per bad class."""
    rows = []
    for c in bad_classes:
        mask = labels == c
        support = int(mask.sum())
        if support == 0:
            continue
        counts = defaultdict(int)
        for pred in preds[mask]:
            pred = int(pred)
            if pred != c:
                counts[pred] += 1
        for rank, (target, count) in enumerate(
            sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k], start=1
        ):
            rows.append({
                "true_class": c,
                "pred_class": target,
                "rank_for_bad_class": rank,
                "confusion_count": count,
                "confusion_rate": count / support,
                "support": support,
            })
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Frozen prototype-aware recalibration for TLS-Year22."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", default="outputs/prototype_recalibration_tls22")
    parser.add_argument(
        "--alphas",
        default="0,0.05,0.1,0.3,0.5,1.0,2.0,5.0",
        help="Comma/space-separated alpha values for logit + alpha*prototype_score.",
    )
    parser.add_argument("--bad-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--min-prototype-support", type=int, default=1)
    parser.add_argument("--top-k-confusions", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    alphas = [float(x) for x in args.alphas.replace(",", " ").split()]
    bad_classes = parse_class_list(args.bad_classes, DEFAULT_BAD_CLASSES)
    stable_classes = parse_class_list(args.stable_classes, DEFAULT_STABLE_CLASSES)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    eval_cfg = load_config(args.config)
    print("Loading frozen source model...")
    model, _, num_classes = load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes
    print(f"Num classes: {num_classes}")

    print(f"Building reference loader: {args.reference_period}")
    ref_loader, _ = make_test_loader(eval_cfg, args.reference_period)
    print(f"Building target loader: {args.target_period}")
    target_loader, _ = make_test_loader(eval_cfg, args.target_period)

    ref = collect_outputs(model, ref_loader, device, f"Reference {args.reference_period}")
    target = collect_outputs(model, target_loader, device, f"Target {args.target_period}")

    prototypes, proto_support, valid_mask = build_prototypes(
        ref["features"], ref["labels"], num_classes, args.min_prototype_support
    )
    print(
        f"Built prototypes for {int(valid_mask.sum())}/{num_classes} classes "
        f"(min_support={args.min_prototype_support})."
    )

    logits = target["logits"]
    labels = target["labels"]
    proto_for_logit = prototype_scores(
        target["features"], prototypes, valid_mask, invalid_value=0.0
    )
    proto_only_scores = prototype_scores(
        target["features"], prototypes, valid_mask, invalid_value=-1e9
    )

    static_preds = logits.argmax(dim=1).numpy()
    proto_only_preds = proto_only_scores.argmax(dim=1).numpy()

    summary_rows = []
    predictions_by_name = {
        "static": static_preds,
        "prototype_only": proto_only_preds,
    }

    for method_name, preds in predictions_by_name.items():
        row = {
            "method": method_name,
            "alpha": "",
            **summarize_predictions(labels, preds, bad_classes, stable_classes),
        }
        summary_rows.append(row)

    best_alpha = None
    best_score = -float("inf")
    best_preds = None
    for alpha in alphas:
        preds = (logits + alpha * proto_for_logit).argmax(dim=1).numpy()
        metrics = summarize_predictions(labels, preds, bad_classes, stable_classes)
        row = {"method": "logit_plus_proto", "alpha": alpha, **metrics}
        summary_rows.append(row)
        # Primary MVP criterion: maximize bad-class F1, tie-break by stable F1.
        bad_f1 = metrics["bad_macro_f1"] if metrics["bad_macro_f1"] is not None else -1.0
        stable_f1 = (
            metrics["stable_macro_f1"] if metrics["stable_macro_f1"] is not None else -1.0
        )
        score = bad_f1 + 1e-3 * stable_f1
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_preds = preds

    summary_fields = [
        "method", "alpha",
        "overall_accuracy", "overall_macro_f1",
        "bad_accuracy", "bad_macro_f1", "bad_support",
        "stable_accuracy", "stable_macro_f1", "stable_support",
    ]
    write_csv(
        os.path.join(args.output_dir, "results_by_alpha.csv"),
        summary_rows,
        summary_fields,
    )

    per_rows = []
    static_by_class = per_class_f1(labels, static_preds, num_classes)
    proto_by_class = per_class_f1(labels, best_preds, num_classes)
    for s, p in zip(static_by_class, proto_by_class):
        c = s["class"]
        group = "bad" if c in bad_classes else "stable" if c in stable_classes else "other"
        per_rows.append({
            "class": c,
            "group": group,
            "reference_support": int(proto_support[c]),
            "target_support": s["support"],
            "static_f1": s["f1"],
            "best_proto_f1": p["f1"],
            "delta_f1": p["f1"] - s["f1"],
            "static_recall": s["recall"],
            "best_proto_recall": p["recall"],
            "delta_recall": p["recall"] - s["recall"],
        })
    write_csv(
        os.path.join(args.output_dir, "per_class_metrics_m12.csv"),
        per_rows,
        [
            "class", "group", "reference_support", "target_support",
            "static_f1", "best_proto_f1", "delta_f1",
            "static_recall", "best_proto_recall", "delta_recall",
        ],
    )

    before_conf = top_bad_confusions(
        labels, static_preds, bad_classes, args.top_k_confusions
    )
    after_conf = top_bad_confusions(
        labels, best_preds, bad_classes, args.top_k_confusions
    )
    for row in before_conf:
        row["method"] = "static"
        row["alpha"] = ""
    for row in after_conf:
        row["method"] = "best_logit_plus_proto"
        row["alpha"] = best_alpha
    write_csv(
        os.path.join(args.output_dir, "bad_confusion_before_after_m12.csv"),
        before_conf + after_conf,
        [
            "method", "alpha", "true_class", "pred_class", "rank_for_bad_class",
            "confusion_count", "confusion_rate", "support",
        ],
    )

    meta = {
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "alphas": alphas,
        "best_alpha_by_bad_macro_f1": best_alpha,
        "bad_classes": bad_classes,
        "stable_classes": stable_classes,
        "min_prototype_support": args.min_prototype_support,
        "num_valid_prototypes": int(valid_mask.sum()),
        "num_classes": num_classes,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Prototype recalibration summary ===")
    for row in summary_rows:
        if row["method"] == "static" or row.get("alpha") == best_alpha:
            print(
                f"{row['method']:<18} alpha={str(row['alpha']):<5} "
                f"macro_f1={row['overall_macro_f1']:.4f} "
                f"bad_f1={row['bad_macro_f1']:.4f} "
                f"stable_f1={row['stable_macro_f1']:.4f}"
            )
    print(f"Best alpha by bad macro-F1: {best_alpha}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
