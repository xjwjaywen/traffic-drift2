"""
Oracle class-pair-aware recalibration for TLS-Year22.

This is an upper-bound diagnostic, not a deployable method. It uses target-label
confusion pairs from bad_confusion_targets.csv to test whether explicitly
counteracting bad-class absorber pairs can recover class-conditional collapse.

Score update for each bad class c:
    sim_k = cos(normalize(h), normalize(p_k))
    z'_c = z_c + beta * mean_{j in A_c} (sim_c - sim_j)
    z'_k = z_k for all other classes

where A_c is the oracle top absorber set for bad class c.

Usage:
    python scripts/oracle_pair_recalibration_tls22.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --pair-csv outputs/class_conditional_drift_tls22_summary/bad_confusion_targets.csv \
        --reference-period M-2022-4 \
        --target-period M-2022-12 \
        --output-dir outputs/oracle_pair_recalibration_tls22
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

# Allow running as: python scripts/oracle_pair_recalibration_tls22.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


def parse_float_list(value):
    return [float(x) for x in value.replace(",", " ").split()]


def parse_int_list(value):
    return [int(x) for x in value.replace(",", " ").split()]


def read_oracle_pairs(pair_csv, target_period, bad_classes, max_rank):
    """Read target-label oracle bad->absorber pairs."""
    bad_set = set(bad_classes)
    pairs = defaultdict(list)
    rows = []
    with open(pair_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("period") != target_period:
                continue
            true_class = int(float(row["true_class"]))
            pred_class = int(float(row["pred_class"]))
            rank = int(float(row["rank_for_bad_class"]))
            if true_class not in bad_set or rank > max_rank:
                continue
            pairs[true_class].append((rank, pred_class))
            rows.append({
                "true_class": true_class,
                "pred_class": pred_class,
                "oracle_rank": rank,
                "oracle_confusion_count": int(float(row["confusion_count"])),
                "oracle_confusion_rate": float(row["confusion_rate"]),
            })

    out = {}
    for c, ranked in pairs.items():
        out[c] = [p for _, p in sorted(ranked)]
    return out, sorted(rows, key=lambda r: (r["true_class"], r["oracle_rank"]))


def cosine_scores(features, prototypes, valid_mask):
    """Compute cosine similarity from features to all prototypes."""
    feat_n = F.normalize(features, dim=1)
    proto_n = F.normalize(prototypes, dim=1)
    scores = feat_n @ proto_n.t()
    if valid_mask is not None:
        scores[:, ~valid_mask] = 0.0
    return scores


def apply_pair_recalibration(logits, sims, oracle_pairs, beta, top_absorbers):
    """Apply oracle pairwise margin correction to bad-class logits."""
    adjusted = logits.clone()
    for c, absorbers in oracle_pairs.items():
        selected = absorbers[:top_absorbers]
        if not selected:
            continue
        margins = sims[:, c].unsqueeze(1) - sims[:, selected]
        adjusted[:, c] = adjusted[:, c] + beta * margins.mean(dim=1)
    return adjusted


def pair_confusion_rate(labels, preds, true_class, pred_class):
    mask = labels == true_class
    support = int(mask.sum())
    if support == 0:
        return 0, 0.0, 0
    count = int((preds[mask] == pred_class).sum())
    return count, count / support, support


def make_pair_summary(labels, static_preds, best_preds, oracle_rows, beta, top_absorbers):
    """Summarize before/after rates for oracle pairs."""
    rows = []
    seen = set()
    for row in oracle_rows:
        c = row["true_class"]
        j = row["pred_class"]
        key = (c, j)
        if key in seen:
            continue
        seen.add(key)
        s_count, s_rate, support = pair_confusion_rate(labels, static_preds, c, j)
        b_count, b_rate, _ = pair_confusion_rate(labels, best_preds, c, j)
        rows.append({
            **row,
            "best_beta": beta,
            "best_top_absorbers": top_absorbers,
            "support": support,
            "static_count": s_count,
            "static_rate": s_rate,
            "best_count": b_count,
            "best_rate": b_rate,
            "delta_count": b_count - s_count,
            "delta_rate": b_rate - s_rate,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Oracle class-pair-aware recalibration diagnostic for TLS-Year22."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--pair-csv",
        default="outputs/class_conditional_drift_tls22_summary/bad_confusion_targets.csv",
        help="CSV from summarize_class_conditional_drift.py; uses target labels.",
    )
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", default="outputs/oracle_pair_recalibration_tls22")
    parser.add_argument("--betas", default="0,0.1,0.5,1,2,5,10")
    parser.add_argument("--top-absorbers", default="1,3,5")
    parser.add_argument("--bad-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--min-prototype-support", type=int, default=1)
    parser.add_argument("--top-k-confusions", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    betas = parse_float_list(args.betas)
    top_absorber_values = parse_int_list(args.top_absorbers)
    max_top_absorbers = max(top_absorber_values)
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
    print("WARNING: oracle pairs come from target-label confusion; diagnostic only.")

    eval_cfg = proto.load_config(args.config)
    print("Loading frozen source model...")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes
    print(f"Num classes: {num_classes}")

    oracle_pairs, oracle_rows = read_oracle_pairs(
        args.pair_csv, args.target_period, bad_classes, max_top_absorbers
    )
    if not oracle_pairs:
        raise ValueError(
            f"No oracle pairs found in {args.pair_csv} for {args.target_period}."
        )
    print(
        f"Loaded oracle pairs for {len(oracle_pairs)} bad classes "
        f"(top_absorbers up to {max_top_absorbers})."
    )

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

    prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref["features"], ref["labels"], num_classes, args.min_prototype_support
    )
    print(
        f"Built prototypes for {int(valid_mask.sum())}/{num_classes} classes "
        f"(min_support={args.min_prototype_support})."
    )

    labels = target["labels"]
    logits = target["logits"]
    sims = cosine_scores(target["features"], prototypes, valid_mask)
    static_preds = logits.argmax(dim=1).numpy()

    summary_rows = [{
        "method": "static",
        "beta": "",
        "top_absorbers": "",
        **proto.summarize_predictions(labels, static_preds, bad_classes, stable_classes),
    }]

    best = {
        "score": -float("inf"),
        "beta": None,
        "top_absorbers": None,
        "preds": None,
        "metrics": None,
    }
    for top_absorbers in top_absorber_values:
        for beta in betas:
            adjusted = apply_pair_recalibration(
                logits, sims, oracle_pairs, beta, top_absorbers
            )
            preds = adjusted.argmax(dim=1).numpy()
            metrics = proto.summarize_predictions(
                labels, preds, bad_classes, stable_classes
            )
            summary_rows.append({
                "method": "oracle_pair",
                "beta": beta,
                "top_absorbers": top_absorbers,
                **metrics,
            })
            bad_f1 = metrics["bad_macro_f1"] or -1.0
            stable_f1 = metrics["stable_macro_f1"] or -1.0
            score = bad_f1 + 1e-3 * stable_f1
            if score > best["score"]:
                best.update({
                    "score": score,
                    "beta": beta,
                    "top_absorbers": top_absorbers,
                    "preds": preds,
                    "metrics": metrics,
                })

    summary_fields = [
        "method", "beta", "top_absorbers",
        "overall_accuracy", "overall_macro_f1",
        "bad_accuracy", "bad_macro_f1", "bad_support",
        "stable_accuracy", "stable_macro_f1", "stable_support",
    ]
    proto.write_csv(
        os.path.join(args.output_dir, "results_by_beta.csv"),
        summary_rows,
        summary_fields,
    )

    static_by_class = proto.per_class_f1(labels, static_preds, num_classes)
    best_by_class = proto.per_class_f1(labels, best["preds"], num_classes)
    per_rows = []
    for s, b in zip(static_by_class, best_by_class):
        c = s["class"]
        group = "bad" if c in bad_classes else "stable" if c in stable_classes else "other"
        per_rows.append({
            "class": c,
            "group": group,
            "reference_support": int(proto_support[c]),
            "target_support": s["support"],
            "static_f1": s["f1"],
            "best_pair_f1": b["f1"],
            "delta_f1": b["f1"] - s["f1"],
            "static_recall": s["recall"],
            "best_pair_recall": b["recall"],
            "delta_recall": b["recall"] - s["recall"],
        })
    proto.write_csv(
        os.path.join(args.output_dir, "per_class_metrics_m12.csv"),
        per_rows,
        [
            "class", "group", "reference_support", "target_support",
            "static_f1", "best_pair_f1", "delta_f1",
            "static_recall", "best_pair_recall", "delta_recall",
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
        row["beta"] = ""
        row["top_absorbers"] = ""
    for row in after_conf:
        row["method"] = "best_oracle_pair"
        row["beta"] = best["beta"]
        row["top_absorbers"] = best["top_absorbers"]
    proto.write_csv(
        os.path.join(args.output_dir, "bad_confusion_before_after_m12.csv"),
        before_conf + after_conf,
        [
            "method", "beta", "top_absorbers",
            "true_class", "pred_class", "rank_for_bad_class",
            "confusion_count", "confusion_rate", "support",
        ],
    )

    pair_rows = make_pair_summary(
        labels, static_preds, best["preds"],
        oracle_rows, best["beta"], best["top_absorbers"]
    )
    proto.write_csv(
        os.path.join(args.output_dir, "pair_summary.csv"),
        pair_rows,
        [
            "true_class", "pred_class", "oracle_rank",
            "oracle_confusion_count", "oracle_confusion_rate",
            "best_beta", "best_top_absorbers", "support",
            "static_count", "static_rate", "best_count", "best_rate",
            "delta_count", "delta_rate",
        ],
    )

    meta = {
        "diagnostic_type": "oracle_upper_bound",
        "leakage_warning": (
            "Uses target-label bad->absorber confusion pairs; not deployable."
        ),
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "pair_csv": args.pair_csv,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "betas": betas,
        "top_absorbers": top_absorber_values,
        "best_beta_by_bad_macro_f1": best["beta"],
        "best_top_absorbers_by_bad_macro_f1": best["top_absorbers"],
        "best_metrics": best["metrics"],
        "bad_classes": bad_classes,
        "stable_classes": stable_classes,
        "num_classes": num_classes,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Oracle pair recalibration summary ===")
    static_row = summary_rows[0]
    print(
        f"{'static':<18} beta={'':<5} top={'':<3} "
        f"macro_f1={static_row['overall_macro_f1']:.4f} "
        f"bad_f1={static_row['bad_macro_f1']:.4f} "
        f"stable_f1={static_row['stable_macro_f1']:.4f}"
    )
    print(
        f"{'best_oracle_pair':<18} beta={best['beta']:<5} "
        f"top={best['top_absorbers']:<3} "
        f"macro_f1={best['metrics']['overall_macro_f1']:.4f} "
        f"bad_f1={best['metrics']['bad_macro_f1']:.4f} "
        f"stable_f1={best['metrics']['stable_macro_f1']:.4f}"
    )
    print("Diagnostic only: oracle pairs use target labels.")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
