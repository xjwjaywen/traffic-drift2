"""
Reject-option ablation for TLS22 collapsed classes.

This is a diagnostic selective-classification experiment, not a new trained
model. It asks whether collapsed samples can be detected as unreliable
predictions by confidence, margin, source-prototype distance, absorber-risk,
or hybrid rules.

Usage from Experiment/core_code/:
    python scripts/reject_option_ablation_tls22.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --reference-period M-2022-4 \
      --target-period M-2022-12 \
      --output-dir outputs/reject_option_ablation_tls22_m12
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tta_tc_matplotlib_cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "tta_tc_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto
from tta_tc.utils.config import load_config


DEFAULT_COLLAPSE_CLASSES = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
DEFAULT_STABLE_CLASSES = [
    8, 15, 44, 57, 59, 62, 64, 76, 94, 98,
    99, 107, 113, 119, 128, 130, 131, 132, 144, 145,
]


def sanitize_period(period):
    match = re.match(r"([MW])-2022-(\d+)$", period)
    if match:
        return f"{match.group(1).lower()}{match.group(2)}"
    return re.sub(r"[^A-Za-z0-9]+", "_", period).strip("_").lower()


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=np.nan):
    if value in (None, ""):
        return default
    return float(value)


def as_int(value, default=0):
    if value in (None, ""):
        return default
    return int(float(value))


def parse_int_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [int(x) for x in str(value).replace(",", " ").split()]


def load_collapse_metadata(path, threshold):
    rows = read_csv(path)
    collapse_classes = []
    abrupt = []
    gradual = []
    absorber_by_class = {}
    for row in rows:
        first = row.get("first_collapse_period") or ""
        final_recall = as_float(row.get("final_recall"))
        if not first or not np.isfinite(final_recall) or final_recall >= threshold:
            continue
        class_id = as_int(row["class_id"])
        collapse_classes.append(class_id)
        pattern = row.get("collapse_pattern") or ""
        if pattern == "abrupt":
            abrupt.append(class_id)
        elif pattern == "gradual":
            gradual.append(class_id)
        absorber = row.get("final_top_confusion_target")
        if absorber not in (None, ""):
            absorber_by_class[class_id] = as_int(absorber)
    return collapse_classes, abrupt, gradual, absorber_by_class


def class_mask(labels, classes):
    return np.isin(labels, np.array(classes, dtype=np.int64))


def softmax_signals(logits):
    probs = F.softmax(logits, dim=1).numpy()
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    preds = probs.argmax(axis=1).astype(np.int64, copy=False)
    return probs, confidence, margin, preds


def prototype_distance_signals(features, prototypes, valid_mask):
    feat_n = F.normalize(features, dim=1)
    proto_n = F.normalize(prototypes, dim=1)
    sims = (feat_n @ proto_n.t()).numpy()
    if valid_mask is not None:
        invalid = ~valid_mask.numpy()
        sims[:, invalid] = -1e9
    nearest_proto = sims.argmax(axis=1).astype(np.int64, copy=False)
    nearest_sim = sims.max(axis=1)
    nearest_distance = 1.0 - nearest_sim
    pred_proto_distance = nearest_distance
    return nearest_distance.astype(np.float32), nearest_proto


def percentile_threshold(values, percentile):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, percentile))


def evaluate_reject_rule(
    rule_name,
    threshold_name,
    threshold_value,
    rejected,
    labels,
    preds,
    collapse_classes,
    abrupt_classes,
    gradual_classes,
    stable_classes,
    absorber_classes,
    absorber_by_class,
):
    rejected = np.asarray(rejected, dtype=bool)
    accepted = ~rejected
    n = labels.shape[0]
    collapse = class_mask(labels, collapse_classes)
    abrupt = class_mask(labels, abrupt_classes)
    gradual = class_mask(labels, gradual_classes)
    stable = class_mask(labels, stable_classes)
    absorber_pred = np.isin(preds, np.array(absorber_classes, dtype=np.int64))
    collapse_absorber_error = collapse & absorber_pred & (preds != labels)

    pair_absorber_error = np.zeros_like(collapse, dtype=bool)
    for true_class, absorber in absorber_by_class.items():
        pair_absorber_error |= (labels == true_class) & (preds == absorber)

    def safe_rate(mask):
        denom = int(mask.sum())
        if denom == 0:
            return ""
        return float(rejected[mask].mean())

    def kept_rate(mask):
        denom = int(mask.sum())
        if denom == 0:
            return ""
        return float(accepted[mask].mean())

    accepted_accuracy = ""
    accepted_macro_f1 = ""
    if int(accepted.sum()) > 0:
        accepted_accuracy = float((labels[accepted] == preds[accepted]).mean())
        accepted_macro_f1 = proto.compute_metrics(labels[accepted], preds[accepted])["macro_f1"]

    original_abs_errors = int(collapse_absorber_error.sum())
    kept_abs_errors = int((collapse_absorber_error & accepted).sum())
    original_pair_errors = int(pair_absorber_error.sum())
    kept_pair_errors = int((pair_absorber_error & accepted).sum())

    return {
        "rule": rule_name,
        "threshold_name": threshold_name,
        "threshold_value": threshold_value,
        "coverage": float(accepted.mean()),
        "reject_rate": float(rejected.mean()),
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
        "collapsed_reject_rate": safe_rate(collapse),
        "abrupt_reject_rate": safe_rate(abrupt),
        "gradual_reject_rate": safe_rate(gradual),
        "stable_false_reject_rate": safe_rate(stable),
        "stable_coverage": kept_rate(stable),
        "collapsed_coverage": kept_rate(collapse),
        "original_collapse_absorber_errors": original_abs_errors,
        "kept_collapse_absorber_errors": kept_abs_errors,
        "absorber_error_reduction": (
            float(1.0 - kept_abs_errors / original_abs_errors) if original_abs_errors else ""
        ),
        "original_pair_absorber_errors": original_pair_errors,
        "kept_pair_absorber_errors": kept_pair_errors,
        "pair_absorber_error_reduction": (
            float(1.0 - kept_pair_errors / original_pair_errors) if original_pair_errors else ""
        ),
        "num_rejected": int(rejected.sum()),
        "num_accepted": int(accepted.sum()),
        "num_collapsed_samples": int(collapse.sum()),
        "num_stable_samples": int(stable.sum()),
    }


def best_rows_by_rule(rows):
    best = {}
    for row in rows:
        rule = row["rule"]
        collapsed_reject = as_float(row["collapsed_reject_rate"], 0.0)
        stable_false = as_float(row["stable_false_reject_rate"], 1.0)
        absorber_reduction = as_float(row["absorber_error_reduction"], 0.0)
        coverage = as_float(row["coverage"], 0.0)
        score = collapsed_reject - stable_false + 0.25 * absorber_reduction + 0.05 * coverage
        if rule not in best or score > best[rule][0]:
            best[rule] = (score, row)
    return [value[1] for _, value in sorted(best.items())]


def plot_tradeoff(rows, output_dir):
    plt.figure(figsize=(8.2, 5.8))
    for rule in sorted({row["rule"] for row in rows}):
        subset = [row for row in rows if row["rule"] == rule]
        x = [as_float(row["stable_false_reject_rate"], np.nan) for row in subset]
        y = [as_float(row["collapsed_reject_rate"], np.nan) for row in subset]
        c = [as_float(row["coverage"], np.nan) for row in subset]
        plt.scatter(x, y, s=42, alpha=0.75, label=rule)
        for xi, yi, ci in zip(x, y, c):
            if np.isfinite(xi) and np.isfinite(yi) and ci >= 0.8:
                plt.text(xi, yi, f"{ci:.2f}", fontsize=7, alpha=0.75)
    plt.xlabel("Stable false reject rate")
    plt.ylabel("Collapsed reject rate")
    plt.title("Reject trade-off: collapsed detection vs stable damage")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    path = os.path.join(output_dir, "reject_tradeoff_collapsed_vs_stable.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_best_bar(best_rows, output_dir):
    labels = [row["rule"] for row in best_rows]
    collapsed = [as_float(row["collapsed_reject_rate"], 0.0) for row in best_rows]
    stable = [as_float(row["stable_false_reject_rate"], 0.0) for row in best_rows]
    absorber = [as_float(row["absorber_error_reduction"], 0.0) for row in best_rows]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(max(8.0, len(labels) * 1.2), 5.4))
    plt.bar(x - width, collapsed, width, label="collapsed reject")
    plt.bar(x, stable, width, label="stable false reject")
    plt.bar(x + width, absorber, width, label="absorber error reduction")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Best source-calibrated reject rules")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    path = os.path.join(output_dir, "reject_best_rules_summary.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def write_report(path, args, static_summary, best_rows, tradeoff_path, best_plot_path):
    lines = [
        "# Reject-Option Ablation",
        "",
        f"- Reference period: `{args.reference_period}`",
        f"- Target period: `{args.target_period}`",
        "",
        "## Static Baseline",
        "",
        f"- macro-F1: `{static_summary['overall_macro_f1']:.4f}`",
        f"- collapsed-class macro-F1: `{static_summary['bad_macro_f1']:.4f}`",
        f"- stable-class macro-F1: `{static_summary['stable_macro_f1']:.4f}`",
        "",
        "## Best Rules",
        "",
        "| rule | threshold | coverage | collapsed reject | stable false reject | absorber error reduction | accepted macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {rule} | {thr} | {cov:.3f} | {cr:.3f} | {sr:.3f} | {ar:.3f} | {am} |".format(
                rule=row["rule"],
                thr=row["threshold_name"],
                cov=as_float(row["coverage"], 0.0),
                cr=as_float(row["collapsed_reject_rate"], 0.0),
                sr=as_float(row["stable_false_reject_rate"], 0.0),
                ar=as_float(row["absorber_error_reduction"], 0.0),
                am=(
                    f"{as_float(row['accepted_macro_f1']):.4f}"
                    if row["accepted_macro_f1"] != ""
                    else ""
                ),
            )
        )
    lines.extend([
        "",
        "## Figures",
        "",
        f"- Trade-off: `{tradeoff_path}`",
        f"- Best-rule bar chart: `{best_plot_path}`",
        "",
        "## Interpretation Guide",
        "",
        "- A useful reject rule should reject many collapsed samples while keeping stable false rejections low.",
        "- If confidence or margin performs poorly but prototype/absorber-risk performs better, softmax confidence alone is insufficient.",
        "- If all rules either miss collapsed samples or reject stable samples heavily, post-hoc reject alone is not enough.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--collapse-report", default="outputs/per_class_collapse_tls22_monthly/collapse_classes.csv")
    parser.add_argument("--output-dir", default="outputs/reject_option_ablation_tls22_m12")
    parser.add_argument("--collapse-threshold", type=float, default=0.1)
    parser.add_argument("--collapse-classes", default="")
    parser.add_argument("--stable-classes", default="")
    parser.add_argument("--absorber-classes", default="")
    parser.add_argument("--quantiles", default="1,2,5,10,15,20,30")
    parser.add_argument("--distance-quantiles", default="70,80,85,90,95,97,99")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    period_slug = sanitize_period(args.target_period)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report_collapse, abrupt_classes, gradual_classes, absorber_by_class = load_collapse_metadata(
        args.collapse_report,
        args.collapse_threshold,
    )
    collapse_classes = parse_int_list(args.collapse_classes, report_collapse or DEFAULT_COLLAPSE_CLASSES)
    stable_classes = parse_int_list(args.stable_classes, DEFAULT_STABLE_CLASSES)
    if args.absorber_classes.strip():
        absorber_classes = parse_int_list(args.absorber_classes, [])
    else:
        absorber_classes = sorted(set(absorber_by_class.values()))
    if not absorber_classes:
        absorber_classes = [96, 46, 2, 14, 45, 105, 5, 71, 156, 13]
    if not abrupt_classes:
        abrupt_classes = collapse_classes
    if not gradual_classes:
        gradual_classes = []

    eval_cfg = load_config(args.config)
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes

    print(f"Using device: {device}")
    print(f"Reference period: {args.reference_period}")
    print(f"Target period: {args.target_period}")
    print(f"Collapse classes: {collapse_classes}")
    print(f"Absorber classes: {absorber_classes}")

    ref_loader, ref_classes = proto.make_test_loader(eval_cfg, args.reference_period)
    if ref_classes != num_classes:
        print(f"WARNING: reference loader classes={ref_classes}, model classes={num_classes}")
    ref = proto.collect_outputs(model, ref_loader, device, f"Reference {args.reference_period}")
    prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref["features"],
        ref["labels"],
        num_classes,
        min_support=1,
    )
    target_loader, target_classes = proto.make_test_loader(eval_cfg, args.target_period)
    if target_classes != num_classes:
        print(f"WARNING: target loader classes={target_classes}, model classes={num_classes}")
    target = proto.collect_outputs(model, target_loader, device, f"Target {args.target_period}")

    ref_probs, ref_conf, ref_margin, ref_preds = softmax_signals(ref["logits"])
    tgt_probs, tgt_conf, tgt_margin, tgt_preds = softmax_signals(target["logits"])
    ref_dist, ref_nearest_proto = prototype_distance_signals(ref["features"], prototypes, valid_mask)
    tgt_dist, tgt_nearest_proto = prototype_distance_signals(target["features"], prototypes, valid_mask)

    labels = target["labels"].astype(np.int64, copy=False)
    quantiles = [float(x) for x in args.quantiles.replace(",", " ").split()]
    dist_quantiles = [float(x) for x in args.distance_quantiles.replace(",", " ").split()]

    static_summary = proto.summarize_predictions(labels, tgt_preds, collapse_classes, stable_classes)
    rows = []

    absorber_pred = np.isin(tgt_preds, np.array(absorber_classes, dtype=np.int64))
    proto_disagree = tgt_nearest_proto != tgt_preds

    for q in quantiles:
        conf_thr = percentile_threshold(ref_conf, q)
        margin_thr = percentile_threshold(ref_margin, q)
        rows.append(evaluate_reject_rule(
            "confidence",
            f"ref_p{q:g}",
            conf_thr,
            tgt_conf < conf_thr,
            labels,
            tgt_preds,
            collapse_classes,
            abrupt_classes,
            gradual_classes,
            stable_classes,
            absorber_classes,
            absorber_by_class,
        ))
        rows.append(evaluate_reject_rule(
            "margin",
            f"ref_p{q:g}",
            margin_thr,
            tgt_margin < margin_thr,
            labels,
            tgt_preds,
            collapse_classes,
            abrupt_classes,
            gradual_classes,
            stable_classes,
            absorber_classes,
            absorber_by_class,
        ))

    for q in dist_quantiles:
        dist_thr = percentile_threshold(ref_dist, q)
        rows.append(evaluate_reject_rule(
            "prototype_distance",
            f"ref_p{q:g}",
            dist_thr,
            tgt_dist > dist_thr,
            labels,
            tgt_preds,
            collapse_classes,
            abrupt_classes,
            gradual_classes,
            stable_classes,
            absorber_classes,
            absorber_by_class,
        ))
        rows.append(evaluate_reject_rule(
            "absorber_distance",
            f"ref_p{q:g}",
            dist_thr,
            absorber_pred & (tgt_dist > dist_thr),
            labels,
            tgt_preds,
            collapse_classes,
            abrupt_classes,
            gradual_classes,
            stable_classes,
            absorber_classes,
            absorber_by_class,
        ))
        rows.append(evaluate_reject_rule(
            "absorber_proto_disagree",
            f"ref_p{q:g}",
            dist_thr,
            absorber_pred & proto_disagree & (tgt_dist > dist_thr),
            labels,
            tgt_preds,
            collapse_classes,
            abrupt_classes,
            gradual_classes,
            stable_classes,
            absorber_classes,
            absorber_by_class,
        ))

    for q in quantiles:
        conf_thr = percentile_threshold(ref_conf, q)
        margin_thr = percentile_threshold(ref_margin, q)
        for dq in dist_quantiles:
            dist_thr = percentile_threshold(ref_dist, dq)
            rows.append(evaluate_reject_rule(
                "hybrid",
                f"conf_p{q:g}_margin_p{q:g}_dist_p{dq:g}",
                dist_thr,
                (tgt_conf < conf_thr)
                | (tgt_margin < margin_thr)
                | (absorber_pred & (tgt_dist > dist_thr))
                | (absorber_pred & proto_disagree),
                labels,
                tgt_preds,
                collapse_classes,
                abrupt_classes,
                gradual_classes,
                stable_classes,
                absorber_classes,
                absorber_by_class,
            ))

    all_path = os.path.join(args.output_dir, f"reject_ablation_all_{period_slug}.csv")
    write_csv(all_path, rows)
    best_rows = best_rows_by_rule(rows)
    best_path = os.path.join(args.output_dir, f"reject_ablation_best_{period_slug}.csv")
    write_csv(best_path, best_rows)
    tradeoff_path = plot_tradeoff(rows, args.output_dir)
    best_plot_path = plot_best_bar(best_rows, args.output_dir)

    summary = {
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "collapse_classes": collapse_classes,
        "abrupt_classes": abrupt_classes,
        "gradual_classes": gradual_classes,
        "stable_classes": stable_classes,
        "absorber_classes": absorber_classes,
        "absorber_by_class": absorber_by_class,
        "static_summary": static_summary,
        "num_valid_prototypes": int(valid_mask.sum()),
        "prototype_support_min": int(proto_support[valid_mask.numpy()].min()) if int(valid_mask.sum()) else 0,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    report_path = os.path.join(args.output_dir, f"reject_option_ablation_{period_slug}.md")
    write_report(report_path, args, static_summary, best_rows, tradeoff_path, best_plot_path)

    print("\n=== Reject-Option Ablation Summary ===")
    print(
        f"static macro={static_summary['overall_macro_f1']:.4f} "
        f"collapsed_f1={static_summary['bad_macro_f1']:.4f} "
        f"stable_f1={static_summary['stable_macro_f1']:.4f}"
    )
    for row in best_rows:
        print(
            f"{row['rule']:<24} {row['threshold_name']:<24} "
            f"coverage={as_float(row['coverage'], 0.0):.3f} "
            f"collapsed_reject={as_float(row['collapsed_reject_rate'], 0.0):.3f} "
            f"stable_false={as_float(row['stable_false_reject_rate'], 0.0):.3f} "
            f"absorber_reduction={as_float(row['absorber_error_reduction'], 0.0):.3f}"
        )
    print(f"Saved all rows: {all_path}")
    print(f"Saved best rows: {best_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
