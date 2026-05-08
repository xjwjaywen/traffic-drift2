"""
Summarize TLS22 normalization ablations by drift type.

This evaluates static checkpoints trained with different normalization layers
(GN/IN/BN/LN) and reports overall metrics plus group metrics for stable,
collapsed, abrupt-collapse, gradual-collapse, degraded, and absorber classes.

Usage from Experiment/core_code/:
    python scripts/norm_drift_type_ablation_tls22.py \
      --config configs/eval_tls22.yaml \
      --checkpoints gn=outputs/tls22_cnn/best_model.pt \
                    in=outputs/tls22_cnn_in/best_model.pt \
                    bn=outputs/tls22_cnn_bn/best_model.pt \
                    ln=outputs/tls22_cnn_ln/best_model.pt \
      --periods M-2022-7 M-2022-10 M-2022-12 \
      --output-dir outputs/norm_drift_type_ablation_tls22
"""
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


DEFAULT_STABLE_CLASSES = [
    8, 15, 44, 57, 59, 62, 64, 76, 94, 98,
    99, 107, 113, 119, 128, 130, 131, 132, 144, 145,
]
DEFAULT_FINAL_COLLAPSED = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
DEFAULT_ABRUPT_COLLAPSED = [56, 174, 48, 38, 69, 66, 109, 26]
DEFAULT_GRADUAL_COLLAPSED = [163, 104, 47, 10]
DEFAULT_ABSORBERS = [96, 46, 2, 14, 45, 105, 5, 71, 156, 13]


def parse_checkpoint_specs(values):
    result = OrderedDict()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint spec must be name=path, got: {value}")
        name, path = value.split("=", 1)
        result[name.strip()] = path.strip()
    return result


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


def parse_class_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [int(x) for x in str(value).replace(",", " ").split()]


def load_drift_groups(path, recall_threshold, degraded_drop_threshold, min_support):
    groups = OrderedDict()
    groups["stable"] = list(DEFAULT_STABLE_CLASSES)
    groups["final_collapsed"] = list(DEFAULT_FINAL_COLLAPSED)
    groups["abrupt_collapsed"] = list(DEFAULT_ABRUPT_COLLAPSED)
    groups["gradual_collapsed"] = list(DEFAULT_GRADUAL_COLLAPSED)
    groups["absorber"] = list(DEFAULT_ABSORBERS)
    groups["degraded_noncollapsed"] = []

    if not path or not os.path.exists(path):
        return groups, False

    final_collapsed = []
    abrupt = []
    gradual = []
    degraded = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row["class_id"])
            support = int(float(row.get("final_support") or 0))
            final_recall = float(row.get("final_recall") or 0.0)
            first_collapse = row.get("first_collapse_period") or ""
            pattern = row.get("collapse_pattern") or ""
            delta_recall = float(row.get("delta_final_recall_from_ref") or 0.0)
            is_final_collapsed = first_collapse != "" and final_recall < recall_threshold
            if is_final_collapsed:
                final_collapsed.append(class_id)
                if pattern == "abrupt":
                    abrupt.append(class_id)
                elif pattern == "gradual":
                    gradual.append(class_id)
            elif support >= min_support and delta_recall <= -degraded_drop_threshold:
                degraded.append(class_id)

    if final_collapsed:
        groups["final_collapsed"] = final_collapsed
    if abrupt:
        groups["abrupt_collapsed"] = abrupt
    if gradual:
        groups["gradual_collapsed"] = gradual
    groups["degraded_noncollapsed"] = degraded
    return groups, True


def compute_report(labels, preds, num_classes):
    all_labels = list(range(num_classes))
    report = classification_report(
        labels,
        preds,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=all_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=all_labels, average="weighted", zero_division=0)),
        "report": report,
    }


def summarize_group(report, classes, recall_threshold, severe_threshold):
    values_f1 = []
    values_recall = []
    support = 0
    collapsed = 0
    severe = 0
    for class_id in classes:
        item = report.get(str(class_id), {})
        f1 = float(item.get("f1-score", 0.0))
        recall = float(item.get("recall", 0.0))
        cls_support = int(item.get("support", 0))
        values_f1.append(f1)
        values_recall.append(recall)
        support += cls_support
        if recall < recall_threshold:
            collapsed += 1
        if recall < severe_threshold:
            severe += 1
    return {
        "n_classes": len(classes),
        "support": int(support),
        "macro_f1": float(np.mean(values_f1)) if values_f1 else None,
        "macro_recall": float(np.mean(values_recall)) if values_recall else None,
        "collapsed_count": int(collapsed),
        "severe_count": int(severe),
    }


def class_memberships(groups, class_id):
    names = []
    for name, classes in groups.items():
        if class_id in set(classes):
            names.append(name)
    return ";".join(names)


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_report(path, overall_rows, delta_rows, baseline_norm):
    lines = [
        "# TLS22 Normalization Drift-Type Ablation",
        "",
        f"- Baseline norm: `{baseline_norm}`",
        "- Metrics are static predictions from each trained checkpoint.",
        "",
        "## Overall Metrics",
        "",
        "| norm | period | macro-F1 | final-collapsed F1 | stable F1 | collapsed count | severe count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['norm']} | {row['period']} | {fmt(row['macro_f1'])} | "
            f"{fmt(row['final_collapsed_macro_f1'])} | {fmt(row.get('stable_macro_f1'))} | "
            f"{row['final_collapsed_count']} | {row['final_severe_count']} |"
        )
    lines.extend([
        "",
        "## Group Deltas vs Baseline",
        "",
        "| norm | period | group | ΔF1 | ΔRecall | Δcollapsed | Δsevere |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    important = {
        "stable",
        "final_collapsed",
        "abrupt_collapsed",
        "gradual_collapsed",
        "degraded_noncollapsed",
        "absorber",
    }
    for row in delta_rows:
        if row["group"] not in important:
            continue
        lines.append(
            f"| {row['norm']} | {row['period']} | {row['group']} | "
            f"{fmt(row['delta_macro_f1'])} | {fmt(row['delta_macro_recall'])} | "
            f"{row['delta_collapsed_count']} | {row['delta_severe_count']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--periods",
        nargs="+",
        default=["M-2022-7", "M-2022-10", "M-2022-12"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-norm", default="gn")
    parser.add_argument(
        "--collapse-report",
        default="outputs/per_class_collapse_tls22_monthly/collapse_classes.csv",
    )
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    parser.add_argument("--degraded-drop-threshold", type=float, default=0.3)
    parser.add_argument("--degraded-min-support", type=int, default=50)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--absorber-classes", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints = parse_checkpoint_specs(args.checkpoints)
    groups, loaded_report = load_drift_groups(
        args.collapse_report,
        args.collapse_recall_threshold,
        args.degraded_drop_threshold,
        args.degraded_min_support,
    )
    groups["stable"] = parse_class_list(args.stable_classes, groups["stable"])
    groups["absorber"] = parse_class_list(args.absorber_classes, groups["absorber"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = proto.load_config(args.config)
    print(f"Using device: {device}")
    print(f"Loaded collapse report: {loaded_report} ({args.collapse_report})")
    print("Groups:")
    for name, classes in groups.items():
        print(f"  {name}: n={len(classes)} classes={classes[:20]}")

    overall_rows = []
    group_rows = []
    per_class_rows = []

    for norm_name, checkpoint_path in checkpoints.items():
        print(f"\n=== Norm: {norm_name} ===")
        model, _, num_classes = proto.load_source_model(checkpoint_path, device)
        eval_cfg["data"]["num_classes"] = num_classes
        for period in args.periods:
            loader, loader_classes = proto.make_test_loader(eval_cfg, period)
            if loader_classes != num_classes:
                print(f"WARNING {period}: loader classes={loader_classes}, model classes={num_classes}")
            outputs = proto.collect_outputs(model, loader, device, desc=f"{norm_name}@{period}")
            labels = outputs["labels"]
            preds = outputs["logits"].argmax(dim=1).numpy()
            result = compute_report(labels, preds, num_classes)
            report = result["report"]
            collapsed_summary = summarize_group(
                report,
                groups["final_collapsed"],
                args.collapse_recall_threshold,
                args.severe_recall_threshold,
            )
            row = {
                "norm": norm_name,
                "period": period,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "final_collapsed_macro_f1": collapsed_summary["macro_f1"],
                "final_collapsed_macro_recall": collapsed_summary["macro_recall"],
                "final_collapsed_count": collapsed_summary["collapsed_count"],
                "final_severe_count": collapsed_summary["severe_count"],
            }
            for group_name, classes in groups.items():
                summary = summarize_group(
                    report,
                    classes,
                    args.collapse_recall_threshold,
                    args.severe_recall_threshold,
                )
                group_rows.append({
                    "norm": norm_name,
                    "period": period,
                    "group": group_name,
                    **summary,
                })
                row[f"{group_name}_macro_f1"] = summary["macro_f1"]
                row[f"{group_name}_macro_recall"] = summary["macro_recall"]
            overall_rows.append(row)

            for class_id in range(num_classes):
                item = report.get(str(class_id), {})
                per_class_rows.append({
                    "norm": norm_name,
                    "period": period,
                    "class_id": class_id,
                    "groups": class_memberships(groups, class_id),
                    "support": int(item.get("support", 0)),
                    "precision": float(item.get("precision", 0.0)),
                    "recall": float(item.get("recall", 0.0)),
                    "f1": float(item.get("f1-score", 0.0)),
                })
            print(
                f"{period}: macro_f1={result['macro_f1']:.4f} "
                f"collapsed_f1={collapsed_summary['macro_f1']:.4f} "
                f"collapsed={collapsed_summary['collapsed_count']}"
            )

    delta_rows = []
    baseline = {
        (row["period"], row["group"]): row
        for row in group_rows
        if row["norm"] == args.baseline_norm
    }
    for row in group_rows:
        base = baseline.get((row["period"], row["group"]))
        if base is None or row["norm"] == args.baseline_norm:
            continue
        delta_rows.append({
            "baseline_norm": args.baseline_norm,
            "norm": row["norm"],
            "period": row["period"],
            "group": row["group"],
            "delta_macro_f1": None if row["macro_f1"] is None or base["macro_f1"] is None else row["macro_f1"] - base["macro_f1"],
            "delta_macro_recall": None if row["macro_recall"] is None or base["macro_recall"] is None else row["macro_recall"] - base["macro_recall"],
            "delta_collapsed_count": row["collapsed_count"] - base["collapsed_count"],
            "delta_severe_count": row["severe_count"] - base["severe_count"],
        })

    write_csv(os.path.join(args.output_dir, "norm_period_metrics.csv"), overall_rows)
    write_csv(os.path.join(args.output_dir, "norm_group_metrics.csv"), group_rows)
    write_csv(os.path.join(args.output_dir, "norm_delta_vs_baseline.csv"), delta_rows)
    write_csv(os.path.join(args.output_dir, "norm_per_class_metrics.csv"), per_class_rows)
    write_report(
        os.path.join(args.output_dir, "norm_drift_type_report.md"),
        overall_rows,
        delta_rows,
        args.baseline_norm,
    )
    summary = {
        "checkpoints": checkpoints,
        "periods": args.periods,
        "baseline_norm": args.baseline_norm,
        "collapse_report": args.collapse_report,
        "collapse_report_loaded": loaded_report,
        "groups": {name: classes for name, classes in groups.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
