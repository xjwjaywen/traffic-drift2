"""
Evaluate TTA methods by TLS22 drift-type groups.

Existing evaluate_tta.py stores only overall sequential metrics. This script
reruns selected TTA methods and computes group-level metrics for stable,
final-collapsed, abrupt-collapsed, gradual-collapsed, absorber, and degraded
non-collapsed classes.

Usage from Experiment/core_code/:
    python scripts/tta_drift_type_ablation_tls22.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --methods static,eata,cotta,sar,tta_tc \
      --output-dir outputs/tta_drift_type_ablation_tls22
"""
import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import norm_drift_type_ablation_tls22 as norm_eval
import prototype_recalibration_tls22 as proto
from evaluate_tta import evaluate_static, evaluate_tta_method, load_source_model
from tta_tc.baselines import BNAdapt, CoTTA, EATA, NOTE, SAR, Tent
from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.tta import TTAEngine


METHOD_CLASSES = {
    "bn_adapt": BNAdapt,
    "tent": Tent,
    "eata": EATA,
    "cotta": CoTTA,
    "sar": SAR,
    "note": NOTE,
}


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


def load_train_source_accuracy(checkpoint_path):
    path = os.path.join(os.path.dirname(checkpoint_path), "train_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("test_accuracy")


def summarize_predictions(labels, preds, num_classes, groups, args):
    result = norm_eval.compute_report(labels, preds, num_classes)
    report = result["report"]
    collapsed = norm_eval.summarize_group(
        report,
        groups["final_collapsed"],
        args.collapse_recall_threshold,
        args.severe_recall_threshold,
    )
    overall = {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "final_collapsed_macro_f1": collapsed["macro_f1"],
        "final_collapsed_macro_recall": collapsed["macro_recall"],
        "final_collapsed_count": collapsed["collapsed_count"],
        "final_severe_count": collapsed["severe_count"],
    }
    group_rows = []
    for group_name, classes in groups.items():
        summary = norm_eval.summarize_group(
            report,
            classes,
            args.collapse_recall_threshold,
            args.severe_recall_threshold,
        )
        overall[f"{group_name}_macro_f1"] = summary["macro_f1"]
        overall[f"{group_name}_macro_recall"] = summary["macro_recall"]
        group_rows.append({"group": group_name, **summary})

    per_class_rows = []
    for class_id in range(num_classes):
        item = report.get(str(class_id), {})
        per_class_rows.append({
            "class_id": class_id,
            "groups": norm_eval.class_memberships(groups, class_id),
            "support": int(item.get("support", 0)),
            "precision": float(item.get("precision", 0.0)),
            "recall": float(item.get("recall", 0.0)),
            "f1": float(item.get("f1-score", 0.0)),
        })
    return overall, group_rows, per_class_rows


def load_optional_tensor(path, device):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=True)


def make_method(method_name, base_model, checkpoint_dir, num_classes, eval_cfg, device):
    model = copy.deepcopy(base_model).to(device)
    adapt_cfg = {"num_classes": num_classes, **eval_cfg.get("tta", {})}
    if method_name == "tta_tc":
        prototypes = load_optional_tensor(
            os.path.join(checkpoint_dir, "class_prototypes.pt"),
            device,
        )
        if prototypes is None:
            raise RuntimeError("TTA-TC requires class_prototypes.pt next to checkpoint.")
        position_stats = load_optional_tensor(
            os.path.join(checkpoint_dir, "position_stats.pt"),
            device,
        )
        return model, TTAEngine(
            model,
            adapt_cfg,
            prototypes=prototypes,
            position_stats=position_stats,
        )
    cls = METHOD_CLASSES[method_name]
    return model, cls(model, adapt_cfg)


def write_report(path, overall_rows, delta_rows, methods):
    lines = [
        "# TLS22 TTA Drift-Type Ablation",
        "",
        f"- Methods: `{', '.join(methods)}`",
        "- Metrics are computed on the same drift groups used in the normalization ablation.",
        "",
        "## Overall Metrics",
        "",
        "| method | period | macro-F1 | final-collapsed F1 | stable F1 | collapsed count | severe count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['method']} | {row['period']} | {norm_eval.fmt(row['macro_f1'])} | "
            f"{norm_eval.fmt(row['final_collapsed_macro_f1'])} | "
            f"{norm_eval.fmt(row.get('stable_macro_f1'))} | "
            f"{row['final_collapsed_count']} | {row['final_severe_count']} |"
        )
    lines.extend([
        "",
        "## Group Deltas vs Static",
        "",
        "| method | period | group | Delta F1 | Delta Recall | Delta collapsed | Delta severe |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    important = {
        "stable",
        "final_collapsed",
        "abrupt_collapsed",
        "gradual_collapsed",
        "absorber",
        "degraded_noncollapsed",
    }
    for row in delta_rows:
        if row["group"] not in important:
            continue
        lines.append(
            f"| {row['method']} | {row['period']} | {row['group']} | "
            f"{norm_eval.fmt(row['delta_macro_f1'])} | "
            f"{norm_eval.fmt(row['delta_macro_recall'])} | "
            f"{row['delta_collapsed_count']} | {row['delta_severe_count']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--methods",
        default="static,eata,cotta,sar,tta_tc",
        help="Comma-separated methods. Supported: static,bn_adapt,tent,eata,cotta,sar,note,tta_tc",
    )
    parser.add_argument("--output-dir", required=True)
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
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in {"static", "tta_tc", *METHOD_CLASSES}]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = proto.load_config(args.config)
    base_model, train_cfg, num_classes = load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes
    checkpoint_dir = os.path.dirname(args.checkpoint)
    source_acc = load_train_source_accuracy(args.checkpoint)

    groups, loaded_report = norm_eval.load_drift_groups(
        args.collapse_report,
        args.collapse_recall_threshold,
        args.degraded_drop_threshold,
        args.degraded_min_support,
    )
    groups["stable"] = norm_eval.parse_class_list(args.stable_classes, groups["stable"])
    groups["absorber"] = norm_eval.parse_class_list(args.absorber_classes, groups["absorber"])

    loaders, loader_classes = build_sequential_test_loaders(eval_cfg["data"])
    if loader_classes != num_classes:
        print(f"WARNING: loader classes={loader_classes}, model classes={num_classes}")

    print(f"Using device: {device}")
    print(f"Loaded collapse report: {loaded_report} ({args.collapse_report})")
    print(f"Methods: {methods}")
    print("Groups:")
    for name, classes in groups.items():
        print(f"  {name}: n={len(classes)} classes={classes[:20]}")

    overall_rows = []
    group_rows = []
    per_class_rows = []

    for method in methods:
        print(f"\n=== TTA method: {method} ===")
        if method == "static":
            engine = None
            method_model = base_model
        else:
            method_model, engine = make_method(
                method,
                base_model,
                checkpoint_dir,
                num_classes,
                eval_cfg,
                device,
            )

        for period, loader in loaders:
            t0 = time.time()
            if method == "static":
                labels, preds = evaluate_static(method_model, loader, device)
            elif method == "tta_tc":
                engine.reset_period()
                labels, preds = engine.adapt_period(loader, period)
            else:
                labels, preds, _ = evaluate_tta_method(
                    engine,
                    loader,
                    device,
                    method_name=f"{method}@{period}",
                )
            elapsed = time.time() - t0
            overall, groups_out, per_class = summarize_predictions(
                labels,
                preds,
                num_classes,
                groups,
                args,
            )
            if source_acc:
                overall["arr"] = overall["accuracy"] / source_acc
            overall.update({"method": method, "period": period, "time_s": elapsed})
            overall_rows.append(overall)
            for row in groups_out:
                row.update({"method": method, "period": period})
                group_rows.append(row)
            for row in per_class:
                row.update({"method": method, "period": period})
                per_class_rows.append(row)
            print(
                f"{method}@{period}: macro_f1={overall['macro_f1']:.4f} "
                f"collapsed_f1={overall['final_collapsed_macro_f1']:.4f} "
                f"stable_f1={overall.get('stable_macro_f1', float('nan')):.4f} "
                f"time={elapsed:.1f}s"
            )
        if method != "static":
            del method_model

    static_by_group = {
        (row["period"], row["group"]): row
        for row in group_rows
        if row["method"] == "static"
    }
    delta_rows = []
    for row in group_rows:
        if row["method"] == "static":
            continue
        base = static_by_group.get((row["period"], row["group"]))
        if base is None:
            continue
        delta_rows.append({
            "baseline_method": "static",
            "method": row["method"],
            "period": row["period"],
            "group": row["group"],
            "delta_macro_f1": (
                None if row["macro_f1"] is None or base["macro_f1"] is None
                else row["macro_f1"] - base["macro_f1"]
            ),
            "delta_macro_recall": (
                None if row["macro_recall"] is None or base["macro_recall"] is None
                else row["macro_recall"] - base["macro_recall"]
            ),
            "delta_collapsed_count": row["collapsed_count"] - base["collapsed_count"],
            "delta_severe_count": row["severe_count"] - base["severe_count"],
        })

    write_csv(os.path.join(args.output_dir, "tta_drift_period_metrics.csv"), overall_rows)
    write_csv(os.path.join(args.output_dir, "tta_drift_group_metrics.csv"), group_rows)
    write_csv(os.path.join(args.output_dir, "tta_drift_delta_vs_static.csv"), delta_rows)
    write_csv(os.path.join(args.output_dir, "tta_drift_per_class_metrics.csv"), per_class_rows)
    write_report(
        os.path.join(args.output_dir, "tta_drift_type_report.md"),
        overall_rows,
        delta_rows,
        methods,
    )
    summary = {
        "checkpoint": args.checkpoint,
        "methods": methods,
        "source_accuracy": source_acc,
        "collapse_report": args.collapse_report,
        "collapse_report_loaded": loaded_report,
        "groups": {name: classes for name, classes in groups.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

