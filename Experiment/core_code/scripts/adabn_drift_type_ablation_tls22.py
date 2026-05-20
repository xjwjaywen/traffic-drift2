"""
Evaluate Adaptive BatchNorm (AdaBN) on TLS22 drift-type groups.

This script is intentionally separate from evaluate_tta.py because AdaBN should
be evaluated on a BatchNorm checkpoint, not on the default GroupNorm model.
For each target period, it:

  1. evaluates the BN checkpoint without adaptation;
  2. reloads the same checkpoint;
  3. resets BN running statistics and recomputes them on unlabeled target data;
  4. evaluates the adapted model on the same target period;
  5. reports overall and drift-type group metrics.

Usage from Experiment/core_code/:
    python scripts/adabn_drift_type_ablation_tls22.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn_bn/best_model.pt \
      --periods M-2022-7 M-2022-10 M-2022-12 \
      --output-dir outputs/adabn_drift_type_ablation_tls22
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import norm_drift_type_ablation_tls22 as norm_eval
import prototype_recalibration_tls22 as proto


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


def count_bn_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))


def configure_bn_for_adabn(model):
    """Reset BN stats and make only BN layers update running statistics."""
    model.eval()
    bn_layers = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average over batches
            module.train()
            module.requires_grad_(False)
            bn_layers.append(module)
    return bn_layers


@torch.no_grad()
def adapt_bn_stats(model, loader, device, desc):
    """Run an unlabeled target pass to update BN running mean/variance."""
    bn_layers = configure_bn_for_adabn(model)
    if not bn_layers:
        raise RuntimeError(
            "No BatchNorm1d layers found. AdaBN requires a BN checkpoint "
            "(for example outputs/tls22_cnn_bn/best_model.pt)."
        )

    for batch in tqdm(loader, desc=desc):
        ppi = batch["ppi"].to(device)
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)
        _ = model(ppi, flow_stats)

    model.eval()
    return len(bn_layers)


def evaluate_method(model, eval_cfg, period, device, method_name, groups, args):
    loader, num_classes = proto.make_test_loader(eval_cfg, period)
    outputs = proto.collect_outputs(
        model, loader, device, desc=f"{method_name}@{period}"
    )
    labels = outputs["labels"]
    preds = outputs["logits"].argmax(dim=1).numpy()
    result = norm_eval.compute_report(labels, preds, num_classes)
    report = result["report"]

    collapsed = norm_eval.summarize_group(
        report,
        groups["final_collapsed"],
        args.collapse_recall_threshold,
        args.severe_recall_threshold,
    )
    overall = {
        "method": method_name,
        "period": period,
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
        group_rows.append({
            "method": method_name,
            "period": period,
            "group": group_name,
            **summary,
        })

    per_class_rows = []
    for class_id in range(num_classes):
        item = report.get(str(class_id), {})
        per_class_rows.append({
            "method": method_name,
            "period": period,
            "class_id": class_id,
            "groups": norm_eval.class_memberships(groups, class_id),
            "support": int(item.get("support", 0)),
            "precision": float(item.get("precision", 0.0)),
            "recall": float(item.get("recall", 0.0)),
            "f1": float(item.get("f1-score", 0.0)),
        })

    return overall, group_rows, per_class_rows


def write_report(path, overall_rows, delta_rows, bn_layers):
    lines = [
        "# TLS22 AdaBN Drift-Type Ablation",
        "",
        f"- BatchNorm layers adapted: `{bn_layers}`",
        "- AdaBN recomputes BN running statistics on unlabeled target-period data.",
        "- Learned model weights are not updated.",
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
        "## AdaBN Deltas vs BN Static",
        "",
        "| period | group | Delta F1 | Delta Recall | Delta collapsed | Delta severe |",
        "|---|---|---:|---:|---:|---:|",
    ])
    important_groups = {
        "stable",
        "final_collapsed",
        "abrupt_collapsed",
        "gradual_collapsed",
        "absorber",
        "degraded_noncollapsed",
    }
    for row in delta_rows:
        if row["group"] not in important_groups:
            continue
        lines.append(
            f"| {row['period']} | {row['group']} | "
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
        "--periods",
        nargs="+",
        default=["M-2022-7", "M-2022-10", "M-2022-12"],
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = proto.load_config(args.config)

    groups, loaded_report = norm_eval.load_drift_groups(
        args.collapse_report,
        args.collapse_recall_threshold,
        args.degraded_drop_threshold,
        args.degraded_min_support,
    )
    groups["stable"] = norm_eval.parse_class_list(args.stable_classes, groups["stable"])
    groups["absorber"] = norm_eval.parse_class_list(args.absorber_classes, groups["absorber"])

    print(f"Using device: {device}")
    print(f"Loaded collapse report: {loaded_report} ({args.collapse_report})")
    print("Groups:")
    for name, classes in groups.items():
        print(f"  {name}: n={len(classes)} classes={classes[:20]}")

    probe_model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    bn_layers = count_bn_layers(probe_model)
    if bn_layers == 0:
        raise RuntimeError(
            "The checkpoint has no BatchNorm1d layers. Use the BN model checkpoint."
        )
    del probe_model
    eval_cfg["data"]["num_classes"] = num_classes

    overall_rows = []
    group_rows = []
    per_class_rows = []

    for period in args.periods:
        print(f"\n=== Period: {period} ===")

        static_model, _, _ = proto.load_source_model(args.checkpoint, device)
        overall, groups_out, per_class = evaluate_method(
            static_model,
            eval_cfg,
            period,
            device,
            "bn_static",
            groups,
            args,
        )
        overall_rows.append(overall)
        group_rows.extend(groups_out)
        per_class_rows.extend(per_class)
        print(
            f"bn_static@{period}: macro_f1={overall['macro_f1']:.4f} "
            f"collapsed_f1={overall['final_collapsed_macro_f1']:.4f}"
        )
        del static_model

        adabn_model, _, _ = proto.load_source_model(args.checkpoint, device)
        adapt_loader, _ = proto.make_test_loader(eval_cfg, period)
        adapted_layers = adapt_bn_stats(
            adabn_model,
            adapt_loader,
            device,
            desc=f"adabn_stats@{period}",
        )
        overall, groups_out, per_class = evaluate_method(
            adabn_model,
            eval_cfg,
            period,
            device,
            "bn_adabn",
            groups,
            args,
        )
        overall_rows.append(overall)
        group_rows.extend(groups_out)
        per_class_rows.extend(per_class)
        print(
            f"bn_adabn@{period}: macro_f1={overall['macro_f1']:.4f} "
            f"collapsed_f1={overall['final_collapsed_macro_f1']:.4f} "
            f"adapted_bn_layers={adapted_layers}"
        )
        del adabn_model

    static_by_group = {
        (row["period"], row["group"]): row
        for row in group_rows
        if row["method"] == "bn_static"
    }
    delta_rows = []
    for row in group_rows:
        if row["method"] != "bn_adabn":
            continue
        base = static_by_group.get((row["period"], row["group"]))
        if base is None:
            continue
        delta_rows.append({
            "baseline_method": "bn_static",
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

    write_csv(os.path.join(args.output_dir, "adabn_period_metrics.csv"), overall_rows)
    write_csv(os.path.join(args.output_dir, "adabn_group_metrics.csv"), group_rows)
    write_csv(os.path.join(args.output_dir, "adabn_delta_vs_bn_static.csv"), delta_rows)
    write_csv(os.path.join(args.output_dir, "adabn_per_class_metrics.csv"), per_class_rows)
    write_report(
        os.path.join(args.output_dir, "adabn_drift_type_report.md"),
        overall_rows,
        delta_rows,
        bn_layers,
    )
    summary = {
        "checkpoint": args.checkpoint,
        "periods": args.periods,
        "bn_layers": bn_layers,
        "collapse_report": args.collapse_report,
        "collapse_report_loaded": loaded_report,
        "groups": {name: classes for name, classes in groups.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

