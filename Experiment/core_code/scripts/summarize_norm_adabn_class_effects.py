"""
Summarize per-class effects of IN/BN/LN/AdaBN on collapsed TLS22 classes.

This script complements the group-level normalization ablation by producing:

  - a per-class table for final collapsed classes in the final period;
  - a focused stable/abrupt/gradual group plot;
  - a class-by-method recall heatmap for collapsed classes.

Usage from Experiment/core_code/:
    python scripts/summarize_norm_adabn_class_effects.py \
      --output-dir outputs/teacher_result_visuals
"""
import argparse
import csv
import os
import re
import tempfile
from collections import OrderedDict

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tta_tc_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "gn": "GN",
    "in": "IN",
    "bn": "BN",
    "ln": "LN",
    "bn_static": "BN Static",
    "bn_adabn": "BN + AdaBN",
}


def read_csv(path):
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


def load_collapse_groups(path, final_period, threshold, min_support):
    rows = read_csv(path)
    final_collapsed = []
    abrupt = []
    gradual = []
    for row in rows:
        class_id = as_int(row["class_id"])
        support = as_int(row.get("final_support"))
        first = row.get("first_collapse_period", "")
        final_recall = as_float(row.get("final_recall"))
        if first and support >= min_support and final_recall < threshold:
            final_collapsed.append(class_id)
            if row.get("collapse_pattern") == "abrupt":
                abrupt.append(class_id)
            elif row.get("collapse_pattern") == "gradual":
                gradual.append(class_id)
    return {
        "final_collapsed": final_collapsed,
        "abrupt_collapsed": abrupt,
        "gradual_collapsed": gradual,
    }, {as_int(row["class_id"]): row for row in rows}


def load_per_class(norm_path, adabn_path, period):
    rows = []
    if os.path.exists(norm_path):
        for row in read_csv(norm_path):
            if row["period"] != period:
                continue
            item = dict(row)
            item["method"] = item.get("norm", "")
            rows.append(item)
    if os.path.exists(adabn_path):
        for row in read_csv(adabn_path):
            if row["period"] != period:
                continue
            rows.append(dict(row))
    return rows


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def write_markdown_table(path, rows):
    columns = [
        "class_id",
        "pattern",
        "absorber",
        "gn_recall",
        "in_recall",
        "bn_recall",
        "ln_recall",
        "adabn_recall",
        "delta_in_vs_gn",
        "delta_adabn_vs_bn",
    ]
    lines = [
        "# M12 Collapsed-Class Normalization / AdaBN Effects",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm-per-class-csv", default="outputs/norm_drift_type_ablation_tls22/norm_per_class_metrics.csv")
    parser.add_argument("--adabn-per-class-csv", default="outputs/adabn_drift_type_ablation_tls22/adabn_per_class_metrics.csv")
    parser.add_argument("--collapse-classes-csv", default="outputs/per_class_collapse_tls22_monthly/collapse_classes.csv")
    parser.add_argument("--period", default="M-2022-12")
    parser.add_argument("--output-dir", default="outputs/teacher_result_visuals")
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument(
        "--stable-classes",
        default="8,15,44,57,59,62,64,76,94,98,99,107,113,119,128,130,131,132,144,145",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.norm_per_class_csv):
        raise FileNotFoundError(args.norm_per_class_csv)
    if not os.path.exists(args.adabn_per_class_csv):
        raise FileNotFoundError(args.adabn_per_class_csv)
    if not os.path.exists(args.collapse_classes_csv):
        raise FileNotFoundError(args.collapse_classes_csv)

    groups, collapse_by_class = load_collapse_groups(
        args.collapse_classes_csv,
        args.period,
        args.collapse_recall_threshold,
        args.min_support,
    )
    stable = [as_int(x) for x in args.stable_classes.replace(",", " ").split()]
    groups["stable"] = stable

    per_class_rows = load_per_class(
        args.norm_per_class_csv,
        args.adabn_per_class_csv,
        args.period,
    )
    by_method_class = {
        (row["method"], as_int(row["class_id"])): row
        for row in per_class_rows
    }
    methods = [
        method for method in ["gn", "in", "bn", "ln", "bn_static", "bn_adabn"]
        if any(row["method"] == method for row in per_class_rows)
    ]

    collapsed_table = []
    for class_id in groups["final_collapsed"]:
        meta = collapse_by_class.get(class_id, {})
        row = {
            "class_id": class_id,
            "pattern": meta.get("collapse_pattern", ""),
            "absorber": meta.get("final_top_confusion_target", ""),
            "final_support": meta.get("final_support", ""),
        }
        for method in methods:
            item = by_method_class.get((method, class_id), {})
            prefix = "adabn" if method == "bn_adabn" else method
            row[f"{prefix}_recall"] = as_float(item.get("recall"))
            row[f"{prefix}_f1"] = as_float(item.get("f1"))
        row["delta_in_vs_gn"] = row.get("in_recall", np.nan) - row.get("gn_recall", np.nan)
        row["delta_adabn_vs_bn"] = row.get("adabn_recall", np.nan) - row.get("bn_recall", np.nan)
        collapsed_table.append(row)

    csv_path = os.path.join(args.output_dir, "m12_collapsed_norm_adabn_per_class.csv")
    md_path = os.path.join(args.output_dir, "m12_collapsed_norm_adabn_per_class.md")
    write_csv(csv_path, collapsed_table)
    write_markdown_table(md_path, collapsed_table)

    # Focused group plot: stable / abrupt / gradual.
    focused_groups = OrderedDict([
        ("stable", groups["stable"]),
        ("abrupt_collapsed", groups["abrupt_collapsed"]),
        ("gradual_collapsed", groups["gradual_collapsed"]),
    ])
    x = np.arange(len(focused_groups))
    width = 0.8 / max(len(methods), 1)
    plt.figure(figsize=(9.2, 4.8))
    for idx, method in enumerate(methods):
        vals = []
        for class_ids in focused_groups.values():
            f1s = [
                as_float(by_method_class.get((method, c), {}).get("f1"))
                for c in class_ids
            ]
            vals.append(float(np.nanmean(f1s)) if f1s else np.nan)
        plt.bar(x + idx * width, vals, width=width, label=METHOD_LABELS.get(method, method))
    plt.xticks(
        x + width * (len(methods) - 1) / 2,
        ["stable", "abrupt", "gradual"],
    )
    plt.title(f"TLS22 {args.period}: normalization effects by drift type")
    plt.ylabel("Group macro-F1")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    focused_plot = os.path.join(args.output_dir, "tls22_m12_stable_abrupt_gradual_norm_adabn.png")
    savefig(focused_plot)

    # Recall heatmap for final-collapsed classes.
    heat_methods = [m for m in ["gn", "in", "bn", "ln", "bn_adabn"] if m in methods]
    matrix = np.array([
        [
            as_float(by_method_class.get((method, class_id), {}).get("recall"))
            for method in heat_methods
        ]
        for class_id in groups["final_collapsed"]
    ])
    plt.figure(figsize=(max(6, len(heat_methods) * 1.2), max(4, len(groups["final_collapsed"]) * 0.38)))
    im = plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.2, np.nanmax(matrix)))
    plt.colorbar(im, label="Recall")
    plt.xticks(np.arange(len(heat_methods)), [METHOD_LABELS.get(m, m) for m in heat_methods], rotation=25, ha="right")
    plt.yticks(np.arange(len(groups["final_collapsed"])), [str(c) for c in groups["final_collapsed"]])
    plt.title(f"TLS22 {args.period}: collapsed-class recall by normalization")
    plt.xlabel("Method")
    plt.ylabel("Class")
    heatmap = os.path.join(args.output_dir, "tls22_m12_collapsed_class_norm_adabn_recall_heatmap.png")
    savefig(heatmap)

    print(f"Saved per-class table: {csv_path}")
    print(f"Saved markdown table: {md_path}")
    print(f"Saved focused group plot: {focused_plot}")
    print(f"Saved collapsed-class heatmap: {heatmap}")


if __name__ == "__main__":
    main()

