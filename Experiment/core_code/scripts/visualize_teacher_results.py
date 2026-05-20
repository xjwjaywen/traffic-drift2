"""
Create advisor-facing visualizations for TTA, normalization, and collapse results.

The script is deliberately tolerant of missing files: it generates every figure
that can be produced from the available outputs and records missing inputs in a
Markdown summary. This makes it usable both locally and on the GPU server.

Usage from Experiment/core_code/:
    python scripts/visualize_teacher_results.py \
      --output-dir outputs/teacher_result_visuals
"""
import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tta_tc_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "static": "Static",
    "bn_adapt": "BN-Adapt",
    "tent": "Tent",
    "eata": "EATA",
    "cotta": "CoTTA",
    "sar": "SAR",
    "note": "NOTE",
    "tta_tc": "TTA-TC",
    "gn": "GN",
    "in": "IN",
    "bn": "BN",
    "ln": "LN",
    "bn_static": "BN Static",
    "bn_adabn": "BN + AdaBN",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def exists(path):
    return path and os.path.exists(path)


def period_sort_key(period):
    match = re.match(r"([MW])-2022-(\d+)$", str(period))
    if not match:
        return (str(period), 0)
    return (match.group(1), int(match.group(2)))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tta_results(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data)


def method_label(method):
    return METHOD_LABELS.get(method, method)


def as_float(value, default=np.nan):
    if value in (None, ""):
        return default
    return float(value)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_tta_curves(path, out_dir, dataset_name):
    if not exists(path):
        return None, f"Missing TTA result: {path}"
    data = load_tta_results(path)
    methods = [
        m for m in ["static", "eata", "cotta", "sar", "tta_tc", "tent", "note", "bn_adapt"]
        if m in data
    ]
    if not methods:
        return None, f"No known TTA methods in: {path}"

    periods = sorted(
        {p for m in methods for p in data[m].get("periods", {})},
        key=period_sort_key,
    )
    plt.figure(figsize=(9, 5))
    for method in methods:
        values = [
            data[method]["periods"].get(period, {}).get("macro_f1", np.nan)
            for period in periods
        ]
        linewidth = 2.4 if method in {"static", "tta_tc", "cotta", "sar"} else 1.5
        alpha = 1.0 if method in {"static", "tta_tc", "cotta", "sar"} else 0.65
        plt.plot(
            periods,
            values,
            marker="o",
            linewidth=linewidth,
            alpha=alpha,
            label=method_label(method),
        )
    plt.title(f"{dataset_name}: sequential TTA macro-F1")
    plt.xlabel("Target period")
    plt.ylabel("Macro-F1")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.legend(ncol=2, fontsize=8)
    out_path = os.path.join(out_dir, f"{dataset_name.lower()}_tta_macro_f1_curve.png")
    savefig(out_path)

    final_period = periods[-1]
    final_rows = []
    for method in methods:
        item = data[method]["periods"].get(final_period, {})
        final_rows.append({
            "method": method,
            "macro_f1": item.get("macro_f1", np.nan),
            "accuracy": item.get("accuracy", np.nan),
            "aurc": data[method].get("aurc", np.nan),
        })
    final_rows = sorted(final_rows, key=lambda r: r["macro_f1"], reverse=True)

    plt.figure(figsize=(8, 4.8))
    labels = [method_label(r["method"]) for r in final_rows]
    values = [r["macro_f1"] for r in final_rows]
    colors = ["#4C78A8" if r["method"] == "static" else "#F58518" for r in final_rows]
    plt.bar(labels, values, color=colors)
    plt.title(f"{dataset_name}: final-period macro-F1 ({final_period})")
    plt.xlabel("Method")
    plt.ylabel("Macro-F1")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    out_bar = os.path.join(out_dir, f"{dataset_name.lower()}_tta_final_macro_f1_bar.png")
    savefig(out_bar)
    return {
        "curve": out_path,
        "bar": out_bar,
        "final_period": final_period,
        "final_rows": final_rows,
    }, None


def load_period_metric_rows(norm_path, adabn_path):
    rows = []
    if exists(norm_path):
        for row in read_csv(norm_path):
            row = dict(row)
            row["method"] = row.get("norm", "")
            rows.append(row)
    if exists(adabn_path):
        for row in read_csv(adabn_path):
            row = dict(row)
            rows.append(row)
    return rows


def load_group_metric_rows(norm_path, adabn_path):
    rows = []
    if exists(norm_path):
        for row in read_csv(norm_path):
            row = dict(row)
            row["method"] = row.get("norm", "")
            rows.append(row)
    if exists(adabn_path):
        rows.extend(read_csv(adabn_path))
    return rows


def plot_norm_results(args, out_dir):
    period_rows = load_period_metric_rows(
        args.norm_period_csv,
        args.adabn_period_csv,
    )
    group_rows = load_group_metric_rows(
        args.norm_group_csv,
        args.adabn_group_csv,
    )
    if not period_rows:
        return None, "Missing norm/AdaBN period metrics."

    periods = sorted({r["period"] for r in period_rows}, key=period_sort_key)
    methods = [m for m in ["gn", "in", "bn", "ln", "bn_static", "bn_adabn"]
               if any(r["method"] == m for r in period_rows)]

    plt.figure(figsize=(8.5, 4.8))
    for method in methods:
        values = []
        for period in periods:
            match = [r for r in period_rows if r["method"] == method and r["period"] == period]
            values.append(as_float(match[0]["macro_f1"]) if match else np.nan)
        plt.plot(periods, values, marker="o", linewidth=2, label=method_label(method))
    plt.title("TLS22 normalization / AdaBN macro-F1 by period")
    plt.xlabel("Target period")
    plt.ylabel("Macro-F1")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    out_curve = os.path.join(out_dir, "tls22_norm_adabn_macro_f1_curve.png")
    savefig(out_curve)

    final_period = periods[-1]
    final_rows = [
        r for r in period_rows
        if r["period"] == final_period and r["method"] in methods
    ]
    final_rows = sorted(final_rows, key=lambda r: as_float(r["macro_f1"]), reverse=True)
    plt.figure(figsize=(8.2, 4.8))
    labels = [method_label(r["method"]) for r in final_rows]
    values = [as_float(r["macro_f1"]) for r in final_rows]
    plt.bar(labels, values, color="#4C78A8")
    plt.title(f"TLS22 normalization / AdaBN final macro-F1 ({final_period})")
    plt.ylabel("Macro-F1")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    out_bar = os.path.join(out_dir, "tls22_norm_adabn_final_macro_f1_bar.png")
    savefig(out_bar)

    group_plot = None
    important_groups = [
        "stable",
        "final_collapsed",
        "abrupt_collapsed",
        "gradual_collapsed",
        "absorber",
        "degraded_noncollapsed",
    ]
    final_group_rows = [
        r for r in group_rows
        if r.get("period") == final_period and r.get("group") in important_groups
    ]
    if final_group_rows:
        available_methods = [
            m for m in methods
            if any(r["method"] == m for r in final_group_rows)
        ]
        x = np.arange(len(important_groups))
        width = 0.8 / max(len(available_methods), 1)
        plt.figure(figsize=(11, 5.2))
        for i, method in enumerate(available_methods):
            vals = []
            for group in important_groups:
                match = [
                    r for r in final_group_rows
                    if r["method"] == method and r["group"] == group
                ]
                vals.append(as_float(match[0].get("macro_f1")) if match else np.nan)
            plt.bar(x + i * width, vals, width=width, label=method_label(method))
        plt.xticks(
            x + width * (len(available_methods) - 1) / 2,
            important_groups,
            rotation=25,
            ha="right",
        )
        plt.title(f"TLS22 drift-type group F1 by normalization ({final_period})")
        plt.ylabel("Group macro-F1")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(ncol=3, fontsize=8)
        group_plot = os.path.join(out_dir, "tls22_norm_adabn_m12_group_f1.png")
        savefig(group_plot)

    return {
        "curve": out_curve,
        "bar": out_bar,
        "group_bar": group_plot,
        "final_period": final_period,
        "final_rows": final_rows,
    }, None


def load_final_collapsed_classes(collapse_classes_path):
    if not exists(collapse_classes_path):
        return []
    classes = []
    for row in read_csv(collapse_classes_path):
        first = row.get("first_collapse_period", "")
        final_recall = as_float(row.get("final_recall"))
        if first and final_recall < 0.1:
            classes.append(int(float(row["class_id"])))
    return classes


def plot_collapse_heatmap(args, out_dir):
    if not exists(args.collapse_timeline_csv):
        return None, f"Missing collapse timeline: {args.collapse_timeline_csv}"
    rows = read_csv(args.collapse_timeline_csv)
    collapsed_classes = load_final_collapsed_classes(args.collapse_classes_csv)
    if not collapsed_classes:
        collapsed_classes = sorted({int(float(r["class_id"])) for r in rows})[:20]
    periods = sorted({r["period"] for r in rows}, key=period_sort_key)
    by_key = {
        (int(float(r["class_id"])), r["period"]): as_float(r.get("recall"))
        for r in rows
    }
    matrix = np.array([
        [by_key.get((class_id, period), np.nan) for period in periods]
        for class_id in collapsed_classes
    ])
    plt.figure(figsize=(max(8, len(periods) * 0.8), max(4, len(collapsed_classes) * 0.35)))
    im = plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="Recall")
    plt.xticks(np.arange(len(periods)), periods, rotation=35, ha="right")
    plt.yticks(np.arange(len(collapsed_classes)), [str(c) for c in collapsed_classes])
    plt.xlabel("Period")
    plt.ylabel("Class")
    plt.title("TLS22 final-collapsed class recall timeline")
    out_path = os.path.join(out_dir, "tls22_collapse_recall_heatmap.png")
    savefig(out_path)
    return {"heatmap": out_path, "classes": collapsed_classes, "periods": periods}, None


def format_table(rows, headers, keys, float_digits=4):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(path, generated, missing):
    lines = [
        "# Advisor-Facing Result Visualizations",
        "",
        "This folder contains figures generated from existing TTA, normalization, AdaBN, and collapse outputs.",
        "",
        "## Generated Figures",
        "",
    ]
    if not generated:
        lines.append("- No figures were generated.")
    else:
        for name, value in generated.items():
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    if isinstance(sub_value, str) and sub_value.endswith(".png"):
                        lines.append(f"- **{name}/{sub_name}**: `{sub_value}`")
            elif isinstance(value, str):
                lines.append(f"- **{name}**: `{value}`")

    if missing:
        lines.extend(["", "## Missing Inputs", ""])
        for item in missing:
            lines.append(f"- {item}")

    for key in ["tls22_tta", "quic22_tta", "norm"]:
        info = generated.get(key)
        if not isinstance(info, dict) or "final_rows" not in info:
            continue
        lines.extend(["", f"## {key} Final-Period Table", ""])
        rows = []
        for row in info["final_rows"]:
            rows.append({
                "method": method_label(row.get("method", "")),
                "macro_f1": as_float(row.get("macro_f1")),
                "accuracy": as_float(row.get("accuracy")),
                "aurc": as_float(row.get("aurc")),
            })
        lines.append(format_table(
            rows,
            ["method", "macro_f1", "accuracy", "aurc"],
            ["method", "macro_f1", "accuracy", "aurc"],
        ))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/teacher_result_visuals")
    parser.add_argument("--tls22-tta-json", default="outputs/eval_tls22/results_sequential.json")
    parser.add_argument("--quic22-tta-json", default="outputs/eval_quic22/results_sequential.json")
    parser.add_argument("--norm-period-csv", default="outputs/norm_drift_type_ablation_tls22/norm_period_metrics.csv")
    parser.add_argument("--norm-group-csv", default="outputs/norm_drift_type_ablation_tls22/norm_group_metrics.csv")
    parser.add_argument("--adabn-period-csv", default="outputs/adabn_drift_type_ablation_tls22/adabn_period_metrics.csv")
    parser.add_argument("--adabn-group-csv", default="outputs/adabn_drift_type_ablation_tls22/adabn_group_metrics.csv")
    parser.add_argument("--collapse-timeline-csv", default="outputs/per_class_collapse_tls22_monthly/collapse_timeline.csv")
    parser.add_argument("--collapse-classes-csv", default="outputs/per_class_collapse_tls22_monthly/collapse_classes.csv")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    generated = {}
    missing = []

    info, err = plot_tta_curves(args.tls22_tta_json, args.output_dir, "TLS22")
    if err:
        missing.append(err)
    else:
        generated["tls22_tta"] = info

    info, err = plot_tta_curves(args.quic22_tta_json, args.output_dir, "QUIC22")
    if err:
        missing.append(err)
    else:
        generated["quic22_tta"] = info

    info, err = plot_norm_results(args, args.output_dir)
    if err:
        missing.append(err)
    else:
        generated["norm"] = info

    info, err = plot_collapse_heatmap(args, args.output_dir)
    if err:
        missing.append(err)
    else:
        generated["collapse"] = info

    summary_path = os.path.join(args.output_dir, "teacher_result_visuals_summary.md")
    write_summary(summary_path, generated, missing)
    print(f"Saved visualizations to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    if missing:
        print("Missing inputs:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
