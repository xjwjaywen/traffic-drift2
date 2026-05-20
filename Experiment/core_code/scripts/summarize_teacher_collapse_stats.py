"""
Create compact advisor-facing collapse statistics from existing TLS22 outputs.

This script does not rerun any model. It consumes the normalization/AdaBN/TTA
CSV outputs and summarizes whether the additional methods actually recover
collapsed classes.

Usage from Experiment/core_code/:
    python scripts/summarize_teacher_collapse_stats.py \
      --output-dir outputs/teacher_result_visuals
"""
import argparse
import csv
import os
import tempfile

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
    "static": "Static",
    "tta_tc": "TTA-TC",
    "eata": "EATA",
    "cotta": "CoTTA",
    "sar": "SAR",
    "tent": "Tent",
    "note": "NOTE",
    "bn_adapt": "BN-Adapt",
}


def read_csv(path):
    if not os.path.exists(path):
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


def method_label(method):
    return METHOD_LABELS.get(method, method)


def load_collapse_groups(path, collapse_threshold):
    rows = read_csv(path)
    final_collapsed = []
    abrupt = []
    gradual = []
    for row in rows:
        first = row.get("first_collapse_period") or ""
        final_recall = as_float(row.get("final_recall"))
        if not first or not np.isfinite(final_recall) or final_recall >= collapse_threshold:
            continue
        class_id = as_int(row["class_id"])
        final_collapsed.append(class_id)
        pattern = row.get("collapse_pattern") or ""
        if pattern == "abrupt":
            abrupt.append(class_id)
        elif pattern == "gradual":
            gradual.append(class_id)
    return {
        "final_collapsed": final_collapsed,
        "abrupt_collapsed": abrupt,
        "gradual_collapsed": gradual,
    }


def load_method_class_metrics(norm_path, adabn_path, period):
    by_method_class = {}
    for row in read_csv(norm_path):
        if row.get("period") != period:
            continue
        method = row.get("norm", "")
        class_id = as_int(row.get("class_id"))
        by_method_class[(method, class_id)] = row
    for row in read_csv(adabn_path):
        if row.get("period") != period:
            continue
        method = row.get("method", "")
        class_id = as_int(row.get("class_id"))
        by_method_class[(method, class_id)] = row
    return by_method_class


def summarize_thresholds(by_method_class, groups, methods, thresholds):
    rows = []
    for group_name, class_ids in groups.items():
        for method in methods:
            recalls = [
                as_float(by_method_class.get((method, class_id), {}).get("recall"))
                for class_id in class_ids
            ]
            f1s = [
                as_float(by_method_class.get((method, class_id), {}).get("f1"))
                for class_id in class_ids
            ]
            recalls = np.array([x for x in recalls if np.isfinite(x)], dtype=float)
            f1s = np.array([x for x in f1s if np.isfinite(x)], dtype=float)
            row = {
                "scope": "norm_adabn",
                "group": group_name,
                "method": method,
                "method_label": method_label(method),
                "n_classes": len(class_ids),
                "mean_recall": float(np.mean(recalls)) if recalls.size else "",
                "median_recall": float(np.median(recalls)) if recalls.size else "",
                "mean_f1": float(np.mean(f1s)) if f1s.size else "",
            }
            for threshold in thresholds:
                key = str(threshold).replace(".", "_")
                row[f"recall_lt_{key}"] = int(np.sum(recalls < threshold)) if recalls.size else ""
            rows.append(row)
    return rows


def summarize_pairwise_delta(by_method_class, groups, comparisons, epsilon):
    rows = []
    for group_name, class_ids in groups.items():
        for method, baseline in comparisons:
            deltas = []
            for class_id in class_ids:
                cur = as_float(by_method_class.get((method, class_id), {}).get("recall"))
                base = as_float(by_method_class.get((baseline, class_id), {}).get("recall"))
                if np.isfinite(cur) and np.isfinite(base):
                    deltas.append(cur - base)
            deltas = np.array(deltas, dtype=float)
            rows.append({
                "scope": "method_delta",
                "group": group_name,
                "method": method,
                "baseline": baseline,
                "method_label": method_label(method),
                "baseline_label": method_label(baseline),
                "n_classes": len(class_ids),
                "improved_classes": int(np.sum(deltas > epsilon)) if deltas.size else "",
                "harmed_classes": int(np.sum(deltas < -epsilon)) if deltas.size else "",
                "unchanged_classes": int(np.sum(np.abs(deltas) <= epsilon)) if deltas.size else "",
                "mean_delta_recall": float(np.mean(deltas)) if deltas.size else "",
                "median_delta_recall": float(np.median(deltas)) if deltas.size else "",
            })
    return rows


def summarize_tta_group_metrics(path, period):
    rows = []
    wanted_groups = {"stable", "final_collapsed", "abrupt_collapsed", "gradual_collapsed"}
    wanted_methods = {"static", "tta_tc", "eata", "cotta", "sar", "tent", "note", "bn_adapt"}
    for row in read_csv(path):
        if row.get("period") != period:
            continue
        if row.get("group") not in wanted_groups or row.get("method") not in wanted_methods:
            continue
        rows.append({
            "scope": "tta_group",
            "group": row["group"],
            "method": row["method"],
            "method_label": method_label(row["method"]),
            "n_classes": row.get("n_classes", ""),
            "mean_recall": row.get("macro_recall", ""),
            "mean_f1": row.get("macro_f1", ""),
            "recall_lt_0_01": row.get("severe_count", ""),
            "recall_lt_0_1": row.get("collapsed_count", ""),
        })
    return rows


def format_value(value, digits=4):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path, threshold_rows, delta_rows, tta_rows, period):
    lines = [
        "# Collapse Statistics Summary",
        "",
        f"- Final period: `{period}`",
        "- `recall_lt_0_01` is the number of classes with recall < 0.01.",
        "- `recall_lt_0_05` is the number of classes with recall < 0.05.",
        "- `recall_lt_0_1` is the number of classes with recall < 0.1.",
        "",
        "## Normalization / AdaBN Threshold Counts",
        "",
        "| group | method | n | mean recall | mean F1 | <0.01 | <0.05 | <0.1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        if row["group"] not in {"final_collapsed", "abrupt_collapsed", "gradual_collapsed"}:
            continue
        lines.append(
            "| {group} | {method} | {n} | {recall} | {f1} | {lt001} | {lt005} | {lt01} |".format(
                group=row["group"],
                method=row["method_label"],
                n=row["n_classes"],
                recall=format_value(row["mean_recall"]),
                f1=format_value(row["mean_f1"]),
                lt001=row.get("recall_lt_0_01", ""),
                lt005=row.get("recall_lt_0_05", ""),
                lt01=row.get("recall_lt_0_1", ""),
            )
        )

    lines.extend([
        "",
        "## Class-Level Delta Counts",
        "",
        "| group | method vs baseline | improved | harmed | unchanged | mean delta recall |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in delta_rows:
        if row["group"] not in {"final_collapsed", "abrupt_collapsed", "gradual_collapsed"}:
            continue
        lines.append(
            "| {group} | {method} vs {base} | {imp} | {harm} | {unch} | {delta} |".format(
                group=row["group"],
                method=row["method_label"],
                base=row["baseline_label"],
                imp=row["improved_classes"],
                harm=row["harmed_classes"],
                unch=row["unchanged_classes"],
                delta=format_value(row["mean_delta_recall"]),
            )
        )

    if tta_rows:
        lines.extend([
            "",
            "## TTA Drift-Type Group Summary",
            "",
            "| group | method | n | mean recall | mean F1 | severe | collapsed |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in tta_rows:
            if row["group"] not in {"stable", "final_collapsed", "abrupt_collapsed", "gradual_collapsed"}:
                continue
            lines.append(
                "| {group} | {method} | {n} | {recall} | {f1} | {sev} | {col} |".format(
                    group=row["group"],
                    method=row["method_label"],
                    n=row.get("n_classes", ""),
                    recall=format_value(row.get("mean_recall")),
                    f1=format_value(row.get("mean_f1")),
                    sev=row.get("recall_lt_0_01", ""),
                    col=row.get("recall_lt_0_1", ""),
                )
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The key diagnostic is not whether a method moves macro-F1 by a small amount, but whether it reduces the number of collapsed classes with near-zero recall. If IN, AdaBN, or TTA methods improve only a few individual classes while most abrupt/gradual collapsed classes remain below 0.05 recall, the result supports a negative finding: normalization-statistics adaptation and generic TTA do not solve class-conditional collapse.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_threshold_counts(path, threshold_rows):
    rows = [
        row for row in threshold_rows
        if row["group"] == "final_collapsed"
        and row["method"] in {"gn", "in", "bn", "ln", "bn_adabn"}
    ]
    if not rows:
        return
    methods = [row["method_label"] for row in rows]
    lt001 = [as_float(row.get("recall_lt_0_01"), 0.0) for row in rows]
    lt005 = [as_float(row.get("recall_lt_0_05"), 0.0) for row in rows]
    lt01 = [as_float(row.get("recall_lt_0_1"), 0.0) for row in rows]
    x = np.arange(len(methods))
    width = 0.25
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - width, lt001, width=width, label="recall < 0.01")
    plt.bar(x, lt005, width=width, label="recall < 0.05")
    plt.bar(x + width, lt01, width=width, label="recall < 0.10")
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Number of final-collapsed classes")
    plt.title("TLS22 M12 collapsed-class recall threshold counts")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm-per-class-csv", default="outputs/norm_drift_type_ablation_tls22/norm_per_class_metrics.csv")
    parser.add_argument("--adabn-per-class-csv", default="outputs/adabn_drift_type_ablation_tls22/adabn_per_class_metrics.csv")
    parser.add_argument("--tta-group-csv", default="outputs/tta_drift_type_ablation_tls22/tta_drift_group_metrics.csv")
    parser.add_argument("--collapse-classes-csv", default="outputs/per_class_collapse_tls22_monthly/collapse_classes.csv")
    parser.add_argument("--period", default="M-2022-12")
    parser.add_argument("--output-dir", default="outputs/teacher_result_visuals")
    parser.add_argument("--collapse-threshold", type=float, default=0.1)
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    groups = load_collapse_groups(args.collapse_classes_csv, args.collapse_threshold)
    by_method_class = load_method_class_metrics(
        args.norm_per_class_csv,
        args.adabn_per_class_csv,
        args.period,
    )
    methods = [
        method for method in ["gn", "in", "bn", "ln", "bn_static", "bn_adabn"]
        if any(key[0] == method for key in by_method_class)
    ]
    threshold_rows = summarize_thresholds(
        by_method_class,
        groups,
        methods,
        thresholds=[0.01, 0.05, 0.1],
    )
    delta_rows = summarize_pairwise_delta(
        by_method_class,
        groups,
        comparisons=[("in", "gn"), ("bn_adabn", "bn"), ("bn_adabn", "bn_static")],
        epsilon=args.delta_epsilon,
    )
    tta_rows = summarize_tta_group_metrics(args.tta_group_csv, args.period)

    all_rows = threshold_rows + delta_rows + tta_rows
    csv_path = os.path.join(args.output_dir, "collapse_stat_summary.csv")
    md_path = os.path.join(args.output_dir, "collapse_stat_summary.md")
    plot_path = os.path.join(args.output_dir, "tls22_m12_collapse_recall_threshold_counts.png")
    write_csv(csv_path, all_rows)
    write_markdown(md_path, threshold_rows, delta_rows, tta_rows, args.period)
    plot_threshold_counts(plot_path, threshold_rows)

    print(f"Saved collapse statistic CSV: {csv_path}")
    print(f"Saved collapse statistic report: {md_path}")
    print(f"Saved threshold count plot: {plot_path}")


if __name__ == "__main__":
    main()
