"""
Summarize class-conditional drift diagnostics.

Reads outputs from scripts/class_conditional_drift.py and produces:
  - group_comparison.csv: bad vs stable group statistics and effect sizes
  - per_class_correlations.csv: class-level correlations with F1 drop
  - bad_confusion_targets.csv: top confusion targets for selected bad classes
  - summary_stats.json: compact machine-readable summary

Usage from Experiment/core_code/:
    python scripts/summarize_class_conditional_drift.py \
        --input-dir outputs/class_conditional_drift_tls22
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats


GROUP_METRICS = [
    "centroid_shift_from_ref",
    "delta_radius_from_ref",
    "delta_nearest_distance_from_ref",
    "delta_margin_from_ref",
    "delta_current_partner_distance_from_ref",
    "size_sum_w1",
    "direction_front_0_9_sum_w1",
    "ipt_tail_20_29_sum_w1",
    "total_norm_w1_std_floor_1",
    "total_log_w1",
    "delta_f1_from_ref",
]

CORRELATION_METRICS = [
    "centroid_shift_from_ref",
    "delta_radius_from_ref",
    "delta_nearest_distance_from_ref",
    "delta_margin_from_ref",
    "delta_current_partner_distance_from_ref",
    "size_sum_w1",
    "direction_front_0_9_sum_w1",
    "ipt_tail_20_29_sum_w1",
    "total_norm_w1_std_floor_1",
    "total_log_w1",
]


def parse_value(value):
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and value.lower() in {"none", "nan"}:
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return value


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def write_csv(rows, path, fieldnames=None):
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


def index_rows(rows, keys):
    indexed = {}
    for row in rows:
        indexed[tuple(row[key] for key in keys)] = row
    return indexed


def merge_class_rows(per_class, feature, input_drift):
    feature_idx = index_rows(feature, ["period", "class_id"])
    input_idx = index_rows(input_drift, ["period", "class_id"])
    merged = []
    for row in per_class:
        key = (row["period"], row["class_id"])
        out = dict(row)
        if key in feature_idx:
            out.update({
                k: v for k, v in feature_idx[key].items()
                if k not in {"period", "class_id", "support"}
            })
        if key in input_idx:
            out.update({
                k: v for k, v in input_idx[key].items()
                if k not in {"period", "class_id", "support"}
            })
        merged.append(out)
    return merged


def numeric_values(rows, metric):
    vals = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            vals.append(float(value))
    return np.asarray(vals, dtype=np.float64)


def describe_group(rows, metric):
    vals = numeric_values(rows, metric)
    if len(vals) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
    }


def cohens_d(a, b):
    a_vals = np.asarray(a, dtype=np.float64)
    b_vals = np.asarray(b, dtype=np.float64)
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None
    var_a = a_vals.var(ddof=1)
    var_b = b_vals.var(ddof=1)
    pooled = ((len(a_vals) - 1) * var_a + (len(b_vals) - 1) * var_b) / (
        len(a_vals) + len(b_vals) - 2
    )
    if pooled <= 0:
        return None
    return float((a_vals.mean() - b_vals.mean()) / math.sqrt(pooled))


def mannwhitney_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def group_comparison(merged_rows, bad_ids, stable_ids, final_period):
    period_rows = [row for row in merged_rows if row["period"] == final_period]
    bad_rows = [row for row in period_rows if int(row["class_id"]) in bad_ids]
    stable_rows = [row for row in period_rows if int(row["class_id"]) in stable_ids]

    rows = []
    for metric in GROUP_METRICS:
        bad_vals = numeric_values(bad_rows, metric)
        stable_vals = numeric_values(stable_rows, metric)
        bad_desc = describe_group(bad_rows, metric)
        stable_desc = describe_group(stable_rows, metric)
        rows.append({
            "period": final_period,
            "metric": metric,
            "bad_n": bad_desc["n"],
            "bad_mean": bad_desc["mean"],
            "bad_median": bad_desc["median"],
            "bad_std": bad_desc["std"],
            "stable_n": stable_desc["n"],
            "stable_mean": stable_desc["mean"],
            "stable_median": stable_desc["median"],
            "stable_std": stable_desc["std"],
            "mean_diff_bad_minus_stable": (
                bad_desc["mean"] - stable_desc["mean"]
                if bad_desc["mean"] is not None and stable_desc["mean"] is not None
                else None
            ),
            "cohens_d_bad_minus_stable": cohens_d(bad_vals, stable_vals),
            "mannwhitney_p": mannwhitney_p(bad_vals, stable_vals),
        })
    return rows


def safe_corr(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if isinstance(x, (int, float))
        and isinstance(y, (int, float))
        and math.isfinite(x)
        and math.isfinite(y)
    ]
    if len(pairs) < 3:
        return None
    x_arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    pearson = stats.pearsonr(x_arr, y_arr)
    spearman = stats.spearmanr(x_arr, y_arr)
    return {
        "n": int(len(pairs)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def per_class_correlations(merged_rows, final_period, min_support):
    rows = [
        row for row in merged_rows
        if row["period"] == final_period
        and isinstance(row.get("support"), (int, float))
        and row["support"] >= min_support
        and isinstance(row.get("delta_f1_from_ref"), (int, float))
    ]
    out = []
    y = [row["delta_f1_from_ref"] for row in rows]
    y_drop = [-row["delta_f1_from_ref"] for row in rows]
    for metric in CORRELATION_METRICS:
        xs = [row.get(metric) for row in rows]
        corr_delta = safe_corr(xs, y)
        corr_drop = safe_corr(xs, y_drop)
        if corr_delta is None:
            continue
        out.append({
            "period": final_period,
            "metric": metric,
            "target": "delta_f1_from_ref",
            **corr_delta,
        })
        out.append({
            "period": final_period,
            "metric": metric,
            "target": "f1_drop_from_ref",
            **corr_drop,
        })
    return out


def bad_confusion_targets(confusion_rows, bad_ids, final_period, top_per_class):
    rows = [
        row for row in confusion_rows
        if row["period"] == final_period and int(row["true_class"]) in bad_ids
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["true_class"])].append(row)

    out = []
    for true_class, items in grouped.items():
        items = sorted(items, key=lambda r: (r["confusion_rate"], r["confusion_count"]), reverse=True)
        for rank, row in enumerate(items[:top_per_class], start=1):
            out.append({
                "period": final_period,
                "true_class": true_class,
                "pred_class": int(row["pred_class"]),
                "rank_for_bad_class": rank,
                "confusion_count": int(row["confusion_count"]),
                "confusion_rate": float(row["confusion_rate"]),
                "delta_count_from_ref": row.get("delta_count_from_ref"),
                "delta_rate_from_ref": row.get("delta_rate_from_ref"),
            })
    return sorted(out, key=lambda r: (r["true_class"], r["rank_for_bad_class"]))


def infer_periods(summary, per_class_rows):
    reference = summary.get("reference_period")
    final = summary.get("final_period")
    if reference and final:
        return reference, final
    periods = []
    for row in per_class_rows:
        if row["period"] not in periods:
            periods.append(row["period"])
    return periods[0], periods[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--top-confusions-per-class", type=int, default=5)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    per_class = read_csv(os.path.join(input_dir, "per_class_metrics.csv"))
    feature = read_csv(os.path.join(input_dir, "feature_geometry.csv"))
    input_drift = read_csv(os.path.join(input_dir, "class_input_drift.csv"))
    confusion = read_csv(os.path.join(input_dir, "confusion_pairs.csv"))
    bad = read_csv(os.path.join(input_dir, "selected_bad_classes.csv"))
    stable = read_csv(os.path.join(input_dir, "selected_stable_classes.csv"))

    reference_period, final_period = infer_periods(summary, per_class)
    bad_ids = {int(row["class_id"]) for row in bad}
    stable_ids = {int(row["class_id"]) for row in stable}
    merged = merge_class_rows(per_class, feature, input_drift)

    group_rows = group_comparison(merged, bad_ids, stable_ids, final_period)
    corr_rows = per_class_correlations(merged, final_period, args.min_support)
    bad_conf_rows = bad_confusion_targets(
        confusion, bad_ids, final_period, args.top_confusions_per_class
    )

    write_csv(group_rows, os.path.join(output_dir, "group_comparison.csv"))
    write_csv(corr_rows, os.path.join(output_dir, "per_class_correlations.csv"))
    write_csv(bad_conf_rows, os.path.join(output_dir, "bad_confusion_targets.csv"))

    compact = {
        "input_dir": input_dir,
        "reference_period": reference_period,
        "final_period": final_period,
        "bad_class_ids": sorted(bad_ids),
        "stable_class_ids": sorted(stable_ids),
        "top_group_effects": sorted(
            group_rows,
            key=lambda r: abs(r["cohens_d_bad_minus_stable"] or 0.0),
            reverse=True,
        )[:10],
        "top_correlations_with_f1_drop": sorted(
            [row for row in corr_rows if row["target"] == "f1_drop_from_ref"],
            key=lambda r: abs(r["spearman_r"]),
            reverse=True,
        )[:10],
    }
    with open(os.path.join(output_dir, "summary_stats.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print(f"Reference period: {reference_period}")
    print(f"Final period: {final_period}")
    print(f"Bad classes: {sorted(bad_ids)}")
    print(f"Stable classes: {sorted(stable_ids)}")
    print("\nTop group effects:")
    for row in compact["top_group_effects"][:8]:
        print(
            f"  {row['metric']:<40} "
            f"d={row['cohens_d_bad_minus_stable'] if row['cohens_d_bad_minus_stable'] is not None else float('nan'):+.3f} "
            f"bad_med={row['bad_median']} stable_med={row['stable_median']}"
        )
    print("\nTop correlations with F1 drop:")
    for row in compact["top_correlations_with_f1_drop"][:8]:
        print(
            f"  {row['metric']:<40} "
            f"Spearman={row['spearman_r']:+.3f} p={row['spearman_p']:.3g}"
        )
    print(f"\nSaved summaries to: {output_dir}")


if __name__ == "__main__":
    main()
