"""
Summarize CAPS target-prototype experiments across periods.

Reads output directories from scripts/caps_target_prototype_tls22.py and writes:
  - caps_period_summary.csv
  - caps_class_recovery.csv
  - caps_recovery_bins.csv
  - caps_pair_effects.csv

Usage:
    python scripts/summarize_caps_experiments.py \
        --input-dirs \
          outputs/caps_target_prototype_tls22_M-2022-7 \
          outputs/caps_target_prototype_tls22_M-2022-10 \
          outputs/caps_target_prototype_tls22_M-2022-12 \
        --output-dir outputs/caps_target_prototype_summary
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key, default=None):
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def as_int(row, key, default=0):
    value = row.get(key)
    if value in {None, ""}:
        return default
    return int(float(value))


def find_one(pattern, base_dir, required=True):
    matches = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not matches and required:
        raise FileNotFoundError(f"No file matching {pattern} in {base_dir}")
    return matches[0] if matches else None


def summarize_period(input_dir):
    with open(os.path.join(input_dir, "summary.json")) as f:
        meta = json.load(f)
    results = read_csv(os.path.join(input_dir, "results_by_params.csv"))
    static = next(row for row in results if row["method"] == "static")
    best = meta["best_metrics"]

    return {
        "period": meta["target_period"],
        "input_dir": input_dir,
        "static_overall_macro_f1": as_float(static, "overall_macro_f1"),
        "caps_overall_macro_f1": best["overall_macro_f1"],
        "delta_overall_macro_f1": best["overall_macro_f1"] - as_float(static, "overall_macro_f1"),
        "static_bad_macro_f1": as_float(static, "bad_macro_f1"),
        "caps_bad_macro_f1": best["bad_macro_f1"],
        "delta_bad_macro_f1": best["bad_macro_f1"] - as_float(static, "bad_macro_f1"),
        "static_stable_macro_f1": as_float(static, "stable_macro_f1"),
        "caps_stable_macro_f1": best["stable_macro_f1"],
        "delta_stable_macro_f1": best["stable_macro_f1"] - as_float(static, "stable_macro_f1"),
        "best_alpha": meta["best_alpha_by_bad_macro_f1"],
        "best_tau_conf": meta["best_tau_conf_by_bad_macro_f1"],
        "best_momentum": meta["best_momentum_by_bad_macro_f1"],
        "accepted_rate": meta["best_update_stats"]["accepted_rate"],
        "num_updated_classes": meta["best_update_stats"]["num_updated_classes"],
    }, meta


def load_class_rows(input_dir, period):
    path = find_one("per_class_metrics_*.csv", input_dir)
    rows = read_csv(path)
    out = []
    for row in rows:
        out.append({
            "period": period,
            "class": as_int(row, "class"),
            "group": row["group"],
            "reference_support": as_int(row, "reference_support"),
            "target_support": as_int(row, "target_support"),
            "accepted_updates": as_int(row, "accepted_updates"),
            "accepted_rate": (
                as_int(row, "accepted_updates") / as_int(row, "target_support")
                if as_int(row, "target_support") else 0.0
            ),
            "static_f1": as_float(row, "static_f1", 0.0),
            "caps_f1": as_float(row, "best_caps_f1", 0.0),
            "delta_f1": as_float(row, "delta_f1", 0.0),
            "static_recall": as_float(row, "static_recall", 0.0),
            "caps_recall": as_float(row, "best_caps_recall", 0.0),
            "delta_recall": as_float(row, "delta_recall", 0.0),
        })
    return out


def recovery_label(delta, recovered_threshold, harmed_threshold):
    if delta >= recovered_threshold:
        return "recovered"
    if delta <= harmed_threshold:
        return "harmed"
    return "unchanged"


def summarize_bins(class_rows, recovered_threshold, harmed_threshold):
    grouped = defaultdict(list)
    for row in class_rows:
        label = recovery_label(row["delta_f1"], recovered_threshold, harmed_threshold)
        grouped[(row["period"], row["group"], label)].append(row)

    rows = []
    for (period, group, label), items in sorted(grouped.items()):
        deltas = np.asarray([x["delta_f1"] for x in items], dtype=float)
        accepts = np.asarray([x["accepted_rate"] for x in items], dtype=float)
        rows.append({
            "period": period,
            "group": group,
            "recovery_bin": label,
            "n": len(items),
            "mean_delta_f1": float(deltas.mean()),
            "median_delta_f1": float(np.median(deltas)),
            "mean_accepted_rate": float(accepts.mean()),
            "median_accepted_rate": float(np.median(accepts)),
        })
    return rows


def load_pair_rows(input_dir, period):
    path = find_one("pair_summary_*.csv", input_dir, required=False)
    if path is None:
        return []
    rows = []
    for row in read_csv(path):
        rows.append({
            "period": period,
            "true_class": as_int(row, "true_class"),
            "pred_class": as_int(row, "pred_class"),
            "rank_for_bad_class": as_int(row, "rank_for_bad_class"),
            "support": as_int(row, "support"),
            "static_rate": as_float(row, "static_rate", 0.0),
            "caps_rate": as_float(row, "caps_rate", 0.0),
            "delta_rate": as_float(row, "delta_rate", 0.0),
            "static_count": as_int(row, "static_count"),
            "caps_count": as_int(row, "caps_count"),
            "delta_count": as_int(row, "delta_count"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/caps_target_prototype_summary")
    parser.add_argument("--recovered-threshold", type=float, default=0.01)
    parser.add_argument("--harmed-threshold", type=float, default=-0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    period_rows = []
    class_rows = []
    pair_rows = []
    for input_dir in args.input_dirs:
        period_row, meta = summarize_period(input_dir)
        period_rows.append(period_row)
        period = meta["target_period"]
        class_rows.extend(load_class_rows(input_dir, period))
        pair_rows.extend(load_pair_rows(input_dir, period))

    write_csv(os.path.join(args.output_dir, "caps_period_summary.csv"), period_rows)
    write_csv(os.path.join(args.output_dir, "caps_class_recovery.csv"), class_rows)
    write_csv(
        os.path.join(args.output_dir, "caps_recovery_bins.csv"),
        summarize_bins(class_rows, args.recovered_threshold, args.harmed_threshold),
    )
    write_csv(os.path.join(args.output_dir, "caps_pair_effects.csv"), pair_rows)

    print("=== CAPS Period Summary ===")
    for row in period_rows:
        print(
            f"{row['period']}: macro_f1 {row['static_overall_macro_f1']:.4f}"
            f" -> {row['caps_overall_macro_f1']:.4f}"
            f" | bad {row['static_bad_macro_f1']:.4f}"
            f" -> {row['caps_bad_macro_f1']:.4f}"
            f" | stable {row['static_stable_macro_f1']:.4f}"
            f" -> {row['caps_stable_macro_f1']:.4f}"
            f" | accepted={row['accepted_rate']:.3f}"
        )
    print(f"Saved summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()
