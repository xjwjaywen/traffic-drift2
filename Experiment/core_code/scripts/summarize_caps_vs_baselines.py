"""
Build a compact CAPS-vs-baselines table.

Inputs:
  - evaluate_tta.py sequential JSON, e.g. outputs/eval_tls22/results_sequential.json
  - CAPS period summary, e.g.
    outputs/caps_target_prototype_summary/caps_period_summary.csv

The sequential baseline JSON only contains overall accuracy/macro-F1 by period.
Bad/stable macro-F1 is available only for static and CAPS from the CAPS summary.

Usage:
    python scripts/summarize_caps_vs_baselines.py \
        --baseline-json outputs/eval_tls22/results_sequential.json \
        --caps-summary outputs/caps_target_prototype_summary/caps_period_summary.csv \
        --output-dir outputs/caps_vs_baselines
"""
import argparse
import csv
import json
import os


METHOD_LABELS = {
    "static": "Static",
    "bn_adapt": "BN-Adapt",
    "tent": "Tent",
    "eata": "EATA",
    "cotta": "CoTTA",
    "sar": "SAR",
    "note": "NOTE",
    "mvfc": "MVFC",
    "tta_tc": "TTA-TC",
    "knn_labeled": "KNN-Labeled",
    "ft_head": "FT-Head",
    "supervised_norm": "Supervised-Norm",
    "caps": "CAPS",
}


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


def as_float(value):
    if value in {None, ""}:
        return None
    return float(value)


def load_baseline_results(path):
    with open(path) as f:
        payload = json.load(f)
    return payload.get("results", payload)


def load_caps_summary(path):
    rows = read_csv(path)
    return {row["period"]: row for row in rows}


def fmt(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def build_long_rows(baseline_results, caps_by_period, periods):
    rows = []
    for method_key, result in baseline_results.items():
        if not isinstance(result, dict) or "periods" not in result:
            continue
        for period in periods:
            period_metrics = result["periods"].get(period)
            if not period_metrics:
                continue
            row = {
                "method": METHOD_LABELS.get(method_key, method_key),
                "method_key": method_key,
                "period": period,
                "accuracy": period_metrics.get("accuracy"),
                "macro_f1": period_metrics.get("macro_f1"),
                "bad_macro_f1": "",
                "stable_macro_f1": "",
                "source": "evaluate_tta",
            }
            if method_key == "static" and period in caps_by_period:
                caps_row = caps_by_period[period]
                row["bad_macro_f1"] = caps_row["static_bad_macro_f1"]
                row["stable_macro_f1"] = caps_row["static_stable_macro_f1"]
            rows.append(row)

    for period in periods:
        if period not in caps_by_period:
            continue
        caps = caps_by_period[period]
        rows.append({
            "method": "CAPS",
            "method_key": "caps",
            "period": period,
            "accuracy": "",
            "macro_f1": caps["caps_overall_macro_f1"],
            "bad_macro_f1": caps["caps_bad_macro_f1"],
            "stable_macro_f1": caps["caps_stable_macro_f1"],
            "source": "caps",
            "accepted_rate": caps["accepted_rate"],
            "best_alpha": caps["best_alpha"],
            "best_tau_conf": caps["best_tau_conf"],
            "best_momentum": caps["best_momentum"],
        })
    return rows


def build_wide_rows(long_rows, periods):
    method_order = []
    by_method = {}
    for row in long_rows:
        key = row["method_key"]
        if key not in by_method:
            by_method[key] = {
                "method": row["method"],
                "method_key": key,
            }
            method_order.append(key)
        by_method[key][f"{row['period']}_macro_f1"] = row["macro_f1"]
        if row.get("bad_macro_f1") not in {None, ""}:
            by_method[key][f"{row['period']}_bad_macro_f1"] = row["bad_macro_f1"]
        if row.get("stable_macro_f1") not in {None, ""}:
            by_method[key][f"{row['period']}_stable_macro_f1"] = row["stable_macro_f1"]

    rows = []
    for key in method_order:
        out = by_method[key]
        formatted = {"method": out["method"], "method_key": key}
        for period in periods:
            formatted[f"{period}_macro_f1"] = fmt(out.get(f"{period}_macro_f1"))
        # Keep bad/stable columns for the last period if present; this is the
        # main collapse-focused comparison.
        last = periods[-1]
        formatted[f"{last}_bad_macro_f1"] = fmt(out.get(f"{last}_bad_macro_f1"))
        formatted[f"{last}_stable_macro_f1"] = fmt(out.get(f"{last}_stable_macro_f1"))
        rows.append(formatted)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--caps-summary", required=True)
    parser.add_argument("--output-dir", default="outputs/caps_vs_baselines")
    parser.add_argument("--periods", default="M-2022-7,M-2022-10,M-2022-12")
    args = parser.parse_args()

    periods = [p.strip() for p in args.periods.replace(" ", "").split(",") if p.strip()]
    baseline_results = load_baseline_results(args.baseline_json)
    caps_by_period = load_caps_summary(args.caps_summary)

    long_rows = build_long_rows(baseline_results, caps_by_period, periods)
    wide_rows = build_wide_rows(long_rows, periods)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "caps_vs_baselines_long.csv"), long_rows)
    write_csv(os.path.join(args.output_dir, "caps_vs_baselines_wide.csv"), wide_rows)

    print("=== CAPS vs Baselines ===")
    for row in wide_rows:
        values = " | ".join(
            f"{p}={row.get(f'{p}_macro_f1', '')}" for p in periods
        )
        last = periods[-1]
        print(
            f"{row['method']:<16} {values}"
            f" | {last} bad={row.get(f'{last}_bad_macro_f1', '')}"
            f" stable={row.get(f'{last}_stable_macro_f1', '')}"
        )
    print(f"Saved tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
