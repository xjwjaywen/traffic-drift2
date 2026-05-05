"""
Summarize per-class collapse timelines from class-conditional drift outputs.

This script consumes CSVs produced by scripts/class_conditional_drift.py and
writes paper-ready diagnostics:
  - collapse_timeline.csv: per-period recall/F1/collapse status per class
  - collapse_classes.csv: first collapse period, final status, top absorbers
  - collapse_pairs.csv: top confusion targets for collapse-prone classes
  - summary.json and collapse_report.md

Usage from Experiment/core_code/:
    python scripts/summarize_per_class_collapse.py \
        --input-dir outputs/class_conditional_drift_tls22 \
        --output-dir outputs/per_class_collapse_tls22
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np


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


def period_sort_key(period):
    match = re.match(r"([MW])-2022-(\d+)$", period)
    if not match:
        return (period, 0)
    kind_order = 0 if match.group(1) == "M" else 1
    return (kind_order, int(match.group(2)))


def period_number(period):
    match = re.match(r"[MW]-2022-(\d+)$", period)
    return int(match.group(1)) if match else None


def load_inputs(args):
    per_class_path = args.per_class_csv
    confusion_path = args.confusion_csv
    if args.input_dir:
        per_class_path = per_class_path or os.path.join(args.input_dir, "per_class_metrics.csv")
        confusion_path = confusion_path or os.path.join(args.input_dir, "confusion_pairs.csv")
    if not per_class_path or not os.path.exists(per_class_path):
        raise FileNotFoundError(f"Missing per-class CSV: {per_class_path}")
    if not confusion_path or not os.path.exists(confusion_path):
        raise FileNotFoundError(f"Missing confusion CSV: {confusion_path}")
    return read_csv(per_class_path), read_csv(confusion_path), per_class_path, confusion_path


def top_confusions_by_class_period(confusion_rows, top_k):
    grouped = defaultdict(list)
    for row in confusion_rows:
        period = row["period"]
        true_class = as_int(row, "true_class")
        grouped[(period, true_class)].append(row)

    top_map = {}
    top_rows = []
    for key, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda r: (
                -as_float(r, "confusion_rate", 0.0),
                -as_int(r, "confusion_count"),
                as_int(r, "pred_class"),
            ),
        )
        top_map[key] = rows[0] if rows else None
        for rank, row in enumerate(rows[:top_k], start=1):
            out = {
                "period": row["period"],
                "true_class": as_int(row, "true_class"),
                "pred_class": as_int(row, "pred_class"),
                "rank_for_class": rank,
                "confusion_count": as_int(row, "confusion_count"),
                "confusion_rate": as_float(row, "confusion_rate", 0.0),
                "delta_count_from_ref": as_float(row, "delta_count_from_ref", ""),
                "delta_rate_from_ref": as_float(row, "delta_rate_from_ref", ""),
            }
            top_rows.append(out)
    return top_map, top_rows


def classify_collapse_pattern(
    period_rows,
    reference_period,
    first_collapse_period,
    threshold,
    abrupt_drop_threshold,
):
    if first_collapse_period is None:
        return "not_collapsed"
    ref = period_rows.get(reference_period)
    if ref and ref["recall"] < threshold:
        return "already_collapsed_at_reference"

    ordered = [period_rows[p] for p in sorted(period_rows, key=period_sort_key)]
    prev = None
    for row in ordered:
        if row["period"] == first_collapse_period:
            if prev is not None and prev["recall"] - row["recall"] >= abrupt_drop_threshold:
                return "abrupt"
            return "gradual"
        prev = row
    return "collapsed"


def summarize(per_class_rows, confusion_rows, args):
    periods = sorted({row["period"] for row in per_class_rows}, key=period_sort_key)
    if args.reference_period not in periods:
        raise ValueError(f"Reference period {args.reference_period} not found in per-class CSV.")
    final_period = args.final_period or periods[-1]
    if final_period not in periods:
        raise ValueError(f"Final period {final_period} not found in per-class CSV.")

    per_class = defaultdict(dict)
    for row in per_class_rows:
        class_id = as_int(row, "class_id")
        period = row["period"]
        per_class[class_id][period] = {
            "period": period,
            "class_id": class_id,
            "support": as_int(row, "support"),
            "precision": as_float(row, "precision", 0.0),
            "recall": as_float(row, "recall", 0.0),
            "f1": as_float(row, "f1", 0.0),
            "delta_recall_from_ref": as_float(row, "delta_recall_from_ref", None),
            "delta_f1_from_ref": as_float(row, "delta_f1_from_ref", None),
        }

    top_conf_map, top_pair_rows_all = top_confusions_by_class_period(
        confusion_rows, args.top_k_confusions
    )

    timeline_rows = []
    class_rows = []
    collapse_class_ids = set()
    final_collapsed_ids = set()

    for class_id in sorted(per_class):
        rows_by_period = per_class[class_id]
        first_collapse_period = None
        final_row = rows_by_period.get(final_period)
        ref_row = rows_by_period.get(args.reference_period)
        supported_periods = [
            p for p in periods
            if rows_by_period.get(p, {}).get("support", 0) >= args.min_support
        ]

        for period in periods:
            row = rows_by_period.get(period)
            if row is None:
                continue
            collapsed = (
                row["support"] >= args.min_support
                and row["recall"] < args.collapse_recall_threshold
            )
            if collapsed and first_collapse_period is None:
                first_collapse_period = period
            top = top_conf_map.get((period, class_id))
            timeline_rows.append({
                **row,
                "is_collapsed": int(collapsed),
                "top_confusion_target": as_int(top, "pred_class", "") if top else "",
                "top_confusion_count": as_int(top, "confusion_count", "") if top else "",
                "top_confusion_rate": as_float(top, "confusion_rate", "") if top else "",
            })

        if first_collapse_period is not None:
            collapse_class_ids.add(class_id)
        if (
            final_row
            and final_row["support"] >= args.min_support
            and final_row["recall"] < args.collapse_recall_threshold
        ):
            final_collapsed_ids.add(class_id)

        first_row = rows_by_period.get(first_collapse_period) if first_collapse_period else None
        final_top = top_conf_map.get((final_period, class_id))
        first_top = top_conf_map.get((first_collapse_period, class_id)) if first_collapse_period else None

        recalls = []
        period_nums = []
        for period in supported_periods:
            num = period_number(period)
            if num is not None:
                period_nums.append(num)
                recalls.append(rows_by_period[period]["recall"])
        recall_slope = ""
        if len(period_nums) >= 2:
            recall_slope = float(np.polyfit(period_nums, recalls, deg=1)[0])

        class_rows.append({
            "class_id": class_id,
            "reference_support": ref_row["support"] if ref_row else "",
            "reference_recall": ref_row["recall"] if ref_row else "",
            "reference_f1": ref_row["f1"] if ref_row else "",
            "final_support": final_row["support"] if final_row else "",
            "final_recall": final_row["recall"] if final_row else "",
            "final_f1": final_row["f1"] if final_row else "",
            "delta_final_recall_from_ref": (
                final_row["recall"] - ref_row["recall"]
                if final_row and ref_row else ""
            ),
            "delta_final_f1_from_ref": (
                final_row["f1"] - ref_row["f1"]
                if final_row and ref_row else ""
            ),
            "min_recall": min(
                (rows_by_period[p]["recall"] for p in supported_periods),
                default="",
            ),
            "first_collapse_period": first_collapse_period or "",
            "first_collapse_recall": first_row["recall"] if first_row else "",
            "collapse_pattern": classify_collapse_pattern(
                rows_by_period,
                args.reference_period,
                first_collapse_period,
                args.collapse_recall_threshold,
                args.abrupt_drop_threshold,
            ),
            "num_supported_periods": len(supported_periods),
            "num_collapsed_periods": sum(
                1 for p in supported_periods
                if rows_by_period[p]["recall"] < args.collapse_recall_threshold
            ),
            "recall_slope_per_period": recall_slope,
            "final_top_confusion_target": as_int(final_top, "pred_class", "") if final_top else "",
            "final_top_confusion_rate": as_float(final_top, "confusion_rate", "") if final_top else "",
            "first_collapse_top_confusion_target": as_int(first_top, "pred_class", "") if first_top else "",
            "first_collapse_top_confusion_rate": as_float(first_top, "confusion_rate", "") if first_top else "",
        })

    collapse_pair_rows = [
        row for row in top_pair_rows_all
        if row["true_class"] in collapse_class_ids
    ]

    period_summary = []
    for period in periods:
        supported = [
            r for r in timeline_rows
            if r["period"] == period and r["support"] >= args.min_support
        ]
        collapsed = [r for r in supported if r["is_collapsed"]]
        severe = [
            r for r in supported
            if r["recall"] < args.severe_recall_threshold
        ]
        period_summary.append({
            "period": period,
            "supported_classes": len(supported),
            "collapsed_classes": len(collapsed),
            "severely_collapsed_classes": len(severe),
            "collapsed_fraction": len(collapsed) / len(supported) if supported else 0.0,
            "mean_recall_supported": float(np.mean([r["recall"] for r in supported])) if supported else "",
            "median_recall_supported": float(np.median([r["recall"] for r in supported])) if supported else "",
        })

    class_rows = sorted(
        class_rows,
        key=lambda r: (
            r["first_collapse_period"] == "",
            period_sort_key(r["first_collapse_period"]) if r["first_collapse_period"] else (99, 99),
            r["final_recall"] if r["final_recall"] != "" else 999,
            r["class_id"],
        ),
    )

    summary = {
        "reference_period": args.reference_period,
        "final_period": final_period,
        "collapse_recall_threshold": args.collapse_recall_threshold,
        "severe_recall_threshold": args.severe_recall_threshold,
        "min_support": args.min_support,
        "num_classes": len(per_class),
        "num_ever_collapsed_classes": len(collapse_class_ids),
        "num_final_collapsed_classes": len(final_collapsed_ids),
        "period_summary": period_summary,
        "top_final_collapsed_classes": [
            {
                "class_id": row["class_id"],
                "final_recall": row["final_recall"],
                "final_f1": row["final_f1"],
                "first_collapse_period": row["first_collapse_period"],
                "final_top_confusion_target": row["final_top_confusion_target"],
                "final_top_confusion_rate": row["final_top_confusion_rate"],
            }
            for row in class_rows
            if row["class_id"] in final_collapsed_ids
        ][:20],
    }

    return timeline_rows, class_rows, collapse_pair_rows, period_summary, summary


def write_report(path, summary, class_rows, pair_rows):
    lines = []
    lines.append("# Per-Class Collapse Diagnosis")
    lines.append("")
    lines.append(f"- Reference period: `{summary['reference_period']}`")
    lines.append(f"- Final period: `{summary['final_period']}`")
    lines.append(f"- Collapse threshold: recall < `{summary['collapse_recall_threshold']}`")
    lines.append(f"- Min support: `{summary['min_support']}`")
    lines.append(f"- Ever-collapsed classes: `{summary['num_ever_collapsed_classes']}`")
    lines.append(f"- Final collapsed classes: `{summary['num_final_collapsed_classes']}`")
    lines.append("")
    lines.append("## Period Summary")
    lines.append("")
    lines.append("| period | supported | collapsed | severe | collapsed fraction | median recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["period_summary"]:
        lines.append(
            f"| {row['period']} | {row['supported_classes']} | "
            f"{row['collapsed_classes']} | {row['severely_collapsed_classes']} | "
            f"{row['collapsed_fraction']:.3f} | "
            f"{row['median_recall_supported']:.4f} |"
        )
    lines.append("")
    lines.append("## Final Collapsed Classes")
    lines.append("")
    lines.append("| class | first collapse | final recall | final F1 | absorber | absorber rate | pattern |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    final_ids = {item["class_id"] for item in summary["top_final_collapsed_classes"]}
    for row in class_rows:
        if row["class_id"] not in final_ids:
            continue
        lines.append(
            f"| {row['class_id']} | {row['first_collapse_period']} | "
            f"{float(row['final_recall']):.4f} | {float(row['final_f1']):.4f} | "
            f"{row['final_top_confusion_target']} | "
            f"{float(row['final_top_confusion_rate'] or 0.0):.4f} | "
            f"{row['collapse_pattern']} |"
        )
    lines.append("")
    lines.append("## Top Collapse-Pair Rows")
    lines.append("")
    lines.append("| period | true | pred | rank | rate | count |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in pair_rows[:30]:
        lines.append(
            f"| {row['period']} | {row['true_class']} | {row['pred_class']} | "
            f"{row['rank_for_class']} | {row['confusion_rate']:.4f} | "
            f"{row['confusion_count']} |"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--per-class-csv", default=None)
    parser.add_argument("--confusion-csv", default=None)
    parser.add_argument("--output-dir", default="outputs/per_class_collapse_tls22")
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--final-period", default=None)
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--top-k-confusions", type=int, default=5)
    parser.add_argument("--abrupt-drop-threshold", type=float, default=0.3)
    args = parser.parse_args()

    per_class_rows, confusion_rows, per_class_path, confusion_path = load_inputs(args)
    os.makedirs(args.output_dir, exist_ok=True)

    timeline_rows, class_rows, pair_rows, period_summary, summary = summarize(
        per_class_rows, confusion_rows, args
    )
    summary["per_class_csv"] = per_class_path
    summary["confusion_csv"] = confusion_path

    write_csv(os.path.join(args.output_dir, "collapse_timeline.csv"), timeline_rows)
    write_csv(os.path.join(args.output_dir, "collapse_classes.csv"), class_rows)
    write_csv(os.path.join(args.output_dir, "collapse_pairs.csv"), pair_rows)
    write_csv(os.path.join(args.output_dir, "collapse_period_summary.csv"), period_summary)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(
        os.path.join(args.output_dir, "collapse_report.md"),
        summary,
        class_rows,
        pair_rows,
    )

    print("=== Per-Class Collapse Diagnosis ===")
    print(
        f"Reference={summary['reference_period']} final={summary['final_period']} "
        f"threshold={summary['collapse_recall_threshold']} min_support={summary['min_support']}"
    )
    for row in period_summary:
        print(
            f"{row['period']}: collapsed={row['collapsed_classes']}/"
            f"{row['supported_classes']} severe={row['severely_collapsed_classes']} "
            f"median_recall={row['median_recall_supported']:.4f}"
        )
    print(
        f"Ever collapsed: {summary['num_ever_collapsed_classes']} | "
        f"Final collapsed: {summary['num_final_collapsed_classes']}"
    )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
