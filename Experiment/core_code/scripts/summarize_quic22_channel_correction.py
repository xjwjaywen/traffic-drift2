"""
Summarize QUIC22 channel-level drift correction results.

Inputs:
  - quantile_correct_eval.py summary.csv
  - quantify_drift.py summary.csv

Outputs:
  - quic22_channel_correction_report.md
  - quic22_channel_correction_delta_f1.png
  - quic22_channel_drift_w1.png
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=np.nan):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def save_correction_plot(rows, output_path):
    by_period = defaultdict(list)
    for row in rows:
        if row["setting"] == "raw":
            continue
        by_period[row["period"]].append(row)

    periods = sorted(by_period)
    settings = []
    for rows_for_period in by_period.values():
        for row in rows_for_period:
            if row["setting"] not in settings:
                settings.append(row["setting"])

    if not periods or not settings:
        return

    x = np.arange(len(settings))
    width = 0.8 / max(1, len(periods))

    plt.figure(figsize=(max(9.0, len(settings) * 1.6), 5.2))
    for idx, period in enumerate(periods):
        values_by_setting = {
            row["setting"]: as_float(row, "delta_macro_f1_vs_raw")
            for row in by_period[period]
        }
        values = [values_by_setting.get(setting, np.nan) for setting in settings]
        plt.bar(x + (idx - (len(periods) - 1) / 2.0) * width, values, width=width, label=period)

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, settings, rotation=25, ha="right")
    plt.ylabel("Delta macro-F1 vs raw")
    plt.title("QUIC22 channel quantile correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_drift_plot(rows, output_path):
    periods = [row["period"] for row in rows]
    if not periods:
        return

    channels = ["size", "direction", "ipt"]
    x = np.arange(len(periods))
    width = 0.24

    plt.figure(figsize=(8.8, 5.2))
    for idx, channel in enumerate(channels):
        values = [as_float(row, f"{channel}_sum_w1") for row in rows]
        plt.bar(x + (idx - 1) * width, values, width=width, label=channel)

    plt.xticks(x, periods, rotation=20, ha="right")
    plt.ylabel("Wasserstein-1 sum over positions")
    plt.title("QUIC22 input-channel drift magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def best_rows_by_period(correction_rows):
    grouped = defaultdict(list)
    for row in correction_rows:
        grouped[row["period"]].append(row)

    best = {}
    raw = {}
    for period, rows in grouped.items():
        raw_rows = [row for row in rows if row["setting"] == "raw"]
        if raw_rows:
            raw[period] = raw_rows[0]
        candidates = [row for row in rows if row["setting"] != "raw"]
        if candidates:
            best[period] = max(candidates, key=lambda r: as_float(r, "macro_f1"))
    return raw, best


def write_report(correction_rows, drift_rows, output_path):
    raw_by_period, best_by_period = best_rows_by_period(correction_rows)
    settings = []
    for row in correction_rows:
        if row["setting"] not in settings:
            settings.append(row["setting"])

    lines = [
        "# QUIC22 Channel Drift and Correction Summary",
        "",
        "This report checks whether QUIC22 temporal drift is tied to a specific PPI channel and whether channel-level correction can recover frozen-model performance.",
        "",
        "## Best Channel Correction",
        "",
        "| period | raw macro-F1 | best setting | best macro-F1 | delta |",
        "|---|---:|---|---:|---:|",
    ]

    for period in sorted(raw_by_period):
        raw_row = raw_by_period[period]
        best_row = best_by_period.get(period)
        if best_row is None:
            continue
        lines.append(
            f"| {period} | {as_float(raw_row, 'macro_f1'):.4f} | "
            f"{best_row['setting']} | {as_float(best_row, 'macro_f1'):.4f} | "
            f"{as_float(best_row, 'delta_macro_f1_vs_raw'):+.4f} |"
        )

    lines += [
        "",
        "## All Correction Settings",
        "",
        "| period | setting | macro-F1 | delta vs raw | corrected-region W1 reduction |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(correction_rows, key=lambda r: (r["period"], r["setting"])):
        lines.append(
            f"| {row['period']} | {row['setting']} | "
            f"{as_float(row, 'macro_f1'):.4f} | "
            f"{as_float(row, 'delta_macro_f1_vs_raw'):+.4f} | "
            f"{as_float(row, 'setting_region_w1_reduction'):.4f} |"
        )

    if drift_rows:
        lines += [
            "",
            "## Channel Drift Magnitude",
            "",
            "| period | macro-F1 | W1 size | W1 direction | W1 IPT | drifted positions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(drift_rows, key=lambda r: r["period"]):
            lines.append(
                f"| {row['period']} | {as_float(row, 'macro_f1'):.4f} | "
                f"{as_float(row, 'size_sum_w1'):.4f} | "
                f"{as_float(row, 'direction_sum_w1'):.4f} | "
                f"{as_float(row, 'ipt_sum_w1'):.4f} | "
                f"{int(as_float(row, 'total_drifted_count', 0))} |"
            )

    lines += [
        "",
        "## Reading",
        "",
        "- If one channel correction gives a positive delta while others do not, QUIC drift is likely channel/position-specific rather than class-collapse dominated.",
        "- If correcting all channels hurts, broad augmentation can be worse than targeted augmentation.",
        "- A positive channel-correction result is diagnostic; deployment still needs a practical augmentation or calibration rule that does not use labels.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction-summary", required=True)
    parser.add_argument("--drift-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    correction_rows = read_csv(args.correction_summary)
    drift_rows = read_csv(args.drift_summary) if os.path.exists(args.drift_summary) else []

    report_path = os.path.join(args.output_dir, "quic22_channel_correction_report.md")
    correction_plot = os.path.join(args.output_dir, "quic22_channel_correction_delta_f1.png")
    drift_plot = os.path.join(args.output_dir, "quic22_channel_drift_w1.png")

    write_report(correction_rows, drift_rows, report_path)
    save_correction_plot(correction_rows, correction_plot)
    save_drift_plot(drift_rows, drift_plot)

    print(f"Saved report: {report_path}")
    print(f"Saved correction plot: {correction_plot}")
    if drift_rows:
        print(f"Saved drift plot: {drift_plot}")


if __name__ == "__main__":
    main()

