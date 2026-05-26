"""
Summarize reject-option ablations across TLS22 target periods.

Reads output directories from scripts/reject_option_ablation_tls22.py and writes:
  - reject_period_summary.csv
  - reject_rule_summary.csv
  - reject_option_period_report.md
  - reject_absorber_distance_periods.png
  - reject_rule_period_collapsed_vs_stable.png

Usage from Experiment/core_code/:
    python scripts/summarize_reject_option_experiments.py \
      --input-dirs \
        outputs/reject_option_ablation_tls22_M-2022-7 \
        outputs/reject_option_ablation_tls22_M-2022-10 \
        outputs/reject_option_ablation_tls22_M-2022-12 \
      --output-dir outputs/reject_option_ablation_tls22_summary
"""
import argparse
import csv
import glob
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tta_tc_matplotlib_cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "tta_tc_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FOCUS_RULES = [
    "absorber_distance",
    "absorber_proto_disagree",
    "prototype_distance",
    "confidence",
    "margin",
    "hybrid",
]


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


def find_one(input_dir, pattern):
    matches = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {input_dir}")
    return matches[0]


def period_sort_key(period):
    digits = "".join(ch for ch in period if ch.isdigit())
    if digits:
        return int(digits[-2:])
    return 0


def load_period(input_dir):
    with open(os.path.join(input_dir, "summary.json"), encoding="utf-8") as f:
        meta = json.load(f)
    period = meta["target_period"]
    best_rows = read_csv(find_one(input_dir, "reject_ablation_best_*.csv"))
    all_rows = read_csv(find_one(input_dir, "reject_ablation_all_*.csv"))
    static = meta.get("static_summary", {})
    return period, static, best_rows, all_rows


def normalize_best_row(period, static, row):
    collapsed = as_float(row.get("collapsed_reject_rate"), 0.0)
    stable_false = as_float(row.get("stable_false_reject_rate"), 0.0)
    absorber_red = as_float(row.get("absorber_error_reduction"), 0.0)
    coverage = as_float(row.get("coverage"), 0.0)
    deploy_score = collapsed - stable_false + 0.25 * absorber_red + 0.05 * coverage
    return {
        "period": period,
        "rule": row["rule"],
        "threshold_name": row.get("threshold_name", ""),
        "threshold_value": row.get("threshold_value", ""),
        "coverage": coverage,
        "reject_rate": as_float(row.get("reject_rate"), 0.0),
        "accepted_macro_f1": as_float(row.get("accepted_macro_f1"), 0.0),
        "collapsed_reject_rate": collapsed,
        "abrupt_reject_rate": as_float(row.get("abrupt_reject_rate"), 0.0),
        "gradual_reject_rate": as_float(row.get("gradual_reject_rate"), 0.0),
        "stable_false_reject_rate": stable_false,
        "absorber_error_reduction": absorber_red,
        "pair_absorber_error_reduction": as_float(row.get("pair_absorber_error_reduction"), 0.0),
        "original_collapse_absorber_errors": as_float(row.get("original_collapse_absorber_errors"), 0.0),
        "kept_collapse_absorber_errors": as_float(row.get("kept_collapse_absorber_errors"), 0.0),
        "num_rejected": as_float(row.get("num_rejected"), 0.0),
        "num_accepted": as_float(row.get("num_accepted"), 0.0),
        "static_macro_f1": as_float(static.get("overall_macro_f1"), 0.0),
        "static_collapsed_f1": as_float(static.get("bad_macro_f1"), 0.0),
        "static_stable_f1": as_float(static.get("stable_macro_f1"), 0.0),
        "deploy_score": deploy_score,
    }


def summarize_rules(rule_rows):
    grouped = {}
    for row in rule_rows:
        grouped.setdefault(row["rule"], []).append(row)
    out = []
    for rule, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: period_sort_key(r["period"]))
        out.append({
            "rule": rule,
            "periods": " ".join(row["period"] for row in rows),
            "mean_coverage": float(np.mean([row["coverage"] for row in rows])),
            "mean_collapsed_reject_rate": float(np.mean([row["collapsed_reject_rate"] for row in rows])),
            "mean_stable_false_reject_rate": float(np.mean([row["stable_false_reject_rate"] for row in rows])),
            "mean_absorber_error_reduction": float(np.mean([row["absorber_error_reduction"] for row in rows])),
            "mean_deploy_score": float(np.mean([row["deploy_score"] for row in rows])),
        })
    return out


def plot_absorber_distance(rows, output_dir):
    selected = [row for row in rows if row["rule"] == "absorber_distance"]
    if not selected:
        return None
    selected = sorted(selected, key=lambda r: period_sort_key(r["period"]))
    periods = [row["period"] for row in selected]
    x = np.arange(len(periods))
    width = 0.2
    plt.figure(figsize=(9.0, 5.2))
    plt.bar(x - 1.5 * width, [row["coverage"] for row in selected], width, label="coverage")
    plt.bar(x - 0.5 * width, [row["collapsed_reject_rate"] for row in selected], width, label="collapsed reject")
    plt.bar(x + 0.5 * width, [row["stable_false_reject_rate"] for row in selected], width, label="stable false reject")
    plt.bar(x + 1.5 * width, [row["absorber_error_reduction"] for row in selected], width, label="absorber error reduction")
    plt.xticks(x, periods)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Absorber-distance reject across periods")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    path = os.path.join(output_dir, "reject_absorber_distance_periods.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_rule_tradeoff(rows, output_dir):
    periods = sorted({row["period"] for row in rows}, key=period_sort_key)
    rules = [rule for rule in FOCUS_RULES if any(row["rule"] == rule for row in rows)]
    fig, axes = plt.subplots(1, len(periods), figsize=(5.0 * len(periods), 4.6), sharey=True)
    if len(periods) == 1:
        axes = [axes]
    for ax, period in zip(axes, periods):
        sub = [row for row in rows if row["period"] == period and row["rule"] in rules]
        for row in sub:
            ax.scatter(
                row["stable_false_reject_rate"],
                row["collapsed_reject_rate"],
                s=70,
                label=row["rule"],
            )
            ax.text(
                row["stable_false_reject_rate"],
                row["collapsed_reject_rate"],
                row["rule"].replace("_", "\n"),
                fontsize=7,
                ha="left",
                va="bottom",
            )
        ax.set_title(period)
        ax.set_xlabel("Stable false reject")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Collapsed reject")
    path = os.path.join(output_dir, "reject_rule_period_collapsed_vs_stable.png")
    fig.suptitle("Reject rule trade-off by period")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt(value):
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def write_report(path, period_rows, rule_summary, absorber_plot, tradeoff_plot):
    lines = [
        "# Reject-Option Multi-Period Summary",
        "",
        "This summary uses the final-collapsed class set from the collapse report as the risk group across periods.",
        "",
        "## Best Rule Rows",
        "",
        "| period | rule | coverage | collapsed reject | stable false reject | absorber error reduction | accepted macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(period_rows, key=lambda r: (period_sort_key(r["period"]), r["rule"])):
        if row["rule"] not in FOCUS_RULES:
            continue
        lines.append(
            f"| {row['period']} | {row['rule']} | {fmt(row['coverage'])} | "
            f"{fmt(row['collapsed_reject_rate'])} | {fmt(row['stable_false_reject_rate'])} | "
            f"{fmt(row['absorber_error_reduction'])} | {fmt(row['accepted_macro_f1'])} |"
        )
    lines.extend([
        "",
        "## Mean Across Periods",
        "",
        "| rule | mean coverage | mean collapsed reject | mean stable false reject | mean absorber error reduction |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rule_summary:
        lines.append(
            f"| {row['rule']} | {fmt(row['mean_coverage'])} | "
            f"{fmt(row['mean_collapsed_reject_rate'])} | "
            f"{fmt(row['mean_stable_false_reject_rate'])} | "
            f"{fmt(row['mean_absorber_error_reduction'])} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
    ])
    if absorber_plot:
        lines.append(f"- Absorber-distance periods: `{absorber_plot}`")
    if tradeoff_plot:
        lines.append(f"- Rule trade-off by period: `{tradeoff_plot}`")
    lines.extend([
        "",
        "## Reading",
        "",
        "- A practical rule should keep coverage high and stable false reject low.",
        "- A high collapsed reject rate with low coverage is diagnostic but may be too conservative for deployment.",
        "- If absorber-distance remains stable over periods, it supports a lightweight collapse-aware selective rejection mechanism.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/reject_option_ablation_tls22_summary")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    period_rows = []
    all_rows = []
    for input_dir in args.input_dirs:
        period, static, best_rows, full_rows = load_period(input_dir)
        for row in best_rows:
            period_rows.append(normalize_best_row(period, static, row))
        for row in full_rows:
            row = dict(row)
            row["period"] = period
            row["input_dir"] = input_dir
            all_rows.append(row)

    period_rows = sorted(period_rows, key=lambda r: (period_sort_key(r["period"]), r["rule"]))
    rule_summary = summarize_rules(period_rows)
    write_csv(os.path.join(args.output_dir, "reject_period_summary.csv"), period_rows)
    write_csv(os.path.join(args.output_dir, "reject_rule_summary.csv"), rule_summary)
    write_csv(os.path.join(args.output_dir, "reject_all_thresholds.csv"), all_rows)

    absorber_plot = plot_absorber_distance(period_rows, args.output_dir)
    tradeoff_plot = plot_rule_tradeoff(period_rows, args.output_dir)
    report_path = os.path.join(args.output_dir, "reject_option_period_report.md")
    write_report(report_path, period_rows, rule_summary, absorber_plot, tradeoff_plot)

    print("=== Reject-Option Multi-Period Summary ===")
    for row in period_rows:
        if row["rule"] == "absorber_distance":
            print(
                f"{row['period']} absorber_distance: "
                f"coverage={row['coverage']:.3f} "
                f"collapsed_reject={row['collapsed_reject_rate']:.3f} "
                f"stable_false={row['stable_false_reject_rate']:.3f} "
                f"absorber_reduction={row['absorber_error_reduction']:.3f}"
            )
    print(f"Saved summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()
