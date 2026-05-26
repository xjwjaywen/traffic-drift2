"""
Summarize collapse-aware active maintenance results.

Usage from Experiment/core_code/:
    python scripts/summarize_active_maintenance_results.py \
      --input-dir outputs/collapse_active_maintenance_tls22_M-2022-12
"""
import argparse
import csv
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


FOCUS_STRATEGIES = [
    "random",
    "entropy",
    "margin",
    "absorber_random",
    "absorber_margin",
    "absorber_distance",
    "absorber_proto_disagree",
    "hybrid_risk",
    "oracle_collapse_random",
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


def as_int(value, default=0):
    if value in (None, ""):
        return default
    return int(float(value))


def best_by_budget(rows):
    best = {}
    for row in rows:
        if row.get("method") == "static":
            continue
        budget = as_int(row.get("budget"))
        key = (
            as_float(row.get("bad_macro_f1"), -1.0),
            as_float(row.get("overall_macro_f1"), -1.0),
            -as_float(row.get("stable_macro_f1"), 0.0),
        )
        if budget not in best or key > best[budget][0]:
            best[budget] = (key, row)
    return [best[budget][1] for budget in sorted(best)]


def best_by_strategy(rows):
    best = {}
    for row in rows:
        if row.get("method") == "static":
            continue
        strategy = row.get("strategy", "")
        key = (
            as_float(row.get("bad_macro_f1"), -1.0),
            as_float(row.get("overall_macro_f1"), -1.0),
        )
        if strategy not in best or key > best[strategy][0]:
            best[strategy] = (key, row)
    return [best[strategy][1] for strategy in sorted(best)]


def plot_budget_curve(rows, output_dir):
    plt.figure(figsize=(9.5, 5.8))
    strategies = [s for s in FOCUS_STRATEGIES if any(row.get("strategy") == s for row in rows)]
    for strategy in strategies:
        sub = [
            row for row in rows
            if row.get("strategy") == strategy and row.get("method") != "static"
        ]
        sub = sorted(sub, key=lambda r: as_int(r.get("budget")))
        if not sub:
            continue
        x = [as_int(row.get("budget")) for row in sub]
        y = [as_float(row.get("bad_macro_f1"), 0.0) for row in sub]
        plt.plot(x, y, marker="o", linewidth=1.7, label=strategy)
    static = next((row for row in rows if row.get("method") == "static"), None)
    if static:
        plt.axhline(
            as_float(static.get("bad_macro_f1"), 0.0),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="static",
        )
    plt.xlabel("Label budget")
    plt.ylabel("Collapsed-class macro-F1")
    plt.title("Active maintenance: collapsed-class F1 vs label budget")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = os.path.join(output_dir, "active_maintenance_budget_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_selection_curve(rows, output_dir):
    plt.figure(figsize=(9.5, 5.8))
    strategies = [s for s in FOCUS_STRATEGIES if any(row.get("strategy") == s for row in rows)]
    for strategy in strategies:
        sub = [
            row for row in rows
            if row.get("strategy") == strategy and row.get("method") != "static"
        ]
        sub = sorted(sub, key=lambda r: as_int(r.get("budget")))
        if not sub:
            continue
        x = [as_int(row.get("budget")) for row in sub]
        y = [as_int(row.get("selected_collapse_labels")) for row in sub]
        plt.plot(x, y, marker="o", linewidth=1.7, label=strategy)
    plt.xlabel("Label budget")
    plt.ylabel("Selected true collapsed samples")
    plt.title("Active maintenance: selected collapsed samples")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = os.path.join(output_dir, "active_maintenance_selected_collapse.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def fmt(value):
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def write_report(path, rows, best_budget_rows, best_strategy_rows, budget_plot, selection_plot):
    static = next((row for row in rows if row.get("method") == "static"), None)
    lines = [
        "# Collapse-Aware Active Maintenance Summary",
        "",
    ]
    replay_mode = next((row.get("replay_mode") for row in rows if row.get("replay_mode")), "")
    replay_samples = next((row.get("replay_samples") for row in rows if row.get("replay_samples")), "")
    target_repeat = next((row.get("target_repeat") for row in rows if row.get("target_repeat")), "")
    if replay_mode:
        lines.extend([
            "## Replay Setting",
            "",
            f"- replay mode: `{replay_mode}`",
            f"- replay samples: `{as_int(replay_samples)}`",
            f"- target repeat: `{as_int(target_repeat, 1)}`",
            "",
        ])
    if static:
        lines.extend([
            "## Static Baseline",
            "",
            f"- macro-F1: `{fmt(static.get('overall_macro_f1'))}`",
            f"- collapsed-class macro-F1: `{fmt(static.get('bad_macro_f1'))}`",
            f"- stable-class macro-F1: `{fmt(static.get('stable_macro_f1'))}`",
            "",
        ])
    lines.extend([
        "## Best Strategy Per Budget",
        "",
        "| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in best_budget_rows:
        lines.append(
            f"| {as_int(row.get('budget'))} | {row.get('strategy')} | "
            f"{fmt(row.get('overall_macro_f1'))} | {fmt(row.get('bad_macro_f1'))} | "
            f"{fmt(row.get('stable_macro_f1'))} | {as_int(row.get('selected_collapse_labels'))} | "
            f"{as_int(row.get('selected_absorber_preds'))} |"
        )
    lines.extend([
        "",
        "## Best Run Per Strategy",
        "",
        "| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in best_strategy_rows:
        lines.append(
            f"| {row.get('strategy')} | {as_int(row.get('budget'))} | "
            f"{fmt(row.get('overall_macro_f1'))} | {fmt(row.get('bad_macro_f1'))} | "
            f"{fmt(row.get('stable_macro_f1'))} | {as_int(row.get('selected_collapse_labels'))} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        f"- Budget curve: `{budget_plot}`",
        f"- Selected collapsed samples: `{selection_plot}`",
        "",
        "## Reading",
        "",
        "- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.",
        "- Oracle-collapse sampling is an upper bound because it uses target labels for selection.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    rows = read_csv(os.path.join(args.input_dir, "results_by_budget.csv"))
    best_budget_rows = best_by_budget(rows)
    best_strategy_rows = best_by_strategy(rows)
    write_csv(os.path.join(args.input_dir, "best_by_budget.csv"), best_budget_rows)
    write_csv(os.path.join(args.input_dir, "best_by_strategy.csv"), best_strategy_rows)
    budget_plot = plot_budget_curve(rows, args.input_dir)
    selection_plot = plot_selection_curve(rows, args.input_dir)
    report_path = os.path.join(args.input_dir, "active_maintenance_report.md")
    write_report(report_path, rows, best_budget_rows, best_strategy_rows, budget_plot, selection_plot)

    print("=== Active Maintenance Summary ===")
    for row in best_budget_rows:
        print(
            f"budget={as_int(row.get('budget')):<5d} "
            f"best={row.get('strategy'):<24} "
            f"macro={fmt(row.get('overall_macro_f1'))} "
            f"collapse={fmt(row.get('bad_macro_f1'))} "
            f"stable={fmt(row.get('stable_macro_f1'))} "
            f"sel_collapse={as_int(row.get('selected_collapse_labels'))}"
        )
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
