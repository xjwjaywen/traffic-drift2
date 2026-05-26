"""
Summarize multi-period collapse-aware active replay experiments.

Usage from Experiment/core_code/:
    python scripts/summarize_active_replay_multiperiod.py \
      --input-dir M-2022-7:outputs/collapse_active_replay_tls22_M-2022-7_all_r5_tr2 \
      --input-dir M-2022-10:outputs/collapse_active_replay_tls22_M-2022-10_all_r5_tr2 \
      --input-dir M-2022-12:outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2 \
      --output-dir outputs/collapse_active_replay_tls22_summary_all_r5_tr2
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


def fmt(value):
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def parse_input_dir(spec):
    if ":" not in spec:
        raise ValueError(f"Expected PERIOD:PATH, got {spec}")
    period, path = spec.split(":", 1)
    return period, path


def period_key(period):
    try:
        year_part, month_part = period.replace("M-", "").split("-")
        return int(year_part), int(month_part)
    except Exception:
        return 9999, 9999, period


def load_period_rows(period, input_dir):
    rows = read_csv(os.path.join(input_dir, "results_by_budget.csv"))
    static = next(row for row in rows if row.get("method") == "static")
    loaded = []
    for row in rows:
        out = dict(row)
        out["period"] = period
        out["input_dir"] = input_dir
        out["delta_macro_f1_vs_static"] = (
            as_float(row.get("overall_macro_f1")) - as_float(static.get("overall_macro_f1"))
        )
        out["delta_collapse_f1_vs_static"] = (
            as_float(row.get("bad_macro_f1")) - as_float(static.get("bad_macro_f1"))
        )
        out["delta_stable_f1_vs_static"] = (
            as_float(row.get("stable_macro_f1")) - as_float(static.get("stable_macro_f1"))
        )
        loaded.append(out)
    return loaded


def best_by_period(rows):
    best = {}
    for row in rows:
        if row.get("method") == "static":
            continue
        period = row.get("period")
        key = (
            as_float(row.get("bad_macro_f1"), -1.0),
            as_float(row.get("overall_macro_f1"), -1.0),
        )
        if period not in best or key > best[period][0]:
            best[period] = (key, row)
    return [best[period][1] for period in sorted(best, key=period_key)]


def selected_budget_rows(rows, budget):
    return [
        row for row in rows
        if row.get("method") != "static" and as_int(row.get("budget")) == budget
    ]


def plot_period_budget(rows, output_dir, metric, ylabel, filename):
    plt.figure(figsize=(9.5, 5.8))
    strategies = sorted({
        row.get("strategy")
        for row in rows
        if row.get("method") != "static" and row.get("strategy")
    })
    periods = sorted({row.get("period") for row in rows}, key=period_key)
    for strategy in strategies:
        ys = []
        xs = []
        for period in periods:
            sub = [
                row for row in rows
                if row.get("period") == period
                and row.get("strategy") == strategy
                and row.get("method") != "static"
            ]
            if not sub:
                continue
            best = max(
                sub,
                key=lambda r: (
                    as_float(r.get(metric), -1.0),
                    as_float(r.get("overall_macro_f1"), -1.0),
                ),
            )
            xs.append(period)
            ys.append(as_float(best.get(metric), 0.0))
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=strategy)
    static_by_period = []
    static_periods = []
    for period in periods:
        static = next(
            (row for row in rows if row.get("period") == period and row.get("method") == "static"),
            None,
        )
        if static is not None:
            static_periods.append(period)
            static_by_period.append(as_float(static.get(metric), 0.0))
    if static_by_period:
        plt.plot(
            static_periods,
            static_by_period,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="static",
        )
    plt.xlabel("Target period")
    plt.ylabel(ylabel)
    plt.title(f"Active replay best-over-budget {ylabel}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_budget_tradeoff(rows, output_dir, period):
    sub = [
        row for row in rows
        if row.get("period") == period and row.get("method") != "static"
    ]
    if not sub:
        return ""
    plt.figure(figsize=(8.5, 5.8))
    strategies = sorted({row.get("strategy") for row in sub})
    for strategy in strategies:
        srows = sorted(
            [row for row in sub if row.get("strategy") == strategy],
            key=lambda r: as_int(r.get("budget")),
        )
        x = [as_float(row.get("stable_macro_f1"), 0.0) for row in srows]
        y = [as_float(row.get("bad_macro_f1"), 0.0) for row in srows]
        labels = [str(as_int(row.get("budget"))) for row in srows]
        plt.plot(x, y, marker="o", linewidth=1.6, label=strategy)
        for xi, yi, label in zip(x, y, labels):
            plt.annotate(label, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    static = next(
        (row for row in rows if row.get("period") == period and row.get("method") == "static"),
        None,
    )
    if static is not None:
        plt.scatter(
            [as_float(static.get("stable_macro_f1"), 0.0)],
            [as_float(static.get("bad_macro_f1"), 0.0)],
            color="black",
            marker="x",
            s=70,
            label="static",
        )
    plt.xlabel("Stable-class macro-F1")
    plt.ylabel("Collapsed-class macro-F1")
    plt.title(f"Active replay trade-off ({period})")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = os.path.join(output_dir, f"active_replay_tradeoff_{period}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def write_report(path, rows, best_rows, figures):
    lines = [
        "# Active Replay Multi-Period Summary",
        "",
        "## Best Run Per Period",
        "",
        "| period | strategy | budget | macro-F1 | collapse F1 | stable F1 | Δmacro | Δcollapse | Δstable | selected collapsed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row.get('period')} | {row.get('strategy')} | {as_int(row.get('budget'))} | "
            f"{fmt(row.get('overall_macro_f1'))} | {fmt(row.get('bad_macro_f1'))} | "
            f"{fmt(row.get('stable_macro_f1'))} | {fmt(row.get('delta_macro_f1_vs_static'))} | "
            f"{fmt(row.get('delta_collapse_f1_vs_static'))} | {fmt(row.get('delta_stable_f1_vs_static'))} | "
            f"{as_int(row.get('selected_collapse_labels'))} |"
        )
    lines.extend([
        "",
        "## Static Baselines",
        "",
        "| period | macro-F1 | collapse F1 | stable F1 | collapsed count |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row.get("method") != "static":
            continue
        lines.append(
            f"| {row.get('period')} | {fmt(row.get('overall_macro_f1'))} | "
            f"{fmt(row.get('bad_macro_f1'))} | {fmt(row.get('stable_macro_f1'))} | "
            f"{as_int(row.get('collapsed_count'))} |"
        )
    lines.extend(["", "## Figures", ""])
    for label, figure in figures:
        if figure:
            lines.append(f"- {label}: `{figure}`")
    lines.extend([
        "",
        "## Reading",
        "",
        "- The key question is whether the M-2022-12 signal repeats on M-2022-7 and M-2022-10.",
        "- A method-oriented result should improve collapsed-class F1 while keeping stable-class F1 close to the static baseline.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for spec in args.input_dir:
        period, input_dir = parse_input_dir(spec)
        rows.extend(load_period_rows(period, input_dir))

    write_csv(os.path.join(args.output_dir, "active_replay_period_rows.csv"), rows)
    best_rows = best_by_period(rows)
    write_csv(os.path.join(args.output_dir, "active_replay_best_by_period.csv"), best_rows)

    collapse_fig = plot_period_budget(
        rows,
        args.output_dir,
        "bad_macro_f1",
        "Collapsed-class macro-F1",
        "active_replay_collapse_f1_by_period.png",
    )
    macro_fig = plot_period_budget(
        rows,
        args.output_dir,
        "overall_macro_f1",
        "Overall macro-F1",
        "active_replay_macro_f1_by_period.png",
    )
    tradeoff_figs = [
        (f"Trade-off {period}", plot_budget_tradeoff(rows, args.output_dir, period))
        for period in sorted({row.get("period") for row in rows}, key=period_key)
    ]
    figures = [
        ("Collapsed F1 by period", collapse_fig),
        ("Macro-F1 by period", macro_fig),
        *tradeoff_figs,
    ]
    report_path = os.path.join(args.output_dir, "active_replay_multiperiod_report.md")
    write_report(report_path, rows, best_rows, figures)

    print("=== Active Replay Multi-Period Summary ===")
    for row in best_rows:
        print(
            f"{row.get('period')}: {row.get('strategy')} budget={as_int(row.get('budget'))} "
            f"macro={fmt(row.get('overall_macro_f1'))} "
            f"collapse={fmt(row.get('bad_macro_f1'))} "
            f"stable={fmt(row.get('stable_macro_f1'))} "
            f"d_macro={fmt(row.get('delta_macro_f1_vs_static'))}"
        )
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
