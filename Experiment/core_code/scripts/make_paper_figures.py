#!/usr/bin/env python
"""
Generate all paper figures from existing experimental results.

Usage:
    python scripts/make_paper_figures.py --output-dir Publication/figures

Generates:
  1. fig_collapse_timeline.pdf  - Per-class recall trajectories (M4-M12)
  2. fig_strategy_comparison.pdf - Selection strategy comparison
  3. fig_budget_sweep.pdf       - Label budget vs performance
  4. fig_ablation_bar.pdf       - Ablation component bar chart
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

COLLAPSE_CLASSES = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================
# Figure 1: Collapse timeline
# ============================================================
def fig_collapse_timeline(output_dir):
    timeline_path = "outputs/per_class_collapse_tls22_monthly/collapse_timeline.csv"
    if not os.path.exists(timeline_path):
        print(f"SKIP fig_collapse_timeline: {timeline_path} not found")
        print("  Run: bash scripts/run_tls22_collapse_diagnosis.sh")
        return

    rows = read_csv(timeline_path)
    periods = sorted(set(r["period"] for r in rows))
    period_labels = [p.replace("M-2022-", "M") for p in periods]

    fig, ax = plt.subplots(figsize=(8, 4))
    for cls in COLLAPSE_CLASSES:
        recalls = []
        for period in periods:
            match = [r for r in rows
                     if int(r["class_id"]) == cls and r["period"] == period]
            if match:
                recalls.append(float(match[0].get("recall", 0)))
            else:
                recalls.append(np.nan)
        ax.plot(range(len(periods)), recalls, marker="o", markersize=3,
                label=f"Class {cls}", alpha=0.8)

    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Collapse threshold")
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(period_labels, rotation=45)
    ax.set_xlabel("Test Period")
    ax.set_ylabel("Recall")
    ax.set_title("Recall Trajectories of Collapse Classes")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_collapse_timeline.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_collapse_timeline.pdf")


# ============================================================
# Figure 2: Strategy comparison
# ============================================================
def fig_strategy_comparison(output_dir):
    path = "outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2_distill0.5/results_by_budget.csv"
    if not os.path.exists(path):
        print(f"SKIP fig_strategy_comparison: {path} not found")
        return

    rows = read_csv(path)
    strategies = {}
    for r in rows:
        if r["method"] == "static":
            continue
        s = r["strategy"]
        b = int(r["budget"])
        if s not in strategies:
            strategies[s] = {"budgets": [], "macro_f1": [], "collapse_f1": []}
        strategies[s]["budgets"].append(b)
        strategies[s]["macro_f1"].append(float(r["overall_macro_f1"]))
        strategies[s]["collapse_f1"].append(float(r["bad_macro_f1"]))

    # Get static baseline
    static = [r for r in rows if r["method"] == "static"]
    static_macro = float(static[0]["overall_macro_f1"]) if static else 0.629
    static_collapse = float(static[0]["bad_macro_f1"]) if static else 0.028

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for s_name, data in strategies.items():
        idx = np.argsort(data["budgets"])
        budgets = np.array(data["budgets"])[idx]
        macro = np.array(data["macro_f1"])[idx]
        collapse = np.array(data["collapse_f1"])[idx]
        ax1.plot(budgets, macro, marker="o", label=s_name)
        ax2.plot(budgets, collapse, marker="o", label=s_name)

    for ax, baseline, ylabel, title in [
        (ax1, static_macro, "Overall Macro-F1", "Overall Performance"),
        (ax2, static_collapse, "Collapse-Class F1", "Collapse Class Recovery"),
    ]:
        ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.6,
                    label="Static baseline")
        ax.set_xlabel("Label Budget")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_strategy_comparison.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_strategy_comparison.pdf")


# ============================================================
# Figure 3: Budget sweep (best strategy)
# ============================================================
def fig_budget_sweep(output_dir):
    path = "outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2_distill0.5/results_by_budget.csv"
    if not os.path.exists(path):
        print(f"SKIP fig_budget_sweep: {path} not found")
        return

    rows = read_csv(path)
    static = [r for r in rows if r["method"] == "static"]
    static_macro = float(static[0]["overall_macro_f1"]) if static else 0.629

    # Use absorber_margin as primary, random as comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    for strategy, color, marker in [
        ("random", "#2196F3", "o"),
        ("absorber_margin", "#F44336", "s"),
        ("hybrid_risk", "#4CAF50", "^"),
    ]:
        data = [(int(r["budget"]), float(r["overall_macro_f1"]),
                 float(r["bad_macro_f1"]), int(r["collapsed_count"]))
                for r in rows if r["strategy"] == strategy]
        if not data:
            continue
        data.sort()
        budgets, macros, collapses, counts = zip(*data)
        ax.plot(budgets, macros, marker=marker, color=color,
                label=f"{strategy} (macro-F1)")

    ax.axhline(y=static_macro, color="gray", linestyle="--", alpha=0.6,
                label="Static baseline")
    ax.set_xlabel("Label Budget")
    ax.set_ylabel("Overall Macro-F1")
    ax.set_title("Label Efficiency: Budget vs. Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_budget_sweep.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_budget_sweep.pdf")


# ============================================================
# Figure 4: Ablation bar chart
# ============================================================
def fig_ablation(output_dir):
    configs = [
        ("Static", 0.629, 0.028, 0.903),
        ("FT only", 0.607, 0.077, 0.862),
        ("Replay only", 0.573, 0.076, 0.833),
        ("Distill only", 0.623, 0.068, 0.892),
        ("FT+Replay", 0.604, 0.116, 0.849),
        ("CARE (full)", 0.683, 0.426, 0.891),
    ]

    names = [c[0] for c in configs]
    macro = [c[1] for c in configs]
    collapse = [c[2] for c in configs]
    stable = [c[3] for c in configs]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width, macro, width, label="Overall Macro-F1",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, collapse, width, label="Collapse-Class F1",
                   color="#F44336", alpha=0.85)
    bars3 = ax.bar(x + width, stable, width, label="Stable-Class F1",
                   color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Ablation Study: Component Contribution")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")

    # Highlight full method
    for bar in [bars1[-1], bars2[-1], bars3[-1]]:
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_ablation_bar.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablation_bar.pdf")


# ============================================================
# Figure 5: Multi-period TTA failure
# ============================================================
def fig_tta_failure(output_dir):
    """Bar chart: collapse F1 across M7/M10/M12 for baselines vs CARE."""
    methods = ["Static", "Tent", "SAR", "CARE"]
    m7 =  [0.503, 0.505, 0.504, 0.559]
    m10 = [0.258, 0.256, 0.255, 0.298]
    m12 = [0.028, 0.026, 0.026, 0.426]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, m7, width, label="M-7", color="#81D4FA")
    ax.bar(x, m10, width, label="M-10", color="#42A5F5")
    ax.bar(x + width, m12, width, label="M-12", color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Collapse-Class F1")
    ax.set_title("Collapse-Class F1 Across Time Points")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_tta_failure.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_tta_failure.pdf")


# ============================================================
# Figure 6: Per-class recovery waterfall
# ============================================================
def fig_per_class_recovery(output_dir):
    classes = [48, 56, 47, 104, 10, 38, 163, 66, 174, 26, 69, 109]
    before = [0.001, 0.000, 0.000, 0.010, 0.098, 0.052, 0.015, 0.001, 0.000, 0.019, 0.005, 0.000]
    after =  [0.913, 0.733, 0.699, 0.624, 0.623, 0.534, 0.455, 0.355, 0.274, 0.051, 0.000, 0.000]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, before, width, label="Before (Static)",
           color="#FFCDD2", edgecolor="#E53935", linewidth=0.8)
    ax.bar(x + width/2, after, width, label="After (CARE)",
           color="#C8E6C9", edgecolor="#43A047", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], rotation=45)
    ax.set_xlabel("Collapse Class ID")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recovery: Before vs. After CARE")
    ax.legend()
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.4,
               label="Collapse threshold")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_per_class_recovery.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_per_class_recovery.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="Publication/figures")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fig_collapse_timeline(args.output_dir)
    fig_strategy_comparison(args.output_dir)
    fig_budget_sweep(args.output_dir)
    fig_ablation(args.output_dir)
    fig_tta_failure(args.output_dir)
    fig_per_class_recovery(args.output_dir)


if __name__ == "__main__":
    main()
