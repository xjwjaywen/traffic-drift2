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
    periods = sorted(set(r["period"] for r in rows),
                     key=lambda p: int(p.split("-")[-1]))
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
    """AL strategy comparison using strict 5-seed aggregated data."""
    path = "outputs/al_baselines_strict/aggregated_mean_std.csv"
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
            strategies[s] = {"budgets": [], "macro_f1": [], "collapse_f1": [],
                             "macro_std": [], "collapse_std": []}
        strategies[s]["budgets"].append(b)
        strategies[s]["macro_f1"].append(float(r["strict_overall_macro_f1_mean"]))
        strategies[s]["collapse_f1"].append(float(r["strict_bad_macro_f1_mean"]))
        strategies[s]["macro_std"].append(float(r.get("strict_overall_macro_f1_std", 0)))
        strategies[s]["collapse_std"].append(float(r.get("strict_bad_macro_f1_std", 0)))

    static_macro = 0.629
    static_collapse = 0.028

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = {"random": "#2196F3", "entropy": "#FF9800", "margin": "#F44336",
              "coreset": "#9C27B0", "badge": "#4CAF50"}
    markers = {"random": "o", "entropy": "v", "margin": "s",
               "coreset": "^", "badge": "D"}
    # Override BADGE with dedicated 5-seed run (al_baselines_strict has only 2 seeds)
    badge_path = "outputs/badge_5seeds_strict/aggregated_mean_std.csv"
    if os.path.exists(badge_path):
        strategies.pop("badge", None)
        badge_rows = read_csv(badge_path)
        for r in badge_rows:
            if r["method"] == "static" or r["strategy"] != "badge":
                continue
            s = "badge"
            b = int(r["budget"])
            if s not in strategies:
                strategies[s] = {"budgets": [], "macro_f1": [], "collapse_f1": [],
                                 "macro_std": [], "collapse_std": []}
            strategies[s]["budgets"].append(b)
            strategies[s]["macro_f1"].append(float(r["strict_overall_macro_f1_mean"]))
            strategies[s]["collapse_f1"].append(float(r["strict_bad_macro_f1_mean"]))
            strategies[s]["macro_std"].append(float(r.get("strict_overall_macro_f1_std", 0)))
            strategies[s]["collapse_std"].append(float(r.get("strict_bad_macro_f1_std", 0)))

    for s_name in ["entropy", "coreset", "random", "badge", "margin"]:
        if s_name not in strategies:
            continue
        data = strategies[s_name]
        idx = np.argsort(data["budgets"])
        budgets = np.array(data["budgets"])[idx]
        macro = np.array(data["macro_f1"])[idx]
        collapse = np.array(data["collapse_f1"])[idx]
        color = colors.get(s_name, "gray")
        mk = markers.get(s_name, "o")
        lw = 2.0 if s_name == "margin" else 1.2
        ax1.plot(budgets, macro, marker=mk, label=s_name, color=color, linewidth=lw)
        ax2.plot(budgets, collapse, marker=mk, label=s_name, color=color, linewidth=lw)

    for ax, baseline, ylabel, title in [
        (ax1, static_macro, "Overall Macro-F1", "Overall Performance"),
        (ax2, static_collapse, "Collapse-Class F1", "Collapse Class Recovery"),
    ]:
        ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.6,
                    label="Static")
        ax.set_xlabel("Label Budget")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
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
    """Budget sweep using strict 5-seed aggregated data."""
    path = "outputs/care_5seeds_strict_cnn/aggregated_mean_std.csv"
    if not os.path.exists(path):
        path = "outputs/al_baselines_strict/aggregated_mean_std.csv"
    if not os.path.exists(path):
        print(f"SKIP fig_budget_sweep: no aggregated data found")
        return

    rows = read_csv(path)
    static_macro = 0.629

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for strategy, color, marker in [
        ("random", "#2196F3", "o"),
        ("margin", "#F44336", "s"),
    ]:
        data = [(int(r["budget"]),
                 float(r["strict_overall_macro_f1_mean"]),
                 float(r["strict_bad_macro_f1_mean"]),
                 float(r.get("strict_overall_macro_f1_std", 0)),
                 float(r.get("strict_bad_macro_f1_std", 0)))
                for r in rows if r["strategy"] == strategy and r["method"] != "static"]
        if not data:
            continue
        data.sort()
        budgets, macros, collapses, m_std, c_std = zip(*data)
        ax1.errorbar(budgets, macros, yerr=m_std, marker=marker, color=color,
                     label=strategy, capsize=3, linewidth=1.5)
        ax2.errorbar(budgets, collapses, yerr=c_std, marker=marker, color=color,
                     label=strategy, capsize=3, linewidth=1.5)

    ax1.axhline(y=static_macro, color="gray", linestyle="--", alpha=0.6, label="Static")
    ax2.axhline(y=0.028, color="gray", linestyle="--", alpha=0.6, label="Static")
    for ax, ylabel, title in [
        (ax1, "Overall Macro-F1", "Label Efficiency: Macro-F1"),
        (ax2, "Collapse-Class F1", "Label Efficiency: Collapse Recovery"),
    ]:
        ax.set_xlabel("Label Budget")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
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
    # Strict evaluation, 5-seed mean±std, margin@1000
    configs = [
        #  (name,       macro, collapse, stable, m_std, c_std, s_std)
        ("Static",     0.629, 0.028, 0.903, 0, 0, 0),
        ("FT only",    0.638, 0.134, 0.838, 0.001, 0.001, 0.001),
        ("FT+Replay",  0.620, 0.224, 0.840, 0.001, 0.002, 0.004),
        ("FT+Distill", 0.670, 0.152, 0.887, 0.000, 0.007, 0.003),
        ("CARE (full)", 0.673, 0.233, 0.885, 0.001, 0.002, 0.004),
    ]

    names = [c[0] for c in configs]
    macro = [c[1] for c in configs]
    collapse = [c[2] for c in configs]
    stable = [c[3] for c in configs]
    m_std = [c[4] for c in configs]
    c_std = [c[5] for c in configs]
    s_std = [c[6] for c in configs]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width, macro, width, yerr=m_std, capsize=2,
                   label="Overall Macro-F1", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, collapse, width, yerr=c_std, capsize=2,
                   label="Collapse-Class F1", color="#F44336", alpha=0.85)
    bars3 = ax.bar(x + width, stable, width, yerr=s_std, capsize=2,
                   label="Stable-Class F1", color="#4CAF50", alpha=0.85)

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
    """Line chart: macro-F1 across M7/M9/M11/M12 for Static vs TTA vs CARE."""
    periods = ["M7", "M9", "M11", "M12"]
    # Static and TTA macro-F1 (all TTA ≈ Static)
    static_macro = [0.740, 0.691, 0.649, 0.629]
    tent_macro =   [0.742, 0.691, 0.649, 0.631]
    sar_macro =    [0.740, 0.690, 0.649, 0.629]
    # CARE strict multi-period (margin@1000, auto-discovery, 3 seeds)
    care_macro =   [0.750, 0.722, 0.689, 0.674]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(periods))

    # Left: Macro-F1
    ax1.plot(x, static_macro, "o--", color="gray", label="Static", alpha=0.8)
    ax1.plot(x, tent_macro, "s--", color="#90CAF9", label="Tent", alpha=0.6)
    ax1.plot(x, sar_macro, "^--", color="#CE93D8", label="SAR", alpha=0.6)
    ax1.plot(x, care_macro, "D-", color="#F44336", label="CARE", linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods)
    ax1.set_ylabel("Overall Macro-F1")
    ax1.set_title("Overall Performance")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.55, 0.80)

    # Right: Collapse-class F1
    static_coll = [0.023, 0.055, 0.017, 0.028]
    tent_coll =   [0.025, 0.053, 0.017, 0.026]
    sar_coll =    [0.024, 0.054, 0.017, 0.026]
    care_coll =   [0.456, 0.159, 0.277, 0.238]

    ax2.plot(x, static_coll, "o--", color="gray", label="Static", alpha=0.8)
    ax2.plot(x, tent_coll, "s--", color="#90CAF9", label="Tent", alpha=0.6)
    ax2.plot(x, sar_coll, "^--", color="#CE93D8", label="SAR", alpha=0.6)
    ax2.plot(x, care_coll, "D-", color="#F44336", label="CARE", linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods)
    ax2.set_ylabel("Collapse-Class F1")
    ax2.set_title("Collapse Class Recovery")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 0.55)

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
