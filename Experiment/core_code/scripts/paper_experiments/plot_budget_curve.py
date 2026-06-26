"""Plot label budget curve from seed results or aggregated CSV."""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_from_seeds(base, num_seeds=5):
    """Read per-seed results_by_budget.csv and aggregate."""
    by_budget = defaultdict(lambda: {"macro": [], "col": [], "stable": [], "collapsed": []})
    for seed in range(num_seeds):
        path = os.path.join(base, f"seed_{seed}", "results_by_budget.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                b = int(row["budget"])
                by_budget[b]["macro"].append(float(row["strict_overall_macro_f1"]))
                by_budget[b]["col"].append(float(row["strict_bad_macro_f1"]))
                by_budget[b]["stable"].append(float(row["strict_stable_macro_f1"]))
                by_budget[b]["collapsed"].append(float(row["strict_collapsed_count"]))
    return by_budget


def load_from_aggregated(agg_path):
    """Read aggregated_mean_std.csv."""
    by_budget = defaultdict(lambda: {"macro": [], "col": [], "stable": [], "collapsed": []})
    with open(agg_path) as f:
        for row in csv.DictReader(f):
            b = int(row["budget"])
            n = int(row["n_seeds"])
            by_budget[b]["macro"] = [float(row["strict_overall_macro_f1_mean"])] * n
            by_budget[b]["col"] = [float(row["strict_bad_macro_f1_mean"])] * n
            by_budget[b]["stable"] = [float(row["strict_stable_macro_f1_mean"])] * n
            by_budget[b]["collapsed"] = [float(row["strict_collapsed_count_mean"])] * n
            by_budget[b]["macro_std"] = float(row["strict_overall_macro_f1_std"])
            by_budget[b]["col_std"] = float(row["strict_bad_macro_f1_std"])
            by_budget[b]["stable_std"] = float(row["strict_stable_macro_f1_std"])
            by_budget[b]["collapsed_std"] = float(row["strict_collapsed_count_std"])
    return by_budget


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..",
                        "outputs", "paper_experiments", "budget_curve")
    agg_path = os.path.join(base, "aggregated_mean_std.csv")

    if os.path.exists(agg_path):
        by_budget = load_from_aggregated(agg_path)
    else:
        by_budget = load_from_seeds(base)

    budgets_sorted = sorted(by_budget.keys())
    budgets = np.array(budgets_sorted)

    macro_m, macro_s = [], []
    col_m, col_s = [], []
    stable_m, stable_s = [], []
    collapsed_m, collapsed_s = [], []

    for b in budgets_sorted:
        d = by_budget[b]
        macro_m.append(np.mean(d["macro"]))
        macro_s.append(d.get("macro_std", np.std(d["macro"], ddof=1) if len(d["macro"]) > 1 else 0))
        col_m.append(np.mean(d["col"]))
        col_s.append(d.get("col_std", np.std(d["col"], ddof=1) if len(d["col"]) > 1 else 0))
        stable_m.append(np.mean(d["stable"]))
        stable_s.append(d.get("stable_std", np.std(d["stable"], ddof=1) if len(d["stable"]) > 1 else 0))
        collapsed_m.append(np.mean(d["collapsed"]))
        collapsed_s.append(d.get("collapsed_std", np.std(d["collapsed"], ddof=1) if len(d["collapsed"]) > 1 else 0))

    macro_m, macro_s = np.array(macro_m), np.array(macro_s)
    col_m, col_s = np.array(col_m), np.array(col_s)
    stable_m, stable_s = np.array(stable_m), np.array(stable_s)
    collapsed_m, collapsed_s = np.array(collapsed_m), np.array(collapsed_s)

    x_pos = np.arange(len(budgets))
    rec_idx = list(budgets).index(2000) if 2000 in budgets else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: F1 curves
    ax1.errorbar(x_pos, macro_m, yerr=macro_s, marker='o', capsize=3,
                 label='Overall Macro F1', linewidth=2, markersize=6)
    ax1.errorbar(x_pos, col_m, yerr=col_s, marker='s', capsize=3,
                 label='Collapse F1', linewidth=2, markersize=6)
    ax1.errorbar(x_pos, stable_m, yerr=stable_s, marker='^', capsize=3,
                 label='Stable F1', linewidth=2, markersize=6)
    ax1.axhline(y=macro_m[0], color='gray', linestyle='--', alpha=0.5, linewidth=1)
    if rec_idx is not None:
        ax1.axvline(x=rec_idx, color='red', linestyle=':', alpha=0.4, linewidth=1.5,
                    label='Recommended B=2000')
    ax1.set_xlabel('Label Budget (B)', fontsize=12)
    ax1.set_ylabel('Macro F1', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(b) for b in budgets], fontsize=9)
    ax1.legend(fontsize=9, loc='center right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_title('(a) F1 vs. Label Budget', fontsize=12)

    # Right: Collapsed class count
    ax2.errorbar(x_pos, collapsed_m, yerr=collapsed_s, marker='D', capsize=3,
                 color='#d62728', linewidth=2, markersize=6)
    if rec_idx is not None:
        ax2.axvline(x=rec_idx, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
    ax2.set_xlabel('Label Budget (B)', fontsize=12)
    ax2.set_ylabel('# Collapsed Classes (recall < 0.1)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(b) for b in budgets], fontsize=9)
    ax2.set_ylim(-0.5, 13)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('(b) Collapsed Classes vs. Label Budget', fontsize=12)

    plt.tight_layout()
    out_dir = base
    pdf_path = os.path.join(out_dir, "budget_curve.pdf")
    png_path = os.path.join(out_dir, "budget_curve.png")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
