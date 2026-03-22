"""
Result visualization for TTA-TC experiments.

Generates publication-quality figures:
  1. ARR curves over time (QUIC22 + TLS22)
  2. Method comparison bar chart (W-45 accuracy)
  3. Ablation bar charts (SSL tasks, mask ratio, anti-forgetting)
  4. Per-class recall decay heatmap (from sequential results)

Usage (from Experiment/core_code/):
    python ../analysis/visualize_results.py --outputs-dir outputs/ \
        --save-dir ../analysis/figures/
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #

PALETTE = {
    "static":   "#d62728",
    "bn_adapt": "#ff7f0e",
    "tent":     "#9467bd",
    "eata":     "#8c564b",
    "cotta":    "#e377c2",
    "sar":      "#7f7f7f",
    "note":     "#bcbd22",
    "tta_tc":   "#2ca02c",
}
METHOD_LABELS = {
    "static":   "Static",
    "bn_adapt": "BN-Adapt",
    "tent":     "Tent",
    "eata":     "EATA",
    "cotta":    "CoTTA",
    "sar":      "SAR",
    "note":     "NOTE",
    "tta_tc":   "TTA-TC (Ours)",
}
LINESTYLES = {
    "static":   "--",
    "bn_adapt": "-.",
    "tent":     ":",
    "eata":     ":",
    "cotta":    "-.",
    "sar":      ":",
    "note":     "-.",
    "tta_tc":   "-",
}
MARKERS = {
    "static": "x", "bn_adapt": "s", "tent": "^", "eata": "v",
    "cotta": "D", "sar": "P", "note": "h", "tta_tc": "o",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

# --------------------------------------------------------------------------- #
#  Helper
# --------------------------------------------------------------------------- #

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _methods_in_results(results):
    order = ["static", "bn_adapt", "tent", "eata", "cotta", "sar", "note", "tta_tc"]
    return [m for m in order if m in results]


# --------------------------------------------------------------------------- #
#  Figure 1: ARR curves
# --------------------------------------------------------------------------- #

def plot_arr_curves(seq_results, save_path, title="Sequential ARR — CESNET-QUIC22"):
    """Plot per-method ARR over test periods."""
    methods = _methods_in_results(seq_results)
    if not methods:
        print(f"  [skip] No sequential data for {title}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for method in methods:
        data = seq_results[method]
        periods = data.get("periods", {})
        sorted_periods = sorted(periods.keys())
        arrs = [periods[p]["arr"] for p in sorted_periods if periods[p].get("arr") is not None]
        if not arrs:
            continue
        x = range(1, len(arrs) + 1)
        ax.plot(
            x, arrs,
            label=METHOD_LABELS.get(method, method),
            color=PALETTE.get(method, "gray"),
            linestyle=LINESTYLES.get(method, "-"),
            marker=MARKERS.get(method, "o"),
            markersize=5,
            linewidth=2 if method == "tta_tc" else 1.2,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Test Period (weeks after training)")
    ax.set_ylabel("Accuracy Retention Ratio (ARR)")
    ax.set_title(title)
    ax.set_ylim(0.5, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
#  Figure 2: Bar chart — W-45 accuracy
# --------------------------------------------------------------------------- #

def plot_comparison_bar(single_results, save_path,
                        title="Accuracy on CESNET-QUIC22 W-2022-45"):
    """Grouped bar chart comparing methods on single test period."""
    methods = _methods_in_results(single_results)
    if not methods:
        print(f"  [skip] No single-period data")
        return

    accs = [single_results[m].get("accuracy", 0.0) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [PALETTE.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(accs) * 1.15) if accs else 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
#  Figure 3: Ablation bar charts
# --------------------------------------------------------------------------- #

def plot_ablation_bars(ablation_results, ablation_name, save_path,
                       metric="accuracy", title=None):
    """Bar chart for one ablation study."""
    if not ablation_results:
        print(f"  [skip] No ablation data for {ablation_name}")
        return

    settings = list(ablation_results.keys())
    values = [ablation_results[s].get(metric, 0.0) for s in settings]
    title = title or f"Ablation: {ablation_name} ({metric})"

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("muted", len(settings))
    bars = ax.bar(settings, values, color=colors, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
#  Figure 4: AURC summary bar
# --------------------------------------------------------------------------- #

def plot_aurc_bar(seq_results, save_path, title="AURC — Sequential Evaluation"):
    """Bar chart of AURC values per method."""
    methods = _methods_in_results(seq_results)
    aurcs = []
    valid_methods = []
    for m in methods:
        aurc = seq_results[m].get("aurc")
        if aurc is not None:
            aurcs.append(aurc)
            valid_methods.append(m)
    if not valid_methods:
        print(f"  [skip] No AURC data")
        return

    labels = [METHOD_LABELS.get(m, m) for m in valid_methods]
    colors = [PALETTE.get(m, "gray") for m in valid_methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, aurcs, color=colors, edgecolor="white")
    for bar, v in zip(bars, aurcs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylabel("AURC")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
#  Figure 5: TLS-Year22 long-term ARR curve (highlight)
# --------------------------------------------------------------------------- #

def plot_long_term_arr(seq_tls22, save_path):
    """Highlight TTA-TC vs Static on the 9-month TLS-Year22 evaluation."""
    highlight = ["static", "tta_tc"]
    present = [m for m in highlight if m in seq_tls22]
    if not present:
        print("  [skip] No TLS-Year22 data")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for method in present:
        periods = seq_tls22[method].get("periods", {})
        sorted_p = sorted(periods.keys())
        arrs = [periods[p].get("arr") for p in sorted_p if periods[p].get("arr") is not None]
        if not arrs:
            continue
        x = range(1, len(arrs) + 1)
        ax.plot(
            x, arrs,
            label=METHOD_LABELS.get(method, method),
            color=PALETTE.get(method, "gray"),
            linestyle=LINESTYLES.get(method, "-"),
            marker=MARKERS.get(method, "o"),
            markersize=6,
            linewidth=2,
        )

    ax.axhline(0.90, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7,
               label="H5 target (ARR=0.90)")
    ax.set_xlabel("Months after training (M-4 → M-12)")
    ax.set_ylabel("Accuracy Retention Ratio (ARR)")
    ax.set_title("Long-Term Evaluation — CESNET-TLS-Year22 (9 months)")
    ax.set_ylim(0.3, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="TTA-TC result visualization")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Root outputs/ directory from experiments")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save figures (default: analysis/figures/)")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    os.makedirs(save_dir, exist_ok=True)
    out = args.outputs_dir

    def _try_load(path):
        if os.path.exists(path):
            return _load_json(path)
        print(f"  [not found] {path}")
        return {}

    # Load results
    single = _try_load(os.path.join(out, "eval_quic22_single", "results_single.json"))
    seq_quic22 = _try_load(os.path.join(out, "eval_quic22_sequential", "results_sequential.json"))
    seq_tls22  = _try_load(os.path.join(out, "eval_tls22_sequential", "results_sequential.json"))

    ablation_names = ["ssl_tasks", "mask_ratio", "adapt_depth", "anti_forgetting", "norm_type"]
    ablations = {}
    for name in ablation_names:
        path = os.path.join(out, "ablations", name, "ablation_results.json")
        d = _try_load(path)
        if d:
            ablations[name] = d

    print(f"\nSaving figures to {save_dir}/")

    # Figure 1: ARR curves QUIC22
    plot_arr_curves(
        seq_quic22,
        os.path.join(save_dir, "fig1_arr_quic22.pdf"),
        title="Sequential ARR — CESNET-QUIC22 (W-45 → W-47)",
    )

    # Figure 2: Comparison bar W-45
    plot_comparison_bar(
        single,
        os.path.join(save_dir, "fig2_comparison_w45.pdf"),
    )

    # Figure 3: AURC bar QUIC22
    plot_aurc_bar(
        seq_quic22,
        os.path.join(save_dir, "fig3_aurc_quic22.pdf"),
        title="AURC — CESNET-QUIC22 Sequential",
    )

    # Figure 4: Long-term TLS-Year22
    plot_long_term_arr(
        seq_tls22,
        os.path.join(save_dir, "fig4_arr_tls22.pdf"),
    )

    # Figure 5+: Ablations
    ablation_titles = {
        "ssl_tasks":       "Ablation A1: SSL Task Selection",
        "mask_ratio":      "Ablation A2: MPFP Mask Ratio",
        "adapt_depth":     "Ablation A3: Adaptation Depth",
        "anti_forgetting": "Ablation A5: Anti-Forgetting Mechanisms",
        "norm_type":       "Ablation A8: Normalization Type",
    }
    for i, (name, data) in enumerate(ablations.items(), start=5):
        plot_ablation_bars(
            data, name,
            os.path.join(save_dir, f"fig{i}_ablation_{name}.pdf"),
            title=ablation_titles.get(name, f"Ablation: {name}"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
