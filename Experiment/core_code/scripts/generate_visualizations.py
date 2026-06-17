"""
Generate paper visualizations: confusion heatmap and t-SNE/UMAP plots.

1. Confusion heatmap: 12 collapse classes x their absorbers, at M3 vs M12
2. t-SNE/UMAP: 3-4 representative absorber-victim pairs, showing feature
   space before (M3/M4) and after (M12) drift

Requires loading the source model, extracting features from reference and
target periods, and computing confusion matrices.

Outputs saved to Publication/figures/.

Usage from Experiment/core_code/:
    python scripts/generate_visualizations.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir Publication/figures
"""
import argparse
import os
import sys
import tempfile
from collections import Counter, OrderedDict

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tta_tc_matplotlib_cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto

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

DEFAULT_COLLAPSE_CLASSES = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
DEFAULT_ABSORBER_MAP = {
    56: 96, 163: 96, 174: 46, 48: 14, 38: 45, 69: 105,
    104: 2, 47: 71, 66: 5, 10: 156, 109: 71, 26: 13,
}
# Representative pairs for t-SNE (high-support, diverse patterns)
DEFAULT_TSNE_PAIRS = [
    (56, 96),   # largest collapse pair
    (48, 14),   # gradual collapse
    (104, 2),   # sudden collapse
    (10, 156),  # low-support collapse
]


def parse_int_list(value, default=None):
    if value is None:
        return list(default or [])
    return [int(x) for x in str(value).replace(",", " ").split()]


def period_slug(period):
    return period.lower().replace("m-2022-", "m").replace("-", "_")


# ============================================================
# Figure 1: Confusion heatmap -- collapse classes x absorbers
# ============================================================

def build_confusion_matrix(labels, preds, collapse_classes, absorber_map, num_classes):
    """Build a confusion matrix restricted to collapse classes as rows
    and relevant predicted classes as columns."""
    # Determine column classes: absorbers + self + any high-frequency predictions
    column_classes = OrderedDict()
    for c in collapse_classes:
        column_classes[c] = None  # self
        if c in absorber_map:
            column_classes[absorber_map[c]] = None

    # Add top-k predicted classes for each collapse class
    for c in collapse_classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        pred_counts = Counter(preds[mask].tolist())
        for pred_class, _ in pred_counts.most_common(5):
            column_classes[int(pred_class)] = None

    columns = list(column_classes.keys())
    matrix = np.zeros((len(collapse_classes), len(columns)), dtype=np.float32)

    for row_idx, c in enumerate(collapse_classes):
        mask = labels == c
        support = int(mask.sum())
        if support == 0:
            continue
        pred_counts = Counter(preds[mask].tolist())
        for col_idx, pred_c in enumerate(columns):
            matrix[row_idx, col_idx] = pred_counts.get(pred_c, 0) / support

    return matrix, columns


def save_confusion_heatmap(labels, preds, collapse_classes, absorber_map,
                           num_classes, output_dir, period, tag=""):
    """Generate and save confusion heatmap."""
    matrix, columns = build_confusion_matrix(
        labels, preds, collapse_classes, absorber_map, num_classes
    )

    slug = period_slug(period)
    absorber_set = set(absorber_map.values())

    fig_w = max(8.0, 0.5 * len(columns) + 3.0)
    fig_h = max(5.0, 0.45 * len(collapse_classes) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(0.1, float(np.max(matrix)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Prediction rate")

    # Column labels with absorber markers
    col_labels = []
    for c in columns:
        label = str(c)
        if c in absorber_set:
            label += " (A)"
        col_labels.append(label)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(collapse_classes)))
    ax.set_yticklabels([str(c) for c in collapse_classes], fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True (collapsed) class")
    ax.set_title(f"Collapse-class confusion heatmap ({period})")

    # Annotate cells with values >= 0.05
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            val = matrix[row_idx, col_idx]
            if val >= 0.05:
                color = "white" if val > vmax * 0.45 else "black"
                ax.text(col_idx, row_idx, f"{val:.2f}",
                        ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    filename = f"fig_confusion_heatmap_{slug}{tag}.pdf"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================
# Figure 2: t-SNE / UMAP of absorber-victim pairs
# ============================================================

def sample_indices_by_class(labels, classes, max_per_class, seed):
    """Sample up to max_per_class indices for each class."""
    rng = np.random.default_rng(seed)
    indices = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        if idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        indices.extend(idx.tolist())
    return np.array(sorted(indices), dtype=np.int64)


def compute_embedding(features, method="tsne", seed=42, perplexity=30):
    """Compute 2D embedding using t-SNE or UMAP."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    x = StandardScaler().fit_transform(x)

    # PCA pre-reduction for speed
    if x.shape[1] > 50 and x.shape[0] > 60:
        n_components = min(50, x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=n_components, random_state=seed).fit_transform(x)

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                                min_dist=0.1, metric="cosine")
            return reducer.fit_transform(x), "UMAP"
        except ImportError:
            print("  UMAP not available, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        actual_perplexity = min(perplexity, max(5, (x.shape[0] - 1) // 3))
        embedding = TSNE(
            n_components=2, init="pca", learning_rate="auto",
            perplexity=actual_perplexity, random_state=seed,
        ).fit_transform(x)
        return embedding, "t-SNE"

    raise ValueError(f"Unknown embedding method: {method}")


def save_pair_tsne(features_ref, labels_ref, preds_ref,
                   features_tgt, labels_tgt, preds_tgt,
                   pair, output_dir, ref_period, tgt_period,
                   method="tsne", max_per_class=500, seed=42):
    """Generate side-by-side t-SNE/UMAP for one absorber-victim pair."""
    victim, absorber = pair
    classes = [victim, absorber]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (features, labels, preds, period) in zip(
        axes,
        [
            (features_ref, labels_ref, preds_ref, ref_period),
            (features_tgt, labels_tgt, preds_tgt, tgt_period),
        ],
    ):
        idx = sample_indices_by_class(labels, classes, max_per_class, seed)
        if idx.size < 10:
            ax.set_title(f"{period}: insufficient data")
            continue

        sub_features = features[idx]
        sub_labels = labels[idx]
        sub_preds = preds[idx]

        embedding, method_name = compute_embedding(sub_features, method=method, seed=seed)

        # Color by true class, marker by prediction correctness
        colors = {victim: "#d62728", absorber: "#1f77b4"}
        for cls, label_str, marker in [
            (victim, f"Class {victim} (victim)", "o"),
            (absorber, f"Class {absorber} (absorber)", "s"),
        ]:
            mask = sub_labels == cls
            if not np.any(mask):
                continue

            # Correct predictions: filled; misclassified: hollow
            correct = sub_preds[mask] == sub_labels[mask]
            if np.any(correct):
                ax.scatter(
                    embedding[mask][correct, 0], embedding[mask][correct, 1],
                    s=20, marker=marker, c=colors[cls], alpha=0.6,
                    label=f"{label_str} (correct)", edgecolors="none",
                )
            if np.any(~correct):
                ax.scatter(
                    embedding[mask][~correct, 0], embedding[mask][~correct, 1],
                    s=20, marker=marker, facecolors="none", edgecolors=colors[cls],
                    alpha=0.6, label=f"{label_str} (misclassified)", linewidths=1.0,
                )

        # Compute recall for title
        v_mask = sub_labels == victim
        v_recall = float((sub_preds[v_mask] == victim).sum()) / max(v_mask.sum(), 1)
        ax.set_title(f"{period} ({method_name})\n"
                     f"Victim {victim} recall: {v_recall:.2f}")
        ax.legend(loc="best", fontsize=7, markerscale=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Class {victim} (victim) vs Class {absorber} (absorber)",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    slug_ref = period_slug(ref_period)
    slug_tgt = period_slug(tgt_period)
    filename = f"fig_tsne_pair_{victim}_to_{absorber}_{slug_ref}_vs_{slug_tgt}.pdf"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================
# Figure 3: Combined multi-pair t-SNE overview
# ============================================================

def save_multi_pair_overview(features_tgt, labels_tgt, preds_tgt,
                             pairs, output_dir, tgt_period,
                             method="tsne", max_per_class=300, seed=42):
    """Generate a 2x2 grid of t-SNE plots for multiple pairs."""
    n_pairs = len(pairs)
    ncols = 2
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols, idx % ncols]
        victim, absorber = pair
        classes = [victim, absorber]

        sample_idx = sample_indices_by_class(labels_tgt, classes, max_per_class, seed)
        if sample_idx.size < 10:
            ax.set_title(f"Pair {victim}->{absorber}: insufficient data")
            continue

        sub_features = features_tgt[sample_idx]
        sub_labels = labels_tgt[sample_idx]
        sub_preds = preds_tgt[sample_idx]

        embedding, method_name = compute_embedding(sub_features, method=method, seed=seed)

        colors = {victim: "#d62728", absorber: "#1f77b4"}
        for cls, role in [(victim, "victim"), (absorber, "absorber")]:
            mask = sub_labels == cls
            if not np.any(mask):
                continue
            marker = "o" if role == "victim" else "s"
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=18, marker=marker, c=colors[cls], alpha=0.6,
                label=f"{cls} ({role})", edgecolors="none",
            )

        v_mask = sub_labels == victim
        v_recall = float((sub_preds[v_mask] == victim).sum()) / max(v_mask.sum(), 1)
        ax.set_title(f"{victim} -> {absorber} (recall={v_recall:.2f})")
        ax.legend(loc="best", fontsize=7, markerscale=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    slug = period_slug(tgt_period)
    fig.suptitle(f"Absorber-victim feature overlap ({tgt_period})", fontsize=13, y=1.01)
    fig.tight_layout()

    filename = f"fig_tsne_multi_pair_{slug}.pdf"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper visualizations: confusion heatmap and t-SNE/UMAP."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-3",
                        help="Early reference period for before-drift comparison")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--collapse-classes", default=None)
    parser.add_argument("--embedding-method", choices=["tsne", "umap"], default="tsne",
                        help="Dimensionality reduction method (default: tsne)")
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="Max samples per class for t-SNE (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="Publication/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    collapse_classes = parse_int_list(args.collapse_classes, DEFAULT_COLLAPSE_CLASSES)

    # Load reference period (before drift)
    print(f"\n=== Loading reference period: {args.reference_period} ===")
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_out = proto.collect_outputs(model, ref_loader, device,
                                    desc=f"Ref {args.reference_period}")
    ref_features = ref_out["features"].numpy()
    ref_labels = ref_out["labels"]
    ref_preds = ref_out["logits"].argmax(dim=1).numpy()

    # Load target period (after drift)
    print(f"\n=== Loading target period: {args.target_period} ===")
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device,
                                    desc=f"Tgt {args.target_period}")
    tgt_features = tgt_out["features"].numpy()
    tgt_labels = tgt_out["labels"]
    tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()

    print(f"\nReference: {len(ref_labels)} samples")
    print(f"Target:    {len(tgt_labels)} samples")

    # ---- Figure 1: Confusion heatmaps (M3 vs M12) ----
    print(f"\n=== Figure 1: Confusion heatmaps ===")
    save_confusion_heatmap(
        ref_labels, ref_preds, collapse_classes, DEFAULT_ABSORBER_MAP,
        num_classes, args.output_dir, args.reference_period,
    )
    save_confusion_heatmap(
        tgt_labels, tgt_preds, collapse_classes, DEFAULT_ABSORBER_MAP,
        num_classes, args.output_dir, args.target_period,
    )

    # ---- Figure 2: Per-pair t-SNE (before vs after) ----
    print(f"\n=== Figure 2: Per-pair {args.embedding_method.upper()} (before vs after) ===")
    for pair in DEFAULT_TSNE_PAIRS:
        save_pair_tsne(
            ref_features, ref_labels, ref_preds,
            tgt_features, tgt_labels, tgt_preds,
            pair, args.output_dir,
            args.reference_period, args.target_period,
            method=args.embedding_method,
            max_per_class=args.max_per_class,
            seed=args.seed,
        )

    # ---- Figure 3: Multi-pair overview (target period) ----
    print(f"\n=== Figure 3: Multi-pair overview ({args.target_period}) ===")
    save_multi_pair_overview(
        tgt_features, tgt_labels, tgt_preds,
        DEFAULT_TSNE_PAIRS, args.output_dir, args.target_period,
        method=args.embedding_method,
        max_per_class=args.max_per_class // 2,
        seed=args.seed,
    )

    # Print summary of recall changes
    print(f"\n=== Collapse class recall summary ===")
    print(f"{'Class':>6} {'Absorber':>8} {args.reference_period:>10} {args.target_period:>10} {'Delta':>8}")
    print("-" * 50)
    for c in collapse_classes:
        absorber = DEFAULT_ABSORBER_MAP.get(c, "?")
        ref_mask = ref_labels == c
        tgt_mask = tgt_labels == c
        ref_recall = float((ref_preds[ref_mask] == c).sum()) / max(ref_mask.sum(), 1)
        tgt_recall = float((tgt_preds[tgt_mask] == c).sum()) / max(tgt_mask.sum(), 1)
        delta = tgt_recall - ref_recall
        print(f"{c:>6} {str(absorber):>8} {ref_recall:>10.3f} {tgt_recall:>10.3f} {delta:>+8.3f}")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
