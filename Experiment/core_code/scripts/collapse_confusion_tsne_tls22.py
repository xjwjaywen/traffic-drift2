"""
Visual diagnostics for TLS22 collapsed classes.

This script is for advisor-requested analysis, not a new adaptation method. It
answers two questions:

  1. Which absorber classes do collapsed classes get predicted as?
  2. In representation space, are collapsed-class samples clustered, mixed with
     absorbers, or scattered?

Outputs:
  - selected_collapse_confusion_<period>.csv
  - selected_collapse_confusion_heatmap_<period>.png
  - selected_collapse_pair_summary_<period>.csv
  - tsne_selected_collapse_absorbers_<period>.png
  - tsne_pair_<true>_to_<absorber>_<period>.png

Usage from Experiment/core_code/:
    python scripts/collapse_confusion_tsne_tls22.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --target-period M-2022-12 \
      --output-dir outputs/collapse_visual_diagnostics_tls22
"""
import argparse
import csv
import os
import re
import sys
import tempfile
from collections import Counter, OrderedDict

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
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import prototype_recalibration_tls22 as proto
from tta_tc.utils.config import load_config


DEFAULT_PAIRS = [
    (56, 96),
    (48, 14),
    (104, 2),
    (10, 156),
    (109, 71),
    (26, 13),
]


def sanitize_period(period):
    match = re.match(r"([MW])-2022-(\d+)$", period)
    if match:
        return f"{match.group(1).lower()}{match.group(2)}"
    return re.sub(r"[^A-Za-z0-9]+", "_", period).strip("_").lower()


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


def parse_pairs(value):
    if not value:
        return list(DEFAULT_PAIRS)
    pairs = []
    for token in value.replace(",", " ").split():
        if ":" in token:
            left, right = token.split(":", 1)
        elif "->" in token:
            left, right = token.split("->", 1)
        else:
            raise ValueError(f"Bad pair token: {token}. Use true:absorber.")
        pairs.append((int(left), int(right)))
    return pairs


def load_pairs_from_collapse_report(path, target_period, top_k, fallback_pairs):
    if not path or not os.path.exists(path):
        return list(fallback_pairs), False
    rows = read_csv(path)
    candidates = []
    for row in rows:
        first = row.get("first_collapse_period") or ""
        final_recall = as_float(row.get("final_recall"))
        absorber = row.get("final_top_confusion_target")
        support = as_int(row.get("final_support"), 0)
        if not first or not np.isfinite(final_recall) or final_recall >= 0.1:
            continue
        if absorber in (None, ""):
            continue
        candidates.append({
            "true_class": as_int(row["class_id"]),
            "absorber": as_int(absorber),
            "support": support,
            "final_recall": final_recall,
            "confusion_rate": as_float(row.get("final_top_confusion_rate"), 0.0),
            "pattern": row.get("collapse_pattern", ""),
        })
    if not candidates:
        return list(fallback_pairs), False
    candidates.sort(key=lambda r: (-r["support"], r["final_recall"], -r["confusion_rate"]))
    selected = [(row["true_class"], row["absorber"]) for row in candidates[:top_k]]
    return selected, True


def stable_classes_from_string(value):
    if not value:
        return []
    return [int(x) for x in value.replace(",", " ").split()]


def sample_indices_by_class(labels, classes, max_per_class, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for class_id in classes:
        cls_idx = np.flatnonzero(labels == class_id)
        if cls_idx.size == 0:
            continue
        if cls_idx.size > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        indices.extend(cls_idx.tolist())
    return np.array(sorted(indices), dtype=np.int64)


def prepare_tsne(features, random_state, perplexity):
    x = np.asarray(features, dtype=np.float32)
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 50 and x.shape[0] > 60:
        n_components = min(50, x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=n_components, random_state=random_state).fit_transform(x)
    actual_perplexity = min(perplexity, max(5, (x.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=actual_perplexity,
        random_state=random_state,
    ).fit_transform(x)


def class_color_map(classes):
    cmap = plt.get_cmap("tab20")
    return {class_id: cmap(i % 20) for i, class_id in enumerate(classes)}


def save_confusion_heatmap(labels, preds, pairs, top_k, output_dir, period_slug):
    selected_true = [true for true, _ in pairs]
    columns = []
    confusion_rows = []

    for true_class, absorber in pairs:
        mask = labels == true_class
        support = int(mask.sum())
        pred_counts = Counter(preds[mask].tolist())
        columns.extend([true_class, absorber])
        for pred_class, _ in pred_counts.most_common(top_k):
            columns.append(int(pred_class))
        for rank, (pred_class, count) in enumerate(pred_counts.most_common(top_k), start=1):
            confusion_rows.append({
                "period": period_slug,
                "true_class": true_class,
                "pred_class": int(pred_class),
                "rank": rank,
                "count": int(count),
                "support": support,
                "rate": float(count / support) if support else 0.0,
                "is_absorber": int(int(pred_class) == absorber),
                "is_correct": int(int(pred_class) == true_class),
            })

    columns = list(OrderedDict((c, None) for c in columns).keys())
    matrix = np.zeros((len(selected_true), len(columns)), dtype=np.float32)
    for row_idx, true_class in enumerate(selected_true):
        mask = labels == true_class
        support = int(mask.sum())
        pred_counts = Counter(preds[mask].tolist())
        for col_idx, pred_class in enumerate(columns):
            matrix[row_idx, col_idx] = pred_counts.get(pred_class, 0) / support if support else 0.0

    fig_w = max(8.0, 0.55 * len(columns) + 2.5)
    fig_h = max(4.5, 0.48 * len(selected_true) + 2.0)
    plt.figure(figsize=(fig_w, fig_h))
    vmax = max(0.1, float(np.max(matrix)))
    im = plt.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    plt.colorbar(im, label="Prediction rate within true class")
    plt.xticks(np.arange(len(columns)), [str(c) for c in columns], rotation=45, ha="right")
    plt.yticks(np.arange(len(selected_true)), [str(c) for c in selected_true])
    plt.xlabel("Predicted class")
    plt.ylabel("True collapsed class")
    plt.title("Collapsed-class confusion heatmap")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value >= 0.05:
                color = "white" if value > vmax * 0.45 else "black"
                plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    path = os.path.join(output_dir, f"selected_collapse_confusion_heatmap_{period_slug}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(output_dir, f"selected_collapse_confusion_{period_slug}.csv")
    write_csv(csv_path, confusion_rows)
    return path, csv_path


def pair_summary(labels, preds, logits, features, pairs):
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    top2 = np.sort(probs, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]

    rows = []
    for true_class, absorber in pairs:
        true_mask = labels == true_class
        absorber_mask = labels == absorber
        support = int(true_mask.sum())
        absorber_count = int(np.sum(preds[true_mask] == absorber))
        correct_count = int(np.sum(preds[true_mask] == true_class))
        true_feat = features[true_mask]
        absorber_feat = features[absorber_mask]
        centroid_dist = ""
        if true_feat.size and absorber_feat.size:
            true_centroid = true_feat.mean(axis=0)
            absorber_centroid = absorber_feat.mean(axis=0)
            true_centroid = true_centroid / (np.linalg.norm(true_centroid) + 1e-12)
            absorber_centroid = absorber_centroid / (np.linalg.norm(absorber_centroid) + 1e-12)
            centroid_dist = float(1.0 - np.dot(true_centroid, absorber_centroid))
        rows.append({
            "true_class": true_class,
            "absorber_class": absorber,
            "true_support": support,
            "absorber_support": int(absorber_mask.sum()),
            "true_recall": float(correct_count / support) if support else 0.0,
            "absorber_confusion_rate": float(absorber_count / support) if support else 0.0,
            "absorber_confusion_count": absorber_count,
            "mean_confidence_true_samples": float(probs[true_mask].max(axis=1).mean()) if support else "",
            "mean_margin_true_samples": float(margins[true_mask].mean()) if support else "",
            "target_centroid_cosine_distance": centroid_dist,
        })
    return rows


def save_global_tsne(features, labels, preds, pairs, stable_classes, output_dir, period_slug, max_per_class, seed, perplexity):
    collapsed = [true for true, _ in pairs]
    absorbers = [absorber for _, absorber in pairs]
    classes = list(OrderedDict((c, None) for c in collapsed + absorbers + stable_classes).keys())
    idx = sample_indices_by_class(labels, classes, max_per_class, seed)
    if idx.size < 10:
        return None

    xy = prepare_tsne(features[idx], seed, perplexity)
    sample_labels = labels[idx]
    sample_preds = preds[idx]
    colors = class_color_map(classes)

    plt.figure(figsize=(10.5, 8.0))
    for class_id in classes:
        mask = sample_labels == class_id
        if not np.any(mask):
            continue
        role = "collapsed" if class_id in collapsed else "absorber" if class_id in absorbers else "stable"
        marker = "x" if role == "collapsed" else "o" if role == "absorber" else "^"
        alpha = 0.78 if role != "stable" else 0.45
        plt.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=16 if role != "stable" else 12,
            marker=marker,
            color=colors[class_id],
            alpha=alpha,
            label=f"{class_id} ({role})",
        )

    wrong_mask = sample_preds != sample_labels
    plt.scatter(
        xy[wrong_mask, 0],
        xy[wrong_mask, 1],
        s=34,
        facecolors="none",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.45,
        label="misclassified",
    )
    plt.title(f"t-SNE of selected collapsed / absorber / stable classes ({period_slug})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=7, ncol=2, loc="best")
    plt.grid(True, alpha=0.2)
    path = os.path.join(output_dir, f"tsne_selected_collapse_absorbers_{period_slug}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def save_pair_tsne(features, labels, preds, pair, output_dir, period_slug, max_per_class, seed, perplexity):
    true_class, absorber = pair
    idx = sample_indices_by_class(labels, [true_class, absorber], max_per_class, seed + true_class)
    if idx.size < 10:
        return None

    xy = prepare_tsne(features[idx], seed + true_class, perplexity)
    sample_labels = labels[idx]
    sample_preds = preds[idx]
    colors = {
        true_class: "#1f77b4",
        absorber: "#ff7f0e",
        "other": "#7f7f7f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for class_id, label_name in [(true_class, f"true {true_class}"), (absorber, f"true absorber {absorber}")]:
        mask = sample_labels == class_id
        axes[0].scatter(xy[mask, 0], xy[mask, 1], s=16, alpha=0.75, color=colors[class_id], label=label_name)
    axes[0].set_title("Colored by true label")
    axes[0].legend(fontsize=8)

    pred_groups = [
        (sample_preds == true_class, f"pred {true_class}", colors[true_class]),
        (sample_preds == absorber, f"pred absorber {absorber}", colors[absorber]),
        ((sample_preds != true_class) & (sample_preds != absorber), "pred other", colors["other"]),
    ]
    for mask, label_name, color in pred_groups:
        if np.any(mask):
            axes[1].scatter(xy[mask, 0], xy[mask, 1], s=16, alpha=0.75, color=color, label=label_name)
    axes[1].set_title("Colored by predicted label")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"t-SNE pair diagnostic: {true_class} -> {absorber} ({period_slug})")
    path = os.path.join(output_dir, f"tsne_pair_{true_class}_to_{absorber}_{period_slug}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(path, period, pairs, confusion_path, heatmap_path, global_tsne_path, pair_paths, pair_summary_path):
    lines = [
        "# Collapse Confusion and t-SNE Diagnostics",
        "",
        f"- Target period: `{period}`",
        f"- Selected collapse pairs: `{', '.join(f'{a}->{b}' for a, b in pairs)}`",
        "",
        "## Outputs",
        "",
        f"- Confusion CSV: `{confusion_path}`",
        f"- Confusion heatmap: `{heatmap_path}`",
        f"- Pair summary CSV: `{pair_summary_path}`",
    ]
    if global_tsne_path:
        lines.append(f"- Global t-SNE: `{global_tsne_path}`")
    for path_item in pair_paths:
        lines.append(f"- Pair t-SNE: `{path_item}`")
    lines.extend([
        "",
        "## How To Read",
        "",
        "- The confusion heatmap shows where selected collapsed classes are predicted.",
        "- The global t-SNE shows selected collapsed classes, their absorber classes, and optional stable anchors.",
        "- Each pair t-SNE has two panels: true labels on the left and predicted labels on the right. If collapsed-class points overlap with absorber points and are mostly colored as absorber on the prediction panel, the failure is representation/decision collapse rather than a simple global shift.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--collapse-report", default="outputs/per_class_collapse_tls22_monthly/collapse_classes.csv")
    parser.add_argument("--output-dir", default="outputs/collapse_visual_diagnostics_tls22")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--top-pairs-from-report", type=int, default=6)
    parser.add_argument("--top-confusions", type=int, default=5)
    parser.add_argument("--max-samples-per-class", type=int, default=250)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--stable-classes",
        default="98,107,145",
        help="Optional stable anchor classes to include in the global t-SNE.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    period_slug = sanitize_period(args.target_period)

    fallback_pairs = parse_pairs(args.pairs)
    if args.pairs:
        pairs = fallback_pairs
        report_loaded = False
    else:
        pairs, report_loaded = load_pairs_from_collapse_report(
            args.collapse_report,
            args.target_period,
            args.top_pairs_from_report,
            fallback_pairs,
        )
    stable_classes = stable_classes_from_string(args.stable_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = load_config(args.config)
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes
    loader, loader_classes = proto.make_test_loader(eval_cfg, args.target_period)
    if loader_classes != num_classes:
        print(f"WARNING: loader classes={loader_classes}, model classes={num_classes}")

    print(f"Using device: {device}")
    print(f"Collapse report loaded: {report_loaded} ({args.collapse_report})")
    print(f"Selected pairs: {pairs}")
    outputs = proto.collect_outputs(model, loader, device, f"Target {args.target_period}")
    features = outputs["features"].numpy().astype(np.float32, copy=False)
    logits = outputs["logits"].numpy().astype(np.float32, copy=False)
    labels = outputs["labels"].astype(np.int64, copy=False)
    preds = logits.argmax(axis=1).astype(np.int64, copy=False)

    heatmap_path, confusion_path = save_confusion_heatmap(
        labels,
        preds,
        pairs,
        args.top_confusions,
        args.output_dir,
        period_slug,
    )
    summary_rows = pair_summary(labels, preds, logits, features, pairs)
    pair_summary_path = os.path.join(args.output_dir, f"selected_collapse_pair_summary_{period_slug}.csv")
    write_csv(pair_summary_path, summary_rows)

    global_tsne_path = save_global_tsne(
        features,
        labels,
        preds,
        pairs,
        stable_classes,
        args.output_dir,
        period_slug,
        args.max_samples_per_class,
        args.seed,
        args.tsne_perplexity,
    )
    pair_paths = []
    for pair in pairs:
        path = save_pair_tsne(
            features,
            labels,
            preds,
            pair,
            args.output_dir,
            period_slug,
            args.max_samples_per_class,
            args.seed,
            args.tsne_perplexity,
        )
        if path:
            pair_paths.append(path)

    report_path = os.path.join(args.output_dir, f"collapse_visual_diagnostics_{period_slug}.md")
    write_report(
        report_path,
        args.target_period,
        pairs,
        confusion_path,
        heatmap_path,
        global_tsne_path,
        pair_paths,
        pair_summary_path,
    )
    print(f"Saved confusion heatmap: {heatmap_path}")
    print(f"Saved confusion CSV: {confusion_path}")
    print(f"Saved pair summary CSV: {pair_summary_path}")
    if global_tsne_path:
        print(f"Saved global t-SNE: {global_tsne_path}")
    for path in pair_paths:
        print(f"Saved pair t-SNE: {path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
