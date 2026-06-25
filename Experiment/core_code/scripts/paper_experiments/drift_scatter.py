"""
Drift mechanism scatter plot data (Paper experiment 4).

For each collapse class, computes:
  x-axis: feature drift intensity (Fisher ratio change, centroid shift)
  y-axis: boundary drift intensity (margin drop, absorber confidence change)

Outputs a CSV + a matplotlib scatter plot.

Usage from Experiment/core_code/:
    python scripts/paper_experiments/drift_scatter.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/paper_experiments/drift_scatter
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(PARENT_DIR))

import prototype_recalibration_tls22 as proto

ABSORBER_PAIRS = [
    (56, 96), (163, 46), (174, 2), (48, 14), (38, 45), (69, 105),
    (104, 2), (47, 5), (66, 71), (10, 156), (109, 71), (26, 13),
]

# Labels from Table 2 (Fisher trajectory analysis in main paper).
# Feature drift: ΔJ < -10%. Boundary shift: ΔJ > +20%. Mixed: otherwise.
MECHANISM_LABELS = {
    56: "feature", 174: "feature", 38: "feature", 163: "feature",
    48: "boundary", 104: "boundary",
    47: "mixed", 69: "mixed", 66: "mixed", 10: "mixed", 109: "mixed", 26: "mixed",
}


def compute_fisher_ratio(features, labels, victim, absorber):
    """Fisher discriminant ratio between victim and absorber classes."""
    v_mask = labels == victim
    a_mask = labels == absorber
    v_n = int(v_mask.sum())
    a_n = int(a_mask.sum())
    if v_n < 5 or a_n < 5:
        return None
    v_feat = features[v_mask].numpy()
    a_feat = features[a_mask].numpy()
    mu_v = v_feat.mean(axis=0)
    mu_a = a_feat.mean(axis=0)
    spread_v = np.mean(np.linalg.norm(v_feat - mu_v, axis=1))
    spread_a = np.mean(np.linalg.norm(a_feat - mu_a, axis=1))
    dist = np.linalg.norm(mu_v - mu_a)
    return float(dist / (spread_v + spread_a + 1e-8))


def compute_centroid_shift(ref_features, ref_labels, tgt_features, tgt_labels, cls):
    """L2 distance between class centroid at ref vs target."""
    ref_mask = ref_labels == cls
    tgt_mask = tgt_labels == cls
    if ref_mask.sum() < 5 or tgt_mask.sum() < 5:
        return None
    ref_centroid = ref_features[ref_mask].numpy().mean(axis=0)
    tgt_centroid = tgt_features[tgt_mask].numpy().mean(axis=0)
    return float(np.linalg.norm(tgt_centroid - ref_centroid))


def compute_margin_stats(logits, labels, cls):
    """Mean margin (top-2 logit gap) for samples of a given class."""
    mask = labels == cls
    if mask.sum() < 5:
        return None
    cls_logits = logits[mask]
    top2 = torch.topk(cls_logits, 2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    return float(np.mean(margin))


def compute_absorber_confidence(logits, labels, victim, absorber):
    """Mean softmax confidence of absorber class for victim samples."""
    import torch.nn.functional as F
    mask = labels == victim
    if mask.sum() < 5:
        return None
    probs = F.softmax(logits[mask], dim=1)
    return float(probs[:, absorber].mean())


def main():
    parser = argparse.ArgumentParser(description="Drift mechanism scatter plot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    # Collect ref and target
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_out = proto.collect_outputs(model, ref_loader, device, desc=f"Ref {args.reference_period}")

    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device, desc=f"Tgt {args.target_period}")

    print(f"=== Drift Mechanism Scatter Data ===")
    print(f"{'Victim':>6} {'Abs':>4} {'Mech':<8} {'Fisher_ref':>10} {'Fisher_tgt':>10} "
          f"{'ΔFisher%':>9} {'CentShift':>9} {'MarginRef':>9} {'MarginTgt':>9} "
          f"{'AbsConf_r':>9} {'AbsConf_t':>9}")
    print("-" * 110)

    rows = []
    for victim, absorber in ABSORBER_PAIRS:
        fisher_ref = compute_fisher_ratio(ref_out["features"], ref_out["labels"], victim, absorber)
        fisher_tgt = compute_fisher_ratio(tgt_out["features"], tgt_out["labels"], victim, absorber)
        centroid_shift = compute_centroid_shift(
            ref_out["features"], ref_out["labels"],
            tgt_out["features"], tgt_out["labels"], victim)
        margin_ref = compute_margin_stats(ref_out["logits"], ref_out["labels"], victim)
        margin_tgt = compute_margin_stats(tgt_out["logits"], tgt_out["labels"], victim)
        abs_conf_ref = compute_absorber_confidence(ref_out["logits"], ref_out["labels"], victim, absorber)
        abs_conf_tgt = compute_absorber_confidence(tgt_out["logits"], tgt_out["labels"], victim, absorber)

        delta_fisher = None
        if fisher_ref and fisher_tgt and fisher_ref > 0:
            delta_fisher = (fisher_tgt - fisher_ref) / fisher_ref * 100

        mechanism = MECHANISM_LABELS.get(victim, "unknown")

        row = {
            "victim": victim, "absorber": absorber, "mechanism": mechanism,
            "fisher_ref": fisher_ref, "fisher_tgt": fisher_tgt,
            "delta_fisher_pct": delta_fisher,
            "centroid_shift": centroid_shift,
            "margin_ref": margin_ref, "margin_tgt": margin_tgt,
            "margin_drop": (margin_ref - margin_tgt) if margin_ref and margin_tgt else None,
            "absorber_conf_ref": abs_conf_ref, "absorber_conf_tgt": abs_conf_tgt,
            "absorber_conf_rise": (abs_conf_tgt - abs_conf_ref) if abs_conf_ref and abs_conf_tgt else None,
        }
        rows.append(row)

        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print(f"  {victim:>4d} {absorber:>4d} {mechanism:<8} "
              f"{_f(fisher_ref, '.3f'):>10} {_f(fisher_tgt, '.3f'):>10} "
              f"{_f(delta_fisher, '+.1f'):>8}% {_f(centroid_shift, '.2f'):>9} "
              f"{_f(margin_ref, '.3f'):>9} {_f(margin_tgt, '.3f'):>9} "
              f"{_f(abs_conf_ref, '.4f'):>9} {_f(abs_conf_tgt, '.4f'):>9}")

    fieldnames = list(rows[0].keys())
    with open(os.path.join(args.output_dir, "drift_scatter_data.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Generate scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        colors = {"feature": "#e74c3c", "boundary": "#3498db", "mixed": "#95a5a6"}
        markers = {"feature": "o", "boundary": "s", "mixed": "D"}

        for row in rows:
            x = row.get("delta_fisher_pct")
            y = row.get("margin_drop")
            if x is None or y is None:
                continue
            mech = row["mechanism"]
            ax.scatter(x, y, c=colors[mech], marker=markers[mech], s=100,
                       edgecolors="black", linewidth=0.5, zorder=3)
            ax.annotate(str(row["victim"]), (x, y), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fisher Ratio Change (%)\n← feature drift (converging) | boundary shift (separating) →", fontsize=11)
        ax.set_ylabel("Margin Drop\n← stable | boundary confusion →", fontsize=11)
        ax.set_title("Drift Mechanism: Feature vs Boundary", fontsize=12)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=8, label="Feature drift"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
                   markersize=8, label="Boundary shift"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#95a5a6",
                   markersize=8, label="Mixed"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "drift_scatter.pdf"), dpi=150)
        plt.savefig(os.path.join(args.output_dir, "drift_scatter.png"), dpi=150)
        print(f"\nScatter plot saved to {args.output_dir}/drift_scatter.pdf")
    except ImportError:
        print("\nmatplotlib not available, scatter plot skipped")

    print(f"Data saved to {args.output_dir}/drift_scatter_data.csv")


if __name__ == "__main__":
    main()
