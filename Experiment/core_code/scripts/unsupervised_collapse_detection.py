"""
Unsupervised per-class collapse detection via Frechet Distance on embeddings.

Groups features by PREDICTED class (no labels needed), computes per-class
Frechet Distance between reference and target periods. High FD = that class's
prediction region has shifted, indicating potential collapse or absorption.

Usage:
    python scripts/unsupervised_collapse_detection.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --reference-period M-2022-4 \
      --target-period M-2022-12 \
      --output-dir outputs/unsupervised_collapse_detection
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(fd, 0))


def compute_per_class_signals(ref_features, ref_preds, ref_logits,
                              tgt_features, tgt_preds, tgt_logits,
                              num_classes, min_samples=10):
    """Compute multiple unsupervised signals per predicted class."""
    import torch.nn.functional as F

    ref_probs = F.softmax(ref_logits, dim=1)
    tgt_probs = F.softmax(tgt_logits, dim=1)
    ref_margin = (torch.topk(ref_logits, 2, dim=1).values[:, 0] -
                  torch.topk(ref_logits, 2, dim=1).values[:, 1]).numpy()
    tgt_margin = (torch.topk(tgt_logits, 2, dim=1).values[:, 0] -
                  torch.topk(tgt_logits, 2, dim=1).values[:, 1]).numpy()
    ref_conf = ref_probs.max(dim=1).values.numpy()
    tgt_conf = tgt_probs.max(dim=1).values.numpy()
    ref_ent = (-(ref_probs * torch.log(ref_probs.clamp(min=1e-12))).sum(dim=1)).numpy()
    tgt_ent = (-(tgt_probs * torch.log(tgt_probs.clamp(min=1e-12))).sum(dim=1)).numpy()

    ref_total = len(ref_preds)
    tgt_total = len(tgt_preds)

    results = []
    for c in range(num_classes):
        ref_mask = ref_preds == c
        tgt_mask = tgt_preds == c
        ref_n = int(ref_mask.sum())
        tgt_n = int(tgt_mask.sum())

        # Normalize by total period size to handle different volumes
        ref_frac = ref_n / max(ref_total, 1)
        tgt_frac = tgt_n / max(tgt_total, 1)
        count_ratio = tgt_frac / max(ref_frac, 1e-8)

        if ref_n < min_samples and tgt_n < min_samples:
            results.append({
                "class": c, "fd": None,
                "ref_count": ref_n, "tgt_count": tgt_n,
                "count_ratio": count_ratio,
                "margin_drop": None, "conf_drop": None, "entropy_rise": None,
                "status": "both_insufficient",
            })
            continue

        if tgt_n < min_samples:
            # Near-zero predictions in target = strong collapse signal
            results.append({
                "class": c, "fd": None,
                "ref_count": ref_n, "tgt_count": tgt_n,
                "count_ratio": count_ratio,
                "margin_drop": None, "conf_drop": None, "entropy_rise": None,
                "status": "count_only",
            })
            continue

        if ref_n < min_samples:
            results.append({
                "class": c, "fd": None,
                "ref_count": ref_n, "tgt_count": tgt_n,
                "count_ratio": count_ratio,
                "margin_drop": None, "conf_drop": None, "entropy_rise": None,
                "status": "ref_insufficient",
            })
            continue

        ref_feat = ref_features[ref_mask].numpy()
        tgt_feat = tgt_features[tgt_mask].numpy()

        mu_ref = ref_feat.mean(axis=0)
        mu_tgt = tgt_feat.mean(axis=0)
        sigma_ref = np.cov(ref_feat, rowvar=False) + np.eye(ref_feat.shape[1]) * 1e-6
        sigma_tgt = np.cov(tgt_feat, rowvar=False) + np.eye(tgt_feat.shape[1]) * 1e-6
        fd = frechet_distance(mu_ref, sigma_ref, mu_tgt, sigma_tgt)

        # Per-class margin drop (lower margin in target = more boundary confusion)
        margin_drop = float(np.mean(ref_margin[ref_mask]) - np.mean(tgt_margin[tgt_mask]))
        # Per-class confidence drop
        conf_drop = float(np.mean(ref_conf[ref_mask]) - np.mean(tgt_conf[tgt_mask]))
        # Per-class entropy rise
        entropy_rise = float(np.mean(tgt_ent[tgt_mask]) - np.mean(ref_ent[ref_mask]))

        results.append({
            "class": c, "fd": fd,
            "ref_count": ref_n, "tgt_count": tgt_n,
            "count_ratio": count_ratio,
            "margin_drop": margin_drop,
            "conf_drop": conf_drop,
            "entropy_rise": entropy_rise,
            "status": "ok",
        })

    return results


DEFAULT_WEIGHTS = [0.40, 0.20, 0.15, 0.15, 0.10]


def compute_collapse_score(r, fd_median, fd_mad, margin_stats, conf_stats, ent_stats,
                           weights=None):
    """Combined collapse risk score using 5 unsupervised signals.

    1. count_drop: prediction count decreased (being absorbed)
    2. FD anomaly: feature distribution shifted
    3. margin_drop: predictions near class became less decisive
    4. conf_drop: predictions became less confident
    5. entropy_rise: prediction entropy increased
    """
    w = weights or DEFAULT_WEIGHTS
    fd = r.get("fd", 0) or 0
    fd_z = (fd - fd_median) / (fd_mad + 1e-8)
    count_drop = max(0, 1.0 - r["count_ratio"])
    fd_norm = min(max(fd_z / 5.0, 0), 1.0)

    margin_drop = r.get("margin_drop") or 0
    margin_z = (margin_drop - margin_stats[0]) / (margin_stats[1] + 1e-8)
    margin_norm = min(max(margin_z / 3.0, 0), 1.0)

    conf_drop = r.get("conf_drop") or 0
    conf_z = (conf_drop - conf_stats[0]) / (conf_stats[1] + 1e-8)
    conf_norm = min(max(conf_z / 3.0, 0), 1.0)

    ent_rise = r.get("entropy_rise") or 0
    ent_z = (ent_rise - ent_stats[0]) / (ent_stats[1] + 1e-8)
    ent_norm = min(max(ent_z / 3.0, 0), 1.0)

    score = (w[0] * count_drop + w[1] * fd_norm + w[2] * margin_norm +
             w[3] * conf_norm + w[4] * ent_norm)
    return float(score)


def _robust_stats(values):
    """Compute median and MAD for robust normalization."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return (med, mad)


def detect_collapse_candidates(fd_results, top_k=20, score_threshold=0.12, weights=None):
    """Identify collapse and absorber candidates using multi-signal scoring."""
    valid = [r for r in fd_results if r["fd"] is not None]
    count_only = [r for r in fd_results if r.get("status") == "count_only"]
    for r in count_only:
        r["collapse_score"] = 0.95
        r["fd_zscore"] = 0.0
    if not valid and not count_only:
        return [], []

    fds = np.array([r["fd"] for r in valid])
    fd_median, fd_mad = _robust_stats(fds)

    margins = np.array([r["margin_drop"] for r in valid if r.get("margin_drop") is not None])
    margin_stats = _robust_stats(margins) if len(margins) > 0 else (0, 1)

    confs = np.array([r["conf_drop"] for r in valid if r.get("conf_drop") is not None])
    conf_stats = _robust_stats(confs) if len(confs) > 0 else (0, 1)

    ents = np.array([r["entropy_rise"] for r in valid if r.get("entropy_rise") is not None])
    ent_stats = _robust_stats(ents) if len(ents) > 0 else (0, 1)

    collapse_candidates = []
    absorber_candidates = []

    for r in valid:
        z_score = (r["fd"] - fd_median) / (fd_mad + 1e-8)
        r["fd_zscore"] = float(z_score)
        r["collapse_score"] = compute_collapse_score(
            r, fd_median, fd_mad, margin_stats, conf_stats, ent_stats, weights=weights
        )

        if r["collapse_score"] > score_threshold:
            collapse_candidates.append(r)

        if r["count_ratio"] > 1.3:
            absorber_candidates.append(r)

    # Add count_only entries (classes with near-zero target predictions)
    for r in count_only:
        collapse_candidates.append(r)

    collapse_candidates.sort(key=lambda x: x["collapse_score"], reverse=True)
    absorber_candidates.sort(key=lambda x: x["count_ratio"], reverse=True)

    return collapse_candidates[:top_k], absorber_candidates[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.12,
                        help="Collapse score threshold (not tuned on target labels)")
    parser.add_argument("--weights", default="0.40,0.20,0.15,0.15,0.10",
                        help="Weights for count_drop,fd,margin,conf,entropy")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    # Collect reference period
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_out = proto.collect_outputs(model, ref_loader, device,
                                    desc=f"Ref {args.reference_period}")
    ref_preds = ref_out["logits"].argmax(dim=1).numpy()

    # Collect target period
    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device,
                                    desc=f"Tgt {args.target_period}")
    tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()
    tgt_labels = tgt_out["labels"]

    print(f"\nReference: {args.reference_period} ({len(ref_preds)} samples)")
    print(f"Target: {args.target_period} ({len(tgt_preds)} samples)")

    # Compute per-class signals (using PREDICTIONS, no labels)
    print("\nComputing per-class collapse signals (unsupervised)...")
    fd_results = compute_per_class_signals(
        ref_out["features"], ref_preds, ref_out["logits"],
        tgt_out["features"], tgt_preds, tgt_out["logits"],
        num_classes, args.min_samples,
    )

    # Detect candidates
    weights = [float(w) for w in args.weights.split(",")]
    assert len(weights) == 5, f"Expected 5 weights, got {len(weights)}"
    collapse_cands, absorber_cands = detect_collapse_candidates(
        fd_results, top_k=args.top_k, score_threshold=args.score_threshold,
        weights=weights,
    )

    # Ground truth: actual collapsed classes (using labels)
    # Definition: recall < 0.1 AND support >= 50 (matches paper's 12-class set)
    actual_collapsed = []
    for c in range(num_classes):
        mask = tgt_labels == c
        support = int(mask.sum())
        if support >= 50:
            recall = float((tgt_preds[mask] == c).sum()) / float(support)
            if recall < 0.1:
                actual_collapsed.append(c)

    # Evaluate detection quality
    detected_classes = set(r["class"] for r in collapse_cands)
    actual_set = set(actual_collapsed)
    true_positives = detected_classes & actual_set
    precision = len(true_positives) / max(len(detected_classes), 1)
    recall = len(true_positives) / max(len(actual_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n=== Unsupervised Collapse Detection Results ===")
    print(f"Actual collapsed classes ({len(actual_collapsed)}): {sorted(actual_collapsed)}")
    print(f"Detected candidates ({len(collapse_cands)}): {sorted(r['class'] for r in collapse_cands)}")
    print(f"True positives: {sorted(true_positives)}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    print(f"\nDetected absorber candidates ({len(absorber_cands)}):")
    for r in absorber_cands[:10]:
        print(f"  class {r['class']}: count_ratio={r['count_ratio']:.2f}, FD={r['fd']:.2f}")

    print(f"\nTop-15 by Collapse Score (5-signal composite):")
    valid_scored = sorted([r for r in fd_results if r.get("collapse_score") is not None],
                          key=lambda x: x.get("collapse_score", 0), reverse=True)
    for r in valid_scored[:15]:
        is_actual = "← COLLAPSED" if r["class"] in actual_set else ""
        md = r.get('margin_drop') or 0
        cd = r.get('conf_drop') or 0
        cr = r.get('count_ratio') or 0
        fd = r.get('fd') or 0
        print(f"  class {r['class']:3d}: score={r.get('collapse_score', 0):.3f}, "
              f"cnt={cr:.2f}, FD={fd:6.1f}, "
              f"margin_drop={md:+.2f}, conf_drop={cd:+.3f} {is_actual}")

    # Save results
    out = {
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "num_classes": num_classes,
        "config": {
            "min_samples": args.min_samples,
            "top_k": args.top_k,
            "score_threshold": args.score_threshold,
            "weights": weights,
        },
        "actual_collapsed": sorted(actual_collapsed),
        "detected_collapse": sorted(r["class"] for r in collapse_cands),
        "detected_absorbers": sorted(r["class"] for r in absorber_cands),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    with open(os.path.join(args.output_dir, "detection_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    fieldnames = ["class", "fd", "ref_count", "tgt_count", "count_ratio", "margin_drop", "conf_drop", "entropy_rise", "fd_zscore", "collapse_score", "status"]
    with open(os.path.join(args.output_dir, "per_class_fd.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(fd_results, key=lambda x: x.get("fd") or 0, reverse=True):
            writer.writerow(r)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
