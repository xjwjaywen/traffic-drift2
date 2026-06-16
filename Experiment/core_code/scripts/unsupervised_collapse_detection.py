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


def compute_per_class_fd(ref_features, ref_preds, tgt_features, tgt_preds,
                         num_classes, min_samples=10):
    """Compute Frechet Distance per predicted class (no labels needed)."""
    results = []
    for c in range(num_classes):
        ref_mask = ref_preds == c
        tgt_mask = tgt_preds == c
        ref_n = int(ref_mask.sum())
        tgt_n = int(tgt_mask.sum())

        if ref_n < min_samples or tgt_n < min_samples:
            results.append({
                "class": c,
                "fd": None,
                "ref_count": ref_n,
                "tgt_count": tgt_n,
                "count_ratio": tgt_n / max(ref_n, 1),
                "status": "insufficient_samples",
            })
            continue

        ref_feat = ref_features[ref_mask].numpy()
        tgt_feat = tgt_features[tgt_mask].numpy()

        mu_ref = ref_feat.mean(axis=0)
        mu_tgt = tgt_feat.mean(axis=0)
        sigma_ref = np.cov(ref_feat, rowvar=False) + np.eye(ref_feat.shape[1]) * 1e-6
        sigma_tgt = np.cov(tgt_feat, rowvar=False) + np.eye(tgt_feat.shape[1]) * 1e-6

        fd = frechet_distance(mu_ref, sigma_ref, mu_tgt, sigma_tgt)

        results.append({
            "class": c,
            "fd": fd,
            "ref_count": ref_n,
            "tgt_count": tgt_n,
            "count_ratio": tgt_n / max(ref_n, 1),
            "status": "ok",
        })

    return results


def detect_collapse_candidates(fd_results, top_k=15, count_change_threshold=0.3):
    """Identify collapse candidates from FD scores + count changes.

    A class is a collapse candidate if:
    - Its FD is in the top-K highest (feature distribution shifted)
    - OR its prediction count dropped significantly (being absorbed)

    A class is an absorber candidate if:
    - Its prediction count increased significantly (absorbing others)
    - AND its FD is elevated (feature distribution changed due to incoming victims)
    """
    valid = [r for r in fd_results if r["fd"] is not None]
    if not valid:
        return [], []

    fds = np.array([r["fd"] for r in valid])
    fd_median = np.median(fds)
    fd_mad = np.median(np.abs(fds - fd_median)) + 1e-8

    collapse_candidates = []
    absorber_candidates = []

    for r in valid:
        z_score = (r["fd"] - fd_median) / fd_mad
        r["fd_zscore"] = float(z_score)

        # Collapse: count dropped (being absorbed by others) OR high FD
        if r["count_ratio"] < (1 - count_change_threshold):
            collapse_candidates.append(r)
        elif z_score > 3.0:
            collapse_candidates.append(r)

        # Absorber: count increased (absorbing others)
        if r["count_ratio"] > (1 + count_change_threshold):
            absorber_candidates.append(r)

    collapse_candidates.sort(key=lambda x: x["fd"], reverse=True)
    absorber_candidates.sort(key=lambda x: x["count_ratio"], reverse=True)

    return collapse_candidates[:top_k], absorber_candidates[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--min-samples", type=int, default=10)
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

    # Compute per-class Frechet Distance (using PREDICTIONS, no labels)
    print("\nComputing per-class Frechet Distance (unsupervised)...")
    fd_results = compute_per_class_fd(
        ref_out["features"], ref_preds,
        tgt_out["features"], tgt_preds,
        num_classes, args.min_samples,
    )

    # Detect candidates
    collapse_cands, absorber_cands = detect_collapse_candidates(fd_results)

    # Ground truth: actual collapsed classes (using labels)
    actual_collapsed = []
    for c in range(num_classes):
        mask = tgt_labels == c
        if mask.sum() > 0:
            recall = float((tgt_preds[mask] == c).sum()) / float(mask.sum())
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

    print(f"\nTop-10 by Frechet Distance:")
    valid_sorted = sorted([r for r in fd_results if r["fd"] is not None],
                          key=lambda x: x["fd"], reverse=True)
    for r in valid_sorted[:10]:
        is_actual = "← COLLAPSED" if r["class"] in actual_set else ""
        print(f"  class {r['class']:3d}: FD={r['fd']:8.2f}, "
              f"count_ratio={r['count_ratio']:.2f}, "
              f"z={r.get('fd_zscore', 0):.1f} {is_actual}")

    # Save results
    out = {
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "num_classes": num_classes,
        "actual_collapsed": sorted(actual_collapsed),
        "detected_collapse": sorted(r["class"] for r in collapse_cands),
        "detected_absorbers": sorted(r["class"] for r in absorber_cands),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    with open(os.path.join(args.output_dir, "detection_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    fieldnames = ["class", "fd", "ref_count", "tgt_count", "count_ratio", "fd_zscore", "status"]
    with open(os.path.join(args.output_dir, "per_class_fd.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(fd_results, key=lambda x: x.get("fd") or 0, reverse=True):
            writer.writerow(r)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
