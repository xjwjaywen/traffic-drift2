"""
Deployment trigger analysis: when should CARE be activated?

Monitors per-period statistics and determines when collapse occurs,
triggering CARE maintenance. Evaluates trigger mechanisms:
  1. Mean confidence drop
  2. Entropy increase
  3. High-reject rate (low max-prob samples)
  4. Prototype distance increase

For each period M5-M12, computes trigger signals and checks whether
CARE would have been triggered before collapse became severe.

Usage:
    python scripts/deployment_trigger_analysis.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --output-dir outputs/deployment_trigger
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


PERIODS_TLS22 = [f"M-2022-{m}" for m in range(4, 13)]


def compute_signals(model, loader, device, prototypes, valid_mask):
    """Compute deployment monitoring signals for a period."""
    outputs = proto.collect_outputs(model, loader, device, desc="")
    logits = outputs["logits"]
    features = outputs["features"]
    labels = outputs["labels"]

    probs = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    preds = probs.argmax(dim=1).numpy()

    # Prototype distance
    feat_n = F.normalize(features, dim=1)
    proto_n = F.normalize(prototypes, dim=1)
    sims = feat_n @ proto_n.t()
    if valid_mask is not None:
        sims[:, ~valid_mask] = -1e9
    max_sim = sims.max(dim=1).values

    # Reject rate: fraction with max_prob < 0.5
    reject_rate = float((confidence < 0.5).float().mean())

    # Per-class recall for collapse counting
    report = proto.compute_metrics(labels, preds)
    num_collapsed = 0
    for c in range(logits.shape[1]):
        item = report["classification_report"].get(str(c), {})
        support = int(item.get("support", 0))
        recall = float(item.get("recall", 0.0))
        if support > 0 and recall < 0.1:
            num_collapsed += 1

    return {
        "mean_confidence": float(confidence.mean()),
        "std_confidence": float(confidence.std()),
        "mean_entropy": float(entropy.mean()),
        "mean_proto_sim": float(max_sim.mean()),
        "mean_proto_dist": float((1 - max_sim).mean()),
        "reject_rate_05": reject_rate,
        "macro_f1": report["macro_f1"],
        "accuracy": report["accuracy"],
        "num_collapsed": num_collapsed,
        "num_samples": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--periods", default=None,
                        help="Comma-separated periods (default: M-2022-4 to M-2022-12)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    periods = args.periods.split(",") if args.periods else PERIODS_TLS22

    # Build reference prototypes
    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_outputs = proto.collect_outputs(model, ref_loader, device, desc=f"Ref {args.reference_period}")
    prototypes, _, valid_mask = proto.build_prototypes(
        ref_outputs["features"], ref_outputs["labels"], num_classes, 1
    )

    rows = []
    ref_signals = None

    for period in periods:
        print(f"Processing {period}...")
        loader, _ = proto.make_test_loader(eval_cfg, period)
        signals = compute_signals(model, loader, device, prototypes, valid_mask)
        signals["period"] = period

        if ref_signals is None:
            ref_signals = signals.copy()
            signals["confidence_delta"] = 0.0
            signals["entropy_delta"] = 0.0
            signals["proto_dist_delta"] = 0.0
        else:
            signals["confidence_delta"] = signals["mean_confidence"] - ref_signals["mean_confidence"]
            signals["entropy_delta"] = signals["mean_entropy"] - ref_signals["mean_entropy"]
            signals["proto_dist_delta"] = signals["mean_proto_dist"] - ref_signals["mean_proto_dist"]

        # Simple trigger rules
        signals["trigger_confidence"] = signals["confidence_delta"] < -0.05
        signals["trigger_entropy"] = signals["entropy_delta"] > 0.3
        signals["trigger_reject"] = signals["reject_rate_05"] > 0.3
        signals["trigger_any"] = (
            signals["trigger_confidence"]
            or signals["trigger_entropy"]
            or signals["trigger_reject"]
        )

        rows.append(signals)
        print(f"  macro_f1={signals['macro_f1']:.4f} collapsed={signals['num_collapsed']} "
              f"conf_delta={signals['confidence_delta']:+.4f} "
              f"trigger={'YES' if signals['trigger_any'] else 'no'}")

    # Write CSV
    fieldnames = list(rows[0].keys())
    out_path = os.path.join(args.output_dir, "trigger_analysis.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write summary
    summary = {
        "reference_period": args.reference_period,
        "periods": periods,
        "trigger_thresholds": {
            "confidence_delta": -0.05,
            "entropy_delta": 0.3,
            "reject_rate": 0.3,
        },
        "results": [
            {
                "period": r["period"],
                "macro_f1": r["macro_f1"],
                "collapsed": r["num_collapsed"],
                "triggered": r["trigger_any"],
            }
            for r in rows
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")
    print("\n=== Trigger Summary ===")
    for r in rows:
        flag = "TRIGGER" if r["trigger_any"] else "      "
        print(f"  {r['period']} macro={r['macro_f1']:.4f} collapsed={r['num_collapsed']:2d} "
              f"conf={r['confidence_delta']:+.4f} ent={r['entropy_delta']:+.4f} "
              f"reject={r['reject_rate_05']:.3f} [{flag}]")


if __name__ == "__main__":
    main()
