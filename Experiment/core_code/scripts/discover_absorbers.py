"""
Auto-discover absorber and collapse classes from a prior period's predictions.

Instead of using oracle (ground-truth) absorber lists from the target period,
this script uses an earlier "probe" period to build absorber/collapse lists,
simulating a deployable pipeline:

  1. Run static model on probe period (e.g., M-2022-11)
  2. Find collapse classes: recall < threshold
  3. Find absorbers: classes that receive the most false positives from collapse classes
  4. Output class lists that can be fed to collapse_active_maintenance_tls22.py

Usage:
    python scripts/discover_absorbers.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --probe-period M-2022-11 \
      --output-dir outputs/auto_absorber_discovery
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


def discover_from_confusion(labels, preds, num_classes, recall_threshold=0.1, top_k_absorbers=10):
    """Discover collapse and absorber classes from predictions.

    Collapse classes: recall < recall_threshold.
    Absorber classes: top-K classes by total false positives received from collapse classes.
    """
    collapse_classes = []
    per_class_recall = {}

    for c in range(num_classes):
        mask = labels == c
        support = int(mask.sum())
        if support == 0:
            continue
        correct = int((preds[mask] == c).sum())
        recall = correct / support
        per_class_recall[c] = recall
        if recall < recall_threshold:
            collapse_classes.append(c)

    absorber_counter = Counter()
    for c in collapse_classes:
        mask = labels == c
        wrong_preds = preds[mask]
        for p in wrong_preds:
            if p != c:
                absorber_counter[int(p)] += 1

    absorber_classes = [cls for cls, _ in absorber_counter.most_common(top_k_absorbers)]

    return collapse_classes, absorber_classes, per_class_recall, absorber_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-period", required=True,
                        help="Period to use for discovering absorbers (e.g., M-2022-11)")
    parser.add_argument("--recall-threshold", type=float, default=0.1)
    parser.add_argument("--top-k-absorbers", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    print(f"Probe period: {args.probe_period}")
    print(f"Recall threshold: {args.recall_threshold}")

    loader, _ = proto.make_test_loader(eval_cfg, args.probe_period)
    outputs = proto.collect_outputs(model, loader, device, desc=f"Probe {args.probe_period}")
    labels = outputs["labels"]
    preds = outputs["logits"].argmax(dim=1).numpy()

    collapse_classes, absorber_classes, per_class_recall, absorber_counter = discover_from_confusion(
        labels, preds, num_classes, args.recall_threshold, args.top_k_absorbers
    )

    print(f"\nDiscovered {len(collapse_classes)} collapse classes (recall < {args.recall_threshold}):")
    for c in collapse_classes:
        print(f"  class {c}: recall={per_class_recall[c]:.4f}")

    print(f"\nDiscovered {len(absorber_classes)} absorber classes (top FP receivers):")
    for c in absorber_classes:
        print(f"  class {c}: absorbed {absorber_counter[c]} samples from collapse classes")

    result = {
        "probe_period": args.probe_period,
        "recall_threshold": args.recall_threshold,
        "top_k_absorbers": args.top_k_absorbers,
        "collapse_classes": collapse_classes,
        "absorber_classes": absorber_classes,
        "collapse_details": {str(c): per_class_recall[c] for c in collapse_classes},
        "absorber_details": {str(c): absorber_counter[c] for c in absorber_classes},
    }

    out_path = os.path.join(args.output_dir, f"discovered_{args.probe_period.replace('-', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    print(f"\n--- CLI args for CARE ---")
    collapse_str = ",".join(str(c) for c in collapse_classes)
    absorber_str = ",".join(str(c) for c in absorber_classes)
    print(f"  --collapse-classes \"{collapse_str}\"")
    print(f"  --absorber-classes \"{absorber_str}\"")


if __name__ == "__main__":
    main()
