"""
DT-TTA Step 1: Verify drift topology is stable across test periods.

For each test period, compute the per-(layer, channel) drift score using the
saved source stats, and classify topology as focal / diffuse / mixed via the
Gini coefficient. Output a per-period topology table.

Pass condition: same protocol shows the SAME topology label across periods.
"""
import argparse
import json
import os
import sys
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "DT_TTA"))

from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.utils.config import load_config
from methods.topology import (
    collect_groupnorm_input_stats,
    compute_channel_drift_scores,
    classify_topology,
    gini_coefficient,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-stats", required=True,
                        help="Path to .pt produced by compute_source_stats.py")
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--focal-thresh", type=float, default=0.6)
    parser.add_argument("--diffuse-thresh", type=float, default=0.3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["data"]["dataset"]
    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "DT_TTA",
                                        "outputs", "step1_topology")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]
    train_cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(train_cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load source stats
    raw = torch.load(args.source_stats, map_location="cpu", weights_only=False)
    source_stats = {name: {"mean": v["mean"].numpy(),
                            "var": v["var"].numpy(),
                            "n": int(v["n"])}
                    for name, v in raw.items()}
    print(f"Loaded source stats: {list(source_stats.keys())}")

    # Per-test-period diagnosis
    loaders, _ = build_sequential_test_loaders(cfg["data"])

    rows = []
    print("\n" + "=" * 84)
    print(f"Drift topology per test period — {dataset}")
    print("=" * 84)
    print(f"{'Period':<14} {'Overall Gini':>13} {'Topology':>12}  Per-layer Gini")
    print("-" * 84)
    for period_name, loader in loaders:
        target_stats = collect_groupnorm_input_stats(
            model, loader, device, max_batches=args.max_batches)
        scores = compute_channel_drift_scores(source_stats, target_stats)
        label, gini_per_layer, overall = classify_topology(
            scores, focal_gini_threshold=args.focal_thresh,
            diffuse_gini_threshold=args.diffuse_thresh)
        per_layer_str = ", ".join(f"{n.split('.')[-1]}={g:.2f}"
                                   for n, g in gini_per_layer.items())
        print(f"{period_name:<14} {overall:>13.4f} {label:>12}  {per_layer_str}")
        rows.append({
            "period": period_name,
            "overall_gini": overall,
            "topology": label,
            "gini_per_layer": gini_per_layer,
            "drift_scores_summary": {
                name: {
                    "max": float(np.max(s)),
                    "mean": float(np.mean(s)),
                    "top5_indices": np.argsort(s)[-5:].tolist(),
                }
                for name, s in scores.items()
            },
        })
    print("=" * 84)

    # Pass/fail decision
    labels = [r["topology"] for r in rows]
    if len(set(labels)) == 1:
        decision = f"PASS — all {len(labels)} periods classified as {labels[0]}"
    else:
        decision = f"VARIES — periods classified as: {dict(zip([r['period'] for r in rows], labels))}"
    print(f"\nDecision: {decision}")

    out_file = os.path.join(args.output_dir, f"{dataset}_topology_stability.json")
    with open(out_file, "w") as f:
        json.dump({"dataset": dataset, "decision": decision, "rows": rows}, f,
                  indent=2, default=str)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
