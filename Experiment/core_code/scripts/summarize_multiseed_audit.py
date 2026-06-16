"""
Summarize multi-training-seed CARE experiment results.
Computes mean/std across training seeds (each already aggregated over FT seeds).

Usage:
    python scripts/summarize_multiseed_audit.py \
        --base-dir outputs/multiseed_audit \
        --num-train-seeds 3
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_static_baselines(base_dir, num_train_seeds):
    """Load static baseline metrics per training seed."""
    results = []
    for ts in range(num_train_seeds):
        path = os.path.join(base_dir, f"trainseed{ts}", "static",
                            "baselines_group_metrics.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("method", "").lower() in ("static", ""):
                    results.append({
                        "train_seed": ts,
                        "overall_macro_f1": float(row.get("overall_macro_f1", 0)),
                        "collapse_macro_f1": float(row.get("collapse_macro_f1",
                                                           row.get("bad_macro_f1", 0))),
                        "stable_macro_f1": float(row.get("stable_macro_f1", 0)),
                    })
                    break
    return results


def load_care_aggregated(base_dir, num_train_seeds):
    """Load per-training-seed aggregated CARE results."""
    results = []
    for ts in range(num_train_seeds):
        path = os.path.join(base_dir, f"trainseed{ts}", "care",
                            "aggregated_mean_std.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                row["train_seed"] = ts
                results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--num-train-seeds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Training-Seed Audit Summary")
    print("=" * 80)

    # Static baselines across training seeds
    statics = load_static_baselines(args.base_dir, args.num_train_seeds)
    if statics:
        macro_vals = [s["overall_macro_f1"] for s in statics]
        collapse_vals = [s["collapse_macro_f1"] for s in statics]
        stable_vals = [s["stable_macro_f1"] for s in statics]

        print(f"\nStatic baseline across {len(statics)} training seeds:")
        print(f"  Overall macro-F1:  {np.mean(macro_vals):.4f} ± {np.std(macro_vals, ddof=1):.4f}")
        print(f"  Collapse macro-F1: {np.mean(collapse_vals):.4f} ± {np.std(collapse_vals, ddof=1):.4f}")
        print(f"  Stable macro-F1:   {np.mean(stable_vals):.4f} ± {np.std(stable_vals, ddof=1):.4f}")
        print(f"  Per-seed: {[f'{v:.4f}' for v in macro_vals]}")
    else:
        print("\nNo static baseline results found.")

    # CARE results across training seeds
    care = load_care_aggregated(args.base_dir, args.num_train_seeds)
    if care:
        grouped = defaultdict(list)
        for row in care:
            key = (row.get("strategy", ""), row.get("budget", ""))
            grouped[key].append(row)

        print(f"\nCARE results across training seeds:")
        print(f"{'Strategy':<22} {'Budget':>6}  {'Overall F1':>18}  {'Collapse F1':>18}  {'Stable F1':>18}")
        print("-" * 90)

        for (strategy, budget), rows in sorted(grouped.items()):
            macro_means = []
            collapse_means = []
            stable_means = []
            for r in rows:
                v = r.get("strict_overall_macro_f1_mean", "")
                if v:
                    macro_means.append(float(v))
                v = r.get("strict_bad_macro_f1_mean", "")
                if v:
                    collapse_means.append(float(v))
                v = r.get("strict_stable_macro_f1_mean", "")
                if v:
                    stable_means.append(float(v))

            if macro_means:
                mm, ms = np.mean(macro_means), np.std(macro_means, ddof=1) if len(macro_means) > 1 else 0
                cm, cs = np.mean(collapse_means), np.std(collapse_means, ddof=1) if len(collapse_means) > 1 else 0
                sm, ss = np.mean(stable_means), np.std(stable_means, ddof=1) if len(stable_means) > 1 else 0
                print(f"{strategy:<22} {budget:>6}  {mm:.4f}±{ms:.4f}      {cm:.4f}±{cs:.4f}      {sm:.4f}±{ss:.4f}")
    else:
        print("\nNo CARE results found.")

    # Save summary JSON
    summary = {
        "static_baselines": statics,
        "num_training_seeds": args.num_train_seeds,
    }
    out_path = os.path.join(args.base_dir, "multiseed_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
