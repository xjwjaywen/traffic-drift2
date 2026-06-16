"""
Aggregate AL sweep results into a comparison table.

Reads outputs/al_sweep/<dataset>/results_sequential_<method>_<sampler>_seed<N>.json
and prints / writes a per-dataset table with mean ± std AURC across seeds.

Usage:
    python scripts/aggregate_al_sweep.py
"""
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


ROOT = "outputs/al_sweep"


def load_runs(dataset: str):
    """
    Returns list of dicts: {method, sampler, seed, aurc, per_period}
    """
    runs = []
    pattern = os.path.join(ROOT, dataset, "results_sequential_*.json")
    for path in sorted(glob(pattern)):
        with open(path) as f:
            payload = json.load(f)
        results = payload.get("results", payload)  # backward compat
        sampler = payload.get("sampler", "random")
        seed = payload.get("seed", -1)
        # Parse method from filename suffix
        fname = os.path.basename(path)
        suffix = fname[len("results_sequential_"):-len(".json")]
        # suffix = "<method>_<sampler>_seed<N>"
        # method may be one of {tta_tc, knn_labeled, ft_head}; rest is sampler+seed
        method = None
        for m in ("tta_tc", "knn_labeled", "ft_head", "supervised_norm"):
            if suffix.startswith(m + "_"):
                method = m
                break
        if method is None:
            continue
        for method_key, summary in results.items():
            if not isinstance(summary, dict):
                continue
            aurc = summary.get("aurc")
            if aurc is None:
                continue
            runs.append({
                "method": method,
                "sampler": sampler,
                "seed": seed,
                "aurc": aurc,
                "periods": summary.get("periods", {}),
            })
    return runs


def aggregate(runs):
    """
    Group by (method, sampler) and compute mean / std AURC.
    Returns dict[(method, sampler)] -> (mean, std, n)
    """
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["method"], r["sampler"])].append(r["aurc"])
    out = {}
    for k, vs in by_key.items():
        arr = np.array(vs)
        out[k] = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr))
    return out


def print_table(dataset, agg):
    print(f"\n{'='*72}")
    print(f"Dataset: {dataset}")
    print(f"{'='*72}")
    print(f"{'Method':<14} {'Sampler':<18} {'AURC mean':>10} {'± std':>10} {'n':>5}")
    print("-" * 72)
    method_order = ["tta_tc", "knn_labeled", "ft_head", "supervised_norm"]
    sampler_order = ["random", "entropy", "margin", "coreset", "class_balanced"]
    sorted_keys = sorted(
        agg.keys(),
        key=lambda k: (
            method_order.index(k[0]) if k[0] in method_order else 99,
            sampler_order.index(k[1]) if k[1] in sampler_order else 99,
        ),
    )
    for k in sorted_keys:
        mean, std, n = agg[k]
        method, sampler = k
        print(f"{method:<14} {sampler:<18} {mean:>10.4f} {std:>10.4f} {n:>5}")
    print("=" * 72)


def main():
    for dataset in ("quic22", "tls22"):
        path = os.path.join(ROOT, dataset)
        if not os.path.isdir(path):
            print(f"[skip] {path} does not exist")
            continue
        runs = load_runs(dataset)
        if not runs:
            print(f"[skip] no runs in {path}")
            continue
        agg = aggregate(runs)
        print_table(dataset, agg)
        # Save aggregated CSV
        out_csv = os.path.join(path, "aggregate.csv")
        with open(out_csv, "w") as f:
            f.write("method,sampler,aurc_mean,aurc_std,n\n")
            for (method, sampler), (mean, std, n) in sorted(agg.items()):
                f.write(f"{method},{sampler},{mean:.6f},{std:.6f},{n}\n")
        print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
