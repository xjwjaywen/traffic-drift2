"""
Aggregate Step 2 sweep results into a comparison table per dataset.

Reads outputs/step2_sweep/<dataset>/results_sequential_<method>_seed<N>.json
and prints / writes mean ± std AURC across 3 seeds for each method.

Decision criteria (DT-TTA framework verification):
  QUIC22: focal_strategy > diffuse_strategy by ≥ 0.005 → focal arm valid
  TLS22 : diffuse_strategy > focal_strategy by ≥ 0.005 → diffuse arm valid
  If both → DT-TTA framework verified
"""
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "outputs", "step2_sweep"))


def load_runs(dataset: str):
    runs = []
    pattern = os.path.join(ROOT, dataset, "results_sequential_*.json")
    for path in sorted(glob(pattern)):
        with open(path) as f:
            payload = json.load(f)
        results = payload.get("results", payload)
        seed = payload.get("seed", -1)
        fname = os.path.basename(path)
        suffix = fname[len("results_sequential_"):-len(".json")]
        method = None
        for m in ("static", "ft_head", "supervised_norm",
                  "selective_norm", "focal_strategy", "diffuse_strategy"):
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
            runs.append({"method": method, "seed": seed, "aurc": aurc,
                         "periods": summary.get("periods", {})})
    return runs


def aggregate(runs):
    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r["aurc"])
    return {m: (float(np.mean(vs)), float(np.std(vs)), len(vs))
            for m, vs in by_method.items()}


def print_table(dataset, agg):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"{'Method':<22} {'AURC mean':>10} {'± std':>10} {'n':>4}")
    print("-" * 60)
    method_order = ["static", "ft_head", "supervised_norm",
                    "selective_norm", "focal_strategy", "diffuse_strategy"]
    for m in method_order:
        if m in agg:
            mean, std, n = agg[m]
            print(f"{m:<22} {mean:>10.4f} {std:>10.4f} {n:>4}")
    print("=" * 60)


def verify_framework(quic_agg, tls_agg, gap_threshold=0.005):
    print("\n" + "=" * 60)
    print("DT-TTA framework verification")
    print("=" * 60)
    if "focal_strategy" in quic_agg and "diffuse_strategy" in quic_agg:
        gap_q = quic_agg["focal_strategy"][0] - quic_agg["diffuse_strategy"][0]
        ok_q = gap_q >= gap_threshold
        print(f"QUIC22: focal_strategy - diffuse_strategy = {gap_q:+.4f}  "
              f"{'OK focal arm valid' if ok_q else 'FAIL focal arm not better'}")
    else:
        gap_q, ok_q = None, False
        print("QUIC22: missing focal/diffuse data")

    if "focal_strategy" in tls_agg and "diffuse_strategy" in tls_agg:
        gap_t = tls_agg["diffuse_strategy"][0] - tls_agg["focal_strategy"][0]
        ok_t = gap_t >= gap_threshold
        print(f"TLS22 : diffuse_strategy - focal_strategy = {gap_t:+.4f}  "
              f"{'OK diffuse arm valid' if ok_t else 'FAIL diffuse arm not better'}")
    else:
        gap_t, ok_t = None, False
        print("TLS22 : missing focal/diffuse data")

    if ok_q and ok_t:
        verdict = "FRAMEWORK VERIFIED - proceed to full paper"
    elif ok_q or ok_t:
        verdict = "PARTIAL - only one arm valid; rethink the other"
    else:
        verdict = "FAIL - neither topology-conditioned strategy beats its mismatch"
    print(f"\nVerdict: {verdict}")
    print("=" * 60)


def main():
    all_agg = {}
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
        all_agg[dataset] = agg
        print_table(dataset, agg)
        out_csv = os.path.join(path, "aggregate.csv")
        with open(out_csv, "w") as f:
            f.write("method,aurc_mean,aurc_std,n\n")
            for m, (mean, std, n) in agg.items():
                f.write(f"{m},{mean:.6f},{std:.6f},{n}\n")
        print(f"Saved CSV: {out_csv}")

    if "quic22" in all_agg and "tls22" in all_agg:
        verify_framework(all_agg["quic22"], all_agg["tls22"])


if __name__ == "__main__":
    main()
