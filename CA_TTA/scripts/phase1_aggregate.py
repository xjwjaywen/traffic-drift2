"""
CA-TTA Phase 1 aggregation.

Reads outputs/phase1_adapt_then_certify/<dataset>_<method>_sigma<S>.json
Compares all methods on:
  - Average certified accuracy (across radii × periods)
  - Per-radius certified accuracy table
  - Improvement of CA-TTA over each baseline

Decision criteria:
  CA-TTA wins if its avg certified accuracy is higher than ALL baselines
  by at least 0.005, on at least one of the two datasets.
"""
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "outputs",
                                      "phase1_adapt_then_certify"))


def load_runs(dataset: str):
    """Returns dict[method] -> summary dict."""
    out = {}
    pattern = os.path.join(ROOT, f"{dataset}_*_sigma*.json")
    for path in sorted(glob(pattern)):
        with open(path) as f:
            payload = json.load(f)
        method = payload.get("method")
        if method is None:
            continue
        # If the same method has multiple files (e.g., different ca_lambda),
        # tag them with a suffix from filename
        fname = os.path.basename(path)
        # Format: <dataset>_<method>_sigma<S>[_<suffix>].json
        body = fname[len(dataset) + 1:-len(".json")]
        # body = "<method>_sigma<S>" or "<method>_sigma<S>_<suffix>"
        parts = body.split("_sigma")
        if len(parts) >= 2:
            tail = parts[1]
            sub_parts = tail.split("_", 1)
            sigma_str = sub_parts[0]
            suffix = sub_parts[1] if len(sub_parts) > 1 else ""
            key = f"{method}@s{sigma_str}" + (f"_{suffix}" if suffix else "")
        else:
            key = method
        out[key] = payload
    return out


def avg_cert_acc(summary):
    """Mean certified accuracy across all (period, radius) cells."""
    radii = summary.get("radii", [])
    vals = []
    for p_data in summary["periods"].values():
        for r in radii:
            v = p_data["certified_accuracy"].get(str(r),
                p_data["certified_accuracy"].get(r))
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def per_radius_avg(summary):
    """For each radius, mean cert acc across periods."""
    radii = summary.get("radii", [])
    out = {}
    for r in radii:
        vals = []
        for p_data in summary["periods"].values():
            v = p_data["certified_accuracy"].get(str(r),
                p_data["certified_accuracy"].get(r))
            if v is not None:
                vals.append(v)
        out[r] = float(np.mean(vals)) if vals else 0.0
    return out


def print_dataset(dataset, runs):
    print("\n" + "#" * 80)
    print(f"# Dataset: {dataset}")
    print("#" * 80)

    if not runs:
        print("(no runs found)")
        return

    # Build a combined table
    all_radii = sorted(set(
        r for s in runs.values() for r in s.get("radii", [])))

    print(f"\n{'Method':<28} {'avg':>8}" +
          "".join(f"  r={r:.2f}".rjust(10) for r in all_radii))
    print("-" * (28 + 8 + 10 * len(all_radii)))
    for method, summary in sorted(runs.items()):
        avg = avg_cert_acc(summary)
        per_r = per_radius_avg(summary)
        row = f"{method:<28} {avg:>8.4f}"
        for r in all_radii:
            v = per_r.get(r, None)
            row += f"  {v:>8.4f}" if v is not None else "  " + " " * 8
        print(row)
    print("-" * (28 + 8 + 10 * len(all_radii)))

    # Verdict: does CA-TTA win?
    ca_keys = [k for k in runs if "ca_tta" in k]
    if not ca_keys:
        print("\n(no CA-TTA runs to compare)")
        return
    print(f"\n[Verdict — {dataset}]")
    for ca_key in ca_keys:
        ca_avg = avg_cert_acc(runs[ca_key])
        print(f"\n{ca_key}: avg cert acc = {ca_avg:.4f}")
        wins = []
        losses = []
        for other, summary in runs.items():
            if other == ca_key or "ca_tta" in other:
                continue
            other_avg = avg_cert_acc(summary)
            delta = ca_avg - other_avg
            if delta >= 0.005:
                wins.append((other, delta))
            elif delta <= -0.005:
                losses.append((other, delta))
        for o, d in wins:
            print(f"  beats {o:<28} (delta = {d:+.4f})")
        for o, d in losses:
            print(f"  loses to {o:<28} (delta = {d:+.4f})")
        if not losses and wins:
            print(f"  ==> {ca_key} dominates all baselines on {dataset}")


def main():
    for dataset in ("quic22", "tls22"):
        runs = load_runs(dataset)
        print_dataset(dataset, runs)


if __name__ == "__main__":
    main()
