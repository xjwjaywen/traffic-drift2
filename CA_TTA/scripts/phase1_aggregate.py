"""
CA-TTA Phase 1 aggregation (multi-seed, multi-metric).

Reads outputs/phase1_adapt_then_certify/<dataset>_<method>_sigma<S>[_<suffix>].json

Aggregates seeds (when present) into mean ± std on:
  - clean accuracy
  - smoothed accuracy
  - certified accuracy at each radius

Compares CA-TTA against all baselines on certified accuracy.
"""
import json
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "outputs",
                                      "phase1_adapt_then_certify"))


def load_runs(dataset: str):
    """
    Returns list of summary dicts. Each summary has its own method/seed/etc.
    """
    runs = []
    pattern = os.path.join(ROOT, f"{dataset}_*_sigma*.json")
    for path in sorted(glob(pattern)):
        with open(path) as f:
            payload = json.load(f)
        if payload.get("method") is None:
            continue
        runs.append(payload)
    return runs


def _key(run):
    """A canonical method-key combining method, lambda, loss type, sigma."""
    parts = [run["method"]]
    if run["method"] == "ca_tta":
        parts.append(f"lam{run.get('ca_lambda', '?')}")
        parts.append(f"loss-{run.get('ca_loss_type', '?')}")
        parts.append(f"oc{int(run.get('ca_only_correct', True))}")
    parts.append(f"s{run.get('sigma', '?')}")
    return "|".join(parts)


def aggregate_by_method(runs):
    """
    Group runs by method-key (cross-seed). Returns dict[key] -> aggregated.
    Aggregation: mean and std of each metric across seeds.
    """
    grouped = defaultdict(list)
    for r in runs:
        grouped[_key(r)].append(r)

    out = {}
    for key, rs in grouped.items():
        # All runs in a group share the same radii
        radii = rs[0]["radii"]

        # collect per-period metric arrays then average
        # (Phase 1 first-round JSON files lack clean/smoothed acc — fall
        # back to NaN so we can still aggregate cert acc.)
        per_seed = []
        for r in rs:
            cells = []
            cleans, smoothes = [], []
            cert_per_r = {rad: [] for rad in radii}
            for p_data in r["periods"].values():
                cleans.append(p_data.get("clean_accuracy", float("nan")))
                smoothes.append(p_data.get("smoothed_accuracy", float("nan")))
                for rad in radii:
                    v = p_data["certified_accuracy"].get(str(rad),
                          p_data["certified_accuracy"].get(rad))
                    if v is not None:
                        cert_per_r[rad].append(v)
            per_seed.append({
                "clean": float(np.nanmean(cleans)) if cleans else float("nan"),
                "smoothed": float(np.nanmean(smoothes)) if smoothes else float("nan"),
                "cert": {rad: (np.mean(cert_per_r[rad]) if cert_per_r[rad] else float("nan"))
                         for rad in radii},
                "cert_avg": float(np.mean([np.mean(cert_per_r[rad])
                                            for rad in radii
                                            if cert_per_r[rad]])) if any(cert_per_r.values()) else float("nan"),
            })

        n_seeds = len(per_seed)
        out[key] = {
            "n_seeds": n_seeds,
            "radii": radii,
            "method": rs[0]["method"],
            "ca_lambda": rs[0].get("ca_lambda"),
            "ca_loss_type": rs[0].get("ca_loss_type"),
            "clean_mean": float(np.mean([s["clean"] for s in per_seed])),
            "clean_std": float(np.std([s["clean"] for s in per_seed])),
            "smoothed_mean": float(np.mean([s["smoothed"] for s in per_seed])),
            "smoothed_std": float(np.std([s["smoothed"] for s in per_seed])),
            "cert_avg_mean": float(np.mean([s["cert_avg"] for s in per_seed])),
            "cert_avg_std": float(np.std([s["cert_avg"] for s in per_seed])),
            "cert_per_r_mean": {
                rad: float(np.mean([s["cert"][rad] for s in per_seed]))
                for rad in radii},
            "cert_per_r_std": {
                rad: float(np.std([s["cert"][rad] for s in per_seed]))
                for rad in radii},
        }
    return out


def print_dataset(dataset, agg):
    print("\n" + "#" * 100)
    print(f"# Dataset: {dataset}")
    print("#" * 100)
    if not agg:
        print("(no runs)")
        return

    # Get all unique radii across keys
    all_radii = sorted(set(r for k in agg.values() for r in k["radii"]))

    print(f"\n{'Method-key':<48} {'seeds':>5} {'clean':>14} {'smooth':>14} {'cert avg':>14}",
          end="")
    for r in all_radii:
        print(f"  {f'r={r:.2f}':>12}", end="")
    print()
    print("-" * (48 + 5 + 14 * 3 + 14 * len(all_radii)))

    # Sort by cert_avg_mean descending
    sorted_keys = sorted(agg.keys(),
                          key=lambda k: -agg[k]["cert_avg_mean"])
    def _fmt(v, std=None, w=6, p=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return " " * (w + 6)
        if std is None or (isinstance(std, float) and np.isnan(std)):
            return f"{v:.{p}f}      "
        return f"{v:.{p}f}±{std:.3f}"

    for key in sorted_keys:
        a = agg[key]
        row = f"{key:<48} {a['n_seeds']:>5}"
        row += f"  {_fmt(a['clean_mean'], a['clean_std'])}"
        row += f"  {_fmt(a['smoothed_mean'], a['smoothed_std'])}"
        row += f"  {_fmt(a['cert_avg_mean'], a['cert_avg_std'])}"
        for r in all_radii:
            v_m = a["cert_per_r_mean"].get(r)
            v_s = a["cert_per_r_std"].get(r)
            if v_m is not None and not (isinstance(v_m, float) and np.isnan(v_m)):
                row += f"  {v_m:.4f}±{v_s:.2f}"
            else:
                row += "  " + " " * 10
        print(row)
    print("-" * (48 + 5 + 14 * 3 + 14 * len(all_radii)))


def verify_ca_tta_dominates(dataset, agg):
    """Find best CA-TTA config and check if it beats all non-CA baselines."""
    ca_keys = [k for k in agg if agg[k]["method"] == "ca_tta"]
    other_keys = [k for k in agg if agg[k]["method"] != "ca_tta"]
    if not ca_keys or not other_keys:
        return
    best_ca = max(ca_keys, key=lambda k: agg[k]["cert_avg_mean"])
    print(f"\n[Verdict — {dataset}]")
    print(f"Best CA-TTA: {best_ca}  (cert avg = {agg[best_ca]['cert_avg_mean']:.4f}"
          f" ± {agg[best_ca]['cert_avg_std']:.4f})")
    print(f"\n  Baseline comparisons (delta = best_ca - baseline):")
    all_dominate = True
    for k in other_keys:
        delta = agg[best_ca]["cert_avg_mean"] - agg[k]["cert_avg_mean"]
        # Significance: delta > 2 * sqrt(std_ca^2 + std_baseline^2)
        sig_threshold = 2.0 * np.sqrt(
            agg[best_ca]["cert_avg_std"] ** 2 + agg[k]["cert_avg_std"] ** 2)
        sig = abs(delta) > sig_threshold
        marker = "OK" if delta > 0 else "FAIL"
        sig_marker = " (sig)" if sig else " (~noise)"
        print(f"    vs {k:<45} delta = {delta:+.4f}  {marker}{sig_marker}")
        if delta <= 0:
            all_dominate = False
    if all_dominate:
        print(f"\n  ==> Best CA-TTA dominates all baselines on {dataset}")
    else:
        print(f"\n  ==> Mixed result on {dataset}")


def main():
    for dataset in ("quic22", "tls22"):
        runs = load_runs(dataset)
        if not runs:
            print(f"\n# Dataset {dataset}: no runs found")
            continue
        agg = aggregate_by_method(runs)
        print_dataset(dataset, agg)
        verify_ca_tta_dominates(dataset, agg)


if __name__ == "__main__":
    main()
