"""
Step 0.2: Drift-structure temporal stability diagnostic.

Goal: verify whether the drift pattern observed during the *training* era
(e.g., W-40 vs W-41 vs W-42) predicts the drift pattern observed during the
*test* era (e.g., W-45 vs W-46 vs W-47).

Why this matters: GIRM's whole hypothesis is that "drift dimensions you can
observe at training time will continue to be the dominant drift dimensions at
test time." If this is false, the entire research direction collapses.

Metric: Spearman ρ and cosine similarity between two 90-dim drift signatures
(3 channels × 30 positions, flattened):
  - signature_train = average per-(channel, position) drift score across
                      training-era period pairs
  - signature_test  = average per-(channel, position) drift score across
                      test-era period pairs

Decision thresholds (per dataset):
  ρ > 0.7  : stability assumption holds → continue with GIRM
  ρ ∈ 0.3-0.7 : partial stability → method must be robust to drift
                 misspecification
  ρ < 0.3  : assumption fails → GIRM hypothesis is invalid for this dataset

CPU-only (no GPU needed). Runtime: a few minutes per dataset.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from tqdm import tqdm

# Allow imports from the main project's tta_tc package
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))

from tta_tc.data.cesnet_loader import build_sequential_test_loaders


CHANNEL_NAMES = ("size", "direction", "ipt")
NUM_CHANNELS = 3
SEQ_LEN = 30


def collect_period_ppi(loader, max_batches=None):
    """Collect all PPI features from a loader. Returns array shape (N, 3, 30)."""
    chunks = []
    n_batches = 0
    for batch in tqdm(loader, desc="collecting PPI", leave=False):
        ppi = batch["ppi"].cpu().numpy()
        chunks.append(ppi)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if not chunks:
        raise RuntimeError("Empty loader — no batches collected")
    return np.concatenate(chunks, axis=0)


def per_position_drift(reference: np.ndarray, target: np.ndarray):
    """
    Per (channel, position) drift score = KS statistic between reference and
    target distributions of that scalar feature.

    Args:
        reference: (N1, 3, 30) source PPI samples
        target:    (N2, 3, 30) target PPI samples

    Returns:
        ks_map: (3, 30) KS statistic per (channel, position)
        z_map : (3, 30) signed z-score (target_mean - ref_mean) / ref_std
    """
    ks_map = np.zeros((NUM_CHANNELS, SEQ_LEN), dtype=np.float32)
    z_map = np.zeros((NUM_CHANNELS, SEQ_LEN), dtype=np.float32)
    for c in range(NUM_CHANNELS):
        for p in range(SEQ_LEN):
            ref = reference[:, c, p]
            tgt = target[:, c, p]
            ks, _ = stats.ks_2samp(ref, tgt)
            ks_map[c, p] = ks
            mu_ref = ref.mean()
            sd_ref = ref.std() + 1e-8
            z_map[c, p] = (tgt.mean() - mu_ref) / sd_ref
    return ks_map, z_map


def aggregate_drift(reference: np.ndarray, period_data: dict):
    """
    Compute per-(channel, position) drift signatures for each period, all
    against a shared reference period.

    Returns dict: period_name -> {"ks": (3, 30), "z": (3, 30)}
    """
    out = {}
    for period_name, data in period_data.items():
        if data is reference or np.shares_memory(data, reference):
            # Self-comparison gives 0 drift; skip to keep matrices clean
            ks_map = np.zeros((NUM_CHANNELS, SEQ_LEN), dtype=np.float32)
            z_map = np.zeros((NUM_CHANNELS, SEQ_LEN), dtype=np.float32)
        else:
            ks_map, z_map = per_position_drift(reference, data)
        out[period_name] = {"ks": ks_map, "z": z_map}
    return out


def correlate_signatures(sig_a: np.ndarray, sig_b: np.ndarray):
    """
    Compare two (3, 30) signatures by Spearman ρ and cosine similarity.
    Both metrics are computed on the flattened 90-dim vectors.
    """
    a = sig_a.flatten()
    b = sig_b.flatten()
    rho, p_value = stats.spearmanr(a, b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"spearman_rho": float(rho), "spearman_p": float(p_value),
            "cosine": cos}


def top_k_overlap(sig_a: np.ndarray, sig_b: np.ndarray, k: int = 10):
    """
    Jaccard overlap of top-k drifted (channel, position) cells.
    Tells us: do the same locations dominate drift across time?
    """
    a = sig_a.flatten()
    b = sig_b.flatten()
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else 0.0


def run_dataset(dataset: str, data_dir: str, size: str, periods: list,
                train_periods: list, test_periods: list,
                reference_period: str, max_batches: int):
    """Run drift-stability analysis for one dataset."""
    print("=" * 76)
    print(f"Dataset: {dataset}")
    print(f"All periods loaded: {periods}")
    print(f"Reference period:   {reference_period}")
    print(f"Training-era set:   {train_periods}")
    print(f"Test-era set:       {test_periods}")
    print("=" * 76)

    # Load all periods (use sequential test-loader builder; train_period flag
    # is required by cesnet-datazoo but not used for collection itself)
    loader_cfg = {
        "dataset": dataset,
        "data_dir": data_dir,
        "size": size,
        "train_period": reference_period,
        "test_periods": periods,
        "batch_size": 256,
        "num_workers": 4,
    }
    loaders, _ = build_sequential_test_loaders(loader_cfg)

    period_data = {}
    for period_name, loader in loaders:
        print(f"\n[load] {period_name}")
        ppi = collect_period_ppi(loader, max_batches=max_batches)
        period_data[period_name] = ppi
        print(f"  samples={ppi.shape[0]}")

    if reference_period not in period_data:
        raise RuntimeError(
            f"Reference period {reference_period} not in loaded periods")
    reference = period_data[reference_period]

    print(f"\n[compute] per-period drift signatures vs {reference_period}")
    sigs = aggregate_drift(reference, period_data)

    # Average drift signature over training-era periods and test-era periods
    def _avg(period_list, key):
        valid = [p for p in period_list if p in sigs and p != reference_period]
        if not valid:
            raise RuntimeError(f"No valid periods in {period_list}")
        stack = np.stack([sigs[p][key] for p in valid], axis=0)
        return stack.mean(axis=0), valid

    train_ks_avg, train_used = _avg(train_periods, "ks")
    test_ks_avg, test_used = _avg(test_periods, "ks")
    train_z_avg, _ = _avg(train_periods, "z")
    test_z_avg, _ = _avg(test_periods, "z")

    print(f"\n[result] Comparing drift signatures")
    print(f"  Training-era signature  : mean over {train_used}")
    print(f"  Test-era    signature   : mean over {test_used}")

    ks_corr = correlate_signatures(train_ks_avg, test_ks_avg)
    z_corr = correlate_signatures(train_z_avg, test_z_avg)
    overlap_top10 = top_k_overlap(train_ks_avg, test_ks_avg, k=10)
    overlap_top20 = top_k_overlap(train_ks_avg, test_ks_avg, k=20)

    print("\n  --- KS-statistic signature (magnitude of drift) ---")
    print(f"   Spearman ρ = {ks_corr['spearman_rho']:+.4f}  (p={ks_corr['spearman_p']:.2e})")
    print(f"   Cosine sim = {ks_corr['cosine']:+.4f}")
    print(f"   Top-10 overlap (Jaccard) = {overlap_top10:.4f}")
    print(f"   Top-20 overlap (Jaccard) = {overlap_top20:.4f}")

    print("\n  --- Z-score signature (signed direction of drift) ---")
    print(f"   Spearman ρ = {z_corr['spearman_rho']:+.4f}  (p={z_corr['spearman_p']:.2e})")
    print(f"   Cosine sim = {z_corr['cosine']:+.4f}")

    # Per-channel breakdown
    print("\n  --- Per-channel breakdown (KS Spearman ρ) ---")
    for c, name in enumerate(CHANNEL_NAMES):
        per_c = correlate_signatures(train_ks_avg[c:c+1, :], test_ks_avg[c:c+1, :])
        print(f"   channel {c} ({name:10s}): ρ = {per_c['spearman_rho']:+.4f}")

    # Decision
    rho = ks_corr["spearman_rho"]
    if rho > 0.7:
        decision = "PASS — drift structure is temporally stable; GIRM hypothesis valid"
    elif rho > 0.3:
        decision = "PARTIAL — moderate stability; GIRM viable but needs robustness to misspecification"
    else:
        decision = "FAIL — drift structure is NOT temporally stable; GIRM hypothesis invalid"

    print(f"\n  ==> DECISION ({dataset}): {decision}")

    return {
        "dataset": dataset,
        "reference_period": reference_period,
        "train_periods_used": train_used,
        "test_periods_used": test_used,
        "ks_signature_corr": ks_corr,
        "z_signature_corr": z_corr,
        "top10_overlap": overlap_top10,
        "top20_overlap": overlap_top20,
        "per_channel_ks_rho": [
            correlate_signatures(train_ks_avg[c:c+1, :],
                                  test_ks_avg[c:c+1, :])["spearman_rho"]
            for c in range(NUM_CHANNELS)
        ],
        "decision": decision,
        "raw_signatures": {
            "train_ks_avg": train_ks_avg.tolist(),
            "test_ks_avg": test_ks_avg.tolist(),
            "train_z_avg": train_z_avg.tolist(),
            "test_z_avg": test_z_avg.tolist(),
        },
    }


# Default period plans per dataset
DEFAULT_PLANS = {
    "quic22": {
        # All weeks loaded; reference = first; train-era = early; test-era = late
        "periods": [
            "W-2022-44", "W-2022-45", "W-2022-46", "W-2022-47",
        ],
        "reference": "W-2022-44",
        "train_periods": ["W-2022-45"],
        "test_periods": ["W-2022-46", "W-2022-47"],
    },
    "tls22": {
        "periods": [
            "M-2022-3", "M-2022-4", "M-2022-5", "M-2022-6", "M-2022-7",
            "M-2022-8", "M-2022-9", "M-2022-10", "M-2022-11", "M-2022-12",
        ],
        "reference": "M-2022-3",
        "train_periods": ["M-2022-4", "M-2022-5"],
        "test_periods": ["M-2022-10", "M-2022-11", "M-2022-12"],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["quic22", "tls22"],
                        help="Datasets to analyse")
    parser.add_argument("--data-dir-quic", default="./data/quic22")
    parser.add_argument("--data-dir-tls", default="./data/tls22")
    parser.add_argument("--size", default="S")
    parser.add_argument("--max-batches", type=int, default=200,
                        help="Cap batches per period (None = full)")
    parser.add_argument("--output-dir",
                        default=os.path.join(_REPO_ROOT, "TIF_extension",
                                              "outputs", "step0_2"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for dataset in args.datasets:
        if dataset not in DEFAULT_PLANS:
            print(f"[skip] No default plan for dataset {dataset}")
            continue
        plan = DEFAULT_PLANS[dataset]
        data_dir = (args.data_dir_quic if dataset == "quic22"
                    else args.data_dir_tls)
        try:
            res = run_dataset(
                dataset=dataset,
                data_dir=data_dir,
                size=args.size,
                periods=plan["periods"],
                train_periods=plan["train_periods"],
                test_periods=plan["test_periods"],
                reference_period=plan["reference"],
                max_batches=args.max_batches,
            )
            all_results[dataset] = res
            out_path = os.path.join(args.output_dir, f"{dataset}_drift_stability.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\n  Saved: {out_path}")
        except Exception as e:
            print(f"\n[ERROR] {dataset} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset] = {"error": str(e)}

    # Combined summary
    print("\n" + "=" * 76)
    print("SUMMARY (Step 0.2: Drift Structure Temporal Stability)")
    print("=" * 76)
    print(f"{'Dataset':<10} {'Spearman ρ':>12} {'Cosine':>10} {'Top-10':>8} {'Top-20':>8}  Decision")
    for dataset, res in all_results.items():
        if "error" in res:
            print(f"{dataset:<10} ERROR: {res['error']}")
            continue
        rho = res["ks_signature_corr"]["spearman_rho"]
        cos = res["ks_signature_corr"]["cosine"]
        t10 = res["top10_overlap"]
        t20 = res["top20_overlap"]
        decision_short = res["decision"].split("—")[0].strip()
        print(f"{dataset:<10} {rho:>+12.4f} {cos:>+10.4f} {t10:>8.4f} {t20:>8.4f}  {decision_short}")
    print("=" * 76)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved: {summary_path}")


if __name__ == "__main__":
    # cesnet-datazoo on macOS needs 'fork'; on Linux server this is also fine.
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
