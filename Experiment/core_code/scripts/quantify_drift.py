"""
Quantify feature drift and correlate it with temporal performance decay.

For each test period, this script compares source-period validation PPI
distributions against test-period PPI distributions and computes:
  - per-channel mean/sum KS statistics
  - per-channel count of KS-drifted packet positions
  - per-channel and segment-wise 1D Wasserstein distances
  - Pearson/Spearman correlation with macro-F1 and macro-F1 drop

Usage from Experiment/core_code/:
    python scripts/quantify_drift.py --config configs/eval_tls22.yaml
    python scripts/quantify_drift.py --config configs/eval_quic22.yaml --max-batches 200
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from scipy import stats


CHANNELS = [
    ("size", 0),
    ("direction", 1),
    ("ipt", 2),
]

SEGMENTS = {
    "front_0_9": range(0, 10),
    "middle_10_19": range(10, 20),
    "tail_20_29": range(20, 30),
    "early_0_4": range(0, 5),
    "late_24_29": range(24, 30),
}


def collect_ppi(loader, max_batches=None):
    """Collect PPI arrays as np.ndarray with shape (N, 3, 30)."""
    chunks = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        chunks.append(batch["ppi"].numpy().astype(np.float32, copy=False))
    if not chunks:
        raise RuntimeError("No batches collected. Check dataloader and max_batches.")
    return np.concatenate(chunks, axis=0)


def channel_position_metrics(src_ppi, test_ppi, ks_threshold, pvalue_threshold):
    """Compute per-channel, per-position KS and W1 metrics."""
    result = {}
    for channel_name, channel_idx in CHANNELS:
        src = src_ppi[:, channel_idx, :]
        test = test_ppi[:, channel_idx, :]

        ks_stats = []
        ks_pvals = []
        w1 = []
        for pos in range(src.shape[1]):
            ks, pval = stats.ks_2samp(src[:, pos], test[:, pos])
            ks_stats.append(float(ks))
            ks_pvals.append(float(pval))
            w1.append(float(stats.wasserstein_distance(src[:, pos], test[:, pos])))

        ks_arr = np.asarray(ks_stats)
        pval_arr = np.asarray(ks_pvals)
        w1_arr = np.asarray(w1)
        drifted = np.where((ks_arr > ks_threshold) & (pval_arr < pvalue_threshold))[0]

        segment_metrics = {}
        for segment_name, indices in SEGMENTS.items():
            idx = list(indices)
            segment_metrics[segment_name] = {
                "sum_ks": float(ks_arr[idx].sum()),
                "mean_ks": float(ks_arr[idx].mean()),
                "sum_w1": float(w1_arr[idx].sum()),
                "mean_w1": float(w1_arr[idx].mean()),
                "drifted_count": int(np.isin(idx, drifted).sum()),
            }

        result[channel_name] = {
            "ks_by_position": ks_stats,
            "ks_pvalue_by_position": ks_pvals,
            "w1_by_position": w1,
            "drifted_positions": drifted.astype(int).tolist(),
            "drifted_count": int(len(drifted)),
            "mean_ks": float(ks_arr.mean()),
            "sum_ks": float(ks_arr.sum()),
            "max_ks": float(ks_arr.max()),
            "mean_w1": float(w1_arr.mean()),
            "sum_w1": float(w1_arr.sum()),
            "max_w1": float(w1_arr.max()),
            "segments": segment_metrics,
        }
    return result


def flatten_period_metrics(period_name, period_metrics, perf_metrics):
    """Flatten nested period metrics into one CSV row."""
    row = {"period": period_name}
    if perf_metrics:
        row.update(perf_metrics)

    total_sum_ks = 0.0
    total_sum_w1 = 0.0
    total_drifted = 0
    for channel_name, _ in CHANNELS:
        ch = period_metrics[channel_name]
        prefix = channel_name
        row[f"{prefix}_mean_ks"] = ch["mean_ks"]
        row[f"{prefix}_sum_ks"] = ch["sum_ks"]
        row[f"{prefix}_drifted_count"] = ch["drifted_count"]
        row[f"{prefix}_mean_w1"] = ch["mean_w1"]
        row[f"{prefix}_sum_w1"] = ch["sum_w1"]

        total_sum_ks += ch["sum_ks"]
        total_sum_w1 += ch["sum_w1"]
        total_drifted += ch["drifted_count"]

        for segment_name, seg in ch["segments"].items():
            row[f"{prefix}_{segment_name}_sum_w1"] = seg["sum_w1"]
            row[f"{prefix}_{segment_name}_sum_ks"] = seg["sum_ks"]
            row[f"{prefix}_{segment_name}_drifted_count"] = seg["drifted_count"]

    row["total_sum_ks"] = total_sum_ks
    row["total_mean_ks"] = total_sum_ks / 90.0
    row["total_sum_w1"] = total_sum_w1
    row["total_mean_w1"] = total_sum_w1 / 90.0
    row["total_drifted_count"] = total_drifted
    return row


def load_performance(results_path, method):
    if not results_path or not os.path.exists(results_path):
        return {}
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    method_result = results.get(method, {})
    periods = method_result.get("periods", {})
    if not periods:
        return {}

    ordered_periods = list(periods.keys())
    first_f1 = periods[ordered_periods[0]].get("macro_f1")
    perf = {}
    for period, metrics in periods.items():
        macro_f1 = metrics.get("macro_f1")
        accuracy = metrics.get("accuracy")
        perf[period] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_f1_drop_from_first": (
                float(first_f1 - macro_f1)
                if first_f1 is not None and macro_f1 is not None
                else None
            ),
        }
    return perf


def safe_corr(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
    ]
    if len(pairs) < 3:
        return None
    x_arr = np.asarray([p[0] for p in pairs])
    y_arr = np.asarray([p[1] for p in pairs])
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    pearson = stats.pearsonr(x_arr, y_arr)
    spearman = stats.spearmanr(x_arr, y_arr)
    return {
        "n": int(len(pairs)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def compute_correlations(rows):
    targets = ["macro_f1", "macro_f1_drop_from_first", "accuracy"]
    predictors = [
        "total_sum_ks",
        "total_drifted_count",
        "total_sum_w1",
        "size_sum_ks",
        "direction_sum_ks",
        "ipt_sum_ks",
        "size_drifted_count",
        "direction_drifted_count",
        "ipt_drifted_count",
        "size_sum_w1",
        "direction_sum_w1",
        "ipt_sum_w1",
        "ipt_front_0_9_sum_w1",
        "ipt_tail_20_29_sum_w1",
        "direction_front_0_9_sum_w1",
        "direction_tail_20_29_sum_w1",
        "size_front_0_9_sum_w1",
        "size_tail_20_29_sum_w1",
    ]

    correlations = {}
    for predictor in predictors:
        correlations[predictor] = {}
        xs = [row.get(predictor) for row in rows]
        for target in targets:
            ys = [row.get(target) for row in rows]
            corr = safe_corr(xs, ys)
            if corr is not None:
                correlations[predictor][target] = corr
    return correlations


def infer_results_path(cfg, config_path, explicit_path):
    if explicit_path:
        return explicit_path
    output_dir = cfg.get("output_dir")
    if output_dir:
        candidate = os.path.join(output_dir, "results_sequential.json")
        if os.path.exists(candidate):
            return candidate
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    candidate = os.path.join("outputs", config_name.replace("eval_", "eval_"), "results_sequential.json")
    return candidate if os.path.exists(candidate) else None


def save_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, correlations):
    print("\n=== Drift summary ===")
    header = (
        f"{'Period':<12} {'F1':>8} {'F1drop':>8} "
        f"{'KS(total)':>10} {'#drift':>8} {'W1(total)':>12} "
        f"{'W1_size':>10} {'W1_dir':>10} {'W1_ipt':>10}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['period']:<12} "
            f"{row.get('macro_f1', float('nan')):>8.4f} "
            f"{row.get('macro_f1_drop_from_first', float('nan')):>8.4f} "
            f"{row['total_sum_ks']:>10.4f} "
            f"{row['total_drifted_count']:>8d} "
            f"{row['total_sum_w1']:>12.4f} "
            f"{row['size_sum_w1']:>10.4f} "
            f"{row['direction_sum_w1']:>10.4f} "
            f"{row['ipt_sum_w1']:>10.4f}"
        )

    print("\n=== Strongest correlations with macro_f1_drop_from_first ===")
    scored = []
    for predictor, target_map in correlations.items():
        corr = target_map.get("macro_f1_drop_from_first")
        if corr is not None:
            scored.append((abs(corr["spearman_r"]), predictor, corr))
    for _, predictor, corr in sorted(scored, reverse=True)[:10]:
        print(
            f"{predictor:<32} "
            f"Spearman={corr['spearman_r']:+.3f} (p={corr['spearman_p']:.3g}), "
            f"Pearson={corr['pearson_r']:+.3f} (p={corr['pearson_p']:.3g})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Evaluation config YAML")
    parser.add_argument("--results", default=None, help="Sequential results JSON; defaults to config output_dir")
    parser.add_argument("--method", default="static", help="Method in results JSON to correlate against")
    parser.add_argument("--output-dir", default=None, help="Directory for drift metrics")
    parser.add_argument("--max-batches", type=int, default=200, help="Max batches per period; set <=0 for all")
    parser.add_argument("--ks-threshold", type=float, default=0.05)
    parser.add_argument("--pvalue-threshold", type=float, default=0.001)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_batches = None if args.max_batches is not None and args.max_batches <= 0 else args.max_batches
    data_cfg = cfg["data"]
    results_path = infer_results_path(cfg, args.config, args.results)
    perf_by_period = load_performance(results_path, args.method)

    if results_path:
        print(f"Performance results: {results_path} (method={args.method})")
    else:
        print("Performance results: not found; correlations will be skipped.")

    from tta_tc.data.cesnet_loader import build_dataloaders, build_sequential_test_loaders

    print("Loading source period data...")
    _, val_loader, _, _ = build_dataloaders(data_cfg)
    source_ppi = collect_ppi(val_loader, max_batches=max_batches)
    print(f"Source period: {data_cfg['train_period']}, samples: {source_ppi.shape[0]}")

    print("Loading sequential test periods...")
    test_loaders, _ = build_sequential_test_loaders(data_cfg)

    period_metrics = {}
    rows = []
    for period_name, loader in test_loaders:
        print(f"\nComputing drift metrics for {period_name}...")
        test_ppi = collect_ppi(loader, max_batches=max_batches)
        metrics = channel_position_metrics(
            source_ppi,
            test_ppi,
            ks_threshold=args.ks_threshold,
            pvalue_threshold=args.pvalue_threshold,
        )
        period_metrics[period_name] = metrics
        rows.append(flatten_period_metrics(period_name, metrics, perf_by_period.get(period_name, {})))

    correlations = compute_correlations(rows)

    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = args.output_dir or os.path.join("outputs", "drift_quantification", config_stem)
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "config": args.config,
        "results_path": results_path,
        "method": args.method,
        "source_period": data_cfg["train_period"],
        "test_periods": [p for p, _ in test_loaders],
        "max_batches": max_batches,
        "ks_threshold": args.ks_threshold,
        "pvalue_threshold": args.pvalue_threshold,
        "period_metrics": period_metrics,
        "summary_rows": rows,
        "correlations": correlations,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_csv(rows, summary_path)

    print_summary(rows, correlations)
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
