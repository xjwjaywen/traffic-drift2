"""
Result aggregation for TTA-TC experiments.

Loads all JSON result files produced by evaluate_tta.py / run_ablation.py
and merges them into a single structured summary.

Usage (from Experiment/core_code/):
    python ../analysis/aggregate_results.py --output-dir outputs/
    python ../analysis/aggregate_results.py --output-dir outputs/ \
        --save-csv ../analysis/results/summary.csv

Output structure:
    {
      "comparison": {method: {acc, f1, arr, aurc, time_s}},
      "sequential_quic22": {method: {periods: {...}, aurc}},
      "sequential_tls22":  {method: {periods: {...}, aurc}},
      "ablations": {ablation_name: {setting: {acc, f1}}},
    }
"""
import argparse
import json
import os
import glob
import pandas as pd


# --------------------------------------------------------------------------- #
#  Loader helpers
# --------------------------------------------------------------------------- #

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_single_eval(outputs_dir):
    """
    Load results from single-period comparison evaluation.
    Returns dict: {method: {accuracy, macro_f1, adapt_time_s}}.
    """
    path = os.path.join(outputs_dir, "eval_quic22_single", "results_single.json")
    if not os.path.exists(path):
        return {}
    return _load_json(path)


def load_sequential_eval(outputs_dir, dataset="quic22"):
    """
    Load results from sequential (continual) evaluation.
    Returns dict: {method: {aurc, periods: {period: {accuracy, macro_f1, arr}}}}.
    """
    subdir = f"eval_{dataset}_sequential"
    path = os.path.join(outputs_dir, subdir, "results_sequential.json")
    if not os.path.exists(path):
        return {}
    return _load_json(path)


def load_transformer_eval(outputs_dir):
    """Load Transformer-backbone sequential results."""
    path = os.path.join(outputs_dir, "eval_quic22_transformer", "results_sequential.json")
    if not os.path.exists(path):
        return {}
    return _load_json(path)


def load_ablation(outputs_dir, ablation_name):
    """
    Load ablation results by name (e.g., 'ssl_tasks', 'mask_ratio').
    Returns dict: {setting_name: {accuracy, macro_f1, settings}}.
    """
    path = os.path.join(outputs_dir, "ablations", ablation_name, "ablation_results.json")
    if not os.path.exists(path):
        return {}
    return _load_json(path)


# --------------------------------------------------------------------------- #
#  Summary builders
# --------------------------------------------------------------------------- #

def build_comparison_table(single_results, sequential_results):
    """
    Build a flat method-comparison table combining single-period and sequential metrics.
    Returns a list of dicts, one per method.
    """
    rows = []
    methods_order = ["static", "bn_adapt", "tent", "eata", "cotta", "sar", "note", "tta_tc"]

    for method in methods_order:
        row = {"method": method}

        # Single-period metrics
        if method in single_results:
            s = single_results[method]
            row["acc_w45"] = s.get("accuracy")
            row["f1_w45"] = s.get("macro_f1")
            row["time_s"] = s.get("adapt_time_s")

        # Sequential metrics (AURC, ARR across periods)
        if method in sequential_results:
            sq = sequential_results[method]
            row["aurc"] = sq.get("aurc")
            periods = sq.get("periods", {})
            arrs = [v["arr"] for v in periods.values() if v.get("arr") is not None]
            row["mean_arr"] = sum(arrs) / len(arrs) if arrs else None
            # Last period accuracy (longest drift)
            if periods:
                last_period = sorted(periods.keys())[-1]
                row["acc_last"] = periods[last_period].get("accuracy")

        rows.append(row)

    return rows


def build_ablation_table(ablation_results, ablation_name):
    """Convert ablation dict to list of rows."""
    rows = []
    for setting, metrics in ablation_results.items():
        rows.append({
            "ablation": ablation_name,
            "setting": setting,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "settings": json.dumps(metrics.get("settings", {})),
        })
    return rows


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

ABLATION_NAMES = ["ssl_tasks", "mask_ratio", "adapt_depth", "anti_forgetting", "norm_type"]


def main():
    parser = argparse.ArgumentParser(description="TTA-TC result aggregation")
    parser.add_argument("--output-dir", default="outputs",
                        help="Root outputs directory (default: outputs/)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save CSVs and aggregated JSON")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(save_dir, exist_ok=True)

    out = args.output_dir

    print("Loading results...")

    # ---------- Load ----------
    single = load_single_eval(out)
    seq_quic22 = load_sequential_eval(out, "quic22")
    seq_tls22  = load_sequential_eval(out, "tls22")
    transformer = load_transformer_eval(out)

    ablations = {}
    for name in ABLATION_NAMES:
        abl = load_ablation(out, name)
        if abl:
            ablations[name] = abl

    # ---------- Build tables ----------
    comparison_rows = build_comparison_table(single, seq_quic22)
    comparison_df = pd.DataFrame(comparison_rows)

    abl_rows = []
    for name, data in ablations.items():
        abl_rows.extend(build_ablation_table(data, name))
    ablation_df = pd.DataFrame(abl_rows) if abl_rows else pd.DataFrame()

    # ---------- Save CSVs ----------
    comparison_path = os.path.join(save_dir, "comparison_table.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved: {comparison_path}")

    if not ablation_df.empty:
        abl_path = os.path.join(save_dir, "ablation_table.csv")
        ablation_df.to_csv(abl_path, index=False)
        print(f"Saved: {abl_path}")

    # ---------- Aggregated JSON ----------
    summary = {
        "comparison": single,
        "sequential_quic22": seq_quic22,
        "sequential_tls22": seq_tls22,
        "transformer": transformer,
        "ablations": ablations,
    }
    summary_path = os.path.join(save_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # ---------- Print comparison table ----------
    print("\n" + "=" * 70)
    print("Method Comparison (QUIC22)")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print()

    if not ablation_df.empty:
        print("=" * 70)
        print("Ablations")
        print("=" * 70)
        for name in ablations:
            subset = ablation_df[ablation_df["ablation"] == name]
            if not subset.empty:
                print(f"\n{name}:")
                print(subset[["setting", "accuracy", "macro_f1"]].to_string(index=False))

    return summary


if __name__ == "__main__":
    main()
