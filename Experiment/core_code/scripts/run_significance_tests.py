"""
Statistical significance tests for Table 5 comparisons.

Reads per-seed results from care_5seeds_strict_cnn/ and badge_5seeds_strict/,
then runs paired t-tests and Wilcoxon signed-rank tests for:
  - margin vs random (macro-F1, collapse-F1)
  - margin vs BADGE (macro-F1, collapse-F1)

Output: JSON with test statistics and p-values.

Usage from Experiment/core_code/:
    python scripts/run_significance_tests.py \
      --care-dir outputs/care_5seeds_strict_cnn \
      --badge-dir outputs/badge_5seeds_strict \
      --output-dir outputs/significance_tests
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


def load_seed_results(base_dir, num_seeds=5):
    """Load results_by_budget.csv from each seed directory."""
    all_rows = []
    for seed in range(num_seeds):
        path = os.path.join(base_dir, f"seed_{seed}", "results_by_budget.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_seed"] = seed
                all_rows.append(row)
    return all_rows


def extract_metric_by_strategy(rows, strategy, budget, metric):
    """Extract a metric value for each seed, filtered by strategy and budget."""
    values = {}
    for row in rows:
        s = row.get("strategy", "")
        b = str(row.get("budget", "0"))
        if s == strategy and b == str(budget):
            seed = row["_seed"]
            val = row.get(metric, "")
            if val != "":
                values[seed] = float(val)
    return values


def paired_tests(values_a, values_b, label_a, label_b, metric_name):
    """Run paired t-test and Wilcoxon signed-rank test on matched seed pairs."""
    # Find common seeds
    common_seeds = sorted(set(values_a.keys()) & set(values_b.keys()))
    n = len(common_seeds)

    if n < 2:
        return {
            "comparison": f"{label_a} vs {label_b}",
            "metric": metric_name,
            "n_paired": n,
            "error": "Insufficient paired observations (need >= 2)",
        }

    a = np.array([values_a[s] for s in common_seeds])
    b = np.array([values_b[s] for s in common_seeds])
    diff = a - b

    result = {
        "comparison": f"{label_a} vs {label_b}",
        "metric": metric_name,
        "n_paired": n,
        "seeds": common_seeds,
        f"{label_a}_values": a.tolist(),
        f"{label_b}_values": b.tolist(),
        f"{label_a}_mean": round(float(a.mean()), 6),
        f"{label_b}_mean": round(float(b.mean()), 6),
        "mean_diff": round(float(diff.mean()), 6),
        "std_diff": round(float(diff.std(ddof=1)), 6) if n > 1 else 0.0,
    }

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(a, b)
    result["paired_ttest"] = {
        "t_statistic": round(float(t_stat), 6),
        "p_value": round(float(t_pval), 6),
        "significant_005": bool(t_pval < 0.05),
        "significant_001": bool(t_pval < 0.01),
    }

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        if np.all(diff == 0):
            result["wilcoxon"] = {
                "warning": "All differences are zero; test not applicable",
            }
        else:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
            result["wilcoxon"] = {
                "W_statistic": round(float(w_stat), 6),
                "p_value": round(float(w_pval), 6),
                "significant_005": bool(w_pval < 0.05),
                "significant_001": bool(w_pval < 0.01),
            }
    except ValueError as e:
        result["wilcoxon"] = {"error": str(e)}

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for CARE strategy comparisons."
    )
    parser.add_argument("--care-dir", default="outputs/care_5seeds_strict_cnn",
                        help="Directory with CARE 5-seed results (margin, random)")
    parser.add_argument("--badge-dir", default="outputs/badge_5seeds_strict",
                        help="Directory with BADGE 5-seed results")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=1000,
                        help="Label budget to compare (default: 1000)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading CARE results from: {args.care_dir}")
    care_rows = load_seed_results(args.care_dir, args.num_seeds)
    print(f"  Loaded {len(care_rows)} rows")

    print(f"Loading BADGE results from: {args.badge_dir}")
    badge_rows = load_seed_results(args.badge_dir, args.num_seeds)
    print(f"  Loaded {len(badge_rows)} rows")

    if not care_rows:
        print("ERROR: No CARE results found. Check --care-dir path.")
        sys.exit(1)

    # Define metrics to test
    metrics = [
        ("strict_overall_macro_f1", "Macro-F1 (overall)"),
        ("strict_bad_macro_f1", "Collapse-F1"),
    ]

    budget = args.budget
    all_tests = []

    print(f"\n{'='*70}")
    print(f"Statistical Significance Tests (budget={budget})")
    print(f"{'='*70}")

    # Test 1: margin vs random (from CARE results)
    for metric_key, metric_label in metrics:
        margin_vals = extract_metric_by_strategy(care_rows, "margin", budget, metric_key)
        random_vals = extract_metric_by_strategy(care_rows, "random", budget, metric_key)

        if margin_vals and random_vals:
            result = paired_tests(margin_vals, random_vals, "margin", "random", metric_label)
            all_tests.append(result)

            print(f"\n--- margin vs random: {metric_label} ---")
            print(f"  margin:  {result.get('margin_mean', 'N/A'):.4f}")
            print(f"  random:  {result.get('random_mean', 'N/A'):.4f}")
            print(f"  diff:    {result.get('mean_diff', 'N/A'):+.4f}")
            t_info = result.get("paired_ttest", {})
            print(f"  t-test:  t={t_info.get('t_statistic', 'N/A'):.3f}, "
                  f"p={t_info.get('p_value', 'N/A'):.4f}"
                  f"{' *' if t_info.get('significant_005') else ''}"
                  f"{'*' if t_info.get('significant_001') else ''}")
            w_info = result.get("wilcoxon", {})
            if "p_value" in w_info:
                print(f"  Wilcoxon: W={w_info.get('W_statistic', 'N/A'):.1f}, "
                      f"p={w_info.get('p_value', 'N/A'):.4f}"
                      f"{' *' if w_info.get('significant_005') else ''}"
                      f"{'*' if w_info.get('significant_001') else ''}")
        else:
            print(f"\n--- margin vs random: {metric_label} ---")
            print(f"  SKIPPED: margin seeds={len(margin_vals)}, random seeds={len(random_vals)}")

    # Test 2: margin vs BADGE
    for metric_key, metric_label in metrics:
        margin_vals = extract_metric_by_strategy(care_rows, "margin", budget, metric_key)
        badge_vals = extract_metric_by_strategy(badge_rows, "badge", budget, metric_key)

        # BADGE may use different strategy names; try alternatives
        if not badge_vals:
            badge_vals = extract_metric_by_strategy(badge_rows, "margin", budget, metric_key)
        if not badge_vals:
            # Check all strategies in badge results
            strats = set(r.get("strategy", "") for r in badge_rows)
            for s in strats:
                badge_vals = extract_metric_by_strategy(badge_rows, s, budget, metric_key)
                if badge_vals:
                    break

        if margin_vals and badge_vals:
            result = paired_tests(margin_vals, badge_vals, "margin", "badge", metric_label)
            all_tests.append(result)

            print(f"\n--- margin vs BADGE: {metric_label} ---")
            print(f"  margin:  {result.get('margin_mean', 'N/A'):.4f}")
            print(f"  BADGE:   {result.get('badge_mean', 'N/A'):.4f}")
            print(f"  diff:    {result.get('mean_diff', 'N/A'):+.4f}")
            t_info = result.get("paired_ttest", {})
            print(f"  t-test:  t={t_info.get('t_statistic', 'N/A'):.3f}, "
                  f"p={t_info.get('p_value', 'N/A'):.4f}"
                  f"{' *' if t_info.get('significant_005') else ''}"
                  f"{'*' if t_info.get('significant_001') else ''}")
            w_info = result.get("wilcoxon", {})
            if "p_value" in w_info:
                print(f"  Wilcoxon: W={w_info.get('W_statistic', 'N/A'):.1f}, "
                      f"p={w_info.get('p_value', 'N/A'):.4f}"
                      f"{' *' if w_info.get('significant_005') else ''}"
                      f"{'*' if w_info.get('significant_001') else ''}")
        else:
            print(f"\n--- margin vs BADGE: {metric_label} ---")
            print(f"  SKIPPED: margin seeds={len(margin_vals)}, badge seeds={len(badge_vals)}")

    # Save all results
    output = {
        "care_dir": args.care_dir,
        "badge_dir": args.badge_dir,
        "budget": budget,
        "num_seeds": args.num_seeds,
        "tests": all_tests,
        "interpretation": {
            "p<0.05": "Statistically significant at 5% level",
            "p<0.01": "Statistically significant at 1% level",
            "paired_ttest": "Parametric test assuming normal distribution of differences",
            "wilcoxon": "Non-parametric alternative; more conservative with small n",
            "note": (
                "With n=5 seeds, Wilcoxon's minimum possible p-value is 0.0625 "
                "(cannot reach 0.05), so the t-test is the primary significance "
                "measure for this sample size."
            ),
        },
    }

    out_path = os.path.join(args.output_dir, "significance_tests.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved to {out_path}")

    # Print summary table for paper
    print(f"\n=== Table for paper ===")
    print(f"{'Comparison':<25} {'Metric':<20} {'Diff':>8} {'t':>8} {'p(t)':>8} {'sig':>5}")
    print("-" * 80)
    for t in all_tests:
        comp = t["comparison"]
        metric = t["metric"]
        diff = t.get("mean_diff", 0)
        t_info = t.get("paired_ttest", {})
        t_stat = t_info.get("t_statistic", float("nan"))
        p_val = t_info.get("p_value", float("nan"))
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{comp:<25} {metric:<20} {diff:>+8.4f} {t_stat:>8.3f} {p_val:>8.4f} {sig:>5}")


if __name__ == "__main__":
    main()
