"""Aggregate 5-seed CARE results into mean ± std tables."""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def load_seed_results(base_dir, num_seeds=5):
    """Load results_by_budget.csv from each seed directory."""
    all_rows = []
    for seed in range(num_seeds):
        path = os.path.join(base_dir, f"seed_{seed}", "results_by_budget.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_seed"] = seed
                all_rows.append(row)
    return all_rows


def aggregate(rows):
    """Group by (method, strategy, budget) and compute mean ± std."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["strategy"], row.get("budget", "0"))
        groups[key].append(row)

    metrics = [
        "strict_overall_macro_f1", "strict_bad_macro_f1", "strict_stable_macro_f1",
        "strict_collapsed_count",
        "full_overall_macro_f1", "full_bad_macro_f1", "full_stable_macro_f1",
        "full_collapsed_count",
    ]

    agg_rows = []
    for (method, strategy, budget), seed_rows in sorted(groups.items()):
        agg = {
            "method": method,
            "strategy": strategy,
            "budget": budget,
            "n_seeds": len(seed_rows),
        }
        for m in metrics:
            vals = []
            for r in seed_rows:
                v = r.get(m, "")
                if v != "":
                    vals.append(float(v))
            if vals:
                agg[f"{m}_mean"] = f"{np.mean(vals):.4f}"
                agg[f"{m}_std"] = f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}"
            else:
                agg[f"{m}_mean"] = ""
                agg[f"{m}_std"] = ""
        agg_rows.append(agg)
    return agg_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_seed_results(args.base_dir, args.num_seeds)
    if not rows:
        print("No seed results found.")
        return

    agg = aggregate(rows)
    out_path = args.output or os.path.join(args.base_dir, "aggregated_mean_std.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(agg[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg)

    print(f"Aggregated {len(rows)} rows from {args.num_seeds} seeds -> {out_path}")
    print()
    print(f"{'Method':<18} {'Strategy':<22} {'Budget':>6}  "
          f"{'Strict Macro':>14} {'Strict Collapse':>16} {'Strict Stable':>14} {'Collapsed':>10}")
    print("-" * 110)
    for r in agg:
        sm = r.get("strict_overall_macro_f1_mean", "")
        ss = r.get("strict_overall_macro_f1_std", "")
        cm = r.get("strict_bad_macro_f1_mean", "")
        cs = r.get("strict_bad_macro_f1_std", "")
        stm = r.get("strict_stable_macro_f1_mean", "")
        sts = r.get("strict_stable_macro_f1_std", "")
        cc = r.get("strict_collapsed_count_mean", "")
        ccs = r.get("strict_collapsed_count_std", "")
        macro_str = f"{sm}±{ss}" if sm else ""
        collapse_str = f"{cm}±{cs}" if cm else ""
        stable_str = f"{stm}±{sts}" if stm else ""
        collapsed_str = f"{cc}±{ccs}" if cc else ""
        print(f"{r['method']:<18} {r['strategy']:<22} {r['budget']:>6}  "
              f"{macro_str:>14} {collapse_str:>16} {stable_str:>14} {collapsed_str:>10}")


if __name__ == "__main__":
    main()
