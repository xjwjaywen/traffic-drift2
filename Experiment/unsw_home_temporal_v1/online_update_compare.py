"""Compare frozen online CARE and paired target-only updates by home.

Reads only completed evaluation CSVs. The five seeds are overlapping draws
within each home, not independent experimental units. No archive is opened.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import PRIMARY


def read_method(root, seed, expected):
    path = root / f"seed_{seed}" / "evaluation_all" / "pooled_by_home.csv"
    part = pd.read_csv(path)
    if (set(part.home) != set(PRIMARY) or set(part.method) != expected or
            len(part) != len(PRIMARY) * len(expected) or
            part.duplicated(["home", "method"]).any() or
            set(part.seed) != {seed}):
        raise ValueError(f"Completed evaluation layout differs: {path}")
    values = part[["pooled_macro_f1_fixed24",
                   "pooled_negative_flip_rate_vs_source"]].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Invalid evaluation metric: {path}")
    return part


def compare(care_root, target_root):
    parts = []
    for seed in range(5):
        care = read_method(care_root, seed,
                           {"source", "fixed_care", "updated_care"})
        target = read_method(target_root, seed,
                             {"source", "fixed_target", "updated_target"})
        baseline = care[care.method == "source"].set_index("home")
        target_source = target[target.method == "source"].set_index("home")
        for column in ("pooled_curated_rows", "pooled_present_classes"):
            if not baseline[column].equals(target_source[column]):
                raise ValueError(f"Evaluation rows differ: seed={seed} {column}")
        for column in ("pooled_macro_f1_fixed24",
                       "pooled_negative_flip_rate_vs_source"):
            if not np.allclose(baseline[column], target_source[column],
                               rtol=0, atol=1e-10):
                raise ValueError(f"Source control differs: seed={seed} {column}")
        parts.append(pd.concat((care, target[target.method != "source"]),
                               ignore_index=True))
    frame = pd.concat(parts, ignore_index=True)
    f1 = frame.pivot(index=["home", "seed"], columns="method",
                     values="pooled_macro_f1_fixed24")
    nfr = frame.pivot(index=["home", "seed"], columns="method",
                      values="pooled_negative_flip_rate_vs_source")
    paired = pd.DataFrame({
        "updated_care_minus_updated_target_pp": 100 * (
            f1.updated_care - f1.updated_target),
        "updated_care_minus_source_pp": 100 * (
            f1.updated_care - f1.source),
        "updated_target_minus_source_pp": 100 * (
            f1.updated_target - f1.source),
        "fixed_care_minus_fixed_target_pp": 100 * (
            f1.fixed_care - f1.fixed_target),
        "updated_care_minus_target_nfr_pp": 100 * (
            nfr.updated_care - nfr.updated_target),
    }).reset_index()
    by_home = paired.groupby("home", as_index=False).agg(
        mean_care_minus_target_pp=("updated_care_minus_updated_target_pp", "mean"),
        seeds_care_beats_target=("updated_care_minus_updated_target_pp",
                                 lambda x: int((x > 0).sum())),
        mean_care_minus_source_pp=("updated_care_minus_source_pp", "mean"),
        mean_target_minus_source_pp=("updated_target_minus_source_pp", "mean"),
        mean_fixed_care_minus_target_pp=("fixed_care_minus_fixed_target_pp", "mean"),
        mean_care_minus_target_nfr_pp=("updated_care_minus_target_nfr_pp", "mean"))
    return paired, by_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--care-root", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_care_update_v1"))
    parser.add_argument("--target-root", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_target_update_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_update_compare_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    paired, by_home = compare(args.care_root.resolve(strict=True),
                              args.target_root.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "per_seed.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Exploratory paired online CARE versus target-only update control.",
        "Same source, query labels, prior-day paid labels, 500 steps, and target minibatch stream.",
        "CARE alone adds the original historical replay CE and KD losses.",
        "Development outcomes were examined before this control; final window untouched.",
        "Homes are independent units; five seeds within each home overlap.",
        "All deltas below are F1 or negative-flip percentage points.",
        "Per-home means and paired seed counts:",
        by_home.round(4).to_string(index=False),
        "Across-home means:",
        by_home.drop(columns="home").mean(numeric_only=True).round(4).to_string(),
        "Homes where updated CARE beats updated target on average: " +
        f"{int((by_home.mean_care_minus_target_pp > 0).sum())}/{len(by_home)}",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
