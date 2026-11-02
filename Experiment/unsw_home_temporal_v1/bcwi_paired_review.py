"""Read-only paired review of the completed fixed-grid BCWI control.

Seeds are repeated paid-label draws within homes. This script reads only the
frozen aggregate CSV and never touches the archive or final window.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from online_bcwi_control import METHODS
from predict import PRIMARY


def review(path):
    frame = pd.read_csv(path)
    expected = len(PRIMARY) * 5 * len(METHODS)
    if (len(frame) != expected or set(frame.home) != set(PRIMARY) or
            set(frame.seed) != set(range(5)) or
            set(frame.method) != set(METHODS) or
            frame.duplicated(["home", "seed", "method"]).any()):
        raise ValueError(f"Completed five-seed BCWI layout differs: {path}")
    f1 = frame.pivot(index=["home", "seed"], columns="method",
                     values="pooled_macro_f1_fixed24")
    nfr = frame.pivot(index=["home", "seed"], columns="method",
                      values="pooled_negative_flip_rate_vs_source")
    if not np.isfinite(f1.to_numpy()).all() or not np.isfinite(nfr.to_numpy()).all():
        raise ValueError("Non-finite BCWI metric")
    paired = pd.DataFrame({
        "bcwi_minus_care_f1_pp": 100 * (f1.bcwi_050 - f1.updated_care),
        "bcwi_minus_care_nfr_pp": 100 * (nfr.bcwi_050 - nfr.updated_care),
        "bcwi_minus_target_f1_pp": 100 * (f1.bcwi_050 - f1.updated_target),
        "bcwi_minus_target_nfr_pp": 100 * (nfr.bcwi_050 - nfr.updated_target),
        "bcwi_minus_source_f1_pp": 100 * (f1.bcwi_050 - f1.source),
    }).reset_index()
    paired["dominates_care"] = ((paired.bcwi_minus_care_f1_pp > 0) &
                                (paired.bcwi_minus_care_nfr_pp < 0))
    paired["dominates_target"] = ((paired.bcwi_minus_target_f1_pp > 0) &
                                  (paired.bcwi_minus_target_nfr_pp < 0))
    by_home = paired.groupby("home", as_index=False).agg(
        mean_f1_minus_care_pp=("bcwi_minus_care_f1_pp", "mean"),
        seeds_f1_beats_care=("bcwi_minus_care_f1_pp",
                             lambda x: int((x > 0).sum())),
        mean_nfr_minus_care_pp=("bcwi_minus_care_nfr_pp", "mean"),
        seeds_nfr_below_care=("bcwi_minus_care_nfr_pp",
                              lambda x: int((x < 0).sum())),
        seeds_dominates_care=("dominates_care", "sum"),
        mean_f1_minus_target_pp=("bcwi_minus_target_f1_pp", "mean"),
        mean_nfr_minus_target_pp=("bcwi_minus_target_nfr_pp", "mean"),
        seeds_dominates_target=("dominates_target", "sum"),
        mean_f1_minus_source_pp=("bcwi_minus_source_f1_pp", "mean"))
    return paired, by_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_v1/aggregate_0_1_2_3_4"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_v1/paired_review_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    paired, by_home = review(args.aggregate.resolve(strict=True) / "per_seed.csv")
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "per_seed.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Read-only paired review of prespecified BCWI old-weight alpha=0.50.",
        "The three alpha grid points were fixed before scoring; focusing this review on 0.50 is post-hoc.",
        "F1 and conditional negative-flip differences are percentage points.",
        "A dominance event requires strictly higher F1 and strictly lower negative-flip rate for that home/seed.",
        "The five paid-label seeds overlap within each home; homes are the independent units.",
        "Development outcomes were examined before this baseline; no final-window file opened.",
        "Per-home means and counts out of five seeds:",
        by_home.round(4).to_string(index=False),
        "Across-home means:",
        by_home.drop(columns="home").mean(numeric_only=True).round(4).to_string(),
        "Homes where BCWI dominates CARE on mean metrics: " +
        f"{int(((by_home.mean_f1_minus_care_pp > 0) & (by_home.mean_nfr_minus_care_pp < 0)).sum())}/{len(by_home)}",
        "Homes where BCWI dominates target-only on mean metrics: " +
        f"{int(((by_home.mean_f1_minus_target_pp > 0) & (by_home.mean_nfr_minus_target_pp < 0)).sum())}/{len(by_home)}",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
