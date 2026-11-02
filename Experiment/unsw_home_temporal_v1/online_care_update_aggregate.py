"""Aggregate five paired online CARE update seeds, treating homes as units.

Reads only completed evaluation CSVs. Seeds within a home share the source
model and development outcomes; they are repeated label draws, not
independent datasets. No archive, model, or final-window file is opened.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import PRIMARY

METHODS = {"source", "fixed_care", "updated_care"}
SEEDS = range(5)


def aggregate(root):
    pieces = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "evaluation_all" / "pooled_by_home.csv"
        part = pd.read_csv(path)
        if (set(part.home) != set(PRIMARY) or set(part.method) != METHODS or
                len(part) != len(PRIMARY) * len(METHODS) or
                part.duplicated(["home", "method"]).any() or
                set(part.seed) != {seed}):
            raise ValueError(f"Completed seed evaluation layout differs: {path}")
        metrics = part[["pooled_macro_f1_fixed24",
                        "pooled_negative_flip_rate_vs_source"]].to_numpy()
        if not np.isfinite(metrics).all() or ((metrics < 0) | (metrics > 1)).any():
            raise ValueError(f"Non-finite or out-of-range seed metrics: {path}")
        pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    f1 = frame.pivot(index=["home", "seed"], columns="method",
                     values="pooled_macro_f1_fixed24")
    nfr = frame.pivot(index=["home", "seed"], columns="method",
                      values="pooled_negative_flip_rate_vs_source")
    source_by_home = f1["source"].unstack("seed")
    if not np.allclose(source_by_home, source_by_home.iloc[:, [0]].to_numpy(),
                       rtol=0, atol=1e-10):
        raise ValueError("Source model differs across repair seeds")
    if not np.allclose(nfr["source"], 0, rtol=0, atol=1e-10):
        raise ValueError("Source negative-flip rate must be zero")
    paired = pd.DataFrame({
        "updated_minus_fixed_pp": 100 * (f1["updated_care"] - f1["fixed_care"]),
        "updated_minus_source_pp": 100 * (f1["updated_care"] - f1["source"]),
        "fixed_minus_source_pp": 100 * (f1["fixed_care"] - f1["source"]),
        "updated_minus_fixed_nfr_pp": 100 * (
            nfr["updated_care"] - nfr["fixed_care"]),
        "fixed_nfr": nfr["fixed_care"],
        "updated_nfr": nfr["updated_care"],
    }).reset_index()
    by_home = (paired.groupby("home", as_index=False)
               .agg(mean_updated_minus_fixed_pp=("updated_minus_fixed_pp", "mean"),
                    seeds_updated_beats_fixed=("updated_minus_fixed_pp",
                                               lambda x: int((x > 0).sum())),
                    mean_updated_minus_source_pp=("updated_minus_source_pp", "mean"),
                    seeds_updated_beats_source=("updated_minus_source_pp",
                                                lambda x: int((x > 0).sum())),
                    mean_fixed_minus_source_pp=("fixed_minus_source_pp", "mean"),
                    mean_updated_minus_fixed_nfr_pp=(
                        "updated_minus_fixed_nfr_pp", "mean"),
                    mean_fixed_nfr=("fixed_nfr", "mean"),
                    mean_updated_nfr=("updated_nfr", "mean")))
    return paired, by_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_care_update_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_care_update_v1/aggregate_0_1_2_3_4"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    paired, by_home = aggregate(args.root.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "per_seed.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Exploratory five-seed chronological CARE update aggregation.",
        "Each seed uses the original 1000 paid query labels plus 1000 paid labels per elapsed development day.",
        "Source encoder, historical references, and 500-step CARE retraining schedule are fixed.",
        "Homes, not the five overlapping seed draws, are independent units.",
        "The development period was examined before this control was designed; final window untouched.",
        "Per-home means and seed counts:",
        by_home.round(4).to_string(index=False),
        "Across-home means:",
        by_home.drop(columns="home").mean(numeric_only=True).round(4).to_string(),
        "Homes where updated CARE beats fixed CARE on average: " +
        f"{int((by_home.mean_updated_minus_fixed_pp > 0).sum())}/{len(by_home)}",
        "Homes where updated CARE beats source on average: " +
        f"{int((by_home.mean_updated_minus_source_pp > 0).sum())}/{len(by_home)}",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
