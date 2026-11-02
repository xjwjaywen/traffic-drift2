"""Read-only uncertainty review of existing paid BCWI/source block signs.

Reports individual seeds at fixed cumulative-flow checkpoints. Exact binomial
p-values describe a selected day's paired-correctness sample, not sequential
stopping guarantees, macro-F1 confidence, or independent validation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from predict import PRIMARY

CHECKPOINTS = (0, 1, 2, 4, 6, 9)


def review(frame):
    keys = ["home", "seed", "transition", "block"]
    if (set(frame.home) != set(PRIMARY) or
            set(frame.seed) != set(range(5)) or
            frame.duplicated(keys).any() or
            not frame.groupby(["home", "seed", "transition"]).block.apply(
                lambda x: np.array_equal(np.sort(x.to_numpy()),
                                         np.arange(len(x)))).all()):
        raise ValueError("Paid arrival block layout differs")
    rows = []
    for row in frame.itertuples(index=False):
        s = int(row.cumulative_paid_source_only_correct)
        c = int(row.cumulative_paid_bcwi_only_correct)
        n = s + c
        if (s < 0 or c < 0 or n > int(row.cumulative_paid_rows) or
                c - s != int(row.cumulative_paid_net_bcwi_correct)):
            raise ValueError("Paid paired-correctness counts differ")
        rows.append(dict(
            home=row.home, seed=int(row.seed),
            transition=int(row.transition), block=int(row.block),
            observed_curated_rows=(int(row.block) + 1) * 1000,
            remaining_curated_rows=int(row.remaining_curated_rows),
            cumulative_paid_rows=int(row.cumulative_paid_rows),
            cumulative_exclusive_wins=n,
            cumulative_source_wins=s, cumulative_bcwi_wins=c,
            source_win_fraction=s / n if n else np.nan,
            source_better_one_sided_p=(
                binomtest(s, n, p=.5, alternative="greater").pvalue
                if n else np.nan),
            bcwi_better_one_sided_p=(
                binomtest(c, n, p=.5, alternative="greater").pvalue
                if n else np.nan),
            full_day_bcwi_gain_pp=float(row.full_day_bcwi_gain_pp)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_paid_arrival_audit_v1/per_seed_block.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_paid_arrival_review_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    result = review(pd.read_csv(args.input.resolve(strict=True)))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_seed_checkpoint.csv", index=False)
    focus = result.loc[(result.home == "home07") &
                       result.transition.isin([1, 3]) &
                       result.block.isin(CHECKPOINTS), [
                           "transition", "seed", "block", "cumulative_paid_rows",
                           "cumulative_exclusive_wins", "cumulative_source_wins",
                           "cumulative_bcwi_wins", "source_better_one_sided_p",
                           "bcwi_better_one_sided_p", "full_day_bcwi_gain_pp"]]
    report = "\n".join([
        "Read-only per-seed paid-label uncertainty review on examined development blocks.",
        "One-sided binomial p-values condition on exclusive-correctness events in the existing uniform paid sample.",
        "Repeated checkpoint inspection invalidates ordinary p<0.05 as an anytime stopping rule.",
        "Paired correctness is not macro F1; five seeds overlap; no final-window file opened.",
        "Home07 beneficial transition 1 and harmful transition 3:",
        focus.round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
