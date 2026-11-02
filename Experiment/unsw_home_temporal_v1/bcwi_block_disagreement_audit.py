"""Read-only BCWI/source disagreement beside examined block outcomes.

Disagreement uses only frozen predictions. Previously examined development
labels are joined afterward to describe whether disagreement indicates a
useful correction; no gate, threshold, or final-window result is produced.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from bcwi_early_shift_probe import development_stamps
from online_target_update import read_membership
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
BLOCK = 1000


def audit(archive, windows, members, official, baseline_root, bcwi_root,
          outcome):
    keys = ["home", "seed", "transition", "block"]
    if (set(outcome.home) != set(PRIMARY) or
            outcome.duplicated(keys).any()):
        raise ValueError("Completed block-outcome layout differs")
    records = []
    for home in PRIMARY:
        mapping = (official.loc[official.home == home]
                   .set_index("mac").device_type.to_dict())
        days, curated = read_membership(
            archive, members.loc[home, "member"], home, windows, set(mapping))
        stamps = development_stamps(
            archive, members.loc[home, "member"], home, windows, days)
        source = np.load(verified(baseline_root / home, "static_mlp.npy"),
                         mmap_mode="r")
        if len(source) != len(days):
            raise ValueError(f"Frozen source alignment differs: {home}")
        ordered = np.unique(days)
        for t in range(1, len(ordered) - 1):
            tomorrow_ids = np.flatnonzero(days == ordered[t + 1])
            current_ids = np.flatnonzero((days == ordered[t + 1]) & curated)
            current_ids = current_ids[np.argsort(stamps[current_ids],
                                                  kind="stable")]
            nblocks = len(current_ids) // BLOCK
            local = np.searchsorted(tomorrow_ids, current_ids)
            if (np.any(local >= len(tomorrow_ids)) or
                    not np.array_equal(tomorrow_ids[local], current_ids)):
                raise ValueError(f"Curated row alignment differs: {home} {t}")
            source_pred = np.asarray(source[current_ids], dtype=np.int64)
            for seed in range(5):
                folder = bcwi_root / f"seed_{seed}" / home / f"transition_{t}"
                frozen_ids = np.load(verified(folder, "tomorrow_indices.npy"))
                if not np.array_equal(frozen_ids, tomorrow_ids):
                    raise ValueError(f"Frozen BCWI indices differ: {home} {seed} {t}")
                bcwi_all = np.load(verified(folder, "bcwi_050.npy"),
                                   mmap_mode="r")
                if len(bcwi_all) != len(tomorrow_ids):
                    raise ValueError(f"Frozen BCWI length differs: {home} {seed} {t}")
                bcwi_pred = np.asarray(bcwi_all[local], dtype=np.int64)
                for b in range(nblocks):
                    start, stop = b * BLOCK, (b + 1) * BLOCK
                    disagreements = int(np.sum(source_pred[start:stop] !=
                                               bcwi_pred[start:stop]))
                    records.append(dict(
                        home=home, seed=seed, transition=t, block=b,
                        disagreement_rows=disagreements,
                        disagreement_rate=disagreements / BLOCK))
        print(home, "unlabeled BCWI disagreements checked", flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != len(outcome):
        raise ValueError("Disagreement/outcome block counts differ")
    joined = frame.merge(outcome, on=keys, how="inner", validate="one_to_one")
    if (len(joined) != len(frame) or
            (joined.source_only_correct + joined.bcwi_only_correct >
             joined.disagreement_rows).any() or
            (joined.net_bcwi_correct.abs() > joined.disagreement_rows).any()):
        raise ValueError("Disagreement does not bound observed exclusive wins")
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--bcwi", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_v1"))
    parser.add_argument("--outcome", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_block_outcome_audit_v1/per_seed_block.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_block_disagreement_audit_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())[
        "archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    outcome = pd.read_csv(args.outcome.resolve(strict=True))
    joined = audit(archive, windows, members, official,
                   args.baseline.resolve(strict=True), args.bcwi.resolve(strict=True),
                   outcome)
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "per_seed_block.csv", index=False)
    home07 = joined.loc[(joined.home == "home07") &
                        joined.transition.isin([1, 3])].groupby(
                            ["transition", "block", "end_timestamp",
                             "block_tv_vs_prior_day"], as_index=False).agg(
                                disagreement_rate=("disagreement_rate", "mean"),
                                mean_net_bcwi_correct=("net_bcwi_correct", "mean"))
    focus = home07.loc[(home07.transition == 1) |
                       ((home07.transition == 3) & (home07.block < 10))]
    per_day = joined.groupby(["home", "seed", "transition"],
                             as_index=False).agg(
        mean_disagreement_rate=("disagreement_rate", "mean"),
        max_disagreement_rate=("disagreement_rate", "max"),
        net_bcwi_correct=("net_bcwi_correct", "sum"),
        full_day_bcwi_gain_pp=("full_day_bcwi_gain_pp", "first"))
    per_day.to_csv(output / "per_seed_day.csv", index=False)
    report = "\n".join([
        "Exploratory unlabeled BCWI/source disagreement audit on examined development blocks.",
        "Disagreement rate uses only frozen predictions; correctness is a post-hoc label-based outcome.",
        "A disagreement is not evidence of which model is right. No rule or threshold is selected.",
        "Five seeds and adjacent blocks overlap; final-window labels/features untouched.",
        "Home07 beneficial transition 1 and harmful transition 3 (first ten blocks):",
        focus.round(4).to_string(index=False),
        "Home07 day-level disagreement and BCWI gain:",
        per_day.loc[per_day.home == "home07"].groupby("transition", as_index=False).agg(
            mean_disagreement_rate=("mean_disagreement_rate", "mean"),
            mean_max_block_disagreement_rate=("max_disagreement_rate", "mean"),
            mean_net_bcwi_correct=("net_bcwi_correct", "mean"),
            mean_full_day_bcwi_gain_pp=("full_day_bcwi_gain_pp", "mean")
        ).round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
