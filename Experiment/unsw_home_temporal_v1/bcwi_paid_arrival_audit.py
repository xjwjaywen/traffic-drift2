"""When do previously paid uniform labels reveal BCWI/source block wins?

The same 1000 labels per elapsed day used by the frozen online update are
examined in timestamp order. Their identities were selected from the full day
offline, so this is signal feasibility, not an implementable online sampler.
Full development outcomes are used only for retrospective comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from bcwi_early_shift_probe import development_stamps
from online_target_update import select_paid
from online_transport_feasibility import development_days_and_labels
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
    for home_index, home in enumerate(PRIMARY):
        mapping = (official.loc[official.home == home]
                   .set_index("mac").device_type.to_dict())
        days, y_all = development_days_and_labels(
            archive, members.loc[home, "member"], home, windows, mapping)
        stamps = development_stamps(
            archive, members.loc[home, "member"], home, windows, days)
        source_all = np.load(verified(baseline_root / home, "static_mlp.npy"),
                             mmap_mode="r")
        if len(source_all) != len(days):
            raise ValueError(f"Source prediction alignment differs: {home}")
        eligible = y_all >= 0
        ordered = np.unique(days)
        for seed in range(5):
            paid_days, paid_by_day = select_paid(days, eligible, home_index, seed)
            if not np.array_equal(paid_days, ordered):
                raise ValueError(f"Paid selection day coverage differs: {home} {seed}")
            # The last predicted day has no subsequent model update and no
            # frozen paid sample; therefore only earlier days can be audited.
            for t in range(1, len(ordered) - 2):
                current_day = ordered[t + 1]
                ids = np.flatnonzero((days == current_day) & eligible)
                ids = ids[np.argsort(stamps[ids], kind="stable")]
                nblocks = len(ids) // BLOCK
                selected = paid_by_day[t + 1]
                if len(selected) != 1000 or not np.all(
                        (days[selected] == current_day) & eligible[selected]):
                    raise ValueError(f"Current paid labels differ: {home} {seed} {t}")
                next_folder = (bcwi_root / f"seed_{seed}" / home /
                               f"transition_{t + 1}")
                frozen_paid = np.load(verified(next_folder,
                                               "paid_dev_indices.npy"))
                if not np.array_equal(frozen_paid,
                                      np.concatenate(paid_by_day[:t + 2])):
                    raise ValueError(f"Frozen paid identities differ: {home} {seed} {t}")
                folder = bcwi_root / f"seed_{seed}" / home / f"transition_{t}"
                tomorrow_ids = np.load(verified(folder, "tomorrow_indices.npy"))
                if not np.array_equal(tomorrow_ids,
                                      np.flatnonzero(days == current_day)):
                    raise ValueError(f"Frozen prediction indices differ: {home} {seed} {t}")
                local = np.searchsorted(tomorrow_ids, ids)
                if (np.any(local >= len(tomorrow_ids)) or
                        not np.array_equal(tomorrow_ids[local], ids)):
                    raise ValueError(f"Curated row alignment differs: {home} {seed} {t}")
                bcwi_all = np.load(verified(folder, "bcwi_050.npy"),
                                   mmap_mode="r")
                if len(bcwi_all) != len(tomorrow_ids):
                    raise ValueError(f"Frozen BCWI length differs: {home} {seed} {t}")
                source = np.asarray(source_all[ids], dtype=np.int64)
                bcwi = np.asarray(bcwi_all[local], dtype=np.int64)
                truth = np.asarray(y_all[ids], dtype=np.int64)
                paid_mask = np.isin(ids, selected, assume_unique=True)
                if int(paid_mask.sum()) != 1000:
                    raise ValueError(f"Paid rows missing from current day: {home} {seed} {t}")
                cumulative_paid = 0
                cumulative_source_wins = 0
                cumulative_bcwi_wins = 0
                for b in range(nblocks):
                    start, stop = b * BLOCK, (b + 1) * BLOCK
                    chosen = paid_mask[start:stop]
                    paid_count = int(chosen.sum())
                    y = truth[start:stop][chosen]
                    s = source[start:stop][chosen]
                    c = bcwi[start:stop][chosen]
                    source_wins = int(np.sum((s == y) & (c != y)))
                    bcwi_wins = int(np.sum((c == y) & (s != y)))
                    cumulative_paid += paid_count
                    cumulative_source_wins += source_wins
                    cumulative_bcwi_wins += bcwi_wins
                    records.append(dict(
                        home=home, seed=seed, transition=t, block=b,
                        paid_rows=paid_count,
                        paid_disagreements=int(np.sum(s != c)),
                        paid_source_only_correct=source_wins,
                        paid_bcwi_only_correct=bcwi_wins,
                        paid_net_bcwi_correct=bcwi_wins - source_wins,
                        cumulative_paid_rows=cumulative_paid,
                        cumulative_paid_source_only_correct=cumulative_source_wins,
                        cumulative_paid_bcwi_only_correct=cumulative_bcwi_wins,
                        cumulative_paid_net_bcwi_correct=(
                            cumulative_bcwi_wins - cumulative_source_wins)))
        print(home, "paid-label arrival checked", flush=True)
    frame = pd.DataFrame(records)
    # The last predicted transition is excluded for every home, because it
    # does not have a paid sample used in a subsequent frozen update.
    joined = frame.merge(outcome, on=keys, how="inner",
                         validate="one_to_one")
    if (len(joined) != len(frame) or
            (joined.paid_source_only_correct > joined.source_only_correct).any() or
            (joined.paid_bcwi_only_correct > joined.bcwi_only_correct).any() or
            (joined.paid_disagreements > BLOCK).any()):
        raise ValueError("Paid observations disagree with full block outcomes")
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
        "/data/xjw/tma-care-data/unsw_bcwi_paid_arrival_audit_v1"))
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
    frame = audit(archive, windows, members, official,
                  args.baseline.resolve(strict=True), args.bcwi.resolve(strict=True),
                  outcome)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_block.csv", index=False)
    home07 = frame.loc[(frame.home == "home07") &
                       frame.transition.isin([1, 3]) &
                       (frame.block < 10)].groupby(
                           ["transition", "block", "end_timestamp"],
                           as_index=False).agg(
                               paid_rows=("paid_rows", "mean"),
                               paid_disagreements=("paid_disagreements", "mean"),
                               paid_net_bcwi_correct=("paid_net_bcwi_correct", "mean"),
                               cumulative_paid_rows=("cumulative_paid_rows", "mean"),
                               cumulative_paid_net_bcwi_correct=(
                                   "cumulative_paid_net_bcwi_correct", "mean"),
                               full_net_bcwi_correct=("net_bcwi_correct", "mean"))
    report = "\n".join([
        "Post-hoc paid-label arrival audit using the already frozen 1000 uniform labels/day.",
        "Paid identities were selected from the full day offline, so this is not an online query policy.",
        "No new labels or model training; all outcomes were previously examined development labels.",
        "A paid paired-correctness sign is not a guaranteed macro-F1 sign or next-block sign.",
        "The final predicted day has no frozen current-day paid sample and is excluded.",
        "Five seeds overlap within each home; final-window labels/features untouched.",
        "Home07 beneficial transition 1 and harmful transition 3 (first ten blocks):",
        home07.round(2).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
