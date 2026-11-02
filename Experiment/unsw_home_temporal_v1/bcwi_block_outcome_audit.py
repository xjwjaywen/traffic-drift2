"""Post-hoc BCWI versus source outcomes after within-day unlabeled shifts.

All development labels in this script are evaluation outcomes. It evaluates
each timestamp-ordered 1000-flow block and counterfactual mid-day switches;
none of these switches is an available decision rule. Final window unopened.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from bcwi_early_shift_probe import development_stamps
from bcwi_unlabeled_shift_probe import frozen_outcomes
from f1_feasibility import macro_f1
from online_transport_feasibility import development_days_and_labels
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
CLASSES = np.arange(24)
BLOCK = 1000


def matrix(truth, pred):
    if (len(truth) != len(pred) or np.any((truth < 0) | (truth >= 24)) or
            np.any((pred < 0) | (pred >= 24))):
        raise ValueError("Invalid curated truth/prediction pair")
    return np.bincount(truth * 24 + pred,
                       minlength=24 * 24).reshape(24, 24)


def matrix_macro_f1(confusion):
    tp = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted,
                   out=np.zeros(24, dtype=float),
                   where=(support + predicted) > 0)
    return float(f1.mean())


def audit(archive, windows, members, official, baseline_root, bcwi_root,
          blocks_root):
    frozen = frozen_outcomes(bcwi_root).set_index(
        ["home", "seed", "transition"])
    block_audit = pd.read_csv(blocks_root / "per_block.csv")
    if (set(block_audit.home) != set(PRIMARY) or
            block_audit.duplicated(["home", "transition", "block"]).any()):
        raise ValueError("Within-day block audit layout differs")
    rows = []
    for home in PRIMARY:
        mapping = (official.loc[official.home == home]
                   .set_index("mac").device_type.to_dict())
        days, y_all = development_days_and_labels(
            archive, members.loc[home, "member"], home, windows, mapping)
        stamps = development_stamps(
            archive, members.loc[home, "member"], home, windows, days)
        base = baseline_root / home
        source_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
        if len(source_all) != len(days):
            raise ValueError(f"Source prediction alignment differs: {home}")
        ordered = np.unique(days)
        for t in range(1, len(ordered) - 1):
            tomorrow = ordered[t + 1]
            tomorrow_ids = np.flatnonzero(days == tomorrow)
            curated_ids = np.flatnonzero((days == tomorrow) & (y_all >= 0))
            curated_ids = curated_ids[np.argsort(stamps[curated_ids],
                                                 kind="stable")]
            expected = block_audit.loc[(block_audit.home == home) &
                                       (block_audit.transition == t)].sort_values(
                                           "block")
            nblocks = len(curated_ids) // BLOCK
            if (len(expected) != nblocks or
                    not np.array_equal(expected.block.to_numpy(),
                                       np.arange(nblocks)) or
                    not np.array_equal(expected.observed_curated_rows.to_numpy(),
                                       BLOCK * np.arange(1, nblocks + 1)) or
                    not np.array_equal(expected.end_timestamp.to_numpy(),
                                       [stamps[curated_ids[(b + 1) * BLOCK - 1]].decode()
                                        for b in range(nblocks)])):
                raise ValueError(f"Block identities differ: {home} {t}")
            local = np.searchsorted(tomorrow_ids, curated_ids)
            if (np.any(local >= len(tomorrow_ids)) or
                    not np.array_equal(tomorrow_ids[local], curated_ids)):
                raise ValueError(f"Curated rows do not align: {home} {t}")
            truth = np.asarray(y_all[curated_ids], dtype=np.int64)
            source = np.asarray(source_all[curated_ids], dtype=np.int64)
            source_confusion = matrix(truth, source)
            source_f1 = matrix_macro_f1(source_confusion)
            for seed in range(5):
                folder = bcwi_root / f"seed_{seed}" / home / f"transition_{t}"
                frozen_ids = np.load(verified(folder, "tomorrow_indices.npy"))
                if not np.array_equal(frozen_ids, tomorrow_ids):
                    raise ValueError(f"Frozen BCWI indices differ: {home} {seed} {t}")
                bcwi_all = np.load(verified(folder, "bcwi_050.npy"),
                                   mmap_mode="r")
                if len(bcwi_all) != len(tomorrow_ids):
                    raise ValueError(f"Frozen BCWI prediction length differs: {home} {seed} {t}")
                bcwi = np.asarray(bcwi_all[local], dtype=np.int64)
                bcwi_confusion = matrix(truth, bcwi)
                bcwi_f1 = matrix_macro_f1(bcwi_confusion)
                if (not np.isclose(source_f1, macro_f1(truth, source, CLASSES),
                                   rtol=0, atol=1e-12) or
                        not np.isclose(bcwi_f1, macro_f1(truth, bcwi, CLASSES),
                                       rtol=0, atol=1e-12) or
                        not np.isclose(100 * (bcwi_f1 - source_f1),
                                       frozen.loc[(home, seed, t),
                                                  "delta_vs_source_pp"],
                                       rtol=0, atol=1e-9)):
                    raise ValueError(f"Recomputed full-day BCWI score differs: {home} {seed} {t}")
                bcwi_prefix = np.zeros((24, 24), dtype=np.int64)
                source_suffix = source_confusion.copy()
                for b in range(nblocks):
                    start, stop = b * BLOCK, (b + 1) * BLOCK
                    block_y = truth[start:stop]
                    block_source = source[start:stop]
                    block_bcwi = bcwi[start:stop]
                    bcwi_prefix += matrix(block_y, block_bcwi)
                    source_suffix -= matrix(block_y, block_source)
                    route_confusion = bcwi_prefix + source_suffix
                    if np.any(route_confusion < 0):
                        raise ValueError("Counterfactual confusion count negative")
                    route_f1 = matrix_macro_f1(route_confusion)
                    rows.append(dict(
                        home=home, seed=seed, transition=t, block=b,
                        end_timestamp=expected.iloc[b].end_timestamp,
                        remaining_curated_rows=len(curated_ids) - stop,
                        block_tv_vs_prior_day=expected.iloc[b].block_tv_vs_prior_day,
                        cumulative_tv_vs_prior_day=expected.iloc[b].cumulative_tv_vs_prior_day,
                        source_only_correct=int(np.sum((block_source == block_y) &
                                                       (block_bcwi != block_y))),
                        bcwi_only_correct=int(np.sum((block_bcwi == block_y) &
                                                     (block_source != block_y))),
                        net_bcwi_correct=int(np.sum(block_bcwi == block_y) -
                                             np.sum(block_source == block_y)),
                        full_day_bcwi_gain_pp=100 * (bcwi_f1 - source_f1),
                        switch_after_block_gain_vs_bcwi_pp=100 *
                        (route_f1 - bcwi_f1)))
        print(home, "block outcomes checked", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--bcwi", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_v1"))
    parser.add_argument("--blocks", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_block_shift_probe_v1"))
    parser.add_argument("--output", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_block_outcome_audit_v1"))
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
    frame = audit(archive, windows, members, official,
                  args.baseline.resolve(strict=True), args.bcwi.resolve(strict=True),
                  args.blocks.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_block.csv", index=False)
    home07 = frame.loc[(frame.home == "home07") &
                       frame.transition.isin([1, 3])].groupby(
                           ["transition", "block", "end_timestamp",
                            "remaining_curated_rows", "block_tv_vs_prior_day"],
                           as_index=False).agg(
                               mean_source_only_correct=("source_only_correct", "mean"),
                               mean_bcwi_only_correct=("bcwi_only_correct", "mean"),
                               mean_net_bcwi_correct=("net_bcwi_correct", "mean"),
                               mean_switch_gain_vs_bcwi_pp=(
                                   "switch_after_block_gain_vs_bcwi_pp", "mean"))
    report = "\n".join([
        "Post-hoc BCWI/source block-outcome audit on examined development days.",
        "Each block has 1000 timestamp-ordered curated flows; frozen BCWI scores are reproduced.",
        "Block correctness and counterfactual switch F1 use full development labels.",
        "The switch location is not selected by an available rule; gains are contaminated ceilings.",
        "Five seeds overlap within each home. No final-window label or feature decoded.",
        "Home07 transitions 1 and 3, mean over overlapping seeds:",
        home07.round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
