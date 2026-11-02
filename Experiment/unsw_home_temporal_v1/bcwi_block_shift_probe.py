"""Read-only within-day source-prediction shift timing audit.

Compares successive timestamp-ordered blocks of 1000 curated flows with the
previous full day. The frozen source predictions are label-free; the BCWI gain
columns are previously examined development outcomes, included only to judge
whether a signal appears early enough for a later online policy.
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


def distribution(pred):
    if len(pred) == 0 or np.any((pred < 0) | (pred >= 24)):
        raise ValueError("Empty or out-of-inventory source prediction block")
    return np.bincount(pred, minlength=24).astype(float) / len(pred)


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())


def probe(archive, windows, members, official, baseline_root, full):
    if (set(full.home) != set(PRIMARY) or
            full.duplicated(["home", "transition"]).any()):
        raise ValueError("Completed full-day probe coverage differs")
    rows = []
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
            raise ValueError(f"Source prediction alignment differs: {home}")
        ordered = np.unique(days)
        home_full = full.loc[full.home == home].set_index("transition")
        if not np.array_equal(home_full.index.sort_values(),
                              np.arange(1, len(ordered) - 1)):
            raise ValueError(f"Full-day transition coverage differs: {home}")
        for t in range(1, len(ordered) - 1):
            prior_ids = np.flatnonzero((days == ordered[t]) & curated)
            current_ids = np.flatnonzero((days == ordered[t + 1]) & curated)
            current_ids = current_ids[np.argsort(stamps[current_ids],
                                                  kind="stable")]
            prior_distribution = distribution(np.asarray(source[prior_ids],
                                                         dtype=np.int64))
            current_pred = np.asarray(source[current_ids], dtype=np.int64)
            full_tv = tv(prior_distribution, distribution(current_pred))
            audited = home_full.loc[t]
            if (audited.prior_day != ordered[t] or
                    audited.current_day != ordered[t + 1] or
                    not np.isclose(full_tv, audited.source_prediction_tv,
                                   rtol=0, atol=1e-12)):
                raise ValueError(f"Full-day audit differs: {home} {t}")
            block_count = len(current_ids) // BLOCK
            if block_count == 0:
                raise ValueError(f"No full 1000-flow block: {home} {t}")
            first_distribution = distribution(current_pred[:BLOCK])
            running_counts = np.zeros(24, dtype=np.int64)
            for b in range(block_count):
                start, stop = b * BLOCK, (b + 1) * BLOCK
                block_counts = np.bincount(current_pred[start:stop],
                                           minlength=24)
                running_counts += block_counts
                block_distribution = block_counts / BLOCK
                rows.append(dict(
                    home=home, transition=t, block=b,
                    current_day=ordered[t + 1],
                    end_timestamp=stamps[current_ids[stop - 1]].decode(),
                    observed_curated_rows=stop,
                    remaining_curated_rows=len(current_ids) - stop,
                    block_tv_vs_prior_day=tv(block_distribution,
                                              prior_distribution),
                    cumulative_tv_vs_prior_day=tv(running_counts / stop,
                                                   prior_distribution),
                    block_tv_vs_first=tv(block_distribution,
                                          first_distribution),
                    full_day_tv=full_tv,
                    mean_current_bcwi_gain_pp=audited.mean_current_bcwi_gain_pp,
                    mean_gain_change_pp=audited.mean_gain_change_pp))
        print(home, "within-day unlabeled blocks checked", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--full-probe", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_unlabeled_shift_probe_v1/per_day.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_block_shift_probe_v1"))
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
    full = pd.read_csv(args.full_probe.resolve(strict=True))
    frame = probe(archive, windows, members, official,
                  args.baseline.resolve(strict=True), full)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_block.csv", index=False)
    summary = frame.groupby(["home", "transition"], as_index=False).agg(
        blocks=("block", "count"),
        first_block_tv=("block_tv_vs_prior_day", "first"),
        max_later_block_tv=("block_tv_vs_prior_day",
                            lambda x: x.iloc[1:].max() if len(x) > 1 else np.nan),
        full_day_tv=("full_day_tv", "first"),
        mean_current_bcwi_gain_pp=("mean_current_bcwi_gain_pp", "first"),
        mean_gain_change_pp=("mean_gain_change_pp", "first"))
    summary.to_csv(output / "per_day.csv", index=False)
    columns = ["home", "transition", "block", "end_timestamp",
               "observed_curated_rows", "remaining_curated_rows",
               "block_tv_vs_prior_day", "cumulative_tv_vs_prior_day",
               "block_tv_vs_first", "mean_current_bcwi_gain_pp"]
    report = "\n".join([
        "Exploratory within-day unlabeled source-prediction shift timing audit.",
        "Each block is 1000 curated flows in timestamp order; only complete blocks are shown.",
        "Previous full-day source predictions are the reference; no target labels enter TV.",
        "BCWI gains are post-hoc examined outcomes. No alarm threshold or routed score is estimated.",
        "Full-day shifts cannot be assumed visible in the first block. Final-window labels/features untouched.",
        "Home07 transitions 1 and 3:",
        frame.loc[(frame.home == "home07") &
                  frame.transition.isin([1, 3]), columns].round(4).to_string(index=False),
        "Largest BCWI gain-drop days, with first/max-later block TV:",
        summary.nsmallest(10, "mean_gain_change_pp").round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
