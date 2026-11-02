"""Read-only early-day unlabeled shift feasibility audit.

The first 250 and 1000 curated flows of a day are chosen by timestamp and
compared with the preceding full day's frozen source predictions. BCWI gains
come from a previously completed development audit and are post-hoc outcomes.
This does not choose a threshold or evaluate a deployable route.
"""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import sha256
from bcwi_unlabeled_shift_probe import total_variation
from online_target_update import read_membership
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
PREFIXES = (250, 1000)


def development_stamps(archive, member, home, windows, expected_days):
    """Read timestamp fields only, retaining original development row order."""
    after_query = str(windows.loc[(home, "query"), "last_day"])
    after_dev = str(windows.loc[(home, "development"), "last_day"])
    stamps = []
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {home}")
        for line in stream:
            stamp = line.split(b",", 1)[0]
            day = stamp[:10].decode("ascii")
            if after_query < day <= after_dev:
                stamps.append(stamp)
    if len(stamps) != len(expected_days):
        raise ValueError(f"Development timestamp count differs: {home}")
    values = np.asarray(stamps, dtype="S64")
    if any(len(stamp) > 64 for stamp in stamps):
        raise ValueError(f"Timestamp exceeds fixed sorting width: {home}")
    if not np.array_equal(np.asarray([stamp[:10].decode("ascii")
                                      for stamp in stamps]), expected_days):
        raise ValueError(f"Development timestamps differ from membership: {home}")
    return values


def probe(archive, windows, members, official, baseline_root, full):
    rows = []
    if (set(full.home) != set(PRIMARY) or
            full.duplicated(["home", "transition"]).any()):
        raise ValueError("Full-day shift audit layout differs")
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
            current_order = np.argsort(stamps[current_ids], kind="stable")
            current_ids = current_ids[current_order]
            if len(current_ids) < max(PREFIXES):
                raise ValueError(f"Too few curated flows for early probe: {home} {t}")
            prior_pred = np.asarray(source[prior_ids], dtype=np.int64)
            current_pred = np.asarray(source[current_ids], dtype=np.int64)
            full_tv = total_variation(prior_pred, current_pred)
            audited = home_full.loc[t]
            if (audited.prior_day != ordered[t] or
                    audited.current_day != ordered[t + 1] or
                    not np.isclose(full_tv, audited.source_prediction_tv,
                                   rtol=0, atol=1e-12)):
                raise ValueError(f"Full-day shift audit differs: {home} {t}")
            row = dict(home=home, transition=t, prior_day=ordered[t],
                       current_day=ordered[t + 1],
                       current_curated_rows=len(current_ids),
                       full_day_tv=full_tv,
                       mean_prior_bcwi_gain_pp=audited.mean_prior_bcwi_gain_pp,
                       mean_current_bcwi_gain_pp=audited.mean_current_bcwi_gain_pp,
                       mean_gain_change_pp=audited.mean_gain_change_pp)
            for n in PREFIXES:
                row[f"early_tv_{n}"] = total_variation(
                    prior_pred, current_pred[:n])
                row[f"time_at_{n}"] = stamps[current_ids[n - 1]].decode()
            rows.append(row)
        print(home, "early unlabeled shift checked", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--full-probe", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_unlabeled_shift_probe_v1/per_day.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "/data/xjw/tma-care-data/unsw_bcwi_early_shift_probe_v1"))
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
    frame.to_csv(output / "per_day.csv", index=False)
    columns = ["home", "transition", "early_tv_250", "early_tv_1000",
               "full_day_tv", "mean_prior_bcwi_gain_pp",
               "mean_current_bcwi_gain_pp", "mean_gain_change_pp"]
    report = "\n".join([
        "Exploratory early-day unlabeled source-prediction shift audit.",
        "The first 250/1000 curated flows are selected by timestamp; no target labels enter TV.",
        "The prior full day is available before the current day starts; prefixes require waiting for those flows.",
        "BCWI gain is an already examined, post-hoc development outcome; no threshold or route is selected.",
        "Homes and adjacent days are dependent; no final-window label or feature decoded.",
        "Home07 transitions:",
        frame.loc[frame.home == "home07", columns].round(4).to_string(index=False),
        "Largest early-1000 TV transitions:",
        frame.nlargest(10, "early_tv_1000")[columns].round(4).to_string(index=False),
        "Largest BCWI gain drops:",
        frame.nsmallest(10, "mean_gain_change_pp")[columns].round(4).to_string(index=False),
        "Early/full TV Spearman correlation (descriptive):",
        str(round(frame.early_tv_1000.corr(frame.full_day_tv,
                                           method="spearman"), 4)),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
