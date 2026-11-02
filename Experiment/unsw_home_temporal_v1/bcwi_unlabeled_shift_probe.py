"""Read-only source-prediction shift versus BCWI failure on development days.

Uses only official-device membership and frozen source predictions to measure
adjacent-day total variation. Full-day BCWI F1 is a post-hoc outcome, never a
detector input. Seeing a whole day's unlabeled distribution is not an online
decision rule; this checks signal feasibility only. No final labels/features.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from online_target_update import read_membership
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent


def total_variation(a, b):
    if not len(a) or not len(b):
        raise ValueError("Empty curated day")
    ca = np.bincount(a, minlength=24).astype(float) / len(a)
    cb = np.bincount(b, minlength=24).astype(float) / len(b)
    if len(ca) != 24 or len(cb) != 24:
        raise ValueError("Source prediction outside 24-class inventory")
    return float(0.5 * np.abs(ca - cb).sum())


def frozen_outcomes(root):
    parts = []
    for seed in range(5):
        path = root / f"seed_{seed}" / "evaluation_all" / "per_day.csv"
        part = pd.read_csv(path)
        part = part.loc[part.method == "bcwi_050", [
            "home", "seed", "transition", "tomorrow", "delta_vs_source_pp"]]
        if (set(part.home) != set(PRIMARY) or set(part.seed) != {seed} or
                part.duplicated(["home", "transition"]).any()):
            raise ValueError(f"Frozen BCWI development layout differs: {path}")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def probe(archive, windows, members, official, baseline_root, bcwi_root,
          route_root):
    outcomes = frozen_outcomes(bcwi_root)
    rows = []
    for home in PRIMARY:
        mapping = (official.loc[official.home == home]
                   .set_index("mac").device_type.to_dict())
        days, curated = read_membership(
            archive, members.loc[home, "member"], home, windows, set(mapping))
        base = baseline_root / home
        source = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
        if len(source) != len(days):
            raise ValueError(f"Source prediction alignment differs: {home}")
        ordered = np.unique(days)
        transitions = len(ordered) - 1
        for seed in range(5):
            folder = bcwi_root / f"seed_{seed}" / home
            gate = pd.read_csv(route_root / f"seed_{seed}" / home /
                               "decisions.csv")
            if len(gate) != transitions:
                raise ValueError(f"Paid gate day coverage differs: {home} {seed}")
            scores = outcomes[(outcomes.home == home) &
                              (outcomes.seed == seed)].set_index("transition")
            if not np.array_equal(scores.index.sort_values(),
                                  np.arange(transitions)):
                raise ValueError(f"BCWI score day coverage differs: {home} {seed}")
            for t in range(1, transitions):
                before = folder / f"transition_{t - 1}"
                after = folder / f"transition_{t}"
                before_ids = np.load(verified(before, "tomorrow_indices.npy"))
                after_ids = np.load(verified(after, "tomorrow_indices.npy"))
                if (not np.array_equal(before_ids, np.flatnonzero(
                        days == ordered[t])) or
                        not np.array_equal(after_ids, np.flatnonzero(
                            days == ordered[t + 1]))):
                    raise ValueError(f"Day indices differ: {home} {seed} {t}")
                prior_pred = np.asarray(source[before_ids[curated[before_ids]]],
                                        dtype=np.int64)
                next_pred = np.asarray(source[after_ids[curated[after_ids]]],
                                       dtype=np.int64)
                tv = total_variation(prior_pred, next_pred)
                previous_gain = float(scores.loc[t - 1, "delta_vs_source_pp"])
                current_gain = float(scores.loc[t, "delta_vs_source_pp"])
                # The current day's paid class count is available only after
                # its labels are purchased, so it is descriptive, not a
                # pre-day detector feature.
                paid_classes_current = (int(gate.iloc[t + 1]
                                            .audit_present_classes)
                                        if t + 1 < transitions else None)
                rows.append(dict(
                    home=home, seed=seed, transition=t,
                    prior_day=ordered[t], current_day=ordered[t + 1],
                    prior_curated_rows=len(prior_pred),
                    current_curated_rows=len(next_pred),
                    source_prediction_tv=tv,
                    prior_bcwi_gain_pp=previous_gain,
                    current_bcwi_gain_pp=current_gain,
                    gain_change_pp=current_gain - previous_gain,
                    paid_classes_current_day=paid_classes_current))
        print(home, "unlabeled shift checked", flush=True)
    frame = pd.DataFrame(rows)
    # Source prediction TV is seed-independent. Validate instead of counting
    # five overlapping repair draws as independent shift observations.
    spread = frame.groupby(["home", "transition"]).source_prediction_tv.agg(
        lambda x: float(x.max() - x.min()))
    if (spread > 1e-12).any():
        raise ValueError("Source-prediction TV differs across repair seeds")
    day = frame.groupby(["home", "transition", "prior_day", "current_day"],
                        as_index=False).agg(
        source_prediction_tv=("source_prediction_tv", "first"),
        mean_prior_bcwi_gain_pp=("prior_bcwi_gain_pp", "mean"),
        mean_current_bcwi_gain_pp=("current_bcwi_gain_pp", "mean"),
        mean_gain_change_pp=("gain_change_pp", "mean"),
        mean_paid_classes_current_day=("paid_classes_current_day", "mean"),
        current_curated_rows=("current_curated_rows", "first"))
    return frame, day


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--bcwi", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_v1"))
    parser.add_argument("--route", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_paid_route_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_unlabeled_shift_probe_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    frame, day = probe(archive, windows, members, official,
                       args.baseline.resolve(strict=True),
                       args.bcwi.resolve(strict=True),
                       args.route.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed.csv", index=False)
    day.to_csv(output / "per_day.csv", index=False)
    home07 = day.loc[day.home == "home07"]
    largest = day.nlargest(10, "source_prediction_tv")
    drops = day.nsmallest(10, "mean_gain_change_pp")
    columns = ["home", "transition", "source_prediction_tv",
               "mean_prior_bcwi_gain_pp", "mean_current_bcwi_gain_pp",
               "mean_gain_change_pp", "mean_paid_classes_current_day"]
    report = "\n".join([
        "Read-only full-day unlabeled source-prediction shift feasibility probe.",
        "TV uses official curated rows and frozen source hard-prediction proportions only.",
        "Whole-day TV is not available before the day starts; this is not a deployable alarm.",
        "BCWI gains and paid class counts are post-hoc descriptors, not detector inputs.",
        "Five seeds repeat each home/day; TV is checked identical across seeds and days are summarized once.",
        "No final-window label or feature decoded. Development outcomes had been examined earlier.",
        "Home07 transitions:", home07[columns].round(4).to_string(index=False),
        "Largest prediction-TV transitions:",
        largest[columns].round(4).to_string(index=False),
        "Largest BCWI gain drops:",
        drops[columns].round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
