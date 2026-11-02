"""Test whether yesterday's CARE gain predicts tomorrow's gain.

Read-only exploratory replay on the 4–6 already examined development days.
CARE/source predictions were frozen before development labels were opened.
The previous day's full labels are an oracle diagnostic; a separate random
1000-label audit is the usable information pattern. No final-window label or
feature is decoded, and no route or new model is fitted.
"""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import sha256
from f1_exact_count_audit import CLASSES, paired_delta
from f1_feasibility import macro_f1
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent


def development_days_and_labels(archive, member, home, windows, mapping):
    first = str(windows.loc[(home, "query"), "last_day"])
    last = str(windows.loc[(home, "development"), "last_day"])
    expected = int(windows.loc[(home, "development"), "rows"])
    day_values = []
    labels = []
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {member}")
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= first or day > last:
                continue
            mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
            day_values.append(day)
            labels.append(mapping.get(mac, -1))
    if len(labels) != expected:
        raise ValueError(f"Development rows differ from timestamp audit: {home}")
    return np.asarray(day_values), np.asarray(labels, dtype=np.int64)


def sign(value):
    return int(np.sign(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--audit-labels", type=int, default=1000)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_transport_feasibility_v1"))
    args = parser.parse_args()
    if args.audit_labels < 100:
        parser.error("Audit labels must be at least 100")
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    reviewed = json.loads((manifest / "audit.json").read_text())
    if sha256(archive) != reviewed["archive_sha256"]:
        parser.error("Archive differs from reviewed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    base_root = args.baseline.resolve(strict=True)
    homes = [args.home] if args.home else PRIMARY
    rows = []
    for home_index, home in enumerate(homes):
        mapping = (official.loc[official.home == home]
                   .set_index("mac").device_type.to_dict())
        days_all, y_all = development_days_and_labels(
            archive, members.loc[home, "member"], home, windows, mapping)
        curated = y_all >= 0
        days, y = days_all[curated], y_all[curated]
        ordered_days = np.unique(days)
        expected_days = int(windows.loc[(home, "development"), "observed_days"])
        if len(ordered_days) != expected_days:
            raise ValueError(f"Curated development day coverage differs: {home}")
        base = base_root / home
        source_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
        if len(source_all) != len(y_all):
            raise ValueError(f"Frozen source prediction length differs: {home}")
        source = np.asarray(source_all[curated], dtype=np.int64)
        for seed in range(5):
            candidate_all = np.load(verified(base / f"seed_{seed}", "care.npy"),
                                    mmap_mode="r")
            if len(candidate_all) != len(y_all):
                raise ValueError(f"Frozen CARE prediction length differs: {home} {seed}")
            candidate = np.asarray(candidate_all[curated], dtype=np.int64)
            for transition, (today, tomorrow) in enumerate(zip(
                    ordered_days[:-1], ordered_days[1:])):
                today_mask, tomorrow_mask = days == today, days == tomorrow
                today_y, tomorrow_y = y[today_mask], y[tomorrow_mask]
                today_source, tomorrow_source = source[today_mask], source[tomorrow_mask]
                today_candidate, tomorrow_candidate = (
                    candidate[today_mask], candidate[tomorrow_mask])
                if len(today_y) < args.audit_labels:
                    raise ValueError(f"Too few curated audit rows: {home} {today}")
                full_today = 100 * (
                    macro_f1(today_y, today_candidate, CLASSES) -
                    macro_f1(today_y, today_source, CLASSES))
                full_tomorrow = 100 * (
                    macro_f1(tomorrow_y, tomorrow_candidate, CLASSES) -
                    macro_f1(tomorrow_y, tomorrow_source, CLASSES))
                rng = np.random.default_rng(
                    990000 + 1000 * home_index + 100 * seed + transition)
                paid = rng.choice(len(today_y), args.audit_labels, replace=False)
                p0 = np.bincount(today_source, minlength=24).astype(np.float64)
                p1 = np.bincount(today_candidate, minlength=24).astype(np.float64)
                weights = np.full((1, args.audit_labels),
                                  len(today_y) / args.audit_labels)
                paid_estimate = paired_delta(weights, today_y[paid],
                                             today_source[paid],
                                             today_candidate[paid],
                                             p0, p1, True)[0]
                today_pairs = today_source * 24 + today_candidate
                tomorrow_pairs = tomorrow_source * 24 + tomorrow_candidate
                paid_counts = np.bincount(today_pairs[paid], minlength=576)
                tomorrow_counts = np.bincount(tomorrow_pairs, minlength=576)
                today_counts = np.bincount(today_pairs, minlength=576)
                tomorrow_mass = tomorrow_counts / len(tomorrow_pairs)
                today_mass = today_counts / len(today_pairs)
                rows.append(dict(
                    home=home, seed=seed, today=today, tomorrow=tomorrow,
                    today_curated_rows=len(today_y),
                    tomorrow_curated_rows=len(tomorrow_y),
                    today_present_classes=len(np.unique(today_y)),
                    tomorrow_present_classes=len(np.unique(tomorrow_y)),
                    today_full_delta_pp=full_today,
                    today_audit_delta_pp=paid_estimate,
                    tomorrow_full_delta_pp=full_tomorrow,
                    oracle_direction_matches_next=sign(full_today) == sign(full_tomorrow),
                    audit_direction_matches_next=sign(paid_estimate) == sign(full_tomorrow),
                    pair_mass_tv=float(.5 * np.abs(today_mass-tomorrow_mass).sum()),
                    next_mass_with_lt_5_paid_pair=float(
                        tomorrow_mass[paid_counts < 5].sum()),
                    next_mass_in_unseen_pairs=float(
                        tomorrow_mass[today_counts == 0].sum()),
                ))
            print(home, "seed", seed, "daily transport diagnostic complete", flush=True)
    result = pd.DataFrame(rows)
    result["abs_day_delta_change_pp"] = (
        result.tomorrow_full_delta_pp - result.today_full_delta_pp).abs()
    per_home = (result.groupby("home", as_index=False)
                .agg(transitions_per_seed=("today", lambda x: len(x)//5),
                     oracle_direction_match_rate=("oracle_direction_matches_next", "mean"),
                     audit_direction_match_rate=("audit_direction_matches_next", "mean"),
                     mean_pair_mass_tv=("pair_mass_tv", "mean"),
                     mean_next_mass_lt_5_paid=("next_mass_with_lt_5_paid_pair", "mean"),
                     mean_next_mass_unseen=("next_mass_in_unseen_pairs", "mean"),
                     mean_abs_day_delta_change_pp=("abs_day_delta_change_pp", "mean")))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_transition.csv", index=False)
    per_home.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Exploratory frozen CARE/source day-to-next-day transport audit on examined development days.",
        "Full today and full tomorrow labels are oracle diagnostics; the usable estimate pays 1000 today labels.",
        "The challenger is fixed before development and never refitted here.",
        "Daily macro-F1 delta uses fixed 24-class inventory; compare direction and support as well as magnitude.",
        "Pair-mass TV uses predictions only. <5-paid and unseen-pair mass are coverage diagnostics, not accuracy.",
        "Five seeds per home overlap; no independent validation or deployable route is claimed.",
        "No final-window label or feature decoded.",
        "Per-home results:",
        per_home.round(4).to_string(index=False),
        "Mean across independent homes:",
        per_home.select_dtypes(include="number").mean().round(4).to_string(),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        homes=homes, archive_sha256=reviewed["archive_sha256"],
        audit_labels=args.audit_labels,
        implementation_sha256=sha256(Path(__file__)),
        final_window_read=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
