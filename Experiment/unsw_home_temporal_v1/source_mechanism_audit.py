"""Source-only audit of class composition and within-class feature change.

All labels come from the existing source reservoir. The archive is streamed
only to recover dates at retained source row IDs. Other rows are traversed as
bytes, but their timestamps, features, and labels are never decoded. This is
a descriptive mechanism audit, not a target adaptation result or shift test.
"""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from audit import sha256
from invariant_probe import PRIMARY, source_blocks


MAX_KS_ROWS = 1000
MIN_KS_ROWS = 100
MIN_WITHIN_ROWS = 30
SMOOTHING = 0.5


def retained_dates(archive, member, ids, source_days):
    """Match prepared row IDs; decode no timestamp unless its ID is retained."""
    if len(ids) == 0 or len(np.unique(ids)) != len(ids) or np.min(ids) < 0:
        raise ValueError("Invalid retained source row IDs")
    lookup = {int(row): position for position, row in enumerate(ids)}
    dates = np.full(len(ids), "", dtype="U10")
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        if not stream.readline().decode("utf-8-sig").startswith("start,"):
            raise ValueError(f"Unexpected CSV header: {member}")
        for row_id in range(int(np.max(ids)) + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"Archive ended before retained source row: {member}")
            position = lookup.get(row_id)
            if position is not None:
                day = line[:10].decode("ascii")
                if day not in source_days:
                    raise ValueError(f"Retained source ID has later date: {member} {row_id}")
                dates[position] = day
    if np.any(dates == ""):
        raise ValueError(f"Unmatched retained source IDs: {member}")
    return dates


def capped_rows(x, mask, seed):
    indices = np.flatnonzero(mask)
    if len(indices) > MAX_KS_ROWS:
        indices = np.random.default_rng(seed).choice(
            indices, MAX_KS_ROWS, replace=False)
    return x[indices]


def mean_feature_ks(left, right):
    """Distributional effect size, not a p-value or proof of stable conditionals."""
    return float(np.mean([ks_2samp(left[:, j], right[:, j], method="asymp").statistic
                          for j in range(left.shape[1])]))


def conditional_effects(features, labels, dates, day_groups, home_index):
    rows = []
    for label in range(24):
        for earlier in (0, 1):
            later = earlier + 1
            old_days, new_days = day_groups[earlier], day_groups[later]
            old_mask = (labels == label) & np.isin(dates, old_days)
            new_mask = (labels == label) & np.isin(dates, new_days)
            old_count, new_count = int(old_mask.sum()), int(new_mask.sum())
            result = dict(device_type=label, earlier_block=earlier,
                          later_block=later, earlier_rows=old_count,
                          later_rows=new_count, cross_block_mean_ks=np.nan,
                          within_block_mean_ks=np.nan, within_control_blocks=0)
            if min(old_count, new_count) < MIN_KS_ROWS:
                rows.append(result)
                continue
            seed = 10000 * home_index + 100 * label + 10 * earlier
            old = capped_rows(features, old_mask, seed)
            new = capped_rows(features, new_mask, seed + 1)
            result["cross_block_mean_ks"] = mean_feature_ks(old, new)
            controls = []
            for block, days in ((earlier, old_days), (later, new_days)):
                split = len(days) // 2
                first_mask = (labels == label) & np.isin(dates, days[:split])
                second_mask = (labels == label) & np.isin(dates, days[split:])
                if min(int(first_mask.sum()), int(second_mask.sum())) < MIN_WITHIN_ROWS:
                    continue
                first = capped_rows(features, first_mask, seed + 2 + 2 * block)
                second = capped_rows(features, second_mask, seed + 3 + 2 * block)
                controls.append(mean_feature_ks(first, second))
            if controls:
                result["within_block_mean_ks"] = float(np.mean(controls))
                result["within_control_blocks"] = len(controls)
            rows.append(result)
    return pd.DataFrame(rows)


def composition(counts, raw_totals):
    frame = counts.merge(raw_totals[["device_type", "source_rows"]],
                         on="device_type", validate="one_to_one")
    retained = frame[[f"block_{i}" for i in range(3)]].sum(axis=1)
    if (retained == 0).any() or (frame.source_rows < retained).any():
        raise ValueError("Raw and retained source counts are incompatible")
    frame["retained_total"] = retained
    frame["source_sampling_weight"] = frame.source_rows / retained
    for block in range(3):
        frame[f"estimated_curated_rows_{block}"] = (
            frame[f"block_{block}"] * frame.source_sampling_weight)
        total = frame[f"estimated_curated_rows_{block}"].sum()
        frame[f"estimated_share_{block}"] = (
            (frame[f"estimated_curated_rows_{block}"] + SMOOTHING) /
            (total + 24 * SMOOTHING))
        frame[f"sampled_zero_{block}"] = frame[f"block_{block}"] == 0
        frame[f"zero_is_exact_{block}"] = (
            frame[f"sampled_zero_{block}"] &
            (frame.source_rows == frame.retained_total))
    for earlier in (0, 1):
        later = earlier + 1
        frame[f"log2_share_ratio_{earlier}_{later}"] = np.log2(
            frame[f"estimated_share_{later}"] /
            frame[f"estimated_share_{earlier}"])
    return frame


def run_home(archive, prepared, metadata, output, home, archive_hash):
    folder = output / home
    if folder.is_symlink():
        raise ValueError(f"Output is a symlink: {folder}")
    source_file = prepared / home / "source.npz"
    prepared_config = json.loads((prepared / home / "config.json").read_text())
    prepared_marker = json.loads((prepared / home / "complete.json").read_text())
    source_hash = sha256(source_file)
    if prepared_config["archive_sha256"] != archive_hash or (
            prepared_marker["artifacts"]["source.npz"] != source_hash):
        raise ValueError(f"Prepared source verification failed: {home}")
    signature = dict(home=home, archive_sha256=archive_hash,
                     source_sha256=source_hash,
                     code_sha256=sha256(Path(__file__)),
                     metadata_sha256={name: sha256(metadata / name) for name in
                                      ("windows.csv", "daily_rows.csv",
                                       "official_source_counts.csv")},
                     max_ks_rows=MAX_KS_ROWS, min_ks_rows=MIN_KS_ROWS,
                     min_within_rows=MIN_WITHIN_ROWS, smoothing=SMOOTHING)
    marker = folder / "complete.json"
    if marker.is_file():
        prior = json.loads(marker.read_text())
        if prior["signature"] != signature or any(
                sha256(folder / name) != digest for name, digest in
                prior["artifacts"].items()):
            raise ValueError(f"Completed mechanism audit differs: {home}")
        print(home, "verified completed source audit", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete mechanism audit output: {folder}")
    with np.load(source_file, allow_pickle=False) as data:
        ids, raw_x, labels = data["ids"], data["x"], data["y"]
    day_to_block, day_groups = source_blocks(home, metadata)
    dates = retained_dates(archive, prepared_config["member"], ids, day_to_block)
    blocks = np.asarray([day_to_block[day] for day in dates], np.int8)
    counts = pd.DataFrame([dict(device_type=c, **{f"block_{i}": int(np.sum(
        (labels == c) & (blocks == i))) for i in range(3)}) for c in range(24)])
    raw_totals = pd.read_csv(metadata / "official_source_counts.csv")
    raw_totals = raw_totals.loc[raw_totals.home == home]
    composition_frame = composition(counts, raw_totals)
    features = Preprocessor().fit(raw_x).transform(raw_x)
    conditional = conditional_effects(features, labels, dates, day_groups,
                                      int(home[-2:]))
    composition_frame.insert(0, "home", home)
    conditional.insert(0, "home", home)
    folder.mkdir(parents=True, exist_ok=True)
    composition_frame.to_csv(folder / "composition.csv", index=False)
    conditional.to_csv(folder / "conditional.csv", index=False)
    artifacts = {name: sha256(folder / name) for name in
                 ("composition.csv", "conditional.csv")}
    marker.write_text(json.dumps(dict(signature=signature, artifacts=artifacts),
                                 indent=2) + "\n")
    print(home, "source composition/conditional audit complete", flush=True)


def report(output, review_file):
    composition_frame = pd.concat([pd.read_csv(output / h / "composition.csv")
                                   for h in PRIMARY], ignore_index=True)
    conditional = pd.concat([pd.read_csv(output / h / "conditional.csv")
                             for h in PRIMARY], ignore_index=True)
    later = composition_frame.loc[(composition_frame.block_0 == 0) &
                                  (composition_frame.block_1 >= 100) &
                                  (composition_frame.block_2 >= 100)]
    both = conditional.loc[(conditional.earlier_block == 1) &
                           conditional.cross_block_mean_ks.notna()]
    with_control = both.loc[both.within_block_mean_ks.notna()].copy()
    with_control["cross_minus_within_ks"] = (
        with_control.cross_block_mean_ks - with_control.within_block_mean_ks)
    lines = ["UNSW source-window shift mechanism audit; no target-period labels read.",
             "Class shares are estimated from the class-capped uniform reservoir,",
             "weighted by the reviewed full source class totals. They are not exact",
             "for capped classes; zero retained rows need not mean physical absence.",
             "Mean KS across 56 preprocessed coordinates is a descriptive effect size;",
             "within-block day-split KS is a variability reference, not a formal null.",
             f"Class/home pairs with 0 retained block-0 rows, >=100 in blocks 1 and 2: {len(later)}",
             f"Of these, exact block-0 zero (uncapped source): {int(later.zero_is_exact_0.sum())}",
             f"Pairs with >=100 rows in both blocks 1 and 2: {len(both)}",
             f"Pairs also having within-block day controls: {len(with_control)}",
             "Block 1-to-2 conditional KS, eligible pairs:",
             both[["cross_block_mean_ks", "within_block_mean_ks"]]
             .describe().round(3).to_string(),
             "Cross-minus-within KS where control exists:",
             with_control.cross_minus_within_ks.describe().round(3).to_string()]
    if review_file.is_file():
        review = pd.read_csv(review_file)
        top = review.reindex(review.macro_contribution_pp.abs().sort_values(
            ascending=False).index).head(12)
        top = top[["home", "device_type", "device_name", "macro_contribution_pp"]]
        top = top.merge(composition_frame, on=["home", "device_type"],
                        validate="one_to_one")
        top = top.merge(conditional.loc[conditional.earlier_block == 1, [
            "home", "device_type", "cross_block_mean_ks", "within_block_mean_ks"]],
            on=["home", "device_type"], validate="one_to_one")
        lines += ["Largest prior projection-effect pairs with source shift diagnostics:",
                  top[["home", "device_type", "device_name", "block_0", "block_1",
                       "block_2", "zero_is_exact_0", "log2_share_ratio_1_2",
                       "cross_block_mean_ks", "within_block_mean_ks",
                       "macro_contribution_pp"]].round(3).to_string(index=False)]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print((output / "REPORT.txt").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--metadata", type=Path, default=HERE / "metadata")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_source_mechanism_v1"))
    parser.add_argument("--review", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_invariant_probe_v1/class_home_review.csv"))
    parser.add_argument("--home", choices=PRIMARY)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    prepared = args.prepared.resolve(strict=True)
    metadata = args.metadata.resolve(strict=True)
    output = args.output.resolve()
    expected = json.loads((metadata / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    if output.is_symlink():
        parser.error("Output must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    homes = [args.home] if args.home else PRIMARY
    for home in homes:
        run_home(archive, prepared, metadata, output, home, expected)
    if not args.home:
        report(output, args.review)
    print("SOURCE MECHANISM AUDIT COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
