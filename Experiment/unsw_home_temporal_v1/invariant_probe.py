"""Source-only test of discriminative temporal-invariance directions.

The first two thirds of observed source days fit every projection and classifier.
The last third is evaluated once, using only source-window labels. Archive
bytes for intervening rows may be traversed, but only retained source-row
dates are decoded; later labels, features, and timestamps are not decoded.
This is a diagnostic of a necessary condition, not a target adaptation method.
"""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from audit import sha256


PRIMARY = [f"home{i:02d}" for i in range(1, 13) if i != 10]
CLASSES = 24
DIMENSIONS = (8, 16)
TEMPORAL_PENALTY = 4.0
MIN_BLOCK_SUPPORT = 30
RIDGE = 0.1


def source_blocks(home, metadata):
    windows = pd.read_csv(metadata / "windows.csv")
    source = windows.loc[(windows.home == home) & (windows.split == "source")]
    if len(source) != 1:
        raise ValueError(f"Missing source window for {home}")
    first, last = source.iloc[0][["first_day", "last_day"]]
    daily = pd.read_csv(metadata / "daily_rows.csv")
    days = sorted(daily.loc[(daily.home == home) &
                            (daily.day >= first) & (daily.day <= last), "day"].tolist())
    if len(days) != int(source.iloc[0].observed_days) or len(days) < 9:
        raise ValueError(f"Unexpected observed source days for {home}")
    groups = np.array_split(np.asarray(days), 3)
    return {str(day): i for i, group in enumerate(groups) for day in group}, [
        [str(day) for day in group] for group in groups]


def retained_source_blocks(archive, member, ids, day_to_block):
    """Decode dates only for retained source IDs, even if archive rows interleave."""
    if len(ids) == 0 or np.min(ids) < 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid source row IDs")
    lookup = {int(row): i for i, row in enumerate(ids)}
    blocks = np.full(len(ids), -1, np.int8)
    maximum = int(np.max(ids))
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        if not stream.readline().decode("utf-8-sig").startswith("start,"):
            raise ValueError(f"Unexpected CSV header: {member}")
        for row_id in range(maximum + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"Archive ended before retained source row: {member}")
            position = lookup.get(row_id)
            if position is not None:
                day = line[:10].decode("ascii")
                if day not in day_to_block:
                    raise ValueError(f"Prepared source row outside source days: {member} {row_id}")
                blocks[position] = day_to_block[day]
    if np.any(blocks < 0):
        raise ValueError(f"Source row IDs missing from archive: {member}")
    return blocks


def balanced_scatter(x, y, blocks):
    """Class-balanced Fisher scatter plus within-class temporal mean drift."""
    dimension = x.shape[1]
    eligible = []
    means = []
    within = np.zeros((dimension, dimension), np.float64)
    temporal = np.zeros_like(within)
    for label in range(CLASSES):
        left = x[(y == label) & (blocks == 0)]
        right = x[(y == label) & (blocks == 1)]
        if min(len(left), len(right)) < MIN_BLOCK_SUPPORT:
            continue
        rows = np.concatenate((left, right), axis=0)
        center = rows.mean(axis=0)
        centered = rows - center
        within += centered.T @ centered / len(rows)
        drift = left.mean(axis=0) - right.mean(axis=0)
        temporal += np.outer(drift, drift)
        eligible.append(label)
        means.append(center)
    if len(eligible) < 2:
        raise ValueError("Fewer than two classes have enough rows in both fit blocks")
    means = np.stack(means)
    between = np.cov(means, rowvar=False, bias=True)
    within /= len(eligible)
    temporal /= len(eligible)
    return between, within, temporal, eligible


def projection(between, within, temporal, penalty, dimension):
    regularized = within + penalty * temporal + RIDGE * np.eye(len(within))
    _, vectors = eigh(between, regularized, check_finite=True)
    return vectors[:, -dimension:]


def classify_projected(fit_x, fit_y, test_x, projector):
    """The same uniform-prior, ridge-pooled-covariance classifier for all views."""
    train = fit_x @ projector
    test = test_x @ projector
    present = np.unique(fit_y)
    centers = np.stack([train[fit_y == label].mean(axis=0) for label in present])
    residual = train - centers[np.searchsorted(present, fit_y)]
    covariance = residual.T @ residual / len(train)
    covariance += RIDGE * np.eye(covariance.shape[0])
    coefficients = np.linalg.solve(covariance, centers.T)
    scores = test @ coefficients - 0.5 * np.sum(centers * coefficients.T, axis=1)
    return present[np.argmax(scores, axis=1)].astype(np.int64)


def evaluate(y, prediction):
    _, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(CLASSES), zero_division=0)
    present = support > 0
    summary = dict(macro_f1_present=float(f1[present].mean()),
                   zero_recall_classes=int(np.sum(present & (recall == 0))),
                   present_classes=int(np.sum(present)),
                   evaluated_rows=int(len(y)))
    classes = [dict(device_type=int(c), support=int(support[c]),
                    recall=float(recall[c]), f1=float(f1[c]))
               for c in range(CLASSES)]
    return summary, classes


def heldout_shift_ratio(fit_x, fit_y, valid_x, valid_y, projector):
    """Label-conditioned late-source mean movement / fit class separation."""
    eligible = [c for c in range(CLASSES)
                if (fit_y == c).sum() >= MIN_BLOCK_SUPPORT and
                (valid_y == c).sum() >= MIN_BLOCK_SUPPORT]
    if len(eligible) < 2:
        return None, len(eligible)
    fit_means = np.stack([fit_x[fit_y == c].mean(0) for c in eligible]) @ projector
    valid_means = np.stack([valid_x[valid_y == c].mean(0) for c in eligible]) @ projector
    drift = np.mean(np.sum((valid_means - fit_means) ** 2, axis=1))
    separation = np.mean(np.sum((fit_means[:, None] - fit_means[None, :]) ** 2,
                                axis=2)[~np.eye(len(eligible), dtype=bool)])
    return float(drift / max(separation, 1e-12)), len(eligible)


def probe_arrays(raw_x, y, blocks):
    if len(raw_x) != len(y) or len(y) != len(blocks) or raw_x.ndim != 2:
        raise ValueError("Source array dimensions differ")
    if np.any((y < 0) | (y >= CLASSES)) or set(np.unique(blocks)) != {0, 1, 2}:
        raise ValueError("Unexpected source labels or time blocks")
    fit_mask = blocks < 2
    prep = Preprocessor().fit(raw_x[fit_mask])
    fit_x = prep.transform(raw_x[fit_mask]).astype(np.float64)
    valid_x = prep.transform(raw_x[~fit_mask]).astype(np.float64)
    fit_y, valid_y = y[fit_mask], y[~fit_mask]
    between, within, temporal, eligible = balanced_scatter(
        fit_x, fit_y, blocks[fit_mask])
    summaries, class_rows = [], []
    candidates = [("full", np.eye(fit_x.shape[1]))]
    for dimension in DIMENSIONS:
        for name, penalty in (("fisher", 0.0), ("temporal", TEMPORAL_PENALTY)):
            candidates.append((f"{name}_{dimension}", projection(
                between, within, temporal, penalty, dimension)))
    for method, projector in candidates:
        prediction = classify_projected(fit_x, fit_y, valid_x, projector)
        summary, rows = evaluate(valid_y, prediction)
        shift, shift_classes = heldout_shift_ratio(
            fit_x, fit_y, valid_x, valid_y, projector)
        summaries.append(dict(method=method, **summary,
                              heldout_shift_ratio=shift,
                              shift_classes=shift_classes,
                              fit_projection_classes=len(eligible)))
        class_rows.extend(dict(method=method, **row) for row in rows)
    counts = [dict(device_type=c, **{f"block_{i}": int(np.sum(
        (y == c) & (blocks == i))) for i in range(3)},
                   projection_eligible=bool(c in eligible)) for c in range(CLASSES)]
    return summaries, class_rows, counts


def run_home(archive, prepared, metadata, output, home, archive_hash):
    home_out = output / home
    if home_out.is_symlink():
        raise ValueError(f"Output is a symlink: {home_out}")
    source_path = prepared / home / "source.npz"
    prepared_config = json.loads((prepared / home / "config.json").read_text())
    prepared_marker = json.loads((prepared / home / "complete.json").read_text())
    source_hash = sha256(source_path)
    if prepared_marker["artifacts"]["source.npz"] != source_hash:
        raise ValueError(f"Prepared source checksum differs: {home}")
    if prepared_config["archive_sha256"] != archive_hash:
        raise ValueError(f"Prepared archive differs: {home}")
    signature = dict(home=home, archive_sha256=archive_hash,
                     prepared_source_sha256=source_hash,
                     implementation_sha256=sha256(Path(__file__)),
                     windows_sha256=sha256(metadata / "windows.csv"),
                     daily_rows_sha256=sha256(metadata / "daily_rows.csv"),
                     dimensions=DIMENSIONS, temporal_penalty=TEMPORAL_PENALTY,
                     min_block_support=MIN_BLOCK_SUPPORT, ridge=RIDGE)
    marker = home_out / "complete.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["signature"] != signature or any(
                sha256(home_out / name) != digest for name, digest in
                prior["artifacts"].items()):
            raise ValueError(f"Completed probe differs: {home}")
        print(home, "verified completed source-only probe", flush=True)
        return
    if home_out.exists() and any(home_out.iterdir()):
        raise ValueError(f"Incomplete output directory: {home_out}")
    with np.load(source_path, allow_pickle=False) as data:
        ids, raw_x, y = data["ids"], data["x"], data["y"]
    day_to_block, day_groups = source_blocks(home, metadata)
    blocks = retained_source_blocks(archive, prepared_config["member"],
                                    ids, day_to_block)
    summaries, class_rows, counts = probe_arrays(raw_x, y, blocks)
    home_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).assign(home=home).to_csv(home_out / "summary.csv", index=False)
    pd.DataFrame(class_rows).assign(home=home).to_csv(home_out / "per_class.csv", index=False)
    pd.DataFrame(counts).assign(home=home).to_csv(home_out / "block_counts.csv", index=False)
    (home_out / "blocks.json").write_text(json.dumps(day_groups, indent=2) + "\n")
    artifacts = {name: sha256(home_out / name) for name in
                 ("summary.csv", "per_class.csv", "block_counts.csv", "blocks.json")}
    marker.write_text(json.dumps(dict(signature=signature, artifacts=artifacts),
                                 indent=2) + "\n")
    print(home, "source probe complete; projection-eligible classes",
          sum(row["projection_eligible"] for row in counts), flush=True)


def report(output, homes):
    summaries = pd.concat([pd.read_csv(output / home / "summary.csv") for home in homes])
    pivot = summaries.pivot(index="home", columns="method", values="macro_f1_present")
    lines = ["UNSW source-window temporal-invariance feasibility probe.",
             "First two chronological source-day blocks fit projections and classifier;",
             "third source-day block evaluates them. Later labels/features/timestamps not decoded.",
             "Class-balanced Fisher scatter; temporal variant adds a within-class",
             "early-block mean-drift penalty. All classifiers and dimensions match.",
             "This is a source-only diagnostic, not a target adaptation result.",
             "Macro F1 by home:", pivot.round(4).to_string(),
             "Mean across homes:", summaries.groupby("method")[[
                 "macro_f1_present", "zero_recall_classes", "heldout_shift_ratio",
                 "fit_projection_classes"]].mean().round(4).to_string()]
    for dimension in DIMENSIONS:
        delta = 100 * (pivot[f"temporal_{dimension}"] - pivot[f"fisher_{dimension}"])
        lines += [f"Temporal minus Fisher, {dimension} dimensions, F1 percentage points:",
                  delta.round(3).to_string(),
                  f"Homes improved: {int((delta > 0).sum())}/{len(delta)}"]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print((output / "REPORT.txt").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--metadata", type=Path, default=HERE / "metadata")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_invariant_probe_v1"))
    parser.add_argument("--home", choices=PRIMARY)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    metadata = args.metadata.resolve(strict=True)
    prepared = args.prepared.resolve(strict=True)
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
        report(output, PRIMARY)
    print("SOURCE PROBE COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
