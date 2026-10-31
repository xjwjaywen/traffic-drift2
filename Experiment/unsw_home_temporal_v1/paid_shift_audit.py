"""Exploratory source-to-paid-query shift diagnosis on existing 1000 labels.

Each seed uses only its own 1000 uniformly paid query labels. Seeds are repeats,
not independent homes, and must never be pooled as a deployable 5000-label
method. No development or final-window label, feature, or timestamp is read.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from audit import sha256
from invariant_probe import PRIMARY


BUDGET = 1000
REPEATS = 3
MAX_MATCHED = 300
MIN_MATCHED = 30
WILSON_Z = 1.96


def wilson_interval(count, total):
    proportion = count / total
    z2 = WILSON_Z ** 2
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    half = WILSON_Z * np.sqrt(
        proportion * (1 - proportion) / total + z2 / (4 * total * total)
    ) / denominator
    return float(center - half), float(center + half)


def mean_feature_ks(left, right):
    return float(np.mean([ks_2samp(left[:, j], right[:, j],
                                   method="asymp").statistic
                          for j in range(left.shape[1])]))


def matched_effect(source, query, *, seed):
    size = min(len(query), len(source) // 2, MAX_MATCHED)
    if size < MIN_MATCHED:
        return np.nan, np.nan, size
    rng = np.random.default_rng(seed)
    cross, within = [], []
    for _ in range(REPEATS):
        source_ids = rng.choice(len(source), 2 * size, replace=False)
        query_ids = rng.choice(len(query), size, replace=False)
        first, second = source[source_ids[:size]], source[source_ids[size:]]
        target = query[query_ids]
        cross.append(mean_feature_ks(first, target))
        within.append(mean_feature_ks(first, second))
    return float(np.mean(cross)), float(np.mean(within)), size


def run_home(prepared, baseline, source_mechanism, output, home):
    folder = output / home
    if folder.is_symlink():
        raise ValueError(f"Output is a symlink: {folder}")
    prep_dir, base_dir = prepared / home, baseline / home
    prep_artifacts = json.loads((prep_dir / "complete.json").read_text())["artifacts"]
    file_names = ("source.npz", "query.npz", "query_labels.npy")
    for name in file_names:
        if sha256(prep_dir / name) != prep_artifacts[name]:
            raise ValueError(f"Prepared artifact differs: {home} {name}")
    chosen_hashes = []
    for seed in range(5):
        seed_dir = base_dir / f"seed_{seed}"
        name = "paid_query_pool_indices.npy"
        expected = json.loads((seed_dir / "complete.json").read_text())["artifacts"][name]
        actual = sha256(seed_dir / name)
        if actual != expected:
            raise ValueError(f"Paid query indices changed: {home} {seed}")
        chosen_hashes.append(actual)
    composition_file = source_mechanism / home / "composition.csv"
    composition_artifacts = json.loads(
        (source_mechanism / home / "complete.json").read_text())["artifacts"]
    if sha256(composition_file) != composition_artifacts["composition.csv"]:
        raise ValueError(f"Source composition audit changed: {home}")
    signature = dict(home=home, code_sha256=sha256(Path(__file__)),
                     prepared_sha256={name: prep_artifacts[name] for name in file_names},
                     chosen_hashes=chosen_hashes,
                     source_composition_sha256=composition_artifacts["composition.csv"],
                     budget=BUDGET, repeats=REPEATS,
                     max_matched=MAX_MATCHED, min_matched=MIN_MATCHED)
    marker = folder / "complete.json"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous["signature"] != signature or any(
                sha256(folder / name) != digest for name, digest in
                previous["artifacts"].items()):
            raise ValueError(f"Completed paid shift audit differs: {home}")
        print(home, "verified completed paid shift audit", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete paid shift audit output: {folder}")
    with np.load(prep_dir / "source.npz", allow_pickle=False) as data:
        raw_source, source_labels = data["x"], data["y"]
    with np.load(prep_dir / "query.npz", allow_pickle=False) as data:
        raw_query = data["x"]
    paid_labels = np.load(prep_dir / "query_labels.npy", mmap_mode="r")
    if len(paid_labels) != len(raw_query):
        raise ValueError(f"Query labels/features length differs: {home}")
    prep = Preprocessor().fit(raw_source)
    source_x = prep.transform(raw_source)
    composition = pd.read_csv(composition_file).set_index("device_type")
    results = []
    for seed in range(5):
        chosen = np.load(base_dir / f"seed_{seed}" /
                         "paid_query_pool_indices.npy", allow_pickle=False)
        if chosen.shape != (BUDGET,) or len(np.unique(chosen)) != BUDGET or (
                np.any(chosen < 0) or np.any(chosen >= len(raw_query))):
            raise ValueError(f"Invalid paid query selection: {home} {seed}")
        labels = np.asarray(paid_labels[chosen], dtype=np.int64)
        if np.any((labels < 0) | (labels >= 24)):
            raise ValueError(f"Invalid paid class label: {home} {seed}")
        query_x = prep.transform(raw_query[chosen])
        for c in range(24):
            support = int(np.sum(labels == c))
            low, high = wilson_interval(support, BUDGET)
            source_share = float(composition.loc[c, "estimated_share_2"])
            cross, within, matched = matched_effect(
                source_x[source_labels == c], query_x[labels == c],
                seed=100000 * int(home[-2:]) + 100 * seed + c)
            results.append(dict(home=home, seed=seed, device_type=c,
                                paid_support=support, paid_share=support / BUDGET,
                                paid_share_low=low, paid_share_high=high,
                                late_source_share=source_share,
                                source_share_outside_paid_interval=bool(
                                    source_share < low or source_share > high),
                                matched_rows=matched, cross_mean_ks=cross,
                                within_source_mean_ks=within,
                                excess_ks=cross - within))
        print(home, "seed", seed, "paid-label shift audit complete", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(folder / "per_class.csv", index=False)
    marker.write_text(json.dumps(dict(signature=signature,
                         artifacts={"per_class.csv": sha256(folder / "per_class.csv")}),
                         indent=2) + "\n")


def report(output):
    data = pd.concat([pd.read_csv(output / home / "per_class.csv")
                      for home in PRIMARY], ignore_index=True)
    eligible = data.loc[data.matched_rows >= MIN_MATCHED]
    per_home = data.groupby("home").agg(
        paid_classes_per_seed=("paid_support", lambda x: float((x > 0).sum()) / 5),
        conditional_classes_per_seed=("matched_rows", lambda x: float(
            (x >= MIN_MATCHED).sum()) / 5),
        source_share_outside_wilson_per_seed=(
            "source_share_outside_paid_interval", lambda x: float(x.sum()) / 5),
    )
    conditional_home = eligible.groupby("home").excess_ks.mean()
    per_home["mean_excess_ks_eligible"] = conditional_home
    class_means = eligible.groupby("device_type").agg(
        eligible_home_seed_pairs=("home", "count"),
        eligible_homes=("home", "nunique"),
        mean_paid_support=("paid_support", "mean"),
        mean_excess_ks=("excess_ks", "mean"),
    ).sort_values("mean_excess_ks", ascending=False)
    lines = ["Exploratory source-to-paid-query shift audit, same 1000 random labels per seed.",
             "Five seeds repeat each home; their labels are never pooled into a method.",
             "No development/final-window label, feature, or timestamp was read.",
             "Late-source class share is reservoir-weighted; paid query share has a",
             "95% Wilson interval, which does not include source-estimation error.",
             "Conditional comparison uses matched sample sizes; source-to-query",
             "mean KS minus source-to-source mean KS is an effect-size diagnostic,",
             "not a formal test or evidence of future conditional invariance.",
             "Per-home coverage and class-share changes:",
             per_home.round(4).to_string(),
             f"Class/home/seed pairs with >= {MIN_MATCHED} matched positives: "
             f"{len(eligible)} / {len(data)}",
             "Eligible excess KS distribution:",
             eligible.excess_ks.describe().round(4).to_string(),
             "Class summary among eligible pairs:",
             class_means.round(4).to_string()]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print((output / "REPORT.txt").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--source-mechanism", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_source_mechanism_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_paid_shift_v1"))
    parser.add_argument("--home", choices=PRIMARY)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink():
        parser.error("Output must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    homes = [args.home] if args.home else PRIMARY
    for home in homes:
        run_home(args.prepared.resolve(strict=True),
                 args.baseline.resolve(strict=True),
                 args.source_mechanism.resolve(strict=True), output, home)
    if not args.home:
        report(output)
    print("PAID SHIFT AUDIT COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
