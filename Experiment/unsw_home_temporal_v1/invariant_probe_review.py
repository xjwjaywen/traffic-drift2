"""Read-only class/home review of the source-only invariance probe.

This reads only the probe's source-window CSVs. It does not open the archive,
paid query labels, development results, or the final window.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from invariant_probe import PRIMARY


def review(root):
    pairs = []
    for home in PRIMARY:
        per_class = pd.read_csv(root / home / "per_class.csv")
        counts = pd.read_csv(root / home / "block_counts.csv")
        expected = {"fisher_16", "temporal_16"}
        if not expected.issubset(set(per_class.method)):
            raise ValueError(f"Missing 16-dimensional methods for {home}")
        fisher = per_class.loc[per_class.method == "fisher_16", [
            "device_type", "support", "f1", "recall"]].rename(
                columns={"f1": "fisher_f1", "recall": "fisher_recall"})
        temporal = per_class.loc[per_class.method == "temporal_16", [
            "device_type", "support", "f1", "recall"]].rename(
                columns={"f1": "temporal_f1", "recall": "temporal_recall"})
        pair = fisher.merge(temporal, on="device_type", suffixes=("_fisher", "_temporal"),
                            validate="one_to_one")
        if len(pair) != 24 or not (pair.support_fisher == pair.support_temporal).all():
            raise ValueError(f"Class coverage or support differs: {home}")
        pair = pair.drop(columns="support_temporal").rename(
            columns={"support_fisher": "support"})
        pair = pair.merge(counts.drop(columns="home"), on="device_type",
                          validate="one_to_one")
        pair["home"] = home
        pairs.append(pair)
    data = pd.concat(pairs, ignore_index=True)
    data = data.loc[data.support > 0].copy()
    data["delta_f1_pp"] = 100 * (data.temporal_f1 - data.fisher_f1)
    data["delta_recall_pp"] = 100 * (data.temporal_recall - data.fisher_recall)
    present_per_home = data.groupby("home").device_type.transform("count")
    data["macro_contribution_pp"] = data.delta_f1_pp / present_per_home
    if data.projection_eligible.dtype != bool:
        raise ValueError("Projection eligibility was not parsed as boolean")
    class_summary = data.groupby("device_type").agg(
        present_homes=("home", "count"),
        eligible_homes=("projection_eligible", "sum"),
        improved_homes=("delta_f1_pp", lambda x: int((x > 0).sum())),
        worsened_homes=("delta_f1_pp", lambda x: int((x < 0).sum())),
        median_support=("support", "median"),
        mean_delta_pp=("delta_f1_pp", "mean"),
        median_delta_pp=("delta_f1_pp", "median"),
    ).reset_index()
    named = pd.read_csv(Path(__file__).resolve().parent / "metadata" / "official_devices.csv")
    names = named.groupby("device_type").name.first().rename("device_name")
    class_summary = class_summary.join(names, on="device_type")
    data = data.join(names, on="device_type")
    by_home = data.groupby("home").agg(
        present_classes=("device_type", "count"),
        eligible_classes=("projection_eligible", "sum"),
        macro_delta_pp=("macro_contribution_pp", "sum"),
        positive_class_pairs=("delta_f1_pp", lambda x: int((x > 0).sum())),
        negative_class_pairs=("delta_f1_pp", lambda x: int((x < 0).sum())),
    ).reset_index()
    for row in by_home.itertuples(index=False):
        summary = pd.read_csv(root / row.home / "summary.csv").set_index("method")
        expected = 100 * (summary.loc["temporal_16", "macro_f1_present"] -
                          summary.loc["fisher_16", "macro_f1_present"])
        if not np.isclose(row.macro_delta_pp, expected, atol=1e-6):
            raise ValueError(f"Per-class changes do not reproduce macro F1: {row.home}")
    if not np.isfinite(data.delta_f1_pp).all():
        raise ValueError("Non-finite per-class F1 changes")
    data.to_csv(root / "class_home_review.csv", index=False)
    class_summary.to_csv(root / "class_review.csv", index=False)
    by_home.to_csv(root / "home_review.csv", index=False)
    strongest = data.reindex(data.macro_contribution_pp.abs().sort_values(
        ascending=False).index).head(12)
    consistent = class_summary.loc[(class_summary.present_homes >= 8) &
                                   (class_summary.improved_homes >= 8)]
    lines = ["Source-only invariance probe: paired 16-dimensional per-class review.",
             "Late source-block labels are diagnostic only; no target-period data used.",
             "The class/home pair is descriptive, not an independent replicate.",
             "Per-home F1 difference, temporal minus Fisher (percentage points):",
             by_home.round(3).to_string(index=False),
             "Projection eligibility and per-class direction:",
             class_summary.round(3).to_string(index=False),
             "Class types improved in at least 8 of at least 8 present homes:",
             consistent.round(3).to_string(index=False) if len(consistent) else "none",
             "Largest absolute contributions to home macro-F1 change:",
             strongest[["home", "device_type", "device_name", "support",
                        "projection_eligible", "block_0", "block_1", "block_2",
                        "delta_f1_pp", "delta_recall_pp", "macro_contribution_pp"]]
             .round(3).to_string(index=False),
             "Mean per-class delta by projection eligibility (descriptive):",
             data.groupby("projection_eligible")[["delta_f1_pp", "delta_recall_pp"]]
             .mean().round(3).to_string()]
    report = root / "REPORT_classes.txt"
    report.write_text("\n".join(lines) + "\n")
    print(report.read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_invariant_probe_v1"))
    args = parser.parse_args()
    review(args.root.resolve(strict=True))


if __name__ == "__main__":
    main()
