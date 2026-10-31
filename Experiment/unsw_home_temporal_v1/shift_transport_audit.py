"""Read-only audit of whether paid-query shift signals transport to development.

This is exploratory analysis of already viewed development outcomes, not a
deployable selector or independent test. It reads only completed CSV artifacts;
no archive, development features, or final-window data are opened.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import PRIMARY


CLASSES = 24
SEEDS = range(5)
MIN_MATCHED = 30
MIN_DEVELOPMENT = 100
MIN_ELIGIBLE_SEEDS = 3


def read_paid(root):
    frames = []
    for home in PRIMARY:
        folder = root / home
        marker = json.loads((folder / "complete.json").read_text())
        path = folder / "per_class.csv"
        if sha256(path) != marker["artifacts"]["per_class.csv"]:
            raise ValueError(f"Paid-shift artifact changed: {home}")
        frames.append(pd.read_csv(path))
    table = pd.concat(frames, ignore_index=True)
    expected = {(h, s, c) for h in PRIMARY for s in SEEDS
                for c in range(CLASSES)}
    actual = set(zip(table.home, table.seed, table.device_type))
    if len(table) != len(expected) or actual != expected:
        raise ValueError("Paid-shift table does not cover the fixed protocol")
    if not table.paid_support.groupby([table.home, table.seed]).sum().eq(1000).all():
        raise ValueError("Paid-label budget differs from 1000")
    return table


def read_development(root):
    classes = pd.read_csv(root / "per_class.csv")
    summary = pd.read_csv(root / "summary.csv")
    expected = {(h, -1, "static_mlp", c) for h in PRIMARY
                for c in range(CLASSES)} | {
                    (h, s, "prior_static", c) for h in PRIMARY
                    for s in SEEDS for c in range(CLASSES)}
    actual = set(zip(classes.home, classes.seed, classes.method,
                     classes.device_type))
    if len(classes) != len(expected) or actual != expected:
        raise ValueError("Prior-control development class table is incomplete")
    expected_summary = {(h, -1, "static_mlp") for h in PRIMARY} | {
        (h, s, "prior_static") for h in PRIMARY for s in SEEDS}
    actual_summary = set(zip(summary.home, summary.seed, summary.method))
    if len(summary) != len(expected_summary) or actual_summary != expected_summary:
        raise ValueError("Prior-control development summary is incomplete")
    return classes, summary


def analyze(paid, classes, summary):
    base = classes.loc[classes.method == "static_mlp", [
        "home", "device_type", "support", "recall", "f1"]].rename(columns={
            "support": "dev_support", "recall": "static_recall",
            "f1": "static_f1"})
    prior = classes.loc[classes.method == "prior_static", [
        "home", "seed", "device_type", "support", "recall", "f1"]].rename(
            columns={"support": "prior_support", "recall": "prior_recall",
                     "f1": "prior_f1"})
    paired = paid.merge(base, on=["home", "device_type"],
                        validate="many_to_one")
    paired = paired.merge(prior, on=["home", "seed", "device_type"],
                          validate="one_to_one")
    if not (paired.dev_support == paired.prior_support).all():
        raise ValueError("Static/prior development labels differ")
    paired = paired.drop(columns="prior_support")
    counts = summary.loc[summary.method == "static_mlp", [
        "home", "curated_rows"]]
    paired = paired.merge(counts, on="home", validate="many_to_one")
    if not (paired.groupby(["home", "seed"]).dev_support.sum().to_numpy()
            == paired.groupby(["home", "seed"]).curated_rows.first().to_numpy()
            ).all():
        raise ValueError("Development class supports do not sum to curated rows")
    paired["dev_share"] = paired.dev_support / paired.curated_rows
    paired["prior_delta_f1_pp"] = 100 * (paired.prior_f1 - paired.static_f1)
    paired["prior_lost_present_class"] = (
        (paired.dev_support > 0) & (paired.static_recall > 0)
        & (paired.prior_recall == 0))

    rows = []
    for (home, seed), group in paired.groupby(["home", "seed"], sort=True):
        group = group.sort_values("device_type")
        dev = group.dev_share.to_numpy(float)
        paid_share = group.paid_share.to_numpy(float)
        source_share = group.late_source_share.to_numpy(float)
        if not np.isclose(dev.sum(), 1) or not np.isclose(paid_share.sum(), 1):
            raise ValueError(f"Invalid development/query class shares: {home} {seed}")
        if np.any(source_share < 0) or source_share.sum() <= 0:
            raise ValueError(f"Invalid source class shares: {home} {seed}")
        source_share /= source_share.sum()
        rows.append(dict(home=home, seed=seed,
                         paid_to_dev_tv=.5 * np.abs(paid_share - dev).sum(),
                         source_to_dev_tv=.5 * np.abs(source_share - dev).sum(),
                         paid_minus_source_tv=.5 * (np.abs(paid_share - dev).sum()
                                                    - np.abs(source_share - dev).sum()),
                         lost_present_classes=int(group.prior_lost_present_class.sum())))
    transport = pd.DataFrame(rows)
    home_transport = transport.groupby("home", as_index=False)[[
        "paid_to_dev_tv", "source_to_dev_tv", "paid_minus_source_tv",
        "lost_present_classes"]].mean()

    # The five seeds are repeated measurements of each home/class. Aggregate
    # before examining associations, then report homes as the independent units.
    paired["ks_eligible"] = (paired.matched_rows >= MIN_MATCHED)
    eligible = paired.loc[paired.ks_eligible &
                          (paired.dev_support >= MIN_DEVELOPMENT)]
    geometry = eligible.groupby(["home", "device_type"], as_index=False).agg(
        eligible_seeds=("seed", "nunique"),
        mean_excess_ks=("excess_ks", "mean"))
    outcomes = paired.groupby(["home", "device_type"], as_index=False).agg(
        dev_support=("dev_support", "first"),
        mean_paid_support=("paid_support", "mean"),
        mean_prior_delta_f1_pp=("prior_delta_f1_pp", "mean"),
        lost_present_seed_count=("prior_lost_present_class", "sum"))
    geometry = geometry.merge(outcomes, on=["home", "device_type"],
                              validate="one_to_one")
    geometry = geometry.loc[geometry.eligible_seeds >= MIN_ELIGIBLE_SEEDS].copy()
    # A descriptive rank association is not an out-of-home prediction result.
    by_home = []
    for home, group in geometry.groupby("home"):
        rho = (group.mean_excess_ks.corr(group.mean_prior_delta_f1_pp,
                                         method="spearman")
               if len(group) >= 4 else np.nan)
        by_home.append(dict(home=home, eligible_classes=len(group),
                            ks_vs_prior_delta_spearman=rho))
    geometry_home = pd.DataFrame(by_home)
    home_transport = home_transport.merge(geometry_home, on="home", how="left",
                                           validate="one_to_one")
    home_transport["eligible_classes"] = home_transport.eligible_classes.fillna(0).astype(int)

    lost = paired.loc[paired.prior_lost_present_class, [
        "home", "seed", "device_type", "dev_support", "paid_support",
        "late_source_share", "paid_share", "matched_rows", "excess_ks",
        "static_recall", "prior_recall", "static_f1", "prior_f1"]]
    return paired, transport, home_transport, geometry, lost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paid", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_paid_shift_v1"))
    parser.add_argument("--prior-development", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_prior_control_v1/development"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_shift_transport_audit_v1"))
    args = parser.parse_args()
    paid_root = args.paid.resolve(strict=True)
    prior_root = args.prior_development.resolve(strict=True)
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error("Output must be a new, empty, non-symlink directory")
    paid = read_paid(paid_root)
    classes, summary = read_development(prior_root)
    paired, transport, homes, geometry, lost = analyze(paid, classes, summary)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("paired.csv", paired), ("by_seed.csv", transport),
                        ("by_home.csv", homes), ("geometry.csv", geometry),
                        ("lost_classes.csv", lost)):
        table.to_csv(output / name, index=False)
    report = ("Exploratory read-only transport audit; development outcomes were already seen.\n"
              "No model fitted, no new labels, no final-window data read.\n"
              "Source share is a reservoir-weighted estimate; paid share uses 1000 random labels.\n"
              "TV is total variation between 24-class proportions; positive paid-minus-source\n"
              "means the paid query estimate is farther from development than late source.\n"
              "KS association uses home/class means, >=3 eligible seeds, >=100 development\n"
              "support; five seeds are repeated measurements, not independent homes.\n"
              "Descriptive associations do not validate a deployable routing rule.\n"
              "By home:\n" + homes.round(4).to_string(index=False) + "\n"
              "Across-home mean TV: paid-to-dev="
              f"{homes.paid_to_dev_tv.mean():.4f}, source-to-dev="
              f"{homes.source_to_dev_tv.mean():.4f}; paid closer in "
              f"{int((homes.paid_minus_source_tv < 0).sum())}/{len(homes)} homes.\n"
              f"Geometry-eligible home/class pairs: {len(geometry)}; "
              f"homes with >=4 pairs: {int((homes.eligible_classes >= 4).sum())}.\n"
              f"Prior correction removed recall from {len(lost)} present-class/seed pairs;\n"
              "see lost_classes.csv for paid support and development support.\n")
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        implementation_sha256=sha256(Path(__file__)),
        paid_artifact_sha256={h: sha256(paid_root / h / "per_class.csv")
                              for h in PRIMARY},
        prior_class_sha256=sha256(prior_root / "per_class.csv"),
        prior_summary_sha256=sha256(prior_root / "summary.csv"),
        final_window_read=False), indent=2) + "\n")
    print(report, flush=True)
    print("SHIFT TRANSPORT AUDIT COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
