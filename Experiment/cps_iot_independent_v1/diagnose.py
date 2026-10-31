"""Read-only CARE/RF development diagnostic; no final-test access or routing.

Future development labels are used for diagnosis, so no oracle or routed score
from this file is a deployable method result.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_home(folder):
    if not (folder / "complete.json").is_file():
        raise ValueError(f"Incomplete development output: {folder}")
    classes = pd.read_csv(folder / "per_class.csv")
    queries = pd.read_csv(folder / "queries.csv")
    paid = (queries.groupby(["seed", "true_class"]).size()
            .rename("paid_support").reset_index()
            .rename(columns={"true_class": "device_name"}))
    static = (classes.loc[classes.method == "static_mlp",
                          ["device_name", "f1", "recall", "support"]]
              .rename(columns={"f1": "static_f1", "recall": "static_recall"}))
    paired = classes.loc[classes.method.isin(["care", "frozen_rf"])].pivot(
        index=["seed", "device_name"], columns="method",
        values=["f1", "recall"])
    paired.columns = [f"{method}_{metric}" for metric, method in paired.columns]
    paired = paired.reset_index().merge(static, on="device_name", validate="many_to_one")
    paired = paired.merge(paid, on=["seed", "device_name"], how="left",
                          validate="one_to_one")
    paired["paid_support"] = paired.paid_support.fillna(0).astype(int)
    paired["rf_minus_care_f1"] = paired.frozen_rf_f1 - paired.care_f1
    paired.insert(0, "home", folder.name)
    overlap = []
    for path in sorted(folder.glob("development_predictions_seed_*.npz")):
        seed = int(path.stem.rsplit("_", 1)[1])
        with np.load(path, allow_pickle=False) as data:
            y, static_pred, care, rf = [data[name] for name in
                                        ("y", "static_mlp", "care", "frozen_rf")]
        valid = y >= 0
        source_correct = valid & (static_pred == y)
        rf_correct, care_correct = valid & (rf == y), valid & (care == y)
        overlap.append(dict(
            home=folder.name, seed=seed, rows=len(y),
            disagreements=int((rf != care).sum()),
            rf_only_correct=int((rf_correct & ~care_correct).sum()),
            care_only_correct=int((care_correct & ~rf_correct).sum()),
            rf_source_flips=int((source_correct & ~rf_correct).sum()),
            care_source_flips=int((source_correct & ~care_correct).sum()),
        ))
    return paired, pd.DataFrame(overlap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.development_root.resolve(strict=True)
    output = args.output or root.parent / "cps_iot_diagnostic_v1"
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output directory must be new and empty: {output}")
    homes = sorted(p for p in root.glob("home[0-9][0-9]") if p.is_dir())
    if len(homes) != 12:
        parser.error(f"Expected all 12 homes; found {len(homes)}")
    parts = [inspect_home(folder) for folder in homes]
    class_rows = pd.concat([part[0] for part in parts], ignore_index=True)
    overlap = pd.concat([part[1] for part in parts], ignore_index=True)
    for name in ("disagreements", "rf_only_correct", "care_only_correct",
                 "rf_source_flips", "care_source_flips"):
        overlap[name + "_fraction"] = overlap[name] / overlap.rows
    by_home = overlap.groupby("home")[[
        "disagreements_fraction", "rf_only_correct_fraction",
        "care_only_correct_fraction", "rf_source_flips_fraction",
        "care_source_flips_fraction"]].mean().reset_index()
    class_home = class_rows.groupby(["home", "device_name"]).agg(
        dev_support=("support", "first"), paid_support_mean=("paid_support", "mean"),
        static_f1=("static_f1", "first"), care_f1=("care_f1", "mean"),
        rf_f1=("frozen_rf_f1", "mean"),
        rf_minus_care_f1=("rf_minus_care_f1", "mean")).reset_index()
    supported = class_home.loc[class_home.dev_support >= 20]
    output.mkdir(parents=True)
    class_rows.to_csv(output / "per_seed_class_deltas.csv", index=False)
    class_home.to_csv(output / "per_home_class_deltas.csv", index=False)
    overlap.to_csv(output / "per_seed_overlap.csv", index=False)
    by_home.to_csv(output / "per_home_overlap.csv", index=False)
    top = supported.nlargest(10, "rf_minus_care_f1")
    bottom = supported.nsmallest(10, "rf_minus_care_f1")
    columns = ["home", "device_name", "dev_support", "paid_support_mean",
               "static_f1", "care_f1", "rf_f1", "rf_minus_care_f1"]
    report = "\n".join([
        "CARE/RF disagreement diagnostic on the development sets only.",
        "Uses development labels for analysis; these are not deployable routing results.",
        "No test_full.csv file was opened; do not choose a class gate from this table alone.",
        "Mean per-home disagreement and correct-only rates (fractions of all dev rows):",
        by_home.to_string(index=False),
        "Overall home mean:", by_home.drop(columns="home").mean().to_string(),
        "Top RF-over-CARE class F1 differences with >=20 dev examples:",
        top[columns].to_string(index=False),
        "Top CARE-over-RF class F1 differences with >=20 dev examples:",
        bottom[columns].to_string(index=False),
        "Use per_home_class_deltas.csv and paid support before proposing a routing rule.",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
