"""Read-only review of the completed 11-home IoT development experiment.

This does not open the traffic archive, train a model, or inspect the final
window. It summarizes the already evaluated static-versus-CARE experiment for
deciding what, if anything, can be reported in a paper.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HOMES = [f"home{i:02d}" for i in range(1, 13) if i != 10]
SEEDS = list(range(5))
CLASSES = list(range(24))
BASE = Path("/data/xjw/tma-care-data")


def require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks columns: {sorted(missing)}")


def require_keys(frame, keys, expected, name):
    actual = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
    if len(actual) != len(frame) or actual != expected:
        raise ValueError(f"{name} has missing, duplicate, or unexpected keys: "
                         f"expected {len(expected)}, found {len(actual)} rows "
                         f"and {len(actual)} distinct keys")


def validate(summary, per_class):
    require_columns(summary, ["home", "seed", "method", "macro_f1_present",
                              "negative_flip_rate_vs_static_mlp",
                              "zero_recall_classes", "present_classes"], "summary")
    require_columns(per_class, ["home", "seed", "method", "device_type",
                                "support", "recall", "f1"], "per_class")
    summary = summary[summary.method.isin(["static_mlp", "care"])].copy()
    per_class = per_class[per_class.method.isin(["static_mlp", "care"])].copy()
    summary["seed"] = summary.seed.astype(int)
    per_class["seed"] = per_class.seed.astype(int)
    per_class["device_type"] = per_class.device_type.astype(int)
    base_keys = {(h, -1, "static_mlp") for h in HOMES}
    care_keys = {(h, s, "care") for h in HOMES for s in SEEDS}
    require_keys(summary, ["home", "seed", "method"], base_keys | care_keys,
                 "summary")
    class_keys = {(h, s, m, c)
                  for h, s, m in base_keys | care_keys for c in CLASSES}
    require_keys(per_class, ["home", "seed", "method", "device_type"],
                 class_keys, "per_class")
    if per_class[["support", "recall", "f1"]].isna().any().any():
        raise ValueError("per_class contains missing metrics")
    if (per_class.support < 0).any():
        raise ValueError("Negative class support")
    source = per_class.loc[per_class.method == "static_mlp",
                           ["home", "device_type", "support"]]
    care = per_class.loc[per_class.method == "care",
                         ["home", "device_type", "support"]]
    if not care.merge(source, on=["home", "device_type"],
                      suffixes=("_care", "_source"), validate="many_to_one") \
            .eval("support_care == support_source").all():
        raise ValueError("Development class support differs by method")
    for method in ("static_mlp", "care"):
        frame = per_class[per_class.method == method]
        present = frame[frame.support > 0].groupby(["home", "seed"]).f1.mean()
        rows = summary[summary.method == method].set_index(["home", "seed"])
        if not np.allclose(present.loc[rows.index], rows.macro_f1_present,
                           atol=1e-10):
            raise ValueError(f"Per-class F1 does not reproduce {method} macro F1")
    return summary, per_class


def home_review(summary, per_class):
    static = (summary[summary.method == "static_mlp"]
              .set_index("home")["macro_f1_present"])
    care_rows = summary[summary.method == "care"]
    care = care_rows.groupby("home")
    result = pd.DataFrame({
        "home": HOMES,
        "static_macro_f1": static.reindex(HOMES).to_numpy(),
        "care_macro_f1_mean": care.macro_f1_present.mean().reindex(HOMES).to_numpy(),
        "care_macro_f1_seed_sd": care.macro_f1_present.std().reindex(HOMES).to_numpy(),
        "care_negative_flip_mean":
            care.negative_flip_rate_vs_static_mlp.mean().reindex(HOMES).to_numpy(),
        "care_zero_recall_mean":
            care.zero_recall_classes.mean().reindex(HOMES).to_numpy(),
    })
    result["delta_pp"] = 100 * (result.care_macro_f1_mean - result.static_macro_f1)
    source_class = per_class[per_class.method == "static_mlp"]
    counts = source_class.groupby("home").agg(
        present_classes=("support", lambda s: int((s > 0).sum())),
        present_lt_100=("support", lambda s: int(((s > 0) & (s < 100)).sum())))
    return result.merge(counts, left_on="home", right_index=True,
                        validate="one_to_one")


def class_review(per_class, names, interference):
    source = (per_class[per_class.method == "static_mlp"]
              .drop(columns=["seed", "method"])
              .rename(columns={"recall": "static_recall", "f1": "static_f1"}))
    care = per_class[per_class.method == "care"].copy()
    per_seed = care.merge(source, on=["home", "device_type"],
                          suffixes=("", "_source"), validate="many_to_one")
    per_seed["delta_pp"] = 100 * (per_seed.f1 - per_seed.static_f1)
    means = per_seed.groupby(["home", "device_type"], as_index=False).agg(
        care_recall=("recall", "mean"), care_f1=("f1", "mean"),
        delta_pp=("delta_pp", "mean"),
        seeds_improved=("delta_pp", lambda x: int((x > 0).sum())),
        seeds_harmed=("delta_pp", lambda x: int((x < 0).sum())))
    result = source.merge(means, on=["home", "device_type"],
                          validate="one_to_one")
    result = result.merge(names[["device_type", "name"]].drop_duplicates(),
                          on="device_type", validate="many_to_one")
    if interference is not None:
        require_columns(interference, ["home", "device_type", "paid_support",
                                       "recovered", "spoiled"], "interference")
        result = result.merge(
            interference[["home", "device_type", "paid_support", "recovered",
                          "spoiled"]], on=["home", "device_type"],
            validate="one_to_one")
    return result[result.support > 0].sort_values(["home", "device_type"])


def format_table(frame, columns, digits=3):
    return frame[columns].round(digits).to_string(index=False)


def make_report(homes, classes, interference_used):
    delta = homes.delta_pp
    low = classes[(classes.support >= 50) & (classes.static_recall < 0.1)]
    high = classes[(classes.support >= 50) & (classes.static_recall >= 0.8)]
    count_positive = int((delta > 0).sum())
    lines = [
        "Read-only, post-hoc review of the completed 11-home IoT development comparison.",
        "IoT labels are curated device types, not encrypted application classes.",
        "Each home is split chronologically within that home; no cross-home transfer is tested.",
        "CARE uses a tabular source MLP and uniform 1000-label queries here, unlike the paper's CNN/BADGE primary result.",
        "Five paid-label seeds overlap within each home; homes are the independent units.",
        "All development outcomes have been seen. No final-window label or feature is read by this script.",
        "Baseline low recall on development alone does not establish temporal absorber collapse.",
        "The 50-support screen below matches the manuscript detector support floor; it is descriptive.",
        f"Paired interference audit included: {interference_used}.", "",
        "Across-home macro-F1 summary (equal weight per home):",
        f"  static mean: {homes.static_macro_f1.mean():.6f}",
        f"  CARE mean: {homes.care_macro_f1_mean.mean():.6f}",
        f"  CARE minus static: {delta.mean():+.3f} percentage points",
        f"  home-delta median: {delta.median():+.3f} percentage points",
        f"  home-delta sample SD: {delta.std(ddof=1):.3f} percentage points",
        f"  home-delta range: {delta.min():+.3f} to {delta.max():+.3f} percentage points",
        f"  homes improved: {count_positive}/{len(homes)}", "",
        "Per-home results:",
        format_table(homes, ["home", "present_classes", "present_lt_100",
                             "static_macro_f1", "care_macro_f1_mean",
                             "delta_pp", "care_negative_flip_mean"]), "",
        "Per-class F1 changes across present home/device-type pairs:",
        f"  improved: {int((classes.delta_pp > 0).sum())}",
        f"  harmed: {int((classes.delta_pp < 0).sum())}",
        f"  unchanged: {int((classes.delta_pp == 0).sum())}",
        f"  improved in all 5 seeds: {int((classes.seeds_improved == 5).sum())}",
        f"  harmed in all 5 seeds: {int((classes.seeds_harmed == 5).sum())}", "",
        "Low-baseline-recall candidates (support >= 50, static recall < 0.1):",
        f"  home/device-type pairs: {len(low)}",
        f"  CARE mean recall reaches >= 0.1: {int((low.care_recall >= 0.1).sum())}",
        f"  CARE mean F1 improves: {int((low.delta_pp > 0).sum())}",
        f"  CARE mean F1 worsens: {int((low.delta_pp < 0).sum())}", "",
        "High-baseline-recall candidates (support >= 50, static recall >= 0.8):",
        f"  home/device-type pairs: {len(high)}",
        f"  mean F1 loss > 5 pp: {int((high.delta_pp < -5).sum())}",
        f"  mean F1 gain > 5 pp: {int((high.delta_pp > 5).sum())}", "",
        "Largest class-level mean gains (descriptive, selected after seeing outcomes):",
        format_table(classes.nlargest(10, "delta_pp"),
                     ["home", "device_type", "name", "support", "static_recall",
                      "care_recall", "static_f1", "care_f1", "delta_pp",
                      "seeds_improved"]), "",
        "Largest class-level mean losses (descriptive, selected after seeing outcomes):",
        format_table(classes.nsmallest(10, "delta_pp"),
                     ["home", "device_type", "name", "support", "static_recall",
                      "care_recall", "static_f1", "care_f1", "delta_pp",
                      "seeds_harmed"]), "",
        "These are exploratory descriptive results, not an IoT efficacy claim or a new method result.",
    ]
    if interference_used:
        lines.extend(["", "Paid-label coverage among low-recall candidates:",
                      f"  no paid positives on average: {int((low.paid_support == 0).sum())}/{len(low)}",
                      f"  fewer than 5 paid positives on average: {int((low.paid_support < 5).sum())}/{len(low)}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path,
                        default=BASE / "unsw_home_temporal_development_v1")
    parser.add_argument("--interference", type=Path,
                        default=BASE / "unsw_home_temporal_interference_v1")
    parser.add_argument("--output", type=Path,
                        default=BASE / "unsw_iotj_scope_review_v1")
    args = parser.parse_args()
    summary = pd.read_csv(args.development / "summary.csv")
    per_class = pd.read_csv(args.development / "per_class.csv")
    summary, per_class = validate(summary, per_class)
    names = pd.read_csv(Path(__file__).parent / "metadata" / "official_devices.csv")
    interference_path = args.interference / "by_home_class.csv"
    interference = (pd.read_csv(interference_path)
                    if interference_path.is_file() else None)
    homes = home_review(summary, per_class)
    classes = class_review(per_class, names, interference)
    report = make_report(homes, classes, interference is not None)
    args.output.mkdir(parents=True, exist_ok=True)
    homes.to_csv(args.output / "by_home.csv", index=False)
    classes.to_csv(args.output / "by_home_class.csv", index=False)
    (args.output / "REPORT.txt").write_text(report)
    print(report, flush=True)
    print(f"SCOPE REVIEW COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
