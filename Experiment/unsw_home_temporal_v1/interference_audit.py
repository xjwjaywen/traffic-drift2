"""Read-only, paired source-versus-CARE development interference audit.

The audit reopens development labels only. It does not train, route, tune,
or evaluate on the final time window. Paid support is counted only at the
1,000 query indices selected before development labels were opened.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import PRIMARY, metrics, read_development_labels, verify_predictions


def class_changes(y_all, static_all, care_all, paid_y, home, seed):
    valid = y_all >= 0
    y = y_all[valid].astype(np.int64)
    static = static_all[valid].astype(np.int64)
    care = care_all[valid].astype(np.int64)
    _, static_classes = metrics(y_all, static_all, static_all,
                                home, -1, "static_mlp")
    _, care_classes = metrics(y_all, static_all, care_all,
                              home, seed, "care")
    paid = np.bincount(paid_y, minlength=24)
    rows = []
    for c in range(24):
        truth = y == c
        before = static == c
        after = care == c
        recovered = int(np.sum(truth & ~before & after))
        spoiled = int(np.sum(truth & before & ~after))
        wrong_to_other_wrong = int(np.sum(truth & ~before & ~after &
                                          (static != care)))
        sr, cr = static_classes[c], care_classes[c]
        rows.append(dict(
            home=home, seed=seed, device_type=c,
            support=int(truth.sum()), paid_support=int(paid[c]),
            static_correct=int(np.sum(truth & before)),
            care_correct=int(np.sum(truth & after)),
            recovered=recovered, spoiled=spoiled,
            net_correct=recovered - spoiled,
            wrong_to_other_wrong=wrong_to_other_wrong,
            static_predicted_count=int(sr["predicted_count"]),
            care_predicted_count=int(cr["predicted_count"]),
            static_recall=float(sr["recall"]), care_recall=float(cr["recall"]),
            static_f1=float(sr["f1"]), care_f1=float(cr["f1"]),
            f1_delta_pp=100. * (float(cr["f1"]) - float(sr["f1"])),
        ))
    if sum(row["recovered"] for row in rows) != int(np.sum((static != y) & (care == y))):
        raise AssertionError("Recovered class counts do not sum to paired recovery")
    if sum(row["spoiled"] for row in rows) != int(np.sum((static == y) & (care != y))):
        raise AssertionError("Spoiled class counts do not sum to paired harm")
    return rows


def run(archive, predictions, prepared, manifest, output):
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    frozen_hashes = verify_predictions(predictions, windows)
    expected_archive = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected_archive:
        raise ValueError("Archive differs from reviewed timestamp audit")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Audit output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(manifest / "official_devices.csv")
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    rows = []
    for home in PRIMARY:
        mapping = official.loc[official.home == home].set_index("mac").device_type.to_dict()
        y = read_development_labels(archive, members.loc[home, "member"],
                                    home, windows, mapping)
        static = np.load(predictions / home / "static_mlp.npy", mmap_mode="r")
        paid_labels = np.load(prepared / home / "query_labels.npy", mmap_mode="r")
        for seed in range(5):
            seed_dir = predictions / home / f"seed_{seed}"
            chosen = np.load(seed_dir / "paid_query_pool_indices.npy")
            if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
                raise ValueError(f"Paid query selection invalid: {home} {seed}")
            paid = np.asarray(paid_labels[chosen], dtype=np.int64)
            if not np.all((0 <= paid) & (paid < 24)):
                raise ValueError(f"Paid labels out of range: {home} {seed}")
            care = np.load(seed_dir / "care.npy", mmap_mode="r")
            rows.extend(class_changes(y, static, care, paid, home, seed))
        print(home, "paired class audit complete", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(output / "by_seed_class.csv", index=False)
    names = official[["home", "device_type", "name"]]
    pooled = (data.groupby(["home", "device_type"], as_index=False)
              [["support", "paid_support", "static_correct", "care_correct",
                "recovered", "spoiled", "net_correct",
                "wrong_to_other_wrong", "static_predicted_count",
                "care_predicted_count", "static_recall", "care_recall",
                "static_f1", "care_f1", "f1_delta_pp"]].mean())
    pooled = pooled.merge(names, on=["home", "device_type"], validate="one_to_one")
    pooled.to_csv(output / "by_home_class.csv", index=False)
    homes = (data.groupby(["home", "seed"], as_index=False)
             [["static_correct", "care_correct", "recovered", "spoiled",
               "net_correct", "wrong_to_other_wrong"]].sum()
             .drop(columns="seed")
             .groupby("home", as_index=False).mean(numeric_only=True))
    homes.to_csv(output / "by_home.csv", index=False)
    lines = ["Read-only development paired-flip audit. These target labels are for",
             "diagnosis only; no method or route was chosen with them.",
             "Final-window labels and features remain unopened.",
             "Mean counts per seed, by home:", homes.to_string(index=False), ""]
    for home in PRIMARY:
        subset = pooled[(pooled.home == home) & (pooled.support > 0)]
        columns = ["device_type", "name", "support", "paid_support",
                   "recovered", "spoiled", "net_correct", "static_f1",
                   "care_f1", "f1_delta_pp"]
        lines.extend([home + " most harmed F1 classes:",
                      subset.nsmallest(3, "f1_delta_pp")[columns]
                      .round(3).to_string(index=False),
                      home + " most improved F1 classes:",
                      subset.nlargest(3, "f1_delta_pp")[columns]
                      .round(3).to_string(index=False), ""])
    report = "\n".join(lines) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        archive_sha256=expected_archive,
        frozen_prediction_markers_sha256=frozen_hashes,
        implementation_sha256=sha256(Path(__file__)),
        development_labels_read=True, final_labels_read=False), indent=2) + "\n")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--predictions", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "metadata")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_interference_v1"))
    args = parser.parse_args()
    run(args.archive.resolve(strict=True), args.predictions.resolve(strict=True),
        args.prepared.resolve(strict=True), args.manifest.resolve(strict=True),
        args.output.resolve())


if __name__ == "__main__":
    main()
