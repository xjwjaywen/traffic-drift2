"""Score all frozen 11-home final predictions after global preflight.

Only this stage maps final-window MACs to device types. Evaluation definitions
are fixed in FINAL_HOLDOUT_PROTOCOL.md and do not select a challenger.
"""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import sha256
from predict import PRIMARY, SEEDS


HERE = Path(__file__).resolve().parent
METHODS = ("static_mlp", "target", "care", "bcwi_050")
N_CLASSES = 24


def preflight(predictions, development, original, windows, expected_archive):
    """Verify every prediction and reference before reading one final label."""
    dev_config = json.loads((development / "config.json").read_text())
    if (dev_config.get("stage") != "development_only" or
            dev_config.get("archive_sha256") != expected_archive):
        raise ValueError("Development evaluation provenance differs")
    dev_classes = pd.read_csv(development / "per_class.csv")
    baseline = dev_classes.loc[(dev_classes.method == "static_mlp") &
                               (dev_classes.seed == -1)]
    if (len(baseline) != len(PRIMARY) * N_CLASSES or
            baseline.groupby("home").size().to_dict() !=
            {home: N_CLASSES for home in PRIMARY}):
        raise ValueError("Development static class records incomplete")
    refs = {}
    hashes = {}
    for home in PRIMARY:
        folder = predictions / home
        if folder.is_symlink():
            raise ValueError(f"Prediction folder is a symlink: {folder}")
        marker = folder / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Final predictions incomplete: {home}")
        config = json.loads((folder / "config.json").read_text())
        old_marker = original / home / "complete.json"
        if (config.get("home") != home or
                config.get("archive_sha256") != expected_archive or
                config.get("original_marker_sha256") != sha256(old_marker) or
                config.get("original_marker_sha256") !=
                    dev_config["prediction_markers_sha256"][home] or
                config.get("protocol_sha256") !=
                    sha256(HERE / "FINAL_HOLDOUT_PROTOCOL.md") or
                config.get("implementation_sha256") != sha256(HERE / "final_predict.py") or
                config.get("methods") != list(METHODS) or
                config.get("label_budget") != 1000 or
                config.get("repair_seeds") != list(SEEDS) or
                config.get("old_weight") != 0.5 or
                config.get("final_labels_read") is not False):
            raise ValueError(f"Frozen final protocol differs: {home}")
        artifacts = json.loads(marker.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Final prediction artifact changed: {home}")
        n = int(windows.loc[(home, "final"), "rows"])
        required = [folder / "static_mlp.npy"]
        for seed in SEEDS:
            required.extend(folder / f"seed_{seed}" / f"{name}.npy"
                            for name in METHODS if name != "static_mlp")
        for path in required:
            pred = np.load(path, mmap_mode="r", allow_pickle=False)
            if pred.shape != (n,) or pred.dtype != np.uint8 or pred.max() >= N_CLASSES:
                raise ValueError(f"Invalid final prediction shape/class: {path}")
        ref = baseline.loc[baseline.home == home].sort_values("device_type")
        if ref.device_type.tolist() != list(range(N_CLASSES)):
            raise ValueError(f"Development device inventory differs: {home}")
        refs[home] = ref
        hashes[home] = sha256(marker)
    return refs, hashes


def read_final_labels(archive, member, home, windows, mapping):
    n = int(windows.loc[(home, "final"), "rows"])
    after_dev = str(windows.loc[(home, "development"), "last_day"])
    last_final = str(windows.loc[(home, "final"), "last_day"])
    labels = np.empty(n, dtype=np.int8)
    pos = 0
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {home}")
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= after_dev:
                continue
            if day > last_final or pos >= n:
                raise ValueError(f"Final timestamp/count differs: {home}")
            mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
            labels[pos] = mapping.get(mac, -1)
            pos += 1
    if pos != n:
        raise ValueError(f"Final row count differs: {home}, {pos} != {n}")
    return labels


def class_stats(y, pred):
    cm = np.bincount(y * N_CLASSES + pred, minlength=N_CLASSES**2).reshape(
        N_CLASSES, N_CLASSES)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(N_CLASSES, float), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(N_CLASSES, float),
                   where=support + predicted > 0)
    return support, predicted, recall, f1


def mean_or_none(values, mask):
    return float(np.mean(values[mask])) if mask.any() else None


def score(y_all, source_all, pred_all, dev_ref, home, seed, method, paid_counts):
    curated = y_all >= 0
    y = y_all[curated].astype(np.int64)
    before = source_all[curated].astype(np.int64)
    pred = pred_all[curated].astype(np.int64)
    if not len(y):
        raise ValueError(f"No curated final rows: {home}")
    support, predicted, recall, f1 = class_stats(y, pred)
    _, _, before_recall, before_f1 = class_stats(y, before)
    dev_support = dev_ref.support.to_numpy(np.int64)
    dev_recall = dev_ref.recall.to_numpy(float)
    low = (support >= 50) & (before_recall < 0.1)
    learned = (dev_support >= 50) & (dev_recall >= 0.8)
    incident = low & learned
    stable = learned & (support >= 50) & (before_recall >= 0.8)
    source_correct = before == y
    summary = dict(
        home=home, seed=seed, method=method, all_final_rows=len(y_all),
        curated_rows=len(y), present_classes=int((support > 0).sum()),
        macro_f1_present=mean_or_none(f1, support > 0),
        low_recall_classes=int(low.sum()), low_recall_f1=mean_or_none(f1, low),
        low_recall_recovered_01=int((low & (recall >= 0.1)).sum()),
        low_recall_recovered_05=int((low & (recall >= 0.5)).sum()),
        incident_collapse_classes=int(incident.sum()),
        incident_collapse_f1=mean_or_none(f1, incident),
        incident_recovered_01=int((incident & (recall >= 0.1)).sum()),
        stable_classes=int(stable.sum()),
        stable_f1=mean_or_none(f1, stable),
        stable_f1_delta=mean_or_none(f1 - before_f1, stable),
        stable_losses_over_5pp=int((stable & ((f1 - before_f1) < -0.05)).sum()),
        negative_flip_rate_vs_static=float((pred[source_correct] != y[source_correct]).mean())
            if source_correct.any() else None,
        source_correct_rows=int(source_correct.sum()))
    rows = [dict(home=home, seed=seed, method=method, device_type=c,
                 support=int(support[c]), predicted_count=int(predicted[c]),
                 recall=float(recall[c]), f1=float(f1[c]),
                 static_recall=float(before_recall[c]),
                 static_f1=float(before_f1[c]),
                 development_static_recall=float(dev_recall[c]),
                 development_support=int(dev_support[c]),
                 paid_positives=(int(paid_counts[c]) if paid_counts is not None else None),
                 final_low_recall=bool(low[c]), incident_collapse=bool(incident[c]),
                 stable_recognition=bool(stable[c]))
            for c in range(N_CLASSES)]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--predictions", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/predictions"))
    parser.add_argument("--development", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_development_v1"))
    parser.add_argument("--original", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/evaluation"))
    args = parser.parse_args()
    manifest = HERE / "metadata"
    archive = args.archive.resolve(strict=True)
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from fixed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    original = args.original.resolve(strict=True)
    predictions = args.predictions.resolve(strict=True)
    development = args.development.resolve(strict=True)
    refs, hashes = preflight(predictions, development, original, windows, expected)
    devices = pd.read_csv(manifest / "official_devices.csv")
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    prepared = args.prepared.resolve(strict=True)
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Evaluation output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(dict(
        archive_sha256=expected, prediction_markers_sha256=hashes,
        evaluation_sha256=sha256(Path(__file__)),
        protocol_sha256=sha256(HERE / "FINAL_HOLDOUT_PROTOCOL.md"),
        stage="final_holdout", comparison_frozen=True), indent=2) + "\n")

    summary_rows, class_rows, diagnostics = [], [], []
    for home in PRIMARY:
        mapping = devices.loc[devices.home == home].set_index("mac").device_type.to_dict()
        y_all = read_final_labels(archive, members.loc[home, "member"],
                                  home, windows, mapping)
        folder = predictions / home
        static = np.load(folder / "static_mlp.npy", mmap_mode="r")
        dev_ref = refs[home]
        row, rows = score(y_all, static, static, dev_ref, home, -1,
                          "static_mlp", None)
        summary_rows.append(row)
        class_rows.extend(rows)
        final_support = np.bincount(y_all[y_all >= 0], minlength=N_CLASSES)
        dev_support = dev_ref.support.to_numpy(np.int64)
        dev_share = dev_support / dev_support.sum()
        final_share = final_support / final_support.sum()
        tv = 0.5 * np.abs(dev_share - final_share).sum()
        diagnostics.append(dict(home=home, seed=-1,
                                development_to_final_class_share_tv=float(tv),
                                final_max_class_share=float(final_share.max()),
                                final_present_classes=int((final_support > 0).sum()),
                                low_recall_types_with_lt5_paid=None,
                                mean_paid_positives_in_low_recall=None))
        for seed in SEEDS:
            chosen = np.load(original / home / f"seed_{seed}" /
                             "paid_query_pool_indices.npy")
            paid_y = np.load(prepared / home / "query_labels.npy", mmap_mode="r")[chosen]
            paid_counts = np.bincount(paid_y, minlength=N_CLASSES)
            low = np.array([r["final_low_recall"] for r in rows])
            diagnostics.append(dict(home=home, seed=seed,
                                    development_to_final_class_share_tv=float(tv),
                                    final_max_class_share=float(final_share.max()),
                                    final_present_classes=int((final_support > 0).sum()),
                                    low_recall_types_with_lt5_paid=int((low & (paid_counts < 5)).sum()),
                                    mean_paid_positives_in_low_recall=(float(paid_counts[low].mean())
                                                                        if low.any() else None)))
            for method in METHODS[1:]:
                pred = np.load(folder / f"seed_{seed}" / f"{method}.npy", mmap_mode="r")
                row, records = score(y_all, static, pred, dev_ref, home, seed,
                                      method, paid_counts)
                summary_rows.append(row)
                class_rows.extend(records)
        print(home, "final evaluated; curated", int((y_all >= 0).sum()), flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class.csv", index=False)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(output / "diagnostics.csv", index=False)
    home_means = summary.groupby(["home", "method"], as_index=False).mean(numeric_only=True)
    source_f1 = home_means.loc[home_means.method == "static_mlp"].set_index("home").macro_f1_present
    home_means["delta_vs_static_pp"] = [100 * (row.macro_f1_present - source_f1[row.home])
                                         for row in home_means.itertuples()]
    home_means.to_csv(output / "home_means.csv", index=False)
    seed_spread = summary.loc[summary.method != "static_mlp"].groupby(
        ["home", "method"], as_index=False).agg(
            seed_sd_macro_f1=("macro_f1_present", "std"))
    seed_spread.to_csv(output / "seed_spread.csv", index=False)
    means = home_means.groupby("method")[["macro_f1_present", "delta_vs_static_pp",
        "negative_flip_rate_vs_static", "low_recall_f1", "stable_f1_delta"]].mean()
    care_home = home_means.loc[home_means.method == "care"]
    paired = care_home.sort_values("home").delta_vs_static_pp.to_numpy(float)
    draws = np.random.default_rng(0).integers(len(paired), size=(10000, len(paired)))
    ci_low, ci_high = np.quantile(paired[draws].mean(axis=1), [.025, .975])
    assoc_home = diag.groupby("home")[["development_to_final_class_share_tv",
        "final_max_class_share", "mean_paid_positives_in_low_recall"]].mean()
    assoc_home["care_delta_pp"] = care_home.set_index("home").delta_vs_static_pp
    association = []
    for name in ("development_to_final_class_share_tv", "final_max_class_share",
                 "mean_paid_positives_in_low_recall"):
        eligible = assoc_home[[name, "care_delta_pp"]].dropna()
        rho = (eligible[name].corr(eligible.care_delta_pp, method="spearman")
               if len(eligible) >= 5 and eligible[name].nunique() > 1 else None)
        association.append(dict(indicator=name, eligible_homes=len(eligible),
                                spearman_rho=rho))
    pd.DataFrame(association).to_csv(output / "associations.csv", index=False)
    report = ("Frozen 11-home IoT final-window evaluation; five repeated paid draws/home.\n"
              "Source, target-only, CARE and preselected 0.5 source/target head interpolation.\n"
              "Mixed-protocol curated closed-set device-type task; no gateway deployment.\n"
              "All predictions passed preflight before final labels were decoded.\n"
              "The interpolation was chosen after development inspection; final data did not choose a method.\n"
              "Homes, not overlapping seeds, are independent units.\n"
              "Equal-home means:\n" + means.to_string(float_format=lambda v: f"{v:.5f}") + "\n"
              f"CARE improves {int((care_home.delta_vs_static_pp > 0).sum())}/11 homes.\n"
              f"CARE-minus-static equal-home mean = {paired.mean():.3f} pp; "
              f"descriptive 95% home bootstrap [{ci_low:.3f}, {ci_high:.3f}] pp.\n"
              "Per-home F1 difference (percentage points):\n" +
              home_means.pivot(index="home", columns="method", values="delta_vs_static_pp")
                  .to_string(float_format=lambda v: f"{v:.3f}") + "\n"
              "Group definitions and all per-class outcomes are in the frozen protocol and CSVs.\n")
    report += ("Secondary descriptive associations (not causal):\n" +
               pd.DataFrame(association).to_string(index=False) + "\n")
    (output / "REPORT.txt").write_text(report)
    (output / "complete.json").write_text(json.dumps(dict(
        artifacts={p.name: sha256(p) for p in output.iterdir()
                   if p.is_file() and p.name != "complete.json"}), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
