"""Evaluate frozen UNSW development predictions after every primary home is ready.

No final-window label or feature is decoded. Development labels are read only
after predictions for all 11 primary homes and all five seeds pass preflight.
"""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import sha256


PRIMARY = [f"home{i:02d}" for i in range(1, 13) if i != 10]
METHODS = ("target", "care", "frozen_rf", "nearest_anchor")


def verify_predictions(pred_root, windows):
    """Perform the full preflight before opening any development labels."""
    hashes = {}
    for home in PRIMARY:
        folder = pred_root / home
        marker = folder / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Predictions incomplete: {home}")
        artifacts = json.loads(marker.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Prediction artifacts changed: {home}")
        config = json.loads((folder / "config.json").read_text())
        if config["home"] != home or config["repair_seeds"] != list(range(5)):
            raise ValueError(f"Prediction protocol changed: {home}")
        n = int(windows.loc[(home, "development"), "rows"])
        required = [folder / "static_mlp.npy"]
        for seed in range(5):
            seed_dir = folder / f"seed_{seed}"
            sub_marker = seed_dir / "complete.json"
            if not sub_marker.is_file():
                raise ValueError(f"Seed predictions incomplete: {home} {seed}")
            sub_artifacts = json.loads(sub_marker.read_text())["artifacts"]
            if any(sha256(seed_dir / name) != digest
                   for name, digest in sub_artifacts.items()):
                raise ValueError(f"Seed prediction changed: {home} {seed}")
            required.extend(seed_dir / f"{name}.npy" for name in METHODS)
        for path in required:
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.shape != (n,) or data.dtype != np.uint8:
                raise ValueError(f"Prediction shape or dtype invalid: {path}")
            if data.min() < 0 or data.max() >= 24:
                raise ValueError(f"Prediction class out of range: {path}")
        hashes[home] = sha256(marker)
    return hashes


def read_development_labels(archive, member, home, windows, mapping):
    n = int(windows.loc[(home, "development"), "rows"])
    after_query = str(windows.loc[(home, "query"), "last_day"])
    after_dev = str(windows.loc[(home, "development"), "last_day"])
    y = np.empty(n, dtype=np.int8)
    pos = 0
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {home}")
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= after_query:
                continue
            if day > after_dev:
                continue  # No final-window label or feature is decoded.
            if pos >= n:
                raise ValueError(f"Too many development rows: {home}")
            mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
            y[pos] = mapping.get(mac, -1)
            pos += 1
    if pos != n:
        raise ValueError(f"Development row count differs from timestamp audit: {home}")
    return y


def metrics(y_all, static_all, pred_all, home, seed, method):
    selected = y_all >= 0
    y = y_all[selected].astype(np.int64)
    before = static_all[selected].astype(np.int64)
    pred = pred_all[selected].astype(np.int64)
    if not len(y):
        raise ValueError(f"No curated development flows: {home}")
    matrix = np.bincount(y * 24 + pred, minlength=24 * 24).reshape(24, 24)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    recall = np.divide(tp, support, out=np.zeros(24, float), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(24, float),
                   where=support + predicted > 0)
    present = support > 0
    source_correct = before == y
    summary = dict(home=home, seed=seed, method=method,
                   all_development_rows=len(y_all), curated_rows=len(y),
                   present_classes=int(present.sum()),
                   macro_f1_present=float(f1[present].mean()),
                   zero_recall_classes=int(((recall == 0) & present).sum()),
                   negative_flip_rate_vs_static_mlp=float(
                       (pred[source_correct] != y[source_correct]).mean())
                   if source_correct.any() else None,
                   source_correct_rows=int(source_correct.sum()))
    classes = [dict(home=home, seed=seed, method=method, device_type=c,
                    support=int(support[c]), predicted_count=int(predicted[c]),
                    recall=float(recall[c]), f1=float(f1[c]))
               for c in range(24)]
    return summary, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--predictions", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "metadata")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_development_v1"))
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from the fixed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    pred_root = args.predictions.resolve(strict=True)
    frozen = verify_predictions(pred_root, windows)
    devices = pd.read_csv(manifest / "official_devices.csv")
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(dict(
        archive_sha256=expected, prediction_markers_sha256=frozen,
        evaluation_sha256=sha256(Path(__file__)),
        stage="development_only", final_labels_read=False), indent=2) + "\n")
    summaries, class_rows = [], []
    for home in PRIMARY:
        mapping = devices.loc[devices.home == home].set_index("mac").device_type.to_dict()
        y = read_development_labels(archive, members.loc[home, "member"],
                                    home, windows, mapping)
        folder = pred_root / home
        static = np.load(folder / "static_mlp.npy", mmap_mode="r")
        row, rows = metrics(y, static, static, home, -1, "static_mlp")
        summaries.append(row)
        class_rows.extend(rows)
        for seed in range(5):
            for method in METHODS:
                pred = np.load(folder / f"seed_{seed}" / f"{method}.npy", mmap_mode="r")
                row, rows = metrics(y, static, pred, home, seed, method)
                summaries.append(row)
                class_rows.extend(rows)
        print(home, "development evaluated; curated", int((y >= 0).sum()), flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class.csv", index=False)
    means = summary.groupby("method")[["macro_f1_present",
                "negative_flip_rate_vs_static_mlp", "zero_recall_classes"]].mean()
    report = ("Independent temporal IoT development comparison, 11 homes; "
              "five repair seeds/home.\n"
              "Rows outside the public 24-type inventory are excluded from metrics.\n"
              "No final-window labels or features were read. Homes, not seeds, "
              "are independent units.\n"
              + means.to_string() + "\n")
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
