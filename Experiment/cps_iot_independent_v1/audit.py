"""Read-only coverage audit for the CPSIoTSec'25 smart-home flow dataset.

The held-out test device labels are deliberately never read by this script.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd


LABEL = "deviceName"
TIME = "flowEndMilliseconds"  # Published CSV stores Unix seconds here.


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(path, include_labels):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if LABEL not in header or TIME not in header or len(header) != 24:
        raise ValueError(f"Unexpected schema in {path}: {header}")
    features = tuple(name for name in header if name not in {LABEL, TIME})
    columns = [TIME, LABEL] if include_labels else [TIME]
    frame = pd.read_csv(path, usecols=columns)
    time = pd.to_datetime(pd.to_numeric(frame[TIME], errors="coerce"),
                          unit="s", utc=True, errors="coerce")
    row = {
        "rows": len(frame),
        "invalid_timestamps": int(time.isna().sum()),
        "first_utc": time.min().isoformat() if time.notna().any() else "",
        "last_utc": time.max().isoformat() if time.notna().any() else "",
        "sha256": sha256(path),
    }
    labels = frame[LABEL].value_counts(dropna=False) if include_labels else None
    return features, row, labels


def audit(root):
    coverage, counts = [], []
    common_features = None
    for home in range(1, 13):
        name = f"home{home:02d}"
        folder = root / name
        train_features, train, labels = read_split(folder / "train.csv", True)
        test_features, test, _ = read_split(folder / "test_full.csv", False)
        if train_features != test_features:
            raise ValueError(f"Feature mismatch between train and test in {name}")
        if common_features is None:
            common_features = train_features
        elif common_features != train_features:
            raise ValueError(f"Feature mismatch across homes: {name}")
        if train["invalid_timestamps"] or test["invalid_timestamps"]:
            raise ValueError(f"Invalid timestamp in {name}")
        if train["last_utc"] >= test["first_utc"]:
            raise ValueError(f"Train/test times overlap in {name}")
        for split, row in (("train", train), ("test", test)):
            coverage.append({"home": name, "split": split, **row})
        counts.extend({"home": name, "device_name": str(device), "train_rows": int(n)}
                      for device, n in labels.items())
        print(name, "train", train["rows"], "test", test["rows"], flush=True)
    return pd.DataFrame(coverage), pd.DataFrame(counts), common_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    output = args.output or root.parent / "cps_iot_independent_audit_v1"
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output directory must be new and empty: {output}")
    coverage, counts, features = audit(root)
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output / "coverage.csv", index=False)
    counts.to_csv(output / "train_class_counts.csv", index=False)
    (output / "features.txt").write_text("\n".join(features) + "\n")
    summary = coverage.pivot(index="home", columns="split", values="rows")
    report = "\n".join([
        "CPSIoTSec'25 independent-data coverage audit; no model fitting or evaluation.",
        "Only test timestamps and file hashes were read; test device labels were not read.",
        f"Homes: {summary.shape[0]}; common flow features: {len(features)}.",
        f"Train rows: {int(summary['train'].sum())}; test rows: {int(summary['test'].sum())}.",
        "Rows per home:", summary.to_string(),
        "Next: freeze the chronological source/query split and metrics before test predictions.",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
