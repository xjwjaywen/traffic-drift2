"""Audit daily closed-set support before designing delayed-label online replay.

Reads query and already examined development days only. The official device
inventory identifies curated rows; no model is trained or evaluated. Final
window labels and features are not decoded. This is a capacity audit, not a
method result or independent test.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import sha256
from predict import PRIMARY

HERE = Path(__file__).resolve().parent


def audit_home(files, member, home, windows, mapping, expected_daily):
    source_end = str(windows.loc[(home, "source"), "last_day"])
    query_end = str(windows.loc[(home, "query"), "last_day"])
    development_end = str(windows.loc[(home, "development"), "last_day"])
    totals = Counter()
    classes = Counter()
    with files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {member}")
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= source_end or day > development_end:
                # Only the timestamp prefix is inspected outside the audit window.
                continue
            split = "query" if day <= query_end else "development"
            totals[(split, day)] += 1
            mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
            device_type = mapping.get(mac, -1)
            classes[(split, day, device_type)] += 1
    daily_rows = []
    class_rows = []
    for (split, day), total in sorted(totals.items(), key=lambda item: item[0][1]):
        reviewed = expected_daily.loc[(home, day), "rows"]
        if total != reviewed:
            raise ValueError(f"Timestamp audit row count differs: {home} {day}")
        counts = np.array([classes[(split, day, c)] for c in range(24)],
                          dtype=np.int64)
        curated = int(counts.sum())
        if curated > total:
            raise AssertionError("Curated rows exceed total")
        positive = counts[counts > 0]
        if curated:
            expected_per_1000 = 1000 * counts / curated
            low_one = int(np.sum((counts > 0) & (expected_per_1000 < 1)))
            low_five = int(np.sum((counts > 0) & (expected_per_1000 < 5)))
        else:
            low_one = low_five = 0
        daily_rows.append(dict(home=home, split=split, day=day,
                               total_rows=total, curated_rows=curated,
                               other_rows=total-curated,
                               curated_fraction=curated/total,
                               present_classes=len(positive),
                               min_present_support=int(positive.min()) if len(positive) else 0,
                               present_classes_expected_lt_1_of_1000=low_one,
                               present_classes_expected_lt_5_of_1000=low_five))
        for c, support in enumerate(counts):
            class_rows.append(dict(home=home, split=split, day=day,
                                   device_type=c, support=int(support),
                                   curated_share=support/curated if curated else 0))
    for split in ("query", "development"):
        found = sum(row["total_rows"] for row in daily_rows if row["split"] == split)
        reviewed = int(windows.loc[(home, split), "rows"])
        if found != reviewed:
            raise ValueError(f"Window row count differs: {home} {split}")
    return daily_rows, class_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_day_feasibility_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    reviewed = json.loads((manifest / "audit.json").read_text())
    if sha256(archive) != reviewed["archive_sha256"]:
        parser.error("Archive differs from reviewed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    expected_daily = pd.read_csv(manifest / "daily_rows.csv").set_index(["home", "day"])
    official = pd.read_csv(manifest / "official_devices.csv")
    homes = [args.home] if args.home else PRIMARY
    daily, classes = [], []
    with zipfile.ZipFile(archive) as files:
        for home in homes:
            mapping = (official.loc[official.home == home]
                       .set_index("mac").device_type.to_dict())
            home_daily, home_classes = audit_home(
                files, members.loc[home, "member"], home, windows, mapping,
                expected_daily)
            daily.extend(home_daily)
            classes.extend(home_classes)
            print(home, "audited", len(home_daily), "later days", flush=True)
    daily = pd.DataFrame(daily)
    classes = pd.DataFrame(classes)
    per_home = (daily.groupby(["home", "split"], as_index=False)
                .agg(days=("day", "count"),
                     min_curated_per_day=("curated_rows", "min"),
                     median_curated_per_day=("curated_rows", "median"),
                     min_present_classes=("present_classes", "min"),
                     median_present_classes=("present_classes", "median"),
                     median_classes_lt_5_expected=(
                         "present_classes_expected_lt_5_of_1000", "median")))
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily.csv", index=False)
    classes.to_csv(output / "daily_classes.csv", index=False)
    per_home.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "UNSW closed-set daily online-replay capacity audit; query and examined development only.",
        "The official device inventory defines curated rows; no model fitted or scored.",
        "The <5 column counts present classes whose expected count in 1000 uniform curated labels is <5.",
        "These are oracle class shares used for feasibility, not an available query policy.",
        "No final-window label or feature decoded; final data remains unopened.",
        per_home.round(2).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        homes=homes, archive_sha256=reviewed["archive_sha256"],
        implementation_sha256=sha256(Path(__file__)),
        final_window_read=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
