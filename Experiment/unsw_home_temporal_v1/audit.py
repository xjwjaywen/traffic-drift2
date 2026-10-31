"""Timestamp-only coverage audit of UNSW's 12-home IPFIX flow archive.

No device-label values are decoded or counted. This script only checks archive
structure, CSV headers, timestamps, and daily row coverage. It trains nothing.
"""

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import zipfile

import pandas as pd


ACTIVITY = (
    "PacketCount", "ByteCount", "SmallPktCount", "LargePktCount",
    "NonEmptyPktCount", "DataByteCount", "AvgIAT",
    "FirstNonEmptyPktSize", "MaxPktSize", "StdevPayloadSize", "StdevIAT",
)
FEATURES = tuple(prefix + name for prefix in ("out", "in") for name in ACTIVITY) + (
    "http", "https", "dns", "ntp", "tcp", "udp",
)
HOME = re.compile(r"home(\d{1,2})\.csv$", re.IGNORECASE)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(archive, output):
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Output must be new and empty: {output}")
    daily, members, columns = [], [], None
    with zipfile.ZipFile(archive) as files:
        matched = []
        for info in files.infolist():
            found = HOME.search(Path(info.filename).name)
            if found and not info.is_dir():
                matched.append((int(found.group(1)), info))
        if sorted(number for number, _ in matched) != list(range(1, 13)):
            raise ValueError("Expected exactly home1.csv through home12.csv")
        for number, info in sorted(matched):
            counts = Counter()
            with files.open(info) as stream:
                header = stream.readline().decode("utf-8-sig").strip().split(",")
                if header[0] != "start" or header[-1] != "device" or not set(
                        FEATURES).issubset(header) or len(header) != len(set(header)):
                    raise ValueError(f"Unexpected header: {info.filename}")
                if columns is None:
                    columns = header
                elif columns != header:
                    raise ValueError(f"CSV columns differ: {info.filename}")
                for line in stream:
                    # Only the bytes before the first comma are inspected.
                    separator = line.find(b",")
                    if separator < 10:
                        raise ValueError(f"Malformed timestamp row: {info.filename}")
                    stamp = line[:separator]
                    day = stamp[:10].decode("ascii")
                    date.fromisoformat(day)
                    counts[day] += 1
            for day, rows in sorted(counts.items()):
                daily.append(dict(home=f"home{number:02d}", day=day, rows=rows))
            members.append(dict(home=f"home{number:02d}", member=info.filename,
                                compressed_bytes=info.compress_size,
                                uncompressed_bytes=info.file_size,
                                rows=sum(counts.values()), first_day=min(counts),
                                last_day=max(counts), observed_days=len(counts)))
            print(f"home{number:02d}: {sum(counts.values())} rows, "
                  f"{len(counts)} days, {min(counts)}..{max(counts)}", flush=True)
    output.mkdir(parents=True)
    pd.DataFrame(daily).to_csv(output / "daily_rows.csv", index=False)
    pd.DataFrame(members).to_csv(output / "members.csv", index=False)
    all_days = sorted({row["day"] for row in daily})
    metadata = dict(archive=str(archive), archive_sha256=sha256(archive),
                    total_rows=sum(row["rows"] for row in members),
                    global_first_day=all_days[0], global_last_day=all_days[-1],
                    global_observed_days=len(all_days), homes=len(members),
                    checked_feature_columns=list(FEATURES),
                    label_values_read=False, label_column="device")
    (output / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = "\n".join([
        "UNSW home-IPFIX timestamp-only coverage audit; no device-label values read.",
        f"Archive SHA256: {metadata['archive_sha256']}",
        f"Homes={len(members)}, rows={metadata['total_rows']}, "
        f"global days={len(all_days)} ({all_days[0]}..{all_days[-1]}).",
        "Per-home coverage:", pd.DataFrame(members).to_string(index=False),
        "Inspect daily_rows.csv before fixing chronological development/test windows.",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_audit_v1"))
    args = parser.parse_args()
    audit(args.archive.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
