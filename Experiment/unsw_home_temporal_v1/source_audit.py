"""Count device labels in frozen source days only; never parse later labels."""

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import pandas as pd

from audit import HOME, sha256


def run(archive, manifest, output):
    expected = json.loads((manifest / "audit.json").read_text())
    if sha256(archive) != expected["archive_sha256"]:
        raise ValueError("Archive differs from timestamp audit")
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Output must be new and empty: {output}")
    windows = pd.read_csv(manifest / "windows.csv")
    source_days = {}
    for home, group in windows.groupby("home"):
        source = group.loc[group.split == "source"]
        if len(source) != 1:
            raise ValueError(f"Missing frozen source window: {home}")
        start, end = source.iloc[0][["first_day", "last_day"]]
        source_days[home] = (start, end, int(source.iloc[0]["rows"]))
    counts = []
    with zipfile.ZipFile(archive) as files:
        for info in files.infolist():
            found = HOME.search(Path(info.filename).name)
            if not found or info.is_dir():
                continue
            home = f"home{int(found.group(1)):02d}"
            first, last, expected_rows = source_days[home]
            tally = Counter()
            with files.open(info) as stream:
                header = stream.readline()
                if not header.rstrip().endswith(b",device"):
                    raise ValueError(f"Unexpected label column: {info.filename}")
                for line in stream:
                    day = line[:10].decode("ascii")
                    if first <= day <= last:
                        label = line.rsplit(b",", 1)[-1].strip().decode("ascii")
                        if not label:
                            raise ValueError(f"Empty source label: {info.filename}")
                        tally[label] += 1
            if sum(tally.values()) != expected_rows:
                raise ValueError(f"Source row count differs from timestamp audit: {home}")
            for device, rows in sorted(tally.items()):
                counts.append(dict(home=home, device=device, source_rows=rows))
            print(home, "source rows", expected_rows, "source classes", len(tally),
                  "min support", min(tally.values()), flush=True)
    output.mkdir(parents=True)
    frame = pd.DataFrame(counts)
    frame.to_csv(output / "source_class_counts.csv", index=False)
    grouped = frame.groupby("home").source_rows.agg(
        classes="count", min_support="min", median_support="median",
        max_support="max", rows="sum")
    report = "\n".join([
        "UNSW home-IPFIX source-window label audit only.",
        "Query, development and final-window label values were not parsed.",
        "Frozen source class coverage:", grouped.to_string(),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "metadata")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_source_audit_v1"))
    args = parser.parse_args()
    run(args.archive.resolve(strict=True), args.manifest.resolve(strict=True),
        args.output.resolve())


if __name__ == "__main__":
    main()
