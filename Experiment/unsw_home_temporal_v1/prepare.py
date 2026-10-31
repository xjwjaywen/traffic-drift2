"""Prepare source/query rows and unlabeled development features from all.zip.

Query membership is curated by the published 24-device inventory. Query labels
are used only for that curation and saved for later paid selection; the
reservoir sampler does not use their class values.
Development labels are never decoded; final-window rows are never parsed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import FEATURES, sha256


SOURCE_CAP = 5000
QUERY_CAP = 100000


def add_reservoir(items, seen, cap, value, rng):
    """Algorithm R: each seen row has equal inclusion chance."""
    seen += 1
    if len(items) < cap:
        items.append(value)
    else:
        slot = int(rng.integers(seen))
        if slot < cap:
            items[slot] = value
    return seen


def prepare_home(archive, member, home, manifest, output, archive_hash):
    home_out = output / home
    if home_out.is_symlink():
        raise ValueError(f"Output must not be a symlink: {home_out}")
    windows = pd.read_csv(manifest / "windows.csv")
    intervals = windows.loc[windows.home == home].set_index("split")
    if set(intervals.index) != {"source", "query", "development", "final"}:
        raise ValueError(f"Bad split manifest: {home}")
    cut = [str(intervals.loc[name, "last_day"]) for name in
           ("source", "query", "development", "final")]
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = official.loc[official.home == home].set_index("mac").device_type.to_dict()
    if len(mapping) != 24 or set(mapping.values()) != set(range(24)):
        raise ValueError(f"Incomplete official device map: {home}")
    expected_source = pd.read_csv(manifest / "official_source_counts.csv")
    expected_source = (expected_source.loc[expected_source.home == home]
                       .set_index("device_type").source_rows.reindex(range(24))
                       .to_numpy(np.int64))
    config = dict(schema=1, home=home, archive_sha256=archive_hash,
                  member=member, windows={name: dict(first_day=str(row.first_day),
                  last_day=str(row.last_day), rows=int(row.rows))
                  for name, row in intervals.iterrows()},
                  source_cap_per_class=SOURCE_CAP, query_cap=QUERY_CAP,
                  features=list(FEATURES), source_seed=0, query_seed=0,
                  implementation_sha256=sha256(Path(__file__)),
                  official_mapping_sha256=sha256(manifest / "official_devices.csv"),
                  label_boundary="development features only; final rows skipped")
    marker = home_out / "complete.json"
    if marker.is_file():
        if json.loads((home_out / "config.json").read_text()) != config:
            raise ValueError(f"Completed configuration changed: {home_out}")
        artifacts = json.loads(marker.read_text())["artifacts"]
        if any(sha256(home_out / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Completed preparation artifact changed: {home_out}")
        print(home, "verified completed preparation reused", flush=True)
        return
    if home_out.exists() and any(home_out.iterdir()):
        raise ValueError(f"Incomplete preparation output: {home_out}")
    home_out.mkdir(parents=True)
    (home_out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = [[] for _ in range(24)]
    source_seen = np.zeros(24, np.int64)
    source_rng = [np.random.default_rng(100000 + c) for c in range(24)]
    query = []
    query_seen = 0
    query_rng = np.random.default_rng(200000)
    split_rows = np.zeros(4, np.int64)
    dev_n = int(intervals.loc["development", "rows"])
    dev_x = np.lib.format.open_memmap(home_out / "development_all_x.npy",
                                      mode="w+", dtype="float32",
                                      shape=(dev_n, len(FEATURES)))
    dev_pos = 0
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected columns in {member}")
        feature_indices = [header.index(name) for name in FEATURES]
        for row_id, line in enumerate(stream):
            day = line[:10].decode("ascii")
            if day <= cut[0]:
                stage = 0
            elif day <= cut[1]:
                stage = 1
            elif day <= cut[2]:
                stage = 2
            elif day <= cut[3]:
                stage = 3
            else:
                raise ValueError(f"Timestamp beyond manifest: {home} {day}")
            split_rows[stage] += 1
            if stage == 3:
                # Do not split or decode any final-window label or feature.
                continue
            if stage != 2:
                # The known-device inventory defines the curated source/query
                # pool. The model does not receive this MAC as an input.
                mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
                label = mapping.get(mac)
                if label is None:
                    continue
            else:
                label = None  # No development label is decoded.
            values = line.rstrip(b"\r\n").split(b",")
            if len(values) != len(header):
                raise ValueError(f"Malformed CSV row: {home} row {row_id}")
            x = np.asarray([float(values[i]) if values[i] else np.nan
                            for i in feature_indices], dtype=np.float32)
            x[~np.isfinite(x)] = np.nan
            if stage == 0:
                source_seen[label] = add_reservoir(
                    source[label], int(source_seen[label]), SOURCE_CAP,
                    (row_id, x), source_rng[label])
            elif stage == 1:
                query_seen = add_reservoir(query, query_seen, QUERY_CAP,
                                           (row_id, label, x), query_rng)
            else:
                dev_x[dev_pos] = x
                dev_pos += 1
    dev_x.flush()
    del dev_x
    expected_rows = intervals.loc[["source", "query", "development", "final"],
                                  "rows"].to_numpy(np.int64)
    if not np.array_equal(split_rows, expected_rows) or dev_pos != dev_n:
        raise ValueError(f"Timestamp row counts differ from audit: {home}")
    if not np.array_equal(source_seen, expected_source):
        raise ValueError(f"Official source labels differ from source audit: {home}")
    source_rows = [item for bucket in source for item in bucket]
    source_y = np.concatenate([np.full(len(bucket), c, np.int64)
                               for c, bucket in enumerate(source)])
    source_ids = np.asarray([item[0] for item in source_rows], np.int64)
    source_x = np.stack([item[1] for item in source_rows])
    query_ids = np.asarray([item[0] for item in query], np.int64)
    query_y = np.asarray([item[1] for item in query], np.int64)
    query_x = np.stack([item[2] for item in query])
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("Duplicate sampled row IDs")
    np.savez_compressed(home_out / "source.npz", ids=source_ids, x=source_x, y=source_y)
    # Keep the paid labels in a separate memory-mappable file so a later
    # experiment can index only its already selected query IDs.
    np.savez_compressed(home_out / "query.npz", ids=query_ids, x=query_x)
    np.save(home_out / "query_labels.npy", query_y)
    summary = dict(home=home, raw_rows_by_stage=split_rows.tolist(),
                   curated_source_rows=int(source_seen.sum()),
                   retained_source_rows=len(source_ids),
                   curated_query_rows=query_seen, retained_query_rows=len(query_ids),
                   development_all_rows=dev_n, source_classes_present=int((source_seen > 0).sum()),
                   query_labels_not_used_for_selection=True,
                   development_labels_read=False, final_rows_parsed=False)
    (home_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = [p for p in home_out.iterdir() if p.is_file() and p.name != "complete.json"]
    (home_out / "complete.json").write_text(json.dumps(dict(
        artifacts={p.name: sha256(p) for p in files}), indent=2) + "\n")
    print(home, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "metadata")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--home", choices=[f"home{i:02d}" for i in range(1, 13)])
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    archive_hash = sha256(archive)
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if archive_hash != expected:
        parser.error("Archive SHA256 differs from reviewed timestamp audit")
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for home in ([args.home] if args.home else list(members.index)):
        prepare_home(archive, members.loc[home, "member"], home, manifest,
                     output, archive_hash)


if __name__ == "__main__":
    main()
