"""Relabel the original CESNET S1 archive for the IoT-J manuscript.

The scientific payload is copied byte-for-byte. Only README.txt,
RESULTS_INDEX.json, and MANIFEST.csv are changed. The new archive is never
written over the original one.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
TITLE = "Collapse-Aware Maintenance of Encrypted Traffic Classifiers under Temporal Drift"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_manifest(archive):
    rows = list(csv.DictReader(io.StringIO(archive.read("MANIFEST.csv").decode())))
    names = set(archive.namelist()) - {"MANIFEST.csv"}
    if set(row["path"] for row in rows) != names or len(rows) != len(names):
        raise ValueError("Original S1 manifest does not cover each member once")
    for row in rows:
        data = archive.read(row["path"])
        if len(data) != int(row["bytes"]) or sha256(data) != row["sha256"]:
            raise ValueError(f"Original S1 member differs: {row['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="The verified JNCA S1 ZIP")
    parser.add_argument("--pdf", type=Path, default=HERE / "main.pdf")
    parser.add_argument("--output", type=Path,
                        default=HERE / "submission_files" / "S1_CARE_IoTJ.zip")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    pdf = args.pdf.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or source == output:
        parser.error(f"Output exists or would overwrite source: {output}")

    with zipfile.ZipFile(source) as original:
        if original.testzip() is not None:
            raise ValueError("Original S1 ZIP failed integrity check")
        verify_manifest(original)
        old_readme = original.read("README.txt").decode()
        if "Journal of Network and Computer Applications" not in old_readme:
            raise ValueError("Unexpected original S1 README")
        remainder = old_readme.split("Contents\n", 1)[1]
        readme = (
            "Supplementary Material S1\n"
            "Implementation files and archived experiment records for\n"
            f"{TITLE}\n\n"
            "This archive accompanies the IEEE Internet of Things Journal "
            "submission prepared in November 2026. It preserves the original "
            "CESNET scientific payload and the later S2 reconstruction controls. "
            "Only this README, RESULTS_INDEX.json, and MANIFEST.csv were "
            "updated for the current manuscript. The separate 11-home IoT "
            "device-type results are in Supplementary Material S3, not S1.\n\n"
            "Contents\n" + remainder
        ).encode()
        index = json.loads(original.read("RESULTS_INDEX.json"))
        previous_pdf = index["manuscript_pdf_sha256"]
        index["version"] = "2026-11-07 IEEE IoT-J metadata adaptation"
        index["original_2026_10_17_manuscript_pdf_sha256"] = index.get(
            "previous_manuscript_pdf_sha256")
        index["previous_manuscript_pdf_sha256"] = previous_pdf
        index["manuscript_pdf_sha256"] = sha256(pdf.read_bytes())
        index["external_supplements"] = {
            "S2": "Reconstruction-control explanation; local_controls/ payload is in S1",
            "S3": "11-home IoT device-type final-window records; separate ZIP",
            "tab:iot_homes": "S3 evaluation/home_means.csv and summary.csv",
        }
        index["package_adaptation"] = (
            "Only README, RESULTS_INDEX, and MANIFEST changed relative to the "
            "verified JNCA S1. All scientific payloads are byte-identical."
        )
        replacements = {
            "README.txt": readme,
            "RESULTS_INDEX.json": (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode(),
        }
        entries = {}
        for info in original.infolist():
            if info.is_dir() or info.filename == "MANIFEST.csv":
                continue
            entries[info.filename] = replacements.get(info.filename,
                                                      original.read(info))
        if len(entries) != len(original.namelist()) - 1:
            raise ValueError("Unexpected directory or duplicate member in S1")
        manifest_io = io.StringIO()
        writer = csv.writer(manifest_io, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        for name, data in sorted(entries.items()):
            writer.writerow((name, len(data), sha256(data)))
        entries["MANIFEST.csv"] = manifest_io.getvalue().encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x", allowZip64=True) as updated:
            for info in original.infolist():
                updated.writestr(info, entries[info.filename])
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output) as updated:
        if updated.testzip() is not None:
            raise ValueError("Repacked S1 failed ZIP integrity check")
        verify_manifest(updated)
        for name in set(original.namelist()) - {
                "README.txt", "RESULTS_INDEX.json", "MANIFEST.csv"}:
            if sha256(original.read(name)) != sha256(updated.read(name)):
                raise ValueError(f"Scientific payload changed: {name}")
    print(f"S1 COMPLETE: {output}")
    print(f"members={len(entries)}, bytes={output.stat().st_size}, "
          f"sha256={sha256(output.read_bytes())}")


if __name__ == "__main__":
    main()
