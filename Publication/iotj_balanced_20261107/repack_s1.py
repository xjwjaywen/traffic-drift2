"""Align the verified IoT-J S1 archive with the balanced manuscript.

Only README.txt, RESULTS_INDEX.json, and MANIFEST.csv are changed. All
experiment files are checked byte-for-byte against the full-manuscript S1.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "iotj_minimal_20261103/submission_files/S1_CARE_IoTJ.zip"
RETAINED_LABELS = {
    "tab:absorber_pairs", "tab:main_results", "tab:matched_holdout",
    "tab:cl_baselines", "tab:multiperiod", "tab:ablation",
    "tab:per_class_recovery", "tab:quicext25", "tab:trigger",
    "fig:collapse_timeline", "fig:care_workflow",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_manifest(archive):
    rows = list(csv.DictReader(io.StringIO(archive.read("MANIFEST.csv").decode())))
    names = set(archive.namelist()) - {"MANIFEST.csv"}
    if set(row["path"] for row in rows) != names or len(rows) != len(names):
        raise ValueError("S1 manifest does not cover each member once")
    for row in rows:
        data = archive.read(row["path"])
        if len(data) != int(row["bytes"]) or sha256(data) != row["sha256"]:
            raise ValueError(f"S1 member differs from manifest: {row['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pdf", type=Path, default=HERE / "main.pdf")
    parser.add_argument("--output", type=Path,
                        default=HERE / "submission_files/S1_CARE_IoTJ_balanced.zip")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    pdf = args.pdf.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output == source:
        parser.error(f"Output exists or would overwrite source: {output}")

    with zipfile.ZipFile(source) as old:
        if old.testzip() is not None:
            raise ValueError("Source S1 ZIP failed integrity check")
        verify_manifest(old)
        index = json.loads(old.read("RESULTS_INDEX.json"))
        if index.get("manuscript_pdf_sha256") == sha256(pdf.read_bytes()):
            raise ValueError("Source S1 already indexes the balanced PDF")
        absent = RETAINED_LABELS - (set(index["tables"]) | set(index["figures"]))
        if absent:
            raise ValueError(f"Missing original index entries: {sorted(absent)}")
        manuscript = (HERE / "main.tex").read_text()
        missing = {name for name in RETAINED_LABELS if "\\label{" + name + "}" not in manuscript}
        if missing:
            raise ValueError(f"Retained label absent in manuscript: {sorted(missing)}")
        index["full_iot_manuscript_pdf_sha256"] = index["manuscript_pdf_sha256"]
        index["manuscript_pdf_sha256"] = sha256(pdf.read_bytes())
        index["version"] = "2026-11-07 IEEE IoT-J balanced manuscript metadata"
        index["retained_manuscript_labels"] = sorted(RETAINED_LABELS)
        index["package_adaptation"] = (
            "Only README, RESULTS_INDEX, and MANIFEST changed relative to "
            "the verified full IoT-J S1; all scientific payloads are byte-identical.")
        readme = (
            "This S1 archive accompanies the nine-page balanced IEEE IoT-J "
            "manuscript. RESULTS_INDEX.json retains the original result labels "
            "used by this version. Scientific payloads are unchanged.\n\n"
        ).encode() + old.read("README.txt")
        replacements = {
            "README.txt": readme,
            "RESULTS_INDEX.json": (
                json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode(),
        }
        entries = {}
        for info in old.infolist():
            if info.is_dir() or info.filename == "MANIFEST.csv":
                continue
            entries[info.filename] = replacements.get(info.filename, old.read(info))
        manifest = io.StringIO()
        writer = csv.writer(manifest, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        for name, data in sorted(entries.items()):
            writer.writerow((name, len(data), sha256(data)))
        entries["MANIFEST.csv"] = manifest.getvalue().encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x", allowZip64=True) as new:
            for info in old.infolist():
                new.writestr(info, entries[info.filename])

    with zipfile.ZipFile(source) as old, zipfile.ZipFile(output) as new:
        if new.testzip() is not None:
            raise ValueError("Balanced S1 ZIP failed integrity check")
        verify_manifest(new)
        for name in set(old.namelist()) - {"README.txt", "RESULTS_INDEX.json", "MANIFEST.csv"}:
            if sha256(old.read(name)) != sha256(new.read(name)):
                raise ValueError(f"Scientific payload changed: {name}")
    print(f"S1 BALANCED COMPLETE: {output}")
    print(f"members={len(entries)}, bytes={output.stat().st_size}, "
          f"sha256={sha256(output.read_bytes())}")


if __name__ == "__main__":
    main()
