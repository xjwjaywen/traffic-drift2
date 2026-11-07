"""Export the frozen IoT final-window records for reviewer access.

Run on the server that holds ``unsw_iot_final_holdout_v1``. The public raw
archive, intermediate feature arrays, and training caches are not copied.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOMES = [f"home{i:02d}" for i in range(1, 13) if i != 10]
SEEDS = range(5)
METHODS = ("target", "care", "bcwi_050")
EVALUATION_FILES = (
    "config.json", "summary.csv", "per_class.csv", "diagnostics.csv",
    "home_means.csv", "seed_spread.csv", "associations.csv", "REPORT.txt",
    "complete.json",
)
CODE_FILES = (
    "Experiment/unsw_home_temporal_v1/FINAL_HOLDOUT_PROTOCOL.md",
    "Experiment/unsw_home_temporal_v1/audit.py",
    "Experiment/unsw_home_temporal_v1/prepare.py",
    "Experiment/unsw_home_temporal_v1/predict.py",
    "Experiment/unsw_home_temporal_v1/evaluate.py",
    "Experiment/unsw_home_temporal_v1/final_predict.py",
    "Experiment/unsw_home_temporal_v1/final_evaluate.py",
    "Experiment/tma_pilot_v1/data.py",
    "Experiment/tma_pilot_v1/existing_care.py",
    "Experiment/tma_pilot_v1/pilot.py",
    "Experiment/tma_server_v1/stage0.py",
    "Experiment/cps_iot_independent_v1/development.py",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_artifacts(folder, marker, *, omitted=()):
    recorded = json.loads(marker.read_text())["artifacts"]
    if not isinstance(recorded, dict):
        raise ValueError(f"Invalid artifact marker: {marker}")
    for name, expected in recorded.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path: {name}")
        path = folder / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Artifact missing or changed: {path}")
    if set(omitted) - set(recorded):
        raise ValueError(f"Expected omitted artifact missing from marker: {marker}")
    return recorded


def add_file(files, source, name):
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Missing file or symlink: {source}")
    if name in files:
        raise ValueError(f"Duplicate archive entry: {name}")
    files[name] = source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_iot_final_holdout_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/"
                                     "submission/S3_IoT_final_holdout.zip"))
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    evaluation = root / "evaluation"
    predictions = root / "predictions"
    eval_marker = evaluation / "complete.json"
    eval_artifacts = checked_artifacts(evaluation, eval_marker)
    if set(eval_artifacts) != set(EVALUATION_FILES) - {"complete.json"}:
        raise ValueError("Evaluation file set differs from the frozen protocol")
    eval_config = json.loads((evaluation / "config.json").read_text())
    protocol = REPO / CODE_FILES[0]
    if (eval_config.get("stage") != "final_holdout" or
            eval_config.get("comparison_frozen") is not True or
            eval_config.get("protocol_sha256") != digest(protocol) or
            eval_config.get("evaluation_sha256") !=
                digest(REPO / "Experiment/unsw_home_temporal_v1/final_evaluate.py")):
        raise ValueError("Evaluation config differs from reviewed code/protocol")
    if set(eval_config.get("prediction_markers_sha256", {})) != set(HOMES):
        raise ValueError("Incomplete final prediction markers")

    means = pd.read_csv(evaluation / "home_means.csv")
    if (set(means.home) != set(HOMES) or
            set(means.method) != {"static_mlp", *METHODS} or
            len(means) != len(HOMES) * 4):
        raise ValueError("Home summary does not have 11 homes and four methods")
    reported = means.groupby("method").macro_f1_present.mean()
    for method, expected in (("static_mlp", .52006), ("care", .51239),
                             ("bcwi_050", .52858)):
        if abs(reported[method] - expected) > .00002:
            raise ValueError(f"Paper value differs for {method}: {reported[method]}")

    files = {}
    for name in EVALUATION_FILES:
        add_file(files, evaluation / name, f"evaluation/{name}")
    for home in HOMES:
        folder = predictions / home
        marker = folder / "complete.json"
        if digest(marker) != eval_config["prediction_markers_sha256"][home]:
            raise ValueError(f"Prediction marker differs from scored record: {home}")
        recorded = checked_artifacts(folder, marker, omitted=("final_features.npy",))
        required = {"config.json", "static_mlp.npy", "final_features.npy"}
        required.update(f"seed_{seed}/{method}.npy"
                        for seed in SEEDS for method in METHODS)
        if set(recorded) != required:
            raise ValueError(f"Unexpected prediction file set: {home}")
        config = json.loads((folder / "config.json").read_text())
        if (config.get("home") != home or config.get("label_budget") != 1000 or
                config.get("repair_seeds") != list(SEEDS) or
                config.get("old_weight") != .5 or
                config.get("protocol_sha256") != digest(protocol) or
                config.get("implementation_sha256") !=
                    digest(REPO / "Experiment/unsw_home_temporal_v1/final_predict.py") or
                config.get("final_labels_read") is not False):
            raise ValueError(f"Prediction config differs: {home}")
        for name in sorted(required - {"final_features.npy"}):
            add_file(files, folder / name, f"predictions/{home}/{name}")
        add_file(files, marker, f"predictions/{home}/complete.json")
    for relative in CODE_FILES:
        add_file(files, REPO / relative, f"code/{relative}")
    metadata = REPO / "Experiment/unsw_home_temporal_v1/metadata"
    for path in sorted(metadata.iterdir()):
        if path.is_file():
            add_file(files, path, f"code/Experiment/unsw_home_temporal_v1/metadata/{path.name}")

    readme = ("Supplementary Material S3: 11-home IoT final-window records\n\n"
              "This archive contains the frozen evaluation CSVs and reports, "
              "per-row final-window predictions for all four methods, and the "
              "protocol, code, and metadata used for this analysis. Home/seed "
              "predictions use raw archive row order after the development "
              "window; the public archive and official device inventory are "
              "needed to rescore them. See code/Experiment/unsw_home_temporal_v1/"
              "FINAL_HOLDOUT_PROTOCOL.md.\n\n"
              "The raw public traffic archive, intermediate final_features.npy "
              "arrays, source checkpoints, and training caches are not copied "
              "here. Their hashes remain in the original complete.json markers "
              "and were verified at export. This is a curated mixed-protocol, "
              "closed-set device-type stress test, not an encrypted-only IoT "
              "deployment or an independent-site replication. The five seeds "
              "within each home are overlapping repeats.\n\n"
              "MANIFEST.json lists the SHA256 and byte size of each copied file.\n")
    manifest = {name: {"sha256": digest(path), "bytes": path.stat().st_size}
                for name, path in sorted(files.items())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        archive.writestr("README.txt", readme)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    print(f"S3 COMPLETE: {output}")
    print(f"files={len(files) + 2}, bytes={output.stat().st_size}, sha256={digest(output)}")


if __name__ == "__main__":
    main()
