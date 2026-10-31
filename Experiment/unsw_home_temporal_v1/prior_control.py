"""Exploratory paid-prior logit calibration control on frozen source logits.

This is a standard, low-cost control, not a novel method. Prediction uses only
the fixed source model, unlabeled query predictions, and the already selected
1000 paid labels per seed. Development labels are read only by the separate
evaluate stage after predictions for all primary homes have been frozen.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
from audit import sha256
from evaluate import PRIMARY, metrics, read_development_labels, verify_predictions
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model


BATCH = 4096
WEIGHT = .5
PSEUDOCOUNT = 1.
EFFECTIVE_QUERY_N = 1000.


def prior_bias(paid_count, predicted_count, query_rows):
    """Shrink predicted and labeled priors on the same effective sample scale."""
    paid = (np.asarray(paid_count, np.float64) + PSEUDOCOUNT)
    predicted = (np.asarray(predicted_count, np.float64)
                 * (EFFECTIVE_QUERY_N / query_rows) + PSEUDOCOUNT)
    if len(paid) != 24 or len(predicted) != 24 or np.any(paid <= 0):
        raise ValueError("Invalid class counts for prior correction")
    return (WEIGHT * (np.log(paid) - np.log(predicted))).astype(np.float32)


def freeze_home(prepared, baseline, output, home, device):
    base = baseline / home
    prep_dir = prepared / home
    folder = output / "predictions" / home
    base_hash = sha256(base / "complete.json")
    prep_hash = sha256(prep_dir / "complete.json")
    base_artifacts = json.loads((base / "complete.json").read_text())["artifacts"]
    for name in ("source_state.pt", "development_features.npy", "static_mlp.npy"):
        if sha256(base / name) != base_artifacts[name]:
            raise ValueError(f"Baseline artifact changed: {home} {name}")
    prep_artifacts = json.loads((prep_dir / "complete.json").read_text())["artifacts"]
    for name in ("source.npz", "query.npz", "query_labels.npy"):
        if sha256(prep_dir / name) != prep_artifacts[name]:
            raise ValueError(f"Prepared artifact changed: {home} {name}")
    config = dict(home=home, baseline_sha256=base_hash,
                  prepared_sha256=prep_hash, bias_weight=WEIGHT,
                  paid_pseudocount=PSEUDOCOUNT,
                  predicted_effective_sample_size=EFFECTIVE_QUERY_N,
                  implementation_sha256=sha256(Path(__file__)),
                  development_labels_read=False, final_window_read=False)
    marker = folder / "complete.json"
    if marker.is_file():
        if json.loads((folder / "config.json").read_text()) != config:
            raise ValueError(f"Completed prior control configuration changed: {home}")
        artifacts = json.loads(marker.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Completed prior control artifact changed: {home}")
        print(home, "verified prior-control prediction reuse", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete prior-control output; use a new output: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(prep_dir / "source.npz", allow_pickle=False) as data:
        raw_source = data["x"]
    with np.load(prep_dir / "query.npz", allow_pickle=False) as data:
        raw_query = data["x"]
    paid_labels = np.load(prep_dir / "query_labels.npy", mmap_mode="r")
    prep = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(base / "source_state.pt",
                                     map_location="cpu", weights_only=True))
    model.to(device).eval()
    counts = np.zeros(24, np.int64)
    with torch.no_grad():
        for start in range(0, len(raw_query), BATCH):
            x = prep.transform(raw_query[start:start + BATCH])
            logits = model(torch.as_tensor(x, dtype=torch.float32,
                                           device=device))
            counts += np.bincount(logits.argmax(1).cpu().numpy(), minlength=24)
    dev_features = np.load(base / "development_features.npy", mmap_mode="r")
    dev_logits = np.lib.format.open_memmap(folder / "source_dev_logits.npy",
                                           mode="w+", dtype="float32",
                                           shape=(len(dev_features), 24))
    with torch.no_grad():
        for start in range(0, len(dev_features), BATCH):
            features = np.array(dev_features[start:start + BATCH], copy=True)
            logits = model.head(torch.as_tensor(features, dtype=torch.float32,
                                                device=device))
            dev_logits[start:start + BATCH] = logits.cpu().numpy()
    model.cpu()
    dev_logits.flush()
    baseline_static = np.load(base / "static_mlp.npy", mmap_mode="r")
    if not np.array_equal(np.asarray(dev_logits).argmax(1), baseline_static):
        raise ValueError(f"Reconstructed source logits differ from baseline: {home}")
    for seed in range(5):
        chosen = np.load(base / f"seed_{seed}" / "paid_query_pool_indices.npy")
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid query indices differ from protocol: {home} {seed}")
        paid_count = np.bincount(paid_labels[chosen], minlength=24)
        bias = prior_bias(paid_count, counts, len(raw_query))
        pred = np.lib.format.open_memmap(folder / f"seed_{seed}.npy",
                                         mode="w+", dtype="uint8",
                                         shape=(len(dev_logits),))
        for start in range(0, len(dev_logits), BATCH):
            pred[start:start + BATCH] = np.argmax(
                dev_logits[start:start + BATCH] + bias[None, :], axis=1)
        pred.flush()
        del pred
        np.save(folder / f"bias_seed_{seed}.npy", bias)
        np.save(folder / f"paid_count_seed_{seed}.npy", paid_count)
        print(home, "seed", seed, "prior correction complete", flush=True)
    del dev_logits
    np.save(folder / "query_predicted_count.npy", counts)
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    artifacts = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    marker.write_text(json.dumps(dict(artifacts=artifacts), indent=2) + "\n")
    print(home, "all prior-control predictions frozen", flush=True)


def evaluate_frozen(archive, baseline, output, manifest):
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    verify_predictions(baseline, windows)
    expected_archive = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected_archive:
        raise ValueError("Archive differs from reviewed timestamp audit")
    for home in PRIMARY:
        folder = output / "predictions" / home
        marker = folder / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Prior-control predictions incomplete: {home}")
        artifacts = json.loads(marker.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Prior-control artifact changed: {home}")
        n = int(windows.loc[(home, "development"), "rows"])
        for seed in range(5):
            pred = np.load(folder / f"seed_{seed}.npy", mmap_mode="r")
            if pred.shape != (n,) or pred.dtype != np.uint8 or pred.max() >= 24:
                raise ValueError(f"Prior-control prediction invalid: {home} {seed}")
    result = output / "development"
    if result.exists() and any(result.iterdir()):
        raise ValueError(f"Prior-control evaluation output must be empty: {result}")
    result.mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(manifest / "official_devices.csv")
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    rows, classes = [], []
    for home in PRIMARY:
        mapping = official.loc[official.home == home].set_index("mac").device_type.to_dict()
        y = read_development_labels(archive, members.loc[home, "member"],
                                    home, windows, mapping)
        static = np.load(baseline / home / "static_mlp.npy", mmap_mode="r")
        base_row, base_classes = metrics(y, static, static, home, -1, "static_mlp")
        rows.append(base_row)
        classes.extend(base_classes)
        for seed in range(5):
            pred = np.load(output / "predictions" / home / f"seed_{seed}.npy",
                           mmap_mode="r")
            row, per_class = metrics(y, static, pred, home, seed, "prior_static")
            rows.append(row)
            classes.extend(per_class)
        print(home, "prior control development evaluated", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(result / "summary.csv", index=False)
    pd.DataFrame(classes).to_csv(result / "per_class.csv", index=False)
    means = table.groupby("method")[["macro_f1_present",
                "negative_flip_rate_vs_static_mlp", "zero_recall_classes"]].mean()
    home_means = (table.groupby(["home", "method"])["macro_f1_present"]
                  .mean().unstack("method"))
    home_means["prior_minus_static_pp"] = 100 * (
        home_means["prior_static"] - home_means["static_mlp"])
    report = ("Exploratory standard class-prior correction on source logits.\n"
              "Fixed log-ratio weight 0.5, pseudocount 1, same paid 1000 labels.\n"
              "No development labels inform biases; final window unopened.\n"
              "This was proposed after viewing development results; not blind.\n"
              + means.to_string() + "\n\nPer-home macro F1:\n"
              + home_means.to_string() + "\n")
    (result / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["predict", "evaluate"])
    parser.add_argument("--archive", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal/all.zip"))
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_prior_control_v1"))
    parser.add_argument("--manifest", type=Path, default=HERE / "metadata")
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    output = args.output.resolve()
    if args.stage == "predict":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA unavailable; use --device cpu")
        prepared = args.prepared.resolve(strict=True)
        for home in ([args.home] if args.home else PRIMARY):
            freeze_home(prepared, baseline, output, home,
                        torch.device(args.device))
    else:
        if args.home:
            parser.error("Evaluation requires predictions from all primary homes")
        evaluate_frozen(args.archive.resolve(strict=True), baseline, output,
                        args.manifest.resolve(strict=True))


if __name__ == "__main__":
    main()
