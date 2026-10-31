"""Freeze standard-control predictions on UNSW development flows before labels.

Only the 1000 randomly selected query labels are indexed for each repair seed.
Neither development nor final labels are opened by this program.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
from audit import sha256
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
sys.path.insert(0, str(HERE.parent / "cps_iot_independent_v1"))
from data import Preprocessor
from existing_care import fit_controlled
from pilot import Model, forward_batches
from development import source_model


PRIMARY = [f"home{i:02d}" for i in range(1, 13) if i != 10]
SEEDS = range(5)
SOURCE_EPOCHS = 30
REPAIR_STEPS = 500
BUDGET = 1000
CHUNK = 4096
RF = dict(n_estimators=200, min_samples_leaf=2,
          class_weight="balanced_subsample", max_features="sqrt")


def verify_prepared(folder):
    marker = folder / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Preparation incomplete: {folder}")
    artifacts = json.loads(marker.read_text())["artifacts"]
    if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
        raise ValueError(f"Preparation artifact changed: {folder}")
    config = json.loads((folder / "config.json").read_text())
    if config["source_cap_per_class"] != 5000 or config["query_cap"] != 100000:
        raise ValueError(f"Preparation differs from fixed protocol: {folder}")
    return sha256(marker)


def predict_head(head, features, output, device):
    predictions = np.lib.format.open_memmap(output, mode="w+", dtype="uint8",
                                             shape=(len(features),))
    with torch.no_grad():
        for start in range(0, len(features), CHUNK):
            batch = torch.as_tensor(np.array(features[start:start + CHUNK], copy=True),
                                    dtype=torch.float32, device=device)
            predictions[start:start + CHUNK] = head(batch).argmax(1).cpu().numpy()
    predictions.flush()
    del predictions


def predict_nearest(features, support_x, support_y, output, device):
    support = F.normalize(torch.as_tensor(support_x, dtype=torch.float32,
                                          device=device), dim=1)
    labels = torch.as_tensor(support_y, dtype=torch.long, device=device)
    predictions = np.lib.format.open_memmap(output, mode="w+", dtype="uint8",
                                             shape=(len(features),))
    with torch.no_grad():
        for start in range(0, len(features), CHUNK):
            batch = F.normalize(torch.as_tensor(np.array(features[start:start + CHUNK], copy=True),
                                                dtype=torch.float32, device=device), dim=1)
            predictions[start:start + CHUNK] = labels[(batch @ support.T).argmax(1)].cpu().numpy()
    predictions.flush()
    del predictions


def predict_rf(forest, features, output):
    predictions = np.lib.format.open_memmap(output, mode="w+", dtype="uint8",
                                             shape=(len(features),))
    for start in range(0, len(features), CHUNK):
        predictions[start:start + CHUNK] = forest.predict(
            np.asarray(features[start:start + CHUNK])).astype(np.uint8)
    predictions.flush()
    del predictions


def run_home(prepared, output, home, device, threads):
    source_dir = prepared / home
    prepared_hash = verify_prepared(source_dir)
    folder = output / home
    if folder.is_symlink():
        raise ValueError(f"Output must not be a symlink: {folder}")
    config = dict(home=home, prepared_complete_sha256=prepared_hash,
                  source_seed=0, source_epochs=SOURCE_EPOCHS,
                  repair_seeds=list(SEEDS), repair_steps=REPAIR_STEPS,
                  label_budget=BUDGET, query_selector="uniform_without_replacement",
                  references_per_source_class=5, rf=RF,
                  device=str(device), threads=threads,
                  implementation_sha256={name: sha256(path) for name, path in {
                      "predict.py": Path(__file__),
                      "data.py": HERE.parent / "tma_pilot_v1/data.py",
                      "pilot.py": HERE.parent / "tma_pilot_v1/pilot.py",
                      "existing_care.py": HERE.parent / "tma_pilot_v1/existing_care.py",
                      "development.py": HERE.parent / "cps_iot_independent_v1/development.py",
                  }.items()},
                  dev_labels_read=False, final_window_read=False)
    folder.mkdir(parents=True, exist_ok=True)
    config_file = folder / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError(f"Completed configuration changed: {folder}")
    elif any(folder.iterdir()):
        raise ValueError(f"Unexpected partial output: {folder}")
    else:
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    done = folder / "complete.json"
    if done.exists():
        artifacts = json.loads(done.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Completed prediction changed: {folder}")
        print(home, "verified completed predictions reused", flush=True)
        return
    with np.load(source_dir / "source.npz", allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(source_dir / "query.npz", allow_pickle=False) as data:
        raw_query, query_ids = data["x"], data["ids"]
    if len(raw_query) < BUDGET or set(source_y) != set(range(24)):
        raise ValueError(f"Source classes or query pool incomplete: {home}")
    prep = Preprocessor().fit(raw_source)
    source_x = prep.transform(raw_source)
    query_x = prep.transform(raw_query)
    source_file = folder / "source_state.pt"
    source_marker = folder / "source_ready.json"
    if source_marker.is_file():
        if sha256(source_file) != json.loads(source_marker.read_text())["sha256"]:
            raise ValueError(f"Saved source model changed: {folder}")
        source = Model(source_x.shape[1], 24)
        source.load_state_dict(torch.load(source_file, map_location="cpu",
                                          weights_only=True))
        source.eval()
    else:
        if source_file.exists():
            raise ValueError(f"Incomplete source checkpoint: {folder}")
        source = source_model(source_x, source_y, 24, seed=0,
                              epochs=SOURCE_EPOCHS, device=device)
        torch.save(source.state_dict(), source_file)
        source_marker.write_text(json.dumps(dict(sha256=sha256(source_file))) + "\n")
    reference_rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([reference_rng.choice(np.flatnonzero(source_y == c),
                                                    5, replace=False)
                              for c in range(24)])
    reference_features = forward_batches(source.encoder, source_x[ref_ids], device)
    teacher = forward_batches(source.head, reference_features.numpy(), device)
    query_features = forward_batches(source.encoder, query_x, device)
    raw_dev = np.load(source_dir / "development_all_x.npy", mmap_mode="r")
    dev_file = folder / "development_features.npy"
    static_file = folder / "static_mlp.npy"
    dev_marker = folder / "development_ready.json"
    if dev_marker.is_file():
        artifacts = json.loads(dev_marker.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Frozen development features changed: {folder}")
    else:
        if dev_file.exists() or static_file.exists():
            raise ValueError(f"Incomplete development feature cache: {folder}")
        dev_features = np.lib.format.open_memmap(dev_file, mode="w+", dtype="float32",
                                                 shape=(len(raw_dev), 128))
        source.encoder.to(device).eval()
        with torch.no_grad():
            for start in range(0, len(raw_dev), CHUNK):
                x = prep.transform(np.asarray(raw_dev[start:start + CHUNK]))
                dev_features[start:start + CHUNK] = source.encoder(
                    torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
        source.encoder.cpu()
        dev_features.flush()
        del dev_features
        dev_features = np.load(dev_file, mmap_mode="r")
        source.head.to(device).eval()
        predict_head(source.head, dev_features, static_file, device)
        source.head.cpu()
        dev_marker.write_text(json.dumps(dict(artifacts={
            name: sha256(folder / name) for name in
            ("development_features.npy", "static_mlp.npy")}), indent=2) + "\n")
    dev_features = np.load(dev_file, mmap_mode="r")
    query_labels = np.load(source_dir / "query_labels.npy", mmap_mode="r")
    for seed in SEEDS:
        seed_dir = folder / f"seed_{seed}"
        marker = seed_dir / "complete.json"
        if marker.is_file():
            artifacts = json.loads(marker.read_text())["artifacts"]
            if any(sha256(seed_dir / name) != digest for name, digest in artifacts.items()):
                raise ValueError(f"Completed seed prediction changed: {seed_dir}")
            print(home, "seed", seed, "verified reuse", flush=True)
            continue
        if seed_dir.exists() and any(seed_dir.iterdir()):
            raise ValueError(f"Incomplete seed output; use new --output: {seed_dir}")
        seed_dir.mkdir(parents=True)
        chosen = np.random.default_rng(300000 + seed).choice(
            len(raw_query), BUDGET, replace=False)
        paid_y = np.asarray(query_labels[chosen], dtype=np.int64)
        paid_x = query_features[chosen]
        np.save(seed_dir / "paid_query_pool_indices.npy", chosen)
        np.save(seed_dir / "paid_archive_row_ids.npy", query_ids[chosen])
        for name in ("target", "care"):
            head, _ = fit_controlled(
                source.head, paid_x, paid_y, reference_features, source_y[ref_ids],
                teacher, replay_ce=name == "care",
                kd_weight=.5 if name == "care" else 0., temperature=2.,
                target_weight=1., replay_weight=1., steps=REPAIR_STEPS,
                batch_size=128, lr=1e-3, weight_decay=1e-4, seed=seed,
                device=device, log_every=0)
            head.to(device).eval()
            predict_head(head, dev_features, seed_dir / f"{name}.npy", device)
            head.cpu()
            print(home, "seed", seed, name, "complete", flush=True)
        support_x = np.concatenate((paid_x.numpy(), reference_features.numpy()))
        support_y = np.concatenate((paid_y, source_y[ref_ids]))
        forest = RandomForestClassifier(**RF, random_state=seed, n_jobs=threads)
        forest.fit(support_x, support_y)
        predict_rf(forest, dev_features, seed_dir / "frozen_rf.npy")
        predict_nearest(dev_features, support_x, support_y,
                        seed_dir / "nearest_anchor.npy", device)
        print(home, "seed", seed, "RF and 1-NN complete", flush=True)
        artifacts = {p.name: sha256(p) for p in seed_dir.iterdir() if p.is_file()}
        marker.write_text(json.dumps(dict(artifacts=artifacts), indent=2) + "\n")
    artifacts = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    done.write_text(json.dumps(dict(artifacts=artifacts), indent=2) + "\n")
    print(home, "all development predictions frozen", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --device cpu")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    for home in ([args.home] if args.home else PRIMARY):
        run_home(args.prepared.resolve(strict=True), args.output.resolve(),
                 home, torch.device(args.device), args.threads)


if __name__ == "__main__":
    main()
