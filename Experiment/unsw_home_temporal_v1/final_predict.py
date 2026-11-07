"""Freeze original one-shot heads' predictions on the unopened final windows.

Final device/MAC values are not decoded. Every refitted target/CARE head must
reproduce its entire previously frozen development prediction exactly.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
from audit import FEATURES, sha256
from predict import (BUDGET, CHUNK, PRIMARY, REPAIR_STEPS, SEEDS,
                     predict_head, verify_prepared)

sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from existing_care import fit_controlled
from pilot import Model, forward_batches


def verify_old_home(folder, home, windows):
    marker = folder / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Frozen development predictions missing: {home}")
    config = json.loads((folder / "config.json").read_text())
    if (config["home"] != home or config["repair_seeds"] != list(SEEDS)
            or config["label_budget"] != BUDGET
            or config["repair_steps"] != REPAIR_STEPS):
        raise ValueError(f"Original prediction protocol differs: {home}")
    artifacts = json.loads(marker.read_text())["artifacts"]
    if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
        raise ValueError(f"Original home artifact changed: {home}")
    n = int(windows.loc[(home, "development"), "rows"])
    for seed in SEEDS:
        sub = folder / f"seed_{seed}"
        sub_marker = sub / "complete.json"
        if not sub_marker.is_file():
            raise ValueError(f"Frozen development seed missing: {home} {seed}")
        sub_artifacts = json.loads(sub_marker.read_text())["artifacts"]
        if any(sha256(sub / name) != digest
               for name, digest in sub_artifacts.items()):
            raise ValueError(f"Original seed artifact changed: {home} {seed}")
        for method in ("target", "care"):
            pred = np.load(sub / f"{method}.npy", mmap_mode="r")
            if pred.shape != (n,) or pred.dtype != np.uint8:
                raise ValueError(f"Invalid old prediction: {home} {seed} {method}")
    return sha256(marker)


def verify_reproduction(head, features, frozen_path, device):
    expected = np.load(frozen_path, mmap_mode="r", allow_pickle=False)
    head.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(features), CHUNK):
            batch = torch.as_tensor(np.array(features[start:start + CHUNK], copy=True),
                                    dtype=torch.float32, device=device)
            actual = head(batch).argmax(1).cpu().numpy()
            if not np.array_equal(actual, expected[start:start + len(actual)]):
                raise ValueError(f"Head cannot reproduce {frozen_path} at row {start}")
    head.cpu()


def final_features(archive, member, after_dev, last_final, expected_rows,
                   prep, encoder, source_head, folder, device):
    features = np.lib.format.open_memmap(folder / "final_features.npy", mode="w+",
                                         dtype="float32", shape=(expected_rows, 128))
    static = np.lib.format.open_memmap(folder / "static_mlp.npy", mode="w+",
                                       dtype="uint8", shape=(expected_rows,))
    encoder.to(device).eval()
    source_head.to(device).eval()
    raw, pos = [], 0

    def flush():
        nonlocal pos
        if not raw:
            return
        x = prep.transform(np.asarray(raw, dtype=np.float32))
        with torch.no_grad():
            z = encoder(torch.as_tensor(x, dtype=torch.float32, device=device))
            pred = source_head(z).argmax(1)
        end = pos + len(raw)
        if end > expected_rows:
            raise ValueError("More final rows than timestamp manifest")
        features[pos:end] = z.cpu().numpy()
        static[pos:end] = pred.cpu().numpy().astype(np.uint8)
        pos = end
        raw.clear()

    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {member}")
        indices = [header.index(name) for name in FEATURES]
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= after_dev:
                continue
            if day > last_final:
                raise ValueError(f"Row outside frozen final window: {member} {day}")
            values = line.rstrip(b"\r\n").split(b",")
            if len(values) != len(header):
                raise ValueError(f"Malformed final row: {member}")
            # The last `device` field is never read or mapped to a label here.
            x = [float(values[i]) if values[i] else np.nan for i in indices]
            raw.append(x)
            if len(raw) == CHUNK:
                flush()
    flush()
    encoder.cpu()
    source_head.cpu()
    if pos != expected_rows:
        raise ValueError(f"Final row count {pos} differs from {expected_rows}")
    features.flush()
    static.flush()
    del features, static


def run_home(archive, prepared, original, output, windows, members,
             home, device):
    source_dir = prepared / home
    prepared_hash = verify_prepared(source_dir)
    old_dir = original / home
    original_hash = verify_old_home(old_dir, home, windows)
    folder = output / home
    if folder.is_symlink():
        raise ValueError(f"Output must not be a symlink: {folder}")
    marker = folder / "complete.json"
    if marker.is_file():
        config = json.loads((folder / "config.json").read_text())
        if (config["prepared_marker_sha256"] != prepared_hash or
                config["original_marker_sha256"] != original_hash or
                config["implementation_sha256"] != sha256(Path(__file__)) or
                config["protocol_sha256"] !=
                    sha256(HERE / "FINAL_HOLDOUT_PROTOCOL.md")):
            raise ValueError(f"Completed final prediction protocol changed: {home}")
        artifacts = json.loads(marker.read_text())["artifacts"]
        if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
            raise ValueError(f"Completed final predictions changed: {home}")
        print(home, "verified final predictions reused", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete final output; remove only this folder: {folder}")

    with np.load(source_dir / "source.npz", allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(source_dir / "query.npz", allow_pickle=False) as data:
        raw_query, query_ids = data["x"], data["ids"]
    query_labels = np.load(source_dir / "query_labels.npy", mmap_mode="r")
    prep = Preprocessor().fit(raw_source)
    source_x = prep.transform(raw_source)
    query_x = prep.transform(raw_query)
    source = Model(source_x.shape[1], 24)
    source_file = old_dir / "source_state.pt"
    saved_hash = json.loads((old_dir / "source_ready.json").read_text())["sha256"]
    if sha256(source_file) != saved_hash:
        raise ValueError(f"Frozen source checkpoint changed: {home}")
    source.load_state_dict(torch.load(source_file, map_location="cpu",
                                      weights_only=True))
    source.eval()
    ref_rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([ref_rng.choice(np.flatnonzero(source_y == c),
                                              5, replace=False) for c in range(24)])
    ref_features = forward_batches(source.encoder, source_x[ref_ids], device)
    teacher = forward_batches(source.head, ref_features.numpy(), device)
    query_features = forward_batches(source.encoder, query_x, device)
    dev_features = np.load(old_dir / "development_features.npy", mmap_mode="r")

    heads = {}
    for seed in SEEDS:
        old_seed = old_dir / f"seed_{seed}"
        chosen = np.random.default_rng(300000 + seed).choice(
            len(raw_query), BUDGET, replace=False)
        if not np.array_equal(chosen, np.load(old_seed / "paid_query_pool_indices.npy")):
            raise ValueError(f"Paid query identities differ: {home} {seed}")
        if not np.array_equal(query_ids[chosen], np.load(old_seed / "paid_archive_row_ids.npy")):
            raise ValueError(f"Paid archive row IDs differ: {home} {seed}")
        paid_y = np.asarray(query_labels[chosen], dtype=np.int64)
        for method in ("target", "care"):
            head, _ = fit_controlled(
                source.head, query_features[chosen], paid_y,
                ref_features, source_y[ref_ids], teacher,
                replay_ce=method == "care", kd_weight=.5 if method == "care" else 0.,
                temperature=2., target_weight=1., replay_weight=1.,
                steps=REPAIR_STEPS, batch_size=128, lr=1e-3,
                weight_decay=1e-4, seed=seed, device=device, log_every=0)
            verify_reproduction(head, dev_features, old_seed / f"{method}.npy", device)
            heads[(seed, method)] = head
            print(home, "seed", seed, method, "development reproduction exact", flush=True)

    folder.mkdir(parents=True, exist_ok=True)
    config = dict(home=home, archive_sha256=sha256(archive),
                  prepared_marker_sha256=prepared_hash,
                  original_marker_sha256=original_hash,
                  source_checkpoint_sha256=saved_hash,
                  implementation_sha256=sha256(Path(__file__)),
                  protocol_sha256=sha256(HERE / "FINAL_HOLDOUT_PROTOCOL.md"),
                  label_budget=BUDGET, repair_seeds=list(SEEDS),
                  methods=["static_mlp", "target", "care", "bcwi_050"],
                  old_weight=0.5, final_labels_read=False)
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    final_features(
        archive, members.loc[home, "member"],
        str(windows.loc[(home, "development"), "last_day"]),
        str(windows.loc[(home, "final"), "last_day"]),
        int(windows.loc[(home, "final"), "rows"]),
        prep, source.encoder, source.head, folder, device)
    features = np.load(folder / "final_features.npy", mmap_mode="r")
    for seed in SEEDS:
        sub = folder / f"seed_{seed}"
        sub.mkdir()
        for method in ("target", "care"):
            head = heads[(seed, method)]
            head.to(device).eval()
            predict_head(head, features, sub / f"{method}.npy", device)
            head.cpu()
        blend = copy.deepcopy(source.head)
        with torch.no_grad():
            for old, new in zip(blend.parameters(), heads[(seed, "target")].parameters()):
                old.mul_(0.5).add_(new, alpha=0.5)
        blend.to(device).eval()
        predict_head(blend, features, sub / "bcwi_050.npy", device)
        blend.cpu()
        print(home, "seed", seed, "final predictions frozen", flush=True)
    artifacts = {p.relative_to(folder).as_posix(): sha256(p)
                 for p in folder.rglob("*") if p.is_file() and p.name != "complete.json"}
    marker.write_text(json.dumps({"artifacts": artifacts}, indent=2) + "\n")
    print(home, "final prediction complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--original", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/predictions"))
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --device cpu")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from frozen timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for home in ([args.home] if args.home else PRIMARY):
        run_home(archive, args.prepared.resolve(strict=True),
                 args.original.resolve(strict=True), output, windows,
                 members, home, torch.device(args.device))


if __name__ == "__main__":
    main()
