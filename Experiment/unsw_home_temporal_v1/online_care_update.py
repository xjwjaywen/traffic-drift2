"""Exploratory chronological CARE-head update with 1000 paid labels per day.

Predict stage selects official-inventory rows using MAC membership, reads only
their paid class labels, and freezes next-day predictions before evaluation.
The source encoder and historical references remain fixed. A fresh CARE head
is fitted from the source head on the original 1000 paid query rows plus all
paid development rows through the previous day. This is a standard update
control, not a new method. The final window is never decoded.
"""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

from audit import sha256
from f1_feasibility import macro_f1
from online_transport_feasibility import development_days_and_labels
from predict import (BUDGET, CHUNK, PRIMARY, REPAIR_STEPS, predict_head,
                     verify_prepared)
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from existing_care import fit_controlled
from pilot import Model, forward_batches


def read_membership(archive, member, home, windows, official_mac):
    after_query = str(windows.loc[(home, "query"), "last_day"])
    after_dev = str(windows.loc[(home, "development"), "last_day"])
    expected = int(windows.loc[(home, "development"), "rows"])
    days, eligible = [], []
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {home}")
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= after_query or day > after_dev:
                continue
            mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
            days.append(day)
            eligible.append(mac in official_mac)
    if len(days) != expected:
        raise ValueError(f"Development row count differs: {home}")
    return np.asarray(days), np.asarray(eligible, dtype=bool)


def read_selected_labels(archive, member, home, windows, mapping, wanted):
    after_query = str(windows.loc[(home, "query"), "last_day"])
    after_dev = str(windows.loc[(home, "development"), "last_day"])
    labels = {}
    pos = 0
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError(f"Unexpected archive schema: {home}")
        for line in stream:
            day = line[:10].decode("ascii")
            if day <= after_query or day > after_dev:
                continue
            if pos in wanted:
                mac = line.rsplit(b",", 1)[-1].strip().decode("ascii")
                label = mapping.get(mac)
                if label is None:
                    raise ValueError(f"Paid row is outside official inventory: {home} {pos}")
                labels[pos] = label
            pos += 1
    if set(labels) != wanted:
        raise ValueError(f"Selected development labels missing: {home}")
    return labels


def select_paid(days, eligible, home_index, seed):
    ordered = np.unique(days)
    chosen = []
    for transition, today in enumerate(ordered[:-1]):
        pool = np.flatnonzero((days == today) & eligible)
        if len(pool) < BUDGET:
            raise ValueError(f"Too few curated daily rows: {today}")
        rng = np.random.default_rng(990000 + 1000 * home_index +
                                    100 * seed + transition)
        chosen.append(np.sort(rng.choice(pool, BUDGET, replace=False)))
    return ordered, chosen


def predict_home(args, archive, windows, members, official, home, device):
    home_index = PRIMARY.index(home)
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    days, eligible = read_membership(
        archive, members.loc[home, "member"], home, windows, set(mapping))
    ordered, paid_by_day = select_paid(days, eligible, home_index, args.seed)
    paid_labels = {}
    prepared = args.prepared.resolve(strict=True) / home
    baseline = args.baseline.resolve(strict=True) / home
    prepared_hash = verify_prepared(prepared)
    base_marker = sha256(baseline / "complete.json")
    folder = args.output.resolve() / f"seed_{args.seed}" / home
    if folder.is_symlink():
        raise ValueError(f"Output must not be a symlink: {folder}")
    config = dict(home=home, seed=args.seed,
                  archive_sha256=args.archive_sha256,
                  prepared_marker_sha256=prepared_hash,
                  baseline_marker_sha256=base_marker,
                  paid_query_labels=BUDGET, paid_development_labels_per_day=BUDGET,
                  repair_steps=REPAIR_STEPS, source_references_per_class=5,
                  source_encoder_frozen=True,
                  implementation_sha256=sha256(Path(__file__)),
                  final_window_read=False)
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Prediction configuration changed: {folder}")
    elif any(folder.iterdir()):
        raise ValueError(f"Unexpected partial output: {folder}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    if all((folder / f"transition_{t}" / "complete.json").is_file()
           for t in range(len(paid_by_day))):
        for t in range(len(paid_by_day)):
            marker = json.loads((folder / f"transition_{t}" / "complete.json").read_text())
            if any(sha256(folder / f"transition_{t}" / name) != digest
                   for name, digest in marker["artifacts"].items()):
                raise ValueError(f"Frozen update prediction changed: {home} {t}")
        print(home, "all frozen updates verified", flush=True)
        return
    with np.load(verified(prepared, "source.npz"), allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(verified(prepared, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    chosen_query = np.load(verified(baseline / f"seed_{args.seed}",
                                    "paid_query_pool_indices.npy"))
    if (chosen_query.shape != (BUDGET,) or
            len(np.unique(chosen_query)) != BUDGET or
            chosen_query.min() < 0 or chosen_query.max() >= len(raw_query)):
        raise ValueError(f"Original paid query indices invalid: {home}")
    query_y_all = np.load(verified(prepared, "query_labels.npy"), mmap_mode="r")
    query_y = np.asarray(query_y_all[chosen_query], dtype=np.int64)
    prep = Preprocessor().fit(raw_source)
    source = Model(raw_source.shape[1] * 2, 24)
    source.load_state_dict(torch.load(verified(baseline, "source_state.pt"),
                                      map_location="cpu", weights_only=True))
    source.eval()
    ref_rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        ref_rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(24)])
    ref_features = forward_batches(source.encoder,
                                   prep.transform(raw_source[ref_ids]), device)
    teacher = forward_batches(source.head, ref_features.numpy(), device)
    query_features = forward_batches(source.encoder,
                                     prep.transform(raw_query[chosen_query]), device)
    dev_features = np.load(verified(baseline, "development_features.npy"),
                           mmap_mode="r")
    if len(dev_features) != len(days):
        raise ValueError(f"Development feature alignment differs: {home}")
    for transition, (today, tomorrow) in enumerate(zip(
            ordered[:-1], ordered[1:])):
        # Reveal only this day's selected labels. Later paid labels are not
        # decoded until their own transition is reached.
        paid_labels.update(read_selected_labels(
            archive, members.loc[home, "member"], home, windows,
            mapping, set(paid_by_day[transition].tolist())))
        out = folder / f"transition_{transition}"
        marker_path = out / "complete.json"
        if marker_path.is_file():
            artifacts = json.loads(marker_path.read_text())["artifacts"]
            if any(sha256(out / name) != digest for name, digest in artifacts.items()):
                raise ValueError(f"Completed transition changed: {out}")
            print(home, "seed", args.seed, "transition", transition,
                  "verified reuse", flush=True)
            continue
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Incomplete transition; use new output: {out}")
        out.mkdir(parents=True, exist_ok=True)
        paid_so_far = np.concatenate(paid_by_day[:transition + 1])
        if not np.all(days[paid_so_far] <= today):
            raise AssertionError("Future paid row entered training")
        labels = np.asarray([paid_labels[int(i)] for i in paid_so_far],
                            dtype=np.int64)
        target_features = torch.cat((
            query_features,
            torch.from_numpy(np.asarray(dev_features[paid_so_far],
                                        dtype=np.float32))))
        target_y = np.concatenate((query_y, labels))
        head, trace = fit_controlled(
            source.head, target_features, target_y,
            ref_features, source_y[ref_ids], teacher,
            replay_ce=True, kd_weight=.5, temperature=2.,
            target_weight=1., replay_weight=1., steps=REPAIR_STEPS,
            batch_size=128, lr=1e-3, weight_decay=1e-4,
            seed=args.seed, device=device, log_every=0)
        tomorrow_ids = np.flatnonzero(days == tomorrow)
        np.save(out / "paid_dev_indices.npy", paid_so_far)
        np.save(out / "tomorrow_indices.npy", tomorrow_ids)
        predict_head(head, dev_features[tomorrow_ids],
                     out / "updated_care.npy", device)
        artifacts = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
        marker_path.write_text(json.dumps(dict(
            today=today, tomorrow=tomorrow,
            paid_development_labels=len(paid_so_far),
            train_trace=trace, artifacts=artifacts), indent=2) + "\n")
        print(home, "seed", args.seed, "transition", transition,
              "updated CARE frozen", flush=True)


def evaluate_home(args, archive, windows, members, official, home, rows,
                  pooled_rows):
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    days, y = development_days_and_labels(
        archive, members.loc[home, "member"], home, windows, mapping)
    baseline = args.baseline.resolve(strict=True) / home
    folder = args.output.resolve(strict=True) / f"seed_{args.seed}" / home
    source = np.load(verified(baseline, "static_mlp.npy"), mmap_mode="r")
    fixed = np.load(verified(baseline / f"seed_{args.seed}", "care.npy"),
                    mmap_mode="r")
    if len(source) != len(y) or len(fixed) != len(y):
        raise ValueError(f"Baseline prediction alignment differs: {home}")
    ordered = np.unique(days)
    names = ("source", "fixed_care", "updated_care")
    matrices = {name: np.zeros((24, 24), dtype=np.int64) for name in names}
    flip_counts = {name: 0 for name in names}
    source_correct_total = 0
    for transition, (today, tomorrow) in enumerate(zip(
            ordered[:-1], ordered[1:])):
        out = folder / f"transition_{transition}"
        marker = json.loads((out / "complete.json").read_text())
        ids = np.load(verified(out, "tomorrow_indices.npy"))
        updated = np.load(verified(out, "updated_care.npy"))
        paid = np.load(verified(out, "paid_dev_indices.npy"))
        if not np.array_equal(ids, np.flatnonzero(days == tomorrow)):
            raise ValueError(f"Next-day indices differ: {home} {transition}")
        if len(updated) != len(ids) or len(paid) != BUDGET * (transition+1):
            raise ValueError(f"Prediction or paid-label length differs: {home} {transition}")
        if len(np.unique(paid)) != len(paid) or not np.all(days[paid] <= today):
            raise ValueError(f"Paid-label chronology differs: {home} {transition}")
        curated = y[ids] >= 0
        truth = y[ids][curated]
        base_pred = np.asarray(source[ids][curated], dtype=np.int64)
        fixed_pred = np.asarray(fixed[ids][curated], dtype=np.int64)
        updated_pred = np.asarray(updated[curated], dtype=np.int64)
        static_f1 = macro_f1(truth, base_pred, np.arange(24))
        source_correct = base_pred == truth
        source_correct_total += int(source_correct.sum())
        for method, pred in (("source", base_pred),
                             ("fixed_care", fixed_pred),
                             ("updated_care", updated_pred)):
            matrices[method] += np.bincount(
                truth * 24 + pred, minlength=24 * 24).reshape(24, 24)
            flip_counts[method] += int(np.sum(
                pred[source_correct] != truth[source_correct]))
            rows.append(dict(
                home=home, seed=args.seed, today=today, tomorrow=tomorrow,
                method=method, curated_rows=len(truth),
                present_classes=len(np.unique(truth)),
                paid_query_labels=BUDGET,
                paid_development_labels=(transition+1)*BUDGET
                    if method == "updated_care" else 0,
                macro_f1_fixed24=macro_f1(truth, pred, np.arange(24)),
                macro_delta_vs_source_pp=100 * (
                    macro_f1(truth, pred, np.arange(24)) - static_f1),
                negative_flip_rate_vs_source=float(
                    (pred[source_correct] != truth[source_correct]).mean())
                    if source_correct.any() else None))
    for method in names:
        matrix = matrices[method]
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        tp = np.diag(matrix)
        f1 = np.divide(2 * tp, support + predicted,
                       out=np.zeros(24, dtype=float),
                       where=(support + predicted) > 0)
        pooled_rows.append(dict(
            home=home, seed=args.seed, method=method,
            pooled_curated_rows=int(support.sum()),
            pooled_present_classes=int(np.sum(support > 0)),
            pooled_macro_f1_fixed24=float(f1.mean()),
            pooled_macro_f1_present=float(f1[support > 0].mean()),
            pooled_negative_flip_rate_vs_source=(
                flip_counts[method] / source_correct_total
                if source_correct_total else None)))
    print(home, "seed", args.seed, "development updates evaluated", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "evaluate"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_care_update_v1"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    args.archive_sha256 = expected
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    homes = [args.home] if args.home else PRIMARY
    if args.stage == "predict":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA unavailable; select --device cpu")
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        for home in homes:
            predict_home(args, archive, windows, members, official, home, device)
    else:
        # Freeze and verify every requested prediction before reading full labels.
        for home in homes:
            folder = args.output.resolve(strict=True) / f"seed_{args.seed}" / home
            expected_transitions = int(windows.loc[(home, "development"),
                                                    "observed_days"]) - 1
            for t in range(expected_transitions):
                out = folder / f"transition_{t}"
                marker = json.loads((out / "complete.json").read_text())
                if any(sha256(out / name) != digest
                       for name, digest in marker["artifacts"].items()):
                    raise ValueError(f"Frozen prediction changed: {out}")
        rows, pooled_rows = [], []
        for home in homes:
            evaluate_home(args, archive, windows, members, official, home,
                          rows, pooled_rows)
        frame = pd.DataFrame(rows)
        pooled = pd.DataFrame(pooled_rows)
        source_pooled = (pooled.loc[pooled.method == "source"]
                         .set_index("home").pooled_macro_f1_fixed24)
        pooled["pooled_delta_vs_source_pp"] = 100 * (
            pooled.pooled_macro_f1_fixed24 - pooled.home.map(source_pooled))
        report_name = f"evaluation_{args.home}" if args.home else "evaluation_all"
        report_root = args.output.resolve() / f"seed_{args.seed}" / report_name
        if report_root.exists() and any(report_root.iterdir()):
            parser.error(f"Evaluation output must be new and empty: {report_root}")
        report_root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(report_root / "per_day.csv", index=False)
        pooled.to_csv(report_root / "pooled_by_home.csv", index=False)
        by_home = (frame.groupby(["home", "method"], as_index=False)
                   [["macro_f1_fixed24", "macro_delta_vs_source_pp",
                     "negative_flip_rate_vs_source"]].mean())
        by_home.to_csv(report_root / "per_home.csv", index=False)
        means = by_home.groupby("method")[["macro_f1_fixed24",
                                          "macro_delta_vs_source_pp",
                                          "negative_flip_rate_vs_source"]].mean()
        pooled_means = pooled.groupby("method")[[
            "pooled_macro_f1_fixed24", "pooled_macro_f1_present",
            "pooled_delta_vs_source_pp",
            "pooled_negative_flip_rate_vs_source"]].mean()
        report = "\n".join([
            "Exploratory chronological CARE-head update; source encoder and historical references fixed.",
            "Seed=" + str(args.seed) + "; paid original query labels=1000; extra paid development labels=1000 per prior day.",
            "Each updated head starts from the frozen source head and uses only selected labels through yesterday.",
            "All next-day predictions were frozen and verified before full development labels were scored.",
            "This is an ordinary CARE retraining control, not a new method or independent validation.",
            "Final-window labels and features were not decoded.",
            "Across-home pooled-day means (primary):",
            pooled_means.round(5).to_string(),
            "Per-home pooled-day metrics:", pooled.round(5).to_string(index=False),
            "Across-home mean of daily metrics (diagnostic):",
            means.round(5).to_string(),
        ]) + "\n"
        (report_root / "REPORT.txt").write_text(report)
        print(report, flush=True)


if __name__ == "__main__":
    main()
