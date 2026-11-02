"""Fixed-grid head-only BCWI control for chronological target-only updates.

The new head is refitted from exactly the same paid labels and target minibatch
stream as the frozen target-only run. Old/new weight interpolation follows
Schumann et al. (2023), arXiv:2301.10546. This is an established baseline.
Predict freezes all outputs before evaluate reads full development labels.
No final-window label or feature is decoded.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit import sha256
from f1_feasibility import macro_f1
from online_target_update import (read_membership, read_selected_labels,
                                  select_paid)
from online_transport_feasibility import development_days_and_labels
from predict import (BUDGET, PRIMARY, REPAIR_STEPS, predict_head,
                     verify_prepared)
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from existing_care import fit_controlled
from pilot import Model, forward_batches

ALPHAS = ((0.25, "bcwi_025"), (0.50, "bcwi_050"),
          (0.75, "bcwi_075"))
METHODS = ("source", "updated_target", "updated_care") + tuple(
    name for _, name in ALPHAS)


def interpolate_head(old, new, old_weight):
    """Interpolate the shared frozen-encoder linear head in weight space."""
    if not (0 <= old_weight <= 1):
        raise ValueError("old_weight must be in [0, 1]")
    mixed = copy.deepcopy(old)
    old_state, new_state = old.state_dict(), new.state_dict()
    if old_state.keys() != new_state.keys():
        raise ValueError("Old/new head parameters differ")
    with torch.no_grad():
        state = {key: old_weight * old_state[key] +
                 (1 - old_weight) * new_state[key]
                 for key in old_state}
        mixed.load_state_dict(state)
    return mixed.eval()


def checked_target_days(args, archive, windows, members, official, home):
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    days, eligible = read_membership(
        archive, members.loc[home, "member"], home, windows, set(mapping))
    ordered, paid_by_day = select_paid(days, eligible, PRIMARY.index(home),
                                       args.seed)
    target_folder = (args.target_output.resolve(strict=True) /
                     f"seed_{args.seed}" / home)
    target_config = target_folder / "config.json"
    for transition, tomorrow in enumerate(ordered[1:]):
        out = target_folder / f"transition_{transition}"
        marker = json.loads((out / "complete.json").read_text())
        if any(sha256(out / name) != digest
               for name, digest in marker["artifacts"].items()):
            raise ValueError(f"Frozen target-only prediction changed: {out}")
        paid = np.load(verified(out, "paid_dev_indices.npy"))
        ids = np.load(verified(out, "tomorrow_indices.npy"))
        if (not np.array_equal(paid, np.concatenate(paid_by_day[:transition + 1]))
                or not np.array_equal(ids, np.flatnonzero(days == tomorrow))):
            raise ValueError(f"BCWI/target-only pairing differs: {out}")
    return mapping, days, ordered, paid_by_day, target_folder, sha256(target_config)


def check_output_config(folder, config, transitions):
    if folder.is_symlink():
        raise ValueError(f"Output must not be a symlink: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != config:
            change = {key for key in config if previous.get(key) != config[key]}
            completed = any(folder.glob("transition_*/complete.json"))
            if change != {"implementation_sha256"} or completed:
                raise ValueError(f"Prediction configuration changed: {folder}")
            path.write_text(json.dumps(config, indent=2) + "\n")
    elif any(folder.iterdir()):
        raise ValueError(f"Unexpected partial output: {folder}")
    else:
        path.write_text(json.dumps(config, indent=2) + "\n")
    completed = all((folder / f"transition_{t}" / "complete.json").is_file()
                    for t in range(transitions))
    if completed:
        for t in range(transitions):
            out = folder / f"transition_{t}"
            marker = json.loads((out / "complete.json").read_text())
            if any(sha256(out / name) != digest
                   for name, digest in marker["artifacts"].items()):
                raise ValueError(f"Frozen BCWI prediction changed: {out}")
    return completed


def predict_home(args, archive, windows, members, official, home, device):
    (mapping, days, ordered, paid_by_day, target_folder,
     target_config_hash) = checked_target_days(
         args, archive, windows, members, official, home)
    prepared = args.prepared.resolve(strict=True) / home
    baseline = args.baseline.resolve(strict=True) / home
    prepared_hash = verify_prepared(prepared)
    folder = args.output.resolve() / f"seed_{args.seed}" / home
    config = dict(home=home, seed=args.seed, archive_sha256=args.archive_sha256,
                  prepared_marker_sha256=prepared_hash,
                  baseline_marker_sha256=sha256(baseline / "complete.json"),
                  target_config_sha256=target_config_hash,
                  paid_query_labels=BUDGET,
                  paid_development_labels_per_day=BUDGET,
                  repair_steps=REPAIR_STEPS, old_weight_grid=[a for a, _ in ALPHAS],
                  interpolation="alpha*source_head+(1-alpha)*updated_target_head",
                  implementation_sha256=sha256(Path(__file__)),
                  final_window_read=False)
    if check_output_config(folder, config, len(paid_by_day)):
        print(home, "seed", args.seed, "all BCWI outputs verified", flush=True)
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
    query_labels = np.load(verified(prepared, "query_labels.npy"),
                           mmap_mode="r")
    query_y = np.asarray(query_labels[chosen_query], dtype=np.int64)
    prep = Preprocessor().fit(raw_source)
    source = Model(raw_source.shape[1] * 2, 24)
    source.load_state_dict(torch.load(verified(baseline, "source_state.pt"),
                                      map_location="cpu", weights_only=True))
    source.eval()
    rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(24)])
    ref_features = forward_batches(source.encoder,
                                   prep.transform(raw_source[ref_ids]), device)
    teacher = forward_batches(source.head, ref_features.numpy(), device)
    query_features = forward_batches(source.encoder,
                                     prep.transform(raw_query[chosen_query]),
                                     device)
    dev_features = np.load(verified(baseline, "development_features.npy"),
                           mmap_mode="r")
    source_pred = np.load(verified(baseline, "static_mlp.npy"), mmap_mode="r")
    if len(dev_features) != len(days) or len(source_pred) != len(days):
        raise ValueError(f"Development alignment differs: {home}")
    paid_labels = {}
    for transition, (today, tomorrow) in enumerate(zip(
            ordered[:-1], ordered[1:])):
        paid_labels.update(read_selected_labels(
            archive, members.loc[home, "member"], home, windows,
            mapping, set(paid_by_day[transition].tolist())))
        out = folder / f"transition_{transition}"
        marker_path = out / "complete.json"
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            if any(sha256(out / name) != digest
                   for name, digest in marker["artifacts"].items()):
                raise ValueError(f"Frozen BCWI transition changed: {out}")
            print(home, "seed", args.seed, "transition", transition,
                  "verified reuse", flush=True)
            continue
        if out.is_symlink():
            raise ValueError(f"Transition output must not be a symlink: {out}")
        if out.exists() and any(out.iterdir()):
            expected_partial = {"paid_dev_indices.npy", "tomorrow_indices.npy",
                                "alpha_000.npy", "alpha_100.npy"} | {
                                    f"{name}.npy" for _, name in ALPHAS}
            partial = list(out.iterdir())
            if any(p.is_symlink() or not p.is_file() or
                   p.name not in expected_partial for p in partial):
                raise ValueError(f"Unexpected partial BCWI transition: {out}")
            for path in partial:
                path.unlink()
        out.mkdir(parents=True, exist_ok=True)
        paid = np.concatenate(paid_by_day[:transition + 1])
        labels = np.asarray([paid_labels[int(i)] for i in paid], dtype=np.int64)
        features = torch.cat((query_features,
            torch.from_numpy(np.asarray(dev_features[paid], dtype=np.float32))))
        target_y = np.concatenate((query_y, labels))
        updated, trace = fit_controlled(
            source.head, features, target_y,
            ref_features, source_y[ref_ids], teacher,
            replay_ce=False, kd_weight=0., temperature=2.,
            target_weight=1., replay_weight=1., steps=REPAIR_STEPS,
            batch_size=128, lr=1e-3, weight_decay=1e-4,
            seed=args.seed, device=device, log_every=0)
        frozen_target = target_folder / f"transition_{transition}"
        prior_trace = json.loads((frozen_target / "complete.json").read_text())[
            "train_trace"]
        if trace["target_stream_sha256"] != prior_trace["target_stream_sha256"]:
            raise ValueError(f"Target minibatch stream changed: {home} {transition}")
        ids = np.flatnonzero(days == tomorrow)
        np.save(out / "paid_dev_indices.npy", paid)
        np.save(out / "tomorrow_indices.npy", ids)
        updated.to(device).eval()
        predict_head(updated, dev_features[ids], out / "alpha_000.npy", device)
        updated.cpu()
        expected_target = np.load(verified(frozen_target, "updated_target.npy"))
        if not np.array_equal(np.load(out / "alpha_000.npy"), expected_target):
            raise ValueError(f"Refitted target endpoint differs: {home} {transition}")
        source.head.to(device).eval()
        predict_head(source.head, dev_features[ids], out / "alpha_100.npy", device)
        source.head.cpu()
        if not np.array_equal(np.load(out / "alpha_100.npy"), source_pred[ids]):
            raise ValueError(f"Source endpoint differs: {home} {transition}")
        for alpha, name in ALPHAS:
            mixed = interpolate_head(source.head, updated, alpha)
            mixed.to(device).eval()
            predict_head(mixed, dev_features[ids], out / f"{name}.npy", device)
            mixed.cpu()
        artifacts = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
        marker_path.write_text(json.dumps(dict(
            today=today, tomorrow=tomorrow,
            paid_development_labels=len(paid),
            train_trace=trace, artifacts=artifacts), indent=2) + "\n")
        print(home, "seed", args.seed, "transition", transition,
              "BCWI grid frozen", flush=True)


def evaluate_home(args, archive, windows, members, official, home, rows,
                  pooled_rows):
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    days, y = development_days_and_labels(
        archive, members.loc[home, "member"], home, windows, mapping)
    baseline = args.baseline.resolve(strict=True) / home
    bcwi_folder = args.output.resolve(strict=True) / f"seed_{args.seed}" / home
    target_folder = (args.target_output.resolve(strict=True) /
                     f"seed_{args.seed}" / home)
    care_folder = (args.care_output.resolve(strict=True) /
                   f"seed_{args.seed}" / home)
    source = np.load(verified(baseline, "static_mlp.npy"), mmap_mode="r")
    matrices = {method: np.zeros((24, 24), dtype=np.int64)
                for method in METHODS}
    flips = {method: 0 for method in METHODS}
    source_correct_total = 0
    for transition, (today, tomorrow) in enumerate(zip(
            np.unique(days)[:-1], np.unique(days)[1:])):
        out = bcwi_folder / f"transition_{transition}"
        ids = np.load(verified(out, "tomorrow_indices.npy"))
        if not np.array_equal(ids, np.flatnonzero(days == tomorrow)):
            raise ValueError(f"Next-day indices differ: {home} {transition}")
        curated = y[ids] >= 0
        truth = y[ids][curated]
        source_p = np.asarray(source[ids][curated], dtype=np.int64)
        target_p = np.asarray(np.load(verified(
            target_folder / f"transition_{transition}",
            "updated_target.npy"))[curated], dtype=np.int64)
        care_p = np.asarray(np.load(verified(
            care_folder / f"transition_{transition}",
            "updated_care.npy"))[curated], dtype=np.int64)
        predictions = dict(source=source_p, updated_target=target_p,
                           updated_care=care_p)
        for _, name in ALPHAS:
            predictions[name] = np.asarray(
                np.load(verified(out, f"{name}.npy"))[curated],
                dtype=np.int64)
        correct = source_p == truth
        source_correct_total += int(correct.sum())
        base_f1 = macro_f1(truth, source_p, np.arange(24))
        for method, pred in predictions.items():
            if len(pred) != len(truth) or np.any((pred < 0) | (pred >= 24)):
                raise ValueError(f"Prediction shape/class invalid: {home} {method}")
            matrices[method] += np.bincount(
                truth * 24 + pred, minlength=24 * 24).reshape(24, 24)
            flips[method] += int(np.sum(pred[correct] != truth[correct]))
            rows.append(dict(home=home, seed=args.seed, today=today,
                             tomorrow=tomorrow, method=method,
                             curated_rows=len(truth),
                             macro_f1_fixed24=macro_f1(
                                 truth, pred, np.arange(24)),
                             delta_vs_source_pp=100 * (
                                 macro_f1(truth, pred, np.arange(24)) - base_f1),
                             negative_flip_rate_vs_source=float(
                                 (pred[correct] != truth[correct]).mean())
                                 if correct.any() else None))
    for method in METHODS:
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
                flips[method] / source_correct_total
                if source_correct_total else None)))
    print(home, "seed", args.seed, "BCWI development evaluated", flush=True)


def evaluate(args, archive, windows, members, official, homes):
    # Freeze every requested prediction file before opening full labels.
    for home in homes:
        folder = args.output.resolve(strict=True) / f"seed_{args.seed}" / home
        expected = int(windows.loc[(home, "development"), "observed_days"]) - 1
        for transition in range(expected):
            out = folder / f"transition_{transition}"
            marker = json.loads((out / "complete.json").read_text())
            if any(sha256(out / name) != digest
                   for name, digest in marker["artifacts"].items()):
                raise ValueError(f"Frozen BCWI prediction changed: {out}")
    rows, pooled_rows = [], []
    for home in homes:
        evaluate_home(args, archive, windows, members, official, home,
                      rows, pooled_rows)
    pooled = pd.DataFrame(pooled_rows)
    if args.home is None:
        for previous_root, method in ((args.target_output, "updated_target"),
                                      (args.care_output, "updated_care")):
            previous = pd.read_csv(previous_root.resolve(strict=True) /
                                   f"seed_{args.seed}" / "evaluation_all" /
                                   "pooled_by_home.csv")
            old = previous.loc[previous.method == method].set_index("home")
            new = pooled.loc[pooled.method == method].set_index("home")
            if set(old.index) != set(new.index):
                raise ValueError(f"Previous {method} homes differ")
            for column in ("pooled_curated_rows", "pooled_present_classes",
                           "pooled_macro_f1_fixed24",
                           "pooled_negative_flip_rate_vs_source"):
                if not np.allclose(old.loc[new.index, column], new[column],
                                   rtol=0, atol=1e-10):
                    raise ValueError(f"Recomputed {method} differs: {column}")
    report_name = f"evaluation_{args.home}" if args.home else "evaluation_all"
    output = args.output.resolve() / f"seed_{args.seed}" / report_name
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Evaluation output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_day.csv", index=False)
    pooled.to_csv(output / "pooled_by_home.csv", index=False)
    means = pooled.groupby("method")[["pooled_macro_f1_fixed24",
                                       "pooled_negative_flip_rate_vs_source"]].mean()
    report = "\n".join([
        "Exploratory fixed-grid head-only BCWI control on examined UNSW development days.",
        f"Seed={args.seed}; old-weight alpha=[0.25, 0.50, 0.75].",
        "Alpha 0 and 1 exactly reproduce frozen target-only and source predictions.",
        "Same paid labels and target minibatch stream as the prior target-only run.",
        "Negative-flip rate divides by source-correct rows, unlike the original BCWI paper's all-row denominator.",
        "No alpha is selected using development outcomes; all grid points are reported.",
        "This is a published-method baseline; no final-window label or feature decoded.",
        "Across-home pooled means:", means.round(5).to_string(),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


def aggregate(root):
    parts = []
    for seed in range(5):
        path = root / f"seed_{seed}" / "evaluation_all" / "pooled_by_home.csv"
        part = pd.read_csv(path)
        if (len(part) != len(PRIMARY) * len(METHODS) or
                set(part.home) != set(PRIMARY) or
                set(part.method) != set(METHODS) or
                set(part.seed) != {seed} or
                part.duplicated(["home", "method"]).any()):
            raise ValueError(f"Completed evaluation layout differs: {path}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    f1 = frame.pivot(index=["home", "seed"], columns="method",
                     values="pooled_macro_f1_fixed24")
    nfr = frame.pivot(index=["home", "seed"], columns="method",
                      values="pooled_negative_flip_rate_vs_source")
    if not np.allclose(nfr.source, 0, rtol=0, atol=1e-10):
        raise ValueError("Source negative-flip rate must be zero")
    if not np.isfinite(f1.to_numpy()).all() or not np.isfinite(nfr.to_numpy()).all():
        raise ValueError("Non-finite BCWI metric")
    by_home = frame.groupby(["home", "method"], as_index=False)[[
        "pooled_macro_f1_fixed24",
        "pooled_negative_flip_rate_vs_source"]].mean()
    means = by_home.groupby("method")[[
        "pooled_macro_f1_fixed24",
        "pooled_negative_flip_rate_vs_source"]].mean()
    f1_home = by_home.pivot(index="home", columns="method",
                            values="pooled_macro_f1_fixed24")
    nfr_home = by_home.pivot(index="home", columns="method",
                             values="pooled_negative_flip_rate_vs_source")
    output = root / "aggregate_0_1_2_3_4"
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Aggregate output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Five-seed exploratory fixed-grid head-only BCWI baseline, 11 independent homes.",
        "All alpha values were fixed before this run and every grid point is reported.",
        "Five seeds within each home overlap; development already examined.",
        "F1 is fixed-24-class macro F1, pooled over next-day rows within each home.",
        "Negative-flip rate is conditional on source-correct rows.",
        "No final-window label or feature decoded. This is an existing method, not a new one.",
        "Across-home means:", means.round(5).to_string(),
        "Per-home macro F1:", f1_home.round(5).to_string(),
        "Per-home conditional negative-flip rate:",
        nfr_home.round(5).to_string(),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "evaluate", "aggregate"))
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--target-output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_target_update_v1"))
    parser.add_argument("--care-output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_care_update_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_v1"))
    args = parser.parse_args()
    if args.stage == "aggregate":
        if args.archive is not None:
            parser.error("aggregate does not take an archive")
        aggregate(args.output.resolve(strict=True))
        return
    if args.archive is None or args.threads < 1:
        parser.error("predict/evaluate require an archive and positive threads")
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
        evaluate(args, archive, windows, members, official, homes)


if __name__ == "__main__":
    main()
