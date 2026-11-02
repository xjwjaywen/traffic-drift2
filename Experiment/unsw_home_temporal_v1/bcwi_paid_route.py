"""Exploratory paid-label next-day fallback for frozen BCWI 0.50 predictions.

At the first predicted development day, use BCWI 0.50. Thereafter choose
BCWI 0.50 for tomorrow iff its fixed-24 macro F1 exceeds source on today's
already paid 1000 labels; otherwise use source. Today's models were trained
only through yesterday. The same paid labels already train tomorrow's model,
so this rule buys no additional labels. It is a simple control, not novelty.

The prepare stage stores decisions before evaluate opens full development
labels. No final-window label or feature is decoded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from f1_feasibility import macro_f1
from online_target_update import (read_membership, read_selected_labels,
                                  select_paid)
from online_transport_feasibility import development_days_and_labels
from predict import BUDGET, PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
CLASSES = np.arange(24)
METHODS = ("source", "bcwi_050", "updated_care", "paid_sign_route")


def checked_days(args, archive, windows, members, official, home, seed):
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    days, eligible = read_membership(
        archive, members.loc[home, "member"], home, windows, set(mapping))
    ordered, paid_by_day = select_paid(days, eligible, PRIMARY.index(home), seed)
    bcwi_folder = args.bcwi.resolve(strict=True) / f"seed_{seed}" / home
    for t, tomorrow in enumerate(ordered[1:]):
        out = bcwi_folder / f"transition_{t}"
        marker = json.loads((out / "complete.json").read_text())
        if any(sha256(out / name) != digest
               for name, digest in marker["artifacts"].items()):
            raise ValueError(f"Frozen BCWI prediction changed: {out}")
        if not np.array_equal(np.load(verified(out, "paid_dev_indices.npy")),
                              np.concatenate(paid_by_day[:t + 1])):
            raise ValueError(f"Paid identities differ: {out}")
        if not np.array_equal(np.load(verified(out, "tomorrow_indices.npy")),
                              np.flatnonzero(days == tomorrow)):
            raise ValueError(f"Next-day indices differ: {out}")
    return mapping, days, ordered, paid_by_day, bcwi_folder


def prepare_home(args, archive, windows, members, official, home, seed):
    (mapping, days, ordered, paid_by_day,
     bcwi_folder) = checked_days(
         args, archive, windows, members, official, home, seed)
    baseline = args.baseline.resolve(strict=True) / home
    source = np.load(verified(baseline, "static_mlp.npy"), mmap_mode="r")
    if len(source) != len(days):
        raise ValueError(f"Source prediction alignment differs: {home}")
    out = args.output.resolve() / f"seed_{seed}" / home
    if out.is_symlink():
        raise ValueError(f"Gate output must not be a symlink: {out}")
    config = dict(home=home, seed=seed, archive_sha256=args.archive_sha256,
                  bcwi_config_sha256=sha256(bcwi_folder / "config.json"),
                  baseline_marker_sha256=sha256(baseline / "complete.json"),
                  rule="first=bcwi_050; later=bcwi_050 iff previous-day paid fixed24 macro F1 exceeds source; tie=source",
                  paid_labels_per_elapsed_day=BUDGET,
                  implementation_sha256=sha256(Path(__file__)),
                  final_window_labels_or_features_read=False)
    marker_path = out / "complete.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text())
        if marker["config"] != config or sha256(out / "decisions.csv") != marker[
                "decisions_sha256"]:
            raise ValueError(f"Frozen gate changed: {out}")
        print(home, "seed", seed, "gate verified reuse", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Incomplete gate output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Decode only labels whose identities were already paid in the target-only
    # update. Each decision below accesses only its own previous-day slice.
    wanted = set(np.concatenate(paid_by_day).tolist())
    paid_labels = read_selected_labels(
        archive, members.loc[home, "member"], home, windows, mapping, wanted)
    rows = []
    for t, (today, tomorrow) in enumerate(zip(ordered[:-1], ordered[1:])):
        if t == 0:
            choice, delta = "bcwi_050", None
            audit_day, audit_classes = None, 0
        else:
            audit_day = today
            paid = paid_by_day[t]
            if not np.all(days[paid] == today):
                raise ValueError(f"Future paid label entered gate: {home} {t}")
            prior = bcwi_folder / f"transition_{t - 1}"
            prior_ids = np.load(verified(prior, "tomorrow_indices.npy"))
            local = np.searchsorted(prior_ids, paid)
            if np.any(local >= len(prior_ids)) or not np.array_equal(
                    prior_ids[local], paid):
                raise ValueError(f"Paid audit not in frozen prior predictions: {home} {t}")
            bcwi = np.load(verified(prior, "bcwi_050.npy"), mmap_mode="r")
            truth = np.asarray([paid_labels[int(i)] for i in paid],
                               dtype=np.int64)
            audit_classes = len(np.unique(truth))
            delta = 100 * (
                macro_f1(truth, np.asarray(bcwi[local], dtype=np.int64), CLASSES) -
                macro_f1(truth, np.asarray(source[paid], dtype=np.int64), CLASSES))
            choice = "bcwi_050" if delta > 0 else "source"
        rows.append(dict(home=home, seed=seed, transition=t,
                         audit_day=audit_day, today=today, tomorrow=tomorrow,
                         audit_paid_rows=BUDGET if t else 0,
                         audit_present_classes=audit_classes,
                         audit_delta_pp=delta, choice=choice))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "decisions.csv", index=False)
    marker_path.write_text(json.dumps(dict(
        config=config, decisions_sha256=sha256(out / "decisions.csv")),
        indent=2) + "\n")
    print(home, "seed", seed, "gate frozen", flush=True)


def evaluate_home(args, archive, windows, members, official, home, seed,
                  rows, pooled_rows):
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    days, y = development_days_and_labels(
        archive, members.loc[home, "member"], home, windows, mapping)
    gate = pd.read_csv(args.output.resolve(strict=True) /
                       f"seed_{seed}" / home / "decisions.csv")
    ordered = np.unique(days)
    if (len(gate) != len(ordered) - 1 or
            not np.array_equal(gate.transition.to_numpy(),
                               np.arange(len(ordered) - 1)) or
            set(gate.choice) - {"source", "bcwi_050"}):
        raise ValueError(f"Gate layout differs: {home} {seed}")
    baseline = args.baseline.resolve(strict=True) / home
    source = np.load(verified(baseline, "static_mlp.npy"), mmap_mode="r")
    bcwi_folder = args.bcwi.resolve(strict=True) / f"seed_{seed}" / home
    care_folder = args.care.resolve(strict=True) / f"seed_{seed}" / home
    matrices = {method: np.zeros((24, 24), dtype=np.int64)
                for method in METHODS}
    flips = {method: 0 for method in METHODS}
    source_correct_total = 0
    for t, tomorrow in enumerate(ordered[1:]):
        prediction_out = bcwi_folder / f"transition_{t}"
        ids = np.load(verified(prediction_out, "tomorrow_indices.npy"))
        if not np.array_equal(ids, np.flatnonzero(days == tomorrow)):
            raise ValueError(f"Frozen prediction day differs: {home} {t}")
        curated = y[ids] >= 0
        truth = y[ids][curated]
        source_p = np.asarray(source[ids][curated], dtype=np.int64)
        bcwi_p = np.asarray(np.load(verified(
            prediction_out, "bcwi_050.npy"))[curated], dtype=np.int64)
        care_p = np.asarray(np.load(verified(
            care_folder / f"transition_{t}", "updated_care.npy"))[curated],
                            dtype=np.int64)
        route_p = bcwi_p if gate.iloc[t].choice == "bcwi_050" else source_p
        predictions = dict(source=source_p, bcwi_050=bcwi_p,
                           updated_care=care_p, paid_sign_route=route_p)
        correct = source_p == truth
        source_correct_total += int(correct.sum())
        source_f1 = macro_f1(truth, source_p, CLASSES)
        for method, pred in predictions.items():
            matrices[method] += np.bincount(
                truth * 24 + pred, minlength=24 * 24).reshape(24, 24)
            flips[method] += int(np.sum(pred[correct] != truth[correct]))
            rows.append(dict(home=home, seed=seed, transition=t,
                             tomorrow=tomorrow, method=method,
                             choice=gate.iloc[t].choice,
                             curated_rows=len(truth),
                             macro_f1_fixed24=macro_f1(truth, pred, CLASSES),
                             delta_vs_source_pp=100 * (
                                 macro_f1(truth, pred, CLASSES) - source_f1),
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
            home=home, seed=seed, method=method,
            pooled_curated_rows=int(support.sum()),
            pooled_present_classes=int(np.sum(support > 0)),
            pooled_macro_f1_fixed24=float(f1.mean()),
            pooled_negative_flip_rate_vs_source=(
                flips[method] / source_correct_total
                if source_correct_total else None),
            chosen_bcwi_days=int((gate.choice == "bcwi_050").sum()),
            transition_days=len(gate)))
    print(home, "seed", seed, "paid route evaluated", flush=True)


def evaluate(args, archive, windows, members, official, homes):
    # Verify every gate before any full development label is read.
    for home in homes:
        out = args.output.resolve(strict=True) / f"seed_{args.seed}" / home
        marker = json.loads((out / "complete.json").read_text())
        if sha256(out / "decisions.csv") != marker["decisions_sha256"]:
            raise ValueError(f"Frozen gate changed: {out}")
    rows, pooled_rows = [], []
    for home in homes:
        evaluate_home(args, archive, windows, members, official, home,
                      args.seed, rows, pooled_rows)
    pooled = pd.DataFrame(pooled_rows)
    if args.home is None:
        previous = pd.read_csv(args.bcwi.resolve(strict=True) /
                               f"seed_{args.seed}" / "evaluation_all" /
                               "pooled_by_home.csv")
        for method in ("source", "bcwi_050", "updated_care"):
            old = previous.loc[previous.method == method].set_index("home")
            new = pooled.loc[pooled.method == method].set_index("home")
            if set(old.index) != set(new.index):
                raise ValueError(f"Previous {method} home coverage differs")
            for column in ("pooled_curated_rows", "pooled_present_classes",
                           "pooled_macro_f1_fixed24",
                           "pooled_negative_flip_rate_vs_source"):
                if not np.allclose(old.loc[new.index, column], new[column],
                                   rtol=0, atol=1e-10):
                    raise ValueError(f"Recomputed {method} differs: {column}")
    output = (args.output.resolve() / f"seed_{args.seed}" /
              (f"evaluation_{args.home}" if args.home else "evaluation_all"))
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Evaluation output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_day.csv", index=False)
    pooled.to_csv(output / "pooled_by_home.csv", index=False)
    means = pooled.groupby("method")[["pooled_macro_f1_fixed24",
                                       "pooled_negative_flip_rate_vs_source"]].mean()
    report = "\n".join([
        "Exploratory paid-sign next-day BCWI/source fallback on examined development days.",
        f"Seed={args.seed}; first predicted day defaults to BCWI 0.50.",
        "Later gates use only 1000 already paid labels from yesterday's frozen model predictions.",
        "No additional labels; model-update schedule and prediction identities are fixed.",
        "Every gate was frozen before full development labels were scored here.",
        "Macro F1 is fixed-24; negative-flip denominator is source-correct rows.",
        "Development outcomes were viewed before this rule; not independent validation or a new method.",
        "No final-window label or feature decoded.",
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
                set(part.seed) != {seed} or
                set(part.method) != set(METHODS) or
                part.duplicated(["home", "method"]).any()):
            raise ValueError(f"Completed evaluation layout differs: {path}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    f1 = frame.pivot(index=["home", "seed"], columns="method",
                     values="pooled_macro_f1_fixed24")
    nfr = frame.pivot(index=["home", "seed"], columns="method",
                      values="pooled_negative_flip_rate_vs_source")
    paired = pd.DataFrame({
        "route_minus_bcwi_f1_pp": 100 * (f1.paid_sign_route - f1.bcwi_050),
        "route_minus_source_f1_pp": 100 * (f1.paid_sign_route - f1.source),
        "route_minus_care_f1_pp": 100 * (f1.paid_sign_route - f1.updated_care),
        "route_minus_bcwi_nfr_pp": 100 * (nfr.paid_sign_route - nfr.bcwi_050),
    }).reset_index()
    by_home = paired.groupby("home", as_index=False).agg(
        mean_route_minus_bcwi_f1_pp=("route_minus_bcwi_f1_pp", "mean"),
        seeds_route_beats_bcwi=("route_minus_bcwi_f1_pp",
                                lambda x: int((x > 0).sum())),
        mean_route_minus_source_f1_pp=("route_minus_source_f1_pp", "mean"),
        mean_route_minus_care_f1_pp=("route_minus_care_f1_pp", "mean"),
        mean_route_minus_bcwi_nfr_pp=("route_minus_bcwi_nfr_pp", "mean"))
    output = root / "aggregate_0_1_2_3_4"
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Aggregate output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    means = frame.groupby("method")[["pooled_macro_f1_fixed24",
                                       "pooled_negative_flip_rate_vs_source"]].mean()
    report = "\n".join([
        "Exploratory five-seed paid-sign BCWI/source next-day fallback.",
        "Same paid labels as online updates; first predicted day uses BCWI 0.50.",
        "Adjacent days and seeds within homes are correlated; homes are the independent units.",
        "Rule was designed after viewing development outcomes; final window untouched.",
        "F1 is fixed-24-class macro F1; negative-flip rate is conditional on source-correct rows.",
        "Across-home pooled means:", means.round(5).to_string(),
        "Per-home paired differences in percentage points:",
        by_home.round(4).to_string(index=False),
        "Across-home paired means:",
        by_home.drop(columns="home").mean(numeric_only=True).round(4).to_string(),
        "Homes where route beats fixed BCWI 0.50: " +
        f"{int((by_home.mean_route_minus_bcwi_f1_pp > 0).sum())}/{len(by_home)}",
        "Homes where route beats source: " +
        f"{int((by_home.mean_route_minus_source_f1_pp > 0).sum())}/{len(by_home)}",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate", "aggregate"))
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--bcwi", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_v1"))
    parser.add_argument("--care", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_care_update_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_paid_route_v1"))
    args = parser.parse_args()
    if args.stage == "aggregate":
        if args.archive is not None:
            parser.error("aggregate does not take an archive")
        aggregate(args.output.resolve(strict=True))
        return
    if args.archive is None:
        parser.error("prepare/evaluate require an archive")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    reviewed = json.loads((manifest / "audit.json").read_text())
    if sha256(archive) != reviewed["archive_sha256"]:
        parser.error("Archive differs from reviewed timestamp audit")
    args.archive_sha256 = reviewed["archive_sha256"]
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    homes = [args.home] if args.home else PRIMARY
    if args.stage == "prepare":
        for home in homes:
            prepare_home(args, archive, windows, members, official,
                         home, args.seed)
    else:
        evaluate(args, archive, windows, members, official, homes)


if __name__ == "__main__":
    main()
