"""Paid-label-only feasibility audit for macro-F1-aware RF routing.

Reproduce the previously fixed five-fold RF predictions on each 1000-label
query draw. Evaluate one fixed policy: accept every RF proposal to the
preselected target class. Row and query-day bootstraps describe uncertainty
conditional on those OOF predictions. They do not certify future performance.
Only paid query row timestamps are decoded; other rows are streamed past
without decoding their timestamp, feature, or label fields.
"""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from audit import sha256
from predict import RF, verify_prepared
from rf_disagreement_probe import compare, verified

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model, forward_batches

N_CLASSES = 24


def macro_f1(y, pred, present, weights=None):
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)
    matrix = np.bincount(y * N_CLASSES + pred, weights=weights,
                         minlength=N_CLASSES * N_CLASSES).reshape(
                             N_CLASSES, N_CLASSES)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    score = np.divide(2 * tp, support + predicted,
                      out=np.zeros(N_CLASSES), where=(support + predicted) > 0)
    return float(score[present].mean())


def bootstrap_delta(y, source, routed, days, draws, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    present = np.unique(y)
    day_labels, day_index = np.unique(days, return_inverse=True)
    if len(day_labels) < 2:
        raise ValueError("Need at least two paid query days")
    row_delta = np.empty(draws)
    day_delta = np.empty(draws)
    for i in range(draws):
        row_weight = rng.multinomial(n, np.full(n, 1 / n))
        row_delta[i] = macro_f1(y, routed, present, row_weight) - macro_f1(
            y, source, present, row_weight)
        sampled_days = rng.integers(len(day_labels), size=len(day_labels))
        day_weight = np.bincount(sampled_days, minlength=len(day_labels))[
            day_index]
        day_delta[i] = macro_f1(y, routed, present, day_weight) - macro_f1(
            y, source, present, day_weight)
    return (np.quantile(row_delta, [.025, .5, .975]),
            np.quantile(day_delta, [.025, .5, .975]))


def selected_query_days(archive, member, last_query_day, selected_ids):
    found = {}
    last_id = max(selected_ids)
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        if stream.readline().split(b",", 1)[0].strip() != b"start":
            raise ValueError("Unexpected archive header")
        for row_id in range(last_id + 1):
            line = stream.readline()
            if not line:
                raise ValueError("Archive ended before selected query rows")
            if row_id in selected_ids:
                day = line[:10].decode("ascii")
                if day > last_query_day:
                    raise ValueError(f"Selected paid row is after query boundary: {row_id}")
                found[row_id] = day
    if found.keys() != selected_ids:
        raise ValueError(f"Missing query timestamps: {len(selected_ids - found.keys())}")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--gate", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_target_route_v1/home07/gate.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_f1_feasibility_v1"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.target_class < N_CLASSES or args.draws < 100:
        parser.error("Invalid target class or bootstrap count")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    out = args.output.resolve() / args.home
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output must be new and empty: {out}")
    archive = args.archive.resolve(strict=True)
    expected = json.loads((HERE / "metadata/audit.json").read_text())[
        "archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    prep = args.prepared.resolve(strict=True) / args.home
    base = args.baseline.resolve(strict=True) / args.home
    verify_prepared(prep)
    frozen_gate = json.loads(args.gate.resolve(strict=True).read_text())
    if (frozen_gate["home"] != args.home or
            frozen_gate["target_class"] != args.target_class or
            len(frozen_gate["decisions"]) != 5):
        raise ValueError("Frozen prior gate does not match this audit")
    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        raw_query, query_ids = data["x"], data["ids"]
    labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    if len(labels) != len(raw_query):
        raise ValueError("Prepared query features/labels differ")
    chosen_by_seed = {}
    all_ids = set()
    for seed in range(5):
        chosen = np.load(verified(base / f"seed_{seed}",
                                  "paid_query_pool_indices.npy"))
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid indices differ for seed {seed}")
        chosen_by_seed[seed] = chosen
        all_ids.update(map(int, query_ids[chosen]))
    members = pd.read_csv(HERE / "metadata/members.csv").set_index("home")
    windows = pd.read_csv(HERE / "metadata/windows.csv").set_index(
        ["home", "split"])
    member = str(members.loc[args.home, "member"])
    last_query_day = str(windows.loc[(args.home, "query"), "last_day"])
    days_by_id = selected_query_days(archive, member, last_query_day, all_ids)

    prep_model = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, N_CLASSES)
    model.load_state_dict(torch.load(verified(base, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(N_CLASSES)])
    device = torch.device(args.device)
    model.encoder.to(device).eval()
    ref_x = forward_batches(model.encoder,
                            prep_model.transform(raw_source[ref_ids]), device).numpy()
    ref_y = source_y[ref_ids]
    model.head.to(device).eval()
    summary, edges = [], []
    for seed, chosen in chosen_by_seed.items():
        y = np.asarray(labels[chosen], dtype=np.int64)
        paid_x = forward_batches(model.encoder,
                                 prep_model.transform(raw_query[chosen]),
                                 device).numpy()
        source = forward_batches(model.head, paid_x, device).numpy().argmax(1)
        rf = np.empty(len(chosen), dtype=np.int64)
        folds = KFold(n_splits=5, shuffle=True, random_state=9100 + seed)
        for fold, (train, valid) in enumerate(folds.split(paid_x)):
            forest = RandomForestClassifier(**RF, random_state=seed + 100 * fold,
                                            n_jobs=args.threads)
            forest.fit(np.concatenate((paid_x[train], ref_x)),
                       np.concatenate((y[train], ref_y)))
            rf[valid] = forest.predict(paid_x[valid])
        mask = (rf == args.target_class) & (source != rf)
        actual = compare(mask, y, source, rf)
        frozen = frozen_gate["decisions"][seed]
        if any(actual[key] != frozen[key] for key in
               ("rows", "rf_wins", "source_wins", "both_wrong")):
            raise ValueError(f"OOF predictions differ from frozen audit: seed {seed}")
        routed = np.where(mask, rf, source)
        present = np.unique(y)
        source_f1 = macro_f1(y, source, present)
        routed_f1 = macro_f1(y, routed, present)
        days = np.asarray([days_by_id[int(i)] for i in query_ids[chosen]])
        row_ci, day_ci = bootstrap_delta(y, source, routed, days,
                                         args.draws, 52000 + seed)
        source_correct = source == y
        nfr = float((routed[source_correct] != y[source_correct]).mean())
        summary.append(dict(seed=seed, paid_rows=len(y), present_classes=len(present),
                            target_support=int((y == args.target_class).sum()),
                            query_days=len(np.unique(days)), proposal_rows=actual["rows"],
                            source_macro_f1=source_f1, all_route_macro_f1=routed_f1,
                            delta_pp=100 * (routed_f1 - source_f1),
                            row_boot_low_pp=100 * row_ci[0],
                            row_boot_med_pp=100 * row_ci[1],
                            row_boot_high_pp=100 * row_ci[2],
                            day_boot_low_pp=100 * day_ci[0],
                            day_boot_med_pp=100 * day_ci[1],
                            day_boot_high_pp=100 * day_ci[2],
                            negative_flip=nfr))
        for donor in np.unique(source[mask]):
            edge = mask & (source == donor)
            route_edge = np.where(edge, rf, source)
            counts = compare(edge, y, source, rf)
            edges.append(dict(seed=seed, source_predicted_class=int(donor),
                              **counts,
                              target_true=int(np.sum(edge & (y == args.target_class))),
                              source_true=int(np.sum(edge & (y == donor))),
                              edge_macro_delta_pp=100 * (
                                  macro_f1(y, route_edge, present) - source_f1)))
        print("seed", seed, "complete", flush=True)
    model.cpu()
    out.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary)
    edges_frame = pd.DataFrame(edges)
    summary_frame.to_csv(out / "summary.csv", index=False)
    edges_frame.to_csv(out / "edges.csv", index=False)
    report = "\n".join([
        "Exploratory paid-label-only macro-F1 feasibility audit, home07 class 9.",
        "Fixed policy: accept every RF-to-class-9 disagreement; no OOF gate selection.",
        "RF predictions are 5-fold out-of-fold on each existing 1000-label draw.",
        "Class 9 was selected after viewing development results; seeds overlap.",
        "Bootstrap intervals describe paid-query sampling conditional on OOF models.",
        "Row bootstrap ignores temporal dependence; day bootstrap has few days.",
        "Neither interval guarantees future performance under drift.",
        "Only selected paid-row timestamps were decoded from the archive.",
        "Summary (F1 changes and intervals in percentage points):",
        summary_frame.round(4).to_string(index=False),
        "Donor edges (source prediction -> RF class 9):",
        edges_frame.sort_values(["seed", "rows"], ascending=[True, False])
        .round(4).to_string(index=False),
    ]) + "\n"
    (out / "REPORT.txt").write_text(report)
    (out / "config.json").write_text(json.dumps(dict(
        home=args.home, target_class=args.target_class, draws=args.draws,
        archive_sha256=expected, frozen_gate_sha256=sha256(args.gate),
        implementation_sha256=sha256(Path(__file__)),
        later_period_fields_decoded=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
