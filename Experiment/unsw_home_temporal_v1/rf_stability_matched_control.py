"""Falsify whether fold agreement adds value beyond accepting fewer RF proposals.

This home07 class-9 rule was chosen after development outcomes were inspected.
The comparison is exploratory. It reuses the original 1000 paid labels per
seed and the fixed curated development sample; final-window data stay closed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from audit import sha256
from evaluate import read_development_labels
from f1_feasibility import macro_f1
from predict import RF, verify_prepared
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model, forward_batches


def target_f1(y, pred, target):
    true_positive = int(np.sum((y == target) & (pred == target)))
    support = int(np.sum(y == target))
    predicted = int(np.sum(pred == target))
    return 2 * true_positive / (support + predicted) if support + predicted else 0.


def evaluate_mask(y, source, mask, target, present, static_macro, static_target):
    route = np.where(mask, target, source)
    source_correct = source == y
    return dict(proposals=int(mask.sum()),
                macro_delta_pp=100 * (macro_f1(y, route, present) - static_macro),
                target_f1=target_f1(y, route, target),
                target_f1_delta_pp=100 * (target_f1(y, route, target) -
                                           static_target),
                negative_flip=float(np.mean(mask[source_correct])))


def random_controls(y, source, full_mask, selected_mask, target, present,
                    static_macro, static_target, draws, seed):
    rng = np.random.default_rng(430000 + seed)
    full_ids = np.flatnonzero(full_mask)
    accepted = int(selected_mask.sum())
    per_donor = {}
    for donor in np.unique(source[selected_mask]):
        per_donor[int(donor)] = (
            np.flatnonzero(full_mask & (source == donor)),
            int(np.sum(selected_mask & (source == donor))))
    out = []
    for draw in range(draws):
        global_mask = np.zeros(len(y), dtype=bool)
        if accepted:
            global_mask[rng.choice(full_ids, accepted, replace=False)] = True
        donor_mask = np.zeros(len(y), dtype=bool)
        for pool, keep in per_donor.values():
            if keep:
                donor_mask[rng.choice(pool, keep, replace=False)] = True
        for name, mask in (("random_global", global_mask),
                           ("random_by_donor", donor_mask)):
            out.append(dict(seed=seed, draw=draw, method=name,
                            **evaluate_mask(y, source, mask, target, present,
                                            static_macro, static_target)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--donor-class", type=int, default=14)
    parser.add_argument("--dev-sample", type=int, default=20000)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--stability-report", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_refit_stability_v2/home07"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_stability_matched_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (args.target_class not in range(24) or args.donor_class not in range(24)
            or args.target_class == args.donor_class or args.dev_sample < 100
            or args.draws < 10 or args.threads < 1):
        parser.error("Invalid class, sample size, draw count, or thread count")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    output = args.output.resolve() / args.home
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    audit = args.stability_report.resolve(strict=True)
    audit_config = json.loads((audit / "config.json").read_text())
    if (audit_config["home"] != args.home or
            audit_config["target_class"] != args.target_class or
            audit_config["donor_class"] != args.donor_class or
            audit_config["dev_sample"] != args.dev_sample or
            audit_config["archive_sha256"] != expected or
            audit_config["implementation_sha256"] !=
            sha256(HERE / "rf_refit_stability.py")):
        raise ValueError("Completed stability audit differs from this control")
    reference_folds = pd.read_csv(audit / "folds.csv").set_index(["seed", "fold"])
    reference_summary = pd.read_csv(audit / "summary.csv").set_index("seed")
    prep = args.prepared.resolve(strict=True) / args.home
    base = args.baseline.resolve(strict=True) / args.home
    verify_prepared(prep)
    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    if len(labels) != len(raw_query):
        raise ValueError("Paid query arrays differ")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = (official.loc[official.home == args.home]
               .set_index("mac").device_type.to_dict())
    y_all = read_development_labels(archive, members.loc[args.home, "member"],
                                    args.home, windows, mapping)
    curated = np.flatnonzero(y_all >= 0)
    sample = np.sort(np.random.default_rng(8800).choice(
        curated, size=min(args.dev_sample, len(curated)), replace=False))
    if len(sample) != args.dev_sample:
        raise ValueError("Development sample differs from stability audit")
    y = y_all[sample].astype(np.int64)
    source = np.asarray(np.load(verified(base, "static_mlp.npy"),
                                  mmap_mode="r")[sample], dtype=np.int64)
    dev_features = np.asarray(np.load(verified(base, "development_features.npy"),
                                        mmap_mode="r")[sample], dtype=np.float32)
    present = np.unique(y)
    static_macro = macro_f1(y, source, present)
    static_target = target_f1(y, source, args.target_class)
    preprocessing = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(verified(base, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    device = torch.device(args.device)
    model.encoder.to(device).eval()
    ref_rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        ref_rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(24)])
    ref_x = forward_batches(model.encoder,
                            preprocessing.transform(raw_source)[ref_ids],
                            device).numpy()
    ref_y = source_y[ref_ids]
    all_query_x = forward_batches(model.encoder,
                                  preprocessing.transform(raw_query),
                                  device).numpy()
    method_rows, random_rows, detail_rows = [], [], []
    for seed in range(5):
        chosen = np.load(verified(base / f"seed_{seed}",
                                  "paid_query_pool_indices.npy"))
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid indices differ: seed {seed}")
        paid_x = all_query_x[chosen]
        paid_y = np.asarray(labels[chosen], dtype=np.int64)
        votes = np.zeros(len(sample), dtype=np.uint8)
        folds = KFold(n_splits=5, shuffle=True, random_state=9100 + seed)
        for fold, (train, _) in enumerate(folds.split(paid_x)):
            forest = RandomForestClassifier(**RF, random_state=seed + 100 * fold,
                                            n_jobs=args.threads)
            forest.fit(np.concatenate((paid_x[train], ref_x)),
                       np.concatenate((paid_y[train], ref_y)))
            proposed = ((forest.predict(dev_features) == args.target_class) &
                        (source != args.target_class))
            expected_fold = reference_folds.loc[(seed, fold)]
            if (int(proposed.sum()) != expected_fold.dev_proposals or
                    int(np.sum(proposed & (source == args.donor_class))) !=
                    expected_fold.dev_donor_proposals):
                raise ValueError(f"Fold proposals differ: seed {seed}, fold {fold}")
            votes += proposed.astype(np.uint8)
        full_rf = np.asarray(np.load(verified(base / f"seed_{seed}",
                                            "frozen_rf.npy"), mmap_mode="r")[sample],
                             dtype=np.int64)
        full = (full_rf == args.target_class) & (source != args.target_class)
        consensus = votes >= 4
        selected = full & consensus
        if (int(full.sum()) != reference_summary.loc[seed, "full_dev_proposals"]
                or int(consensus.sum()) !=
                reference_summary.loc[seed, "consensus_dev_proposals"]):
            raise ValueError(f"Stability mask differs: seed {seed}")
        for method, mask in (("full_rf", full), ("consensus_only", consensus),
                             ("full_intersect_consensus", selected)):
            method_rows.append(dict(seed=seed, method=method,
                                    **evaluate_mask(y, source, mask,
                                                    args.target_class, present,
                                                    static_macro, static_target)))
        random_rows.extend(random_controls(y, source, full, selected,
                                           args.target_class, present,
                                           static_macro, static_target,
                                           args.draws, seed))
        donor = source == args.donor_class
        detail_rows.append(dict(seed=seed, full_proposals=int(full.sum()),
                                consensus_proposals=int(consensus.sum()),
                                full_and_consensus=int(selected.sum()),
                                consensus_outside_full=int(np.sum(consensus & ~full)),
                                full_donor_proposals=int(np.sum(full & donor)),
                                selected_donor_proposals=int(np.sum(selected & donor)),
                                full_donor_true_target=int(np.sum(
                                    full & donor & (y == args.target_class))),
                                selected_donor_true_target=int(np.sum(
                                    selected & donor & (y == args.target_class)))))
        print("seed", seed, "matched controls complete", flush=True)
    method = pd.DataFrame(method_rows)
    random = pd.DataFrame(random_rows)
    detail = pd.DataFrame(detail_rows)
    random_summary = (random.groupby(["seed", "method"])
                      [["macro_delta_pp", "target_f1_delta_pp", "negative_flip"]]
                      .agg(["mean", "median", lambda x: x.quantile(.025),
                            lambda x: x.quantile(.975)]))
    random_summary.columns = ["_".join((a, b if isinstance(b, str) else
                                        "quantile")) for a, b in random_summary.columns]
    random_summary = random_summary.reset_index()
    output.mkdir(parents=True, exist_ok=True)
    method.to_csv(output / "methods.csv", index=False)
    random.to_csv(output / "random_draws.csv", index=False)
    random_summary.to_csv(output / "random_summary.csv", index=False)
    detail.to_csv(output / "overlap.csv", index=False)
    report = "\n".join([
        "Exploratory full-RF stability intersection versus count-matched random routes.",
        "Class 9, home07, 4-of-5 threshold and donor 14 were selected after development inspection.",
        "Full RF, intersection and random controls share the same full-RF proposal set; consensus-only is a separate diagnostic.",
        f"Every route uses the same paid-label draw, source references and {len(sample)} development rows.",
        "Random-global matches total accepted full proposals; random-by-donor also matches source-predicted class counts.",
        "Random intervals are only assignment variability conditional on this examined development sample.",
        "Five paid seeds overlap; they are not five independent homes. Final-window data unopened.",
        "Per-seed methods (percentage-point deltas versus frozen source):",
        method.round(5).to_string(index=False),
        "Random-control summaries:",
        random_summary.round(5).to_string(index=False),
        "Overlap and focus donor:",
        detail.to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        home=args.home, target_class=args.target_class,
        donor_class=args.donor_class, dev_sample=len(sample),
        random_draws=args.draws, archive_sha256=expected,
        stability_report_config_sha256=sha256(audit / "config.json"),
        implementation_sha256=sha256(Path(__file__)),
        final_labels_or_features_decoded=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
