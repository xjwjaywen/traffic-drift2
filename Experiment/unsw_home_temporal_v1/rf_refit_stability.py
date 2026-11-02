"""Read-only probe of RF refit instability versus temporal transport.

Each previously fixed 5-fold, 800-paid-label RF predicts both its held-out
paid fold and the same uniform curated development sample. This checks
whether full-RF proposals, especially home07 seed-4 source-14 to target-9,
are also made by the smaller fold models. A fixed 4-of-5 consensus route is
shown only as an exploratory control. Final-window data stay unopened.
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
from rf_disagreement_probe import compare, verified

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model, forward_batches


def outcome(y, static, route, target, donor):
    changed = route != static
    source_correct = static == y
    return dict(proposals=int(changed.sum()),
                donor_proposals=int(np.sum(changed & (static == donor))),
                donor_true_target=int(np.sum(changed & (static == donor)
                                             & (y == target))),
                macro_delta_pp=100 * (
                    macro_f1(y, route, np.unique(y)) -
                    macro_f1(y, static, np.unique(y))),
                negative_flip=float(np.mean(route[source_correct] !=
                                            y[source_correct])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--donor-class", type=int, default=14)
    parser.add_argument("--dev-sample", type=int, default=20000)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--gate", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_target_route_v1/home07/gate.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_refit_stability_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (not 0 <= args.target_class < 24 or not 0 <= args.donor_class < 24
            or args.target_class == args.donor_class or args.dev_sample < 100):
        parser.error("Invalid class or sample size")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    output = args.output.resolve() / args.home
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())[
        "archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    gate = json.loads(args.gate.resolve(strict=True).read_text())
    if (gate["home"] != args.home or gate["target_class"] != args.target_class
            or len(gate["decisions"]) != 5):
        raise ValueError("Frozen paid gate differs")
    prep = args.prepared.resolve(strict=True) / args.home
    base = args.baseline.resolve(strict=True) / args.home
    verify_prepared(prep)
    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    if len(raw_query) != len(labels):
        raise ValueError("Prepared query labels/features differ")
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
    dev_y = y_all[sample].astype(np.int64)
    dev_static = np.asarray(np.load(verified(base, "static_mlp.npy"),
                                    mmap_mode="r")[sample], dtype=np.int64)
    dev_features = np.asarray(np.load(verified(base, "development_features.npy"),
                                      mmap_mode="r")[sample], dtype=np.float32)
    if len(dev_y) != len(dev_static) or len(dev_y) != len(dev_features):
        raise ValueError("Development sample alignment differs")
    prep_model = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(verified(base, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(24)])
    device = torch.device(args.device)
    model.encoder.to(device).eval()
    ref_x = forward_batches(model.encoder,
                            prep_model.transform(raw_source[ref_ids]), device).numpy()
    ref_y = source_y[ref_ids]
    model.head.to(device).eval()
    summaries, fold_rows = [], []
    for seed in range(5):
        chosen = np.load(verified(base / f"seed_{seed}",
                                  "paid_query_pool_indices.npy"))
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid indices differ: seed {seed}")
        paid_y = np.asarray(labels[chosen], dtype=np.int64)
        paid_x = forward_batches(model.encoder,
                                 prep_model.transform(raw_query[chosen]),
                                 device).numpy()
        paid_static = forward_batches(model.head, paid_x, device).numpy().argmax(1)
        oof_rf = np.empty(1000, dtype=np.int64)
        fold_proposal = np.zeros((5, len(sample)), dtype=bool)
        folds = KFold(n_splits=5, shuffle=True, random_state=9100 + seed)
        for fold, (train, valid) in enumerate(folds.split(paid_x)):
            forest = RandomForestClassifier(**RF, random_state=seed + 100 * fold,
                                            n_jobs=args.threads)
            forest.fit(np.concatenate((paid_x[train], ref_x)),
                       np.concatenate((paid_y[train], ref_y)))
            oof_rf[valid] = forest.predict(paid_x[valid])
            dev_rf = forest.predict(dev_features)
            proposed = (dev_rf == args.target_class) & (
                dev_static != args.target_class)
            fold_proposal[fold] = proposed
            heldout = (oof_rf[valid] == args.target_class) & (
                paid_static[valid] != args.target_class)
            fold_rows.append(dict(seed=seed, fold=fold,
                                  heldout_paid_rows=len(valid),
                                  heldout_paid_proposals=int(heldout.sum()),
                                  dev_sample_rows=len(sample),
                                  dev_proposals=int(proposed.sum()),
                                  dev_donor_proposals=int(np.sum(
                                      proposed & (dev_static == args.donor_class))),
                                  dev_donor_true_target=int(np.sum(
                                      proposed & (dev_static == args.donor_class)
                                      & (dev_y == args.target_class)))))
        oof_mask = (oof_rf == args.target_class) & (
            paid_static != args.target_class)
        actual = compare(oof_mask, paid_y, paid_static, oof_rf)
        frozen = gate["decisions"][seed]
        if frozen["seed"] != seed or any(actual[key] != frozen[key] for key in
               ("rows", "rf_wins", "source_wins", "both_wrong")):
            raise ValueError(f"OOF RF differs from frozen paid gate: seed {seed}")
        votes = fold_proposal.sum(axis=0)
        consensus_route = np.where(votes >= 4, args.target_class, dev_static)
        full_rf = np.asarray(np.load(verified(base / f"seed_{seed}",
                                            "frozen_rf.npy"), mmap_mode="r")[sample],
                             dtype=np.int64)
        full_route = np.where((full_rf == args.target_class) &
                              (dev_static != args.target_class),
                              args.target_class, dev_static)
        full = outcome(dev_y, dev_static, full_route,
                       args.target_class, args.donor_class)
        consensus = outcome(dev_y, dev_static, consensus_route,
                            args.target_class, args.donor_class)
        summaries.append(dict(seed=seed, sampled_dev_rows=len(sample),
                              oof_paid_proposals=actual["rows"],
                              full_dev_proposals=full["proposals"],
                              consensus_dev_proposals=consensus["proposals"],
                              full_donor_proposals=full["donor_proposals"],
                              consensus_donor_proposals=consensus["donor_proposals"],
                              full_donor_true_target=full["donor_true_target"],
                              consensus_donor_true_target=consensus[
                                  "donor_true_target"],
                              full_macro_delta_pp=full["macro_delta_pp"],
                              consensus_macro_delta_pp=consensus["macro_delta_pp"],
                              full_negative_flip=full["negative_flip"],
                              consensus_negative_flip=consensus["negative_flip"]))
        print("seed", seed, "matched-fold prediction complete", flush=True)
    model.cpu()
    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(fold_rows)
    summary["full_dev_proposal_rate"] = (summary.full_dev_proposals /
                                         summary.sampled_dev_rows)
    summary["consensus_dev_proposal_rate"] = (
        summary.consensus_dev_proposals / summary.sampled_dev_rows)
    summary["full_donor_target_fraction"] = np.divide(
        summary.full_donor_true_target, summary.full_donor_proposals,
        out=np.full(len(summary), np.nan), where=summary.full_donor_proposals > 0)
    summary["consensus_donor_target_fraction"] = np.divide(
        summary.consensus_donor_true_target, summary.consensus_donor_proposals,
        out=np.full(len(summary), np.nan),
        where=summary.consensus_donor_proposals > 0)
    detail["heldout_paid_proposal_rate"] = (detail.heldout_paid_proposals /
                                             detail.heldout_paid_rows)
    detail["dev_proposal_rate"] = (detail.dev_proposals /
                                    detail.dev_sample_rows)
    detail["dev_donor_target_fraction"] = np.divide(
        detail.dev_donor_true_target, detail.dev_donor_proposals,
        out=np.full(len(detail), np.nan), where=detail.dev_donor_proposals > 0)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    detail.to_csv(output / "folds.csv", index=False)
    report = "\n".join([
        "Exploratory RF refit-stability audit on a fixed uniform curated development sample.",
        "Each 800-label fold RF predicts both its held-out paid rows and the same development rows.",
        "Full RF uses all 1000 paid rows and a different random state.",
        "Consensus route requires at least 4 of 5 fold RFs to propose target class.",
        "Fold forests share most training rows; consensus is a stability heuristic, not a confidence interval.",
        "Development labels were examined before this audit; no final labels/features decoded.",
        "Paid and development samples have different sizes and sampling frames; use rates for comparison.",
        "Sample size and all rules were fixed before this run.",
        "By seed:",
        summary.round(5).to_string(index=False),
        "By fold:",
        detail.to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        home=args.home, target_class=args.target_class,
        donor_class=args.donor_class, dev_sample=len(sample),
        sample_rng_seed=8800, archive_sha256=expected,
        gate_sha256=sha256(args.gate),
        implementation_sha256=sha256(Path(__file__)),
        final_labels_or_features_decoded=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
