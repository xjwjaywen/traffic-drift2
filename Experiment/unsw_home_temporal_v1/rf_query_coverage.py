"""Check whether unstable RF override regions exist in the earlier query pool.

Home07 seed4 and the 14->9 edge were selected after development inspection.
This is a descriptive, read-only diagnostic. Only the already paid 1000 query
labels train RFs; all other query labels and final-window rows stay unopened.
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
from predict import RF, verify_prepared
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model, forward_batches


def counts(source, challenger, target, donor):
    source_donor = source == donor
    proposal = (challenger == target) & (source != target)
    edge = source_donor & proposal
    return dict(rows=len(source), source_donor_rows=int(source_donor.sum()),
                proposals=int(proposal.sum()), donor_proposals=int(edge.sum()),
                proposal_rate=float(proposal.mean()),
                donor_proposal_rate=float(edge.mean()),
                donor_given_source_donor=float(edge.sum() / source_donor.sum())
                if source_donor.any() else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--donor-class", type=int, default=14)
    parser.add_argument("--dev-sample", type=int, default=20000)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_query_coverage_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (args.seed not in range(5) or args.target_class not in range(24)
            or args.donor_class not in range(24)
            or args.target_class == args.donor_class or args.dev_sample < 100):
        parser.error("Invalid seed, class, or development sample size")
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
    prep = args.prepared.resolve(strict=True) / args.home
    base = args.baseline.resolve(strict=True) / args.home
    verify_prepared(prep)
    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    paid_labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    chosen = np.load(verified(base / f"seed_{args.seed}",
                              "paid_query_pool_indices.npy"))
    if (chosen.shape != (1000,) or len(np.unique(chosen)) != 1000
            or len(raw_query) != len(paid_labels)):
        raise ValueError("Paid query selection differs")
    unpaid_mask = np.ones(len(raw_query), dtype=bool)
    unpaid_mask[chosen] = False
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = (official.loc[official.home == args.home]
               .set_index("mac").device_type.to_dict())
    dev_y = read_development_labels(archive, members.loc[args.home, "member"],
                                    args.home, windows, mapping)
    curated = np.flatnonzero(dev_y >= 0)
    dev_sample = np.sort(np.random.default_rng(8800).choice(
        curated, size=min(args.dev_sample, len(curated)), replace=False))
    dev_x = np.asarray(np.load(verified(base, "development_features.npy"),
                               mmap_mode="r")[dev_sample], dtype=np.float32)
    dev_source = np.asarray(np.load(verified(base, "static_mlp.npy"),
                                    mmap_mode="r")[dev_sample], dtype=np.int64)
    preprocessing = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(verified(base, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    device = torch.device(args.device)
    model.encoder.to(device).eval()
    rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(24)])
    ref_x = forward_batches(model.encoder,
                            preprocessing.transform(raw_source[ref_ids]),
                            device).numpy()
    paid_x = forward_batches(model.encoder,
                             preprocessing.transform(raw_query[chosen]),
                             device).numpy()
    paid_y = np.asarray(paid_labels[chosen], dtype=np.int64)
    query_x = forward_batches(model.encoder,
                              preprocessing.transform(raw_query[unpaid_mask]),
                              device).numpy()
    model.head.to(device).eval()
    query_source = forward_batches(model.head, query_x, device).numpy().argmax(1)
    rows = []
    folds = list(KFold(n_splits=5, shuffle=True,
                       random_state=9100 + args.seed).split(paid_x))
    for name, train, random_state in [
        ("fold_3", folds[3][0], args.seed + 300),
        ("fold_4", folds[4][0], args.seed + 400),
        ("full", np.arange(1000), args.seed),
    ]:
        forest = RandomForestClassifier(**RF, random_state=random_state,
                                        n_jobs=args.threads)
        forest.fit(np.concatenate((paid_x[train], ref_x)),
                   np.concatenate((paid_y[train], source_y[ref_ids])))
        query_rf = forest.predict(query_x)
        dev_rf = forest.predict(dev_x)
        if name == "full":
            frozen = np.load(verified(base / f"seed_{args.seed}",
                                      "frozen_rf.npy"), mmap_mode="r")
            if not np.array_equal(dev_rf, frozen[dev_sample]):
                raise ValueError("Reproduced full RF differs from frozen predictions")
        for period, source, challenger in (
            ("unpaid_query_pool", query_source, query_rf),
            ("development_sample", dev_source, dev_rf),
        ):
            rows.append(dict(model=name, period=period, train_paid_rows=len(train),
                             **counts(source, challenger, args.target_class,
                                      args.donor_class)))
        print(name, "query and development predictions complete", flush=True)
    report_rows = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    report_rows.to_csv(output / "summary.csv", index=False)
    report = "\n".join([
        "Exploratory unlabeled query-pool coverage audit; home/edge selected after development inspection.",
        "Only already paid 1000 query labels and five-per-class source references train RFs.",
        "The earlier query pool is a fixed uniform reservoir; all paid rows are excluded from its counts.",
        "The development sample matches the previous refit-stability audit; development labels select curated rows only.",
        "No unpaid query labels or final-window labels/features were read.",
        "Full RF predictions are checked against the frozen development predictions.",
        "Proposal rates compare feature-region coverage; they do not measure correctness in the unpaid query pool.",
        report_rows.round(6).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        home=args.home, seed=args.seed, target_class=args.target_class,
        donor_class=args.donor_class, dev_sample=len(dev_sample),
        archive_sha256=expected, implementation_sha256=sha256(Path(__file__)),
        unpaid_query_labels_read=False, final_window_read=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
