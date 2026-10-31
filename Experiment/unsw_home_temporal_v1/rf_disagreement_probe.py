"""Paid-label-only OOF audit of RF overrides of the frozen source model.

The home07 class-9 focus and class-6 edge were selected after development
results were viewed. This is exploratory; no development or final-window
row is read. Each paid row is predicted by an RF trained without that row.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from audit import sha256
from predict import RF, verify_prepared

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model, forward_batches


def verified(folder, name):
    marker = json.loads((folder / "complete.json").read_text())
    path = folder / name
    if sha256(path) != marker["artifacts"][name]:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def compare(mask, truth, source_pred, rf_pred):
    wins = int(np.sum(mask & (rf_pred == truth) & (source_pred != truth)))
    losses = int(np.sum(mask & (source_pred == truth) & (rf_pred != truth)))
    both_wrong = int(np.sum(mask & (source_pred != truth) & (rf_pred != truth)))
    return dict(rows=int(mask.sum()), rf_wins=wins, source_wins=losses,
                both_wrong=both_wrong, net_wins=wins - losses)


def run(args):
    home = args.home
    prep = args.prepared.resolve(strict=True) / home
    base = args.baseline.resolve(strict=True) / home
    verify_prepared(prep)
    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        raw_source, source_y = data["x"], data["y"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    if len(labels) != len(raw_query):
        raise ValueError("Prepared query features/labels differ")
    prep_model = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(verified(base, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    ref_rng = np.random.default_rng(10007)
    ref_ids = np.concatenate([
        ref_rng.choice(np.flatnonzero(source_y == c), 5, replace=False)
        for c in range(24)])
    device = torch.device(args.device)
    model.encoder.to(device).eval()
    ref_x = forward_batches(model.encoder,
                            prep_model.transform(raw_source[ref_ids]), device).numpy()
    ref_y = source_y[ref_ids]
    model.head.to(device).eval()
    for seed in range(5):
        chosen = np.load(verified(base / f"seed_{seed}",
                                  "paid_query_pool_indices.npy"))
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid indices differ for seed {seed}")
        y = np.asarray(labels[chosen], dtype=np.int64)
        paid_x = forward_batches(model.encoder,
                                 prep_model.transform(raw_query[chosen]),
                                 device).numpy()
        source_logits = forward_batches(model.head, paid_x, device).numpy()
        source_pred = source_logits.argmax(axis=1)
        rf_pred = np.empty(1000, dtype=np.int64)
        rf_probability = np.empty(1000, dtype=np.float64)
        folds = KFold(n_splits=5, shuffle=True, random_state=9100 + seed)
        for fold, (train, valid) in enumerate(folds.split(paid_x)):
            forest = RandomForestClassifier(**RF, random_state=seed + 100 * fold,
                                            n_jobs=args.threads)
            forest.fit(np.concatenate((paid_x[train], ref_x)),
                       np.concatenate((y[train], ref_y)))
            rf_pred[valid] = forest.predict(paid_x[valid])
            column = np.flatnonzero(forest.classes_ == args.target_class)
            if len(column) != 1:
                raise ValueError("Target class missing despite source references")
            rf_probability[valid] = forest.predict_proba(paid_x[valid])[:, column[0]]
        masks = {
            "all_rf_to_target": (rf_pred == args.target_class)
                                & (source_pred != rf_pred),
            "source_pair_to_target": (source_pred == args.source_class)
                                     & (rf_pred == args.target_class),
        }
        for name, mask in masks.items():
            row = compare(mask, y, source_pred, rf_pred)
            print(f"seed={seed} group={name} {row}", flush=True)
        pair = source_pred == args.source_class
        pair_y = y[pair] == args.target_class
        if pair_y.any() and (~pair_y).any():
            source_margin = (source_logits[pair, args.target_class]
                             - source_logits[pair, args.source_class])
            ap_rf = float(average_precision_score(pair_y, rf_probability[pair]))
            auc_rf = float(roc_auc_score(pair_y, rf_probability[pair]))
            ap_source = float(average_precision_score(pair_y, source_margin))
            auc_source = float(roc_auc_score(pair_y, source_margin))
            print(f"seed={seed} source_pred={args.source_class} rows={pair.sum()} "
                  f"true_target={pair_y.sum()} prevalence={pair_y.mean():.4f} "
                  f"rf_ap={ap_rf:.4f} rf_auc={auc_rf:.4f} "
                  f"source_ap={ap_source:.4f} source_auc={auc_source:.4f}",
                  flush=True)
    model.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--source-class", type=int, default=6)
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (args.source_class == args.target_class or
            not 0 <= args.source_class < 24 or
            not 0 <= args.target_class < 24):
        parser.error("Select two distinct class IDs in 0..23")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    print("Paid-label-only five-fold OOF RF disagreement audit.")
    print("Class pair chosen after seeing development outcomes; exploratory.")
    print("Five paid seeds overlap. No development or final-window rows read.")
    run(args)


if __name__ == "__main__":
    main()
