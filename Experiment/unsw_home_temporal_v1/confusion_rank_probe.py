"""Diagnostic: can source logits rank a paid confusion edge locally?

For the fixed home07 6-to-9 edge, assess score z_9-z_6 among flows the source
predicts as class 6. Query scores use only each seed's existing 1000 paid
labels; development labels are read for descriptive evaluation. No threshold
is chosen, no model is fitted, and no final-window row is decoded. This edge
was chosen after inspecting development results, so it is not blind evidence.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from audit import sha256
from evaluate import read_development_labels

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model


def verified(folder, name):
    marker = json.loads((folder / "complete.json").read_text())
    path = folder / name
    if sha256(path) != marker["artifacts"][name]:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def rank_metrics(labels, scores, source_pred, source_class, target_class):
    mask = source_pred == source_class
    truth = labels[mask] == target_class
    n = int(mask.sum())
    positive = int(truth.sum())
    return dict(source_predicted_rows=n, target_true_rows=positive,
                target_prevalence=positive / n if n else np.nan,
                ap=(float(average_precision_score(truth, scores[mask]))
                    if 0 < positive < n else np.nan),
                auc=(float(roc_auc_score(truth, scores[mask]))
                     if 0 < positive < n else np.nan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--source-class", type=int, default=6)
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--prior", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_prior_control_v1"))
    parser.add_argument("--archive", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal/all.zip"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (not 0 <= args.source_class < 24 or not 0 <= args.target_class < 24
            or args.source_class == args.target_class):
        parser.error("Classes must be distinct integers from 0 to 23")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    home = args.home
    source_class, target_class = args.source_class, args.target_class
    prep_dir = args.prepared.resolve(strict=True) / home
    base_dir = args.baseline.resolve(strict=True) / home
    prior_dir = args.prior.resolve(strict=True) / "predictions" / home
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected_archive = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected_archive:
        raise ValueError("Archive differs from reviewed timestamp audit")
    with np.load(verified(prep_dir, "source.npz"), allow_pickle=False) as data:
        raw_source = data["x"]
    with np.load(verified(prep_dir, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    paid_labels = np.load(verified(prep_dir, "query_labels.npy"), mmap_mode="r")
    if len(raw_query) != len(paid_labels):
        raise ValueError("Query labels/features length differs")
    prep = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(verified(base_dir, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.to(args.device).eval()
    rows = []
    with torch.no_grad():
        for seed in range(5):
            chosen = np.load(verified(base_dir / f"seed_{seed}",
                                      "paid_query_pool_indices.npy"))
            if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
                raise ValueError(f"Paid query indices differ: {home} {seed}")
            x = prep.transform(raw_query[chosen])
            logits = model(torch.as_tensor(x, dtype=torch.float32,
                                           device=args.device)).cpu().numpy()
            score = logits[:, target_class] - logits[:, source_class]
            labels = np.asarray(paid_labels[chosen])
            rows.append(dict(period="paid_query", seed=seed,
                             **rank_metrics(labels, score, logits.argmax(1),
                                            source_class, target_class)))
    model.cpu()

    dev_logits = np.load(verified(prior_dir, "source_dev_logits.npy"),
                         mmap_mode="r")
    static = np.load(verified(base_dir, "static_mlp.npy"), mmap_mode="r")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = official.loc[official.home == home].set_index("mac").device_type.to_dict()
    y = read_development_labels(archive, members.loc[home, "member"],
                                home, windows, mapping)
    if dev_logits.shape != (len(y), 24) or not np.array_equal(
            dev_logits.argmax(1), static):
        raise ValueError("Frozen development source logits differ")
    curated = y >= 0
    source_pred = np.asarray(static[curated])
    score = (np.asarray(dev_logits[curated, target_class])
             - np.asarray(dev_logits[curated, source_class]))
    rows.append(dict(period="development", seed=-1,
                     **rank_metrics(y[curated], score, source_pred,
                                    source_class, target_class)))
    result = pd.DataFrame(rows)
    print(f"Exploratory source-predicted {source_class} to true {target_class} rank probe.")
    print("AP baseline is target_prevalence within source-predicted rows.")
    print("Five paid seeds overlap; they are not independent datasets.")
    print("Development labels are descriptive; selected edge was already seen.")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
