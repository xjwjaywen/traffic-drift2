"""Exploratory one-home soft-count control for the paid-prior correction.

The fixed source model and 1000 paid query indices are reused. The only
change from prior_control is summing source softmax probabilities rather than
hard argmax predictions in the query-prior denominator. Development labels
are read for evaluation only; no final-window row is decoded. This is an
already-viewed development diagnostic, not a new method or blind test.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from audit import sha256
from evaluate import metrics, read_development_labels
from prior_control import BATCH, EFFECTIVE_QUERY_N, PSEUDOCOUNT, WEIGHT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model


FOCUS = (6, 8, 9, 10)


def verified_artifact(folder, name):
    marker = json.loads((folder / "complete.json").read_text())
    path = folder / name
    if sha256(path) != marker["artifacts"][name]:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="home07")
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    device = torch.device(args.device)
    home = args.home
    prep_dir = args.prepared.resolve(strict=True) / home
    base_dir = args.baseline.resolve(strict=True) / home
    prior_dir = args.prior.resolve(strict=True) / "predictions" / home
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected_archive = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected_archive:
        raise ValueError("Archive differs from reviewed timestamp audit")

    source_file = verified_artifact(prep_dir, "source.npz")
    query_file = verified_artifact(prep_dir, "query.npz")
    paid_file = verified_artifact(prep_dir, "query_labels.npy")
    model_file = verified_artifact(base_dir, "source_state.pt")
    logit_file = verified_artifact(prior_dir, "source_dev_logits.npy")
    hard_count_file = verified_artifact(prior_dir, "query_predicted_count.npy")
    static_file = verified_artifact(base_dir, "static_mlp.npy")
    with np.load(source_file, allow_pickle=False) as source:
        raw_source = source["x"]
    with np.load(query_file, allow_pickle=False) as query:
        raw_query = query["x"]
    paid_labels = np.load(paid_file, mmap_mode="r")
    if len(paid_labels) != len(raw_query):
        raise ValueError("Query labels/features length differs")
    prep = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(model_file, map_location="cpu",
                                     weights_only=True))
    model.to(device).eval()
    soft_count = np.zeros(24, dtype=np.float64)
    hard_count = np.zeros(24, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(raw_query), BATCH):
            x = prep.transform(raw_query[start:start + BATCH])
            logits = model(torch.as_tensor(x, dtype=torch.float32,
                                           device=device))
            soft_count += torch.softmax(logits, dim=1).sum(0).cpu().numpy()
            hard_count += np.bincount(logits.argmax(1).cpu().numpy(),
                                      minlength=24)
    model.cpu()
    saved_hard = np.load(hard_count_file)
    if not np.array_equal(hard_count, saved_hard):
        raise ValueError("Recomputed hard query counts differ from frozen control")
    if not np.isclose(soft_count.sum(), len(raw_query), atol=.1):
        raise ValueError("Soft query counts do not sum to pool size")

    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = official.loc[official.home == home].set_index("mac").device_type.to_dict()
    y = read_development_labels(archive, members.loc[home, "member"],
                                home, windows, mapping)
    logits = np.load(logit_file, mmap_mode="r")
    static = np.load(static_file, mmap_mode="r")
    if logits.shape != (len(y), 24) or not np.array_equal(logits.argmax(1), static):
        raise ValueError("Frozen source logits and static predictions differ")

    rows = []
    for seed in range(5):
        chosen = np.load(verified_artifact(base_dir / f"seed_{seed}",
                                           "paid_query_pool_indices.npy"))
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid query indices differ: {home} {seed}")
        paid_count = np.bincount(paid_labels[chosen], minlength=24)
        soft_effective = soft_count * (EFFECTIVE_QUERY_N / len(raw_query))
        soft_bias = (WEIGHT * (np.log(paid_count + PSEUDOCOUNT)
                               - np.log(soft_effective + PSEUDOCOUNT)))
        hard_bias = np.load(verified_artifact(prior_dir,
                                              f"bias_seed_{seed}.npy"))
        prediction = np.empty(len(y), dtype=np.uint8)
        for start in range(0, len(y), BATCH):
            prediction[start:start + BATCH] = np.argmax(
                logits[start:start + BATCH] + soft_bias[None, :], axis=1)
        hard_prediction = np.load(verified_artifact(prior_dir,
                                                    f"seed_{seed}.npy"),
                                  mmap_mode="r")
        for method, pred in (("static", static), ("hard", hard_prediction),
                             ("soft", prediction)):
            overall, classes = metrics(y, static, pred, home, seed, method)
            rows.append(dict(seed=seed, method=method, device_type=-1,
                             paid_support=np.nan, query_hard_effective=np.nan,
                             query_soft_effective=np.nan, bias=np.nan,
                             support=overall["curated_rows"], predicted_count=np.nan,
                             recall=np.nan, f1=overall["macro_f1_present"],
                             zero_recall=overall["zero_recall_classes"],
                             negative_flip=overall["negative_flip_rate_vs_static_mlp"]))
            for c in FOCUS:
                item = classes[c]
                rows.append(dict(seed=seed, method=method, device_type=c,
                                 paid_support=int(paid_count[c]),
                                 query_hard_effective=hard_count[c] * 1000 / len(raw_query),
                                 query_soft_effective=soft_effective[c],
                                 bias=(0. if method == "static" else
                                       hard_bias[c] if method == "hard" else soft_bias[c]),
                                 support=item["support"],
                                 predicted_count=item["predicted_count"],
                                 recall=item["recall"], f1=item["f1"],
                                 zero_recall=np.nan, negative_flip=np.nan))
    result = pd.DataFrame(rows)
    print("Exploratory one-home hard-vs-soft query count control; same labels and weight.")
    print("Development outcomes already viewed; no final-window data read.")
    print("Whole-model macro F1 and negative-flip rate:")
    print(result.loc[result.device_type == -1, ["seed", "method", "f1",
                    "zero_recall", "negative_flip"]].round(4).to_string(index=False))
    print("Focus classes; F1 and recall are development outcomes:")
    print(result.loc[result.device_type >= 0, ["seed", "method", "device_type",
                    "paid_support", "query_hard_effective", "query_soft_effective",
                    "bias", "support", "predicted_count", "recall", "f1"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
