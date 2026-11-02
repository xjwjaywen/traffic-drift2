"""Audit whether paid labels cover source-predicted groups that later grow.

This is a source-only, class-agnostic feasibility audit, not an adaptation
method. It reads labels only for each already paid 1000-row query draw.
Development labels identify rows in the reviewed 24-device inventory but
are never used to fit a model or to choose a route. Final rows stay closed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from audit import sha256
from evaluate import read_development_labels
from predict import PRIMARY, verify_prepared
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model, forward_batches


def one_home(home, archive, prepared, baseline, windows, members, official,
             device):
    prep = prepared / home
    base = baseline / home
    verify_prepared(prep)
    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        raw_source = data["x"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        raw_query = data["x"]
    labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    if len(raw_query) != len(labels):
        raise ValueError(f"Prepared query arrays differ: {home}")
    mapping = (official.loc[official.home == home]
               .set_index("mac").device_type.to_dict())
    y_dev = read_development_labels(archive, members.loc[home, "member"],
                                    home, windows, mapping)
    curated = y_dev >= 0
    static_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
    if len(static_all) != len(y_dev):
        raise ValueError(f"Development source predictions differ: {home}")
    dev_pred = np.asarray(static_all[curated], dtype=np.int64)
    preprocessing = Preprocessor().fit(raw_source)
    model = Model(raw_source.shape[1] * 2, 24)
    model.load_state_dict(torch.load(verified(base, "source_state.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    features = forward_batches(model.encoder,
                               preprocessing.transform(raw_query),
                               device).numpy()
    query_pred = forward_batches(model.head, features, device).numpy().argmax(1)
    query_count = np.bincount(query_pred, minlength=24)
    dev_count = np.bincount(dev_pred, minlength=24)
    query_share = query_count / len(query_pred)
    dev_share = dev_count / len(dev_pred)
    ratio = np.divide(dev_share, query_share,
                      out=np.full(24, np.inf), where=query_share > 0)
    ratio[(query_share == 0) & (dev_share == 0)] = np.nan
    rows = []
    for seed in range(5):
        chosen = np.load(verified(base / f"seed_{seed}",
                                  "paid_query_pool_indices.npy"))
        if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid query indices differ: {home} seed {seed}")
        paid_pred = query_pred[chosen]
        paid_y = np.asarray(labels[chosen], dtype=np.int64)
        for donor in range(24):
            selected = paid_pred == donor
            rows.append(dict(
                home=home, seed=seed, donor=donor,
                query_rows=len(query_pred), dev_curated_rows=len(dev_pred),
                query_source_pred_rows=int(query_count[donor]),
                dev_source_pred_rows=int(dev_count[donor]),
                query_source_pred_share=float(query_share[donor]),
                dev_source_pred_share=float(dev_share[donor]),
                dev_to_query_source_pred_ratio=float(ratio[donor]),
                paid_source_pred_rows=int(selected.sum()),
                paid_source_correct=int(np.sum(paid_y[selected] == donor)),
                paid_source_errors=int(np.sum(paid_y[selected] != donor)),
                paid_true_class_rows=int(np.sum(paid_y == donor))))
    print(home, "donor coverage complete", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_donor_coverage_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    homes = [args.home] if args.home else PRIMARY
    rows = []
    for home in homes:
        rows.extend(one_home(home, archive,
                             args.prepared.resolve(strict=True),
                             args.baseline.resolve(strict=True),
                             windows, members, official,
                             torch.device(args.device)))
    result = pd.DataFrame(rows)
    # These cutoffs describe coverage; they are not a proposed routing rule.
    risky = ((result.dev_source_pred_rows >= 100) &
             (result.dev_to_query_source_pred_ratio >= 4))
    per_seed = (result.assign(shifted=risky,
                              sparse_shifted=risky &
                              (result.paid_source_pred_rows < 10),
                              shifted_dev_mass=np.where(
                                  risky, result.dev_source_pred_share, 0),
                              sparse_shifted_dev_mass=np.where(
                                  risky & (result.paid_source_pred_rows < 10),
                                  result.dev_source_pred_share, 0))
                .groupby(["home", "seed"], as_index=False)
                .agg(shifted_donors=("shifted", "sum"),
                     sparse_shifted_donors=("sparse_shifted", "sum"),
                     shifted_dev_mass=("shifted_dev_mass", "sum"),
                     sparse_shifted_dev_mass=("sparse_shifted_dev_mass", "sum")))
    pair = (result.groupby(["home", "donor"], as_index=False)
            .agg(query_source_pred_rows=("query_source_pred_rows", "first"),
                 dev_source_pred_rows=("dev_source_pred_rows", "first"),
                 dev_to_query_source_pred_ratio=("dev_to_query_source_pred_ratio", "first"),
                 paid_source_pred_median=("paid_source_pred_rows", "median"),
                 dev_source_pred_share=("dev_source_pred_share", "first")))
    top = (pair.loc[(pair.dev_source_pred_rows >= 100) &
                    (pair.dev_to_query_source_pred_ratio >= 4)]
           .sort_values("dev_source_pred_share", ascending=False).head(25))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_seed_class.csv", index=False)
    per_seed.to_csv(output / "per_seed_summary.csv", index=False)
    pair.to_csv(output / "per_home_class.csv", index=False)
    report = "\n".join([
        "UNSW source-predicted donor coverage audit across reviewed homes; no challenger or new method fitted.",
        "Only the existing 1000 paid labels per seed are indexed; five seeds overlap within each home.",
        "Development labels only select the already curated 24-type population; no final-window labels/features decoded.",
        "Shift ratio compares source-predicted class shares, not true class prevalence or conditional error.",
        "Descriptive flags: >=100 development predictions, >=4x share growth, <10 paid donor rows.",
        "These cutoffs are not a proposed routing policy or a significance test.",
        "Per-home means over overlapping seeds:",
        per_seed.groupby("home")[["shifted_donors", "sparse_shifted_donors",
                                  "shifted_dev_mass", "sparse_shifted_dev_mass"]]
        .mean().round(4).to_string(),
        "Largest shifted donor groups by development share:",
        top.round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        homes=homes, archive_sha256=expected,
        implementation_sha256=sha256(Path(__file__)),
        paid_labels_per_seed=1000, final_window_read=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
