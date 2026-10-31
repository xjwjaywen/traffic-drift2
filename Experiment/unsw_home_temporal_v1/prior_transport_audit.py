"""Can paid, time-forward evidence predict prior-calibration benefit by home?

Diagnostic only. For each existing 1000-label query seed, fit the same prior
correction on the earlier 700 paid rows and score it on the later 300 paid
rows. Then check whether that signal predicts the already measured development
gain on homes held out from a tiny cross-home logistic model. No final data.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
from audit import sha256
from evaluate import PRIMARY, metrics
from prior_control import WEIGHT, PSEUDOCOUNT, prior_bias
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import Model


def forward_signal(logits, labels, archive_ids, predicted_count, query_rows, home, seed):
    """Fit on earlier selected rows; score only later selected rows."""
    order = np.argsort(archive_ids, kind="stable")
    train, holdout = order[:700], order[700:]
    if len(logits) != 1000 or len(labels) != 1000 or len(holdout) != 300:
        raise ValueError("Forward split requires the fixed 1000 paid queries")
    paid_count = np.bincount(labels[train], minlength=24)
    predicted_at_700 = predicted_count * (700. / query_rows)
    bias = (WEIGHT * (np.log(paid_count + PSEUDOCOUNT)
                      - np.log(predicted_at_700 + PSEUDOCOUNT))).astype(np.float32)
    static = logits[holdout].argmax(1).astype(np.uint8)
    calibrated = (logits[holdout] + bias[None, :]).argmax(1).astype(np.uint8)
    y = labels[holdout].astype(np.int8)
    before, _ = metrics(y, static, static, home, seed, "forward_static")
    after, _ = metrics(y, static, calibrated, home, seed, "forward_prior")
    return dict(home=home, seed=seed,
                forward_present_classes=before["present_classes"],
                forward_macro_f1_static=before["macro_f1_present"],
                forward_macro_f1_prior=after["macro_f1_present"],
                forward_macro_delta_pp=100 * (after["macro_f1_present"]
                                              - before["macro_f1_present"]),
                forward_nfr=after["negative_flip_rate_vs_static_mlp"],
                forward_static_correct=before["source_correct_rows"])


def leave_one_home_out(home_table):
    """Fit on ten independent home means, predict only the held-out home."""
    rows = []
    for home in home_table.home:
        train = home_table[home_table.home != home]
        held = home_table[home_table.home == home].iloc[0]
        x = train.forward_macro_delta_pp.to_numpy(float)
        y = (train.dev_macro_delta_pp > 0).to_numpy(int)
        if len(np.unique(y)) < 2:
            probability = float(y.mean())
        else:
            mean, std = x.mean(), max(float(x.std()), 1e-6)
            model = LogisticRegression(C=1., solver="lbfgs", random_state=0)
            model.fit(((x - mean) / std)[:, None], y)
            probability = float(model.predict_proba(
                np.array([[(held.forward_macro_delta_pp - mean) / std]]))[0, 1])
        rows.append(dict(home=home, loo_prior_probability=probability,
                         loo_choose_prior=probability > .5))
    return pd.DataFrame(rows)


def run(prepared, baseline, prior, development, output, device):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Audit output must be new and empty: {output}")
    dev = pd.read_csv(development / "summary.csv")
    expected = {(home, seed, method) for home in PRIMARY
                for seed in range(5) for method in ("prior_static",)}
    found = set(zip(dev.home, dev.seed, dev.method))
    if not expected.issubset(found):
        raise ValueError("Prior development summary is incomplete")
    rows = []
    for home in PRIMARY:
        prep_dir = prepared / home
        base_dir = baseline / home
        prior_dir = prior / "predictions" / home
        for folder, names in (
            (prep_dir, ("source.npz", "query.npz", "query_labels.npy")),
            (base_dir, ("source_state.pt",)),
            (prior_dir, ("query_predicted_count.npy",))):
            artifacts = json.loads((folder / "complete.json").read_text())["artifacts"]
            if any(sha256(folder / name) != artifacts[name] for name in names):
                raise ValueError(f"Changed audit input: {folder}")
        with np.load(prep_dir / "source.npz", allow_pickle=False) as data:
            source_x = data["x"]
        with np.load(prep_dir / "query.npz", allow_pickle=False) as data:
            query_x, query_ids = data["x"], data["ids"]
        query_labels = np.load(prep_dir / "query_labels.npy", mmap_mode="r")
        predicted_count = np.load(prior_dir / "query_predicted_count.npy")
        if predicted_count.shape != (24,) or predicted_count.sum() != len(query_x):
            raise ValueError(f"Unlabeled query count invalid: {home}")
        prep = Preprocessor().fit(source_x)
        model = Model(source_x.shape[1] * 2, 24)
        model.load_state_dict(torch.load(base_dir / "source_state.pt",
                                         map_location="cpu", weights_only=True))
        model.to(device).eval()
        for seed in range(5):
            seed_dir = base_dir / f"seed_{seed}"
            artifacts = json.loads((seed_dir / "complete.json").read_text())["artifacts"]
            name = "paid_query_pool_indices.npy"
            if sha256(seed_dir / name) != artifacts[name]:
                raise ValueError(f"Paid query indices changed: {home} {seed}")
            chosen = np.load(seed_dir / name)
            if chosen.shape != (1000,) or len(np.unique(chosen)) != 1000:
                raise ValueError(f"Invalid selected query set: {home} {seed}")
            x = prep.transform(query_x[chosen])
            with torch.no_grad():
                logits = model(torch.as_tensor(x, dtype=torch.float32,
                                               device=device)).cpu().numpy()
            labels = np.asarray(query_labels[chosen], dtype=np.int64)
            # Reproduce the already frozen full-1000 bias before the audit.
            full_bias = prior_bias(np.bincount(labels, minlength=24),
                                   predicted_count, len(query_x))
            saved_bias = np.load(prior_dir / f"bias_seed_{seed}.npy")
            if not np.allclose(full_bias, saved_bias, rtol=0, atol=1e-6):
                raise ValueError(f"Frozen prior bias mismatch: {home} {seed}")
            rows.append(forward_signal(logits, labels, query_ids[chosen],
                                       predicted_count, len(query_x), home, seed))
        model.cpu()
        print(home, "time-forward paid signal complete", flush=True)
    signal = pd.DataFrame(rows)
    prior_dev = dev[dev.method == "prior_static"][
        ["home", "seed", "macro_f1_present"]].rename(
            columns={"macro_f1_present": "dev_prior_f1"})
    static_dev = dev[dev.method == "static_mlp"][
        ["home", "macro_f1_present"]].rename(
            columns={"macro_f1_present": "dev_static_f1"})
    paired = signal.merge(prior_dev, on=["home", "seed"], validate="one_to_one")
    paired = paired.merge(static_dev, on="home", validate="many_to_one")
    paired["dev_macro_delta_pp"] = 100 * (paired.dev_prior_f1 - paired.dev_static_f1)
    home_table = paired.groupby("home", as_index=False)[
        ["forward_present_classes", "forward_macro_delta_pp", "forward_nfr",
         "dev_static_f1", "dev_prior_f1", "dev_macro_delta_pp"]].mean()
    loo = leave_one_home_out(home_table)
    home_table = home_table.merge(loo, on="home", validate="one_to_one")
    home_table["fixed_choose_prior"] = home_table.forward_macro_delta_pp > 0
    for kind in ("fixed", "loo"):
        choose = home_table[f"{kind}_choose_prior"]
        home_table[f"{kind}_routed_f1"] = np.where(
            choose, home_table.dev_prior_f1, home_table.dev_static_f1)
        home_table[f"{kind}_correct_direction"] = (
            choose == (home_table.dev_macro_delta_pp > 0))
    # The oracle uses development labels and is an upper bound, not a method.
    home_table["oracle_f1_contaminated"] = home_table[
        ["dev_static_f1", "dev_prior_f1"]].max(axis=1)
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "by_seed.csv", index=False)
    home_table.to_csv(output / "by_home.csv", index=False)
    columns = ["home", "forward_present_classes", "forward_macro_delta_pp",
               "dev_macro_delta_pp", "fixed_choose_prior",
               "loo_prior_probability", "loo_choose_prior",
               "fixed_correct_direction", "loo_correct_direction"]
    report = ("Exploratory query-to-development signal transport audit.\n"
              "Each signal fits on the earlier 700 of the same paid 1000 rows, "
              "scores the later 300; no extra paid target-home labels.\n"
              "LOO logistic uses the development outcomes of ten other homes "
              "to predict the held-out home; no held-out-home development "
              "label enters its choice. Seeds are repeats, not independent homes.\n"
              "Development outcomes were seen before designing this audit; "
              "this is not blind validation. Final-window data untouched.\n"
              + home_table[columns].round(3).to_string(index=False) + "\n"
              "Mean development macro F1 by home route:\n"
              + home_table[["dev_static_f1", "dev_prior_f1", "fixed_routed_f1",
                            "loo_routed_f1", "oracle_f1_contaminated"]]
              .mean().to_string() + "\n")
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        implementation_sha256=sha256(Path(__file__)),
        prior_summary_sha256=sha256(development / "summary.csv"),
        signal="earlier_700_to_later_300_paid_rows_by_archive_order",
        cross_home="leave_one_home_out_logistic_C_1_one_feature",
        final_window_read=False), indent=2) + "\n")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--prior", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_prior_control_v1"))
    parser.add_argument("--development", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_prior_control_v1/development"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_prior_transport_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    run(args.prepared.resolve(strict=True), args.baseline.resolve(strict=True),
        args.prior.resolve(strict=True), args.development.resolve(strict=True),
        args.output.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()
