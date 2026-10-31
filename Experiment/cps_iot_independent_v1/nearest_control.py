"""Develop fixed 1-NN and three-paid-neighbor controls on CPS train.csv only.

The published test_full.csv is never opened. This is development evaluation
of controls proposed after viewing TMA, not independent confirmation.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tma_pilot_v1"))
from data import Preprocessor
from pilot import forward_batches
from development import evaluate, read_train, sha256, source_model


def nearest_and_consensus(query_features, dev_features, paid_indices, paid_y,
                          reference_features, reference_y, care, device):
    """Use the same cosine 1-NN and paid-target K=3 rule as the TMA control."""
    usable = paid_y >= 0
    if usable.sum() < 3:
        raise ValueError("Fewer than three paid source-vocabulary labels")
    paid_x = F.normalize(torch.as_tensor(query_features[paid_indices[usable]],
                                         dtype=torch.float32, device=device), dim=1)
    ref_x = F.normalize(torch.as_tensor(reference_features, dtype=torch.float32,
                                        device=device), dim=1)
    support = torch.cat((paid_x, ref_x))
    support_y = torch.as_tensor(np.concatenate((paid_y[usable], reference_y)),
                                dtype=torch.long, device=device)
    paid_labels = support_y[:len(paid_x)]
    nn_parts, accept_parts = [], []
    dev_x = F.normalize(torch.as_tensor(dev_features, dtype=torch.float32,
                                        device=device), dim=1)
    with torch.no_grad():
        for offset in range(0, len(dev_x), 4096):
            chunk = dev_x[offset:offset + 4096]
            cosine = chunk @ support.T
            nn = support_y[cosine.argmax(1)]
            paid_top = paid_labels[(chunk @ paid_x.T).topk(3, dim=1).indices]
            agreeing = (paid_top == paid_top[:, :1]).all(dim=1)
            accept = agreeing & (paid_top[:, 0] == nn)
            nn_parts.append(nn.cpu().numpy())
            accept_parts.append(accept.cpu().numpy())
    nearest = np.concatenate(nn_parts).astype(np.int64)
    accept = np.concatenate(accept_parts) & (nearest != care)
    routed = care.copy()
    routed[accept] = nearest[accept]
    return nearest, routed, accept


def completed(folder, config):
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    if json.loads((folder / "config.json").read_text()) != config:
        raise ValueError(f"Completed configuration changed: {folder}")
    artifacts = json.loads(marker.read_text())["artifacts"]
    if any(sha256(folder / name) != digest for name, digest in artifacts.items()):
        raise ValueError(f"Completed artifact changed: {folder}")
    return True


def run_home(dataset, baseline, output, home, device, threads):
    train_file = dataset / home / "train.csv"
    base_dir = baseline / home
    if not (base_dir / "complete.json").is_file():
        raise ValueError(f"Incomplete baseline: {base_dir}")
    base_config = json.loads((base_dir / "config.json").read_text())
    if sha256(train_file) != base_config["train_sha256"]:
        raise ValueError(f"Training data changed: {train_file}")
    if base_config["device"] != device or base_config["threads"] != threads:
        raise ValueError("Use the baseline's device and thread count for reproduction")
    config = dict(home=home, baseline_config_sha256=sha256(base_dir / "config.json"),
                  train_sha256=base_config["train_sha256"],
                  implementation_sha256=sha256(Path(__file__)),
                  methods="cosine_1nn_plus_5_source_refs_and_paid_target_3nn_consensus",
                  device=device, threads=threads)
    home_out = output / home
    if completed(home_out, config):
        print(home, "verified completed controls reused", flush=True)
        return
    if home_out.exists() and any(home_out.iterdir()):
        raise ValueError(f"Incomplete control output: {home_out}")
    x_raw, y, labels, classes, features, (a, b) = read_train(train_file)
    if (classes != base_config["source_classes"] or features != base_config["features"]
            or [a, b, len(y)] != base_config["split"]):
        raise ValueError("Chronological split differs from baseline")
    with np.load(base_dir / "split.npz", allow_pickle=False) as data:
        ref_ids = data["reference_rows"]
        if not (np.array_equal(data["source_rows"], np.arange(a))
                and np.array_equal(data["query_rows"], np.arange(a, b))
                and np.array_equal(data["dev_rows"], np.arange(b, len(y)))):
            raise ValueError("Saved split differs from baseline")
    prep = Preprocessor().fit(x_raw[:a])
    x = prep.transform(x_raw)
    source = source_model(x[:a], y[:a], len(classes),
                          seed=base_config["source_seed"],
                          epochs=base_config["source_epochs"],
                          device=torch.device(device))
    query_x = forward_batches(source.encoder, x[a:b], device).numpy()
    dev_x = forward_batches(source.encoder, x[b:], device).numpy()
    ref_x = forward_batches(source.encoder, x[ref_ids], device).numpy()
    static = forward_batches(source.head, dev_x, device).argmax(1).numpy()
    queries = pd.read_csv(base_dir / "queries.csv")
    seeds = base_config["repair_seeds"]
    summaries, per_class = [], []
    home_out.mkdir(parents=True)
    (home_out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for seed in seeds:
        with np.load(base_dir / f"development_predictions_seed_{seed}.npz",
                     allow_pickle=False) as saved:
            baseline_pred = {name: saved[name] for name in ("y", "static_mlp", "care")}
        if not np.array_equal(static, baseline_pred["static_mlp"]):
            raise ValueError(f"Source model did not reproduce baseline: {home} seed {seed}")
        if not np.array_equal(y[b:], baseline_pred["y"]):
            raise ValueError(f"Development labels differ: {home} seed {seed}")
        query = queries.loc[queries.seed == seed].sort_values("query_order")
        if (len(query) != base_config["label_budget"] or
                not np.array_equal(query.query_order.to_numpy(),
                                   np.arange(len(query)))):
            raise ValueError("Paid query order or count changed")
        ids = query.train_row.to_numpy(np.int64) - a
        if not ((ids >= 0) & (ids < b-a)).all():
            raise ValueError("Paid row lies outside query period")
        paid_y = y[a:b][ids]
        if not np.array_equal(query.true_class.astype(str).to_numpy(), labels[a:b][ids]):
            raise ValueError("Paid query labels changed")
        near, routed, accept = nearest_and_consensus(
            query_x, dev_x, ids, paid_y, ref_x, y[ref_ids],
            baseline_pred["care"], torch.device(device))
        for name, pred in (("nearest_anchor", near), ("consensus", routed)):
            summary, rows = evaluate(y[b:], static, pred, classes)
            summaries.append(dict(home=home, seed=seed, method=name,
                                  accepted=int(accept.sum()) if name == "consensus" else None,
                                  **summary))
            per_class.extend(dict(home=home, seed=seed, method=name, **row)
                             for row in rows)
        np.savez_compressed(home_out / f"predictions_seed_{seed}.npz",
                            nearest_anchor=near, consensus=routed, accept=accept)
        print(home, "seed", seed, "accepted", int(accept.sum()), flush=True)
    pd.DataFrame(summaries).to_csv(home_out / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(home_out / "per_class.csv", index=False)
    (home_out / "complete.json").write_text(json.dumps(dict(
        artifacts={p.name: sha256(p) for p in home_out.iterdir()
                   if p.is_file() and p.name != "complete.json"}), indent=2) + "\n")
    print(home, "development controls complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/cps_iot_development_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/cps_iot_nearest_control_v1"))
    parser.add_argument("--home", choices=[f"home{i:02d}" for i in range(1, 13)])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or (args.device.startswith("cuda") and not torch.cuda.is_available()):
        parser.error("Require positive threads and an available device")
    torch.set_num_threads(args.threads)
    dataset = args.dataset_root.resolve(strict=True)
    baseline = args.baseline.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for home in ([args.home] if args.home else [f"home{i:02d}" for i in range(1, 13)]):
        run_home(dataset, baseline, output, home, args.device, args.threads)
    controls = [pd.read_csv(p) for p in sorted(output.glob("home*/summary.csv"))]
    baselines = [pd.read_csv(p) for p in sorted(baseline.glob("home*/summary.csv"))]
    result = pd.concat(controls + baselines, ignore_index=True)
    methods = ["static_mlp", "care", "frozen_rf", "nearest_anchor", "consensus"]
    result = result.loc[result.method.isin(methods)]
    columns = ["macro_f1_present", "negative_flip_rate_vs_static_mlp",
               "zero_recall_classes"]
    home_means = result.groupby(["home", "method"], as_index=False)[columns].mean()
    overall = home_means.groupby("method")[columns].mean().reindex(methods)
    paired = home_means.pivot(index="home", columns="method", values=columns)
    deltas = pd.DataFrame({
        f"{method}_minus_care_{metric}": paired[(metric, method)] - paired[(metric, "care")]
        for method in ("nearest_anchor", "consensus") for metric in columns
    }).reset_index()
    home_means.to_csv(output / "home_means.csv", index=False)
    deltas.to_csv(output / "paired_home_deltas.csv", index=False)
    result.to_csv(output / "summary.csv", index=False)
    report = "\n".join([
        "CPSIoTSec'25 train.csv chronological development only; test_full.csv unopened.",
        "Same paid query identities and five source references per class as baseline.",
        "1-NN and paid-target K=3 consensus were fixed after TMA exploration.",
        "Homes are independent units; repair seeds are paired repeats.",
        "Across-home means:", overall.to_string(),
        "Paired home deltas versus CARE (positive macro F1, negative flip/zero-recall deltas preferred):",
        deltas.to_string(index=False),
        "Per-home means:", home_means.to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
