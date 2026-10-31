"""Chronological CPSIoTSec'25 development baselines; never opens test_full.csv.

This is a port of the existing tabular TMA pilot controls, not a new method.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tma_pilot_v1"))
from data import Preprocessor
from existing_care import badge_selection, fit_controlled
from pilot import Model, forward_batches

LABEL = "deviceName"
TIME = "flowEndMilliseconds"
RF = dict(n_estimators=200, min_samples_leaf=2,
          class_weight="balanced_subsample", max_features="sqrt")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_train(path):
    frame = pd.read_csv(path)
    if LABEL not in frame or TIME not in frame or len(frame.columns) != 24:
        raise ValueError(f"Unexpected source schema: {path}")
    names = [c for c in frame.columns if c not in (LABEL, TIME)]
    times = pd.to_numeric(frame[TIME], errors="coerce").to_numpy()
    if not np.isfinite(times).all():
        raise ValueError(f"Invalid timestamp: {path}")
    order = np.argsort(times, kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    n = len(frame)
    a, b = int(.6 * n), int(.8 * n)
    if a < 100 or b - a < 1000 or n - b < 100:
        raise ValueError(f"Insufficient chronological rows: {path}")
    x = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    y = frame[LABEL].astype(str).to_numpy()
    classes = sorted(set(y[:a]))
    if len(classes) < 2:
        raise ValueError(f"Insufficient source classes: {path}")
    lookup = {name: i for i, name in enumerate(classes)}
    mapped = np.array([lookup.get(name, -1) for name in y], np.int64)
    return x, mapped, y, classes, names, (a, b)


def source_model(x, y, classes, *, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=rng)
        losses = []
        for ids in order.split(512):
            loss = torch.nn.functional.cross_entropy(
                model(tx[ids].to(device)), ty[ids].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite source loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"  source epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}", flush=True)
    return model.cpu().eval()


def hybrid_query(features, logits, budget, seed):
    n = len(features)
    if n < budget:
        raise ValueError("Query pool smaller than label budget")
    rng = np.random.default_rng(seed)
    anchor = rng.choice(n, budget // 2, replace=False)
    remaining = np.setdiff1d(np.arange(n), anchor, assume_unique=True)
    active = badge_selection(features[remaining], logits[remaining],
                             logits.shape[1], budget - len(anchor), seed).numpy()
    ids = np.concatenate((anchor, remaining[active]))
    if len(np.unique(ids)) != budget:
        raise ValueError("Repeated paid query")
    return ids


def evaluate(y, static, pred, class_names):
    n = len(class_names)
    labels = np.arange(n)
    _, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    correct = (y >= 0) & (static == y)
    present = support > 0
    summary = dict(
        evaluated_rows=len(y), known_rows=int((y >= 0).sum()),
        out_of_vocabulary_rows=int((y < 0).sum()), present_classes=int(present.sum()),
        macro_f1_present=float(f1[present].mean()) if present.any() else None,
        zero_recall_classes=int(((recall == 0) & present).sum()),
        negative_flip_rate_vs_static_mlp=float((pred[correct] != y[correct]).mean())
        if correct.any() else None,
    )
    rows = [dict(device_name=name, support=int(support[i]),
                 recall=float(recall[i]), f1=float(f1[i]))
            for i, name in enumerate(class_names)]
    return summary, rows


def run_home(root, output, home, args):
    train_file = root / home / "train.csv"
    if not train_file.is_file():
        raise FileNotFoundError(train_file)
    home_out = output / home
    x_raw, y, labels, classes, features, (a, b) = read_train(train_file)
    train_hash = sha256(train_file)
    config = dict(home=home, train_sha256=train_hash, source_seed=args.source_seed,
                  repair_seeds=args.seeds, source_epochs=args.epochs,
                  repair_steps=args.steps, label_budget=args.budget,
                  rf={**RF, "n_estimators": args.trees}, features=features,
                  source_classes=classes, split=[a, b, len(y)],
                  device=args.device, threads=args.threads,
                  implementation_sha256={name: sha256(path) for name, path in {
                      "development.py": Path(__file__),
                      "data.py": Path(__file__).resolve().parents[1] / "tma_pilot_v1/data.py",
                      "pilot.py": Path(__file__).resolve().parents[1] / "tma_pilot_v1/pilot.py",
                      "existing_care.py": Path(__file__).resolve().parents[1] / "tma_pilot_v1/existing_care.py",
                  }.items()},
                  selector="500 uniform anchor + 500 BADGE from remaining",
                  note="Development only; test_full.csv is never opened")
    marker = home_out / "complete.json"
    if marker.is_file():
        if json.loads((home_out / "config.json").read_text()) != config:
            raise ValueError(f"Completed configuration changed: {home_out}")
        expected = json.loads(marker.read_text())["artifacts"]
        if any(sha256(home_out / name) != digest for name, digest in expected.items()):
            raise ValueError(f"Completed artifact changed: {home_out}")
        print(home, "verified completed run reused", flush=True)
        return
    if home_out.exists() and any(home_out.iterdir()):
        raise ValueError(f"Incomplete output directory; use a new --output: {home_out}")
    prep = Preprocessor().fit(x_raw[:a])
    x = prep.transform(x_raw)
    source = source_model(x[:a], y[:a], len(classes),
                          seed=args.source_seed, epochs=args.epochs,
                          device=torch.device(args.device))
    query_features = forward_batches(source.encoder, x[a:b], args.device)
    dev_features = forward_batches(source.encoder, x[b:], args.device)
    query_logits = forward_batches(source.head, query_features.numpy(), args.device)
    dev_logits = forward_batches(source.head, dev_features.numpy(), args.device)
    static = dev_logits.argmax(1).numpy()
    rng = np.random.default_rng(args.source_seed + 10007)
    reference = np.concatenate([rng.choice(np.flatnonzero(y[:a] == c), 5, replace=False)
                                for c in range(len(classes))])
    reference_features = forward_batches(source.encoder, x[reference], args.device)
    teacher = forward_batches(source.head, reference_features.numpy(), args.device)
    home_out.mkdir(parents=True)
    np.savez_compressed(home_out / "split.npz", source_rows=np.arange(a),
                        query_rows=np.arange(a, b), dev_rows=np.arange(b, len(y)),
                        reference_rows=reference)
    split_counts = []
    for split, segment in (("source", y[:a]), ("query", y[a:b]),
                           ("development", y[b:])):
        for c, name in enumerate(classes):
            split_counts.append(dict(home=home, split=split, device_name=name,
                                     rows=int((segment == c).sum())))
    pd.DataFrame(split_counts).to_csv(home_out / "split_class_counts.csv", index=False)
    (home_out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summaries, class_rows, query_rows = [], [], []
    base_summary, base_classes = evaluate(y[b:], static, static, classes)
    summaries.append(dict(home=home, seed=-1, method="static_mlp", **base_summary))
    class_rows.extend(dict(home=home, seed=-1, method="static_mlp", **row)
                      for row in base_classes)
    for seed in args.seeds:
        ids = hybrid_query(query_features, query_logits, args.budget, seed)
        paid = y[a:b][ids]
        usable = ids[paid >= 0]
        for j, idx in enumerate(ids):
            query_rows.append(dict(home=home, seed=seed, query_order=j,
                                   train_row=int(a + idx), true_class=labels[a + idx],
                                   source_class=bool(paid[j] >= 0)))
        print(home, "seed", seed, "paid", len(ids), "known", len(usable), flush=True)
        preds = {"static_mlp": static}
        for variant in ("target", "care"):
            head, _ = fit_controlled(
                source.head, query_features[usable], y[a:b][usable],
                reference_features, y[reference], teacher,
                replay_ce=variant == "care", kd_weight=.5 if variant == "care" else 0.,
                temperature=2., target_weight=1., replay_weight=1.,
                steps=args.steps, batch_size=128, lr=1e-3, weight_decay=1e-4,
                seed=seed, device=torch.device(args.device), log_every=0)
            preds[variant] = forward_batches(head, dev_features.numpy(), args.device).argmax(1).numpy()
        forest = RandomForestClassifier(**{**RF, "n_estimators": args.trees},
                                        random_state=seed, n_jobs=args.threads)
        forest.fit(np.concatenate((query_features[usable].numpy(),
                                   reference_features.numpy())),
                   np.concatenate((y[a:b][usable], y[reference])))
        preds["frozen_rf"] = forest.predict(dev_features.numpy()).astype(np.int64)
        for method in ("target", "care", "frozen_rf"):
            summary, rows = evaluate(y[b:], static, preds[method], classes)
            summaries.append(dict(home=home, seed=seed, method=method,
                                  paid_labels=len(ids), usable_labels=len(usable), **summary))
            class_rows.extend(dict(home=home, seed=seed, method=method, **row)
                              for row in rows)
        np.savez_compressed(home_out / f"development_predictions_seed_{seed}.npz",
                            y=y[b:], **preds)
    pd.DataFrame(summaries).to_csv(home_out / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(home_out / "per_class.csv", index=False)
    pd.DataFrame(query_rows).to_csv(home_out / "queries.csv", index=False)
    (home_out / "complete.json").write_text(json.dumps(dict(
        config_sha256=sha256(home_out / "config.json"),
        artifacts={p.name: sha256(p) for p in home_out.iterdir()
                   if p.is_file() and p.name != "complete.json"}), indent=2) + "\n")
    print(home, "development complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/cps_iot_development_v1"))
    parser.add_argument("--home", choices=[f"home{i:02d}" for i in range(1, 13)])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()
    args.seeds = sorted(set(int(s) for s in args.seeds.split(",")))
    if not args.seeds or min(args.epochs, args.steps, args.budget, args.trees, args.threads) < 1:
        parser.error("Epochs, steps, budget, trees and threads must be positive")
    if args.budget != 1000:
        parser.error("Published development protocol fixes the label budget at 1000")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; choose --device cpu for a local smoke run")
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root.resolve(strict=True)
    for home in ([args.home] if args.home else [f"home{i:02d}" for i in range(1, 13)]):
        run_home(root, output, home, args)
    reports = [pd.read_csv(p) for p in sorted(output.glob("home*/summary.csv"))]
    table = pd.concat(reports, ignore_index=True)
    table.to_csv(output / "summary.csv", index=False)
    columns = ["macro_f1_present", "negative_flip_rate_vs_static_mlp",
               "zero_recall_classes"]
    home_means = table.groupby(["home", "method"])[columns].mean().reset_index()
    home_means.to_csv(output / "home_means.csv", index=False)
    means = home_means.groupby("method")[columns].mean()
    report = ("CPSIoTSec'25 chronological development baselines only.\n"
              "No test_full.csv file was opened. Homes are the independent units; "
              f"{len(args.seeds)} repair seeds are paired repeats.\n"
              "This port uses a newly trained tabular source MLP, not the paper CNN.\n"
              + "Across-home means:\n" + means.to_string() +
              "\n\nPer-home means:\n" + home_means.to_string(index=False) + "\n")
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
