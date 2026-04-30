"""
H3: Compare WST features vs raw PPI under temporal drift.

Train on source period, test on multiple later periods, measure
accuracy drop. WST should drop less if Mallat stability theorem
applies in practice.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "WST_validation"))

from tta_tc.data.cesnet_loader import (
    build_dataloaders, build_sequential_test_loaders)
from tta_tc.utils.config import load_config

from scripts.h2_iid_compare import (
    collect_ppi_and_labels, featurize, train_clf, MLPClf, LinearClf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-test-per-period", type=int, default=5000)
    parser.add_argument("--clf", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["data"]["dataset"]
    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "WST_validation",
                                        "outputs", "h3_drift")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load source training data
    train_loader, _, _, num_classes = build_dataloaders(cfg["data"])
    print(f"num_classes = {num_classes}")

    print("\n[collect source training data]")
    X_train_raw, y_train = collect_ppi_and_labels(
        train_loader, device, max_samples=args.max_train)
    print(f"  shape: {X_train_raw.shape}")

    # Featurize once
    print("\n[featurize PPI]")
    X_train_ppi = featurize(X_train_raw, mode="ppi", device=device)
    print("[featurize WST]")
    X_train_wst = featurize(X_train_raw, mode="wst", device=device)
    print(f"  PPI dim: {X_train_ppi.size(1)}, WST dim: {X_train_wst.size(1)}")

    # Train two classifiers (one per representation)
    ClfClass = MLPClf if args.clf == "mlp" else LinearClf
    print(f"\n[train PPI + {args.clf}]")
    ppi_clf = ClfClass(X_train_ppi.size(1), num_classes).to(device)
    opt = torch.optim.Adam(ppi_clf.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    n_train = X_train_ppi.size(0)
    for ep in range(args.epochs):
        ppi_clf.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, 256):
            b = perm[s:s + 256]
            loss = crit(ppi_clf(X_train_ppi[b]), y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    ppi_clf.eval()

    print(f"[train WST + {args.clf}]")
    wst_clf = ClfClass(X_train_wst.size(1), num_classes).to(device)
    opt = torch.optim.Adam(wst_clf.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        wst_clf.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, 256):
            b = perm[s:s + 256]
            loss = crit(wst_clf(X_train_wst[b]), y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    wst_clf.eval()

    # Loop through test periods
    test_loaders, _ = build_sequential_test_loaders(cfg["data"])

    results = {"dataset": dataset, "num_classes": num_classes,
               "clf": args.clf,
               "ppi_dim": int(X_train_ppi.size(1)),
               "wst_dim": int(X_train_wst.size(1)),
               "periods": {}}

    print("\n" + "=" * 76)
    print(f"H3 — Drift evaluation ({dataset}, classifier: {args.clf})")
    print("=" * 76)
    print(f"{'Period':<14}{'PPI acc':>10}{'WST acc':>10}{'WST-PPI':>12}")
    print("-" * 76)

    for period_name, loader in test_loaders:
        X_test_raw, y_test = collect_ppi_and_labels(
            loader, device, max_samples=args.max_test_per_period)
        X_test_ppi = featurize(X_test_raw, mode="ppi", device=device)
        X_test_wst = featurize(X_test_raw, mode="wst", device=device)
        with torch.no_grad():
            ppi_acc = (ppi_clf(X_test_ppi).argmax(1) == y_test).float().mean().item()
            wst_acc = (wst_clf(X_test_wst).argmax(1) == y_test).float().mean().item()
        delta = wst_acc - ppi_acc
        print(f"{period_name:<14}{ppi_acc:>10.4f}{wst_acc:>10.4f}{delta:>+12.4f}")
        results["periods"][period_name] = {
            "ppi_acc": ppi_acc, "wst_acc": wst_acc, "delta": delta,
            "n_test": int(X_test_raw.size(0)),
        }

    print("=" * 76)

    # Compute drop from first to last period
    period_keys = list(results["periods"].keys())
    if len(period_keys) >= 2:
        first = results["periods"][period_keys[0]]
        last = results["periods"][period_keys[-1]]
        ppi_drop = first["ppi_acc"] - last["ppi_acc"]
        wst_drop = first["wst_acc"] - last["wst_acc"]
        rel_diff = ppi_drop - wst_drop
        print(f"\nDrop ({period_keys[0]} -> {period_keys[-1]}):")
        print(f"  PPI: {ppi_drop:+.4f}")
        print(f"  WST: {wst_drop:+.4f}")
        print(f"  WST is {rel_diff:+.4f} more robust than PPI")

        if rel_diff >= 0.05:
            verdict = "PASS — WST significantly more drift-robust"
        elif rel_diff >= 0.02:
            verdict = "WEAK PASS — WST slightly more drift-robust"
        elif rel_diff >= -0.02:
            verdict = "PARTIAL — WST and PPI roughly equally robust"
        else:
            verdict = "FAIL — WST is LESS drift-robust than PPI"
        results["drift_verdict"] = verdict
        results["ppi_drop"] = ppi_drop
        results["wst_drop"] = wst_drop
        print(f"\n==> {verdict}")

    out_path = os.path.join(args.output_dir, f"{dataset}_h3_{args.clf}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
