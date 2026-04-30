"""
H2: Compare WST features vs raw PPI baseline on i.i.d. test data.

Three classifiers per representation:
  - Linear (logistic regression / SVM)
  - MLP (small)

Two representations:
  - Raw PPI (90-d flattened)
  - WST features (~192-d for J=3, Q=4)

Decision criterion (H2 pass):
  WST + linear  >=  PPI + linear     (WST captures discriminative info)
  WST + MLP     >=  PPI + MLP - 0.02 (WST not catastrophically worse)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "WST_validation"))

from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config
from methods.wst_extractor import make_wst, extract_wst_features, wst_feature_dim


# ----------------------------------------------------------- collectors --
@torch.no_grad()
def collect_ppi_and_labels(loader, device, max_samples=20000):
    ppis, labels = [], []
    n = 0
    for batch in tqdm(loader, desc="collect"):
        ppi = batch["ppi"]
        lbl = batch["label"]
        ppis.append(ppi)
        labels.append(lbl)
        n += ppi.size(0)
        if n >= max_samples:
            break
    ppis = torch.cat(ppis, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]
    return ppis.to(device), labels.to(device)


@torch.no_grad()
def featurize(ppis, mode="ppi", device="cuda"):
    """mode: 'ppi' (raw, flattened) or 'wst'."""
    if mode == "ppi":
        return ppis.flatten(1)  # (N, 3*T)
    elif mode == "wst":
        T = ppis.shape[-1]
        # Pad to next power of 2 if needed
        next_pow2 = 1
        while next_pow2 < T:
            next_pow2 *= 2
        sc = make_wst(seq_len=next_pow2, J=3, Q=4).to(device)
        # Process in chunks to avoid OOM
        feats = []
        chunk = 2048
        for s in range(0, ppis.size(0), chunk):
            e = min(s + chunk, ppis.size(0))
            f = extract_wst_features(ppis[s:e], scattering=sc)
            feats.append(f)
        return torch.cat(feats, dim=0)
    raise ValueError(mode)


# ----------------------------------------------------------- classifiers --
class LinearClf(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPClf(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_clf(model, X_train, y_train, X_test, y_test,
              epochs=30, lr=1e-3, batch_size=256, device="cuda"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    n_train = X_train.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            b = perm[s:s + batch_size]
            logits = model(X_train[b])
            loss = crit(logits, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        # Test in chunks
        all_preds = []
        for s in range(0, X_test.size(0), 4096):
            e = min(s + 4096, X_test.size(0))
            preds = model(X_test[s:e]).argmax(1)
            all_preds.append(preds)
        preds = torch.cat(all_preds)
    acc = (preds == y_test).float().mean().item()
    return acc


# ----------------------------------------------------------- main --
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Cap train + test samples")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["data"]["dataset"]
    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "WST_validation",
                                        "outputs", "h2_iid")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        cfg["data"])
    print(f"num_classes = {num_classes}")

    print("\n[collect train]")
    X_train_raw, y_train = collect_ppi_and_labels(
        train_loader, device, max_samples=args.max_samples)
    print(f"  shape: {X_train_raw.shape}")

    print("[collect test]")
    X_test_raw, y_test = collect_ppi_and_labels(
        test_loader, device, max_samples=args.max_samples)
    print(f"  shape: {X_test_raw.shape}")

    results = {"dataset": dataset, "num_classes": num_classes,
               "n_train": X_train_raw.size(0), "n_test": X_test_raw.size(0),
               "settings": {}}

    for repr_name in ("ppi", "wst"):
        print(f"\n=== Representation: {repr_name} ===")
        t0 = time.time()
        X_train = featurize(X_train_raw, mode=repr_name, device=device)
        X_test = featurize(X_test_raw, mode=repr_name, device=device)
        feat_t = time.time() - t0
        in_dim = X_train.size(1)
        print(f"  feature dim: {in_dim}, featurize time: {feat_t:.1f}s")

        for clf_name in ("linear", "mlp"):
            print(f"\n  -- Classifier: {clf_name} --")
            if clf_name == "linear":
                model = LinearClf(in_dim, num_classes)
            else:
                model = MLPClf(in_dim, num_classes)
            t0 = time.time()
            acc = train_clf(model, X_train, y_train, X_test, y_test,
                            epochs=args.epochs, device=device)
            train_t = time.time() - t0
            print(f"  acc: {acc:.4f}  train time: {train_t:.1f}s")
            results["settings"][f"{repr_name}+{clf_name}"] = {
                "acc": acc,
                "feat_dim": in_dim,
                "feat_time_s": feat_t,
                "train_time_s": train_t,
            }

    out_path = os.path.join(args.output_dir, f"{dataset}_h2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Verdict
    print("\n" + "=" * 70)
    print(f"H2 Verdict — {dataset}")
    print("=" * 70)
    s = results["settings"]
    ppi_lin = s["ppi+linear"]["acc"]
    wst_lin = s["wst+linear"]["acc"]
    ppi_mlp = s["ppi+mlp"]["acc"]
    wst_mlp = s["wst+mlp"]["acc"]
    print(f"  PPI + linear:  {ppi_lin:.4f}")
    print(f"  WST + linear:  {wst_lin:.4f}    delta = {wst_lin - ppi_lin:+.4f}")
    print(f"  PPI + MLP:     {ppi_mlp:.4f}")
    print(f"  WST + MLP:     {wst_mlp:.4f}    delta = {wst_mlp - ppi_mlp:+.4f}")
    if wst_lin >= ppi_lin and wst_mlp >= ppi_mlp - 0.02:
        verdict = "PASS — WST captures discriminative info; H2 confirmed"
    elif wst_lin >= ppi_lin - 0.05 and wst_mlp >= ppi_mlp - 0.05:
        verdict = "PARTIAL — WST competitive, but no clear advantage"
    else:
        verdict = "FAIL — WST features substantially worse than raw PPI"
    print(f"\n  ==> {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
