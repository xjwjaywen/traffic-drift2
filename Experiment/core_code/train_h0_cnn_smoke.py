"""
Supervised CNN smoke baseline for checking the H0 dataloader/split.

This is not a new H experiment. It is a diagnostic control: if the existing
CNN-style model cannot learn under the same short smoke setting, the NCDE smoke
result should not be used to judge the research direction.

Usage from Experiment/core_code/:
    python train_h0_cnn_smoke.py --config configs/h0_quic22_cnn_smoke.yaml
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.models import TTATCModel
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(batch, device):
    ppi = batch["ppi"].to(device)
    labels = batch["label"].to(device)
    flow_stats = batch.get("flow_stats")
    if flow_stats is not None:
        flow_stats = flow_stats.to(device)
    return ppi, labels, flow_stats


def train_epoch(model, dataloader, optimizer, device, epoch, max_batches=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"H0 CNN smoke epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        ppi, labels, flow_stats = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(ppi, flow_stats)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, desc="eval", max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        ppi, labels, flow_stats = _move_batch(batch, device)
        logits = model(ppi, flow_stats)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="H0 supervised CNN smoke diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    output_dir = args.output_dir or cfg.get("output_dir", "outputs/h0_quic22_cnn_smoke")
    os.makedirs(output_dir, exist_ok=True)

    device = choose_device(args.device or cfg.get("device"))
    print(f"Using device: {device}")
    print("Loading CESNET dataloaders...")
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg["data"])
    print(f"Num classes: {num_classes}")

    model_cfg = dict(cfg["model"])
    model_cfg["num_classes"] = num_classes
    model = TTATCModel(model_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    epochs = train_cfg.get("epochs", 5)
    max_train_batches = train_cfg.get("max_train_batches")
    max_val_batches = train_cfg.get("max_val_batches")

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, max_train_batches
        )
        val_metrics = evaluate(
            model, val_loader, device, desc=f"H0 CNN smoke val {epoch}", max_batches=max_val_batches
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_f1={row['train_macro_f1']:.4f} | "
            f"val_f1={row['val_macro_f1']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "num_classes": num_classes,
                    "config": cfg,
                },
                os.path.join(output_dir, "best_model.pt"),
            )

    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, desc="H0 CNN smoke test")

    results = {
        "experiment": "H0_CNN_smoke_diagnostic",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "num_classes": num_classes,
        "train_seconds": time.perf_counter() - start_time,
        "device": str(device),
    }
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nH0 CNN smoke complete")
    print(f"Best epoch: {best_epoch} | best_val_f1={best_val_f1:.4f}")
    print(f"Test accuracy={test_metrics['accuracy']:.4f} | test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    import sys
    import multiprocessing as _mp

    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
