"""
H0 feasibility experiment: NCDE baseline on CESNET-QUIC22 W-44.

Usage from Experiment/core_code/:
    python train_h0_ncde.py --config configs/h0_quic22_ncde.yaml
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
from tta_tc.models.ncde import NeuralCDEClassifier
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


def train_epoch(model, dataloader, optimizer, device, epoch, grad_clip=None, max_batches=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"H0 NCDE epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        ppi = batch["ppi"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(ppi)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        ppi = batch["ppi"].to(device)
        labels = batch["label"].to(device)
        logits = model(ppi)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


@torch.no_grad()
def estimate_ppi_channel_stats(dataloader, device, max_batches=None):
    """Estimate channel-wise mean/std over PPI tensors shaped (B, 3, T)."""
    total = torch.zeros(3, device=device)
    total_sq = torch.zeros(3, device=device)
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Estimating PPI stats")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        ppi = batch["ppi"].to(device)
        flat = ppi.transpose(1, 2).reshape(-1, ppi.size(1))
        total += flat.sum(dim=0)
        total_sq += (flat * flat).sum(dim=0)
        count += flat.size(0)

    mean = total / max(count, 1)
    var = (total_sq / max(count, 1)) - mean * mean
    std = var.clamp_min(1e-12).sqrt()
    return mean, std


def load_cnn_reference(comparison_cfg):
    if not comparison_cfg:
        return None
    path = comparison_cfg.get("cnn_results_path")
    metric = comparison_cfg.get("cnn_metric", "best_val_f1")
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if metric not in data:
        return None
    return {
        "path": path,
        "metric": metric,
        "value": float(data[metric]),
    }


def build_comparison(ncde_results, comparison_cfg):
    reference = load_cnn_reference(comparison_cfg)
    if reference is None:
        return None

    ncde_metric = comparison_cfg.get("ncde_metric", "best_val_f1")
    pass_within = float(comparison_cfg.get("pass_within", 0.03))
    stop_if_lower_than = float(comparison_cfg.get("stop_if_lower_than", 0.05))
    ncde_value = float(ncde_results[ncde_metric])
    delta = ncde_value - reference["value"]

    return {
        "cnn_reference": reference,
        "ncde_metric": ncde_metric,
        "ncde_value": ncde_value,
        "delta": delta,
        "abs_delta": abs(delta),
        "within_3pct": abs(delta) <= pass_within,
        "h0_pass": delta >= -pass_within,
        "stop_direction": delta < -stop_if_lower_than,
        "criteria": {
            "pass_within": pass_within,
            "stop_if_lower_than": stop_if_lower_than,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="H0 NCDE feasibility experiment")
    parser.add_argument("--config", required=True, help="Path to H0 YAML config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug override for train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug override for validation batches")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    output_dir = args.output_dir or cfg.get("output_dir", "outputs/h0_quic22_ncde")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(cfg, f, sort_keys=False)

    device = choose_device(args.device or cfg.get("device"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Using device: {device}")
    print("Loading CESNET dataloaders...")
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg["data"])
    print(f"Num classes: {num_classes}")

    model_cfg = cfg.get("model", {})
    model = NeuralCDEClassifier(
        input_channels=model_cfg.get("input_channels", 3),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_classes=num_classes,
        interpolation=model_cfg.get("interpolation", "linear"),
        solver=model_cfg.get("solver", "rk4"),
        solver_step_size=model_cfg.get("solver_step_size", 1.0),
        normalize_input=model_cfg.get("normalize_input", False),
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    norm_cfg = cfg.get("normalization", {})
    if model_cfg.get("normalize_input", False):
        mean, std = estimate_ppi_channel_stats(
            train_loader,
            device=device,
            max_batches=norm_cfg.get("max_batches"),
        )
        model.set_input_stats(mean, std)
        print(f"PPI channel mean: {[round(x, 6) for x in mean.detach().cpu().tolist()]}")
        print(f"PPI channel std : {[round(x, 6) for x in std.detach().cpu().tolist()]}")

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    epochs = train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    start_time = time.perf_counter()
    max_train_batches = (
        args.max_train_batches
        if args.max_train_batches is not None
        else train_cfg.get("max_train_batches")
    )
    max_val_batches = (
        args.max_val_batches
        if args.max_val_batches is not None
        else train_cfg.get("max_val_batches")
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, grad_clip, max_train_batches
        )
        val_metrics = evaluate(
            model, val_loader, device, desc=f"H0 NCDE val {epoch}", max_batches=max_val_batches
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "num_classes": num_classes,
                    "config": cfg,
                },
                os.path.join(output_dir, "best_model.pt"),
            )

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            os.path.join(output_dir, "latest_model.pt"),
        )

    train_seconds = time.perf_counter() - start_time
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, desc="H0 NCDE test")

    results = {
        "experiment": "H0_NCDE_feasibility",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "num_classes": num_classes,
        "train_seconds": train_seconds,
        "device": str(device),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        ),
    }
    comparison = build_comparison(results, cfg.get("comparison", {}))
    if comparison is not None:
        results["comparison"] = comparison

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nH0 NCDE complete")
    print(f"Best epoch: {best_epoch} | best_val_f1={best_val_f1:.4f}")
    print(f"Test accuracy={test_metrics['accuracy']:.4f} | test_macro_f1={test_metrics['macro_f1']:.4f}")
    if comparison is not None:
        print(
            "CNN comparison: "
            f"NCDE {comparison['ncde_metric']}={comparison['ncde_value']:.4f}, "
            f"CNN {comparison['cnn_reference']['metric']}={comparison['cnn_reference']['value']:.4f}, "
            f"delta={comparison['delta']:.4f}, "
            f"h0_pass={comparison['h0_pass']}"
        )
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
