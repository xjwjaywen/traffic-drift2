"""
TTA-TC Training Script.

Joint training with classification + SSL auxiliary tasks.

Usage:
    python train.py --config configs/train_quic22_cnn.yaml
"""
import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tta_tc.models import TTATCModel
from tta_tc.ssl_tasks.combined import CombinedSSLLoss
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


def train_epoch(model, ssl_loss_fn, dataloader, optimizer, device, ssl_weight, epoch):
    """Train one epoch with joint cls + SSL loss."""
    model.train()
    total_cls_loss = 0
    total_ssl_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        ppi = batch["ppi"].to(device)
        labels = batch["label"].to(device)
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)

        optimizer.zero_grad()

        # Classification forward
        logits = model(ppi, flow_stats)
        cls_loss = F.cross_entropy(logits, labels)

        # SSL forward
        ssl_loss, ssl_dict = ssl_loss_fn(model, ppi, flow_stats)

        # Combined loss
        loss = cls_loss + ssl_weight * ssl_loss
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_ssl_loss += ssl_loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

        pbar.set_postfix({
            "cls": f"{cls_loss.item():.4f}",
            "ssl": f"{ssl_loss.item():.4f}",
        })

    metrics = compute_metrics(all_labels, all_preds)
    return {
        "cls_loss": total_cls_loss / num_batches,
        "ssl_loss": total_ssl_loss / num_batches,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        ppi = batch["ppi"].to(device)
        labels = batch["label"].to(device)
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)

        logits = model(ppi, flow_stats)
        loss = F.cross_entropy(logits, labels)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu())
        total_loss += loss.item()
        num_batches += 1

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    metrics["logits"] = torch.cat(all_logits)
    metrics["labels"] = all_labels
    return metrics


def main():
    parser = argparse.ArgumentParser(description="TTA-TC Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/train")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.dump(cfg, f)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg["data"])
    cfg["model"]["num_classes"] = num_classes
    print(f"Num classes: {num_classes}")

    # Model
    model = TTATCModel(cfg["model"]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # SSL loss
    ssl_loss_fn = CombinedSSLLoss(
        alpha=cfg["ssl"].get("alpha", 0.2),
        beta=cfg["ssl"].get("beta", 0.1),
        mask_ratio=cfg["ssl"].get("mask_ratio", 0.15),
        enable_mpfp=cfg["ssl"].get("enable_mpfp", True),
        enable_pop=cfg["ssl"].get("enable_pop", True),
        enable_fsr=cfg["ssl"].get("enable_fsr", True),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("lr", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"].get("epochs", 50),
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, "tb_logs"))

    ssl_weight = cfg["training"].get("ssl_weight", 1.0)
    epochs = cfg["training"].get("epochs", 50)
    best_val_f1 = 0
    best_epoch = 0

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, ssl_loss_fn, train_loader, optimizer, device, ssl_weight, epoch
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        # Log
        writer.add_scalar("train/cls_loss", train_metrics["cls_loss"], epoch)
        writer.add_scalar("train/ssl_loss", train_metrics["ssl_loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save best
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "num_classes": num_classes,
                "config": cfg,
            }, os.path.join(output_dir, "best_model.pt"))

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }, os.path.join(output_dir, "latest_model.pt"))

    # Final test evaluation
    print(f"\nBest epoch: {best_epoch} (Val F1: {best_val_f1:.4f})")
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    # Save baseline entropy for drift detection
    import numpy as np
    val_metrics = evaluate(model, val_loader, device)
    val_logits = val_metrics["logits"]
    val_labels = val_metrics["labels"]
    probs = F.softmax(val_logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    baseline_entropy = np.zeros(num_classes)
    for c in range(num_classes):
        mask = np.array(val_labels) == c
        if mask.sum() > 0:
            baseline_entropy[c] = entropy[mask].mean().item()
    np.save(os.path.join(output_dir, "baseline_entropy.npy"), baseline_entropy)

    # Compute and save source class prototypes for SPA loss
    model.eval()
    proto_sum = torch.zeros(num_classes, cfg["model"]["hidden_dim"], device=device)
    proto_count = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for batch in val_loader:
            ppi = batch["ppi"].to(device)
            labels = batch["label"].to(device)
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(device)
            _, features = model(ppi, flow_stats, return_repr=True)
            for c in range(num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    proto_sum[c] += features[mask].sum(dim=0)
                    proto_count[c] += mask.sum()
    # Average; fall back to zero vector for unseen classes (safe — normalized later)
    proto_count = proto_count.clamp(min=1)
    class_prototypes = proto_sum / proto_count.unsqueeze(1)  # (C, hidden_dim)
    torch.save(class_prototypes.cpu(), os.path.join(output_dir, "class_prototypes.pt"))
    print(f"Class prototypes saved: shape {list(class_prototypes.shape)}")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "num_classes": num_classes,
    }
    with open(os.path.join(output_dir, "train_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\nTraining complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    # macOS requires 'fork' for cesnet-datazoo DataLoader; Windows only supports 'spawn'.
    import sys
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
