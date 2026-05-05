"""
Minimal temporal-invariance training validation for TLS-Year22.

This script tests the core route-B hypothesis:
  pooled ERM on multiple historical months
    vs.
  pooled ERM + class-level temporal prototype invariance.

The temporal loss is computed on a multi-period training step. Each step draws
one batch from every training period, concatenates them for CE, and aligns
same-class prototypes across periods inside the step:

    L = L_ce + lambda_temporal * sum_c mean_t ||norm(p_c^t)-norm(p_c^global)||^2

Usage from Experiment/core_code/:
    python scripts/train_temporal_invariance_tls22.py \
      --config configs/train_tls22_cnn.yaml \
      --method temporal_proto \
      --train-periods M-2022-1 M-2022-2 M-2022-3 M-2022-4 M-2022-5 M-2022-6 \
      --test-periods M-2022-7 M-2022-8 M-2022-9 M-2022-10 M-2022-11 M-2022-12 \
      --output-dir outputs/titc_tls22_temporal_proto
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.models import TTATCModel
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


DEFAULT_COLLAPSE_CLASSES = [
    56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26
]
DEFAULT_STABLE_CLASSES = [
    8, 15, 44, 57, 59, 62, 64, 76, 94, 98,
    99, 107, 113, 119, 128, 130, 131, 132, 144, 145,
]


def parse_periods(value):
    if isinstance(value, list):
        return value
    return [x for x in value.replace(",", " ").split() if x]


def parse_class_list(value, default):
    if value is None or value.strip() == "":
        return list(default)
    return [int(x) for x in value.replace(",", " ").split()]


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_period_loaders(cfg, periods, split, batch_size=None, label_anchor_period=None):
    """Build per-period loaders under one stable label mapping.

    For historical/future periods we keep ``train_period`` fixed to
    ``label_anchor_period`` and vary only ``test_period``. DataZoo derives the
    known-app label space from ``train_period``; changing it per month can
    silently change class ids and make training look random.
    """
    loaders = []
    class_counts = []
    anchor = label_anchor_period or cfg["data"].get("train_period")
    for period in periods:
        data_cfg = dict(cfg["data"])
        data_cfg["train_period"] = anchor if label_anchor_period else period
        data_cfg["test_period"] = period
        if batch_size is not None:
            data_cfg["batch_size"] = batch_size
        train_loader, val_loader, test_loader, n_cls = build_dataloaders(data_cfg)
        class_counts.append(int(n_cls))
        selected = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
        loaders.append((period, selected))
    num_classes = max(class_counts)
    if len(set(class_counts)) > 1:
        pairs = ", ".join(
            f"{period}:{count}" for period, count in zip(periods, class_counts)
        )
        print(
            "WARNING: DataZoo returned different class counts across periods "
            f"({pairs}); using max num_classes={num_classes}."
        )
    return loaders, num_classes


def move_batch(batch, device):
    out = {
        "ppi": batch["ppi"].to(device),
        "label": batch["label"].to(device),
    }
    flow_stats = batch.get("flow_stats")
    if flow_stats is not None:
        out["flow_stats"] = flow_stats.to(device)
    return out


def concat_period_batches(period_batches):
    ppi = torch.cat([b["ppi"] for _, b in period_batches], dim=0)
    labels = torch.cat([b["label"] for _, b in period_batches], dim=0)
    period_ids = []
    for period_idx, (_, batch) in enumerate(period_batches):
        period_ids.append(
            torch.full(
                (batch["label"].shape[0],),
                period_idx,
                dtype=torch.long,
                device=batch["label"].device,
            )
        )
    period_ids = torch.cat(period_ids, dim=0)

    flow_stats_values = [b.get("flow_stats") for _, b in period_batches]
    if all(v is not None for v in flow_stats_values):
        flow_stats = torch.cat(flow_stats_values, dim=0)
    else:
        flow_stats = None
    return ppi, labels, period_ids, flow_stats


def temporal_prototype_loss(features, labels, period_ids, num_periods, min_samples):
    losses = []
    stats = {
        "temporal_classes": 0,
        "temporal_proto_terms": 0,
    }
    for class_id in torch.unique(labels).tolist():
        period_protos = []
        for period_idx in range(num_periods):
            mask = (labels == class_id) & (period_ids == period_idx)
            if int(mask.sum().item()) >= min_samples:
                period_protos.append(features[mask].mean(dim=0))
        if len(period_protos) < 2:
            continue
        proto = F.normalize(torch.stack(period_protos, dim=0), dim=1)
        global_proto = F.normalize(proto.mean(dim=0, keepdim=True), dim=1)
        losses.append((proto - global_proto).pow(2).sum(dim=1).mean())
        stats["temporal_classes"] += 1
        stats["temporal_proto_terms"] += len(period_protos)
    if not losses:
        return features.new_tensor(0.0), stats
    return torch.stack(losses).mean(), stats


def group_macro_f1(report, classes):
    values = []
    support = 0
    for class_id in classes:
        item = report.get(str(class_id), {})
        values.append(float(item.get("f1-score", 0.0)))
        support += int(item.get("support", 0))
    return float(np.mean(values)) if values else None, support


def collapse_recall_counts(report, classes, recall_threshold, severe_threshold):
    collapsed = 0
    severe = 0
    recovered = 0
    per_class = []
    for class_id in classes:
        item = report.get(str(class_id), {})
        support = int(item.get("support", 0))
        recall = float(item.get("recall", 0.0))
        f1 = float(item.get("f1-score", 0.0))
        if recall < recall_threshold:
            collapsed += 1
        else:
            recovered += 1
        if recall < severe_threshold:
            severe += 1
        per_class.append({
            "class_id": class_id,
            "support": support,
            "recall": recall,
            "f1": f1,
        })
    return collapsed, severe, recovered, per_class


@torch.no_grad()
def evaluate_loader(model, loader, device, collapse_classes, stable_classes, thresholds):
    model.eval()
    labels_all = []
    preds_all = []
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["ppi"], batch.get("flow_stats"))
        loss = F.cross_entropy(logits, batch["label"])
        labels_all.extend(batch["label"].cpu().numpy())
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        total_loss += float(loss.item())
        num_batches += 1
    metrics = compute_metrics(labels_all, preds_all)
    report = metrics["classification_report"]
    collapse_f1, collapse_support = group_macro_f1(report, collapse_classes)
    stable_f1, stable_support = group_macro_f1(report, stable_classes)
    collapsed, severe, recovered, per_class = collapse_recall_counts(
        report,
        collapse_classes,
        thresholds["collapse"],
        thresholds["severe"],
    )
    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "collapse_macro_f1": collapse_f1,
        "collapse_support": collapse_support,
        "stable_macro_f1": stable_f1,
        "stable_support": stable_support,
        "collapsed_count": collapsed,
        "severe_collapsed_count": severe,
        "recovered_count": recovered,
        "per_collapse_class": per_class,
    }


def evaluate_periods(model, period_loaders, device, collapse_classes, stable_classes, thresholds):
    rows = []
    per_class_rows = []
    for period, loader in period_loaders:
        metrics = evaluate_loader(
            model, loader, device, collapse_classes, stable_classes, thresholds
        )
        rows.append({
            "period": period,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "collapse_macro_f1": metrics["collapse_macro_f1"],
            "collapse_support": metrics["collapse_support"],
            "stable_macro_f1": metrics["stable_macro_f1"],
            "stable_support": metrics["stable_support"],
            "collapsed_count": metrics["collapsed_count"],
            "severe_collapsed_count": metrics["severe_collapsed_count"],
            "recovered_count": metrics["recovered_count"],
        })
        for row in metrics["per_collapse_class"]:
            per_class_rows.append({"period": period, **row})
    return rows, per_class_rows


def train_one_epoch(
    model,
    period_loaders,
    optimizer,
    device,
    method,
    lambda_temporal,
    min_proto_samples,
    epoch,
    max_steps,
):
    model.train()
    num_periods = len(period_loaders)
    iters = [(period, loader, iter(loader)) for period, loader in period_loaders]
    steps = max_steps or min(len(loader) for _, loader in period_loaders)
    rows = []
    running = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "temporal_loss": 0.0,
        "temporal_classes": 0.0,
    }

    pbar = tqdm(range(steps), desc=f"Epoch {epoch}")
    for _ in pbar:
        period_batches = []
        next_iters = []
        for period, loader, iterator in iters:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            period_batches.append((period, move_batch(batch, device)))
            next_iters.append((period, loader, iterator))
        iters = next_iters
        ppi, labels, period_ids, flow_stats = concat_period_batches(period_batches)

        optimizer.zero_grad()
        logits, features = model(ppi, flow_stats, return_repr=True)
        ce_loss = F.cross_entropy(logits, labels)
        if method == "temporal_proto":
            temp_loss, temp_stats = temporal_prototype_loss(
                features, labels, period_ids, num_periods, min_proto_samples
            )
        else:
            temp_loss = features.new_tensor(0.0)
            temp_stats = {"temporal_classes": 0, "temporal_proto_terms": 0}
        loss = ce_loss + lambda_temporal * temp_loss
        loss.backward()
        optimizer.step()

        batch_row = {
            "loss": float(loss.item()),
            "ce_loss": float(ce_loss.item()),
            "temporal_loss": float(temp_loss.item()),
            **temp_stats,
        }
        rows.append(batch_row)
        for key in running:
            running[key] += batch_row.get(key, 0.0)
        pbar.set_postfix({
            "loss": f"{batch_row['loss']:.4f}",
            "ce": f"{batch_row['ce_loss']:.4f}",
            "tmp": f"{batch_row['temporal_loss']:.4f}",
            "cls": temp_stats["temporal_classes"],
        })

    return {key: value / max(steps, 1) for key, value in running.items()}, rows


def save_checkpoint(path, model, optimizer, cfg, epoch, best_score, num_classes):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_macro_f1": best_score,
        "num_classes": num_classes,
        "config": cfg,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["pooled_erm", "temporal_proto"], required=True)
    parser.add_argument(
        "--train-periods",
        nargs="+",
        default=["M-2022-1", "M-2022-2", "M-2022-3", "M-2022-4", "M-2022-5", "M-2022-6"],
    )
    parser.add_argument(
        "--test-periods",
        nargs="+",
        default=["M-2022-7", "M-2022-8", "M-2022-9", "M-2022-10", "M-2022-11", "M-2022-12"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional train.py checkpoint used to warm-start drift-aware training.",
    )
    parser.add_argument("--per-period-batch-size", type=int, default=256)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--lambda-temporal", type=float, default=0.1)
    parser.add_argument("--min-proto-samples", type=int, default=2)
    parser.add_argument(
        "--label-anchor-period",
        default=None,
        help=(
            "Period used by DataZoo to define the known-app label mapping. "
            "Defaults to data.train_period from the config."
        ),
    )
    parser.add_argument("--collapse-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    train_periods = parse_periods(args.train_periods)
    test_periods = parse_periods(args.test_periods)
    label_anchor_period = args.label_anchor_period or cfg["data"].get("train_period")
    collapse_classes = parse_class_list(args.collapse_classes, DEFAULT_COLLAPSE_CLASSES)
    stable_classes = parse_class_list(args.stable_classes, DEFAULT_STABLE_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Method: {args.method}")
    print(f"Label anchor period: {label_anchor_period}")
    print(f"Train periods: {' '.join(train_periods)}")
    print(f"Test periods: {' '.join(test_periods)}")
    print(f"Per-period batch size: {args.per_period_batch_size}")

    print("Building train loaders...")
    train_loaders, num_classes = make_period_loaders(
        cfg,
        train_periods,
        "test",
        batch_size=args.per_period_batch_size,
        label_anchor_period=label_anchor_period,
    )
    print("Building validation loaders...")
    val_loaders, _ = make_period_loaders(
        cfg,
        [label_anchor_period],
        "val",
        batch_size=args.per_period_batch_size,
        label_anchor_period=label_anchor_period,
    )
    print("Building test loaders...")
    test_loaders, _ = make_period_loaders(
        cfg,
        test_periods,
        "test",
        batch_size=args.per_period_batch_size,
        label_anchor_period=label_anchor_period,
    )

    cfg["model"]["num_classes"] = num_classes
    model = TTATCModel(cfg["model"]).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        ckpt_num_classes = int(checkpoint.get("num_classes", num_classes))
        if ckpt_num_classes != num_classes:
            raise RuntimeError(
                f"Checkpoint num_classes={ckpt_num_classes} does not match "
                f"loader num_classes={num_classes}. Check label_anchor_period."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized model from checkpoint: {args.init_checkpoint}")
    print(f"Num classes: {num_classes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    epochs = args.epochs or cfg["training"].get("epochs", 30)
    lr = args.lr if args.lr is not None else cfg["training"].get("lr", 1e-3)
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else cfg["training"].get("weight_decay", 1e-4)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    thresholds = {
        "collapse": args.collapse_recall_threshold,
        "severe": args.severe_recall_threshold,
    }

    run_cfg = {
        "method": args.method,
        "label_anchor_period": label_anchor_period,
        "train_periods": train_periods,
        "test_periods": test_periods,
        "per_period_batch_size": args.per_period_batch_size,
        "init_checkpoint": args.init_checkpoint,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "lambda_temporal": args.lambda_temporal,
        "min_proto_samples": args.min_proto_samples,
        "collapse_classes": collapse_classes,
        "stable_classes": stable_classes,
        "collapse_recall_threshold": args.collapse_recall_threshold,
        "severe_recall_threshold": args.severe_recall_threshold,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    best_val_f1 = -1.0
    epoch_rows = []
    best_path = os.path.join(args.output_dir, "best_model.pt")
    max_steps = args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else None

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_metrics, _ = train_one_epoch(
            model,
            train_loaders,
            optimizer,
            device,
            args.method,
            args.lambda_temporal if args.method == "temporal_proto" else 0.0,
            args.min_proto_samples,
            epoch,
            max_steps,
        )
        scheduler.step()
        val_rows, _ = evaluate_periods(
            model, val_loaders, device, collapse_classes, stable_classes, thresholds
        )
        val_macro = float(np.mean([row["macro_f1"] for row in val_rows]))
        val_collapse = float(np.mean([row["collapse_macro_f1"] for row in val_rows]))
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_macro_f1_mean": val_macro,
            "val_collapse_macro_f1_mean": val_collapse,
            "lr": scheduler.get_last_lr()[0],
        }
        epoch_rows.append(row)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['ce_loss']:.4f} tmp={train_metrics['temporal_loss']:.4f} "
            f"val_f1={val_macro:.4f} val_collapse_f1={val_collapse:.4f}"
        )
        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            save_checkpoint(best_path, model, optimizer, run_cfg, epoch, best_val_f1, num_classes)
        save_checkpoint(
            os.path.join(args.output_dir, "latest_model.pt"),
            model,
            optimizer,
            run_cfg,
            epoch,
            best_val_f1,
            num_classes,
        )

    write_csv(os.path.join(args.output_dir, "epoch_metrics.csv"), epoch_rows)

    print(f"\nLoading best model: {best_path}")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_rows, test_per_class = evaluate_periods(
        model, test_loaders, device, collapse_classes, stable_classes, thresholds
    )
    write_csv(os.path.join(args.output_dir, "test_period_metrics.csv"), test_rows)
    write_csv(os.path.join(args.output_dir, "test_collapse_class_metrics.csv"), test_per_class)

    summary = {
        "method": args.method,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_macro_f1": float(best_val_f1),
        "test_mean_macro_f1": float(np.mean([row["macro_f1"] for row in test_rows])),
        "test_mean_collapse_macro_f1": float(
            np.mean([row["collapse_macro_f1"] for row in test_rows])
        ),
        "test_mean_stable_macro_f1": float(
            np.mean([row["stable_macro_f1"] for row in test_rows])
        ),
        "final_period": test_rows[-1]["period"],
        "final_macro_f1": test_rows[-1]["macro_f1"],
        "final_collapse_macro_f1": test_rows[-1]["collapse_macro_f1"],
        "final_stable_macro_f1": test_rows[-1]["stable_macro_f1"],
        "final_collapsed_count": test_rows[-1]["collapsed_count"],
        "final_severe_collapsed_count": test_rows[-1]["severe_collapsed_count"],
        "train_periods": train_periods,
        "test_periods": test_periods,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Temporal Invariance Validation Summary ===")
    print(
        f"method={args.method} best_epoch={summary['best_epoch']} "
        f"mean_macro_f1={summary['test_mean_macro_f1']:.4f} "
        f"mean_collapse_f1={summary['test_mean_collapse_macro_f1']:.4f}"
    )
    print(
        f"{summary['final_period']}: macro_f1={summary['final_macro_f1']:.4f} "
        f"collapse_f1={summary['final_collapse_macro_f1']:.4f} "
        f"stable_f1={summary['final_stable_macro_f1']:.4f} "
        f"collapsed={summary['final_collapsed_count']} "
        f"severe={summary['final_severe_collapsed_count']}"
    )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    import multiprocessing as _mp

    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
