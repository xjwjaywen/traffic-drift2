"""
Oracle-style input drift correction with frozen source CNN.

This script evaluates whether correcting input-level marginal drift can recover
performance without updating model parameters. It estimates per-channel,
per-position quantile maps from unlabeled test-period PPI distributions to the
source validation PPI distributions, applies selected corrections, and evaluates
the frozen CNN.

This is a transductive intervention study, not a deployment-ready online method.

Usage from Experiment/core_code/:
    python scripts/quantile_correct_eval.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --periods M-2022-7 M-2022-10 M-2022-12
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from tta_tc.data.cesnet_loader import build_dataloaders, build_sequential_test_loaders
from tta_tc.models import TTATCModel
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


CHANNELS = {
    "size": 0,
    "direction": 1,
    "ipt": 2,
}


def load_source_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["num_classes"], cfg


def collect_dataset(loader, max_batches=None, collect_flow_stats=False):
    ppi_chunks = []
    label_chunks = []
    flow_chunks = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting PPI")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        ppi_chunks.append(batch["ppi"].numpy().astype(np.float32, copy=False))
        label_chunks.append(batch["label"].numpy().astype(np.int64, copy=False))
        if collect_flow_stats and "flow_stats" in batch:
            flow_chunks.append(batch["flow_stats"].numpy().astype(np.float32, copy=False))

    if not ppi_chunks:
        raise RuntimeError("No data collected. Check dataloader and max_batches.")

    ppi = np.concatenate(ppi_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    flow_stats = np.concatenate(flow_chunks, axis=0) if flow_chunks else None
    return ppi, labels, flow_stats


def build_correction_masks():
    all_positions = list(range(30))
    front = list(range(0, 10))
    tail = list(range(20, 30))

    return {
        "raw": [],
        "all": [
            ("size", all_positions),
            ("direction", all_positions),
            ("ipt", all_positions),
        ],
        "size_all": [("size", all_positions)],
        "direction_front_0_9": [("direction", front)],
        "ipt_tail_20_29": [("ipt", tail)],
        "size_direction_front_ipt_tail": [
            ("size", all_positions),
            ("direction", front),
            ("ipt", tail),
        ],
    }


def parse_periods_arg(periods):
    if periods is None:
        return None
    return set(periods)


def quantile_map_values(values, source_reference, test_reference):
    """Map values from test marginal distribution to source marginal distribution."""
    src_sorted = np.sort(source_reference.astype(np.float64, copy=False))
    test_sorted = np.sort(test_reference.astype(np.float64, copy=False))
    if len(src_sorted) == 0 or len(test_sorted) == 0:
        return values

    ranks = np.searchsorted(test_sorted, values, side="left").astype(np.float64)
    if len(test_sorted) > 1:
        quantiles = ranks / float(len(test_sorted) - 1)
    else:
        quantiles = np.zeros_like(ranks)

    src_pos = quantiles * float(len(src_sorted) - 1)
    lo = np.floor(src_pos).astype(np.int64)
    hi = np.ceil(src_pos).astype(np.int64)
    frac = src_pos - lo
    mapped = src_sorted[lo] * (1.0 - frac) + src_sorted[hi] * frac
    return mapped.astype(values.dtype, copy=False)


def apply_quantile_correction(test_ppi, source_ppi, correction_spec):
    corrected = test_ppi.copy()
    for channel_name, positions in correction_spec:
        channel_idx = CHANNELS[channel_name]
        for pos in positions:
            corrected[:, channel_idx, pos] = quantile_map_values(
                values=test_ppi[:, channel_idx, pos],
                source_reference=source_ppi[:, channel_idx, pos],
                test_reference=test_ppi[:, channel_idx, pos],
            )
    return corrected


def iter_eval_batches(ppi, labels, flow_stats, batch_size):
    n = len(labels)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ppi_batch = torch.from_numpy(ppi[start:end])
        labels_batch = labels[start:end]
        if flow_stats is not None:
            flow_batch = torch.from_numpy(flow_stats[start:end])
        else:
            flow_batch = None
        yield ppi_batch, labels_batch, flow_batch


@torch.no_grad()
def evaluate_ppi_array(model, ppi, labels, flow_stats, device, batch_size):
    model.eval()
    all_preds = []
    all_labels = []
    for ppi_batch, labels_batch, flow_batch in tqdm(
        iter_eval_batches(ppi, labels, flow_stats, batch_size),
        total=int(np.ceil(len(labels) / batch_size)),
        desc="Frozen eval",
    ):
        ppi_batch = ppi_batch.to(device)
        if flow_batch is not None:
            flow_batch = flow_batch.to(device)
        logits = model(ppi_batch, flow_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels_batch)
    return compute_metrics(all_labels, all_preds)


def w1_for_positions(source_ppi, test_ppi, correction_spec=None):
    if correction_spec is None:
        correction_spec = [
            ("size", range(30)),
            ("direction", range(30)),
            ("ipt", range(30)),
        ]
    result = {}
    total = 0.0
    for channel_name, positions in correction_spec:
        channel_idx = CHANNELS[channel_name]
        channel_total = 0.0
        for pos in positions:
            value = float(stats.wasserstein_distance(
                source_ppi[:, channel_idx, pos],
                test_ppi[:, channel_idx, pos],
            ))
            channel_total += value
            total += value
        result[channel_name] = result.get(channel_name, 0.0) + channel_total
    result["total"] = total
    return result


def w1_by_named_regions(source_ppi, test_ppi):
    masks = build_correction_masks()
    return {
        name: w1_for_positions(source_ppi, test_ppi, spec)
        for name, spec in masks.items()
        if name != "raw"
    }


def infer_eval_batch_size(cfg, override):
    if override is not None:
        return override
    return int(cfg.get("data", {}).get("batch_size", 256))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Evaluation YAML, e.g. configs/eval_tls22.yaml")
    parser.add_argument("--checkpoint", required=True, help="Frozen source CNN checkpoint")
    parser.add_argument("--periods", nargs="*", default=None, help="Subset of periods, e.g. M-2022-7 M-2022-10")
    parser.add_argument("--settings", nargs="*", default=None, help="Correction settings to run")
    parser.add_argument("--output-dir", default="outputs/quantile_correction_tls22")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-source-batches", type=int, default=1600)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    selected_periods = parse_periods_arg(args.periods)
    correction_masks = build_correction_masks()
    settings = args.settings or [
        "raw",
        "all",
        "size_all",
        "direction_front_0_9",
        "ipt_tail_20_29",
        "size_direction_front_ipt_tail",
    ]
    if "raw" not in settings:
        settings = ["raw"] + settings
    unknown = [name for name in settings if name not in correction_masks]
    if unknown:
        raise ValueError(f"Unknown correction settings: {unknown}. Valid: {sorted(correction_masks)}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading frozen source model...")
    model, num_classes, train_cfg = load_source_model(args.checkpoint, device)
    print(f"Num classes: {num_classes}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading source validation distribution...")
    _, source_loader, _, _ = build_dataloaders(cfg["data"])
    source_ppi, _, _ = collect_dataset(
        source_loader,
        max_batches=args.max_source_batches,
        collect_flow_stats=False,
    )
    print(f"Source samples for quantile reference: {source_ppi.shape[0]}")

    loaders, _ = build_sequential_test_loaders(cfg["data"])
    loaders = [
        (period_name, loader)
        for period_name, loader in loaders
        if selected_periods is None or period_name in selected_periods
    ]
    if not loaders:
        raise RuntimeError(f"No selected test periods found. Requested: {args.periods}")

    eval_batch_size = infer_eval_batch_size(cfg, args.eval_batch_size)
    all_results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "periods": [name for name, _ in loaders],
        "settings": settings,
        "max_source_batches": args.max_source_batches,
        "max_test_batches": args.max_test_batches,
        "eval_batch_size": eval_batch_size,
        "note": "Oracle-style transductive marginal quantile correction; no model parameters updated.",
        "results": {},
    }
    summary_rows = []

    for period_name, loader in loaders:
        print(f"\n{'=' * 70}")
        print(f"Period: {period_name}")
        print(f"{'=' * 70}")

        test_ppi, labels, flow_stats = collect_dataset(
            loader,
            max_batches=args.max_test_batches,
            collect_flow_stats=train_cfg["model"].get("flow_stats_dim", 0) > 0,
        )
        print(f"Test samples: {test_ppi.shape[0]}")

        raw_w1_regions = w1_by_named_regions(source_ppi, test_ppi)
        raw_total_w1 = w1_for_positions(source_ppi, test_ppi)["total"]
        period_results = {
            "num_samples": int(test_ppi.shape[0]),
            "raw_w1_regions": raw_w1_regions,
            "settings": {},
        }

        raw_metrics = None
        for setting_name in settings:
            print(f"\n--- Setting: {setting_name} ---")
            t0 = time.time()
            if setting_name == "raw":
                corrected_ppi = test_ppi
            else:
                corrected_ppi = apply_quantile_correction(
                    test_ppi=test_ppi,
                    source_ppi=source_ppi,
                    correction_spec=correction_masks[setting_name],
                )

            metrics = evaluate_ppi_array(
                model=model,
                ppi=corrected_ppi,
                labels=labels,
                flow_stats=flow_stats,
                device=device,
                batch_size=eval_batch_size,
            )
            elapsed = time.time() - t0
            corrected_w1_regions = w1_by_named_regions(source_ppi, corrected_ppi)
            corrected_total_w1 = w1_for_positions(source_ppi, corrected_ppi)["total"]
            if setting_name == "raw":
                setting_region_w1_before = 0.0
                setting_region_w1_after = 0.0
            else:
                setting_region_w1_before = w1_for_positions(
                    source_ppi, test_ppi, correction_masks[setting_name]
                )["total"]
                setting_region_w1_after = w1_for_positions(
                    source_ppi, corrected_ppi, correction_masks[setting_name]
                )["total"]
            setting_result = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "elapsed_s": elapsed,
                "corrected_regions": correction_masks[setting_name],
                "total_w1_before": raw_total_w1,
                "total_w1_after": corrected_total_w1,
                "total_w1_reduction": raw_total_w1 - corrected_total_w1,
                "setting_region_w1_before": setting_region_w1_before,
                "setting_region_w1_after": setting_region_w1_after,
                "setting_region_w1_reduction": setting_region_w1_before - setting_region_w1_after,
                "w1_regions": corrected_w1_regions,
            }
            if setting_name == "raw":
                raw_metrics = setting_result
                setting_result["delta_macro_f1_vs_raw"] = 0.0
                setting_result["delta_accuracy_vs_raw"] = 0.0
            elif raw_metrics is not None:
                setting_result["delta_macro_f1_vs_raw"] = (
                    setting_result["macro_f1"] - raw_metrics["macro_f1"]
                )
                setting_result["delta_accuracy_vs_raw"] = (
                    setting_result["accuracy"] - raw_metrics["accuracy"]
                )

            period_results["settings"][setting_name] = setting_result
            summary_rows.append({
                "period": period_name,
                "setting": setting_name,
                "accuracy": setting_result["accuracy"],
                "macro_f1": setting_result["macro_f1"],
                "weighted_f1": setting_result["weighted_f1"],
                "delta_macro_f1_vs_raw": setting_result["delta_macro_f1_vs_raw"],
                "delta_accuracy_vs_raw": setting_result["delta_accuracy_vs_raw"],
                "total_w1_before": setting_result["total_w1_before"],
                "total_w1_after": setting_result["total_w1_after"],
                "total_w1_reduction": setting_result["total_w1_reduction"],
                "setting_region_w1_before": setting_result["setting_region_w1_before"],
                "setting_region_w1_after": setting_result["setting_region_w1_after"],
                "setting_region_w1_reduction": setting_result["setting_region_w1_reduction"],
                "elapsed_s": elapsed,
            })
            print(
                f"Acc={setting_result['accuracy']:.4f}, "
                f"Macro-F1={setting_result['macro_f1']:.4f}, "
                f"Delta-F1={setting_result['delta_macro_f1_vs_raw']:+.4f}, "
                f"Time={elapsed:.1f}s"
            )

        all_results["results"][period_name] = period_results

        period_path = os.path.join(args.output_dir, f"{period_name}_results.json")
        with open(period_path, "w", encoding="utf-8") as f:
            json.dump(period_results, f, indent=2)
        print(f"Saved period results: {period_path}")

    full_path = os.path.join(args.output_dir, "results.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Summary ===")
    for period_name, period_result in all_results["results"].items():
        raw_f1 = period_result["settings"]["raw"]["macro_f1"]
        print(f"\n{period_name} raw macro-F1: {raw_f1:.4f}")
        for setting_name, setting_result in period_result["settings"].items():
            print(
                f"  {setting_name:<32} "
                f"F1={setting_result['macro_f1']:.4f} "
                f"delta={setting_result['delta_macro_f1_vs_raw']:+.4f}"
            )
    print(f"\nSaved full results: {full_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    import multiprocessing as _mp

    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
