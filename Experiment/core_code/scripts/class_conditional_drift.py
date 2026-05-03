"""
Class-conditional drift diagnostics for temporal traffic degradation.

This script diagnoses *which classes degrade*, *where they collapse*, and *how
their representation/input distributions change* under temporal drift.

It evaluates a frozen source CNN on selected test periods and writes:
  - per_class_metrics.csv
  - confusion_<period>.csv and confusion_pairs.csv
  - feature_geometry.csv
  - class_input_drift.csv
  - selected_bad_classes.csv / selected_stable_classes.csv
  - summary.json

Usage from Experiment/core_code/:
    python scripts/class_conditional_drift.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --periods M-2022-4 M-2022-7 M-2022-10 M-2022-12 \
        --output-dir outputs/class_conditional_drift_tls22
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.models import TTATCModel
from tta_tc.utils.config import load_config


CHANNELS = {
    "size": 0,
    "direction": 1,
    "ipt": 2,
}

REGIONS = {
    "size_all": [("size", range(30))],
    "direction_front_0_9": [("direction", range(0, 10))],
    "ipt_tail_20_29": [("ipt", range(20, 30))],
}


def load_source_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["num_classes"], cfg


def sanitize_period(period):
    """M-2022-4 -> m4, W-2022-45 -> w45."""
    match = re.match(r"([MW])-2022-(\d+)", period)
    if match:
        return f"{match.group(1).lower()}{match.group(2)}"
    return re.sub(r"[^A-Za-z0-9]+", "_", period).strip("_").lower()


def save_csv(rows, path, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def collect_period_outputs(model, loader, device, num_classes, max_batches=None):
    """Collect labels, predictions, embeddings, and PPI for one period."""
    all_labels = []
    all_preds = []
    all_embeddings = []
    all_ppi = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Frozen CNN forward")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            ppi = batch["ppi"].to(device)
            labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(device)

            logits, embeddings = model(ppi, flow_stats, return_repr=True)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.append(labels.numpy().astype(np.int64, copy=False))
            all_preds.append(preds.astype(np.int64, copy=False))
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            all_ppi.append(batch["ppi"].numpy().astype(np.float32, copy=False))

    if not all_labels:
        raise RuntimeError("No batches collected. Check dataloader and max_batches.")

    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
    ppi = np.concatenate(all_ppi, axis=0)
    present_labels = sorted(set(labels.tolist()))
    labels_for_metrics = list(range(num_classes))
    return {
        "labels": labels,
        "preds": preds,
        "embeddings": embeddings,
        "ppi": ppi,
        "present_labels": present_labels,
        "labels_for_metrics": labels_for_metrics,
    }


def compute_per_class_metrics(period, labels, preds, num_classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    rows = []
    by_class = {}
    for class_id in range(num_classes):
        row = {
            "period": period,
            "class_id": class_id,
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        rows.append(row)
        by_class[class_id] = row
    return rows, by_class


def add_metric_deltas(rows, reference_by_class):
    for row in rows:
        ref = reference_by_class.get(row["class_id"])
        if ref is None or ref["support"] == 0 or row["support"] == 0:
            row["delta_f1_from_ref"] = None
            row["delta_recall_from_ref"] = None
            row["delta_precision_from_ref"] = None
            continue
        row["delta_f1_from_ref"] = row["f1"] - ref["f1"]
        row["delta_recall_from_ref"] = row["recall"] - ref["recall"]
        row["delta_precision_from_ref"] = row["precision"] - ref["precision"]


def compute_confusion_rows(period, labels, preds, num_classes, max_pairs=None):
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    support = cm.sum(axis=1)
    rows = []
    for true_class in range(num_classes):
        if support[true_class] == 0:
            continue
        for pred_class in range(num_classes):
            if true_class == pred_class:
                continue
            count = int(cm[true_class, pred_class])
            if count == 0:
                continue
            rows.append({
                "period": period,
                "true_class": true_class,
                "pred_class": pred_class,
                "confusion_count": count,
                "confusion_rate": float(count / support[true_class]),
            })
    rows.sort(key=lambda r: (r["confusion_rate"], r["confusion_count"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_in_period"] = rank
    if max_pairs is not None and max_pairs > 0:
        return rows[:max_pairs]
    return rows


def add_confusion_deltas(rows_by_period, reference_period):
    ref_map = {
        (row["true_class"], row["pred_class"]): row
        for row in rows_by_period.get(reference_period, [])
    }
    all_rows = []
    for period, rows in rows_by_period.items():
        for row in rows:
            ref = ref_map.get((row["true_class"], row["pred_class"]))
            row = dict(row)
            row["delta_count_from_ref"] = row["confusion_count"] - (ref["confusion_count"] if ref else 0)
            row["delta_rate_from_ref"] = row["confusion_rate"] - (ref["confusion_rate"] if ref else 0.0)
            all_rows.append(row)
    return all_rows


def class_centroids_and_radii(labels, embeddings, num_classes):
    centroids = {}
    radii = {}
    supports = {}
    for class_id in range(num_classes):
        mask = labels == class_id
        support = int(mask.sum())
        supports[class_id] = support
        if support == 0:
            continue
        emb = embeddings[mask]
        centroid = emb.mean(axis=0)
        dist = np.linalg.norm(emb - centroid[None, :], axis=1)
        centroids[class_id] = centroid
        radii[class_id] = float(dist.mean())
    return centroids, radii, supports


def nearest_centroid_info(class_id, centroid, centroids):
    best_class = None
    best_dist = None
    for other_class, other_centroid in centroids.items():
        if other_class == class_id:
            continue
        dist = float(np.linalg.norm(centroid - other_centroid))
        if best_dist is None or dist < best_dist:
            best_class = other_class
            best_dist = dist
    return best_class, best_dist


def top_confusion_partner(confusion_rows, class_id):
    candidates = [row for row in confusion_rows if row["true_class"] == class_id]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["confusion_rate"], r["confusion_count"]))


def compute_feature_geometry(period_outputs, confusion_by_period, reference_period, num_classes):
    period_stats = {}
    for period, data in period_outputs.items():
        centroids, radii, supports = class_centroids_and_radii(
            data["labels"], data["embeddings"], num_classes
        )
        period_stats[period] = {
            "centroids": centroids,
            "radii": radii,
            "supports": supports,
        }

    ref = period_stats[reference_period]
    ref_confusion = confusion_by_period.get(reference_period, [])
    rows = []
    for period, stats_dict in period_stats.items():
        period_confusion = confusion_by_period.get(period, [])
        for class_id in range(num_classes):
            support = stats_dict["supports"].get(class_id, 0)
            if support == 0 or class_id not in stats_dict["centroids"]:
                continue

            centroid = stats_dict["centroids"][class_id]
            radius = stats_dict["radii"][class_id]
            nearest_class, nearest_dist = nearest_centroid_info(
                class_id, centroid, stats_dict["centroids"]
            )
            margin = None if nearest_dist is None else nearest_dist - radius
            partner = top_confusion_partner(period_confusion, class_id)
            confusion_partner = partner["pred_class"] if partner else None
            confusion_partner_rate = partner["confusion_rate"] if partner else None
            confusion_partner_distance = None
            if confusion_partner in stats_dict["centroids"]:
                confusion_partner_distance = float(np.linalg.norm(
                    centroid - stats_dict["centroids"][confusion_partner]
                ))

            ref_centroid = ref["centroids"].get(class_id)
            ref_radius = ref["radii"].get(class_id)
            ref_nearest_dist = None
            ref_margin = None
            ref_top_partner = None
            ref_top_partner_distance = None
            ref_distance_to_current_partner = None
            if class_id in ref["centroids"]:
                _, ref_nearest_dist = nearest_centroid_info(
                    class_id, ref["centroids"][class_id], ref["centroids"]
                )
                if ref_nearest_dist is not None and ref_radius is not None:
                    ref_margin = ref_nearest_dist - ref_radius
                ref_partner = top_confusion_partner(ref_confusion, class_id)
                ref_top_partner = ref_partner["pred_class"] if ref_partner else None
                if ref_top_partner in ref["centroids"]:
                    ref_top_partner_distance = float(np.linalg.norm(
                        ref["centroids"][class_id] - ref["centroids"][ref_top_partner]
                    ))
                if confusion_partner in ref["centroids"]:
                    ref_distance_to_current_partner = float(np.linalg.norm(
                        ref["centroids"][class_id] - ref["centroids"][confusion_partner]
                    ))

            centroid_shift = (
                float(np.linalg.norm(centroid - ref_centroid))
                if ref_centroid is not None else None
            )
            row = {
                "period": period,
                "class_id": class_id,
                "support": support,
                "centroid_shift_from_ref": centroid_shift,
                "radius": radius,
                "delta_radius_from_ref": (
                    radius - ref_radius if ref_radius is not None else None
                ),
                "nearest_other_class": nearest_class,
                "nearest_centroid_distance": nearest_dist,
                "delta_nearest_distance_from_ref": (
                    nearest_dist - ref_nearest_dist
                    if nearest_dist is not None and ref_nearest_dist is not None
                    else None
                ),
                "margin": margin,
                "delta_margin_from_ref": (
                    margin - ref_margin
                    if margin is not None and ref_margin is not None
                    else None
                ),
                "top_confusion_partner": confusion_partner,
                "top_confusion_partner_rate": confusion_partner_rate,
                "confusion_partner_distance": confusion_partner_distance,
                "ref_top_confusion_partner": ref_top_partner,
                "ref_top_confusion_partner_distance": ref_top_partner_distance,
                "ref_distance_to_current_partner": ref_distance_to_current_partner,
                "delta_current_partner_distance_from_ref": (
                    confusion_partner_distance - ref_distance_to_current_partner
                    if confusion_partner_distance is not None and ref_distance_to_current_partner is not None
                    else None
                ),
            }
            rows.append(row)
    return rows


def w1_for_values(src, tgt):
    if len(src) == 0 or len(tgt) == 0:
        return None
    return float(stats.wasserstein_distance(src, tgt))


def region_w1(source_ppi, target_ppi, region_spec, normalize=False, log_space=False):
    total = 0.0
    count = 0
    for channel_name, positions in region_spec:
        channel_idx = CHANNELS[channel_name]
        for pos in positions:
            src = source_ppi[:, channel_idx, pos]
            tgt = target_ppi[:, channel_idx, pos]
            if log_space and channel_name in {"size", "ipt"}:
                src = signed_log1p(src)
                tgt = signed_log1p(tgt)
            w1 = w1_for_values(src, tgt)
            if w1 is None:
                continue
            if normalize:
                denom = float(np.std(src) + 1e-8)
                w1 = w1 / denom
            total += w1
            count += 1
    return total if count > 0 else None


def compute_class_input_drift(period_outputs, reference_period, per_class_ref, min_support):
    ref_data = period_outputs[reference_period]
    ref_ppi = ref_data["ppi"]
    ref_labels = ref_data["labels"]
    rows = []

    all_regions = [
        ("size", range(30)),
        ("direction", range(30)),
        ("ipt", range(30)),
    ]

    for period, data in period_outputs.items():
        ppi = data["ppi"]
        labels = data["labels"]
        for class_id, ref_metric in per_class_ref.items():
            ref_mask = ref_labels == class_id
            tgt_mask = labels == class_id
            ref_support = int(ref_mask.sum())
            support = int(tgt_mask.sum())
            if ref_support < min_support or support < min_support:
                continue

            ref_class_ppi = ref_ppi[ref_mask]
            tgt_class_ppi = ppi[tgt_mask]
            metric = per_class_ref[class_id]
            target_metric = data["per_class_by_class"].get(class_id, {})
            delta_f1 = target_metric.get("f1", 0.0) - metric.get("f1", 0.0)

            row = {
                "period": period,
                "class_id": class_id,
                "support": support,
                "ref_support": ref_support,
                "size_sum_w1": region_w1(ref_class_ppi, tgt_class_ppi, REGIONS["size_all"]),
                "direction_front_0_9_sum_w1": region_w1(
                    ref_class_ppi, tgt_class_ppi, REGIONS["direction_front_0_9"]
                ),
                "ipt_tail_20_29_sum_w1": region_w1(
                    ref_class_ppi, tgt_class_ppi, REGIONS["ipt_tail_20_29"]
                ),
                "total_norm_w1": region_w1(
                    ref_class_ppi, tgt_class_ppi, all_regions, normalize=True
                ),
                "total_log_w1": region_w1(
                    ref_class_ppi, tgt_class_ppi, all_regions, log_space=True
                ),
                "delta_f1_from_ref": delta_f1,
            }
            rows.append(row)
    return rows


def select_bad_and_stable_classes(per_class_rows, reference_period, final_period, min_support, top_k):
    final_rows = [
        row for row in per_class_rows
        if row["period"] == final_period
        and row["support"] >= min_support
        and row.get("delta_f1_from_ref") is not None
    ]
    bad = [dict(row) for row in sorted(
        final_rows, key=lambda r: (r["delta_f1_from_ref"], -r["support"])
    )[:top_k]]
    stable = [dict(row) for row in sorted(
        final_rows, key=lambda r: (abs(r["delta_f1_from_ref"]), -r["support"])
    )[:top_k]]
    for row in bad:
        row["bucket"] = "bad"
        row["reference_period"] = reference_period
        row["selection_period"] = final_period
    for row in stable:
        row["bucket"] = "stable"
        row["reference_period"] = reference_period
        row["selection_period"] = final_period
    return bad, stable


def summarize_groups(class_input_rows, feature_rows, bad_classes, stable_classes):
    bad_ids = {row["class_id"] for row in bad_classes}
    stable_ids = {row["class_id"] for row in stable_classes}

    def mean_for(rows, class_ids, field):
        vals = [
            row.get(field) for row in rows
            if row.get("class_id") in class_ids
            and row.get(field) is not None
            and np.isfinite(row.get(field))
        ]
        return float(np.mean(vals)) if vals else None

    fields_input = [
        "size_sum_w1",
        "direction_front_0_9_sum_w1",
        "ipt_tail_20_29_sum_w1",
        "total_norm_w1",
        "total_log_w1",
        "delta_f1_from_ref",
    ]
    fields_feature = [
        "centroid_shift_from_ref",
        "delta_radius_from_ref",
        "delta_nearest_distance_from_ref",
        "delta_margin_from_ref",
        "delta_current_partner_distance_from_ref",
    ]

    return {
        "bad_class_ids": sorted(bad_ids),
        "stable_class_ids": sorted(stable_ids),
        "bad_means_input": {field: mean_for(class_input_rows, bad_ids, field) for field in fields_input},
        "stable_means_input": {field: mean_for(class_input_rows, stable_ids, field) for field in fields_input},
        "bad_means_feature": {field: mean_for(feature_rows, bad_ids, field) for field in fields_feature},
        "stable_means_feature": {field: mean_for(feature_rows, stable_ids, field) for field in fields_feature},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--periods", nargs="*", default=None, help="Optional period subset; first period is reference")
    parser.add_argument("--output-dir", default="outputs/class_conditional_drift_tls22")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-confusion-pairs", type=int, default=0,
                        help="Limit confusion CSV rows per period; <=0 saves all nonzero off-diagonal pairs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    selected_periods = set(args.periods) if args.periods else None
    max_batches = None if args.max_batches is not None and args.max_batches <= 0 else args.max_batches
    max_confusion_pairs = (
        None if args.max_confusion_pairs is not None and args.max_confusion_pairs <= 0
        else args.max_confusion_pairs
    )

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

    print("Building test loaders...")
    loaders, _ = build_sequential_test_loaders(cfg["data"])
    loaders = [
        (period, loader) for period, loader in loaders
        if selected_periods is None or period in selected_periods
    ]
    if not loaders:
        raise RuntimeError(f"No selected periods found. Requested: {args.periods}")

    reference_period = loaders[0][0]
    final_period = loaders[-1][0]
    print(f"Reference period: {reference_period}")
    print(f"Selection/final period: {final_period}")

    os.makedirs(args.output_dir, exist_ok=True)

    period_outputs = {}
    per_class_rows = []
    per_class_by_period = {}
    confusion_by_period = {}

    for period, loader in loaders:
        print(f"\n{'=' * 70}")
        print(f"Period: {period}")
        print(f"{'=' * 70}")
        outputs = collect_period_outputs(
            model=model,
            loader=loader,
            device=device,
            num_classes=num_classes,
            max_batches=max_batches,
        )
        rows, by_class = compute_per_class_metrics(
            period, outputs["labels"], outputs["preds"], num_classes
        )
        outputs["per_class_by_class"] = by_class
        period_outputs[period] = outputs
        per_class_by_period[period] = by_class
        per_class_rows.extend(rows)

        confusion_rows = compute_confusion_rows(
            period,
            outputs["labels"],
            outputs["preds"],
            num_classes,
            max_pairs=max_confusion_pairs,
        )
        confusion_by_period[period] = confusion_rows
        confusion_path = os.path.join(
            args.output_dir, f"confusion_{sanitize_period(period)}.csv"
        )
        save_csv(confusion_rows, confusion_path)
        print(f"Saved confusion rows: {confusion_path}")

    reference_by_class = per_class_by_period[reference_period]
    add_metric_deltas(per_class_rows, reference_by_class)

    confusion_pairs = add_confusion_deltas(confusion_by_period, reference_period)
    feature_geometry_rows = compute_feature_geometry(
        period_outputs, confusion_by_period, reference_period, num_classes
    )
    class_input_rows = compute_class_input_drift(
        period_outputs,
        reference_period,
        reference_by_class,
        min_support=args.min_support,
    )

    bad_classes, stable_classes = select_bad_and_stable_classes(
        per_class_rows,
        reference_period=reference_period,
        final_period=final_period,
        min_support=args.min_support,
        top_k=args.top_k,
    )
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "periods": [period for period, _ in loaders],
        "reference_period": reference_period,
        "final_period": final_period,
        "num_classes": num_classes,
        "min_support": args.min_support,
        "top_k": args.top_k,
        "group_summary": summarize_groups(
            class_input_rows, feature_geometry_rows, bad_classes, stable_classes
        ),
    }

    save_csv(per_class_rows, os.path.join(args.output_dir, "per_class_metrics.csv"))
    save_csv(confusion_pairs, os.path.join(args.output_dir, "confusion_pairs.csv"))
    save_csv(feature_geometry_rows, os.path.join(args.output_dir, "feature_geometry.csv"))
    save_csv(class_input_rows, os.path.join(args.output_dir, "class_input_drift.csv"))
    save_csv(bad_classes, os.path.join(args.output_dir, "selected_bad_classes.csv"))
    save_csv(stable_classes, os.path.join(args.output_dir, "selected_stable_classes.csv"))

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Selected bad classes ===")
    for row in bad_classes[:10]:
        print(
            f"class={row['class_id']:>4} support={row['support']:>6} "
            f"f1={row['f1']:.4f} delta={row['delta_f1_from_ref']:+.4f}"
        )

    print("\n=== Selected stable classes ===")
    for row in stable_classes[:10]:
        print(
            f"class={row['class_id']:>4} support={row['support']:>6} "
            f"f1={row['f1']:.4f} delta={row['delta_f1_from_ref']:+.4f}"
        )

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    import multiprocessing as _mp

    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
