"""
Analyze whether collapse-prone class pairs are already close in source/history.

Reads collapse_classes.csv from scripts/summarize_per_class_collapse.py, extracts
collapsed class -> absorber pairs, and computes frozen-feature prototype distances
for selected historical periods/splits.

This is meant to answer:
  - Are future collapse pairs already close in M-1/M-2/M-3 or reference data?
  - Is the final absorber among the nearest prototype neighbors before collapse?

Usage from Experiment/core_code/:
    python scripts/analyze_collapse_pair_distances.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --collapse-csv outputs/per_class_collapse_tls22_monthly/collapse_classes.csv \
        --prototype-periods M-2022-1 M-2022-2 M-2022-3 M-2022-4 M-2022-12 \
        --split train \
        --output-dir outputs/collapse_pair_distances_tls22
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto
from tta_tc.data.cesnet_loader import build_dataloaders


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def as_int(row, key, default=None):
    value = row.get(key)
    if value in {None, ""}:
        return default
    return int(float(value))


def as_float(row, key, default=None):
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def load_pairs(collapse_csv, pair_source, final_only):
    rows = read_csv(collapse_csv)
    pairs = []
    seen = set()
    for row in rows:
        first_period = row.get("first_collapse_period", "")
        if final_only and not first_period:
            continue
        class_id = as_int(row, "class_id")
        candidates = []
        if pair_source in {"final", "both"}:
            candidates.append(("final", as_int(row, "final_top_confusion_target")))
        if pair_source in {"first", "both"}:
            candidates.append(("first", as_int(row, "first_collapse_top_confusion_target")))
        for source, absorber in candidates:
            if class_id is None or absorber is None:
                continue
            key = (class_id, absorber, source)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "class_id": class_id,
                "absorber_class": absorber,
                "pair_source": source,
                "first_collapse_period": first_period,
                "collapse_pattern": row.get("collapse_pattern", ""),
                "reference_recall": as_float(row, "reference_recall"),
                "final_recall": as_float(row, "final_recall"),
                "final_f1": as_float(row, "final_f1"),
                "final_top_confusion_rate": as_float(row, "final_top_confusion_rate"),
                "first_collapse_top_confusion_rate": as_float(
                    row, "first_collapse_top_confusion_rate"
                ),
            })
    return pairs


def make_period_loader(eval_cfg, period, split):
    data_cfg = dict(eval_cfg["data"])
    data_cfg["train_period"] = period
    data_cfg["test_period"] = period
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(data_cfg)
    if split == "train":
        return train_loader, num_classes
    if split == "val":
        return val_loader, num_classes
    if split == "test":
        return test_loader, num_classes
    raise ValueError(f"Unknown split: {split}")


def pair_distance_stats(prototypes, support, valid_mask, class_id, absorber_class):
    valid = valid_mask.cpu().numpy().astype(bool)
    if class_id >= len(valid) or absorber_class >= len(valid):
        return None
    if not valid[class_id] or not valid[absorber_class]:
        return None

    proto_n = F.normalize(prototypes, dim=1)
    cls = prototypes[class_id]
    absb = prototypes[absorber_class]
    cls_n = proto_n[class_id]
    abs_n = proto_n[absorber_class]

    cosine = float(torch.dot(cls_n, abs_n).item())
    cosine_distance = float(1.0 - cosine)
    normalized_l2 = float(torch.linalg.norm(cls_n - abs_n).item())
    raw_l2 = float(torch.linalg.norm(cls - absb).item())

    distances = []
    for other in np.flatnonzero(valid):
        other = int(other)
        if other == class_id:
            continue
        dist = float((1.0 - torch.dot(cls_n, proto_n[other])).item())
        distances.append((other, dist))
    distances.sort(key=lambda item: (item[1], item[0]))
    rank = next(
        (idx for idx, (other, _) in enumerate(distances, start=1) if other == absorber_class),
        None,
    )
    nearest_class, nearest_distance = distances[0] if distances else (None, None)
    percentile = rank / len(distances) if rank is not None and distances else None
    all_dists = np.asarray([d for _, d in distances], dtype=float)
    z_score = (
        float((cosine_distance - all_dists.mean()) / (all_dists.std() + 1e-12))
        if len(all_dists) else None
    )

    return {
        "class_support": int(support[class_id]),
        "absorber_support": int(support[absorber_class]),
        "raw_l2_distance": raw_l2,
        "cosine_similarity": cosine,
        "cosine_distance": cosine_distance,
        "normalized_l2_distance": normalized_l2,
        "absorber_neighbor_rank": rank,
        "num_valid_neighbors": len(distances),
        "absorber_rank_percentile": percentile,
        "nearest_class": nearest_class,
        "nearest_cosine_distance": nearest_distance,
        "absorber_is_nearest": int(nearest_class == absorber_class),
        "distance_z_vs_class_neighbors": z_score,
    }


def summarize_rows(rows):
    by_pair = {}
    for row in rows:
        key = (row["class_id"], row["absorber_class"], row["pair_source"])
        by_pair.setdefault(key, []).append(row)

    out = []
    for key, items in sorted(by_pair.items()):
        ranks = [
            float(row["absorber_neighbor_rank"])
            for row in items
            if row.get("absorber_neighbor_rank") not in {None, ""}
        ]
        distances = [
            float(row["cosine_distance"])
            for row in items
            if row.get("cosine_distance") not in {None, ""}
        ]
        nearest_hits = [
            int(row["absorber_is_nearest"])
            for row in items
            if row.get("absorber_is_nearest") not in {None, ""}
        ]
        first = items[0]
        out.append({
            "class_id": key[0],
            "absorber_class": key[1],
            "pair_source": key[2],
            "first_collapse_period": first["first_collapse_period"],
            "collapse_pattern": first["collapse_pattern"],
            "num_periods": len(items),
            "min_cosine_distance": min(distances) if distances else "",
            "mean_cosine_distance": float(np.mean(distances)) if distances else "",
            "min_absorber_rank": min(ranks) if ranks else "",
            "mean_absorber_rank": float(np.mean(ranks)) if ranks else "",
            "absorber_nearest_any_period": int(any(nearest_hits)) if nearest_hits else "",
            "final_recall": first["final_recall"],
            "final_f1": first["final_f1"],
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collapse-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/collapse_pair_distances_tls22")
    parser.add_argument(
        "--prototype-periods",
        nargs="+",
        default=["M-2022-1", "M-2022-2", "M-2022-3", "M-2022-4"],
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--pair-source", choices=["final", "first", "both"], default="final")
    parser.add_argument("--final-collapsed-only", type=parse_bool, default=True)
    parser.add_argument("--min-prototype-support", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pairs = load_pairs(args.collapse_csv, args.pair_source, args.final_collapsed_only)
    if not pairs:
        raise RuntimeError(f"No collapse pairs loaded from {args.collapse_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    eval_cfg = proto.load_config(args.config)
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg["data"]["num_classes"] = num_classes
    print(f"Loaded {len(pairs)} collapse pairs; num_classes={num_classes}")

    rows = []
    period_meta = []
    for period in args.prototype_periods:
        print(f"\nBuilding {args.split} loader for prototype period: {period}")
        loader, loader_num_classes = make_period_loader(eval_cfg, period, args.split)
        if loader_num_classes != num_classes:
            print(f"WARNING: loader classes={loader_num_classes}, model classes={num_classes}")
        outputs = proto.collect_outputs(model, loader, device, f"Prototype {period}/{args.split}")
        prototypes, support, valid_mask = proto.build_prototypes(
            outputs["features"], outputs["labels"], num_classes, args.min_prototype_support
        )
        valid_count = int(valid_mask.sum().item())
        period_meta.append({
            "period": period,
            "split": args.split,
            "num_samples": int(len(outputs["labels"])),
            "valid_prototype_classes": valid_count,
        })
        print(
            f"Built prototypes for {valid_count}/{num_classes} classes "
            f"from {len(outputs['labels'])} samples."
        )

        for pair in pairs:
            stats = pair_distance_stats(
                prototypes,
                support,
                valid_mask,
                pair["class_id"],
                pair["absorber_class"],
            )
            base = {
                "prototype_period": period,
                "split": args.split,
                **pair,
            }
            if stats is None:
                rows.append({
                    **base,
                    "class_support": int(support[pair["class_id"]])
                    if pair["class_id"] < len(support) else "",
                    "absorber_support": int(support[pair["absorber_class"]])
                    if pair["absorber_class"] < len(support) else "",
                    "valid_pair": 0,
                })
            else:
                rows.append({**base, "valid_pair": 1, **stats})

    pair_summary = summarize_rows([row for row in rows if row.get("valid_pair") == 1])
    write_csv(os.path.join(args.output_dir, "collapse_pair_distances.csv"), rows)
    write_csv(os.path.join(args.output_dir, "collapse_pair_distance_summary.csv"), pair_summary)
    write_csv(os.path.join(args.output_dir, "prototype_period_summary.csv"), period_meta)

    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "collapse_csv": args.collapse_csv,
        "prototype_periods": args.prototype_periods,
        "split": args.split,
        "pair_source": args.pair_source,
        "final_collapsed_only": args.final_collapsed_only,
        "min_prototype_support": args.min_prototype_support,
        "num_pairs": len(pairs),
        "num_distance_rows": len(rows),
        "periods": period_meta,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Collapse Pair Distance Summary ===")
    for row in pair_summary[:20]:
        print(
            f"class {row['class_id']} -> {row['absorber_class']} "
            f"min_dist={row['min_cosine_distance']:.4f} "
            f"min_rank={row['min_absorber_rank']} "
            f"nearest_any={row['absorber_nearest_any_period']}"
        )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
