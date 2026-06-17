#!/usr/bin/env python
"""
Evaluate all TTA baselines on a single test period with collapse/stable
group F1 metrics. Produces a clean comparison CSV for the paper.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines_with_groups.py \
        --config configs/eval_tls22.yaml \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --test-period M-2022-12 \
        --output-dir outputs/baselines_group_metrics_M12
"""
import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tta_tc.models import TTATCModel
from tta_tc.baselines import Tent, EATA, CoTTA, SAR, NOTE, BNAdapt, MVFC
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config

COLLAPSE_CLASSES = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
STABLE_CLASSES = [8, 15, 44, 57, 59, 62, 64, 76, 94, 98, 99, 107,
                  113, 119, 128, 130, 131, 132, 144, 145]


def load_model(checkpoint_path, cfg, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model_cfg = ckpt.get("config", cfg)
    if "model" not in model_cfg:
        model_cfg = cfg
    model_cfg["model"]["num_classes"] = num_classes
    model = TTATCModel(model_cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, num_classes


def evaluate_static(model, loader, device):
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Static"):
            ppi = batch["ppi"].to(device)
            fs = batch.get("flow_stats")
            if fs is not None:
                fs = fs.to(device)
            logits = model(ppi, fs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_labels), np.array(all_preds)


def evaluate_tta(method, loader, device, name):
    all_labels, all_preds = [], []
    t0 = time.time()
    for batch in tqdm(loader, desc=name):
        ppi = batch["ppi"].to(device)
        fs = batch.get("flow_stats")
        if fs is not None:
            fs = fs.to(device)
        logits, _ = method.adapt_batch(ppi, fs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return np.array(all_labels), np.array(all_preds), time.time() - t0


def group_f1(labels, preds, class_ids):
    """Compute macro F1 for a group of classes using full-dataset FP/FN."""
    from sklearn.metrics import classification_report
    report = classification_report(labels, preds, output_dict=True,
                                   zero_division=0)
    f1s = []
    for c in class_ids:
        entry = report.get(str(c), {})
        if entry.get("support", 0) > 0:
            f1s.append(entry["f1-score"])
    return float(np.mean(f1s)) if f1s else 0.0


def compute_all_metrics(labels, preds, collapse_classes=None):
    cc = collapse_classes if collapse_classes is not None else COLLAPSE_CLASSES
    return {
        "overall_macro_f1": float(f1_score(labels, preds, average="macro",
                                           zero_division=0)),
        "collapse_macro_f1": group_f1(labels, preds, cc),
        "stable_macro_f1": group_f1(labels, preds, STABLE_CLASSES),
        "collapsed_count": sum(
            1 for c in cc
            if np.isin(labels, [c]).sum() > 0
            and (preds[labels == c] == c).sum() / max((labels == c).sum(), 1) < 0.1
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="static,bn_adapt,tent,eata,cotta,sar,note")
    parser.add_argument("--collapse-classes", default=None,
                        help="Comma-separated collapse class IDs (default: M12 fixed 12-class set)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    cfg["data"]["test_period"] = args.test_period

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test period: {args.test_period}")

    custom_cc = None
    if args.collapse_classes:
        custom_cc = [int(c) for c in args.collapse_classes.split(",")]
        print(f"Using custom collapse classes ({len(custom_cc)}): {custom_cc}")

    model, num_classes = load_model(args.checkpoint, cfg, device)
    _, _, test_loader, _ = build_dataloaders(cfg["data"])

    adapt_cfg = {
        "num_classes": num_classes,
        **cfg.get("tta", {}),
    }

    method_classes = {
        "bn_adapt": ("BN-Adapt", BNAdapt),
        "tent": ("Tent", Tent),
        "eata": ("EATA", EATA),
        "cotta": ("CoTTA", CoTTA),
        "sar": ("SAR", SAR),
        "note": ("NOTE", NOTE),
        "mvfc": ("MVFC", MVFC),
    }

    methods_to_run = [m.strip() for m in args.methods.split(",")]
    rows = []

    for method_key in methods_to_run:
        if method_key == "static":
            print("\n=== Static ===")
            labels, preds = evaluate_static(model, test_loader, device)
            m = compute_all_metrics(labels, preds, custom_cc)
            m["method"] = "Static"
            rows.append(m)
            print(f"  macro={m['overall_macro_f1']:.4f} collapse={m['collapse_macro_f1']:.4f} "
                  f"stable={m['stable_macro_f1']:.4f} collapsed={m['collapsed_count']}")

        elif method_key in method_classes:
            name, MethodClass = method_classes[method_key]
            print(f"\n=== {name} ===")
            m_model = copy.deepcopy(model).to(device)
            method = MethodClass(m_model, adapt_cfg)
            labels, preds, t = evaluate_tta(method, test_loader, device, name)
            m = compute_all_metrics(labels, preds, custom_cc)
            m["method"] = name
            m["time_s"] = round(t, 1)
            rows.append(m)
            print(f"  macro={m['overall_macro_f1']:.4f} collapse={m['collapse_macro_f1']:.4f} "
                  f"stable={m['stable_macro_f1']:.4f} collapsed={m['collapsed_count']} time={t:.1f}s")
            del m_model

    # Save CSV
    out_path = os.path.join(args.output_dir, "baselines_group_metrics.csv")
    fieldnames = ["method", "overall_macro_f1", "collapse_macro_f1",
                  "stable_macro_f1", "collapsed_count", "time_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {out_path}")
    print("\n" + "=" * 70)
    print(f"{'Method':<12} {'macro-F1':>10} {'collapse':>10} {'stable':>10} {'collapsed':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['method']:<12} {r['overall_macro_f1']:10.4f} {r['collapse_macro_f1']:10.4f} "
              f"{r['stable_macro_f1']:10.4f} {r.get('collapsed_count', ''):>10}")


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
