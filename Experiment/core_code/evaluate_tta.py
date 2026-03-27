"""
TTA-TC Evaluation Script.

Evaluates all methods (static, baselines, TTA-TC) on temporal test data.

Usage:
    python evaluate_tta.py --config configs/eval_quic22.yaml --checkpoint outputs/train/best_model.pt
"""
import argparse
import os
import json
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from tta_tc.models import TTATCModel
from tta_tc.tta import TTAEngine
from tta_tc.baselines import Tent, EATA, CoTTA, SAR, NOTE, BNAdapt
from tta_tc.data.cesnet_loader import build_dataloaders, build_sequential_test_loaders
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import MetricsTracker


def load_source_model(checkpoint_path, device):
    """Load trained source model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt["num_classes"]


def evaluate_static(model, test_loader, device):
    """Evaluate without any adaptation (static baseline)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Static"):
            ppi = batch["ppi"].to(device)
            labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(device)

            logits = model(ppi, flow_stats)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate_tta_method(method, test_loader, device, method_name="TTA"):
    """Evaluate a TTA method on test data."""
    all_preds = []
    all_labels = []
    total_time = 0

    for batch in tqdm(test_loader, desc=method_name):
        ppi = batch["ppi"].to(device)
        labels = batch["label"]
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)

        t0 = time.time()
        logits, info = method.adapt_batch(ppi, flow_stats)
        total_time += time.time() - t0

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), total_time


def run_single_period_eval(model_path, eval_cfg, device):
    """Evaluate all methods on a single test period."""
    model, train_cfg, num_classes = load_source_model(model_path, device)
    eval_cfg["data"]["num_classes"] = num_classes

    # Build test loader
    _, _, test_loader, _ = build_dataloaders(eval_cfg["data"])

    results = {}

    # 1. Static baseline
    print("\n=== B1: Static (no adaptation) ===")
    labels, preds = evaluate_static(model, test_loader, device)
    from tta_tc.utils.metrics import compute_metrics
    static_metrics = compute_metrics(labels, preds)
    results["static"] = {
        "accuracy": static_metrics["accuracy"],
        "macro_f1": static_metrics["macro_f1"],
    }
    print(f"Accuracy: {static_metrics['accuracy']:.4f}, F1: {static_metrics['macro_f1']:.4f}")

    # Baseline entropy and source prototypes
    ckpt_dir = os.path.dirname(model_path)
    baseline_entropy = None
    baseline_entropy_path = os.path.join(ckpt_dir, "baseline_entropy.npy")
    if os.path.exists(baseline_entropy_path):
        baseline_entropy = np.load(baseline_entropy_path)

    prototypes = None
    proto_path = os.path.join(ckpt_dir, "class_prototypes.pt")
    if os.path.exists(proto_path):
        prototypes = torch.load(proto_path, map_location=device, weights_only=True)
        print(f"Loaded class prototypes: {prototypes.shape}")

    # Methods to evaluate
    methods_to_eval = eval_cfg.get("methods", ["bn_adapt", "tent", "eata", "cotta", "sar", "note", "tta_tc"])
    adapt_cfg = {
        "num_classes": num_classes,
        "adapt_lr": eval_cfg.get("adapt_lr", 1e-3),
        "ema_momentum": eval_cfg.get("ema_momentum", 0.999),
        "restore_prob": eval_cfg.get("restore_prob", 0.01),
        "fisher_alpha": eval_cfg.get("fisher_alpha", 2000.0),
        "buffer_size": num_classes * 10,
        **eval_cfg.get("tta", {}),
    }

    method_classes = {
        "bn_adapt": ("B3: BN-Adapt", BNAdapt),
        "tent": ("B4: Tent", Tent),
        "eata": ("B5: EATA", EATA),
        "cotta": ("B6: CoTTA", CoTTA),
        "sar": ("B7: SAR", SAR),
        "note": ("B8: NOTE", NOTE),
    }

    for method_key in methods_to_eval:
        if method_key in method_classes:
            name, MethodClass = method_classes[method_key]
            print(f"\n=== {name} ===")
            method_model = copy.deepcopy(model)
            method_model.to(device)
            method = MethodClass(method_model, adapt_cfg)
            labels, preds, t = evaluate_tta_method(method, test_loader, device, name)
            m = compute_metrics(labels, preds)
            results[method_key] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "adapt_time_s": t,
            }
            print(f"Accuracy: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}, Time: {t:.1f}s")
            del method_model

        elif method_key == "tta_tc":
            print("\n=== B10: TTA-TC (Ours) ===")
            tta_model = copy.deepcopy(model)
            tta_model.to(device)
            tta_cfg = {
                "num_classes": num_classes,
                **eval_cfg.get("tta", {}),
            }
            engine = TTAEngine(tta_model, tta_cfg, prototypes=prototypes)
            if baseline_entropy is not None:
                engine.set_baseline_entropy(baseline_entropy)
            labels, preds, t = evaluate_tta_method(engine, test_loader, device, "TTA-TC")
            m = compute_metrics(labels, preds)
            results["tta_tc"] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "adapt_time_s": t,
            }
            print(f"Accuracy: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}, Time: {t:.1f}s")
            del tta_model

    return results


def run_sequential_eval(model_path, eval_cfg, device):
    """Evaluate all methods across sequential test periods (continual TTA)."""
    model, train_cfg, num_classes = load_source_model(model_path, device)

    # Build sequential test loaders
    loaders, _ = build_sequential_test_loaders(eval_cfg["data"])

    # Get source accuracy from training results
    train_dir = os.path.dirname(model_path)
    train_results_path = os.path.join(train_dir, "train_results.json")
    source_acc = None
    if os.path.exists(train_results_path):
        with open(train_results_path) as f:
            source_acc = json.load(f).get("test_accuracy")

    baseline_entropy_path = os.path.join(train_dir, "baseline_entropy.npy")
    baseline_entropy = None
    if os.path.exists(baseline_entropy_path):
        baseline_entropy = np.load(baseline_entropy_path)

    prototypes = None
    proto_path = os.path.join(train_dir, "class_prototypes.pt")
    if os.path.exists(proto_path):
        prototypes = torch.load(proto_path, map_location=device, weights_only=True)
        print(f"Loaded class prototypes: {prototypes.shape}")

    # Methods
    methods_to_eval = eval_cfg.get("methods", ["static", "tent", "eata", "tta_tc"])
    all_results = {}

    for method_key in methods_to_eval:
        print(f"\n{'='*60}")
        print(f"Sequential evaluation: {method_key}")
        print(f"{'='*60}")

        tracker = MetricsTracker(source_accuracy=source_acc)

        if method_key == "static":
            for period_name, test_loader in loaders:
                labels, preds = evaluate_static(model, test_loader, device)
                m = tracker.add_period(period_name, labels, preds)
                print(f"  {period_name}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, ARR={m['arr']:.4f}")

        elif method_key == "tta_tc":
            tta_model = copy.deepcopy(model).to(device)
            tta_cfg = {"num_classes": num_classes, **eval_cfg.get("tta", {})}
            engine = TTAEngine(tta_model, tta_cfg, prototypes=prototypes)
            if baseline_entropy is not None:
                engine.set_baseline_entropy(baseline_entropy)

            # Continual: do NOT reset between periods
            for period_name, test_loader in loaders:
                labels, preds, t = evaluate_tta_method(engine, test_loader, device, f"TTA-TC@{period_name}")
                m = tracker.add_period(period_name, labels, preds)
                print(f"  {period_name}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, ARR={m['arr']:.4f}, Time={t:.1f}s")
            del tta_model

        else:
            # General TTA baselines
            from tta_tc.baselines import Tent, EATA, CoTTA, SAR, NOTE, BNAdapt
            method_map = {
                "bn_adapt": BNAdapt, "tent": Tent, "eata": EATA,
                "cotta": CoTTA, "sar": SAR, "note": NOTE,
            }
            if method_key in method_map:
                MethodClass = method_map[method_key]
                method_model = copy.deepcopy(model).to(device)
                adapt_cfg = {"num_classes": num_classes, **eval_cfg.get("tta", {})}
                method = MethodClass(method_model, adapt_cfg)

                for period_name, test_loader in loaders:
                    labels, preds, t = evaluate_tta_method(
                        method, test_loader, device, f"{method_key}@{period_name}"
                    )
                    m = tracker.add_period(period_name, labels, preds)
                    print(f"  {period_name}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, ARR={m['arr']:.4f}")
                del method_model

        aurc = tracker.compute_aurc()
        print(f"  AURC: {aurc:.4f}" if aurc else "  AURC: N/A")
        all_results[method_key] = tracker.summary()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TTA-TC Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["single", "sequential"], default="single")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/eval")
    os.makedirs(output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.mode == "single":
        results = run_single_period_eval(args.checkpoint, cfg, device)
    else:
        results = run_sequential_eval(args.checkpoint, cfg, device)

    # Save results
    results_path = os.path.join(output_dir, f"results_{args.mode}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Time (s)':>10}")
    print("-" * 70)
    for name, m in results.items():
        if isinstance(m, dict) and "accuracy" in m:
            acc = f"{m['accuracy']:.4f}"
            f1 = f"{m['macro_f1']:.4f}"
            t = f"{m.get('adapt_time_s', 0):.1f}"
            print(f"{name:<20} {acc:>10} {f1:>10} {t:>10}")
    print("=" * 70)


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
