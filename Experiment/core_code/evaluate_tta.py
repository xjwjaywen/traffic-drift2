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
from tta_tc.tta import TTAEngine, CausalStateTTA
from tta_tc.baselines import (
    Tent, EATA, CoTTA, SAR, NOTE, BNAdapt, MVFC,
    KNNLabeled, FineTuneHead, SupervisedNormAdapt,
)
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


def evaluate_tta_method(method, test_loader, device, method_name="TTA",
                        pass_labels=False):
    """Evaluate a TTA method on test data.

    If pass_labels=True, ground-truth labels are passed to adapt_batch
    (for active-learning methods that query an oracle).
    """
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
        if pass_labels:
            logits, info = method.adapt_batch(ppi, flow_stats, labels=labels)
        else:
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

    position_stats = None
    pos_stats_path = os.path.join(ckpt_dir, "position_stats.pt")
    if os.path.exists(pos_stats_path):
        position_stats = torch.load(pos_stats_path, map_location=device, weights_only=True)
        print(f"Loaded position stats: mean/std for {position_stats['mean'].shape[0]} positions")

    causal_mask = None
    causal_mask_path = os.path.join(ckpt_dir, "causal_mask.pt")
    if os.path.exists(causal_mask_path):
        causal_mask = torch.load(causal_mask_path, map_location=device, weights_only=True)
        print(f"Loaded causal mask: {causal_mask.sum().item()}/{causal_mask.numel()} causal dims")

    base_ssl_loss = None
    base_ssl_path = os.path.join(ckpt_dir, "base_ssl_loss.pt")
    if os.path.exists(base_ssl_path):
        base_ssl_loss = torch.load(base_ssl_path, map_location="cpu", weights_only=True).item()
        print(f"Loaded base SSL loss: {base_ssl_loss:.4f}")

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
        "mvfc": ("B9: MVFC", MVFC),
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
            engine = TTAEngine(tta_model, tta_cfg, prototypes=prototypes,
                               position_stats=position_stats)
            labels, preds, t = evaluate_tta_method(engine, test_loader, device, "TTA-TC")
            m = compute_metrics(labels, preds)
            results["tta_tc"] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "adapt_time_s": t,
            }
            print(f"Accuracy: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}, Time: {t:.1f}s")
            del tta_model

        elif method_key == "causal_state":
            print("\n=== CausalState-TTA ===")
            cs_model = copy.deepcopy(model)
            cs_model.to(device)
            cs_cfg = {
                "num_classes": num_classes,
                **eval_cfg.get("tta", {}),
            }
            if base_ssl_loss is not None:
                cs_cfg["base_ssl_loss"] = base_ssl_loss
            engine = CausalStateTTA(cs_model, cs_cfg, prototypes=prototypes,
                                    causal_mask=causal_mask,
                                    position_stats=position_stats)
            labels, preds, t = evaluate_tta_method(engine, test_loader, device, "CausalState")
            m = compute_metrics(labels, preds)
            results["causal_state"] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "adapt_time_s": t,
            }
            print(f"Accuracy: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}, Time: {t:.1f}s")
            del cs_model

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

    position_stats = None
    pos_stats_path = os.path.join(train_dir, "position_stats.pt")
    if os.path.exists(pos_stats_path):
        position_stats = torch.load(pos_stats_path, map_location=device, weights_only=True)
        print(f"Loaded position stats: mean/std for {position_stats['mean'].shape[0]} positions")

    causal_mask = None
    causal_mask_path = os.path.join(train_dir, "causal_mask.pt")
    if os.path.exists(causal_mask_path):
        causal_mask = torch.load(causal_mask_path, map_location=device, weights_only=True)
        print(f"Loaded causal mask: {causal_mask.sum().item()}/{causal_mask.numel()} causal dims")

    base_ssl_loss = None
    base_ssl_path = os.path.join(train_dir, "base_ssl_loss.pt")
    if os.path.exists(base_ssl_path):
        base_ssl_loss = torch.load(base_ssl_path, map_location="cpu", weights_only=True).item()
        print(f"Loaded base SSL loss: {base_ssl_loss:.4f}")

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

        elif method_key in ("tta_tc", "causal_state", "knn_labeled", "ft_head",
                             "supervised_norm", "selective_norm",
                             "focal_strategy", "diffuse_strategy"):
            tta_model = copy.deepcopy(model).to(device)
            tta_cfg = {"num_classes": num_classes, **eval_cfg.get("tta", {})}
            if method_key == "tta_tc":
                engine = TTAEngine(tta_model, tta_cfg, prototypes=prototypes,
                                   position_stats=position_stats)
            elif method_key == "causal_state":
                if base_ssl_loss is not None:
                    tta_cfg["base_ssl_loss"] = base_ssl_loss
                engine = CausalStateTTA(tta_model, tta_cfg, prototypes=prototypes,
                                        causal_mask=causal_mask,
                                        position_stats=position_stats)
            elif method_key == "knn_labeled":
                engine = KNNLabeled(tta_model, tta_cfg)
            elif method_key == "ft_head":
                engine = FineTuneHead(tta_model, tta_cfg)
            elif method_key == "supervised_norm":
                engine = SupervisedNormAdapt(tta_model, tta_cfg)
            else:
                # DT-TTA methods need source_stats
                import sys as _sys
                _sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                                  "..", "..", "DT_TTA"))
                from methods.strategies import (
                    SelectiveNormAdapt, FocalStrategy, DiffuseStrategy)
                src_stats_path = (eval_cfg.get("tta", {}).get("dt_source_stats")
                                  or eval_cfg.get("dt_source_stats")
                                  or os.environ.get("DT_SOURCE_STATS"))
                if src_stats_path is None or not os.path.exists(src_stats_path):
                    raise RuntimeError(
                        f"DT-TTA method {method_key} requires source stats. "
                        f"Pass via --dt-source-stats CLI flag or "
                        f"DT_SOURCE_STATS env var. Got: {src_stats_path!r}")
                raw = torch.load(src_stats_path, map_location="cpu",
                                 weights_only=False)
                source_stats = {n: {"mean": v["mean"].numpy(),
                                     "var": v["var"].numpy(),
                                     "n": int(v["n"])}
                                for n, v in raw.items()}
                cls = {"selective_norm": SelectiveNormAdapt,
                       "focal_strategy": FocalStrategy,
                       "diffuse_strategy": DiffuseStrategy}[method_key]
                engine = cls(tta_model, tta_cfg, source_stats=source_stats)

            for period_name, test_loader in loaders:
                engine.reset_period()
                t0 = time.time()
                labels, preds = engine.adapt_period(test_loader, period_name)
                t = time.time() - t0
                m = tracker.add_period(period_name, labels, preds)
                print(f"  {period_name}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, "
                      f"ARR={m['arr']:.4f}, Labels used={engine.labels_used}, Time={t:.1f}s")
            del tta_model

        else:
            # General TTA baselines
            from tta_tc.baselines import Tent, EATA, CoTTA, SAR, NOTE, BNAdapt, MVFC
            method_map = {
                "bn_adapt": BNAdapt, "tent": Tent, "eata": EATA,
                "cotta": CoTTA, "sar": SAR, "note": NOTE, "mvfc": MVFC,
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
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method override (e.g., 'tta_tc,knn_labeled,ft_head')")
    parser.add_argument("--sampler", type=str, default=None,
                        choices=["random", "entropy", "margin", "coreset", "class_balanced"],
                        help="Active sampling strategy override")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling and torch")
    parser.add_argument("--dt-source-stats", type=str, default=None,
                        help="Path to source GroupNorm stats .pt for DT-TTA methods")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to results filename")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/eval")
    os.makedirs(output_dir, exist_ok=True)

    # Apply CLI overrides
    if args.methods:
        cfg["methods"] = [m.strip() for m in args.methods.split(",")]
    if args.sampler:
        cfg.setdefault("tta", {})["sampler"] = args.sampler
    if args.seed is not None:
        cfg.setdefault("tta", {})["seed"] = args.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.dt_source_stats:
        cfg.setdefault("tta", {})["dt_source_stats"] = args.dt_source_stats

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
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    results_path = os.path.join(output_dir, f"results_{args.mode}{suffix}.json")
    payload = {
        "results": results,
        "sampler": cfg.get("tta", {}).get("sampler", "random"),
        "seed": cfg.get("tta", {}).get("seed"),
        "methods": cfg.get("methods"),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
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
