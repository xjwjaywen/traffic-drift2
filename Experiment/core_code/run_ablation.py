"""
Ablation study runner.

Reads an ablation config, runs TTA-TC with each setting, collects results.

Usage:
    python run_ablation.py --ablation-config configs/ablation_ssl_tasks.yaml \
                           --checkpoint outputs/quic22_cnn/best_model.pt \
                           --output-dir outputs/ablations/ssl_tasks
"""
import argparse
import os
import json
import copy
import torch
import numpy as np
from tqdm import tqdm

from tta_tc.models import TTATCModel
from tta_tc.tta import TTAEngine
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config
from tta_tc.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="TTA-TC Ablation Runner")
    parser.add_argument("--ablation-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/ablations")
    args = parser.parse_args()

    abl_cfg = load_config(args.ablation_config)

    # Load base config
    base_config_path = abl_cfg.get("base_config")
    if base_config_path:
        base_cfg = load_config(base_config_path)
    else:
        base_cfg = load_config("configs/eval_quic22.yaml")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model_cfg = ckpt["config"]["model"]
    model_cfg["num_classes"] = num_classes

    # Load baseline entropy
    baseline_entropy_path = os.path.join(os.path.dirname(args.checkpoint), "baseline_entropy.npy")
    baseline_entropy = None
    if os.path.exists(baseline_entropy_path):
        baseline_entropy = np.load(baseline_entropy_path)

    # Build test loader
    base_cfg["data"]["num_classes"] = num_classes
    _, _, test_loader, _ = build_dataloaders(base_cfg["data"])

    all_results = {}

    for ablation in abl_cfg["ablations"]:
        name = ablation["name"]
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        # Merge ablation settings into base TTA config
        tta_cfg = {
            "num_classes": num_classes,
            **base_cfg.get("tta", {}),
            **ablation.get("tta", {}),
        }

        # Create model and TTA engine
        model = TTATCModel(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        engine = TTAEngine(model, tta_cfg)
        if baseline_entropy is not None:
            engine.set_baseline_entropy(baseline_entropy)

        # Evaluate
        all_preds = []
        all_labels = []
        for batch in tqdm(test_loader, desc=name):
            ppi = batch["ppi"].to(device)
            labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(device)

            logits, info = engine.adapt_batch(ppi, flow_stats)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

        metrics = compute_metrics(all_labels, all_preds)
        all_results[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "settings": ablation.get("tta", {}),
        }
        print(f"  Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

        del model, engine

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*50}")
    print(f"{'Setting':<25} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"{'-'*50}")
    for name, m in all_results.items():
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
    print(f"{'='*50}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
