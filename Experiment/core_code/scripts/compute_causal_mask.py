#!/usr/bin/env python
"""
Compute causal feature mask and base SSL loss from an existing checkpoint.

Use this when you have a trained model but need to generate the causal mask
artifacts without retraining. The artifacts are saved alongside the checkpoint.

Usage:
    python scripts/compute_causal_mask.py \
        --checkpoint outputs/quic22_cnn/best_model.pt \
        --config configs/train_quic22_cnn.yaml \
        --causal-ratio 0.5 \
        --n-folds 5
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta_tc.models import TTATCModel
from tta_tc.ssl_tasks.combined import CombinedSSLLoss
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config
from tta_tc.tta.causal_features import compute_causal_mask, compute_base_ssl_loss


def main():
    parser = argparse.ArgumentParser(
        description="Compute causal mask and base SSL loss from checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True,
                        help="Training config YAML (for data loading)")
    parser.add_argument("--causal-ratio", type=float, default=0.5,
                        help="Fraction of dims to mark causal (default: 0.5)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds for stability analysis")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]
    model_cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(model_cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model: {ckpt['num_classes']} classes, "
          f"hidden_dim={model_cfg['model'].get('hidden_dim', 256)}")

    # Load data
    data_cfg = load_config(args.config)
    _, val_loader, _, num_classes = build_dataloaders(data_cfg["data"])

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    hidden_dim = model_cfg["model"].get("hidden_dim", 256)

    # Compute causal mask
    causal_mask, stability = compute_causal_mask(
        model, val_loader, num_classes, hidden_dim, device,
        n_folds=args.n_folds, causal_ratio=args.causal_ratio,
    )
    torch.save(causal_mask.cpu(), os.path.join(output_dir, "causal_mask.pt"))
    torch.save(stability.cpu(), os.path.join(output_dir, "causal_stability.pt"))

    # Compute base SSL loss
    ssl_cfg = model_cfg.get("ssl", {})
    ssl_loss_fn = CombinedSSLLoss(
        mask_ratio=ssl_cfg.get("mask_ratio", 0.15),
        enable_mpfp=ssl_cfg.get("enable_mpfp", True),
        enable_pop=ssl_cfg.get("enable_pop", True),
        enable_fsr=ssl_cfg.get("enable_fsr", True),
    )
    base_ssl = compute_base_ssl_loss(model, ssl_loss_fn, val_loader, device)
    torch.save(torch.tensor(base_ssl), os.path.join(output_dir, "base_ssl_loss.pt"))

    print(f"\nArtifacts saved to {output_dir}/:")
    print(f"  causal_mask.pt       — {causal_mask.sum().item()}/{hidden_dim} causal dims")
    print(f"  causal_stability.pt  — per-dim stability scores")
    print(f"  base_ssl_loss.pt     — {base_ssl:.4f}")


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
