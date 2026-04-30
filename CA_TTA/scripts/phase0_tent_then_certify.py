"""
CA-TTA Phase 0 — Step 0.4:

Run Tent on each test period, then evaluate the adapted model with
randomized smoothing. Compare certified accuracy:
  - vanilla source model (from phase0_cert_acc_per_month.py)
  - Tent-adapted model (this script)

If Tent INCREASES accuracy but DECREASES certified accuracy, the
"Certification-Aware TTA" framing is empirically motivated:
plain TTA hurts certification, so a cert-aware variant is needed.
"""
import argparse
import copy
import json
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "CA_TTA"))

from tta_tc.models import TTATCModel
from tta_tc.baselines import Tent
from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.utils.config import load_config
from methods.smoothing import SmoothedClassifier, certified_accuracy_at_radii


def adapt_with_tent(base_model, test_loader, device, adapt_cfg):
    """Run Tent on the test loader; returns the adapted model."""
    adapted = copy.deepcopy(base_model).to(device)
    tent = Tent(adapted, adapt_cfg)
    for batch in test_loader:
        ppi = batch["ppi"].to(device)
        fs = batch.get("flow_stats")
        if fs is not None:
            fs = fs.to(device)
        _ = tent.adapt_batch(ppi, fs)
    return adapted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--max-samples-per-period", type=int, default=500)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["data"]["dataset"]
    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "CA_TTA", "outputs",
                                        "phase0_tent_then_certify")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]
    train_cfg["model"]["num_classes"] = ckpt["num_classes"]
    base_model = TTATCModel(train_cfg["model"]).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()
    num_classes = ckpt["num_classes"]

    adapt_cfg = {"num_classes": num_classes, **cfg.get("tta", {})}

    loaders, _ = build_sequential_test_loaders(cfg["data"])

    summary = {"sigma": args.sigma, "n0": args.n0, "n": args.n,
               "alpha": args.alpha, "radii": args.radii, "periods": {}}

    for period_name, loader in loaders:
        print(f"\n{'='*60}")
        print(f"Period: {period_name} — adapt with Tent")
        print(f"{'='*60}")
        # Materialize loader (cesnet wrapped loader is iterator-style)
        # We need to iterate it once to adapt, then again to certify.
        # Convert to list-based loader by caching batches.
        cached_batches = list(loader)

        # Step 1: adapt with Tent
        adapted = adapt_with_tent(base_model, iter(cached_batches), device, adapt_cfg)
        adapted.eval()

        # Step 2: smooth + certify on the adapted model
        smoother = SmoothedClassifier(adapted, num_classes=num_classes,
                                       sigma=args.sigma)
        result = certified_accuracy_at_radii(
            smoother, iter(cached_batches), device,
            radii=args.radii,
            n0=args.n0, n=args.n, alpha=args.alpha,
            max_samples=args.max_samples_per_period,
            desc=f"tent+certify@{period_name}",
        )

        cert_acc = result["certified_accuracy"]
        print(f"  abstain rate: {result['abstain_rate']:.4f}")
        for r in args.radii:
            print(f"  certified acc @ r={r:.2f}: {cert_acc[r]:.4f}")

        summary["periods"][period_name] = {
            "certified_accuracy": cert_acc,
            "abstain_rate": result["abstain_rate"],
            "n_total": result["n_total"],
        }

    out_path = os.path.join(args.output_dir,
                             f"{dataset}_tent_cert_acc_sigma{args.sigma}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 80)
    print(f"Tent-adapted certified accuracy vs period — {dataset} (sigma={args.sigma})")
    print("=" * 80)
    header = f"{'Period':<14}" + "".join(f"  r={r:.2f}".rjust(10) for r in args.radii)
    header += "  abstain"
    print(header)
    print("-" * 80)
    for period_name, p_data in summary["periods"].items():
        row = f"{period_name:<14}"
        for r in args.radii:
            row += f"  {p_data['certified_accuracy'][r]:>8.4f}"
        row += f"  {p_data['abstain_rate']:>8.4f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
