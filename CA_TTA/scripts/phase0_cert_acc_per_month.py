"""
CA-TTA Phase 0 — Step 0.2 + 0.3:

Run randomized smoothing on the source 1D-CNN over each test period of
CESNET-TLS-Year22, plot certified accuracy vs month at multiple radii.

If certified accuracy drops significantly over months, the CA-TTA direction
has empirical motivation. Otherwise, the direction dies.
"""
import argparse
import json
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "CA_TTA"))

from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.utils.config import load_config
from methods.smoothing import SmoothedClassifier, certified_accuracy_at_radii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sigma", type=float, default=0.25,
                        help="Smoothing noise std")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--n0", type=int, default=50,
                        help="Samples for top-class selection")
    parser.add_argument("--n", type=int, default=500,
                        help="Samples for p_A estimation")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--max-samples-per-period", type=int, default=500,
                        help="Cap test samples per period (Cohen-style: 500)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["data"]["dataset"]
    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "CA_TTA", "outputs",
                                        "phase0_cert_acc_per_month")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load source model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]
    train_cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(train_cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    num_classes = ckpt["num_classes"]
    print(f"Loaded model with {num_classes} classes")

    smoother = SmoothedClassifier(model, num_classes=num_classes, sigma=args.sigma)

    loaders, _ = build_sequential_test_loaders(cfg["data"])

    print(f"\nSmoothing: sigma={args.sigma}, n0={args.n0}, n={args.n}, "
          f"alpha={args.alpha}, max_samples={args.max_samples_per_period}")
    print(f"Radii to evaluate: {args.radii}")

    summary = {"sigma": args.sigma, "n0": args.n0, "n": args.n,
               "alpha": args.alpha, "radii": args.radii, "periods": {}}

    for period_name, loader in loaders:
        print(f"\n{'='*60}")
        print(f"Period: {period_name}")
        print(f"{'='*60}")
        result = certified_accuracy_at_radii(
            smoother, loader, device,
            radii=args.radii,
            n0=args.n0, n=args.n, alpha=args.alpha,
            max_samples=args.max_samples_per_period,
            desc=f"certify@{period_name}",
        )
        cert_acc = result["certified_accuracy"]
        print(f"  abstain rate: {result['abstain_rate']:.4f}")
        print(f"  n samples: {result['n_total']}")
        for r in args.radii:
            print(f"  certified acc @ r={r:.2f}: {cert_acc[r]:.4f}")

        summary["periods"][period_name] = {
            "certified_accuracy": cert_acc,
            "abstain_rate": result["abstain_rate"],
            "n_total": result["n_total"],
        }

    out_path = os.path.join(args.output_dir,
                             f"{dataset}_cert_acc_sigma{args.sigma}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print final table
    print("\n" + "=" * 80)
    print(f"Certified accuracy vs period — {dataset} (sigma={args.sigma})")
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
