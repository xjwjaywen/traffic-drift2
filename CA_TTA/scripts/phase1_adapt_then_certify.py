"""
CA-TTA Phase 1: adapt with chosen method, then evaluate certified accuracy.

Compares CA-TTA against baselines on the question:
  "After adaptation, is the model both more accurate AND more certifiably
  robust under temporal drift?"

Methods supported:
  - static          : no adaptation
  - tent            : vanilla Tent (unsupervised)
  - supervised_norm : 500-label supervised norm-only (no cert objective)
  - ft_head         : 500-label classifier-head fine-tune (no cert objective)
  - ca_tta          : CA-TTA — supervised norm + cert margin objective
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
from tta_tc.baselines import Tent, FineTuneHead, SupervisedNormAdapt
from tta_tc.data.cesnet_loader import build_sequential_test_loaders
from tta_tc.utils.config import load_config

from methods.smoothing import SmoothedClassifier, certified_accuracy_at_radii
from methods.ca_tta import CertAwareNormAdapt


def adapt_with_method(method_name, base_model, test_loader, device, num_classes,
                       cfg):
    """Apply the chosen adaptation method, return the adapted model."""
    adapt_cfg = {"num_classes": num_classes, **cfg.get("tta", {})}

    if method_name == "static":
        return copy.deepcopy(base_model).to(device)

    if method_name == "tent":
        adapted = copy.deepcopy(base_model).to(device)
        tent = Tent(adapted, adapt_cfg)
        for batch in test_loader:
            ppi = batch["ppi"].to(device)
            fs = batch.get("flow_stats")
            if fs is not None:
                fs = fs.to(device)
            _ = tent.adapt_batch(ppi, fs)
        return adapted

    if method_name in ("supervised_norm", "ft_head", "ca_tta"):
        cls = {"supervised_norm": SupervisedNormAdapt,
               "ft_head": FineTuneHead,
               "ca_tta": CertAwareNormAdapt}[method_name]
        adapted = copy.deepcopy(base_model).to(device)
        engine = cls(adapted, adapt_cfg)
        # Run adapt_period — returns (labels, preds), we just need the model
        engine.adapt_period(test_loader, period_name="adapt")
        # CertAwareNormAdapt and the labeled baselines deepcopy internally,
        # so the *original* `adapted` may be untouched. Pull the adapted
        # model from the engine if it stashed it.
        if hasattr(engine, "last_adapted_model") and engine.last_adapted_model is not None:
            return engine.last_adapted_model
        # Fall back: rebuild the engine's frozen-copy approach by checking
        # for known attribute names; otherwise use the engine.model.
        return getattr(engine, "model", adapted)

    raise ValueError(f"Unknown method: {method_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True,
                        choices=["static", "tent", "supervised_norm",
                                 "ft_head", "ca_tta"])
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--max-samples-per-period", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    # CA-TTA hyperparams (only relevant when --method ca_tta)
    parser.add_argument("--ca-lambda", type=float, default=1.0)
    parser.add_argument("--ca-sigma", type=float, default=None,
                        help="CA-TTA training noise sigma; defaults to --sigma")
    parser.add_argument("--ca-n-noise", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["data"]["dataset"]
    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, "CA_TTA", "outputs",
                                        "phase1_adapt_then_certify")
    os.makedirs(args.output_dir, exist_ok=True)

    # Inject CA-TTA cfg
    cfg.setdefault("tta", {})
    cfg["tta"]["ca_lambda"] = args.ca_lambda
    cfg["tta"]["ca_sigma"] = args.ca_sigma if args.ca_sigma is not None else args.sigma
    cfg["tta"]["ca_n_noise"] = args.ca_n_noise
    cfg["tta"]["seed"] = args.seed
    # Active sampler default: random (we are not sweeping samplers here)
    cfg["tta"].setdefault("sampler", "random")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]
    train_cfg["model"]["num_classes"] = ckpt["num_classes"]
    base_model = TTATCModel(train_cfg["model"]).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()
    num_classes = ckpt["num_classes"]

    loaders, _ = build_sequential_test_loaders(cfg["data"])

    summary = {"method": args.method, "sigma": args.sigma,
               "radii": args.radii, "n0": args.n0, "n": args.n,
               "alpha": args.alpha, "seed": args.seed,
               "ca_lambda": args.ca_lambda,
               "ca_sigma": cfg["tta"]["ca_sigma"],
               "ca_n_noise": args.ca_n_noise,
               "periods": {}}

    for period_name, loader in loaders:
        print(f"\n{'='*60}")
        print(f"Period: {period_name} | method: {args.method}")
        print(f"{'='*60}")

        # Cache batches because we need to iterate twice (adapt + certify)
        cached = list(loader)

        adapted = adapt_with_method(args.method, base_model, iter(cached),
                                     device, num_classes, cfg)
        adapted.eval()

        smoother = SmoothedClassifier(adapted, num_classes=num_classes,
                                       sigma=args.sigma)
        result = certified_accuracy_at_radii(
            smoother, iter(cached), device,
            radii=args.radii, n0=args.n0, n=args.n, alpha=args.alpha,
            max_samples=args.max_samples_per_period,
            desc=f"{args.method}+certify@{period_name}",
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

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = os.path.join(
        args.output_dir,
        f"{dataset}_{args.method}_sigma{args.sigma}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 80)
    print(f"{args.method} + cert acc — {dataset} (sigma={args.sigma})")
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
