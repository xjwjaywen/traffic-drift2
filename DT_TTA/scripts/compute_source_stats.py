"""
Pre-compute per-channel GroupNorm input statistics on the SOURCE training data.

These are needed by DT-TTA methods to compute drift scores at test time.
Saved as a torch .pt file under DT_TTA/outputs/source_stats/.
"""
import argparse
import os
import sys
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "DT_TTA"))

from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config
from methods.topology import collect_groupnorm_input_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Eval config (configs/eval_quic22.yaml etc.)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--output-path", default=None,
                        help="Where to save .pt; default DT_TTA/outputs/source_stats/<dataset>.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]
    train_cfg["model"]["num_classes"] = ckpt["num_classes"]
    model = TTATCModel(train_cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Use validation loader of source training period to estimate stats
    train_loader, val_loader, _, _ = build_dataloaders(cfg["data"])

    print(f"Collecting GroupNorm input stats over <= {args.max_batches} batches "
          f"of source train period {cfg['data'].get('train_period')}")
    stats = collect_groupnorm_input_stats(
        model, train_loader, device, max_batches=args.max_batches)

    print(f"Collected {len(stats)} layer(s):")
    for name, s in stats.items():
        print(f"  {name}: C={s['mean'].shape[0]}, n_samples={s['n']}, "
              f"mean range=[{s['mean'].min():.4f}, {s['mean'].max():.4f}]")

    out_path = args.output_path
    if out_path is None:
        ds = cfg["data"]["dataset"]
        out_dir = os.path.join(_REPO_ROOT, "DT_TTA", "outputs", "source_stats")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ds}_source_stats.pt")

    # Convert numpy to plain dict for torch.save
    payload = {name: {"mean": torch.tensor(v["mean"]),
                      "var": torch.tensor(v["var"]),
                      "n": int(v["n"])}
               for name, v in stats.items()}
    torch.save(payload, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
