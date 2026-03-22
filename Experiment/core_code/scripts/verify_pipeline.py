"""
Pipeline verification script using synthetic data.

Runs a complete mini end-to-end experiment without requiring CESNET data:
  1. Build a tiny model
  2. Train for 2 epochs on random data
  3. Evaluate static + all baselines + TTA-TC
  4. Run one ablation pass
  5. Save results JSON

Usage (from Experiment/core_code/):
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --output-dir outputs/verify
"""
import argparse
import os
import sys
import json
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Allow imports from parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta_tc.models import TTATCModel
from tta_tc.tta import TTAEngine
from tta_tc.baselines import Tent, EATA, CoTTA, SAR, NOTE, BNAdapt
from tta_tc.ssl_tasks.combined import CombinedSSLLoss
from tta_tc.utils.metrics import compute_metrics, MetricsTracker


# --------------------------------------------------------------------------- #
#  Synthetic data helpers
# --------------------------------------------------------------------------- #

def make_synthetic_loader(n_samples=512, num_classes=10, batch_size=64,
                          add_flow_stats=False, device="cpu"):
    """Return a DataLoader yielding dict batches with synthetic PPI tensors."""
    ppi = torch.randn(n_samples, 3, 30)
    labels = torch.randint(0, num_classes, (n_samples,))
    tensors = [ppi, labels]
    if add_flow_stats:
        flow_stats = torch.randn(n_samples, 6)
        tensors.append(flow_stats)

    ds = TensorDataset(*tensors)

    def collate(batch):
        if add_flow_stats:
            ppi_b, lbl_b, fs_b = zip(*batch)
            return {
                "ppi": torch.stack(ppi_b),
                "label": torch.stack(lbl_b),
                "flow_stats": torch.stack(fs_b),
            }
        else:
            ppi_b, lbl_b = zip(*batch)
            return {"ppi": torch.stack(ppi_b), "label": torch.stack(lbl_b)}

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# --------------------------------------------------------------------------- #
#  Mini training
# --------------------------------------------------------------------------- #

def train_mini(model, ssl_loss_fn, loader, epochs=2, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            ppi = batch["ppi"].to(device)
            labels = batch["label"].to(device)
            fs = batch.get("flow_stats")
            if fs is not None:
                fs = fs.to(device)
            optimizer.zero_grad()
            logits = model(ppi, fs)
            cls_loss = F.cross_entropy(logits, labels)
            ssl_loss, _ = ssl_loss_fn(model, ppi, fs)
            (cls_loss + ssl_loss).backward()
            optimizer.step()
            total_loss += cls_loss.item()
        print(f"  Epoch {epoch}: cls_loss={total_loss/len(loader):.4f}")


# --------------------------------------------------------------------------- #
#  Evaluation helpers
# --------------------------------------------------------------------------- #

def eval_static(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ppi = batch["ppi"].to(device)
            fs = batch.get("flow_stats")
            if fs is not None:
                fs = fs.to(device)
            logits = model(ppi, fs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_labels), np.array(all_preds)


def eval_tta(method, loader, device, name=""):
    all_preds, all_labels = [], []
    for batch in loader:
        ppi = batch["ppi"].to(device)
        fs = batch.get("flow_stats")
        if fs is not None:
            fs = fs.to(device)
        logits, _ = method.adapt_batch(ppi, fs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return np.array(all_labels), np.array(all_preds)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/verify")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    C = args.num_classes
    print(f"\n{'='*55}")
    print("TTA-TC Pipeline Verification (synthetic data)")
    print(f"num_classes={C}, device={device}")
    print(f"{'='*55}")

    # Build tiny model config
    model_cfg = {
        "backbone": "cnn",
        "hidden_dim": 64,
        "num_classes": C,
        "norm_type": "gn",
        "num_groups": 4,
        "flow_stats_dim": 0,
        "enable_mpfp": True,
        "enable_pop": True,
        "enable_fsr": True,
        "fsr_target_dim": 6,
    }

    # ---------- Train ----------
    print("\n[1/4] Training source model (2 epochs on synthetic data)...")
    train_loader = make_synthetic_loader(512, C, batch_size=64)
    val_loader   = make_synthetic_loader(256, C, batch_size=64)
    test_loader  = make_synthetic_loader(256, C, batch_size=64)

    model = TTATCModel(model_cfg).to(device)
    ssl_loss_fn = CombinedSSLLoss(alpha=0.2, beta=0.1, mask_ratio=0.15)
    train_mini(model, ssl_loss_fn, train_loader, epochs=2, device=device)

    # Save baseline entropy
    model.eval()
    all_logits, all_lbls = [], []
    with torch.no_grad():
        for batch in val_loader:
            ppi = batch["ppi"].to(device)
            logits = model(ppi)
            all_logits.append(logits.cpu())
            all_lbls.extend(batch["label"].numpy())
    all_logits = torch.cat(all_logits)
    probs = F.softmax(all_logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(1)
    baseline_entropy = np.zeros(C)
    for c in range(C):
        mask = np.array(all_lbls) == c
        if mask.sum() > 0:
            baseline_entropy[c] = entropy[mask].mean().item()

    # ---------- Evaluate all methods ----------
    print("\n[2/4] Evaluating all methods on synthetic test data...")
    results = {}
    tta_cfg = {
        "num_classes": C,
        "adapt_lr": 1e-4,
        "adapt_mode": "encoder",
        "adapt_batch_size": 32,
        "ema_momentum": 0.999,
        "restore_prob": 0.01,
        "fisher_alpha": 2000.0,
        "mask_ratio": 0.15,
        "enable_mpfp": True,
        "enable_pop": True,
        "enable_fsr": True,
        "ssl_alpha": 0.2,
        "ssl_beta": 0.1,
        "entropy_threshold": 0.5,
        "energy_threshold": -5.0,
        "entropy_filter_ratio": 0.4,
        "buffer_size": C * 10,
        "adapt_steps": 1,
        "abrupt_adapt_steps": 3,
    }

    # Static
    labels, preds = eval_static(model, test_loader, device)
    m = compute_metrics(labels, preds)
    results["static"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    print(f"  static    acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")

    methods = {
        "bn_adapt": BNAdapt,
        "tent":     Tent,
        "eata":     EATA,
        "cotta":    CoTTA,
        "sar":      SAR,
        "note":     NOTE,
    }
    for key, Cls in methods.items():
        m_model = copy.deepcopy(model).to(device)
        method = Cls(m_model, tta_cfg)
        labels, preds = eval_tta(method, test_loader, device, key)
        m = compute_metrics(labels, preds)
        results[key] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        print(f"  {key:<12} acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
        del m_model

    # TTA-TC
    tta_model = copy.deepcopy(model).to(device)
    engine = TTAEngine(tta_model, tta_cfg)
    engine.set_baseline_entropy(baseline_entropy)
    labels, preds = eval_tta(engine, test_loader, device, "tta_tc")
    m = compute_metrics(labels, preds)
    results["tta_tc"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    print(f"  {'tta_tc':<12} acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
    del tta_model

    # ---------- Sequential (2 fake periods) ----------
    print("\n[3/4] Sequential evaluation (2 periods)...")
    tracker = MetricsTracker(source_accuracy=results["static"]["accuracy"])
    tta_model2 = copy.deepcopy(model).to(device)
    engine2 = TTAEngine(tta_model2, tta_cfg)
    engine2.set_baseline_entropy(baseline_entropy)
    for period in ["P1", "P2"]:
        loader_p = make_synthetic_loader(256, C, batch_size=64)
        labels_p, preds_p = eval_tta(engine2, loader_p, device, f"tta_tc@{period}")
        pm = tracker.add_period(period, labels_p, preds_p)
        print(f"  {period}: acc={pm['accuracy']:.4f} arr={pm['arr']:.4f}")
    print(f"  AURC={tracker.compute_aurc():.4f}")
    del tta_model2

    seq_results = {"tta_tc": tracker.summary()}

    # ---------- Ablation ----------
    print("\n[4/4] Ablation: SSL task selection (MPFP-only vs all tasks)...")
    abl_results = {}
    for name, abl_cfg in [
        ("mpfp_only", {"enable_mpfp": True,  "enable_pop": False, "enable_fsr": False}),
        ("pop_only",  {"enable_mpfp": False, "enable_pop": True,  "enable_fsr": False}),
        ("all_tasks", {"enable_mpfp": True,  "enable_pop": True,  "enable_fsr": True}),
    ]:
        m_model = TTATCModel(model_cfg).to(device)
        m_model.load_state_dict(model.state_dict())
        cfg_i = {**tta_cfg, **abl_cfg}
        eng = TTAEngine(m_model, cfg_i)
        eng.set_baseline_entropy(baseline_entropy)
        labels_a, preds_a = eval_tta(eng, test_loader, device, name)
        m = compute_metrics(labels_a, preds_a)
        abl_results[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                              "settings": abl_cfg}
        print(f"  {name:<15} acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
        del m_model

    # ---------- Save ----------
    print(f"\nSaving results to {args.output_dir}/...")
    with open(os.path.join(args.output_dir, "results_single.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "results_sequential.json"), "w") as f:
        json.dump(seq_results, f, indent=2)
    with open(os.path.join(args.output_dir, "ablation_ssl_tasks.json"), "w") as f:
        json.dump(abl_results, f, indent=2)

    print("\n" + "="*55)
    print("VERIFICATION PASSED — all components functional")
    print("="*55)
    print(f"Results: {args.output_dir}/")


if __name__ == "__main__":
    main()
