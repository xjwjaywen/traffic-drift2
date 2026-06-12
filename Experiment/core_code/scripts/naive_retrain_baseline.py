"""Naive full fine-tuning with B labeled samples. No replay, no distill."""
import sys, os, copy, csv, torch, numpy as np
from tqdm import tqdm
sys.path.insert(0, ".")
from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config
from sklearn.metrics import f1_score, classification_report

COLLAPSE = [56,163,174,48,38,69,104,47,66,10,109,26]
STABLE = [8,15,44,57,59,62,64,76,94,98,99,107,113,119,128,130,131,132,144,145]

def group_f1(labels, preds, class_ids):
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    f1s = [report[str(c)]["f1-score"] for c in class_ids if str(c) in report and report[str(c)]["support"]>0]
    return float(np.mean(f1s)) if f1s else 0.0

def run(checkpoint, config, test_period, budget, lr, epochs, seed, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")
    cfg = load_config(config)
    cfg["data"]["test_period"] = test_period

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg["model"] = ckpt["config"]["model"]
    cfg["model"]["num_classes"] = ckpt["num_classes"]
    nc = ckpt["num_classes"]

    _, _, loader, _ = build_dataloaders(cfg["data"])

    # Collect all features and labels
    model = TTATCModel(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_ppi, all_labels, all_fs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect"):
            all_ppi.append(batch["ppi"])
            all_labels.append(batch["label"])
            fs = batch.get("flow_stats")
            if fs is not None:
                all_fs.append(fs)
    all_ppi = torch.cat(all_ppi)
    all_labels = torch.cat(all_labels)

    # Select B random samples
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(all_labels), generator=gen)[:budget]
    train_ppi = all_ppi[idx].to(device)
    train_labels = all_labels[idx].to(device)

    results = []
    for mode in ["head_only", "full_model"]:
        model_ft = TTATCModel(cfg["model"]).to(device)
        model_ft.load_state_dict(ckpt["model_state_dict"])

        if mode == "head_only":
            for p in model_ft.encoder.parameters():
                p.requires_grad_(False)
            for p in model_ft.ssl_head.parameters():
                p.requires_grad_(False)
            params = list(model_ft.cls_head.parameters())
        else:
            params = list(model_ft.parameters())

        optimizer = torch.optim.Adam(params, lr=lr)
        model_ft.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            logits = model_ft(train_ppi)
            loss = torch.nn.functional.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()

        model_ft.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(all_ppi), 256):
                batch_ppi = all_ppi[i:i+256].to(device)
                preds = model_ft(batch_ppi).argmax(1).cpu()
                all_preds.append(preds)
        preds = torch.cat(all_preds).numpy()
        labels = all_labels.numpy()

        macro = f1_score(labels, preds, average="macro", zero_division=0)
        collapse = group_f1(labels, preds, COLLAPSE)
        stable = group_f1(labels, preds, STABLE)
        collapsed = sum(1 for c in COLLAPSE if (labels==c).sum()>0 and (preds[labels==c]==c).sum()/(labels==c).sum()<0.1)

        print(f"{mode} B={budget} seed={seed}: macro={macro:.4f} collapse={collapse:.4f} stable={stable:.4f} collapsed={collapsed}")
        results.append({"mode": mode, "budget": budget, "seed": seed,
                        "macro_f1": macro, "collapse_f1": collapse,
                        "stable_f1": stable, "collapsed": collapsed})

    with open(os.path.join(output_dir, "naive_baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

if __name__ == "__main__":
    for seed in range(3):
        run("outputs/tls22_cnn/best_model.pt", "configs/eval_tls22.yaml",
            "M-2022-12", 1000, 1e-3, 30, seed, "outputs/naive_baseline_M12")
