"""Unsupervised logit adjustment: penalize absorbers via frequency ratio."""
import sys, os, torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
sys.path.insert(0, ".")
from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config

COLLAPSE = [56,163,174,48,38,69,104,47,66,10,109,26]
STABLE = [8,15,44,57,59,62,64,76,94,98,99,107,113,119,128,130,131,132,144,145]

def group_f1(labels, preds, class_ids):
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    f1s = [report[str(c)]["f1-score"] for c in class_ids if str(c) in report and report[str(c)]["support"]>0]
    return float(np.mean(f1s)) if f1s else 0.0

device = torch.device("cuda")
cfg = load_config("configs/eval_tls22.yaml")
ckpt = torch.load("outputs/tls22_cnn/best_model.pt", map_location=device, weights_only=False)
cfg["model"] = ckpt["config"]["model"]
cfg["model"]["num_classes"] = ckpt["num_classes"]
nc = ckpt["num_classes"]
model = TTATCModel(cfg["model"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Step 1: Get reference period (M4) prediction frequencies
cfg["data"]["test_period"] = "M-2022-4"
_, _, ref_loader, _ = build_dataloaders(cfg["data"])
ref_counts = np.zeros(nc)
with torch.no_grad():
    for batch in tqdm(ref_loader, desc="M4 ref"):
        preds = model(batch["ppi"].to(device)).argmax(1).cpu().numpy()
        for p in preds:
            ref_counts[p] += 1
p_ref = ref_counts / ref_counts.sum() + 1e-8

# Step 2: Get target period (M12) logits and labels
cfg["data"]["test_period"] = "M-2022-12"
_, _, tgt_loader, _ = build_dataloaders(cfg["data"])
all_logits, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(tgt_loader, desc="M12 tgt"):
        logits = model(batch["ppi"].to(device))
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
all_logits = torch.cat(all_logits).numpy()
all_labels = torch.cat(all_labels).numpy()

# Target frequencies
tgt_preds = all_logits.argmax(1)
tgt_counts = np.zeros(nc)
for p in tgt_preds:
    tgt_counts[p] += 1
p_tgt = tgt_counts / tgt_counts.sum() + 1e-8

# Step 3: Apply logit adjustment with different strengths
print("\n{:<20} {:>10} {:>12} {:>10} {:>10}".format("Method", "Macro-F1", "Collapse-F1", "Stable-F1", "Collapsed"))
print("-" * 64)

# Static baseline
static_preds = all_logits.argmax(1)
macro = f1_score(all_labels, static_preds, average="macro", zero_division=0)
collapse = group_f1(all_labels, static_preds, COLLAPSE)
stable = group_f1(all_labels, static_preds, STABLE)
collapsed = sum(1 for c in COLLAPSE if (all_labels==c).sum()>0 and (static_preds[all_labels==c]==c).sum()/(all_labels==c).sum()<0.1)
print("{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>10}".format("Static", macro, collapse, stable, collapsed))

# Logit adjustment with various alpha
for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
    adjustment = alpha * (np.log(p_ref) - np.log(p_tgt))
    adjusted_logits = all_logits + adjustment[np.newaxis, :]
    adj_preds = adjusted_logits.argmax(1)

    macro = f1_score(all_labels, adj_preds, average="macro", zero_division=0)
    collapse = group_f1(all_labels, adj_preds, COLLAPSE)
    stable = group_f1(all_labels, adj_preds, STABLE)
    collapsed = sum(1 for c in COLLAPSE if (all_labels==c).sum()>0 and (adj_preds[all_labels==c]==c).sum()/(all_labels==c).sum()<0.1)
    print("{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>10}".format(
        f"LogitAdj a={alpha}", macro, collapse, stable, collapsed))

# Saerens EM
print("\n--- Saerens EM ---")
probs = np.exp(all_logits - all_logits.max(1, keepdims=True))
probs = probs / probs.sum(1, keepdims=True)
pi = p_ref.copy()
for iteration in range(20):
    # E-step: adjust posteriors
    adjusted = probs * (pi / p_ref)[np.newaxis, :]
    adjusted = adjusted / adjusted.sum(1, keepdims=True)
    # M-step: update priors
    pi_new = adjusted.mean(0)
    if np.max(np.abs(pi_new - pi)) < 1e-6:
        break
    pi = pi_new

em_preds = adjusted.argmax(1)
macro = f1_score(all_labels, em_preds, average="macro", zero_division=0)
collapse = group_f1(all_labels, em_preds, COLLAPSE)
stable = group_f1(all_labels, em_preds, STABLE)
collapsed = sum(1 for c in COLLAPSE if (all_labels==c).sum()>0 and (em_preds[all_labels==c]==c).sum()/(all_labels==c).sum()<0.1)
print("{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>10}".format(
    f"Saerens EM (iter={iteration+1})", macro, collapse, stable, collapsed))
