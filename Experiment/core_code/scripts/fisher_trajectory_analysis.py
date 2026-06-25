"""Fisher discriminant ratio trajectory across M4-M12."""
import sys, os, torch, numpy as np
from tqdm import tqdm
sys.path.insert(0, ".")
from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config

device = torch.device("cuda")
cfg = load_config("configs/eval_tls22.yaml")
ckpt = torch.load("outputs/tls22_cnn/best_model.pt", map_location=device, weights_only=False)
cfg["model"] = ckpt["config"]["model"]
cfg["model"]["num_classes"] = ckpt["num_classes"]
nc = ckpt["num_classes"]
model = TTATCModel(cfg["model"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Victim → primary absorber pairs (from confusion matrix analysis)
pairs = [
    (56, 96), (163, 46), (174, 2), (48, 14), (38, 45), (69, 105),
    (104, 2), (47, 5), (66, 71), (10, 156), (109, 71), (26, 13),
]
collapse_classes = [v for v, _ in pairs]
absorbers = [a for _, a in pairs]

all_results = {}

for period in ["M-2022-4","M-2022-5","M-2022-6","M-2022-7","M-2022-8","M-2022-9","M-2022-10","M-2022-11","M-2022-12"]:
    cfg["data"]["test_period"] = period
    _, _, loader, _ = build_dataloaders(cfg["data"])

    feats_by_class = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=period, leave=False):
            ppi = batch["ppi"].to(device)
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(device)
            labels = batch["label"].numpy()
            _, features = model(ppi, flow_stats, return_repr=True)
            features = features.cpu().numpy()
            for c in set(collapse_classes + absorbers):
                mask = labels == c
                if mask.sum() > 0:
                    feats_by_class.setdefault(c, []).append(features[mask])

    prototypes = {}
    for c, fl in feats_by_class.items():
        all_f = np.concatenate(fl)
        prototypes[c] = all_f.mean(axis=0)

    period_data = {}
    for v, a in pairs:
        if v not in prototypes or a not in prototypes:
            period_data[v] = None
            continue
        va_dist = np.linalg.norm(prototypes[v] - prototypes[a])
        v_feats = np.concatenate(feats_by_class[v])
        a_feats = np.concatenate(feats_by_class[a])
        v_spread = np.mean(np.linalg.norm(v_feats - prototypes[v], axis=1))
        a_spread = np.mean(np.linalg.norm(a_feats - prototypes[a], axis=1))
        fisher = va_dist / (v_spread + a_spread + 1e-8)
        period_data[v] = fisher
    all_results[period] = period_data

# Print trajectory table
periods = sorted(all_results.keys(), key=lambda p: int(p.split("-")[-1]))
header = "{:<8}".format("Victim") + "".join("{:>8}".format(p.replace("M-2022-","M")) for p in periods)
print(header)
print("-" * len(header))
for v, a in pairs:
    row = "{:<8}".format(v)
    for p in periods:
        val = all_results[p].get(v)
        if val is not None:
            row += "{:>8.3f}".format(val)
        else:
            row += "{:>8}".format("--")
    print(row)

# Correlation: Fisher M12 vs CARE recall
care_recalls = {56:0.733, 163:0.455, 174:0.274, 48:0.913, 38:0.534,
                69:0.000, 104:0.624, 47:0.699, 66:0.355, 10:0.623}
print("\nCorrelation: Fisher_M12 vs CARE_recall")
fishers_m12 = []
recalls = []
for v, a in pairs:
    f = all_results["M-2022-12"].get(v)
    r = care_recalls.get(v)
    if f is not None and r is not None:
        fishers_m12.append(f)
        recalls.append(r)
corr = np.corrcoef(fishers_m12, recalls)[0,1]
print("Pearson r = {:.3f}".format(corr))
