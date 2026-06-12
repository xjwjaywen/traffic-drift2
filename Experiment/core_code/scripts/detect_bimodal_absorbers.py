import sys, os, torch, numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
sys.path.insert(0, ".")
from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config

cfg = load_config("configs/eval_tls22.yaml")
cfg["data"]["test_period"] = "M-2022-12"
device = torch.device("cuda")
ckpt = torch.load("outputs/tls22_cnn/best_model.pt", map_location=device, weights_only=False)
cfg["model"] = ckpt["config"]["model"]
cfg["model"]["num_classes"] = ckpt["num_classes"]
nc = ckpt["num_classes"]
model = TTATCModel(cfg["model"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

_, _, loader, _ = build_dataloaders(cfg["data"])

all_features, all_preds = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc="Collecting"):
        ppi = batch["ppi"].to(device)
        logits, feats = model(ppi, return_repr=True)
        all_features.append(feats.cpu().numpy().astype(np.float64))
        all_preds.append(logits.argmax(1).cpu().numpy())

features = np.concatenate(all_features)
preds = np.concatenate(all_preds)

known_absorbers = {96, 46, 2, 14, 45, 105, 5, 71, 156, 13}
results = []

for c in tqdm(range(nc), desc="GMM fitting"):
    mask = preds == c
    n = mask.sum()
    if n < 50:
        continue
    X = features[mask]
    try:
        gmm1 = GaussianMixture(n_components=1, reg_covar=1e-4, random_state=0).fit(X)
        gmm2 = GaussianMixture(n_components=2, reg_covar=1e-4, random_state=0).fit(X)
        bic_diff = gmm1.bic(X) - gmm2.bic(X)
        labels_2 = gmm2.predict(X)
        if len(set(labels_2)) > 1:
            sil = silhouette_score(X, labels_2, sample_size=min(5000, n))
        else:
            sil = -1
    except Exception as e:
        continue
    is_absorber = "ABSORBER" if c in known_absorbers else ""
    results.append((c, n, bic_diff, sil, is_absorber))

results.sort(key=lambda x: x[2], reverse=True)

print("\nTop 20 most bimodal classes (candidate absorbers):")
print("{:<8} {:>8} {:>12} {:>10} {:>10}".format("Class", "Samples", "BIC_diff", "Silhouette", "Known?"))
print("-" * 50)
for c, n, bic, sil, tag in results[:20]:
    print("{:<8} {:>8} {:>12.0f} {:>10.3f} {:>10}".format(c, n, bic, sil, tag))

for k in [10, 15, 20, 30]:
    top_k = set(r[0] for r in results[:k])
    hits = top_k & known_absorbers
    print("Top-{}: {}/{} absorbers detected ({})".format(k, len(hits), len(known_absorbers), hits))
