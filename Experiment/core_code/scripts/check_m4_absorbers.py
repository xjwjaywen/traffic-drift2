import torch, numpy as np, sys
from tqdm import tqdm
sys.path.insert(0, ".")
from tta_tc.models import TTATCModel
from tta_tc.data.cesnet_loader import build_dataloaders
from tta_tc.utils.config import load_config

cfg = load_config("configs/eval_tls22.yaml")
cfg["data"]["test_period"] = "M-2022-4"
device = torch.device("cuda")
ckpt = torch.load("outputs/tls22_cnn/best_model.pt", map_location=device, weights_only=False)
cfg["model"] = ckpt["config"]["model"]
cfg["model"]["num_classes"] = ckpt["num_classes"]
nc = ckpt["num_classes"]
model = TTATCModel(cfg["model"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

_, _, loader, _ = build_dataloaders(cfg["data"])

confusion = np.zeros((nc, nc), dtype=np.int64)
with torch.no_grad():
    for batch in tqdm(loader, desc="M4 confusion"):
        ppi = batch["ppi"].to(device)
        labels = batch["label"].numpy()
        preds = model(ppi).argmax(1).cpu().numpy()
        for t, p in zip(labels, preds):
            confusion[t][p] += 1

collapse_classes = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
known_absorbers = [96, 46, 2, 14, 45, 105, 5, 71, 156, 13]

print("Can M4 confusion matrix predict absorbers?")
print("{:<8} {:>10} {:>10} {:>12} {:>10}".format("Victim", "M4_top", "M12_absorb", "Match?", "M4_conf%"))
matches = 0
for i, c in enumerate(collapse_classes):
    row = confusion[c].copy()
    row[c] = 0
    top = row.argmax()
    total = confusion[c].sum()
    conf_rate = row[top] / max(total, 1) * 100
    m12_abs = known_absorbers[i] if i < len(known_absorbers) else "?"
    match = "YES" if top == m12_abs else "no"
    if top == m12_abs:
        matches += 1
    print("{:<8} {:>10} {:>10} {:>12} {:>9.1f}%".format(c, top, m12_abs, match, conf_rate))
print("Match rate: {}/{}".format(matches, len(collapse_classes)))
