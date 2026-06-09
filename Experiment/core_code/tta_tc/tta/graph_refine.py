"""
Graph-based Flow Refinement for Test-Time Adaptation.

Exploits a traffic-specific property: flows from the same application
cluster tightly in feature space, even under drift. By building a k-NN
graph and propagating confident predictions to uncertain neighbors, we
correct misclassified flows using the consensus of their neighborhood.

This is a post-processing step orthogonal to any base TTA method.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class GraphRefineTTA:
    """Test-time graph-based label propagation over flow features.

    Maintains a sliding window of recent flows. For each batch, builds
    a k-NN graph from the window and refines predictions by blending
    each flow's prediction with the similarity-weighted average of its
    neighbors' predictions.
    """

    def __init__(self, model, cfg, prototypes=None, causal_mask=None,
                 position_stats=None):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = next(model.parameters()).device

        self.k = cfg.get("graph_k", 10)
        self.alpha = cfg.get("graph_alpha", 0.4)
        self.temperature = cfg.get("graph_temperature", 0.05)
        self.window_batches = cfg.get("graph_window", 5)

        self._feature_buf = []
        self._logits_buf = []
        self._buf_sizes = []
        self.labels_used = 0

    def reset_period(self):
        self._feature_buf.clear()
        self._logits_buf.clear()
        self._buf_sizes.clear()
        self.labels_used = 0

    def reset(self):
        self.reset_period()

    def set_fisher(self, dataloader):
        pass

    def set_baseline_entropy(self, baseline):
        pass

    @torch.no_grad()
    def adapt_batch(self, ppi, flow_stats=None, labels=None):
        logits, features = self.model(ppi, flow_stats, return_repr=True)
        B = logits.size(0)

        self._feature_buf.append(features)
        self._logits_buf.append(logits)
        self._buf_sizes.append(B)
        while len(self._feature_buf) > self.window_batches:
            self._feature_buf.pop(0)
            self._logits_buf.pop(0)
            self._buf_sizes.pop(0)

        all_features = torch.cat(self._feature_buf, dim=0)
        all_logits = torch.cat(self._logits_buf, dim=0)
        W = all_features.size(0)

        k = min(self.k, W - 1)
        if k < 1:
            return logits, {}

        # k-NN in feature space
        f_norm = F.normalize(all_features, dim=1)
        sim = f_norm @ f_norm.T
        sim.fill_diagonal_(float("-inf"))
        topk_sim, topk_idx = sim.topk(k, dim=1)

        # Neighbor-weighted probability averaging
        probs = F.softmax(all_logits, dim=1)
        neighbor_probs = probs[topk_idx]                        # (W, k, C)
        weights = F.softmax(topk_sim / self.temperature, dim=1) # (W, k)
        neighbor_avg = (weights.unsqueeze(-1) * neighbor_probs).sum(dim=1)

        refined = (1 - self.alpha) * probs + self.alpha * neighbor_avg
        current_refined = refined[-B:]
        return torch.log(current_refined + 1e-8), {}

    @torch.no_grad()
    def adapt_period(self, test_loader, period_name=""):
        self.reset_period()
        all_labels = []
        all_preds = []

        for batch in tqdm(test_loader, desc=f"GraphRefine@{period_name}"):
            ppi = batch["ppi"].to(self.device)
            gt_labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(self.device)

            refined_logits, _ = self.adapt_batch(ppi, flow_stats)
            all_preds.extend(refined_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(gt_labels.numpy())

        return np.array(all_labels), np.array(all_preds)
