"""
NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation.
Gong et al., NeurIPS 2022.

Instance-aware BN + PBRS buffer for non-i.i.d. streams.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from collections import defaultdict


class NOTE:
    """NOTE TTA baseline."""

    def __init__(self, model, cfg: dict):
        self.model = model
        self.model.eval()
        num_classes = cfg["num_classes"]

        # Instance-aware batch normalization
        self._setup_iabn(model)

        # Adapt norm params
        self.params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                for p in m.parameters():
                    p.requires_grad_(True)
                    self.params.append(p)
            else:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.params, lr=cfg.get("adapt_lr", 1e-3))

        # PBRS buffer
        buffer_size = cfg.get("buffer_size", num_classes * 10)
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_labels = []
        self.class_counts = defaultdict(int)

    def _setup_iabn(self, model):
        """Store running stats for instance-aware BN blending."""
        self.bn_stats = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm1d):
                self.bn_stats[name] = {
                    "running_mean": m.running_mean.clone(),
                    "running_var": m.running_var.clone(),
                }

    def _add_to_buffer(self, ppi, pred_labels):
        """Add to PBRS buffer."""
        B = ppi.size(0)
        for i in range(B):
            c = pred_labels[i].item()
            self.class_counts[c] += 1
            entry = ppi[i].detach().cpu()
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(entry)
                self.buffer_labels.append(c)
            else:
                j = np.random.randint(0, self.class_counts[c])
                if j < self.buffer_size:
                    idx = j % len(self.buffer)
                    self.buffer[idx] = entry
                    self.buffer_labels[idx] = c

    def adapt_batch(self, ppi, flow_stats=None):
        """Adapt with PBRS buffer and entropy minimization."""
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
            preds = logits.argmax(dim=1)

        self._add_to_buffer(ppi, preds)

        if len(self.buffer) < 16:
            return logits, {"adapted": False}

        # Sample from buffer
        indices = np.random.choice(len(self.buffer), size=min(64, len(self.buffer)), replace=False)
        buf_ppi = torch.stack([self.buffer[i] for i in indices]).to(ppi.device)

        # Entropy minimization on buffer
        self.model.train()
        self.optimizer.zero_grad()
        buf_logits = self.model(buf_ppi)
        probs = F.softmax(buf_logits, dim=1)
        loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
        return logits, {"entropy_loss": loss.item()}
