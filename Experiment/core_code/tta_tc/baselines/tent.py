"""
Tent: Fully Test-Time Adaptation by Entropy Minimization.
Wang et al., ICLR 2021.

Adapts normalization layer parameters by minimizing prediction entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Tent:
    """Tent TTA baseline."""

    def __init__(self, model, cfg: dict):
        self.model = model
        self.model.eval()

        # Only adapt normalization parameters
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

    def adapt_batch(self, ppi, flow_stats=None):
        """Adapt and classify a batch."""
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(ppi, flow_stats)
        loss = self._entropy_loss(logits)
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
        return logits, {"entropy_loss": loss.item()}

    @staticmethod
    def _entropy_loss(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
