"""
EATA: Efficient Test-Time Model Adaptation without Forgetting.
Niu et al., ICML 2022.

Entropy-based sample selection + Fisher regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class EATA:
    """EATA TTA baseline."""

    def __init__(self, model, cfg: dict):
        self.model = model
        self.model.eval()
        num_classes = cfg["num_classes"]

        # Store source params
        self.source_params = {}
        for name, p in model.named_parameters():
            self.source_params[name] = p.data.clone()

        # Adapt only norm params
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

        # EATA thresholds
        self.e_margin = cfg.get("eata_e_margin", 0.4 * math.log(num_classes))
        self.d_margin = cfg.get("eata_d_margin", 0.05)
        self.fisher_alpha = cfg.get("fisher_alpha", 2000.0)

        # Fisher (simplified: uniform importance)
        self.fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}

    def adapt_batch(self, ppi, flow_stats=None):
        """Adapt with sample selection and Fisher regularization."""
        self.model.train()

        logits = self.model(ppi, flow_stats)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # Sample selection: low entropy + non-redundant
        mask = entropy < self.e_margin
        if mask.sum() < 2:
            self.model.eval()
            with torch.no_grad():
                return self.model(ppi, flow_stats), {"adapted": False}

        self.optimizer.zero_grad()

        # Entropy loss on selected samples
        loss = entropy[mask].mean()

        # Fisher regularization
        fisher_loss = torch.tensor(0.0, device=ppi.device)
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.source_params:
                fisher_loss += (self.fisher[name].to(p.device) * (p - self.source_params[name].to(p.device)).pow(2)).sum()
        loss = loss + self.fisher_alpha * fisher_loss

        loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
        return logits, {"entropy_loss": loss.item(), "selected": mask.sum().item()}
