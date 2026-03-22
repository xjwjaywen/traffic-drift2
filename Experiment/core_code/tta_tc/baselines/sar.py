"""
SAR: Towards Stable Test-Time Adaptation in Dynamic Wild World.
Niu et al., ICLR 2023.

Entropy filtering + sharpness-aware minimization for stable TTA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SAR:
    """SAR TTA baseline."""

    def __init__(self, model, cfg: dict):
        self.model = model
        self.model.eval()
        num_classes = cfg["num_classes"]

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

        self.optimizer = torch.optim.SGD(self.params, lr=cfg.get("adapt_lr", 1e-3))
        self.e_margin = 0.4 * math.log(num_classes)
        self.rho = cfg.get("sar_rho", 0.05)  # SAM radius

    def adapt_batch(self, ppi, flow_stats=None):
        """Adapt with entropy filtering + SAM."""
        self.model.train()

        logits = self.model(ppi, flow_stats)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # Filter reliable samples
        mask = entropy < self.e_margin
        if mask.sum() < 2:
            self.model.eval()
            with torch.no_grad():
                return self.model(ppi, flow_stats), {"adapted": False}

        # SAM: first step (ascend)
        self.optimizer.zero_grad()
        loss = entropy[mask].mean()
        loss.backward()

        # Compute perturbation
        with torch.no_grad():
            grads = []
            for p in self.params:
                if p.grad is not None:
                    grads.append(p.grad.clone())
                else:
                    grads.append(torch.zeros_like(p))

            grad_norm = torch.sqrt(sum((g**2).sum() for g in grads)) + 1e-12
            for p, g in zip(self.params, grads):
                eps = self.rho * g / grad_norm
                p.add_(eps)

        # SAM: second step (descend at perturbed point)
        self.optimizer.zero_grad()
        logits2 = self.model(ppi[mask], flow_stats[mask] if flow_stats is not None else None)
        probs2 = F.softmax(logits2, dim=1)
        loss2 = -(probs2 * torch.log(probs2 + 1e-8)).sum(dim=1).mean()
        loss2.backward()

        # Restore and step
        with torch.no_grad():
            for p, g in zip(self.params, grads):
                eps = self.rho * g / grad_norm
                p.sub_(eps)  # undo perturbation

        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
        return logits, {"sar_loss": loss2.item(), "selected": mask.sum().item()}
