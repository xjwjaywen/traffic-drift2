"""
CoTTA: Continual Test-Time Domain Adaptation.
Wang et al., CVPR 2022.

Augmentation-averaged pseudo-labels + stochastic weight restoration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class CoTTA:
    """CoTTA TTA baseline (simplified for traffic data)."""

    def __init__(self, model, cfg: dict):
        self.model = model
        self.model.eval()

        # Source model
        self.source_model = copy.deepcopy(model)
        self.source_model.eval()
        for p in self.source_model.parameters():
            p.requires_grad_(False)

        # EMA teacher
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

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
        self.restore_prob = cfg.get("restore_prob", 0.01)
        self.ema_momentum = cfg.get("ema_momentum", 0.999)

    def _augment(self, ppi):
        """Simple augmentation: add Gaussian noise to sizes and IPTs."""
        aug = ppi.clone()
        noise = torch.randn_like(ppi[:, 0:1, :]) * 0.05
        aug[:, 0:1, :] = aug[:, 0:1, :] + noise
        noise_ipt = torch.randn_like(ppi[:, 2:3, :]) * 0.05
        aug[:, 2:3, :] = aug[:, 2:3, :] + noise_ipt
        return aug

    def adapt_batch(self, ppi, flow_stats=None):
        """Adapt with augmentation-averaged pseudo-labels."""
        # Get teacher pseudo-labels (averaged over augmentations)
        with torch.no_grad():
            probs_list = [F.softmax(self.teacher(ppi, flow_stats), dim=1)]
            for _ in range(2):
                aug_ppi = self._augment(ppi)
                probs_list.append(F.softmax(self.teacher(aug_ppi, flow_stats), dim=1))
            avg_probs = torch.stack(probs_list).mean(dim=0)
            pseudo_labels = avg_probs.argmax(dim=1)
            confidence = avg_probs.max(dim=1).values

        # Only use high-confidence pseudo-labels
        mask = confidence > 0.5
        if mask.sum() < 2:
            self.model.eval()
            with torch.no_grad():
                return self.model(ppi, flow_stats), {"adapted": False}

        # Cross-entropy with pseudo-labels
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(ppi[mask], flow_stats[mask] if flow_stats is not None else None)
        loss = F.cross_entropy(logits, pseudo_labels[mask])
        loss.backward()
        self.optimizer.step()

        # Stochastic restoration
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    src_param = dict(self.source_model.named_parameters())[name]
                    restore_mask = torch.bernoulli(
                        torch.full_like(param, self.restore_prob)
                    ).bool()
                    param.data[restore_mask] = src_param.data[restore_mask]

        # EMA teacher update
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher.parameters(), self.model.parameters()):
                t_p.data.mul_(self.ema_momentum).add_(s_p.data, alpha=1 - self.ema_momentum)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
        return logits, {"ce_loss": loss.item()}
