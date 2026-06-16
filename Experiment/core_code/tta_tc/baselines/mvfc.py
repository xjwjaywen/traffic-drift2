"""
MVFC: Multi-View Flow Consistency Test-Time Adaptation.

Core idea: at test time, generate K augmented views of each PPI sample
and minimize KL divergence between the views' predictions. Unlike entropy
minimization (which trusts the model's own confidence), consistency does
not require the model to be confident, only stable under flow-level
perturbations that preserve class identity.

Augmentations are traffic-domain specific:
  - Random packet masking (drop 1-2 of 30 packets)
  - Gaussian noise on packet sizes and IPTs (scale-relative)
  - Direction-channel dropout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MVFC:
    """Multi-View Flow Consistency TTA."""

    def __init__(self, model, cfg: dict):
        self.model = model
        self.model.eval()

        # Adapt only normalization layer parameters (Tent-style)
        self.params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                for p in m.parameters():
                    p.requires_grad_(True)
                    self.params.append(p)
            else:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.params, lr=cfg.get("adapt_lr", 1e-4))

        # MVFC hyperparameters
        self.num_views = cfg.get("mvfc_num_views", 2)
        self.mask_prob = cfg.get("mvfc_mask_prob", 0.1)        # P(mask each packet pos)
        self.size_noise_std = cfg.get("mvfc_size_noise", 0.05) # rel. noise std on size
        self.ipt_noise_std = cfg.get("mvfc_ipt_noise", 0.10)   # rel. noise std on IPT
        self.dir_dropout = cfg.get("mvfc_dir_dropout", 0.05)   # P(zero direction)
        self.consistency_weight = cfg.get("mvfc_weight", 1.0)
        self.adapt_steps = cfg.get("adapt_steps", 1)

    @torch.no_grad()
    def _augment(self, ppi):
        """Generate one augmented view of PPI (B, 3, 30)."""
        B, C, L = ppi.shape
        view = ppi.clone()

        # 1. Random packet masking: zero out random positions
        mask = (torch.rand(B, 1, L, device=ppi.device) > self.mask_prob).float()
        view = view * mask

        # 2. Multiplicative Gaussian noise on packet size (channel 0)
        size_noise = 1.0 + torch.randn(B, 1, L, device=ppi.device) * self.size_noise_std
        view[:, 0:1, :] = view[:, 0:1, :] * size_noise

        # 3. Multiplicative Gaussian noise on IPT (channel 2)
        ipt_noise = 1.0 + torch.randn(B, 1, L, device=ppi.device) * self.ipt_noise_std
        view[:, 2:3, :] = view[:, 2:3, :] * ipt_noise

        # 4. Direction channel dropout (channel 1)
        dir_mask = (torch.rand(B, 1, L, device=ppi.device) > self.dir_dropout).float()
        view[:, 1:2, :] = view[:, 1:2, :] * dir_mask

        return view

    def adapt_batch(self, ppi, flow_stats=None):
        """Adapt and classify a batch using multi-view consistency."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                m.train()

        for _ in range(self.adapt_steps):
            self.optimizer.zero_grad()

            # Original (anchor) prediction — no gradient
            with torch.no_grad():
                anchor_logits = self.model(ppi, flow_stats)
                anchor_probs = F.softmax(anchor_logits, dim=1).detach()

            # Generate views and compute consistency loss
            consistency_loss = 0.0
            for _ in range(self.num_views):
                view = self._augment(ppi)
                view_logits = self.model(view, flow_stats)
                view_log_probs = F.log_softmax(view_logits, dim=1)
                # KL(anchor || view): pull views toward anchor
                kl = F.kl_div(view_log_probs, anchor_probs, reduction="batchmean")
                consistency_loss = consistency_loss + kl

            consistency_loss = (consistency_loss / self.num_views) * self.consistency_weight
            consistency_loss.backward()
            self.optimizer.step()

        # Final prediction with adapted norms
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)

        return logits, {"consistency_loss": float(consistency_loss)}
