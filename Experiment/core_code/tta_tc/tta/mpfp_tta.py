"""
MPFP-TTA: Test-Time Training with Masked Packet Feature Prediction.

The core TTA method for this project. Updates the encoder at test time
using the traffic-specific MPFP reconstruction loss. Unlike entropy-based
TTA (Tent) which can collapse, MPFP provides dense reconstruction
gradients that cannot degenerate.

Requirements:
  - Model must be jointly trained with CLS + MPFP (see train.py)
  - GroupNorm (not BatchNorm) for stability under non-i.i.d. traffic
  - Classification head is frozen; only encoder + SSL head are updated

Key differences from Tent:
  - MPFP loss is reconstruction-based -> no collapse risk
  - Entropy filtering -> only reliable samples contribute gradients
  - Traffic-specific: MPFP captures packet size/direction/IPT drift
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..ssl_tasks.combined import CombinedSSLLoss


class MPFPTTA:
    """Test-Time Training via Masked Packet Feature Prediction."""

    def __init__(self, model, cfg, prototypes=None, causal_mask=None,
                 position_stats=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.num_classes = cfg["num_classes"]

        # Freeze classification head
        for p in model.get_cls_params():
            p.requires_grad_(False)

        # Set up adaptation parameters
        adapt_mode = cfg.get("adapt_mode", "encoder")
        self.adapt_params = model.get_adaptation_params(adapt_mode)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in self.adapt_params:
            p.requires_grad_(True)

        lr = cfg.get("mpfp_lr", cfg.get("adapt_lr", 1e-4))
        self.optimizer = torch.optim.Adam(self.adapt_params, lr=lr)

        # SSL loss
        ssl_cfg = cfg.get("ssl", {})
        self.ssl_loss_fn = CombinedSSLLoss(
            mask_ratio=ssl_cfg.get("mask_ratio", 0.15),
            enable_mpfp=True,
            enable_pop=ssl_cfg.get("enable_pop", False),
            enable_fsr=ssl_cfg.get("enable_fsr", False),
        )

        self.entropy_filter_q = cfg.get("entropy_filter_q", 0.4)
        self.adapt_steps = cfg.get("adapt_steps", 1)
        self.min_reliable = cfg.get("min_reliable", 8)

        # Source weights for optional stochastic restoration
        self.restore_prob = cfg.get("restore_prob", 0.0)
        if self.restore_prob > 0:
            self._source_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }

        self.labels_used = 0
        self.steps = 0

    def set_fisher(self, dataloader):
        pass

    def set_baseline_entropy(self, baseline):
        pass

    def reset_period(self):
        self.labels_used = 0
        self.steps = 0

    def reset(self):
        self.reset_period()

    def _stochastic_restore(self):
        """CoTTA-style: randomly restore some params to source values."""
        if self.restore_prob <= 0 or not hasattr(self, "_source_state"):
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._source_state:
                mask = (
                    torch.rand_like(param, dtype=torch.float32)
                    < self.restore_prob
                )
                param.data[mask] = self._source_state[name][mask].to(
                    param.device
                )

    def adapt_batch(self, ppi, flow_stats=None, labels=None):
        """Adapt encoder via MPFP, then classify with updated model."""
        # Identify reliable samples (low entropy)
        self.model.eval()
        with torch.no_grad():
            logits_pre = self.model(ppi, flow_stats)
            probs = F.softmax(logits_pre, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        if entropy.numel() > 1:
            threshold = torch.quantile(entropy, self.entropy_filter_q)
            reliable = entropy <= threshold
        else:
            reliable = torch.ones_like(entropy, dtype=torch.bool)

        # Backprop MPFP loss on reliable samples
        ssl_val = 0.0
        if reliable.sum() >= self.min_reliable:
            rel_ppi = ppi[reliable]
            rel_fs = flow_stats[reliable] if flow_stats is not None else None

            self.model.train()
            for _ in range(self.adapt_steps):
                self.optimizer.zero_grad()
                ssl_loss, _ = self.ssl_loss_fn(self.model, rel_ppi, rel_fs)
                ssl_loss.backward()
                self.optimizer.step()
            ssl_val = ssl_loss.item()

            if self.restore_prob > 0:
                self._stochastic_restore()

        # Classify with updated encoder
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)

        self.steps += 1
        return logits, {"ssl_loss": ssl_val, "reliable": reliable.sum().item()}

    @torch.no_grad()
    def adapt_period(self, test_loader, period_name=""):
        """Period-level: stream adapt_batch over all batches."""
        self.reset_period()
        all_labels = []
        all_preds = []

        for batch in tqdm(test_loader, desc=f"MPFP-TTA@{period_name}"):
            ppi = batch["ppi"].to(self.device)
            gt_labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(self.device)

            logits, _ = self.adapt_batch(ppi, flow_stats)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(gt_labels.numpy())

        return np.array(all_labels), np.array(all_preds)
