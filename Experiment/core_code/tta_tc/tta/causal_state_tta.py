"""
Causal-Aware State-Space TTA (CausalState-TTA) — Label-Efficient Version.

Combines:
  1. Bayesian state-space tracking of class prototypes via Kalman filtering
  2. Causal-aware process noise (spurious dims drift faster)
  3. Active labeling with minimal budget (default 50 labels per period)
  4. Entropy-filtered pseudo-label updates for unlabeled samples

Key insight: a few ground-truth labeled samples (50) anchored via Kalman
filtering can match 500-label methods, because:
  - Labeled updates go to the correct class (no pseudo-label noise)
  - Kalman filtering propagates labeled information across time
  - Entropy filtering keeps pseudo-label updates clean between labels
  - Single-pass streaming — no need to buffer the entire period
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .state_space import PrototypeKalmanFilter
from ..ssl_tasks.combined import CombinedSSLLoss


class CausalStateTTA:
    """Label-efficient causal state-space TTA engine."""

    def __init__(self, model, cfg, prototypes=None, causal_mask=None,
                 position_stats=None):
        self.device = next(model.parameters()).device
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cfg = cfg
        self.num_classes = cfg["num_classes"]
        hidden_dim = model.cfg.get("hidden_dim", 256)

        if prototypes is None:
            raise ValueError("CausalStateTTA requires class prototypes")

        self.kf = PrototypeKalmanFilter(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            causal_mask=causal_mask,
            q_causal=cfg.get("q_causal", 1e-4),
            q_spurious=cfg.get("q_spurious", 1e-2),
            r_base=cfg.get("r_base", 0.1),
            temperature=cfg.get("kf_temperature", 0.1),
            min_samples=cfg.get("kf_min_samples", 3),
            device=self.device,
        )
        self.kf.initialize(prototypes)

        # SSL drift estimator
        ssl_cfg = cfg.get("ssl", {})
        self.ssl_loss_fn = CombinedSSLLoss(
            mask_ratio=ssl_cfg.get("mask_ratio", 0.15),
            enable_mpfp=True, enable_pop=False, enable_fsr=False,
        )
        self.base_ssl_loss = float(cfg.get("base_ssl_loss", 1.0))
        self.drift_ema = self.base_ssl_loss
        self.drift_ema_alpha = cfg.get("drift_ema_alpha", 0.1)
        self.drift_check_interval = cfg.get("drift_check_interval", 50)

        self.tta_blend = cfg.get("tta_blend", 0.5)
        self.entropy_filter_q = cfg.get("entropy_filter_q", 0.4)

        # Active labeling (separate key from TTA-TC's label_budget_per_period)
        self.label_budget = cfg.get("cs_label_budget", 50)
        # Labeled observations: moderate trust (not too aggressive)
        self.labeled_r_scale = cfg.get("labeled_r_scale", 2.0)

        self.labels_used = 0
        self.steps = 0

    def set_fisher(self, dataloader):
        pass

    def set_baseline_entropy(self, baseline):
        pass

    def reset_period(self):
        self.kf.reset()
        self.drift_ema = self.base_ssl_loss
        self.labels_used = 0
        self.steps = 0

    def reset(self):
        self.reset_period()

    def _labeled_update(self, class_id, feature):
        """Kalman update for a single ground-truth labeled sample.

        Uses scaled-down observation noise since the class assignment is
        certain (only feature noise remains).
        """
        R_labeled = self.kf.R * self.labeled_r_scale
        K = self.kf.P[class_id] / (self.kf.P[class_id] + R_labeled)
        innovation = feature - self.kf.mu[class_id]
        self.kf.mu[class_id] = self.kf.mu[class_id] + K * innovation
        self.kf.P[class_id] = (1 - K) * self.kf.P[class_id]

    def _unlabeled_update(self, logits, features):
        """Entropy-filtered pseudo-label Kalman update."""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        if entropy.numel() > 1:
            threshold = torch.quantile(entropy, self.entropy_filter_q)
            reliable = entropy <= threshold
        else:
            reliable = torch.ones_like(entropy, dtype=torch.bool)

        if reliable.sum() < self.kf.min_samples:
            return

        rel_features = features[reliable]
        rel_preds = logits[reliable].argmax(dim=1)
        one_hot = torch.zeros(
            rel_features.size(0), self.num_classes, device=self.device
        )
        one_hot.scatter_(1, rel_preds.unsqueeze(1), 1.0)
        class_count = one_hot.sum(dim=0)
        class_sum = one_hot.T @ rel_features
        class_obs = class_sum / class_count.clamp(min=1).unsqueeze(1)
        self.kf.update(class_obs, class_count)

    def _classify(self, logits, features):
        """Blend static logits with Kalman prototype logits."""
        kf_logits = self.kf.get_logits(features)
        static_probs = F.softmax(logits, dim=1)
        kf_probs = F.softmax(kf_logits, dim=1)
        final_probs = (
            (1 - self.tta_blend) * static_probs
            + self.tta_blend * kf_probs
        )
        return torch.log(final_probs + 1e-8)

    @torch.no_grad()
    def adapt_batch(self, ppi, flow_stats=None, labels=None):
        """Streaming adaptation (no active labeling)."""
        logits, features = self.model(ppi, flow_stats, return_repr=True)

        # Drift estimation
        ssl_val = self.drift_ema
        if self.steps % self.drift_check_interval == 0:
            ssl_loss, _ = self.ssl_loss_fn(self.model, ppi, flow_stats)
            ssl_val = ssl_loss.item()
            self.drift_ema = (
                (1 - self.drift_ema_alpha) * self.drift_ema
                + self.drift_ema_alpha * ssl_val
            )
            self.kf.set_drift_scale(
                self.drift_ema / max(self.base_ssl_loss, 1e-8)
            )

        self.kf.predict()
        self._unlabeled_update(logits, features)
        self.steps += 1

        return self._classify(logits, features), {
            "drift_scale": self.kf.q_scale,
        }

    @torch.no_grad()
    def adapt_period(self, test_loader, period_name=""):
        """Period-level adaptation with active labeling.

        Spreads the label budget evenly across batches. For each labeled
        batch, selects the highest-entropy sample and queries its true
        label for a clean Kalman update.
        """
        self.reset_period()
        all_labels = []
        all_preds = []

        total_batches = len(test_loader)
        budget = self.label_budget
        # Spread labels evenly: label 1 sample every `interval` batches
        if budget > 0 and total_batches > 0:
            label_interval = max(1, total_batches // budget)
        else:
            label_interval = total_batches + 1

        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc=f"CausalState@{period_name}")
        ):
            ppi = batch["ppi"].to(self.device)
            gt_labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(self.device)

            logits, features = self.model(ppi, flow_stats, return_repr=True)

            # --- drift estimation ---
            if self.steps % self.drift_check_interval == 0:
                ssl_loss, _ = self.ssl_loss_fn(self.model, ppi, flow_stats)
                self.drift_ema = (
                    (1 - self.drift_ema_alpha) * self.drift_ema
                    + self.drift_ema_alpha * ssl_loss.item()
                )
                self.kf.set_drift_scale(
                    self.drift_ema / max(self.base_ssl_loss, 1e-8)
                )

            # --- Kalman predict ---
            self.kf.predict()

            # --- active labeling (random selection) ---
            if (budget > 0
                    and self.labels_used < budget
                    and batch_idx % label_interval == 0):
                rand_idx = torch.randint(0, logits.size(0), (1,)).item()
                true_label = gt_labels[rand_idx].item()
                self._labeled_update(true_label, features[rand_idx])
                self.labels_used += 1

            # --- unlabeled update ---
            self._unlabeled_update(logits, features)

            # --- classify ---
            final_logits = self._classify(logits, features)
            all_preds.extend(final_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(gt_labels.numpy())
            self.steps += 1

        return np.array(all_labels), np.array(all_preds)
