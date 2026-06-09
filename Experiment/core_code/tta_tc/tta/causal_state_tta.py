"""
Causal-Aware State-Space TTA (CausalState-TTA).

Novel TTA method combining three ideas:
  1. Causal feature identification — distinguish invariant (causal) vs
     environment-dependent (spurious) hidden dimensions
  2. Bayesian state-space tracking — model class prototype evolution via
     Kalman filtering with causal-aware process noise
  3. SSL drift estimation — use MPFP reconstruction error as a proxy for
     drift magnitude to dynamically modulate the Kalman process noise

Key advantages over entropy-minimization TTA (Tent/EATA/SAR):
  - No gradient-based parameter updates -> no error accumulation
  - Explicit temporal dynamics -> captures how drift evolves, not just
    current statistics
  - Causal-aware noise model -> preserves stable features, adapts drifting
    ones

Key advantage over static prototype anchoring (TTA-TC v9):
  - Streaming (single-pass) — no need to buffer the entire test period
  - Prototype tracking is continuous, adapting incrementally per batch
  - Drift-magnitude-aware: fast adaptation when drift is large, conservative
    when traffic is stable
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .state_space import PrototypeKalmanFilter
from ..ssl_tasks.combined import CombinedSSLLoss


class CausalStateTTA:
    """Causal-aware state-space TTA engine.

    Args:
        model: trained TTATCModel (frozen at test time)
        cfg: dict with keys:
            num_classes (required), q_causal, q_spurious, r_base,
            kf_temperature, kf_min_samples, tta_blend, drift_ema_alpha,
            base_ssl_loss, ssl.mask_ratio
        prototypes: (C, hidden_dim) source class prototypes — required
        causal_mask: (hidden_dim,) boolean tensor — optional, defaults to
            uniform noise if absent
        position_stats: unused, kept for interface compatibility
    """

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

        # Kalman filter
        self.kf = PrototypeKalmanFilter(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            causal_mask=causal_mask,
            q_causal=cfg.get("q_causal", 1e-4),
            q_spurious=cfg.get("q_spurious", 1e-2),
            r_base=cfg.get("r_base", 0.1),
            temperature=cfg.get("kf_temperature", 0.1),
            min_samples=cfg.get("kf_min_samples", 2),
            device=self.device,
        )
        self.kf.initialize(prototypes)

        # SSL drift estimator (MPFP only — lightweight)
        ssl_cfg = cfg.get("ssl", {})
        self.ssl_loss_fn = CombinedSSLLoss(
            mask_ratio=ssl_cfg.get("mask_ratio", 0.15),
            enable_mpfp=True,
            enable_pop=False,
            enable_fsr=False,
        )
        self.base_ssl_loss = float(cfg.get("base_ssl_loss", 1.0))
        self.drift_ema = self.base_ssl_loss
        self.drift_ema_alpha = cfg.get("drift_ema_alpha", 0.1)

        # Blending weight: 0 = pure static, 1 = pure Kalman
        self.tta_blend = cfg.get("tta_blend", 0.3)

        # Only compute SSL drift every N batches (expensive forward pass)
        self.drift_check_interval = cfg.get("drift_check_interval", 50)

        self.labels_used = 0
        self.steps = 0

    # -- interface methods expected by evaluate_tta.py --

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

    # -- core adaptation --

    @torch.no_grad()
    def adapt_batch(self, ppi, flow_stats=None, labels=None):
        """Single-batch streaming adaptation.

        Steps:
          1. Frozen forward pass -> features + static logits
          2. MPFP reconstruction error -> drift magnitude -> modulate Q
          3. Kalman predict -> grow uncertainty
          4. Per-class observations from features grouped by prediction
          5. Kalman update -> shrink uncertainty
          6. Classify with probability-level blend of static + Kalman logits

        Returns:
            (logits, info_dict)
        """
        logits, features = self.model(ppi, flow_stats, return_repr=True)

        # --- drift estimation via SSL reconstruction error ---
        # Only compute every N batches to avoid the expensive SSL forward pass
        ssl_val = self.drift_ema
        if self.steps % self.drift_check_interval == 0:
            ssl_loss, _ = self.ssl_loss_fn(self.model, ppi, flow_stats)
            ssl_val = ssl_loss.item()
            self.drift_ema = (
                (1 - self.drift_ema_alpha) * self.drift_ema
                + self.drift_ema_alpha * ssl_val
            )
            drift_ratio = self.drift_ema / max(self.base_ssl_loss, 1e-8)
            self.kf.set_drift_scale(drift_ratio)

        # --- Kalman predict ---
        self.kf.predict()

        # --- per-class observations (vectorized) ---
        pred_classes = logits.argmax(dim=1)
        one_hot = torch.zeros(
            features.size(0), self.num_classes, device=self.device
        )
        one_hot.scatter_(1, pred_classes.unsqueeze(1), 1.0)
        class_count = one_hot.sum(dim=0)                          # (C,)
        class_sum = one_hot.T @ features                          # (C, D)
        safe_count = class_count.clamp(min=1).unsqueeze(1)
        class_obs = class_sum / safe_count                        # (C, D)

        # --- Kalman update ---
        self.kf.update(class_obs, class_count)

        # --- blended classification ---
        kf_logits = self.kf.get_logits(features)
        static_probs = F.softmax(logits, dim=1)
        kf_probs = F.softmax(kf_logits, dim=1)
        final_probs = (
            (1 - self.tta_blend) * static_probs
            + self.tta_blend * kf_probs
        )

        self.steps += 1
        final_logits = torch.log(final_probs + 1e-8)

        return final_logits, {
            "drift_scale": self.kf.q_scale,
            "ssl_loss": ssl_val,
            "drift_ema": self.drift_ema,
        }

    @torch.no_grad()
    def adapt_period(self, test_loader, period_name=""):
        """Period-level adaptation: stream through all batches."""
        self.reset_period()
        all_labels = []
        all_preds = []

        for batch in tqdm(test_loader, desc=f"CausalState@{period_name}"):
            ppi = batch["ppi"].to(self.device)
            labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(self.device)

            final_logits, _ = self.adapt_batch(ppi, flow_stats)
            all_preds.extend(final_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_preds)
