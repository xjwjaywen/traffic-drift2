"""
Bayesian State-Space Model for Class Prototype Tracking.

Models the temporal evolution of class prototypes using diagonal Kalman filtering.
Key innovation: causal-aware process noise -- spurious feature dimensions get higher
process noise (expected to drift), while causal dimensions get lower noise (expected
to remain stable).

Reference: STAD (TMLR / ICLR 2026 Journal Track) — adapted for traffic classification.
"""
import torch
import torch.nn.functional as F


class PrototypeKalmanFilter:
    """Diagonal Kalman filter tracking per-class prototypes in feature space.

    State model (per class c, per dimension d):
        mu_c,d(t+1) = mu_c,d(t) + w,   w ~ N(0, Q_d)
        z_c,d(t)    = mu_c,d(t) + v,   v ~ N(0, R_d / n_c)

    Q is dimension-dependent via causal mask:
      - causal dims:   low Q  (stable, not expected to drift)
      - spurious dims: high Q (expected to drift with environment changes)
    """

    def __init__(self, num_classes, hidden_dim, causal_mask=None,
                 q_causal=1e-4, q_spurious=1e-2, r_base=0.1,
                 temperature=0.1, min_samples=2, device="cpu"):
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.min_samples = min_samples
        self.device = device

        if causal_mask is not None:
            self.causal_mask = causal_mask.bool().to(device)
        else:
            self.causal_mask = None

        self.q_base = torch.zeros(hidden_dim, device=device)
        if self.causal_mask is not None:
            self.q_base[self.causal_mask] = q_causal
            self.q_base[~self.causal_mask] = q_spurious
        else:
            self.q_base[:] = (q_causal + q_spurious) / 2

        self.R = torch.full((hidden_dim,), r_base, device=device)
        if self.causal_mask is not None:
            self.R[self.causal_mask] *= 0.5

        self.q_scale = 1.0
        self.mu = None
        self.P = None
        self._source_mu = None

    def initialize(self, prototypes):
        """Initialize from source-domain class prototypes."""
        self.mu = prototypes.clone().to(self.device)
        self._source_mu = prototypes.clone().to(self.device)
        self.P = torch.ones_like(self.mu) * 0.1
        self.q_scale = 1.0

    def predict(self):
        """Time prediction: increase uncertainty by process noise."""
        Q = self.q_base.unsqueeze(0) * self.q_scale
        self.P = self.P + Q

    def update(self, class_observations, class_counts):
        """Measurement update given per-class mean features and sample counts.

        Args:
            class_observations: (C, D) per-class mean features
            class_counts: (C,) number of samples per class
        """
        for c in range(self.num_classes):
            if class_counts[c] < self.min_samples:
                continue
            n = class_counts[c].float()
            K = self.P[c] / (self.P[c] + self.R / n)
            innovation = class_observations[c] - self.mu[c]
            self.mu[c] = self.mu[c] + K * innovation
            self.P[c] = (1 - K) * self.P[c]

    def set_drift_scale(self, scale):
        """Dynamically adjust process noise based on estimated drift magnitude."""
        self.q_scale = float(max(0.1, min(10.0, scale)))

    def get_logits(self, features):
        """Cosine similarity logits from features to current prototypes."""
        f_norm = F.normalize(features, dim=1)
        p_norm = F.normalize(self.mu, dim=1)
        return torch.matmul(f_norm, p_norm.T) / self.temperature

    def get_uncertainty(self):
        """Per-class total uncertainty (trace of diagonal covariance)."""
        return self.P.sum(dim=1)

    def reset(self):
        """Reset to source prototypes."""
        if self._source_mu is not None:
            self.initialize(self._source_mu)
