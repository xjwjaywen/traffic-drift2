"""
TTA-TC: Test-Time Adaptation Engine (v6 — Position-Aware Drift Correction + OT).

Two-stage inference-time adaptation:
  1. Drift Detection: compare per-position packet size statistics against
     source baseline; positions with z-score > threshold are flagged.
  2. Input Correction: re-normalize drifted positions to match the source
     distribution (standardize with test stats, rescale with source stats).
     Non-drifted positions are left untouched, preserving all information.
  3. OT Classification: Sinkhorn optimal transport maps the (corrected)
     features toward source prototypes for improved classification.

No gradient updates. No parameter changes. Pure inference-time.

Motivation: Luxemburk & Hynek (TMA 2023) showed that certificate rotation
at packet position ~5 causes 13.48% accuracy drop. Rather than discarding
drifted positions (v5 masking), v6 corrects them: the relative structure
within each class is preserved, only the distributional shift is removed.
"""
import torch
import torch.nn.functional as F
import math


def sinkhorn_transport(cost: torch.Tensor, epsilon: float = 0.1,
                       max_iter: int = 50) -> torch.Tensor:
    """
    Sinkhorn algorithm for entropy-regularized optimal transport.

    Args:
        cost: (K, B) cost matrix between K prototypes and B test samples
        epsilon: regularization strength
        max_iter: Sinkhorn iterations
    Returns:
        gamma: (K, B) optimal transport coupling
    """
    K, B = cost.shape
    log_K = -cost / epsilon
    log_mu = torch.full((K,), -math.log(K), device=cost.device)
    log_nu = torch.full((B,), -math.log(B), device=cost.device)

    log_u = torch.zeros(K, device=cost.device)
    log_v = torch.zeros(B, device=cost.device)

    for _ in range(max_iter):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    return torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))


class TTAEngine:
    """
    Position-Aware Drift Correction + OT Classification.

    Pipeline per batch (all inference, no backward pass):
        1. Per-position z-score drift detection on packet sizes
        2. Correct drifted positions via distribution re-normalization
        3. Forward pass through frozen model with corrected input
        4. OT-based feature transport toward source prototypes
        5. Blend static and OT-adapted predictions
    """

    def __init__(self, model, cfg: dict, prototypes: torch.Tensor = None,
                 position_stats: dict = None):
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.num_classes = cfg["num_classes"]

        # Freeze the entire model
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Source prototypes (frozen)
        if prototypes is not None:
            self.prototypes = F.normalize(prototypes.to(self.device), dim=1)
        else:
            self.prototypes = None

        # Per-position packet size statistics from source domain
        if position_stats is not None:
            self.pos_mean = position_stats["mean"].to(self.device)  # (30,)
            self.pos_std = position_stats["std"].to(self.device)    # (30,)
        else:
            self.pos_mean = None
            self.pos_std = None

        # Hyperparameters
        self.drift_z_threshold = cfg.get("drift_z_threshold", 2.0)
        self.ot_epsilon = cfg.get("ot_epsilon", 0.1)
        self.ot_max_iter = cfg.get("ot_max_iter", 50)
        self.transport_weight = cfg.get("transport_weight", 0.5)
        self.proto_temperature = cfg.get("spa_temperature", 0.1)

        self.step_count = 0

    def set_fisher(self, dataloader):
        """No-op."""
        pass

    def set_baseline_entropy(self, baseline):
        """No-op."""
        pass

    @torch.no_grad()
    def _detect_drifted_positions(self, ppi: torch.Tensor) -> torch.Tensor:
        """
        Detect which packet positions have drifted via z-score test.

        Args:
            ppi: (B, 3, 30) input tensor
        Returns:
            drift_mask: (30,) boolean tensor, True = position drifted
        """
        if self.pos_mean is None:
            return torch.zeros(30, dtype=torch.bool, device=self.device)

        # Batch-level mean of packet sizes at each position
        batch_mean = ppi[:, 0, :].mean(dim=0)  # (30,)

        # Z-score against source distribution
        z_scores = (batch_mean - self.pos_mean) / (self.pos_std + 1e-8)

        return z_scores.abs() > self.drift_z_threshold

    @torch.no_grad()
    def _correct_drifted_positions(self, ppi: torch.Tensor,
                                   drift_mask: torch.Tensor) -> torch.Tensor:
        """
        Re-normalize drifted positions to match source distribution.

        For each drifted position p (packet size channel only):
            x_corrected = (x - μ_test) / σ_test * σ_source + μ_source

        Direction (channel 1) and IPT (channel 2) are left unchanged —
        drift is primarily in packet sizes due to certificate rotation.
        """
        if drift_mask.sum() == 0:
            return ppi
        corrected = ppi.clone()

        # Compute test-batch statistics at drifted positions (packet size only)
        test_sizes = ppi[:, 0, :]  # (B, 30)
        test_mean = test_sizes.mean(dim=0)  # (30,)
        test_std = test_sizes.std(dim=0).clamp(min=1e-8)  # (30,)

        # Correct: standardize with test stats, rescale with source stats
        corrected[:, 0, drift_mask] = (
            (test_sizes[:, drift_mask] - test_mean[drift_mask])
            / test_std[drift_mask]
            * self.pos_std[drift_mask]
            + self.pos_mean[drift_mask]
        )
        return corrected

    @torch.no_grad()
    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        Position-aware drift masking + OT classification.

        Args:
            ppi: (B, 3, 30)
            flow_stats: (B, D) or None
        Returns:
            logits: (B, C) adapted classification output
            info: dict with adaptation stats
        """
        info = {}
        info["total_samples"] = ppi.size(0)

        # Step 1: Detect drifted positions
        drift_mask = self._detect_drifted_positions(ppi)
        num_drifted = drift_mask.sum().item()
        info["drifted_positions"] = num_drifted
        info["drifted_indices"] = drift_mask.nonzero(as_tuple=True)[0].tolist()

        # Step 2: Correct drifted positions (re-normalize to source distribution)
        corrected_ppi = self._correct_drifted_positions(ppi, drift_mask)

        # Step 3: Forward pass with corrected input
        logits, features = self.model(corrected_ppi, flow_stats, return_repr=True)

        # If no prototypes, return cleaned logits directly
        if self.prototypes is None:
            info["adapted"] = num_drifted > 0
            return logits, info

        # Step 4: OT-based classification on cleaned features
        f = F.normalize(features, dim=1)
        K = self.prototypes.size(0)

        sim = torch.matmul(self.prototypes, f.T)           # (K, B)
        cost = 1.0 - sim                                    # cosine distance

        gamma = sinkhorn_transport(cost, self.ot_epsilon, self.ot_max_iter)

        col_sum = gamma.sum(dim=0, keepdim=True).clamp(min=1e-8)
        weights = gamma / col_sum
        transported = F.normalize(torch.matmul(weights.T, self.prototypes), dim=1)

        # Blend original and transported features
        w = self.transport_weight
        blended = F.normalize((1 - w) * f + w * transported, dim=1)
        ot_logits = torch.matmul(blended, self.prototypes.T) / self.proto_temperature

        # Step 5: Blend static and OT predictions
        static_probs = F.softmax(logits, dim=1)
        ot_probs = F.softmax(ot_logits, dim=1)
        final_probs = (1 - w) * static_probs + w * ot_probs
        final_logits = torch.log(final_probs + 1e-8)

        info["adapted"] = True
        info["mean_transport_cost"] = cost.mean().item()
        self.step_count += 1

        return final_logits, info

    def reset(self):
        """Reset state."""
        self.step_count = 0
