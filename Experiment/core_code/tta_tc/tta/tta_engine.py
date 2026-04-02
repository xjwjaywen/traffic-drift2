"""
TTA-TC: Test-Time Adaptation Engine (v7 — RevIN: Reversible Instance Normalization).

Inspired by Kim et al. (2022) "Reversible Instance Normalization for Accurate
Time-Series Forecasting", adapted for encrypted traffic classification.

Core idea: normalize each test instance's per-position, per-channel statistics
to match the source training distribution. This absorbs both focal drift
(e.g., QUIC certificate rotation at position ~5) and diffuse drift (e.g., TLS
gradual application behavior changes across all positions/channels).

Pipeline:
  1. Per-instance, per-channel, per-position normalization:
     x_norm = (x - μ_instance) / σ_instance * σ_source + μ_source
  2. Forward pass through frozen model with normalized input
  3. Direct classification (no OT, no gradient updates)

No gradient updates. No parameter changes. Pure inference-time.
"""
import torch
import torch.nn.functional as F


class TTAEngine:
    """
    RevIN-based Test-Time Adaptation for encrypted traffic.

    For each test sample, re-normalizes PPI features to match source
    distribution statistics. Handles all 3 channels (size, direction, IPT)
    and all 30 positions simultaneously.
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

        # Source prototypes (not used in v7 but kept for interface compatibility)
        if prototypes is not None:
            self.prototypes = F.normalize(prototypes.to(self.device), dim=1)
        else:
            self.prototypes = None

        # Per-position, per-channel source statistics
        if position_stats is not None:
            src_mean = position_stats["mean"].to(self.device)
            src_std = position_stats["std"].to(self.device)
            # Handle both old format (30,) and new format (3, 30)
            if src_mean.dim() == 1:
                # Old format: only packet sizes. Expand to (3, 30) with
                # identity transform for channels 1, 2
                self.src_mean = torch.zeros(3, 30, device=self.device)
                self.src_std = torch.ones(3, 30, device=self.device)
                self.src_mean[0] = src_mean
                self.src_std[0] = src_std
            else:
                self.src_mean = src_mean  # (3, 30)
                self.src_std = src_std    # (3, 30)
        else:
            self.src_mean = None
            self.src_std = None

        self.eps = 1e-8
        self.step_count = 0

    def set_fisher(self, dataloader):
        """No-op."""
        pass

    def set_baseline_entropy(self, baseline):
        """No-op."""
        pass

    @torch.no_grad()
    def _revin_normalize(self, ppi: torch.Tensor) -> torch.Tensor:
        """
        RevIN: normalize each batch's per-position statistics to match source.

        Args:
            ppi: (B, 3, 30) raw PPI input
        Returns:
            ppi_norm: (B, 3, 30) normalized PPI
        """
        if self.src_mean is None:
            return ppi

        # Compute batch-level statistics per channel, per position
        batch_mean = ppi.mean(dim=0)  # (3, 30)
        batch_std = ppi.std(dim=0).clamp(min=self.eps)  # (3, 30)

        # Standardize with batch stats, rescale with source stats
        ppi_norm = (ppi - batch_mean) / batch_std * self.src_std + self.src_mean

        return ppi_norm

    @torch.no_grad()
    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        RevIN normalization + direct classification.

        Args:
            ppi: (B, 3, 30)
            flow_stats: (B, D) or None
        Returns:
            logits: (B, C) classification output
            info: dict with adaptation stats
        """
        info = {}
        info["total_samples"] = ppi.size(0)

        # Step 1: RevIN normalize input
        ppi_norm = self._revin_normalize(ppi)

        # Compute how much correction was applied
        diff = (ppi_norm - ppi).abs().mean().item()
        info["mean_correction"] = diff
        info["adapted"] = diff > self.eps

        # Step 2: Forward pass with normalized input
        logits = self.model(ppi_norm, flow_stats)

        info["method"] = "revin"
        self.step_count += 1

        return logits, info

    def reset(self):
        """Reset state."""
        self.step_count = 0
