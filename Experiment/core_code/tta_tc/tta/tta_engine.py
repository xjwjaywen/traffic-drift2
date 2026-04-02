"""
TTA-TC: Test-Time Adaptation Engine (v7 — RevIN Instance Normalization).

The model itself contains an InstanceNorm1d layer at the input, which
performs per-sample, per-channel normalization. This removes distributional
drift at the input level (both focal drift like QUIC certificate rotation
and diffuse drift like TLS application behavior changes).

Since the normalization is built into the model architecture, the TTA engine
simply runs inference with the frozen model. No gradient updates needed.
"""
import torch
import torch.nn.functional as F


class TTAEngine:
    """
    RevIN-based TTA: model has built-in input instance normalization.
    Engine just runs frozen model inference.
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

        self.prototypes = None
        if prototypes is not None:
            self.prototypes = F.normalize(prototypes.to(self.device), dim=1)

        self.step_count = 0

    def set_fisher(self, dataloader):
        """No-op."""
        pass

    def set_baseline_entropy(self, baseline):
        """No-op."""
        pass

    @torch.no_grad()
    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        Direct inference through model with built-in input normalization.
        """
        info = {"total_samples": ppi.size(0), "adapted": True, "method": "revin"}

        logits = self.model(ppi, flow_stats)

        self.step_count += 1
        return logits, info

    def reset(self):
        """Reset state."""
        self.step_count = 0
