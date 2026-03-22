"""
BN Adaptation (BN-1): Update BatchNorm statistics at test time.
Schneider et al., NeurIPS 2020.

Simply updates BN running mean/var with test data statistics.
No gradient-based adaptation.
"""
import torch
import torch.nn as nn


class BNAdapt:
    """BN statistics adaptation baseline."""

    def __init__(self, model, cfg: dict):
        self.model = model
        # Set BN layers to training mode (updates running stats)
        # but keep everything else in eval mode
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(False)
            else:
                if hasattr(m, 'eval'):
                    pass  # keep default

    def adapt_batch(self, ppi, flow_stats=None):
        """Just forward pass — BN stats updated automatically in train mode."""
        # BN layers are in train mode, so forward pass updates running stats
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)
        return logits, {"adapted": True}
