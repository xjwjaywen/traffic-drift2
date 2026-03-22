"""Combined SSL loss for TTA-TC."""
import torch
import torch.nn as nn
from .mpfp import MPFPTask
from .pop import POPTask
from .fsr import FSRTask


class CombinedSSLLoss(nn.Module):
    """
    Combined self-supervised loss:
        L_ssl = L_MPFP + alpha * L_POP + beta * L_FSR
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.1,
        mask_ratio: float = 0.15,
        enable_mpfp: bool = True,
        enable_pop: bool = True,
        enable_fsr: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.enable_mpfp = enable_mpfp
        self.enable_pop = enable_pop
        self.enable_fsr = enable_fsr

        if enable_mpfp:
            self.mpfp = MPFPTask(mask_ratio=mask_ratio)
        if enable_pop:
            self.pop = POPTask()
        if enable_fsr:
            self.fsr = FSRTask()

    def forward(self, model, ppi, flow_stats=None):
        """
        Compute combined SSL loss.

        Args:
            model: TTATCModel instance
            ppi: (B, 3, 30) original PPI
            flow_stats: optional flow-level stats
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        total_loss = torch.tensor(0.0, device=ppi.device)
        loss_dict = {}
        backbone_type = model.backbone_type

        # === MPFP ===
        if self.enable_mpfp:
            mask_indices = self.mpfp.create_mask(ppi.size(0), ppi.device)
            masked_ppi, targets = self.mpfp.mask_input(ppi, mask_indices)
            l_mpfp = self.mpfp.compute_loss(
                model.ssl_head, model.encoder, masked_ppi, targets,
                mask_indices, flow_stats, backbone_type
            )
            total_loss = total_loss + l_mpfp
            loss_dict["mpfp"] = l_mpfp.item()

        # === POP ===
        if self.enable_pop:
            shuffled_ppi, perm_labels = self.pop.shuffle_segments(ppi)
            if backbone_type == "transformer":
                cls_repr = model.encoder(shuffled_ppi, flow_stats, return_all_tokens=False)
            else:
                cls_repr = model.encoder(shuffled_ppi, flow_stats)
            l_pop = self.pop.compute_loss(model.ssl_head, cls_repr, perm_labels)
            total_loss = total_loss + self.alpha * l_pop
            loss_dict["pop"] = l_pop.item()

        # === FSR ===
        if self.enable_fsr:
            if backbone_type == "transformer":
                cls_repr = model.encoder(ppi, flow_stats, return_all_tokens=False)
            else:
                cls_repr = model.encoder(ppi, flow_stats)
            l_fsr = self.fsr.compute_loss(model.ssl_head, cls_repr, ppi)
            total_loss = total_loss + self.beta * l_fsr
            loss_dict["fsr"] = l_fsr.item()

        loss_dict["total_ssl"] = total_loss.item()
        return total_loss, loss_dict
