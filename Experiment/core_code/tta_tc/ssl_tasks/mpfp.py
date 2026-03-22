"""
Masked Packet Feature Prediction (MPFP) — PRIMARY SSL task.

Randomly masks 15% of PPI positions and predicts:
  - packet size (MSE)
  - direction (BCE)
  - discretized IPT (CE over 8 log bins)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# CESNET PHIST-compatible IPT bins (ms): 0-15, 16-31, 32-63, 64-127, 128-255, 256-511, 512-1024, >1024
IPT_BIN_EDGES = [0.0, 15.0, 31.0, 63.0, 127.0, 255.0, 511.0, 1024.0]


def discretize_ipt(ipt: torch.Tensor) -> torch.Tensor:
    """Convert continuous IPT values (ms) to bin indices (0-7)."""
    edges = torch.tensor(IPT_BIN_EDGES, device=ipt.device)
    # searchsorted returns index where value would be inserted
    bins = torch.searchsorted(edges, ipt.clamp(min=0.0))
    return bins.clamp(max=7).long()


class MPFPTask(nn.Module):
    """
    Masked Packet Feature Prediction.

    For the CNN encoder: works on the raw PPI tensor (B, 3, 30).
    For the Transformer encoder: works on per-token representations.
    """

    def __init__(self, mask_ratio: float = 0.15, seq_len: int = 30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.seq_len = seq_len
        self.num_masked = max(1, int(mask_ratio * seq_len))

    def create_mask(self, batch_size: int, device: torch.device):
        """Create random mask indices for each sample in the batch."""
        # (B, num_masked) — indices of masked positions
        masks = torch.stack([
            torch.randperm(self.seq_len, device=device)[:self.num_masked]
            for _ in range(batch_size)
        ])
        return masks

    def mask_input(self, ppi: torch.Tensor, mask_indices: torch.Tensor):
        """
        Apply masking to PPI input.
        Args:
            ppi: (B, 3, 30) original PPI
            mask_indices: (B, num_masked) positions to mask
        Returns:
            masked_ppi: (B, 3, 30) with masked positions zeroed
            targets: dict with 'size', 'dir', 'ipt' ground truth for masked positions
        """
        B = ppi.size(0)
        masked_ppi = ppi.clone()

        # Gather targets before masking
        # ppi[:, 0, :] = sizes, ppi[:, 1, :] = directions, ppi[:, 2, :] = IPTs
        idx_expanded = mask_indices.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, num_masked)
        targets_all = torch.gather(ppi, 2, idx_expanded)  # (B, 3, num_masked)

        target_size = targets_all[:, 0, :]  # (B, num_masked)
        target_dir = targets_all[:, 1, :]   # (B, num_masked)
        target_ipt = targets_all[:, 2, :]   # (B, num_masked)

        # Zero out masked positions
        for b in range(B):
            masked_ppi[b, :, mask_indices[b]] = 0.0

        return masked_ppi, {
            "size": target_size,
            "dir": target_dir,
            "ipt_continuous": target_ipt,
            "ipt_bins": discretize_ipt(target_ipt),
        }

    def compute_loss(self, ssl_head, encoder, masked_ppi, targets, mask_indices,
                     flow_stats=None, backbone_type="cnn"):
        """
        Compute MPFP loss.

        For CNN: re-encode the masked PPI, extract representations at mask positions.
        For Transformer: use the token representations at masked positions.
        """
        if backbone_type == "transformer":
            cls_repr, all_tokens = encoder(masked_ppi, flow_stats, return_all_tokens=True)
            # Gather token representations at masked positions
            B, S, D = all_tokens.shape
            idx = mask_indices.unsqueeze(-1).expand(-1, -1, D)
            masked_reprs = torch.gather(all_tokens, 1, idx)  # (B, num_masked, D)
        else:
            # CNN: encode masked PPI, then extract per-position features
            # Use the conv feature map before AdaptiveAvgPool1d (last layer of ppi_conv)
            h = encoder.ppi_conv[:-1](masked_ppi)  # (B, C_conv, seq_len)
            idx = mask_indices.unsqueeze(1).expand(-1, h.size(1), -1)  # (B, C_conv, num_masked)
            masked_reprs = torch.gather(h, 2, idx).transpose(1, 2)  # (B, num_masked, C_conv)
            # Project conv feature dim -> hidden_dim using encoder's projection layer
            B_sz, N_mask, feat_dim = masked_reprs.shape
            if feat_dim != encoder.hidden_dim:
                # projection: Linear(feat_dim, hidden_dim) + ReLU
                # reshape to (B*N_mask, feat_dim) for linear layers
                flat = masked_reprs.reshape(-1, feat_dim)
                projected = encoder.projection(flat)   # (B*N_mask, hidden_dim)
                masked_reprs = projected.reshape(B_sz, N_mask, encoder.hidden_dim)

        size_pred, dir_pred, ipt_pred = ssl_head.forward_mpfp(masked_reprs)

        # Normalize size targets with log1p to prevent MSE explosion (raw byte values ~0-1500+)
        target_size_norm = torch.log1p(targets["size"].abs())
        loss_size = F.mse_loss(size_pred, target_size_norm)
        loss_dir = F.binary_cross_entropy_with_logits(
            dir_pred, (targets["dir"] > 0).float()
        )
        loss_ipt = F.cross_entropy(
            ipt_pred.reshape(-1, ipt_pred.size(-1)),
            targets["ipt_bins"].reshape(-1),
        )

        return loss_size + loss_dir + loss_ipt
