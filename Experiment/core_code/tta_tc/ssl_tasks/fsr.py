"""
Flow Statistics Reconstruction (FSR) — TERTIARY SSL task.

From PPI sequences alone, predict aggregated flow statistics:
  - total bytes (sum of |sizes|)
  - duration (sum of IPTs)
  - packet count ratio (forward/total)
  - mean packet size
  - mean IPT
  - byte ratio (forward bytes / total bytes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSRTask(nn.Module):
    """Flow Statistics Reconstruction from PPI features."""

    def __init__(self, target_dim: int = 6):
        super().__init__()
        self.target_dim = target_dim

    @staticmethod
    def compute_flow_stats(ppi: torch.Tensor) -> torch.Tensor:
        """
        Compute flow-level statistics from PPI.

        Args:
            ppi: (B, 3, 30) — sizes, directions, IPTs
        Returns:
            stats: (B, 6) — derived flow statistics
        """
        sizes = ppi[:, 0, :]   # (B, 30)
        dirs = ppi[:, 1, :]    # (B, 30)
        ipts = ppi[:, 2, :]    # (B, 30)

        # Mask for non-padded packets (size > 0)
        mask = (sizes.abs() > 0).float()
        num_packets = mask.sum(dim=1).clamp(min=1)

        total_bytes = sizes.abs().sum(dim=1)
        duration = ipts.sum(dim=1)

        fwd_mask = (dirs > 0).float() * mask
        fwd_count = fwd_mask.sum(dim=1)
        pkt_ratio = fwd_count / num_packets

        mean_size = total_bytes / num_packets
        mean_ipt = duration / num_packets

        fwd_bytes = (sizes.abs() * fwd_mask).sum(dim=1)
        byte_ratio = fwd_bytes / total_bytes.clamp(min=1e-8)

        stats = torch.stack([
            total_bytes, duration, pkt_ratio,
            mean_size, mean_ipt, byte_ratio
        ], dim=1)

        # Normalize for stable training
        stats = torch.log1p(stats.clamp(min=0))

        return stats

    def compute_loss(self, ssl_head, cls_repr: torch.Tensor, ppi: torch.Tensor):
        """
        Compute FSR loss.

        Args:
            ssl_head: SSLHead with fsr_head
            cls_repr: (B, hidden_dim)
            ppi: (B, 3, 30) original PPI for computing target statistics
        """
        target = self.compute_flow_stats(ppi)
        pred = ssl_head.forward_fsr(cls_repr)
        return F.mse_loss(pred, target)
