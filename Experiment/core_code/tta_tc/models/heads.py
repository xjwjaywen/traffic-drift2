"""Classification and SSL heads for TTA-TC."""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Standard classification head: FC -> logits."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class SSLHead(nn.Module):
    """
    Multi-task SSL head for MPFP + POP + FSR.

    Sub-heads:
        mpfp_size: predict masked packet sizes (regression)
        mpfp_dir:  predict masked packet directions (binary)
        mpfp_ipt:  predict masked IPT bins (8-class)
        pop:       predict segment permutation (6-class)
        fsr:       predict flow statistics (regression)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_ipt_bins: int = 8,
        num_pop_classes: int = 6,
        flow_stats_dim: int = 6,
        enable_mpfp: bool = True,
        enable_pop: bool = True,
        enable_fsr: bool = True,
    ):
        super().__init__()
        self.enable_mpfp = enable_mpfp
        self.enable_pop = enable_pop
        self.enable_fsr = enable_fsr

        if enable_mpfp:
            self.mpfp_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.mpfp_size = nn.Linear(hidden_dim, 1)
            self.mpfp_dir = nn.Linear(hidden_dim, 1)
            self.mpfp_ipt = nn.Linear(hidden_dim, num_ipt_bins)

        if enable_pop:
            self.pop_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, num_pop_classes),
            )

        if enable_fsr:
            self.fsr_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, flow_stats_dim),
            )

    def forward_mpfp(self, token_reprs: torch.Tensor):
        """
        Args:
            token_reprs: (B, num_masked, hidden_dim)
        Returns:
            size_pred: (B, num_masked)
            dir_pred:  (B, num_masked)
            ipt_pred:  (B, num_masked, 8)
        """
        h = self.mpfp_proj(token_reprs)
        return (
            self.mpfp_size(h).squeeze(-1),
            self.mpfp_dir(h).squeeze(-1),
            self.mpfp_ipt(h),
        )

    def forward_pop(self, cls_repr: torch.Tensor):
        """
        Args:
            cls_repr: (B, hidden_dim)
        Returns:
            (B, 6) permutation logits
        """
        return self.pop_head(cls_repr)

    def forward_fsr(self, cls_repr: torch.Tensor):
        """
        Args:
            cls_repr: (B, hidden_dim)
        Returns:
            (B, flow_stats_dim) predicted flow statistics
        """
        return self.fsr_head(cls_repr)
