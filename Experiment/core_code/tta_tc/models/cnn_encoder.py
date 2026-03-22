"""1D-CNN encoder for PPI features, compatible with CESNET model style."""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    1D-CNN encoder processing PPI (packet size, direction, IPT) sequences.

    Architecture:
        PPI branch: Conv1d stack -> GlobalAvgPool -> FC
        Flow stats branch: FC -> ReLU -> FC
        Concat -> FC -> representation

    Uses GroupNorm instead of BatchNorm for TTA stability.
    """

    def __init__(
        self,
        ppi_channels: int = 3,
        seq_len: int = 30,
        flow_stats_dim: int = 0,
        hidden_dim: int = 256,
        norm_type: str = "gn",
        num_groups: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.flow_stats_dim = flow_stats_dim

        # PPI branch
        self.ppi_conv = nn.Sequential(
            nn.Conv1d(ppi_channels, 64, kernel_size=3, padding=1),
            self._make_norm(64, norm_type, num_groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            self._make_norm(128, norm_type, num_groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            self._make_norm(256, norm_type, num_groups),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        ppi_out_dim = 256

        # Flow stats branch (optional)
        if flow_stats_dim > 0:
            self.flow_fc = nn.Sequential(
                nn.Linear(flow_stats_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
            )
            concat_dim = ppi_out_dim + 64
        else:
            self.flow_fc = None
            concat_dim = ppi_out_dim

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_norm(channels, norm_type, num_groups):
        if norm_type == "gn":
            return nn.GroupNorm(min(num_groups, channels), channels)
        elif norm_type == "ln":
            return nn.GroupNorm(1, channels)  # GroupNorm(1, C) == LayerNorm
        elif norm_type == "bn":
            return nn.BatchNorm1d(channels)
        else:
            return nn.Identity()

    def forward(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        Args:
            ppi: (B, 3, 30) - packet sizes, directions, IPTs
            flow_stats: (B, D) - optional flow-level statistics
        Returns:
            (B, hidden_dim) representation
        """
        h = self.ppi_conv(ppi).squeeze(-1)  # (B, 256)

        if self.flow_fc is not None and flow_stats is not None:
            z = self.flow_fc(flow_stats)  # (B, 64)
            h = torch.cat([h, z], dim=1)

        return self.projection(h)  # (B, hidden_dim)

    def get_norm_params(self):
        """Return normalization layer parameters (for norm-only adaptation)."""
        params = []
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                params.extend(m.parameters())
        return params
