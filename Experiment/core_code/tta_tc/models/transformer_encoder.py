"""Transformer encoder for PPI features."""
import math
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer encoder processing PPI as per-packet tokens.

    Each packet becomes a token: (size, direction, IPT) -> embedding.
    Adds [CLS] token for classification.
    Uses LayerNorm (default in Transformers), which is TTA-safe.
    """

    def __init__(
        self,
        seq_len: int = 30,
        ppi_channels: int = 3,
        flow_stats_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.flow_stats_dim = flow_stats_dim

        # Per-packet token embedding
        self.token_embed = nn.Linear(ppi_channels, hidden_dim)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, hidden_dim) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Flow stats integration (optional)
        if flow_stats_dim > 0:
            self.flow_fc = nn.Sequential(
                nn.Linear(flow_stats_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, hidden_dim),
            )
            self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.flow_fc = None

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ppi: torch.Tensor,
        flow_stats: torch.Tensor = None,
        return_all_tokens: bool = False,
    ):
        """
        Args:
            ppi: (B, 3, 30) -> reshape to (B, 30, 3) per-packet tokens
            flow_stats: (B, D) optional flow statistics
            return_all_tokens: if True, return all token representations (for SSL)
        Returns:
            cls_repr: (B, hidden_dim) from [CLS] token
            OR (cls_repr, all_tokens) if return_all_tokens
        """
        B = ppi.size(0)

        # (B, 3, 30) -> (B, 30, 3) -> (B, 30, hidden_dim)
        tokens = self.token_embed(ppi.transpose(1, 2))

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 31, hidden_dim)

        # Add positional encoding
        tokens = tokens + self.pos_embed

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        cls_repr = tokens[:, 0]  # (B, hidden_dim)

        # Integrate flow stats
        if self.flow_fc is not None and flow_stats is not None:
            z = self.flow_fc(flow_stats)
            cls_repr = self.combine(torch.cat([cls_repr, z], dim=1))

        if return_all_tokens:
            return cls_repr, tokens[:, 1:]  # skip [CLS]
        return cls_repr

    def get_norm_params(self):
        """Return normalization layer parameters."""
        params = []
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                params.extend(m.parameters())
        return params
