"""TTA-TC: Y-shaped model combining encoder + classification head + SSL head."""
import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder
from .transformer_encoder import TransformerEncoder
from .heads import ClassificationHead, SSLHead


class TTATCModel(nn.Module):
    """
    Y-shaped architecture:
        Encoder (shared) -> Classification Head (frozen at test time)
                         -> SSL Head (updated at test time)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        backbone = cfg.get("backbone", "cnn")
        hidden_dim = cfg.get("hidden_dim", 256)
        num_classes = cfg["num_classes"]
        flow_stats_dim = cfg.get("flow_stats_dim", 0)
        norm_type = cfg.get("norm_type", "gn")

        # Build encoder
        if backbone == "cnn":
            self.encoder = CNNEncoder(
                ppi_channels=3,
                seq_len=30,
                flow_stats_dim=flow_stats_dim,
                hidden_dim=hidden_dim,
                norm_type=norm_type,
            )
        elif backbone == "transformer":
            self.encoder = TransformerEncoder(
                seq_len=30,
                ppi_channels=3,
                flow_stats_dim=flow_stats_dim,
                hidden_dim=hidden_dim,
                num_layers=cfg.get("num_layers", 4),
                num_heads=cfg.get("num_heads", 4),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Classification head
        self.cls_head = ClassificationHead(hidden_dim, num_classes)

        # SSL head
        self.ssl_head = SSLHead(
            hidden_dim=hidden_dim,
            enable_mpfp=cfg.get("enable_mpfp", True),
            enable_pop=cfg.get("enable_pop", True),
            enable_fsr=cfg.get("enable_fsr", True),
            flow_stats_dim=cfg.get("fsr_target_dim", 6),
        )

        self.backbone_type = backbone

    def forward(self, ppi, flow_stats=None, return_repr=False):
        """Standard forward for classification."""
        if self.backbone_type == "transformer":
            h = self.encoder(ppi, flow_stats, return_all_tokens=False)
        else:
            h = self.encoder(ppi, flow_stats)
        logits = self.cls_head(h)
        if return_repr:
            return logits, h
        return logits

    def forward_with_ssl(self, ppi, flow_stats=None):
        """
        Forward pass returning both classification logits and
        intermediate representations for SSL tasks.
        """
        if self.backbone_type == "transformer":
            cls_repr, all_tokens = self.encoder(
                ppi, flow_stats, return_all_tokens=True
            )
        else:
            cls_repr = self.encoder(ppi, flow_stats)
            all_tokens = None  # CNN doesn't have per-token output natively

        logits = self.cls_head(cls_repr)
        return logits, cls_repr, all_tokens

    def get_adaptation_params(self, adapt_mode: str = "encoder"):
        """
        Get parameters to update during TTA.
        adapt_mode:
            'encoder': all encoder params + ssl head
            'norm': only normalization layer params + ssl head
            'last_n': last N layers of encoder + ssl head
        """
        ssl_params = list(self.ssl_head.parameters())

        if adapt_mode == "norm":
            enc_params = self.encoder.get_norm_params()
        elif adapt_mode == "encoder":
            enc_params = list(self.encoder.parameters())
        else:
            enc_params = list(self.encoder.parameters())

        return enc_params + ssl_params

    def get_cls_params(self):
        """Classification head parameters (frozen at test time)."""
        return list(self.cls_head.parameters())
