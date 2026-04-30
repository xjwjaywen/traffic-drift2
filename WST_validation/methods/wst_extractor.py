"""
Wavelet Scattering Transform (WST) feature extractor for PPI sequences.

Wraps Kymatio's Scattering1D with multi-channel concatenation for
encrypted-traffic PPI inputs of shape (batch, 3, T).

H1 verified: kymatio==0.3.0 (3D module broken on scipy>=1.16) — we import
the 1D module directly.
"""
import torch
# Direct 1D import to avoid kymatio's broken 3D module under newer scipy
from kymatio.scattering1d.frontend.torch_frontend import (
    ScatteringTorch1D as Scattering1D,
)


def make_wst(seq_len: int = 32, J: int = 3, Q: int = 4):
    """Build a Scattering1D instance for the given seq_len."""
    return Scattering1D(J=J, shape=(seq_len,), Q=Q)


@torch.no_grad()
def extract_wst_features(ppi: torch.Tensor, scattering=None,
                          J: int = 3, Q: int = 4):
    """
    Extract WST features from multi-channel PPI tensor.

    Args:
        ppi: (B, C=3, T) — packet size, direction, IPT
        scattering: optional pre-built Scattering1D (for repeated calls)

    Returns:
        features: (B, C * n_paths * (T/2^J))
    """
    B, C, T = ppi.shape
    # Pad T to next power of 2 if needed
    if T not in (16, 32, 64, 128, 256):
        next_pow2 = 1
        while next_pow2 < T:
            next_pow2 *= 2
        pad = next_pow2 - T
        ppi = torch.nn.functional.pad(ppi, (0, pad))
        T = next_pow2
    if scattering is None:
        scattering = make_wst(seq_len=T, J=J, Q=Q)
    feats = []
    for c in range(C):
        sc = scattering(ppi[:, c, :].contiguous())
        feats.append(sc.flatten(1))
    return torch.cat(feats, dim=1)


def wst_feature_dim(seq_len: int = 32, J: int = 3, Q: int = 4,
                    n_channels: int = 3) -> int:
    """Compute output dimensionality (run a 1-batch forward to get exact)."""
    sc = make_wst(seq_len, J, Q)
    dummy = torch.zeros(1, seq_len)
    out = sc(dummy)
    per_channel = out.flatten(1).shape[1]
    return n_channels * per_channel
