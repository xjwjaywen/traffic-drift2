"""
Source Prototype Anchoring (SPA) loss for TTA-TC v2.

Replaces the SSL reconstruction tasks (MPFP/POP/FSR) with a contrastive
objective that pulls test-time features toward their source-domain class
prototypes. This directly targets the classification-relevant feature shift
and naturally self-gates: when features are close to prototypes (no drift),
the loss is near zero and causes minimal parameter updates.
"""
import torch
import torch.nn.functional as F


class PrototypeLoss:
    """
    Contrastive prototype anchoring loss.

    For each sample, minimizes the distance to the predicted class prototype
    while maximizing the distance to other class prototypes (InfoNCE-style).

    Loss = -log( exp(-d(f, p_y) / T) / sum_c exp(-d(f, p_c) / T) )

    where d is cosine distance, p_y is the prototype of predicted class y,
    and T is the temperature.
    """

    def __init__(self, prototypes: torch.Tensor, temperature: float = 0.07):
        """
        Args:
            prototypes: (C, hidden_dim) source-domain class centroids
            temperature: softmax temperature (lower = sharper contrast)
        """
        # L2-normalize prototypes for cosine similarity
        self.prototypes = F.normalize(prototypes, dim=1)  # (C, hidden_dim)
        self.temperature = temperature
        self.num_classes = prototypes.size(0)

    def to(self, device):
        self.prototypes = self.prototypes.to(device)
        return self

    def compute_loss(self, features: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype anchoring loss.

        Args:
            features: (B, hidden_dim) encoder output features
            pseudo_labels: (B,) predicted class indices (from teacher model)
        Returns:
            scalar loss
        """
        if features.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        # Normalize features
        f = F.normalize(features, dim=1)  # (B, hidden_dim)

        # Cosine similarity to all prototypes: (B, C)
        sim = torch.matmul(f, self.prototypes.T) / self.temperature

        # InfoNCE loss: maximize similarity to predicted class prototype
        loss = F.cross_entropy(sim, pseudo_labels)

        return loss

    def drift_score(self, features: torch.Tensor, pseudo_labels: torch.Tensor) -> float:
        """
        Compute mean distance from features to their predicted prototypes.
        Used as a drift indicator: higher score = more drift.

        Returns:
            float in [0, 2] (cosine distance range)
        """
        if features.size(0) == 0:
            return 0.0

        with torch.no_grad():
            f = F.normalize(features, dim=1)
            proto_y = self.prototypes[pseudo_labels]  # (B, hidden_dim)
            # Cosine distance = 1 - cosine_similarity
            cos_sim = (f * proto_y).sum(dim=1)
            dist = 1.0 - cos_sim.mean().item()

        return dist
