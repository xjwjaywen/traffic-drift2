"""
Class-Asymmetric Selective Adaptation (CASA) loss for TTA-TC.

Extends Source Prototype Anchoring (SPA) with per-class drift weighting:
classes that have drifted farther from their source prototypes receive
larger gradients, while stable classes are updated minimally.

Key idea: traffic drift is asymmetric — different apps update independently.
Adapting stable classes wastes capacity and can hurt performance.
"""
import torch
import torch.nn.functional as F


class PrototypeLoss:
    """
    CASA: Class-Asymmetric prototype anchoring loss.

    Base loss is InfoNCE-style (same as SPA):
        L = CrossEntropy( cosine_sim(f, P) / T, pseudo_label )

    CASA adds per-class drift weights:
        w_c = softmax( drift_score_c / tau )
        L_weighted = mean( w_c[y_i] * L_i )

    where drift_score_c is the mean cosine distance of class c's features
    from their source prototype, and tau controls weight sharpness.

    Classes with high drift get large weights (adapt aggressively).
    Classes with near-zero drift get near-zero weights (preserve knowledge).
    """

    def __init__(self, prototypes: torch.Tensor, temperature: float = 0.07,
                 weight_tau: float = 1.0):
        """
        Args:
            prototypes: (C, hidden_dim) source-domain class centroids
            temperature: InfoNCE softmax temperature (lower = sharper contrast)
            weight_tau: class weight sharpness (higher = more focus on drifted classes)
        """
        self.prototypes = F.normalize(prototypes, dim=1)  # (C, hidden_dim)
        self.temperature = temperature
        self.weight_tau = weight_tau
        self.num_classes = prototypes.size(0)

    def to(self, device):
        self.prototypes = self.prototypes.to(device)
        return self

    def compute_loss(self, features: torch.Tensor, pseudo_labels: torch.Tensor,
                     class_drift_scores: torch.Tensor = None) -> torch.Tensor:
        """
        Compute CASA loss.

        Args:
            features: (B, hidden_dim) encoder output features
            pseudo_labels: (B,) predicted class indices (from teacher model)
            class_drift_scores: (C,) per-class drift scores. If provided,
                                applies asymmetric weighting. If None,
                                falls back to uniform SPA loss.
        Returns:
            scalar loss
        """
        if features.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        f = F.normalize(features, dim=1)                         # (B, hidden_dim)
        sim = torch.matmul(f, self.prototypes.T) / self.temperature  # (B, C)

        if class_drift_scores is not None:
            # Per-sample weight = drift score of its predicted class
            # softmax over classes so weights sum to num_classes (unbiased mean)
            weights = F.softmax(class_drift_scores / self.weight_tau, dim=0) * self.num_classes
            sample_weights = weights[pseudo_labels]              # (B,)
            per_sample_loss = F.cross_entropy(sim, pseudo_labels, reduction='none')  # (B,)
            loss = (per_sample_loss * sample_weights).mean()
        else:
            loss = F.cross_entropy(sim, pseudo_labels)

        return loss

    @torch.no_grad()
    def per_class_drift_scores(self, features: torch.Tensor,
                               pseudo_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-class mean cosine distance from source prototypes.

        Args:
            features: (B, hidden_dim)
            pseudo_labels: (B,)
        Returns:
            scores: (C,) drift score per class; unobserved classes get 0.0
        """
        f = F.normalize(features, dim=1)
        proto_y = self.prototypes[pseudo_labels]             # (B, hidden_dim)
        cos_sim = (f * proto_y).sum(dim=1)                   # (B,)
        dist = 1.0 - cos_sim                                 # (B,) cosine distance

        scores = torch.zeros(self.num_classes, device=features.device)
        counts = torch.zeros(self.num_classes, device=features.device)
        scores.scatter_add_(0, pseudo_labels, dist)
        counts.scatter_add_(0, pseudo_labels, torch.ones_like(dist))
        counts = counts.clamp(min=1)
        return scores / counts                               # (C,)

    def drift_score(self, features: torch.Tensor, pseudo_labels: torch.Tensor) -> float:
        """Global mean drift score (used for drift gate check)."""
        with torch.no_grad():
            scores = self.per_class_drift_scores(features, pseudo_labels)
            return scores.mean().item()
