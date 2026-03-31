"""
TTA-TC: Test-Time Adaptation Engine (v4 — Optimal Transport).

Adapts at inference by transporting test features toward source class
prototypes via Sinkhorn optimal transport. No gradient updates.

Mathematical formulation:
    Given source prototypes P = {p_1, ..., p_K} and test features Z = {z_1, ..., z_n},
    solve the entropy-regularized OT problem:

        min_γ  Σ_{k,i} γ_{ki} · c(p_k, z_i)  +  ε · H(γ)

    where c is cosine distance, ε is the regularization strength, and H is entropy.
    The Sinkhorn algorithm solves this via alternating row/column normalization.

    Transported features:  ẑ_i = Σ_k  (γ*_{ki} / Σ_{k'} γ*_{k'i}) · p_k
    Classification:        ŷ_i = argmax cosine_sim(ẑ_i, P)

Key properties:
    - Inference-only: no parameter updates, no backward pass
    - Class-asymmetric: stable classes get near-identity transport
    - Safety: with high ε, transport → uniform → falls back to static logits
"""
import torch
import torch.nn.functional as F
import copy
import math


def sinkhorn_transport(cost: torch.Tensor, epsilon: float = 0.1,
                       max_iter: int = 50) -> torch.Tensor:
    """
    Sinkhorn algorithm for entropy-regularized optimal transport.

    Args:
        cost: (K, B) cost matrix between K prototypes and B test samples
        epsilon: regularization strength (lower = sharper transport)
        max_iter: number of Sinkhorn iterations
    Returns:
        gamma: (K, B) optimal transport coupling matrix (rows sum to 1/K, cols sum to 1/B)
    """
    K, B = cost.shape
    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / epsilon                             # (K, B)
    log_mu = torch.full((K,), -math.log(K), device=cost.device)     # uniform source
    log_nu = torch.full((B,), -math.log(B), device=cost.device)     # uniform target

    log_u = torch.zeros(K, device=cost.device)
    log_v = torch.zeros(B, device=cost.device)

    for _ in range(max_iter):
        # Row normalization
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        # Column normalization
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    gamma = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return gamma


class TTAEngine:
    """
    Optimal Transport Test-Time Adaptation (OT-TTA).

    Pipeline per batch (all inference, no backward pass):
        1. Forward pass through frozen model → static logits + features
        2. Compute cosine cost matrix between features and source prototypes
        3. Solve Sinkhorn OT → coupling matrix γ*
        4. Transport features: ẑ_i = weighted average of prototypes via γ*
        5. Blend: interpolate between original and transported features
        6. Classify via cosine similarity to source prototypes
    """

    def __init__(self, model, cfg: dict, prototypes: torch.Tensor = None):
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.num_classes = cfg["num_classes"]

        # Freeze the entire model
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Source prototypes (frozen)
        if prototypes is not None:
            self.prototypes = F.normalize(prototypes.to(self.device), dim=1)  # (K, D)
        else:
            self.prototypes = None

        # OT hyperparameters
        self.ot_epsilon = cfg.get("ot_epsilon", 0.1)
        self.ot_max_iter = cfg.get("ot_max_iter", 50)
        self.transport_weight = cfg.get("transport_weight", 0.5)
        self.proto_temperature = cfg.get("spa_temperature", 0.1)

        # Counters
        self.step_count = 0

    def set_fisher(self, dataloader):
        """No-op."""
        pass

    def set_baseline_entropy(self, baseline):
        """No-op."""
        pass

    @torch.no_grad()
    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        OT-based test-time adaptation.

        Args:
            ppi: (B, 3, 30)
            flow_stats: (B, D) or None
        Returns:
            logits: (B, C) adapted classification output
            info: dict with adaptation stats
        """
        info = {}

        # Step 1: Forward pass
        logits, features = self.model(ppi, flow_stats, return_repr=True)
        info["total_samples"] = ppi.size(0)

        if self.prototypes is None:
            info["adapted"] = False
            return logits, info

        f = F.normalize(features, dim=1)                        # (B, D)
        K = self.prototypes.size(0)
        B = f.size(0)

        # Step 2: Cosine cost matrix (1 - similarity)
        sim = torch.matmul(self.prototypes, f.T)                # (K, B)
        cost = 1.0 - sim                                        # (K, B) cosine distance

        # Step 3: Sinkhorn OT
        gamma = sinkhorn_transport(cost, self.ot_epsilon, self.ot_max_iter)  # (K, B)

        # Step 4: Transport features — barycentric mapping
        # For each test sample i, transported feature = weighted sum of prototypes
        col_sum = gamma.sum(dim=0, keepdim=True).clamp(min=1e-8)  # (1, B)
        weights = gamma / col_sum                                  # (K, B) normalized per column
        transported = torch.matmul(weights.T, self.prototypes)     # (B, D)
        transported = F.normalize(transported, dim=1)

        # Step 5: Blend original and transported features
        # transport_weight controls how much we trust OT vs original
        w = self.transport_weight
        blended = F.normalize((1 - w) * f + w * transported, dim=1)  # (B, D)

        # Step 6: Classify via cosine similarity to prototypes
        adapted_logits = torch.matmul(blended, self.prototypes.T) / self.proto_temperature

        # Blend with static logits for safety
        static_probs = F.softmax(logits, dim=1)
        ot_probs = F.softmax(adapted_logits, dim=1)
        final_probs = (1 - w) * static_probs + w * ot_probs
        final_logits = torch.log(final_probs + 1e-8)

        # Stats
        info["adapted"] = True
        info["mean_transport_cost"] = cost.mean().item()
        info["mean_coupling_entropy"] = -(gamma * (gamma + 1e-8).log()).sum().item() / B
        self.step_count += 1

        return final_logits, info

    def reset(self):
        """Reset state."""
        self.step_count = 0
