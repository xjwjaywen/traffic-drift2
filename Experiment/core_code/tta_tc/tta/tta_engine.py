"""
TTA-TC: Test-Time Adaptation Engine (v3 — Inference-Time Prototype Interpolation).

No gradient updates. Adapts at inference by blending:
  - Static classifier output (frozen source model)
  - Prototype-based output (cosine similarity to running class prototypes)

Blending coefficient α_c is per-class, proportional to drift magnitude:
  - Stable classes (low drift) → α ≈ 0 → output ≈ Static (safe)
  - Drifted classes (high drift) → α → α_max → trust prototype classifier

Safety guarantee: when drift = 0, α = 0, output = Static exactly.
"""
import torch
import torch.nn.functional as F
import copy
import math


class TTAEngine:
    """
    Class-Asymmetric Inference-Time Adaptation (CASA-Inf).

    Pipeline per batch (all inference, no backward pass):
        1. Forward pass through frozen model → static logits + features
        2. Entropy filter: select high-confidence samples for prototype update
        3. EMA update of running class prototypes
        4. Compute per-class drift score: cosine distance(running, source)
        5. Per-class α: drift_score → blending weight
        6. Blend: final = (1-α) × static_probs + α × proto_probs
    """

    def __init__(self, model, cfg: dict, prototypes: torch.Tensor = None):
        """
        Args:
            model: TTATCModel instance (source-trained, will be frozen)
            cfg: configuration dict
            prototypes: (C, hidden_dim) source class prototypes
        """
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.num_classes = cfg["num_classes"]

        # Freeze the entire model — no parameter updates
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Source prototypes (frozen reference)
        if prototypes is not None:
            self.source_protos = F.normalize(prototypes.to(self.device), dim=1)
            # Running prototypes (EMA updated from test features)
            self.running_protos = self.source_protos.clone()
        else:
            self.source_protos = None
            self.running_protos = None

        # Hyperparameters
        self.proto_momentum = cfg.get("proto_momentum", 0.99)
        self.proto_temperature = cfg.get("spa_temperature", 0.1)
        self.alpha_scale = cfg.get("alpha_scale", 5.0)
        self.alpha_max = cfg.get("alpha_max", 0.5)
        self.drift_threshold = cfg.get("proto_drift_threshold", 0.02)

        # Entropy filter for prototype update quality
        self.entropy_filter_ratio = cfg.get("entropy_filter_ratio", 0.4)
        self.max_entropy = self.entropy_filter_ratio * math.log(max(self.num_classes, 2))

        # Counters
        self.step_count = 0

    def set_fisher(self, dataloader):
        """No-op. Fisher regularization is not used in inference-time adaptation."""
        pass

    def set_baseline_entropy(self, baseline):
        """No-op. Entropy baseline not needed for prototype-based adaptation."""
        pass

    @torch.no_grad()
    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        Inference-time adaptation via class-conditional prototype interpolation.

        Args:
            ppi: (B, 3, 30) packet payload input
            flow_stats: (B, D) or None
        Returns:
            logits: (B, C) adapted classification output (log-probs)
            info: dict with adaptation stats
        """
        info = {}

        # Step 1: Forward pass through frozen model
        logits, features = self.model(ppi, flow_stats, return_repr=True)
        info["total_samples"] = ppi.size(0)

        # If no prototypes, return static predictions
        if self.source_protos is None:
            info["adapted"] = False
            return logits, info

        f = F.normalize(features, dim=1)                       # (B, hidden_dim)
        pseudo_labels = logits.argmax(dim=1)                   # (B,)

        # Step 2: Entropy filter — only update prototypes with confident samples
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        confident_mask = entropy < self.max_entropy

        # Step 3: EMA update of running prototypes (confident samples only)
        if confident_mask.sum() > 0:
            conf_features = f[confident_mask]
            conf_labels = pseudo_labels[confident_mask]
            m = self.proto_momentum
            for c in range(self.num_classes):
                mask_c = conf_labels == c
                if mask_c.sum() > 0:
                    class_mean = F.normalize(conf_features[mask_c].mean(dim=0), dim=0)
                    self.running_protos[c] = F.normalize(
                        m * self.running_protos[c] + (1 - m) * class_mean, dim=0
                    )

        # Step 4: Per-class drift score (cosine distance: running vs source)
        drift_scores = 1.0 - (self.running_protos * self.source_protos).sum(dim=1)  # (C,)

        # Step 5: Per-class blending coefficient α
        # α_c = clamp(scale * max(drift_c - threshold, 0), 0, alpha_max)
        alpha = (self.alpha_scale * (drift_scores - self.drift_threshold).clamp(min=0)).clamp(max=self.alpha_max)

        # Step 6: Blend static probs with prototype-based probs
        proto_logits = torch.matmul(f, self.running_protos.T) / self.proto_temperature
        proto_probs = F.softmax(proto_logits, dim=1)           # (B, C)
        static_probs = probs                                   # (B, C), already computed

        sample_alpha = alpha[pseudo_labels].unsqueeze(1)       # (B, 1)
        final_probs = (1 - sample_alpha) * static_probs + sample_alpha * proto_probs
        final_logits = torch.log(final_probs + 1e-8)

        # Stats
        info["adapted"] = True
        info["mean_drift"] = drift_scores.mean().item()
        info["max_drift"] = drift_scores.max().item()
        info["mean_alpha"] = alpha.mean().item()
        info["confident_samples"] = confident_mask.sum().item()
        self.step_count += 1

        return final_logits, info

    def reset(self):
        """Reset running prototypes to source prototypes."""
        if self.source_protos is not None:
            self.running_protos = self.source_protos.clone()
        self.step_count = 0
