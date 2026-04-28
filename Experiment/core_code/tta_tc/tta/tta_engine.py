"""
TTA-TC: Test-Time Adaptation Engine (v8 — Active Prototype Adaptation).

Combines two ideas:
  1. Frozen source model with prototype-based classification.
  2. Streaming active learning: each batch, select top-k highest-entropy
     samples, query their labels (oracle/human), and update the
     corresponding class prototypes online.

Why this works where pure unsupervised TTA fails:
  - Encrypted traffic drift is at the INPUT level (cert rotation, app
    behavior change), not the feature level.
  - No amount of post-hoc feature manipulation can recover what the
    encoder failed to extract correctly.
  - A small label budget (~k samples per period) is enough to anchor
    prototypes to the drifted distribution, recovering most of the gap.

No gradient updates. Pure inference + prototype EMA update.
"""
import torch
import torch.nn.functional as F


class TTAEngine:
    """
    Active prototype-based TTA.

    Per batch:
        1. Forward through frozen encoder
        2. Predict via cosine similarity to (possibly updated) prototypes
        3. If label budget remains, select top-k uncertain samples
        4. Query their labels, EMA-update the corresponding prototypes
    """

    def __init__(self, model, cfg: dict, prototypes: torch.Tensor = None,
                 position_stats: dict = None):
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.num_classes = cfg["num_classes"]

        # Freeze the entire model
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Source prototypes (will be updated online when labels are queried)
        if prototypes is None:
            raise ValueError("TTA-TC v8 requires class prototypes")
        self.prototypes = F.normalize(prototypes.to(self.device).clone(), dim=1)
        self.proto_dim = self.prototypes.size(1)

        # Active learning hyperparameters
        self.label_budget_per_period = cfg.get("label_budget_per_period", 50)
        self.proto_ema = cfg.get("proto_ema", 0.8)  # higher = trust new label more
        self.proto_temperature = cfg.get("spa_temperature", 0.1)

        # Per-period state
        self.budget_remaining = self.label_budget_per_period
        self.labels_used = 0
        self.step_count = 0

    def set_fisher(self, dataloader):
        pass

    def set_baseline_entropy(self, baseline):
        pass

    @torch.no_grad()
    def _proto_logits(self, features):
        f = F.normalize(features, dim=1)
        sim = torch.matmul(f, self.prototypes.T)
        return sim / self.proto_temperature

    @torch.no_grad()
    def _entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -(probs * log_probs).sum(dim=1)

    @torch.no_grad()
    def _update_prototypes(self, features, labels):
        """EMA update: new_proto = ema * old_proto + (1 - ema) * sample_features."""
        f = F.normalize(features, dim=1)
        for c in labels.unique():
            mask = labels == c
            if mask.sum() == 0:
                continue
            class_mean = f[mask].mean(dim=0)
            class_mean = F.normalize(class_mean, dim=0)
            self.prototypes[c] = (
                self.proto_ema * self.prototypes[c]
                + (1 - self.proto_ema) * class_mean
            )
            self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)

    @torch.no_grad()
    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None,
                    labels: torch.Tensor = None):
        """
        Run inference, then optionally use a few labeled samples to update prototypes.

        Args:
            ppi: (B, 3, 30)
            flow_stats: (B, D) or None
            labels: (B,) ground-truth labels (used as oracle for active learning)
        Returns:
            logits: (B, C) classification output (combined static + prototype)
            info: dict with adaptation stats
        """
        info = {"total_samples": ppi.size(0)}

        # Forward through frozen model
        static_logits, features = self.model(ppi, flow_stats, return_repr=True)
        proto_logits = self._proto_logits(features)

        # Combine: average of static and prototype predictions
        combined_logits = 0.5 * static_logits + 0.5 * proto_logits

        # Active learning: query labels for top-k uncertain samples
        info["labels_queried"] = 0
        if labels is not None and self.budget_remaining > 0:
            entropy = self._entropy(combined_logits)
            k = min(self.budget_remaining, ppi.size(0))
            # Select top-k highest entropy
            _, top_idx = torch.topk(entropy, k)
            queried_features = features[top_idx]
            queried_labels = labels.to(self.device)[top_idx]

            self._update_prototypes(queried_features, queried_labels)

            self.budget_remaining -= k
            self.labels_used += k
            info["labels_queried"] = k

            # Recompute prototype logits with updated prototypes for THIS batch
            proto_logits = self._proto_logits(features)
            combined_logits = 0.5 * static_logits + 0.5 * proto_logits

        info["budget_remaining"] = self.budget_remaining
        info["adapted"] = True
        self.step_count += 1
        return combined_logits, info

    def reset_period(self):
        """Reset budget for new test period."""
        self.budget_remaining = self.label_budget_per_period

    def reset(self):
        """Full reset (only used for fresh evaluation)."""
        self.budget_remaining = self.label_budget_per_period
        self.labels_used = 0
        self.step_count = 0
