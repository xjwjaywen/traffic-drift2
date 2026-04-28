"""
TTA-TC: Test-Time Adaptation Engine (v9 — Period-Level Prototype Anchoring).

Two-pass design per test period:
  Pass 1: Forward all batches through frozen model, collect features.
  Sample: Randomly select `label_budget_per_period` samples and query
          oracle for ground-truth labels.
  Build:  Construct period-specific prototypes by averaging features per
          labeled class. Blend with source prototypes for stability on
          classes with few or no labels.
  Pass 2: Predict using a blend of static logits and period-prototype
          cosine logits.

Why this works where v8 failed:
  - v8 selected high-entropy (ambiguous) samples for prototype updates,
    pulling prototypes toward decision boundaries and degrading accuracy.
  - v9 uses random sampling: each labeled sample is a fair representative
    of its class distribution, preserving prototype quality.
  - Two-pass ensures all samples benefit from updated prototypes, not
    just those late in the stream.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .samplers import get_sampler


class TTAEngine:
    """Period-level prototype anchoring with active labels."""

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

        if prototypes is None:
            raise ValueError("TTA-TC v9 requires class prototypes")
        self.source_prototypes = F.normalize(
            prototypes.to(self.device).clone(), dim=1
        )

        # Hyperparameters
        self.label_budget_per_period = cfg.get("label_budget_per_period", 500)
        self.proto_blend = cfg.get("proto_blend", 0.5)  # weight on source proto
        self.tta_blend = cfg.get("tta_blend", 0.5)      # weight on proto logits
        self.proto_temperature = cfg.get("spa_temperature", 0.1)

        # Active sampling strategy (M1 sweep)
        self.sampler_name = cfg.get("sampler", "random")
        self.sampler = get_sampler(self.sampler_name)
        self.seed = cfg.get("seed", None)

        self.labels_used = 0

    def set_fisher(self, dataloader):
        pass

    def set_baseline_entropy(self, baseline):
        pass

    def reset_period(self):
        self.labels_used = 0

    def reset(self):
        self.labels_used = 0

    @torch.no_grad()
    def adapt_period(self, test_loader, period_name: str = ""):
        """
        Two-pass period-level adaptation.

        Returns:
            labels: np.ndarray (N,) ground-truth labels
            preds:  np.ndarray (N,) predicted labels
        """
        # ===== Pass 1: collect features, static logits, labels =====
        all_features = []
        all_labels = []
        all_static_logits = []

        for batch in tqdm(test_loader, desc=f"TTA-TC@{period_name} pass1"):
            ppi = batch["ppi"].to(self.device)
            labels = batch["label"]
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(self.device)

            logits, features = self.model(ppi, flow_stats, return_repr=True)
            all_features.append(features)
            all_labels.append(labels)
            all_static_logits.append(logits)

        all_features = torch.cat(all_features, dim=0)         # (N, D)
        all_labels = torch.cat(all_labels, dim=0)             # (N,)
        all_static_logits = torch.cat(all_static_logits, dim=0)  # (N, C)

        N = all_features.size(0)

        # ===== Sample label budget using configured strategy =====
        budget = min(self.label_budget_per_period, N)
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.seed))
        pred_classes = all_static_logits.argmax(dim=1)
        idx = self.sampler(
            features=all_features,
            static_logits=all_static_logits,
            pred_classes=pred_classes,
            budget=budget,
            num_classes=self.num_classes,
            generator=gen,
        )
        queried_labels = all_labels.to(self.device)[idx]
        queried_features = all_features[idx]
        self.labels_used = idx.size(0)

        # ===== Build period prototypes by per-class averaging =====
        period_prototypes = self.source_prototypes.clone()
        f_norm = F.normalize(queried_features, dim=1)

        for c in queried_labels.unique():
            mask = queried_labels == c
            class_mean = f_norm[mask].mean(dim=0)
            class_mean = F.normalize(class_mean, dim=0)
            # Blend source prototype with new period mean
            blended = (
                self.proto_blend * self.source_prototypes[c]
                + (1 - self.proto_blend) * class_mean
            )
            period_prototypes[c] = F.normalize(blended, dim=0)

        # ===== Pass 2: predict with updated prototypes (chunked to avoid OOM) =====
        w = self.tta_blend
        chunk_size = 8192
        all_preds = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            f_chunk = F.normalize(all_features[start:end], dim=1)
            proto_logits = torch.matmul(f_chunk, period_prototypes.T) / self.proto_temperature
            static_probs = F.softmax(all_static_logits[start:end], dim=1)
            proto_probs = F.softmax(proto_logits, dim=1)
            final_probs = (1 - w) * static_probs + w * proto_probs
            all_preds.append(final_probs.argmax(dim=1))
        preds = torch.cat(all_preds, dim=0)

        return all_labels.cpu().numpy(), preds.cpu().numpy()

    @torch.no_grad()
    def adapt_batch(self, ppi, flow_stats=None, labels=None):
        """Streaming interface kept for compatibility — falls back to static."""
        logits = self.model(ppi, flow_stats)
        return logits, {"adapted": False, "method": "fallback"}
