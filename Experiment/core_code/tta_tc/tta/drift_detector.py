"""Per-class drift detection via entropy monitoring."""
import torch
import numpy as np
from collections import defaultdict


class DriftDetector:
    """
    Monitors per-class prediction entropy to detect drift.

    Compares running entropy against baseline (from training).
    Flags classes where entropy increase exceeds threshold.
    """

    def __init__(
        self,
        num_classes: int,
        entropy_threshold: float = 0.5,
        abrupt_threshold: float = 1.0,
        window_size: int = 100,
    ):
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.abrupt_threshold = abrupt_threshold
        self.window_size = window_size

        # Baseline entropy per class (set from training data)
        self.baseline_entropy = np.zeros(num_classes)
        self.baseline_set = False

        # Running entropy windows
        self.entropy_windows = defaultdict(list)
        self.prev_mean_entropy = np.zeros(num_classes)

    def set_baseline(self, baseline_entropy: np.ndarray):
        """Set per-class baseline entropy from training/validation data."""
        self.baseline_entropy = baseline_entropy.copy()
        self.baseline_set = True

    def compute_baseline_from_logits(self, logits: torch.Tensor, labels: torch.Tensor):
        """Compute baseline entropy from labeled validation data."""
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        per_class_entropy = np.zeros(self.num_classes)
        counts = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_entropy[c] = entropy[mask].mean().item()
                counts[c] = mask.sum().item()

        self.baseline_entropy = per_class_entropy
        self.baseline_set = True
        return per_class_entropy

    def update(self, logits: torch.Tensor):
        """
        Update drift detector with new batch predictions.

        Args:
            logits: (B, C) raw logits
        Returns:
            drifted_classes: set of class indices flagged as drifted
            abrupt_classes: set of class indices with abrupt drift
        """
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        preds = logits.argmax(dim=1)

        # Update per-class entropy windows
        for i in range(logits.size(0)):
            c = preds[i].item()
            self.entropy_windows[c].append(entropy[i].item())
            if len(self.entropy_windows[c]) > self.window_size:
                self.entropy_windows[c].pop(0)

        # Check drift
        drifted = set()
        abrupt = set()

        for c in range(self.num_classes):
            if len(self.entropy_windows[c]) < 10:
                continue

            current_mean = np.mean(self.entropy_windows[c])
            delta = current_mean - self.baseline_entropy[c]

            if delta > self.entropy_threshold:
                drifted.add(c)

            # Check for abrupt drift (rate of change)
            rate = abs(current_mean - self.prev_mean_entropy[c])
            if rate > self.abrupt_threshold:
                abrupt.add(c)

            self.prev_mean_entropy[c] = current_mean

        return drifted, abrupt
