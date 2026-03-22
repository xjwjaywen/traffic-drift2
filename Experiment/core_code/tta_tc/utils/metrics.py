"""Evaluation metrics for TTA-TC."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
import json
import os


def compute_metrics(y_true, y_pred, num_classes=None):
    """
    Compute comprehensive metrics.

    Returns:
        dict with accuracy, macro_f1, per_class_recall, etc.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class recall
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }


class MetricsTracker:
    """Track metrics across temporal periods for computing ARR/AURC."""

    def __init__(self, source_accuracy: float = None):
        self.source_accuracy = source_accuracy
        self.period_metrics = {}

    def add_period(self, period_name: str, y_true, y_pred):
        """Add metrics for a time period."""
        metrics = compute_metrics(y_true, y_pred)
        metrics["period"] = period_name

        if self.source_accuracy is not None:
            metrics["arr"] = metrics["accuracy"] / self.source_accuracy
        else:
            metrics["arr"] = None

        self.period_metrics[period_name] = metrics
        return metrics

    def compute_aurc(self):
        """Compute Area Under Retention Curve."""
        arrs = []
        for name, m in sorted(self.period_metrics.items()):
            if m["arr"] is not None:
                arrs.append(m["arr"])
        if not arrs:
            return None
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        return trapz(arrs) / len(arrs)

    def summary(self):
        """Generate summary dict."""
        return {
            "source_accuracy": self.source_accuracy,
            "aurc": self.compute_aurc(),
            "periods": {
                name: {
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "arr": m["arr"],
                }
                for name, m in self.period_metrics.items()
            },
        }

    def save(self, path: str):
        """Save metrics to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
