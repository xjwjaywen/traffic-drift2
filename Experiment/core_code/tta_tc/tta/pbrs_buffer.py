"""Prediction-Balanced Reservoir Sampling (PBRS) buffer."""
import torch
import numpy as np
from collections import defaultdict


class PBRSBuffer:
    """
    Prediction-Balanced Reservoir Sampling buffer for non-i.i.d. TTA.

    Maintains a class-balanced buffer of test samples using predicted labels.
    Based on NOTE (Gong et al., NeurIPS 2022).
    """

    def __init__(self, buffer_size: int, num_classes: int):
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.per_class_size = buffer_size // num_classes

        # Storage: list of (ppi, flow_stats, pred_label) per class
        self.class_buffers = defaultdict(list)
        self.class_counts = defaultdict(int)

    def add(self, ppi: torch.Tensor, flow_stats: torch.Tensor, pred_labels: torch.Tensor):
        """
        Add samples to buffer using reservoir sampling.

        Args:
            ppi: (B, 3, 30)
            flow_stats: (B, D) or None
            pred_labels: (B,) predicted class labels
        """
        B = ppi.size(0)
        for i in range(B):
            c = pred_labels[i].item()
            self.class_counts[c] += 1

            entry = (
                ppi[i].detach().cpu(),
                flow_stats[i].detach().cpu() if flow_stats is not None else None,
            )

            if len(self.class_buffers[c]) < self.per_class_size:
                self.class_buffers[c].append(entry)
            else:
                # Reservoir sampling
                j = np.random.randint(0, self.class_counts[c])
                if j < self.per_class_size:
                    self.class_buffers[c][j] = entry

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample a class-balanced mini-batch from the buffer.

        Returns:
            ppi: (batch_size, 3, 30)
            flow_stats: (batch_size, D) or None
            weights: (batch_size,) inverse-sqrt class frequency weights
        """
        # Collect available classes
        available = {c: buf for c, buf in self.class_buffers.items() if len(buf) > 0}
        if not available:
            return None, None, None

        classes = list(available.keys())
        samples_per_class = max(1, batch_size // len(classes))

        ppis = []
        stats = []
        weights = []

        for c in classes:
            buf = available[c]
            n_c = len(buf)
            w_c = 1.0 / np.sqrt(max(n_c, 1))

            indices = np.random.choice(n_c, size=min(samples_per_class, n_c), replace=True)
            for idx in indices:
                ppi_i, fs_i = buf[idx]
                ppis.append(ppi_i)
                if fs_i is not None:
                    stats.append(fs_i)
                weights.append(w_c)

        ppi_batch = torch.stack(ppis).to(device)
        flow_stats_batch = torch.stack(stats).to(device) if stats else None
        weight_batch = torch.tensor(weights, device=device)
        weight_batch = weight_batch / weight_batch.sum()  # normalize

        return ppi_batch, flow_stats_batch, weight_batch

    def __len__(self):
        return sum(len(buf) for buf in self.class_buffers.values())
