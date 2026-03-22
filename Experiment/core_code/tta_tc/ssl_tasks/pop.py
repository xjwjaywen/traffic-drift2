"""
Packet Order Prediction (POP) — SECONDARY SSL task.

Divides PPI into 3 segments, shuffles 2 randomly selected segments,
and predicts the correct ordering (6 permutations).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


# All 6 permutations of 3 segments
ALL_PERMS = list(itertools.permutations([0, 1, 2]))
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}


class POPTask(nn.Module):
    """Packet Order Prediction: 6-way classification on segment permutations."""

    def __init__(self, seq_len: int = 30, num_segments: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.num_segments = num_segments
        self.seg_len = seq_len // num_segments  # 10

    def shuffle_segments(self, ppi: torch.Tensor):
        """
        Shuffle PPI segments and return (shuffled_ppi, perm_labels).

        Args:
            ppi: (B, 3, 30)
        Returns:
            shuffled_ppi: (B, 3, 30)
            labels: (B,) permutation class index [0-5]
        """
        B = ppi.size(0)
        device = ppi.device
        shuffled = ppi.clone()
        labels = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            # Random permutation of 3 segments
            perm = torch.randperm(self.num_segments).tolist()
            perm_tuple = tuple(perm)
            labels[b] = PERM_TO_IDX[perm_tuple]

            # Apply permutation
            segments = []
            for seg_idx in perm:
                start = seg_idx * self.seg_len
                end = start + self.seg_len
                segments.append(ppi[b, :, start:end])
            shuffled[b, :, :self.seg_len * self.num_segments] = torch.cat(segments, dim=1)

        return shuffled, labels

    def compute_loss(self, ssl_head, cls_repr: torch.Tensor, labels: torch.Tensor):
        """
        Compute POP loss.

        Args:
            ssl_head: SSLHead with pop_head
            cls_repr: (B, hidden_dim) from encoder
            labels: (B,) permutation class indices
        """
        logits = ssl_head.forward_pop(cls_repr)
        return F.cross_entropy(logits, labels)
