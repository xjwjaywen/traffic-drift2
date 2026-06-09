"""
Causal Feature Identification via Cross-Validation Prototype Stability.

Identifies which hidden dimensions are causally related to class identity
(stable across data subsets) vs. spuriously correlated with deployment
conditions (varying across subsets).

Method:
  1. Split data into K stratified folds
  2. Compute class prototypes from each fold
  3. Measure per-dimension coefficient of variation across folds
  4. Low variation -> causal; high variation -> spurious
"""
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def compute_causal_mask(model, dataloader, num_classes, hidden_dim, device,
                        n_folds=5, causal_ratio=0.5, seed=42):
    """Compute causal feature mask via prototype stability analysis.

    Args:
        model: trained TTATCModel (kept frozen)
        dataloader: training or validation dataloader
        num_classes: number of classes
        hidden_dim: encoder output dimension
        device: torch device
        n_folds: number of random folds for stability estimation
        causal_ratio: fraction of dimensions to mark as causal (0-1)
        seed: random seed for fold splitting

    Returns:
        causal_mask: (hidden_dim,) boolean tensor — True = causal
        stability_scores: (hidden_dim,) float tensor — higher = more stable
    """
    all_features = []
    all_labels = []

    model.eval()
    for batch in tqdm(dataloader, desc="Causal mask: collecting features"):
        ppi = batch["ppi"].to(device)
        labels = batch["label"]
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)
        _, features = model(ppi, flow_stats, return_repr=True)
        all_features.append(features.cpu())
        all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    N = all_features.size(0)

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=gen)
    fold_size = N // n_folds

    fold_prototypes = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else N
        fold_idx = indices[start:end]
        fold_features = all_features[fold_idx]
        fold_labels = all_labels[fold_idx]

        proto = torch.zeros(num_classes, hidden_dim)
        count = torch.zeros(num_classes)
        for c in range(num_classes):
            mask = fold_labels == c
            if mask.sum() > 0:
                proto[c] = fold_features[mask].mean(dim=0)
                count[c] = mask.sum()
        fold_prototypes.append(proto)

    stacked = torch.stack(fold_prototypes)  # (K, C, D)

    mean_proto = stacked.mean(dim=0)
    std_proto = stacked.std(dim=0)
    cv = std_proto / (mean_proto.abs() + 1e-8)

    dim_cv = cv.mean(dim=0)  # (D,)
    stability = 1.0 / (1.0 + dim_cv)

    threshold = stability.quantile(1.0 - causal_ratio)
    causal_mask = stability >= threshold

    n_causal = causal_mask.sum().item()
    print(f"Causal mask: {n_causal}/{hidden_dim} dims marked causal "
          f"(ratio={n_causal / hidden_dim:.2f})")

    return causal_mask, stability


@torch.no_grad()
def compute_base_ssl_loss(model, ssl_loss_fn, dataloader, device):
    """Compute average SSL reconstruction loss on source-domain data.

    This serves as the reference point: when test-time SSL loss exceeds
    this value, drift is occurring.

    Returns:
        base_loss: float — average SSL loss on source data
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Computing base SSL loss"):
        ppi = batch["ppi"].to(device)
        flow_stats = batch.get("flow_stats")
        if flow_stats is not None:
            flow_stats = flow_stats.to(device)
        loss, _ = ssl_loss_fn(model, ppi, flow_stats)
        total_loss += loss.item()
        n_batches += 1

    base_loss = total_loss / max(n_batches, 1)
    print(f"Base SSL loss: {base_loss:.4f}")
    return base_loss
