"""
Active TTA samplers: pick a label budget from a pool of test features.

All samplers share the signature:
    sample(features, static_logits, predicted_classes, budget, num_classes) -> idx

Returns a 1-D LongTensor of selected indices into the pool.
"""
import torch
import torch.nn.functional as F


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    return -(p * log_p).sum(dim=1)


def random_sampler(features, static_logits, pred_classes, budget, num_classes,
                   generator=None):
    N = features.size(0)
    return torch.randperm(N, device=features.device, generator=generator)[:budget]


def entropy_sampler(features, static_logits, pred_classes, budget, num_classes,
                    generator=None):
    """Pick top-budget by predictive entropy (highest = most uncertain)."""
    h = _entropy(static_logits)
    return torch.topk(h, k=min(budget, h.size(0)), largest=True).indices


def margin_sampler(features, static_logits, pred_classes, budget, num_classes,
                   generator=None):
    """Pick top-budget by smallest top-2 margin (smallest = most uncertain)."""
    top2 = torch.topk(static_logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return torch.topk(margin, k=min(budget, margin.size(0)), largest=False).indices


def coreset_sampler(features, static_logits, pred_classes, budget, num_classes,
                    generator=None):
    """k-center greedy on normalized features (no source seeds)."""
    N = features.size(0)
    f = F.normalize(features, dim=1)
    budget = min(budget, N)

    selected = torch.empty(budget, dtype=torch.long, device=features.device)
    # Seed with a random point
    if generator is not None:
        first = torch.randint(0, N, (1,), device=features.device, generator=generator).item()
    else:
        first = torch.randint(0, N, (1,), device=features.device).item()
    selected[0] = first

    # Distance from each unlabeled point to nearest selected point (cosine distance)
    min_dist = 1.0 - f @ f[first]  # (N,)
    min_dist[first] = -float("inf")  # don't reselect

    for i in range(1, budget):
        next_idx = torch.argmax(min_dist).item()
        selected[i] = next_idx
        new_dist = 1.0 - f @ f[next_idx]
        min_dist = torch.minimum(min_dist, new_dist)
        min_dist[next_idx] = -float("inf")

    return selected


def class_balanced_random_sampler(features, static_logits, pred_classes, budget,
                                  num_classes, generator=None):
    """
    Stratified random by predicted class — gives long-tail classes coverage.
    """
    N = features.size(0)
    budget = min(budget, N)
    per_class = max(1, budget // num_classes)

    selected = []
    used = torch.zeros(N, dtype=torch.bool, device=features.device)
    for c in range(num_classes):
        mask = (pred_classes == c) & ~used
        candidates = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            continue
        k = min(per_class, candidates.size(0))
        if generator is not None:
            perm = torch.randperm(candidates.size(0), device=features.device,
                                  generator=generator)[:k]
        else:
            perm = torch.randperm(candidates.size(0), device=features.device)[:k]
        chosen = candidates[perm]
        selected.append(chosen)
        used[chosen] = True

    selected = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long,
                                                                 device=features.device)

    # Top up remaining budget by random over the rest
    remaining = budget - selected.size(0)
    if remaining > 0:
        leftover = torch.nonzero(~used, as_tuple=False).squeeze(1)
        if leftover.numel() > 0:
            k = min(remaining, leftover.size(0))
            if generator is not None:
                perm = torch.randperm(leftover.size(0), device=features.device,
                                      generator=generator)[:k]
            else:
                perm = torch.randperm(leftover.size(0), device=features.device)[:k]
            selected = torch.cat([selected, leftover[perm]])

    return selected


SAMPLERS = {
    "random": random_sampler,
    "entropy": entropy_sampler,
    "margin": margin_sampler,
    "coreset": coreset_sampler,
    "class_balanced": class_balanced_random_sampler,
}


def get_sampler(name: str):
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(SAMPLERS)}")
    return SAMPLERS[name]
