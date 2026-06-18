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


def absorber_aware_sampler(features, static_logits, pred_classes, budget, num_classes,
                           generator=None, source_prototypes=None,
                           aas_ratio=0.7, margin_ratio=0.2, pair_threshold=0.005):
    """Absorption-guided sampling (AAS): scores each sample by how likely it sits
    on an absorber-victim decision boundary, using only unlabeled target data and
    reference-period prototypes.

    Builds a pseudo absorption graph from prototype-vs-head disagreement, then
    scores candidates with:
        score(x) = collapse_risk(v) * p(a|x) * proto_affinity(x,v) * exp(-|z_a - z_v|)
    where a = predicted class (absorber candidate), v = nearest prototype (victim candidate).

    Budget split: 70% AAS-scored, 20% global margin, 10% random.
    """
    N = features.size(0)
    budget = min(budget, N)
    device = features.device

    if source_prototypes is None:
        return margin_sampler(features, static_logits, pred_classes, budget,
                              num_classes, generator)

    # --- Prototype assignment ---
    feat_norm = F.normalize(features, dim=1)
    proto_norm = F.normalize(source_prototypes, dim=1)
    proto_sims = feat_norm @ proto_norm.T  # (N, C)
    proto_classes = proto_sims.argmax(dim=1)

    # --- Per-class collapse risk ---
    # collapse_risk(v) = 1 - pred_count(v) / proto_count(v)
    # High when many prototype-v samples are predicted as something else.
    pred_counts = torch.zeros(num_classes, device=device)
    proto_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        pred_counts[c] = (pred_classes == c).sum()
        proto_counts[c] = (proto_classes == c).sum()
    collapse_risk = (1.0 - pred_counts / proto_counts.clamp(min=1)).clamp(min=0)

    # --- Build pseudo absorption graph (only disagreeing samples) ---
    disagree_mask = pred_classes != proto_classes
    absorption_flow = torch.zeros(num_classes, num_classes, device=device)
    if disagree_mask.any():
        pred_d = pred_classes[disagree_mask]
        proto_d = proto_classes[disagree_mask]
        for i in range(pred_d.size(0)):
            absorption_flow[pred_d[i], proto_d[i]] += 1

    flow_threshold = max(1, int(N * pair_threshold))
    significant_pairs = absorption_flow >= flow_threshold  # (C, C) bool

    if not significant_pairs.any():
        return margin_sampler(features, static_logits, pred_classes, budget,
                              num_classes, generator)

    # --- Budget allocation: 70% AAS / 20% margin / 10% random ---
    aas_budget = max(1, int(budget * aas_ratio))
    margin_budget = max(1, int(budget * margin_ratio))
    random_budget = budget - aas_budget - margin_budget

    # --- Identify significant (absorber, victim) pairs and allocate AAS budget ---
    pair_indices = torch.nonzero(significant_pairs, as_tuple=False)  # (K, 2)
    pair_flows = absorption_flow[pair_indices[:, 0], pair_indices[:, 1]]

    pair_weights = pair_flows / pair_flows.sum()
    pair_budgets = (pair_weights * aas_budget).long()
    shortfall = aas_budget - pair_budgets.sum().item()
    if shortfall > 0:
        top_pairs = torch.argsort(pair_flows, descending=True)[:shortfall]
        pair_budgets[top_pairs] += 1

    # --- Per-pair boundary sampling scored by composite formula ---
    probs = F.softmax(static_logits, dim=1)
    selected = []
    used = torch.zeros(N, dtype=torch.bool, device=device)

    for k in range(pair_indices.size(0)):
        absorber_cls = pair_indices[k, 0].item()
        victim_cls = pair_indices[k, 1].item()
        b = pair_budgets[k].item()
        if b <= 0:
            continue

        # Primary candidates: predicted as absorber, prototype nearest is victim
        candidates = torch.nonzero(
            (pred_classes == absorber_cls) & (proto_classes == victim_cls) & ~used,
            as_tuple=False
        ).squeeze(1)
        if candidates.numel() == 0:
            candidates = torch.nonzero(
                (pred_classes == absorber_cls) & ~used,
                as_tuple=False
            ).squeeze(1)
        if candidates.numel() == 0:
            continue

        cr = collapse_risk[victim_cls]
        pa = probs[candidates, absorber_cls]
        aff = proto_sims[candidates, victim_cls].clamp(min=0)
        pair_unc = torch.exp(-(static_logits[candidates, absorber_cls]
                               - static_logits[candidates, victim_cls]).abs())
        score = cr * pa * aff * pair_unc

        k_sel = min(b, candidates.size(0))
        top_k = torch.topk(score, k=k_sel, largest=True).indices
        chosen = candidates[top_k]
        selected.append(chosen)
        used[chosen] = True

    # (b) Global margin samples (covers non-collapse uncertainty)
    remaining_idx = torch.nonzero(~used, as_tuple=False).squeeze(1)
    if remaining_idx.numel() > 0:
        top2 = torch.topk(static_logits[remaining_idx], k=2, dim=1).values
        m = top2[:, 0] - top2[:, 1]
        k = min(margin_budget, remaining_idx.size(0))
        top_k = torch.topk(m, k=k, largest=False).indices
        chosen = remaining_idx[top_k]
        selected.append(chosen)
        used[chosen] = True

    # (c) Random samples (stable-class preservation + exploration)
    remaining_idx = torch.nonzero(~used, as_tuple=False).squeeze(1)
    if remaining_idx.numel() > 0 and random_budget > 0:
        k = min(random_budget, remaining_idx.size(0))
        if generator is not None:
            perm = torch.randperm(remaining_idx.size(0), device=device,
                                  generator=generator)[:k]
        else:
            perm = torch.randperm(remaining_idx.size(0), device=device)[:k]
        selected.append(remaining_idx[perm])

    if not selected:
        return margin_sampler(features, static_logits, pred_classes, budget,
                              num_classes, generator)

    return torch.cat(selected)[:budget]


SAMPLERS = {
    "random": random_sampler,
    "entropy": entropy_sampler,
    "margin": margin_sampler,
    "coreset": coreset_sampler,
    "class_balanced": class_balanced_random_sampler,
    "absorber_aware": absorber_aware_sampler,
}


def get_sampler(name: str):
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(SAMPLERS)}")
    return SAMPLERS[name]
