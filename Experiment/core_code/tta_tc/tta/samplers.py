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
                           ref_pred_counts=None,
                           aas_ratio=0.5, margin_ratio=0.3):
    """Absorption-guided sampling (AAS): identifies absorber-victim pairs from
    prediction-count shift and softmax leakage, then targets pair-specific
    decision boundaries.

    Two detection signals (no target labels required):
      1. Prediction-count shift: compare per-class prediction counts between
         reference and target periods to find victims (count dropped) and
         absorbers (count increased).
      2. Softmax leakage: among samples predicted as absorber a, the average
         softmax mass on victim v reveals the absorption relationship.

    Per-sample score on each (a,v) pair boundary:
        score(x) = victim_risk(v) * softmax_leakage(a,v) * exp(-|z_a - z_v|)

    Budget split: 50% AAS-scored, 30% global margin, 20% random.
    """
    N = features.size(0)
    budget = min(budget, N)
    device = features.device

    # --- Step 1: Per-class prediction counts on target period ---
    tgt_pred_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        tgt_pred_counts[c] = (pred_classes == c).sum()

    # --- Step 2: Collapse risk from count shift ---
    if ref_pred_counts is not None:
        ref = ref_pred_counts.float().to(device)
        ref_prop = ref / ref.sum().clamp(min=1)
        tgt_prop = tgt_pred_counts / tgt_pred_counts.sum().clamp(min=1)
        count_ratio = tgt_prop / ref_prop.clamp(min=1e-6)
        victim_risk = (1.0 - count_ratio).clamp(min=0)
        absorber_flag = count_ratio > 1.5
    elif source_prototypes is not None:
        feat_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(source_prototypes, dim=1)
        proto_counts = torch.zeros(num_classes, device=device)
        proto_classes = (feat_norm @ proto_norm.T).argmax(dim=1)
        for c in range(num_classes):
            proto_counts[c] = (proto_classes == c).sum()
        victim_risk = (1.0 - tgt_pred_counts / proto_counts.clamp(min=1)).clamp(min=0)
        absorber_flag = tgt_pred_counts > proto_counts * 1.5
    else:
        return margin_sampler(features, static_logits, pred_classes, budget,
                              num_classes, generator)

    # --- Step 3: Build absorption pairs from softmax leakage ---
    probs = F.softmax(static_logits, dim=1)
    absorption_score = torch.zeros(num_classes, num_classes, device=device)

    for a in range(num_classes):
        if not absorber_flag[a]:
            continue
        a_mask = pred_classes == a
        if a_mask.sum() < 10:
            continue
        avg_softmax = probs[a_mask].mean(dim=0)
        absorption_score[a] = avg_softmax * victim_risk
        absorption_score[a, a] = 0

    # Top-K pairs by absorption score (adaptive, no fixed threshold)
    flat_scores = absorption_score.flatten()
    max_pairs = min(30, (flat_scores > 0).sum().item())
    if max_pairs == 0:
        return margin_sampler(features, static_logits, pred_classes, budget,
                              num_classes, generator)

    topk_flat = torch.topk(flat_scores, k=max_pairs, largest=True)
    pair_scores = topk_flat.values
    pair_flat_idx = topk_flat.indices
    pair_indices = torch.stack([pair_flat_idx // num_classes,
                                pair_flat_idx % num_classes], dim=1)

    # --- Step 4: Budget allocation ---
    aas_budget = max(1, int(budget * aas_ratio))
    margin_budget = max(1, int(budget * margin_ratio))
    random_budget = budget - aas_budget - margin_budget

    pair_weights = pair_scores / pair_scores.sum()
    pair_budgets = (pair_weights * aas_budget).long()
    shortfall = aas_budget - pair_budgets.sum().item()
    if shortfall > 0:
        top_idx = torch.argsort(pair_scores, descending=True)[:shortfall]
        pair_budgets[top_idx] += 1

    # --- Step 5: Per-pair boundary sampling ---
    selected = []
    used = torch.zeros(N, dtype=torch.bool, device=device)

    for k in range(pair_indices.size(0)):
        a_cls = pair_indices[k, 0].item()
        v_cls = pair_indices[k, 1].item()
        b = pair_budgets[k].item()
        if b <= 0:
            continue

        candidates = torch.nonzero(
            (pred_classes == a_cls) & ~used, as_tuple=False
        ).squeeze(1)
        if candidates.numel() == 0:
            continue

        pair_unc = torch.exp(-(static_logits[candidates, a_cls]
                               - static_logits[candidates, v_cls]).abs())
        softmax_v = probs[candidates, v_cls]
        score = pair_unc * softmax_v

        k_sel = min(b, candidates.size(0))
        top_k = torch.topk(score, k=k_sel, largest=True).indices
        chosen = candidates[top_k]
        selected.append(chosen)
        used[chosen] = True

    # (b) Global margin
    remaining_idx = torch.nonzero(~used, as_tuple=False).squeeze(1)
    if remaining_idx.numel() > 0 and margin_budget > 0:
        top2 = torch.topk(static_logits[remaining_idx], k=2, dim=1).values
        m = top2[:, 0] - top2[:, 1]
        k = min(margin_budget, remaining_idx.size(0))
        top_k = torch.topk(m, k=k, largest=False).indices
        selected.append(remaining_idx[top_k])
        used[remaining_idx[top_k]] = True

    # (c) Random
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
