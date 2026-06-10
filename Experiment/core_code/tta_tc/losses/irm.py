"""
IRMv1 penalty from Arjovsky et al. 2019.

Encourages the model to learn features that are invariant across
time-period environments. Features that are predictive in M1 but
not in M3 (e.g., certificate fingerprints, CDN patterns) get
penalized, forcing the model to rely on stable, causal features.
"""
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def irm_penalty(logits, labels):
    """IRMv1 gradient penalty for one environment.

    Measures how much the optimal linear classifier (a scalar w)
    deviates from 1.0. If features are invariant, w=1 is optimal
    everywhere and the penalty is zero.

    Args:
        logits: (B, C) classification logits
        labels: (B,) ground truth labels
    Returns:
        scalar penalty
    """
    w = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(w * logits, labels)
    grad = autograd.grad(loss, w, create_graph=True)[0]
    return grad ** 2


def multi_env_irm_penalty(logits, labels, env_ids, num_envs):
    """Mean IRMv1 penalty across multiple environments.

    Args:
        logits: (N, C) pooled logits from all environments
        labels: (N,) pooled labels
        env_ids: (N,) integer environment index per sample
        num_envs: number of environments
    Returns:
        scalar mean penalty
    """
    penalties = []
    for e in range(num_envs):
        mask = env_ids == e
        if mask.sum() < 2:
            continue
        penalties.append(irm_penalty(logits[mask], labels[mask]))
    if not penalties:
        return logits.new_tensor(0.0)
    return torch.stack(penalties).mean()
