"""Existing repository CARE head repair and BADGE approximation. See provenance.json."""
import copy
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F


def fit_controlled(source_head, target_x, target_y, ref_x, ref_y, teacher_logits, *,
                   replay_ce, kd_weight, temperature, target_weight, replay_weight,
                   steps, batch_size, lr, weight_decay, seed, device, log_every=200):
    head = copy.deepcopy(source_head).to(device)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    # Separate CPU generators make the target stream independent of which loss terms are enabled.
    target_idx = torch.randint(len(target_x), (steps, batch_size),
                               generator=torch.Generator().manual_seed(seed + 20011))
    ref_idx = torch.randint(len(ref_x), (steps, batch_size),
                            generator=torch.Generator().manual_seed(seed + 30011))
    trace = {
        "optimizer_steps": steps, "target_presentations": steps * batch_size,
        "reference_ce_presentations": steps * batch_size if replay_ce else 0,
        "reference_kd_presentations": steps * batch_size if kd_weight else 0,
        "target_stream_sha256": hashlib.sha256(target_idx.numpy().tobytes()).hexdigest(),
        "reference_stream_sha256": hashlib.sha256(ref_idx.numpy().tobytes()).hexdigest(),
    }
    x, y = target_x.to(device), torch.as_tensor(target_y, dtype=torch.long, device=device)
    rx, ry = ref_x.to(device), torch.as_tensor(ref_y, dtype=torch.long, device=device)
    teacher = teacher_logits.to(device)
    target_idx, ref_idx = target_idx.to(device), ref_idx.to(device)
    sums = {"target_ce": 0., "reference_ce": 0., "reference_kd": 0., "loss": 0.}
    started = time.perf_counter()
    for step in range(steps):
        tx, ridx = target_idx[step], ref_idx[step]
        ce = F.cross_entropy(head(x[tx]), y[tx])
        loss = target_weight * ce
        rce = kd = torch.zeros((), device=device)
        if replay_ce or kd_weight:
            student = head(rx[ridx])
            if replay_ce:
                rce = F.cross_entropy(student, ry[ridx])
                loss = loss + replay_weight * rce
            if kd_weight:
                kd = F.kl_div(F.log_softmax(student / temperature, dim=1),
                              F.softmax(teacher[ridx] / temperature, dim=1),
                              reduction="batchmean") * temperature ** 2
                loss = loss + kd_weight * kd
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for key, value in [("target_ce", ce), ("reference_ce", rce), ("reference_kd", kd), ("loss", loss)]:
            sums[key] += float(value.detach())
        if log_every and ((step + 1) % log_every == 0 or step + 1 == steps):
            print(f"  step {step + 1}/{steps} loss={float(loss.detach()):.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trace["train_wall_s"] = time.perf_counter() - started
    trace["mean_losses"] = {k: v / steps for k, v in sums.items()}
    return head.cpu().eval(), trace

def badge_selection(features, logits, num_classes, budget, seed):
    """BADGE: Batch Active learning by Diverse Gradient Embeddings.

    Approximates gradient embeddings as (pred_prob - one_hot) ⊗ features,
    then runs k-means++ initialization for diverse selection.
    Optimized: uses top-3 classes + PCA to 50 dims for scalability.
    """
    n = features.shape[0]
    budget = min(budget, n)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, preds.unsqueeze(1), 1.0)
    grad_coeff = probs - one_hot  # [n, C]

    feat_np = features.numpy() if isinstance(features, torch.Tensor) else features
    coeff_np = grad_coeff.numpy()
    top_k = min(3, num_classes)
    top_indices = np.argsort(np.abs(coeff_np), axis=1)[:, -top_k:]
    grad_embed = np.zeros((n, top_k * feat_np.shape[1]), dtype=np.float32)
    for i in range(top_k):
        col = top_indices[:, i]
        coeff_col = coeff_np[np.arange(n), col]
        grad_embed[:, i * feat_np.shape[1]:(i + 1) * feat_np.shape[1]] = (
            feat_np * coeff_col[:, None]
        )

    # PCA to reduce dimensionality for fast k-means++
    pca_dim = min(50, grad_embed.shape[1])
    if grad_embed.shape[1] > pca_dim:
        from sklearn.decomposition import PCA
        grad_embed = PCA(n_components=pca_dim, random_state=seed).fit_transform(grad_embed)

    # k-means++ initialization
    gen = np.random.RandomState(seed)
    selected = [gen.randint(n)]
    min_dist = np.full(n, np.inf, dtype=np.float32)
    for step in range(budget - 1):
        last = grad_embed[selected[-1]]
        dist = np.sum((grad_embed - last[None, :]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected[-1]] = 0
        total = min_dist.sum()
        if total <= 0:
            remaining = np.setdiff1d(np.arange(n), selected)
            selected.append(remaining[gen.randint(len(remaining))])
        else:
            p = min_dist / total
            selected.append(gen.choice(n, p=p))
    return torch.tensor(selected, dtype=torch.long)
