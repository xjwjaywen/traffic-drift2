"""Verbatim original CARE multiset training functions; see provenance.json."""
import copy
import torch
import torch.nn.functional as F


def build_head_training_set(
    target_features,
    target_labels,
    ref_features,
    ref_labels,
    replay_idx,
    target_repeat,
):
    target_repeat = max(1, int(target_repeat))
    feature_parts = [target_features] * target_repeat
    label_parts = [torch.as_tensor(target_labels, dtype=torch.long)] * target_repeat
    if replay_idx.numel() > 0:
        feature_parts.append(ref_features[replay_idx])
        label_parts.append(torch.as_tensor(ref_labels[replay_idx.numpy()], dtype=torch.long))
    return torch.cat(feature_parts, dim=0), torch.cat(label_parts, dim=0)


def fit_head(
    model,
    train_features,
    train_labels,
    lr,
    epochs,
    batch_size,
    weight_decay,
    device,
    distill_features=None,
    distill_logits=None,
    distill_weight=0.0,
    distill_temperature=2.0,
    seed=None,
):
    head = copy.deepcopy(model.cls_head).to(device)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    x = train_features.to(device)
    y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    n = x.shape[0]
    use_distill = (
        distill_weight > 0.0
        and distill_features is not None
        and distill_logits is not None
        and distill_features.shape[0] > 0
    )
    if use_distill:
        dx = distill_features.to(device)
        teacher_logits = distill_logits.to(device)
        dn = dx.shape[0]
        temp = float(distill_temperature)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed if seed is not None else 0)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = F.cross_entropy(head(x[idx]), y[idx])
            if use_distill:
                didx = torch.randint(0, dn, (idx.numel(),), device=device, generator=gen)
                student_log_probs = F.log_softmax(head(dx[didx]) / temp, dim=1)
                teacher_probs = F.softmax(teacher_logits[didx] / temp, dim=1)
                distill_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temp ** 2)
                loss = loss + float(distill_weight) * distill_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head
