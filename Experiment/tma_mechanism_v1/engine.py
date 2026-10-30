"""Matched-step reference-CE and KD ablations of the paper CARE head."""
import copy
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tma_paper_v1'))
from paper_engine import build_head_training_set


def fit_variant(source_head, target_x, target_y, ref_x, ref_y, teacher,
                replay_ce, distill, seed, device, protocol):
    """Use CARE's multiset, shuffles, optimizer steps and KD draws for all variants.

    Masking reference CE keeps target CE contributions at their CARE scale.
    Zero-weight batches still take a weight-decay step, so step counts match.
    """
    n, r = len(target_y), len(ref_y)
    head = copy.deepcopy(source_head).to(device)
    if not n:
        return head.cpu().eval(), dict(skipped='No usable known-class query')
    target_entries = protocol['target_repeat'] * n
    x, y = build_head_training_set(target_x, target_y, ref_x, ref_y,
                                    torch.arange(r), protocol['target_repeat'])
    x = x.to(device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    dx, teacher_logits = ref_x.to(device), teacher.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=protocol['learning_rate'],
                            weight_decay=protocol['weight_decay'])
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    temp = float(protocol['temperature'])
    steps = 0
    head.train()
    for _ in range(protocol['epochs']):
        perm = torch.randperm(len(x), device=device, generator=gen)
        for idx in perm.split(protocol['batch_size']):
            if replay_ce:
                # Exactly the original CARE CE path when both components are on.
                loss = F.cross_entropy(head(x[idx]), y[idx])
            else:
                per_row = F.cross_entropy(head(x[idx]), y[idx], reduction='none')
                loss = (per_row * (idx < target_entries)).mean()
            # Consume the same random draws in all four arms. This matches the
            # original CARE trajectory in the replay+KD arm.
            didx = torch.randint(0, r, (idx.numel(),), device=device, generator=gen)
            if distill:
                student = F.log_softmax(head(dx[didx]) / temp, dim=1)
                teacher_probs = F.softmax(teacher_logits[didx] / temp, dim=1)
                kd = F.kl_div(student, teacher_probs, reduction='batchmean') * temp ** 2
                loss = loss + float(protocol['kd_weight']) * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError('Non-finite repaired head')
    return head.cpu().eval(), dict(known_queries=n, references=r,
        multiset_entries=len(x), optimizer_steps=steps, replay_ce=replay_ce,
        distill=distill, kd_weight=protocol['kd_weight'] if distill else 0.)
