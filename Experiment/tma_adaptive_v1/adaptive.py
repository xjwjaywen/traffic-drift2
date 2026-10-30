"""Class-conditional relaxation of CARE's historical preservation terms."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'tma_paper_v1'))
sys.path.insert(0, str(EXPERIMENT / 'tma_selective_repair_v1'))
from paper_engine import build_head_training_set
from gate import probability_of_net_gain


def drift_flags(paid_labels, source_predictions, historical, n_classes,
                min_paid=5, error_probability=.9):
    """Use paid random-query errors only; historical eligibility is fixed."""
    labels = np.asarray(paid_labels)
    source = np.asarray(source_predictions)
    if labels.shape != source.shape:
        raise ValueError('Paid labels and source predictions must align')
    flags = np.zeros(n_classes, dtype=bool)
    rows = []
    for c in range(n_classes):
        mask = labels == c
        n = int(mask.sum())
        errors = int((source[mask] != c).sum())
        correct = n - errors
        p = probability_of_net_gain(errors, correct)
        eligible = bool(historical['support'][c] >= 50 and historical['recall'][c] >= .8)
        selected = bool(eligible and n >= min_paid and p > error_probability)
        flags[c] = selected
        rows.append(dict(class_index=c, paid_true_class_support=n, source_errors=errors,
                         source_correct=correct, error_probability=p,
                         historically_learned=eligible, relax_preservation=selected))
    return flags, rows


def fit_adaptive(source_head, target_x, target_y, ref_x, ref_y, teacher,
                 relax, seed, device, protocol):
    """CARE schedule with reference CE/KD restricted to unflagged classes.

    If no class is flagged, this is exactly the original CARE training path.
    Target CE is unchanged. The stable-class KD is normalized on the stable
    class subset so flagged logits receive no direct KD gradient.
    """
    relax = np.asarray(relax, dtype=bool)
    if len(relax) != source_head.out_features or len(ref_y) != len(ref_x):
        raise ValueError('Class/reference shape mismatch')
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
    row_weight = torch.ones(len(x), device=device)
    row_weight[target_entries:] = torch.as_tensor(~relax[np.asarray(ref_y)],
                                                   dtype=torch.float32, device=device)
    stable = torch.as_tensor(np.flatnonzero(~relax), dtype=torch.long, device=device)
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
            if relax.any():
                per_row = F.cross_entropy(head(x[idx]), y[idx], reduction='none')
                loss = (per_row * row_weight[idx]).mean()
            else:
                loss = F.cross_entropy(head(x[idx]), y[idx])
            didx = torch.randint(0, r, (idx.numel(),), device=device, generator=gen)
            if len(stable) >= 2:
                student_logits = head(dx[didx])
                target_logits = teacher_logits[didx]
                if relax.any():
                    student_logits = student_logits[:, stable]
                    target_logits = target_logits[:, stable]
                student = F.log_softmax(student_logits / temp, dim=1)
                teacher_probs = F.softmax(target_logits / temp, dim=1)
                kd = F.kl_div(student, teacher_probs, reduction='batchmean') * temp ** 2
                loss = loss + float(protocol['kd_weight']) * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError('Non-finite adaptive head')
    return head.cpu().eval(), dict(known_queries=n, references=r,
        relaxed_classes=int(relax.sum()), stable_classes=int((~relax).sum()),
        reference_ce_rows=int((~relax[np.asarray(ref_y)]).sum()),
        multiset_entries=len(x), optimizer_steps=steps)
