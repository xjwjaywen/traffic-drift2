"""Exploratory CARE with reference KD projected off paid repair gradients.

The paid examples misclassified by the frozen source define a repair gradient.
At each optimizer step, remove only the component of the historical-reference
KD gradient that opposes that repair gradient. Replay CE remains unchanged.
All selection and model weights are frozen before evaluation is opened.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiment import (Reader, PROTOCOL, build_head_training_set,
    check_predictions, complete, context, existing_output, forward_batches,
    indices_from_ids, load_references, load_source, mapped, metrics,
    output_directory, sha, stats, write_csv, write_json)

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent


def identity():
    paths = [HERE / 'projection.py', HERE / 'experiment.py',
             EXPERIMENT / 'tma_paper_v1/paper_engine.py']
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in paths}


def output_for(root, source_seed, repair_seed, month, device_id):
    base = root / 'gradient_safe_kd_v1'
    source = base / f'source_{source_seed}'
    seed = source / f'repair_{repair_seed}'
    period = seed / month
    result = period / f'device_{device_id}'
    if any(p.is_symlink() for p in (base, source, seed, period, result)):
        raise ValueError('Output directory must not be a symlink')
    return result


def project_kd(kd_grads, repair_grads):
    """Return the original KD gradient if compatible, otherwise its projection."""
    dot = sum(torch.sum(k * r) for k, r in zip(kd_grads, repair_grads))
    norm = sum(torch.sum(r * r) for r in repair_grads)
    if bool((dot < 0).item()) and bool((norm > 0).item()):
        scale = dot / norm
        return [k - scale * r for k, r in zip(kd_grads, repair_grads)], True
    return list(kd_grads), False


def fit_projected(source_head, target_x, target_y, source_pred,
                  ref_x, ref_y, teacher, seed, device, protocol):
    """Original CARE CE/replay schedule, with compatibility-projected KD."""
    n, r = len(target_y), len(ref_y)
    if len(source_pred) != n or r != len(ref_x) or r != len(teacher):
        raise ValueError('Paid labels/source predictions/references do not align')
    head = copy.deepcopy(source_head).to(device)
    if not n:
        return head.cpu().eval(), dict(skipped='No known paid labels')
    repair_mask = np.asarray(source_pred) != np.asarray(target_y)
    repair_x = target_x[repair_mask].to(device)
    repair_y = torch.as_tensor(np.asarray(target_y)[repair_mask],
                               dtype=torch.long, device=device)
    x, y = build_head_training_set(target_x, target_y, ref_x, ref_y,
                                   torch.arange(r), protocol['target_repeat'])
    x = x.to(device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    dx, teacher_logits = ref_x.to(device), teacher.to(device)
    parameters = list(head.parameters())
    opt = torch.optim.AdamW(parameters, lr=protocol['learning_rate'],
                            weight_decay=protocol['weight_decay'])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    temp = float(protocol['temperature'])
    steps = conflicts = 0
    head.train()
    for _ in range(protocol['epochs']):
        perm = torch.randperm(len(x), device=device, generator=generator)
        for idx in perm.split(protocol['batch_size']):
            ce = F.cross_entropy(head(x[idx]), y[idx])
            ce_grads = torch.autograd.grad(ce, parameters)
            didx = torch.randint(0, r, (idx.numel(),),
                                  device=device, generator=generator)
            student = F.log_softmax(head(dx[didx]) / temp, dim=1)
            teacher_probs = F.softmax(teacher_logits[didx] / temp, dim=1)
            kd = F.kl_div(student, teacher_probs, reduction='batchmean') * temp ** 2
            kd_grads = torch.autograd.grad(kd, parameters)
            if len(repair_y):
                repair_ce = F.cross_entropy(head(repair_x), repair_y)
                repair_grads = torch.autograd.grad(repair_ce, parameters)
                kd_grads, conflicted = project_kd(kd_grads, repair_grads)
                conflicts += int(conflicted)
            opt.zero_grad()
            with torch.no_grad():
                for param, g_ce, g_kd in zip(parameters, ce_grads, kd_grads):
                    param.grad = g_ce + protocol['kd_weight'] * g_kd
            opt.step()
            steps += 1
    if not all(torch.isfinite(p).all() for p in parameters):
        raise ValueError('Non-finite projected head')
    return head.cpu().eval(), dict(known_queries=n,
        source_errors_in_paid_labels=int(repair_mask.sum()), references=r,
        optimizer_steps=steps, projected_steps=conflicts,
        replay_ce=True, reference_kd=True,
        rule='remove_negative_KD_projection_on_paid_source_error_CE')


def run_seed(root, source_seed, repair_seed, month, device_id, device, threads):
    reader, cache, classes, _ = context(root, source_seed)
    if device_id not in classes:
        raise ValueError(f'Device {device_id} is absent from source classes')
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                    f'repair_{repair_seed}')
    quota = Reader(root / 'quota_query_v1' / f'source_{source_seed}' /
                   f'repair_{repair_seed}')
    ablation = Reader(root / 'component_ablation_v1' / f'source_{source_seed}' /
                      f'repair_{repair_seed}' / month)
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        month=month, device_id=device_id, device=device, threads=threads,
        protocol=PROTOCOL, code=identity(),
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        quota_marker=sha(quota.root / 'complete.json'),
        ablation_marker=sha(ablation.root / 'complete.json'))
    qs = quota.marker['signature']
    ab = ablation.marker['signature']
    if (any(qs[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker')) or
            any(ab[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker',
             'quota_marker')) or
            qs['source_seed'] != source_seed or qs['repair_seed'] != repair_seed or
            qs['protocol'] != PROTOCOL or qs['device'] != device or
            qs['threads'] != threads or ab['month'] != month or
            ab['device_id'] != device_id):
        raise ValueError('Completed inputs differ from fixed source/labels/protocol')
    out = output_for(root, source_seed, repair_seed, month, device_id)
    if existing_output(out, signature):
        print(month, 'seed', repair_seed, 'verified projected KD reused', flush=True)
        return pd.read_csv(out / 'metrics.csv')
    write_json(out / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.manual_seed(repair_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, prep = load_source(reader, classes)
    rx, ry = load_references(reader, cache, paper, classes, source_seed, repair_seed)
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
    qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
    hybrid_ids = pd.read_csv(hybrid.file(month + '/hybrid_queries.csv'),
                             dtype={'record_id': str})['record_id'].tolist()
    selection = json.loads(quota.file(month + '/selection.json').read_text())
    quota_ids = selection['query_record_ids']
    if not np.array_equal(query['y'][indices_from_ids(query['ids'], quota_ids)],
                          np.asarray(selection['paid_device_ids'])):
        raise ValueError('Saved quota labels differ from the fixed query pool')
    heads, traces = {}, {}
    for pair, ids in (('hybrid', hybrid_ids), ('quota', quota_ids)):
        selected = indices_from_ids(query['ids'], ids)
        paid_y = mapped(query['y'][selected], classes)
        usable = paid_y >= 0
        target_x, target_y = qfeat[selected[usable]], paid_y[usable]
        source_pred = forward_batches(model.head,
            target_x.numpy(), device).argmax(1).numpy()
        head, trace = fit_projected(model.head, target_x, target_y,
            source_pred, rfeat, ry, teacher, repair_seed, device, PROTOCOL)
        heads[pair], traces[pair] = head, trace
        torch.save(head.state_dict(), out / (pair + '_head.pt'))
        print(month, 'seed', repair_seed, pair, 'projected KD complete',
              trace['projected_steps'], '/', trace['optimizer_steps'], flush=True)
    write_json(out / 'training.json', traces)

    # Only after both new heads are saved are evaluation predictions/labels read.
    future = quota.arrays(month + '/predictions.npz',
        ['ids', 'labels', 'device_ids', 'static_mlp'])
    target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
    y = check_predictions(future, target, classes)
    efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
    source_pred = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
    if not np.array_equal(source_pred, future['static_mlp']):
        raise ValueError('Reloaded source predictions differ from saved source')
    history = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(history['labels'], history['mlp'], len(classes))
    rows = []
    for pair, head in heads.items():
        pred = forward_batches(head, efeat.numpy(), device).argmax(1).numpy()
        score, per_class, _ = metrics(y, source_pred, pred, historical, classes)
        victim = next(row for row in per_class if row['device_id'] == device_id)
        rows.append(dict(seed=repair_seed, month=month, pair=pair,
            method='projected_kd', paid_labels=1000,
            macro_f1_present=score['macro_f1_present'],
            high_recall_group_f1=score['high_recall_group_f1'],
            negative_flip_rate_vs_static_mlp=score['negative_flip_rate_vs_static_mlp'],
            device_id=device_id, device_support=victim['support'],
            device_recall=victim['after_recall'], device_f1=victim['after_f1'],
            device_predicted_count=victim['predicted_count']))
    write_csv(out / 'metrics.csv', rows)
    complete(out, signature)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    parser.add_argument('--month', default='2020-04')
    parser.add_argument('--device-id', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    seeds = [int(x) for x in args.repair_seeds.split(',')]
    if (not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or
            args.source_seed < 0 or args.threads < 1):
        parser.error('Invalid seed or threads')
    root = args.root.resolve(strict=True)
    projected = pd.concat([run_seed(root, args.source_seed, seed, args.month,
        args.device_id, args.device, args.threads) for seed in seeds],
        ignore_index=True)
    baselines = []
    for seed in seeds:
        path = (root / 'component_ablation_v1' / f'source_{args.source_seed}' /
                f'repair_{seed}' / args.month / 'metrics.csv')
        baselines.append(pd.read_csv(path))
    all_results = pd.concat([pd.concat(baselines, ignore_index=True),
                             projected], ignore_index=True)
    aggregate = (root / 'gradient_safe_kd_v1' / f'source_{args.source_seed}' /
                 f'aggregate_{"_".join(map(str, seeds))}' / args.month /
                 f'device_{args.device_id}')
    if aggregate.is_symlink():
        raise ValueError('Aggregate directory must not be a symlink')
    aggregate.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(aggregate / 'by_seed.csv', index=False)
    columns = ['macro_f1_present', 'device_f1', 'device_recall',
               'negative_flip_rate_vs_static_mlp']
    means = all_results.groupby(['pair', 'method'])[columns].mean().reset_index()
    means.to_csv(aggregate / 'mean.csv', index=False)
    report = '\n'.join([
        'Exploratory paid-error-gradient-projected historical KD.',
        f'Month={args.month}; device={args.device_id}; repair seeds={seeds}.',
        'Projection uses only already paid labels misclassified by the frozen source.',
        'Replay CE and the original CARE optimizer schedule are retained.',
        'All tested months were previously examined; this is not independent validation.',
        'By-seed device F1:',
        all_results.pivot_table(index=['seed', 'pair'], columns='method',
            values='device_f1').to_string(),
        'Across-seed means:',
        means.to_string(index=False),
    ]) + '\n'
    (aggregate / 'REPORT.txt').write_text(report)
    print(report)
    print('PROJECTED KD COMPLETE:', aggregate)


if __name__ == '__main__':
    main()
