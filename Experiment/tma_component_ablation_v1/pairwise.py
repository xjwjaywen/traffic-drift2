"""Exploratory paid-confusion localized KD for frozen TMA query sets.

Only paid labels define a class-confusion graph. Its Laplacian damps the
historical KD gradient on confused class contrasts while leaving replay CE
and the original optimizer schedule intact. Evaluation is read after heads
and graph decisions have been saved.
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
    paths = [HERE / 'pairwise.py', HERE / 'experiment.py',
             EXPERIMENT / 'tma_paper_v1/paper_engine.py']
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in paths}


def output_for(root, source_seed, repair_seed, month):
    base = root / 'pairwise_kd_v1'
    source = base / f'source_{source_seed}'
    seed = source / f'repair_{repair_seed}'
    result = seed / month
    if any(p.is_symlink() for p in (base, source, seed, result)):
        raise ValueError('Output directory must not be a symlink')
    return result


def paid_confusion_operator(labels, source_predictions, classes):
    """Shrink KD gradient differences along paid source-error class pairs.

    A directed paid error c->s has weight count(c->s)/(paid_count(c)+1).
    We symmetrize those weights and use (I+L)^-1, where L is the graph
    Laplacian. No evaluation labels or class-specific thresholds are used.
    """
    labels = np.asarray(labels, dtype=np.int64)
    source_predictions = np.asarray(source_predictions, dtype=np.int64)
    if (labels.ndim != 1 or source_predictions.shape != labels.shape or
            np.any((labels < 0) | (labels >= classes)) or
            np.any((source_predictions < 0) | (source_predictions >= classes))):
        raise ValueError('Invalid paid label or source-prediction arrays')
    support = np.bincount(labels, minlength=classes)
    counts = np.bincount(labels * classes + source_predictions,
                         minlength=classes * classes).reshape(classes, classes)
    np.fill_diagonal(counts, 0)
    directed = counts / (support[:, None] + 1)
    weights = (directed + directed.T) / 2
    laplacian = np.diag(weights.sum(axis=1)) - weights
    operator = np.linalg.solve(np.eye(classes) + laplacian, np.eye(classes))
    if (not np.allclose(operator.sum(axis=1), 1, atol=1e-10) or
            not np.allclose(operator, operator.T, atol=1e-10)):
        raise ValueError('Invalid paid-confusion KD operator')
    return operator, counts, support, weights


def fit_pairwise(source_head, target_x, target_y, source_pred,
                 ref_x, ref_y, teacher, seed, device, protocol):
    n, r = len(target_y), len(ref_y)
    if len(source_pred) != n or r != len(ref_x) or r != len(teacher):
        raise ValueError('Paid labels/source predictions/references do not align')
    head = copy.deepcopy(source_head).to(device)
    if not n:
        return head.cpu().eval(), dict(skipped='No known paid labels'), []
    classes = source_head.out_features
    operator, counts, support, weights = paid_confusion_operator(
        target_y, source_pred, classes)
    p = torch.as_tensor(operator, dtype=target_x.dtype, device=device)
    edges = [dict(true_class_index=i, source_class_index=j,
                  paid_true_class_support=int(support[i]),
                  paid_pair_errors=int(counts[i, j]),
                  symmetric_weight=float(weights[i, j]))
             for i in range(classes) for j in range(classes)
             if i != j and counts[i, j] > 0]
    x, y = build_head_training_set(target_x, target_y, ref_x, ref_y,
                                   torch.arange(r), protocol['target_repeat'])
    x = x.to(device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    dx, teacher_logits = ref_x.to(device), teacher.to(device)
    parameters = list(head.parameters())
    if (len(parameters) != 2 or parameters[0].shape[0] != classes or
            parameters[1].shape != (classes,)):
        raise ValueError('Pairwise KD requires a linear head with bias')
    opt = torch.optim.AdamW(parameters, lr=protocol['learning_rate'],
                            weight_decay=protocol['weight_decay'])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    temp = float(protocol['temperature'])
    steps = 0
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
            opt.zero_grad()
            with torch.no_grad():
                for parameter, g_ce, g_kd in zip(parameters, ce_grads, kd_grads):
                    parameter.grad = g_ce + protocol['kd_weight'] * (p @ g_kd)
            opt.step()
            steps += 1
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError('Non-finite paid-confusion KD head')
    return head.cpu().eval(), dict(known_queries=n, references=r,
        paid_source_errors=int(counts.sum()), directed_edges=len(edges),
        optimizer_steps=steps, replay_ce=True, reference_kd=True,
        rule='symmetric_paid_confusion_laplacian_inverse_no_threshold'), edges


def run_seed(root, source_seed, repair_seed, month, device_id, device, threads):
    reader, cache, classes, _ = context(root, source_seed)
    if device_id not in classes:
        raise ValueError(f'Device {device_id} is absent from source classes')
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                    f'repair_{repair_seed}')
    quota = Reader(root / 'quota_query_v1' / f'source_{source_seed}' /
                   f'repair_{repair_seed}')
    strength = Reader(root / 'kd_strength_control_v1' / f'source_{source_seed}' /
                      f'repair_{repair_seed}' / month / f'device_{device_id}')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        month=month, device_id=device_id, device=device, threads=threads,
        protocol=PROTOCOL, code=identity(),
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        quota_marker=sha(quota.root / 'complete.json'),
        strength_marker=sha(strength.root / 'complete.json'))
    qs = quota.marker['signature']
    ss = strength.marker['signature']
    if (any(qs[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker')) or
            any(ss[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker',
             'quota_marker')) or
            qs['source_seed'] != source_seed or qs['repair_seed'] != repair_seed or
            qs['protocol'] != PROTOCOL or qs['device'] != device or
            qs['threads'] != threads or ss['month'] != month or
            ss['device_id'] != device_id):
        raise ValueError('Completed inputs differ from fixed source/labels/protocol')
    out = output_for(root, source_seed, repair_seed, month)
    if existing_output(out, signature):
        print(month, 'seed', repair_seed, 'verified pairwise KD reused', flush=True)
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
    heads, traces, edge_rows = {}, {}, []
    for pair, ids in (('hybrid', hybrid_ids), ('quota', quota_ids)):
        selected = indices_from_ids(query['ids'], ids)
        paid_y = mapped(query['y'][selected], classes)
        usable = paid_y >= 0
        target_x, target_y = qfeat[selected[usable]], paid_y[usable]
        source_pred = forward_batches(model.head,
            target_x.numpy(), device).argmax(1).numpy()
        head, trace, edges = fit_pairwise(model.head, target_x, target_y,
            source_pred, rfeat, ry, teacher, repair_seed, device, PROTOCOL)
        heads[pair], traces[pair] = head, trace
        edge_rows.extend(dict(pair=pair, true_device_id=int(classes[e['true_class_index']]),
                              source_device_id=int(classes[e['source_class_index']]),
                              **e) for e in edges)
        torch.save(head.state_dict(), out / (pair + '_head.pt'))
        print(month, 'seed', repair_seed, pair, 'pairwise KD complete',
              trace['directed_edges'], 'paid edges', flush=True)
    write_json(out / 'training.json', traces)
    write_csv(out / 'paid_confusion_edges.csv', edge_rows)

    # Both heads and their paid-only graph decisions are frozen before evaluation.
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
            method='pairwise_kd', paid_labels=1000,
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
    pairwise = pd.concat([run_seed(root, args.source_seed, seed, args.month,
        args.device_id, args.device, args.threads) for seed in seeds],
        ignore_index=True)
    old = []
    for seed in seeds:
        base = (root / 'kd_strength_control_v1' / f'source_{args.source_seed}' /
                f'repair_{seed}' / args.month / f'device_{args.device_id}')
        old.append(pd.read_csv(base / 'metrics.csv'))
        ablation = (root / 'component_ablation_v1' / f'source_{args.source_seed}' /
                    f'repair_{seed}' / args.month)
        projection = (root / 'gradient_safe_kd_v1' / f'source_{args.source_seed}' /
                      f'repair_{seed}' / args.month / f'device_{args.device_id}')
        old += [pd.read_csv(ablation / 'metrics.csv'),
                pd.read_csv(projection / 'metrics.csv')]
    all_results = pd.concat([pd.concat(old, ignore_index=True), pairwise],
                            ignore_index=True)
    aggregate = (root / 'pairwise_kd_v1' / f'source_{args.source_seed}' /
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
        'Exploratory paid-confusion localized KD, with no named target class in the rule.',
        f'Month={args.month}; reported device={args.device_id}; repair seeds={seeds}.',
        'The graph uses paid labels/source predictions only; replay CE and CARE optimizer schedule are fixed.',
        'All TMA months were already examined; this is not independent validation.',
        'By-seed device F1:',
        all_results.pivot_table(index=['seed', 'pair'], columns='method',
            values='device_f1').to_string(),
        'Across-seed means:',
        means.to_string(index=False),
    ]) + '\n'
    (aggregate / 'REPORT.txt').write_text(report)
    print(report)
    print('PAIRWISE KD COMPLETE:', aggregate)


if __name__ == '__main__':
    main()
