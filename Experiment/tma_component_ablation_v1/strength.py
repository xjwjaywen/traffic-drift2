"""Matched-schedule half-strength KD control for projected-KD TMA pilot.

This retrospective control asks whether projection adds value beyond weakening
KD globally. It fixes the CARE replay/optimizer schedule and changes only
KD weight from 0.5 to 0.25. All heads are saved before evaluation is opened.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment import (Reader, PROTOCOL, check_predictions, complete, context,
    existing_output, fit_paper, forward_batches, indices_from_ids,
    load_references, load_source, mapped, metrics, output_directory, sha,
    stats, write_csv, write_json)

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
KD_WEIGHT = PROTOCOL['kd_weight'] / 2


def identity():
    paths = [HERE / 'strength.py', HERE / 'experiment.py',
             EXPERIMENT / 'tma_paper_v1/paper_engine.py',
             EXPERIMENT / 'tma_paper_v1/run.py']
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in paths}


def output_for(root, source_seed, repair_seed, month, device_id):
    base = root / 'kd_strength_control_v1'
    source = base / f'source_{source_seed}'
    seed = source / f'repair_{repair_seed}'
    period = seed / month
    result = period / f'device_{device_id}'
    if any(p.is_symlink() for p in (base, source, seed, period, result)):
        raise ValueError('Output directory must not be a symlink')
    return result


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
    projected = Reader(root / 'gradient_safe_kd_v1' / f'source_{source_seed}' /
                       f'repair_{repair_seed}' / month / f'device_{device_id}')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        month=month, device_id=device_id, device=device, threads=threads,
        protocol=dict(PROTOCOL, kd_weight=KD_WEIGHT), code=identity(),
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        quota_marker=sha(quota.root / 'complete.json'),
        ablation_marker=sha(ablation.root / 'complete.json'),
        projected_marker=sha(projected.root / 'complete.json'))
    qs = quota.marker['signature']
    ab = ablation.marker['signature']
    pr = projected.marker['signature']
    if (any(qs[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker')) or
            any(ab[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker',
             'quota_marker')) or
            any(pr[key] != signature[key] for key in
            ('source_marker', 'cache_marker', 'paper_marker', 'hybrid_marker',
             'quota_marker', 'ablation_marker')) or
            qs['source_seed'] != source_seed or qs['repair_seed'] != repair_seed or
            qs['protocol'] != PROTOCOL or qs['device'] != device or
            qs['threads'] != threads or ab['month'] != month or
            ab['device_id'] != device_id or pr['month'] != month or
            pr['device_id'] != device_id):
        raise ValueError('Completed inputs differ from fixed source/labels/protocol')
    out = output_for(root, source_seed, repair_seed, month, device_id)
    if existing_output(out, signature):
        print(month, 'seed', repair_seed, 'verified half-KD control reused', flush=True)
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
        head, trace = fit_paper(model.head, qfeat[selected[usable]],
            paid_y[usable], rfeat, ry, teacher, True, repair_seed,
            device, dict(PROTOCOL, kd_weight=KD_WEIGHT))
        heads[pair], traces[pair] = head, trace
        torch.save(head.state_dict(), out / (pair + '_head.pt'))
        print(month, 'seed', repair_seed, pair, 'half-KD complete', flush=True)
    write_json(out / 'training.json', traces)

    # Neither evaluation predictions nor labels are read before both heads exist.
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
            method='half_kd', paid_labels=1000,
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
    half = pd.concat([run_seed(root, args.source_seed, seed, args.month,
        args.device_id, args.device, args.threads) for seed in seeds],
        ignore_index=True)
    old = []
    for seed in seeds:
        base = (root / 'component_ablation_v1' / f'source_{args.source_seed}' /
                f'repair_{seed}' / args.month)
        projection = (root / 'gradient_safe_kd_v1' / f'source_{args.source_seed}' /
                      f'repair_{seed}' / args.month / f'device_{args.device_id}')
        old += [pd.read_csv(base / 'metrics.csv'),
                pd.read_csv(projection / 'metrics.csv')]
    all_results = pd.concat([pd.concat(old, ignore_index=True), half],
                            ignore_index=True)
    aggregate = (root / 'kd_strength_control_v1' / f'source_{args.source_seed}' /
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
        'Exploratory half-strength KD control; weight 0.25 versus original 0.5.',
        f'Month={args.month}; device={args.device_id}; repair seeds={seeds}.',
        'Same paid labels, replay CE, source, references, and optimizer schedule as CARE and projected KD.',
        'The half-weight control was chosen after viewing 2020-04 results; this is not independent validation.',
        'By-seed device F1:',
        all_results.pivot_table(index=['seed', 'pair'], columns='method',
            values='device_f1').to_string(),
        'Across-seed means:',
        means.to_string(index=False),
    ]) + '\n'
    (aggregate / 'REPORT.txt').write_text(report)
    print(report)
    print('HALF-KD CONTROL COMPLETE:', aggregate)


if __name__ == '__main__':
    main()
