"""Retrospective 2x2 CARE-component ablation on frozen TMA paid queries.

Target-only and full CARE predictions are reused. Replay-only and KD-only
heads are newly fitted with the same paid labels, source, references, and seed.
No evaluation predictions or labels are opened before both heads are saved.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ('tma_selective_repair_v1', 'tma_paper_v1',
               'tma_analysis_v1', 'tma_pilot_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))
from trial import load_references, load_source  # noqa: E402
from run import (PROTOCOL, check_predictions, complete, context, existing_output,
                 fit_paper, metrics, output_directory)  # noqa: E402
from analyze import Reader  # noqa: E402
from paper_engine import build_head_training_set, fit_head  # noqa: E402
from pilot import forward_batches, mapped, stats  # noqa: E402
from data import sha, write_csv, write_json  # noqa: E402


def identity():
    paths = [HERE / 'experiment.py',
             EXPERIMENT / 'tma_paper_v1/paper_engine.py',
             EXPERIMENT / 'tma_paper_v1/run.py',
             EXPERIMENT / 'tma_selective_repair_v1/trial.py']
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in paths}


def output_for(root, source_seed, repair_seed, month):
    base = root / 'component_ablation_v1'
    source = base / f'source_{source_seed}'
    seed = source / f'repair_{repair_seed}'
    result = seed / month
    if any(p.is_symlink() for p in (base, source, seed, result)):
        raise ValueError('Output directory must not be a symlink')
    return result


def indices_from_ids(pool_ids, selected_ids):
    lookup = {str(key): i for i, key in enumerate(pool_ids)}
    if len(lookup) != len(pool_ids):
        raise ValueError('Duplicate query-pool ID')
    try:
        selected = np.asarray([lookup[str(key)] for key in selected_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Paid query absent from fixed pool') from exc
    if len(selected) != 1000 or len(np.unique(selected)) != 1000:
        raise ValueError('Expected 1000 distinct paid queries')
    return selected


def fit_kd_only(source_head, target_x, target_y, ref_x, teacher,
                seed, device, protocol):
    """Target CE plus reference KD, without replay CE or replay batch rows."""
    n = len(target_y)
    if not n:
        return source_head.cpu().eval(), dict(skipped='No known paid labels')
    x, y = build_head_training_set(target_x, target_y, ref_x,
        np.empty(0, dtype=np.int64), torch.empty(0, dtype=torch.long),
        protocol['target_repeat'])
    head = fit_head(SimpleNamespace(cls_head=source_head), x, y,
        protocol['learning_rate'], protocol['epochs'], protocol['batch_size'],
        protocol['weight_decay'], torch.device(device),
        distill_features=ref_x, distill_logits=teacher,
        distill_weight=protocol['kd_weight'],
        distill_temperature=protocol['temperature'], seed=seed)
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError('Non-finite KD-only head')
    return head.cpu().eval(), dict(known_queries=n, reference_rows=len(ref_x),
        multiset_entries=len(x), optimizer_steps=protocol['epochs'] *
        ((len(x) + protocol['batch_size'] - 1) // protocol['batch_size']),
        replay_ce=False, reference_kd=True)


def run_seed(root, source_seed, repair_seed, month, device_id, device, threads):
    reader, cache, classes, _ = context(root, source_seed)
    if device_id not in classes:
        raise ValueError(f'Device {device_id} is absent from source classes')
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                    f'repair_{repair_seed}')
    quota = Reader(root / 'quota_query_v1' / f'source_{source_seed}' /
                   f'repair_{repair_seed}')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        month=month, device_id=device_id, device=device, threads=threads,
        protocol=PROTOCOL, code=identity(),
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        quota_marker=sha(quota.root / 'complete.json'))
    qsig = quota.marker['signature']
    if (qsig['source_marker'] != signature['source_marker'] or
            qsig['paper_marker'] != signature['paper_marker'] or
            qsig['hybrid_marker'] != signature['hybrid_marker'] or
            qsig['cache_marker'] != signature['cache_marker'] or
            qsig['source_seed'] != source_seed or qsig['repair_seed'] != repair_seed or
            qsig['protocol'] != PROTOCOL or qsig['device'] != device or
            qsig['threads'] != threads):
        raise ValueError('Quota run differs from fixed source, labels, or protocol')
    out = output_for(root, source_seed, repair_seed, month)
    if existing_output(out, signature):
        print(month, 'seed', repair_seed, 'verified completed ablation reused', flush=True)
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
        raise ValueError('Frozen quota labels differ from the candidate pool')
    heads, traces = {}, {}
    for pair, ids in (('hybrid', hybrid_ids), ('quota', quota_ids)):
        selected = indices_from_ids(query['ids'], ids)
        paid_y = mapped(query['y'][selected], classes)
        usable = paid_y >= 0
        target_x, target_y = qfeat[selected[usable]], paid_y[usable]
        replay_protocol = dict(PROTOCOL, kd_weight=0.)
        replay_head, replay_trace = fit_paper(model.head, target_x,
            target_y, rfeat, ry, teacher, True, repair_seed, device,
            replay_protocol)
        replay_trace['reference_kd_presentations'] = 0
        kd_head, kd_trace = fit_kd_only(model.head, target_x, target_y,
            rfeat, teacher, repair_seed, device, PROTOCOL)
        for name, head, trace in (('replay_only', replay_head, replay_trace),
                                  ('kd_only', kd_head, kd_trace)):
            key = pair + '_' + name
            heads[key], traces[key] = head, trace
            torch.save(head.state_dict(), out / (key + '_head.pt'))
            print(month, 'seed', repair_seed, key, 'complete', flush=True)
    write_json(out / 'training.json', traces)

    # The two new heads are frozen before any future evaluation file is read.
    future = quota.arrays(month + '/predictions.npz',
        ['ids', 'labels', 'device_ids', 'static_mlp', 'hybrid_target',
         'hybrid_care', 'quota_target', 'quota_care'])
    target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
    y = check_predictions(future, target, classes)
    efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
    source_pred = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
    if not np.array_equal(source_pred, future['static_mlp']):
        raise ValueError('Reloaded source differs from saved completed prediction')
    history = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(history['labels'], history['mlp'], len(classes))
    rows = []
    for pair in ('hybrid', 'quota'):
        predictions = dict(target=future[pair + '_target'],
                           replay_only=forward_batches(heads[pair + '_replay_only'],
                               efeat.numpy(), device).argmax(1).numpy(),
                           kd_only=forward_batches(heads[pair + '_kd_only'],
                               efeat.numpy(), device).argmax(1).numpy(),
                           care=future[pair + '_care'])
        for method, pred in predictions.items():
            score, per_class, _ = metrics(y, source_pred, pred, historical, classes)
            victim = next(row for row in per_class if row['device_id'] == device_id)
            rows.append(dict(seed=repair_seed, month=month, pair=pair, method=method,
                paid_labels=1000, macro_f1_present=score['macro_f1_present'],
                high_recall_group_f1=score['high_recall_group_f1'],
                negative_flip_rate_vs_static_mlp=score['negative_flip_rate_vs_static_mlp'],
                device_id=device_id, device_support=victim['support'],
                device_recall=victim['after_recall'],
                device_f1=victim['after_f1'],
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
    frames = [run_seed(root, args.source_seed, seed, args.month,
                       args.device_id, args.device, args.threads) for seed in seeds]
    results = pd.concat(frames, ignore_index=True)
    aggregate = (root / 'component_ablation_v1' / f'source_{args.source_seed}' /
                 f'aggregate_{"_".join(map(str, seeds))}' / args.month /
                 f'device_{args.device_id}')
    if aggregate.is_symlink():
        raise ValueError('Aggregate directory must not be a symlink')
    aggregate.mkdir(parents=True, exist_ok=True)
    results.to_csv(aggregate / 'by_seed.csv', index=False)
    columns = ['macro_f1_present', 'device_f1', 'device_recall',
               'negative_flip_rate_vs_static_mlp']
    mean = results.groupby(['pair', 'method'])[columns].mean().reset_index()
    mean.to_csv(aggregate / 'mean.csv', index=False)
    report = '\n'.join([
        'Exploratory fixed-label 2x2 CARE-component ablation.',
        f'Month={args.month}; device={args.device_id}; repair seeds={seeds}.',
        'Target and CARE reuse completed predictions; replay-only and KD-only are newly trained.',
        'Each pair uses identical paid queries, source, historical references, and seed.',
        'Replay adds training rows and optimizer steps, so component differences are not pure loss-term causal effects.',
        'All months were previously examined; this is not a blind test.',
        'By-seed device F1:',
        results.pivot_table(index=['seed', 'pair'], columns='method',
                            values='device_f1').to_string(),
        'Across-seed means:',
        mean.to_string(index=False),
    ]) + '\n'
    (aggregate / 'REPORT.txt').write_text(report)
    print(report)
    print('COMPONENT ABLATION COMPLETE:', aggregate)


if __name__ == '__main__':
    main()
