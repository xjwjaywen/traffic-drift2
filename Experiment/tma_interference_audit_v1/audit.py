"""Paired prediction audit of target-only heads versus CARE on completed TMA runs.

This is a retrospective diagnostic. It trains nothing and makes no method decision.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT / 'tma_analysis_v1'))
from analyze import Reader, sha  # noqa: E402


def f1(y, pred, class_index):
    actual = y == class_index
    predicted = pred == class_index
    tp = int((actual & predicted).sum())
    denominator = int(actual.sum() + predicted.sum())
    return 2 * tp / denominator if denominator else float('nan')


def compare(y, source, target, care, class_index, classes, seed, pair, month,
            paid_true_labels, focus_selected):
    """Count paired outcomes on identical examples, retaining class routes."""
    y, source, target, care = (np.asarray(a) for a in (y, source, target, care))
    if any(a.shape != y.shape for a in (source, target, care)) or y.ndim != 1:
        raise ValueError('Prediction arrays must be aligned one-dimensional arrays')
    nclasses = len(classes)
    if (np.any((y < -1) | (y >= nclasses)) or
            any(np.any((p < 0) | (p >= nclasses)) for p in (source, target, care))):
        raise ValueError('Invalid class index in predictions')
    known = y >= 0
    if not np.any(known & (y == class_index)):
        raise ValueError('The requested device has no evaluation support')
    source_ok = known & (source == y)
    target_ok = known & (target == y)
    care_ok = known & (care == y)
    victim = known & (y == class_index)
    others = known & (y != class_index)
    t_only = target_ok & ~care_ok
    c_only = care_ok & ~target_ok

    row = dict(seed=seed, pair=pair, month=month, device_id=int(classes[class_index]),
        paid_true_labels=paid_true_labels, focus_selected=focus_selected,
        support=int(victim.sum()),
        target_f1=f1(y[known], target[known], class_index),
        care_f1=f1(y[known], care[known], class_index),
        target_tp=int((victim & target_ok).sum()),
        care_tp=int((victim & care_ok).sum()),
        target_fp=int((others & (target == class_index)).sum()),
        care_fp=int((others & (care == class_index)).sum()),
        victim_target_only_correct=int((victim & t_only).sum()),
        victim_care_only_correct=int((victim & c_only).sum()),
        target_only_false_device_positives=int((others & (target == class_index) &
            (care != class_index)).sum()),
        care_only_false_device_positives=int((others & (care == class_index) &
            (target != class_index)).sum()),
        all_target_only_correct=int(t_only.sum()),
        all_care_only_correct=int(c_only.sum()),
        source_wrong_target_correct_care_wrong=int((~source_ok & t_only).sum()),
        source_correct_target_wrong_care_correct=int((source_ok & c_only).sum()),
        source_correct_target_correct_care_wrong=int((source_ok & t_only).sum()),
        source_wrong_target_wrong_care_correct=int((~source_ok & c_only).sum()))

    class_rows = []
    for i, device_id in enumerate(classes):
        mask = known & (y == i)
        class_rows.append(dict(seed=seed, pair=pair, month=month,
            device_id=int(device_id), support=int(mask.sum()),
            target_only_correct=int((mask & t_only).sum()),
            care_only_correct=int((mask & c_only).sum()),
            source_correct_target_wrong_care_correct=int((mask & source_ok & c_only).sum()),
            source_wrong_target_correct_care_wrong=int((mask & ~source_ok & t_only).sum())))

    source_id = np.asarray(classes, dtype=np.int64)[source[victim]]
    target_id = np.asarray(classes, dtype=np.int64)[target[victim]]
    care_id = np.asarray(classes, dtype=np.int64)[care[victim]]
    routes = pd.DataFrame(dict(source_prediction=source_id,
                               target_prediction=target_id,
                               care_prediction=care_id))
    routes = routes.value_counts().rename('count').reset_index()
    routes.insert(0, 'seed', seed)
    routes.insert(1, 'pair', pair)
    routes.insert(2, 'month', month)
    routes.insert(3, 'device_id', int(classes[class_index]))
    return row, class_rows, routes


def audit(root, seeds, source_seed, month, device_id):
    source = Reader(root / f'seed_{source_seed}')
    classes = [int(x) for x in json.loads(source.file('run_info.json').read_text())['classes']]
    if device_id not in classes:
        raise ValueError(f'Device {device_id} is absent from source classes')
    class_index = classes.index(device_id)
    rows, class_rows, route_frames = [], [], []
    reference_ids = None
    for seed in seeds:
        quota = Reader(root / 'quota_query_v1' / f'source_{source_seed}' /
                       f'repair_{seed}')
        hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                        f'repair_{seed}')
        signature = quota.marker['signature']
        if (signature['source_seed'] != source_seed or signature['repair_seed'] != seed or
                signature['source_marker'] != sha(source.root / 'complete.json') or
                signature['hybrid_marker'] != sha(hybrid.root / 'complete.json')):
            raise ValueError(f'Seed {seed} does not match its source and hybrid run')
        q = quota.arrays(month + '/predictions.npz',
            ['ids', 'labels', 'device_ids', 'static_mlp', 'hybrid_target',
             'hybrid_care', 'quota_target', 'quota_care'])
        h = hybrid.arrays(month + '/predictions.npz',
            ['ids', 'labels', 'device_ids', 'static_mlp', 'hybrid_target',
             'hybrid_care'])
        for key in h:
            if not np.array_equal(q[key], h[key]):
                raise ValueError(f'Seed {seed}: quota and hybrid {key} differ')
        if reference_ids is None:
            reference_ids = q['ids']
        elif not np.array_equal(q['ids'], reference_ids):
            raise ValueError('Evaluation IDs differ across repair seeds')
        actual = q['labels']
        device_ids = q['device_ids']
        mapped = np.asarray(classes, dtype=np.int64)[actual[actual >= 0]]
        if not np.array_equal(device_ids[actual >= 0], mapped):
            raise ValueError('Saved class indices do not match device IDs')
        coverage = pd.read_csv(quota.file(month + '/coverage.csv'))
        selected = json.loads(quota.file(month + '/selection.json').read_text())
        focus = device_id in [int(x['device_id']) for x in selected['focus']]
        match = coverage[coverage.device_id == device_id]
        if len(match) != 1:
            raise ValueError(f'Seed {seed}: missing or repeated device coverage')
        match = match.iloc[0]
        for pair, target_key, care_key, paid in (
                ('hybrid', 'hybrid_target', 'hybrid_care', int(match.hybrid_true_count)),
                ('quota', 'quota_target', 'quota_care', int(match.quota_true_count))):
            row, per_class, routes = compare(actual, q['static_mlp'],
                q[target_key], q[care_key], class_index, classes, seed, pair,
                month, paid, focus if pair == 'quota' else False)
            rows.append(row)
            class_rows.extend(per_class)
            route_frames.append(routes)
    return (pd.DataFrame(rows), pd.DataFrame(class_rows),
            pd.concat(route_frames, ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    parser.add_argument('--month', default='2020-04')
    parser.add_argument('--device-id', type=int, default=20)
    args = parser.parse_args()
    seeds = [int(x) for x in args.repair_seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error('repair-seeds must be a nonempty unique list of nonnegative integers')
    if args.source_seed < 0:
        parser.error('source-seed must be nonnegative')
    root = args.root.resolve(strict=True)
    summary, classes, routes = audit(root, seeds, args.source_seed,
                                      args.month, args.device_id)
    out = (root / 'interference_audit_v1' / f'source_{args.source_seed}' /
           f'repair_{"_".join(map(str, seeds))}' / args.month /
           f'device_{args.device_id}')
    if out.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'paired_summary.csv', index=False)
    classes.to_csv(out / 'per_class_interference.csv', index=False)
    routes.to_csv(out / 'device_routes.csv', index=False)
    focus = summary[['seed', 'pair', 'paid_true_labels', 'focus_selected',
        'target_f1', 'care_f1', 'target_tp', 'care_tp', 'target_fp', 'care_fp',
        'victim_target_only_correct', 'victim_care_only_correct',
        'source_wrong_target_correct_care_wrong',
        'source_correct_target_wrong_care_correct']]
    report = '\n'.join([
        'Retrospective paired prediction audit; no training or new method decision.',
        f'Month={args.month}, device={args.device_id}, repair seeds={seeds}.',
        'All counts compare target-only and CARE on exactly the same evaluation records.',
        'The prediction differences cannot identify which CARE loss component caused them.',
        focus.to_string(index=False),
        'Device routes (source -> target-only -> CARE), sorted by seed/pair/count:',
        routes.sort_values(['seed', 'pair', 'count'], ascending=[True, True, False])
              .groupby(['seed', 'pair'], group_keys=False).head(8).to_string(index=False),
        'All tested months were previously examined; this is exploratory, not a blind test.',
    ]) + '\n'
    (out / 'REPORT.txt').write_text(report)
    print(report)
    print('INTERFERENCE AUDIT COMPLETE:', out)


if __name__ == '__main__':
    main()
