"""Second-round exploratory controls using completed pilot models and caches."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'tma_pilot_v1'))
sys.path.insert(0, str(EXPERIMENT / 'tma_analysis_v1'))
from analyze import Reader
from feature_audit import timestamps
from data import FEATURES, MONTHS, Preprocessor, finish, implementation, sha, validate_complete, write_csv, write_json
from pilot import Model, evaluate, forward_batches, mapped, select, stats
from existing_care import fit_controlled

NEW = {'random': ['kd'], 'badge': ['replay', 'kd'], 'hybrid': ['target', 'care']}
OLD = {'random': ['target', 'replay', 'care'], 'margin': ['care'], 'badge': ['target', 'care']}
PAIRS = [('random_kd', 'random_target'), ('random_care', 'random_kd'),
         ('badge_replay', 'badge_target'), ('badge_kd', 'badge_target'),
         ('badge_care', 'badge_replay'), ('badge_care', 'badge_kd'),
         ('hybrid_target', 'random_target'), ('hybrid_target', 'badge_target'),
         ('hybrid_care', 'random_care'), ('hybrid_care', 'badge_care')]


def query_indices(reader, month, selector, pool):
    table = pd.read_csv(reader.file(f'{month}/{selector}_queries.csv'), dtype={'record_id': str})
    lookup = {int(k): i for i, k in enumerate(pool['ids'])}
    indices = np.array([lookup[int(k)] for k in table.record_id], dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or not np.array_equal(pool['y'][indices], table.device_id.to_numpy()):
        raise ValueError('Saved query identities or labels disagree with cache')
    return indices


def hybrid_selection(features, logits, random_indices, seed):
    # Use a fixed prefix of the original uniform sample; no labels enter here.
    budget = len(random_indices)
    uniform = np.asarray(random_indices[:budget // 2])
    remaining = np.setdiff1d(np.arange(len(features)), uniform)
    active = remaining[select('badge', features[remaining], logits[remaining], budget - len(uniform), seed)]
    selected = np.concatenate([uniform, active])
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise ValueError('Hybrid query budget or uniqueness failed')
    return selected


def reference_data(reader, cache, classes):
    table = pd.read_csv(reader.file('reference_ids.csv'), dtype={'record_id': str})
    pools = {role: cache.arrays(role + '.npz', ['ids', 'x', 'y']) for role in ['train', 'reference']}
    lookups = {role: {int(k): i for i, k in enumerate(part['ids'])} for role, part in pools.items()}
    xs, ys = [], []
    for row in table.itertuples():
        if row.role not in ('train_fallback', 'reference'):
            raise ValueError('Unexpected reference source')
        role = 'train' if row.role == 'train_fallback' else 'reference'
        i = lookups[role][int(row.record_id)]
        if int(pools[role]['y'][i]) != row.device_id:
            raise ValueError('Reference label differs from cache')
        xs.append(pools[role]['x'][i]); ys.append(row.device_id)
    labels = mapped(np.asarray(ys), classes)
    if (labels < 0).any() or not np.all(np.bincount(labels, minlength=len(classes)) == 5):
        raise ValueError('Expected original five-reference-per-class protocol')
    return np.stack(xs), labels


def query_diagnostics(month, method, classes, y, selected, before, after, ey, epred):
    selected_mask = np.zeros(len(y), dtype=bool); selected_mask[selected] = True
    rows = []
    for i, c in enumerate(classes):
        row = dict(month=month, method=method, device_id=c)
        for split, mask in [('queried', (y == i) & selected_mask), ('unqueried_pool', (y == i) & ~selected_mask)]:
            row[split + '_support'] = int(mask.sum())
            row[split + '_before_recall'] = float((before[mask] == i).mean()) if mask.any() else None
            row[split + '_after_recall'] = float((after[mask] == i).mean()) if mask.any() else None
        mask = ey == i
        row['future_support'] = int(mask.sum())
        row['future_recall'] = float((epred[mask] == i).mean()) if mask.any() else None
        rows.append(row)
    return rows


def run(root, output, seed, device='cuda', threads=8):
    reader, cache = Reader(root / f'seed_{seed}'), Reader(root / 'cache')
    original = reader.marker['signature']
    if original['cache_sha256'] != sha(root / 'cache/complete.json'):
        raise ValueError('Seed and cache differ')
    if original['implementation'] != implementation():
        raise ValueError('Pilot implementation changed; restore the original implementation first')
    versions = {n: importlib.metadata.version(n) for n in original['versions']}
    if original['versions'] != versions:
        raise ValueError('Package versions differ from first round; use its original environment')
    signature = dict(original_seed_sha256=sha(reader.root / 'complete.json'),
                     original_cache_sha256=sha(cache.root / 'complete.json'), seed=seed,
                     device=device, threads=threads, versions=versions, implementation=implementation(),
                     controls_sha256=sha(Path(__file__)), reader_sha256=sha(EXPERIMENT / 'tma_analysis_v1/analyze.py'),
                     time_helper_sha256=sha(EXPERIMENT / 'tma_analysis_v1/feature_audit.py'))
    output.mkdir(parents=True, exist_ok=True)
    if validate_complete(output, signature):
        print((output / 'REPORT.txt').read_text()); print('CONTROLS COMPLETE (verified existing run)', flush=True)
        return
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classes = json.loads(reader.file('run_info.json').read_text())['classes']
    checkpoint = torch.load(reader.file('source.pt'), map_location='cpu', weights_only=True)
    if checkpoint['classes'] != classes or checkpoint['input_features'] != FEATURES:
        raise ValueError('Source class/feature mapping changed')
    model = Model(2 * len(FEATURES), len(classes))
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    prep = Preprocessor()
    with np.load(reader.file('preprocessor.npz'), allow_pickle=False) as f:
        prep.median, prep.mean, prep.std = f['median'], f['mean'], f['std']
    rx, ry = reference_data(reader, cache, classes)
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    history = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(history['labels'], history['mlp'], len(classes))
    base_summary = pd.read_csv(reader.file('summary.csv'))
    base_classes = pd.read_csv(reader.file('per_class.csv'))
    summaries, class_rows, diagnostics, query_dates = [], [], [], []
    for month in MONTHS:
        print(f'{month}: encode existing query/test samples, seed={seed}', flush=True)
        query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y', 'times'])
        target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
        qy, ey = mapped(query['y'], classes), mapped(target['y'], classes)
        qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
        efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
        qlogits = forward_batches(model.head, qfeat.numpy(), device)
        qbefore = qlogits.argmax(1).numpy()
        base = reader.arrays(month + '/predictions.npz', ['ids', 'labels', 'device_ids', 'static_mlp'] +
                             [s + '_' + v for s, variants in OLD.items() for v in variants])
        if not np.array_equal(base['ids'], target['ids']) or not np.array_equal(base['labels'], ey) or not np.array_equal(base['device_ids'], target['y']):
            raise ValueError('Evaluation sample identities or labels changed')
        before = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
        if not np.array_equal(before, base['static_mlp']):
            raise ValueError('Reloaded source predictions differ; investigate environment before comparing controls')
        selected = {s: query_indices(reader, month, s, query) for s in OLD}
        print(f'{month}: hybrid selection (uniform half + BADGE half)', flush=True)
        selected['hybrid'] = hybrid_selection(qfeat, qlogits, selected['random'], seed)
        dates = timestamps(query['times'])[1].strftime('%Y-%m-%d')
        for selector, indices in selected.items():
            frame = pd.DataFrame(dict(device_id=query['y'], date=dates,
                                      selected=np.isin(np.arange(len(qy)), indices)))
            grouped = frame.groupby(['device_id', 'date']).agg(candidates=('selected', 'size'), queried=('selected', 'sum')).reset_index()
            query_dates.extend(dict(month=month, selector=selector, **r) for r in grouped.to_dict('records'))
        part = output / month; part.mkdir(exist_ok=True)
        write_csv(part / 'hybrid_queries.csv', [dict(record_id=str(query['ids'][i]), device_id=int(query['y'][i]),
                                                   known_class=bool(qy[i] >= 0)) for i in selected['hybrid']])
        predictions = dict(ids=target['ids'], labels=ey, device_ids=target['y'])
        for selector, variants in OLD.items():
            for variant in variants:
                method = selector + '_' + variant
                head = copy.deepcopy(model.head)
                head.load_state_dict(torch.load(reader.file(month + '/' + method + '_head.pt'), map_location='cpu', weights_only=True))
                qpred = forward_batches(head, qfeat.numpy(), device).argmax(1).numpy()
                diagnostics.extend(query_diagnostics(month, method, classes, qy, selected[selector], qbefore, qpred, ey, base[method]))
        for selector, variants in NEW.items():
            ids = selected[selector]
            usable = ids[qy[ids] >= 0]
            for variant in variants:
                method = selector + '_' + variant
                print(f'{month}: {method}, paid={len(ids)}, usable={len(usable)}', flush=True)
                if len(usable):
                    head, trace = fit_controlled(model.head, qfeat[usable], qy[usable], rfeat, ry, teacher,
                        replay_ce=variant in ('replay', 'care'), kd_weight=.5 if variant in ('kd', 'care') else 0.,
                        temperature=2., target_weight=1., replay_weight=1., steps=original['repair_steps'],
                        batch_size=128, lr=.001, weight_decay=.0001, seed=seed, device=torch.device(device), log_every=0)
                    if selector in ('random', 'badge'):
                        old_trace = json.loads(reader.file(month + '/' + selector + '_care_trace.json').read_text())
                        for key in ['target_stream_sha256', 'reference_stream_sha256', 'optimizer_steps']:
                            if trace[key] != old_trace[key]:
                                raise ValueError('Batch stream changed in matched loss control: ' + key)
                else:
                    head, trace = copy.deepcopy(model.head), dict(skipped='No known queried labels')
                write_json(part / (method + '_trace.json'), trace)
                torch.save(head.state_dict(), part / (method + '_head.pt'))
                pred = forward_batches(head, efeat.numpy(), device).argmax(1).numpy()
                qpred = forward_batches(head, qfeat.numpy(), device).argmax(1).numpy()
                predictions[method] = pred
                metrics, rows, cm = evaluate(ey, before, pred, historical, classes)
                row = dict(seed=seed, month=month, method=method, label_queries=len(ids), usable_known_labels=len(usable), **metrics)
                summaries.append(row)
                class_rows.extend(dict(seed=seed, month=month, method=method, **r) for r in rows)
                diagnostics.extend(query_diagnostics(month, method, classes, qy, ids, qbefore, qpred, ey, pred))
                np.save(part / (method + '_confusion.npy'), cm)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        np.savez_compressed(part / 'predictions.npz', **predictions)
        combined = pd.concat([base_summary, pd.DataFrame(summaries)], ignore_index=True)
        combined.to_csv(output / 'summary.csv', index=False)
        pd.concat([base_classes, pd.DataFrame(class_rows)], ignore_index=True).to_csv(output / 'per_class.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(output / 'query_generalization.csv', index=False)
        pd.DataFrame(query_dates).to_csv(output / 'query_dates.csv', index=False)
    pairs = []
    for month in MONTHS:
        table = combined[combined.month == month].set_index('method')
        for left, right in PAIRS:
            row = dict(month=month, comparison=left + ' minus ' + right)
            for metric in ['macro_f1_present', 'macro_recall_present', 'accuracy_known', 'negative_flip_rate']:
                row[metric + '_pp'] = 100 * float(table.loc[left, metric] - table.loc[right, metric])
            pairs.append(row)
    write_csv(output / 'paired_deltas.csv', pairs)
    collapse_cases = base_classes[(base_classes.method == 'static_mlp') & base_classes.collapse.astype(str).str.lower().eq('true')][['month', 'device_id']]
    detail = pd.DataFrame(diagnostics).merge(collapse_cases, on=['month', 'device_id'], how='inner')
    date_detail = pd.DataFrame(query_dates).merge(collapse_cases, on=['month', 'device_id'], how='inner')
    date_lines = []
    for (month, c, selector), group in date_detail.groupby(['month', 'device_id', 'selector']):
        paid = group[group.queried > 0]
        date_lines.append(f'{month} device={c} {selector}: ' + json.dumps({str(d): int(n) for d, n in zip(paid.date, paid.queried)}, ensure_ascii=False))
    cols = ['month', 'method', 'device_id', 'queried_support', 'queried_after_recall',
            'unqueried_pool_support', 'unqueried_pool_after_recall', 'future_support', 'future_recall']
    report = '\n'.join(['TMA 第二轮受控探索（沿用首轮源模型、参考样本与测试集）',
        'kd=目标CE+KD；replay=目标CE+参考CE；care=目标CE+参考CE+KD。',
        'hybrid=固定一半随机、一半BADGE，预算与首轮相同，不使用真类决定查询。',
        '\n成对指标差值（百分点）：', pd.DataFrame(pairs).to_string(index=False, float_format=lambda x: f'{x:+.2f}'),
        '\n崩溃设备的拟合与泛化（召回率 0--1）：', detail[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '\n崩溃设备实际查询的日期及条数：', '\n'.join(date_lines),
        '\n已查询样本的召回是训练拟合指标，不是测试成绩。未查询候选也可能存在时间相关性。',
        '查询池所有标签只用于离线诊断；优化仅使用已付费查询与历史参考标签。',
        '未来测试已用于第一轮诊断，本轮属于探索，不能作为新方法的独立确认性测试。',
        '保留全部月份与失败结果；不把单种子或同一环境变化中的多个设备当作独立重复。']) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(seed=reader.consumed, cache=cache.consumed))
    names = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name not in ('complete.json', 'config.json')]
    finish(output, signature, names)
    print(report); print('CONTROLS COMPLETE:', output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Run real experiments on the Linux server only')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; check traffic-ncde environment')
    if a.threads < 1:
        p.error('--threads must be positive')
    output = a.root / 'controls_v1' / f'seed_{a.seed}'
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a.root, output, a.seed, a.device, a.threads)
