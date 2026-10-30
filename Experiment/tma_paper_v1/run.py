"""Paper-protocol TMA head repair. Reads immutable completed pilot artifacts only."""
import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ['tma_pilot_v1', 'tma_analysis_v1', 'tma_controls_v1']:
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader, confusion, class_stats
from controls import query_indices, reference_data
from data import FEATURES, MONTHS, Preprocessor, finish, implementation, sha, validate_complete, write_csv, write_json
from pilot import Model, choose_reference, evaluate, forward_batches, mapped, select, stats
from paper_engine import build_head_training_set, fit_head

METHODS = [('random', 'target'), ('random', 'care'), ('badge', 'target'),
           ('badge', 'care'), ('margin', 'care')]
OLD_METHODS = ['static_mlp', 'static_rf', 'random_target', 'random_replay',
               'random_care', 'margin_care', 'badge_target', 'badge_care']
CONTROL_METHODS = ['random_kd', 'badge_replay', 'badge_kd', 'hybrid_target', 'hybrid_care']
PROTOCOL = dict(epochs=30, batch_size=64, target_repeat=2, learning_rate=.001,
                weight_decay=.0001, kd_weight=.5, temperature=2., references_per_class=5)


def code_identity():
    files = list(HERE.glob('*.py')) + [HERE / 'provenance.json']
    files += [EXPERIMENT / p for p in ['tma_analysis_v1/analyze.py',
              'tma_analysis_v1/feature_audit.py', 'tma_controls_v1/controls.py',
              'tma_server_v1/stage0.py']]
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in files if not p.name.startswith('test_')}


def output_directory(root, source_seed):
    # No arbitrary output destination: never write into cache, old seeds or CESNET.
    base = root.resolve() / 'paper_protocol_v1'
    result = base / f'source_{source_seed}'
    if base.is_symlink() or result.is_symlink():
        raise ValueError('Output directories must not be symlinks')
    return result


def existing_output(output, signature):
    if output.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    if output.exists() and any(output.iterdir()) and not (output / 'config.json').exists():
        raise ValueError('Refusing unrecognized nonempty output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    return validate_complete(output, signature)


def complete(output, signature):
    names = [str(p.relative_to(output)) for p in output.rglob('*')
             if p.is_file() and p.name not in ('complete.json', 'config.json', '.lock')]
    # Include nested completion/config records as well, so resume metadata is protected.
    names += [str(p.relative_to(output)) for p in output.rglob('complete.json') if p.parent != output]
    names += [str(p.relative_to(output)) for p in output.rglob('config.json') if p.parent != output]
    finish(output, signature, names)


def verify_used_inputs(output, **readers):
    for path in output.rglob('inputs.json'):
        for role, files in json.loads(path.read_text()).items():
            reader = readers.get(role)
            if files and reader is None:
                raise ValueError('Missing input reader for completed run: ' + role)
            for name, expected in files.items():
                if reader.marker['artifacts'].get(name) != expected:
                    raise ValueError('Input manifest changed: ' + name)
                reader.file(name)


def context(root, source_seed):
    reader, cache = Reader(root / f'seed_{source_seed}'), Reader(root / 'cache')
    original = reader.marker['signature']
    if original['cache_sha256'] != sha(cache.root / 'complete.json'):
        raise ValueError('Pilot checkpoint and cache do not match')
    if original['implementation'] != implementation():
        raise ValueError('Pilot implementation changed; restore its original .py files')
    versions = {n: importlib.metadata.version(n) for n in original['versions']}
    if original['versions'] != versions:
        raise ValueError('Use the original traffic-ncde package versions')
    classes = json.loads(reader.file('run_info.json').read_text())['classes']
    signature = dict(schema=1, source_seed=source_seed, versions=versions,
                     source_marker=sha(reader.root / 'complete.json'),
                     cache_marker=sha(cache.root / 'complete.json'), pilot_code=implementation(),
                     code=code_identity())
    return reader, cache, classes, signature


def metrics(y, before, after, historical, classes):
    """All methods, including RF, use the SAME static-MLP diagnostic groups."""
    summary, rows, cm = evaluate(y, before, after, historical, classes)
    summary['initial_collapse_count'] = summary.pop('collapse_count')
    summary['remaining_collapse_count'] = summary['initial_collapse_count'] - summary['recovered_to_recall_0_1']
    summary['high_recall_group_f1'] = summary.pop('stable_f1')
    summary['high_recall_group_delta_f1'] = summary.pop('stable_delta_f1')
    summary['negative_flip_rate_vs_static_mlp'] = summary.pop('negative_flip_rate')
    summary['new_collapses_among_historically_learned'] = summary.pop('new_collapses')
    # Independently check confusion, F1 and the remaining-collapse count.
    independent = confusion(y, after, len(classes))
    support, recalls, f1 = class_stats(independent)
    if not np.array_equal(cm, independent):
        raise ValueError('Independent confusion check failed')
    expected = f1[support > 0].mean() if (support > 0).any() else np.nan
    actual = summary['macro_f1_present'] if summary['macro_f1_present'] is not None else np.nan
    if not np.isclose(actual, expected, equal_nan=True):
        raise ValueError('Independent macro-F1 check failed')
    remaining = sum(r['collapse'] and recalls[i] < .1 for i, r in enumerate(rows))
    if summary['remaining_collapse_count'] != remaining:
        raise ValueError('Remaining-collapse check failed')
    for row in rows:
        row['high_recall_group'] = row.pop('stable')
        recall = row['after_recall']
        row['recovery_status_per_run'] = (('recovered' if recall >= .30 else
            'partial' if recall >= .05 else 'failed') if row['collapse'] else 'not_in_collapse_group')
    summary['group_definition'] = 'static_mlp_history_and_target_recall_v1'
    return summary, rows, cm


def check_predictions(base, target, classes):
    y = mapped(target['y'], classes)
    if not (np.array_equal(base['ids'], target['ids']) and np.array_equal(base['labels'], y)
            and np.array_equal(base['device_ids'], target['y'])):
        raise ValueError('Saved prediction identities/labels differ from cache')
    return y


def audit(root, source_seed=0):
    reader, cache, classes, signature = context(root, source_seed)
    output = output_directory(root, source_seed) / 'audit'
    control_root = root / 'controls_v1' / f'seed_{source_seed}'
    controls = Reader(control_root) if (control_root / 'complete.json').exists() else None
    signature = dict(signature, controls_marker=sha(control_root / 'complete.json') if controls else None)
    if controls and controls.marker['signature']['original_seed_sha256'] != signature['source_marker']:
        raise ValueError('Controls refer to a different pilot')
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, controls=controls)
        print('Verified metric audit reused:', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    h = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(h['labels'], h['mlp'], len(classes))
    records, per_class = [], []
    for month in MONTHS:
        target = cache.arrays(month + '_eval.npz', ['ids', 'y'])
        base = reader.arrays(month + '/predictions.npz', ['ids', 'labels', 'device_ids'] + OLD_METHODS)
        y = check_predictions(base, target, classes)
        batches = [('legacy_pilot_equal_ce_fixed_steps', base, OLD_METHODS)]
        if controls:
            extra = controls.arrays(month + '/predictions.npz', ['ids', 'labels', 'device_ids'] + CONTROL_METHODS)
            check_predictions(extra, target, classes)
            batches.append(('legacy_controls_equal_ce_fixed_steps', extra, CONTROL_METHODS))
        for protocol, values, names in batches:
            for method in names:
                s, rows, _ = metrics(y, base['static_mlp'], values[method], historical, classes)
                meta = dict(source_seed=source_seed, month=month, method=method, protocol=protocol)
                records.append(dict(**meta, **s))
                per_class.extend(dict(**meta, **row) for row in rows)
    write_csv(output / 'legacy_metrics_common_groups.csv', records)
    write_csv(output / 'legacy_per_class_common_groups.csv', per_class)
    reference_data(reader, cache, classes)  # Validate original identities and class coverage.
    refs = pd.read_csv(reader.file('reference_ids.csv'), dtype={'record_id': str})
    refs.groupby(['device_id', 'role']).size().rename('rows').reset_index().to_csv(output / 'reference_sources.csv', index=False)
    (output / 'REPORT.txt').write_text(
        '旧结果重统计完成；未训练模型。所有方法使用同一静态 MLP 崩溃/高召回组。\n'
        'initial_collapse_count 是原崩溃组大小；remaining_collapse_count 是修复后仍低于 .1 的类数。\n'
        '高召回组不同于原论文固定20类稳定组；NFR 均相对静态 MLP。\n'
        'RF 原有自身分组诊断仍保留在旧目录；此处增加共同分组对照。\n'
        '恢复等级按单次召回 .05/.30 划分；原稿跨种子等级应由平均召回再划分。\n'
        '旧 care 为等权 CE 固定步数配置，不是原稿混合 epoch 配置。\n')
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
                                          controls=controls.consumed if controls else {}))
    complete(output, signature)
    print('METRIC AUDIT COMPLETE:', output, flush=True)
    return output


def fit_paper(source_head, target_x, target_y, ref_x, ref_y, teacher, care, seed, device, protocol):
    start = time.perf_counter()
    n, r = len(target_y), len(ref_y) if care else 0
    rho, epochs, batch = protocol['target_repeat'], protocol['epochs'], protocol['batch_size']
    entries = rho * n + r
    trace = dict(protocol='paper_multiset_epochs', usable_known_labels=n, reference_rows=r,
                 multiset_entries=entries, optimizer_steps=epochs * math.ceil(entries / batch) if n else 0,
                 target_presentations=epochs * rho * n,
                 reference_ce_presentations=epochs * r if n else 0,
                 reference_kd_presentations=epochs * entries if care and n else 0,
                 target_ce_mixture_weight=rho * n / entries if n else None,
                 reference_ce_mixture_weight=r / entries if n else None,
                 kd_weight=protocol['kd_weight'] if care else 0., seed=seed, **{
                     k: protocol[k] for k in ['epochs', 'batch_size', 'target_repeat', 'learning_rate', 'weight_decay', 'temperature']})
    if not n:
        trace['skipped'] = 'No usable known-class queries; keep source head'
        return copy.deepcopy(source_head).cpu().eval(), trace
    indices = torch.arange(r, dtype=torch.long)
    x, y = build_head_training_set(target_x, target_y, ref_x, ref_y, indices, rho)
    head = fit_head(SimpleNamespace(cls_head=source_head), x, y,
                    protocol['learning_rate'], epochs, batch, protocol['weight_decay'], torch.device(device),
                    distill_features=ref_x if care else None, distill_logits=teacher if care else None,
                    distill_weight=protocol['kd_weight'] if care else 0.,
                    distill_temperature=protocol['temperature'], seed=seed)
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError('Non-finite repaired head')
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    trace['train_wall_s'] = time.perf_counter() - start
    return head.cpu().eval(), trace


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8, protocol=None):
    protocol = dict(PROTOCOL if protocol is None else protocol)  # Overrides for synthetic tests only.
    reader, cache, classes, signature = context(root, source_seed)
    signature = dict(signature, repair_seed=repair_seed, device=device, threads=threads, protocol=protocol,
                     queries='reuse_source_seed' if repair_seed == source_seed else 'select_on_fixed_source')
    output = output_directory(root, source_seed) / f'repair_{repair_seed}'
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache)
        print('PAPER REPAIR COMPLETE (verified existing run):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.manual_seed(repair_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(reader.file('source.pt'), map_location='cpu', weights_only=True)
    if checkpoint['classes'] != classes or checkpoint['input_features'] != FEATURES:
        raise ValueError('Source class/feature mapping changed')
    model = Model(2 * len(FEATURES), len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    prep = Preprocessor()
    with np.load(reader.file('preprocessor.npz'), allow_pickle=False) as f:
        prep.median, prep.mean, prep.std = f['median'], f['mean'], f['std']
    if repair_seed == source_seed:
        rx, ry = reference_data(reader, cache, classes)
        origins = pd.read_csv(reader.file('reference_ids.csv'), dtype={'record_id': str}).to_dict('records')
    else:
        train, ref = [cache.arrays(role + '.npz', ['ids', 'x', 'y']) for role in ['train', 'reference']]
        samples, origins = choose_reference(train, ref, classes, repair_seed)
        rx, ry = samples['x'], mapped(samples['y'], classes)
    write_csv(output / 'reference_ids.csv', origins)
    if (ry < 0).any() or not np.all(np.bincount(ry, minlength=len(classes)) == 5):
        raise ValueError('Expected five references per known class')
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    h = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(h['labels'], h['mlp'], len(classes))
    summaries, per_class, paired = [], [], []
    for month in MONTHS:
        part = output / month
        month_signature = dict(signature, month=month)
        if existing_output(part, month_signature):
            verify_used_inputs(part, pilot=reader, cache=cache)
            print(month, 'verified completed month reused', flush=True)
        else:
            write_json(part / 'config.json', month_signature)
            print(f'{month}: encoding cached samples; source={source_seed}, repair={repair_seed}', flush=True)
            query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
            target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
            if len(np.unique(query['ids'])) != len(query['ids']) or np.intersect1d(query['ids'], target['ids']).size:
                raise ValueError('Duplicate queries or query/test identity overlap')
            base = reader.arrays(month + '/predictions.npz', ['ids', 'labels', 'device_ids'] + OLD_METHODS)
            ey = check_predictions(base, target, classes)
            qy = mapped(query['y'], classes)
            qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
            efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
            qlogits = forward_batches(model.head, qfeat.numpy(), device)
            before = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
            if not np.array_equal(before, base['static_mlp']):
                raise ValueError('Reloaded source predictions differ; check environment')
            budget = reader.marker['signature']['budget']
            selections = {}
            for selector in ['random', 'badge', 'margin']:
                ids = (query_indices(reader, month, selector, query) if repair_seed == source_seed else
                       select(selector, qfeat, qlogits, budget, repair_seed))
                if len(ids) != budget or len(np.unique(ids)) != budget:
                    raise ValueError('Paid-label budget/uniqueness mismatch')
                selections[selector] = ids
                write_csv(part / (selector + '_queries.csv'), [dict(record_id=str(query['ids'][i]),
                          device_id=int(query['y'][i]), known_class=bool(qy[i] >= 0)) for i in ids])
            predictions = dict(ids=target['ids'], labels=ey, device_ids=target['y'],
                               static_mlp=base['static_mlp'], static_rf=base['static_rf'])
            counts = {name: (0, 0) for name in ['static_mlp', 'static_rf']}
            for selector, variant in METHODS:
                name = selector + '_' + variant
                ids = selections[selector]
                usable = ids[qy[ids] >= 0]
                print(f'{month}: {name}, labels={len(ids)}, usable={len(usable)}, {protocol["epochs"]} epochs', flush=True)
                head, trace = fit_paper(model.head, qfeat[usable], qy[usable], rfeat, ry, teacher,
                                        variant == 'care', repair_seed, device, protocol)
                torch.save(head.state_dict(), part / (name + '_head.pt'))
                # Validate the saved artifact, not just the in-memory result.
                loaded = copy.deepcopy(model.head)
                loaded.load_state_dict(torch.load(part / (name + '_head.pt'), map_location='cpu', weights_only=True))
                pred = forward_batches(head, efeat.numpy(), device).argmax(1).numpy()
                if not np.array_equal(pred, forward_batches(loaded, efeat.numpy(), device).argmax(1).numpy()):
                    raise ValueError('Saved-head reload predictions differ')
                predictions[name] = pred
                counts[name] = (len(ids), len(usable))
                write_json(part / (name + '_trace.json'), trace)
            rows_out, classes_out = [], []
            for name in ['static_mlp', 'static_rf'] + [s + '_' + v for s, v in METHODS]:
                s, rows, cm = metrics(ey, before, predictions[name], historical, classes)
                meta = dict(source_seed=source_seed, repair_seed=repair_seed, month=month,
                            method=name, protocol='static' if name.startswith('static_') else 'paper_multiset_epochs')
                rows_out.append(dict(**meta, label_queries=counts[name][0], usable_known_labels=counts[name][1], **s))
                classes_out.extend(dict(**meta, **r) for r in rows)
                np.save(part / (name + '_confusion.npy'), cm)
            np.savez_compressed(part / 'predictions.npz', **predictions)
            write_csv(part / 'summary.csv', rows_out)
            write_csv(part / 'per_class.csv', classes_out)
            current = {r['method']: r for r in rows_out}
            pairs = [('random_care', 'random_target'), ('badge_care', 'badge_target')]
            comparisons = [(left, right, current[right], 'within_paper_protocol') for left, right in pairs]
            if repair_seed == source_seed:
                for selector, variant in METHODS:
                    name = selector + '_' + variant
                    old, _, _ = metrics(ey, before, base[name], historical, classes)
                    comparisons.append((name, name, old, 'paper_minus_legacy_same_queries'))
            deltas = []
            for left, right, old, kind in comparisons:
                row = dict(month=month, comparison=kind, left=left, right=right)
                for metric in ['macro_f1_present', 'collapse_f1', 'high_recall_group_f1',
                               'negative_flip_rate_vs_static_mlp']:
                    a, b = current[left][metric], old[metric]
                    row[metric + '_pp'] = 100 * (a - b) if a is not None and b is not None else None
                row['remaining_collapse_count_delta'] = current[left]['remaining_collapse_count'] - old['remaining_collapse_count']
                deltas.append(row)
            write_csv(part / 'paired_deltas.csv', deltas)
            write_json(part / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed))
            complete(part, month_signature)
        summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
        per_class.extend(pd.read_csv(part / 'per_class.csv').to_dict('records'))
        paired.extend(pd.read_csv(part / 'paired_deltas.csv').to_dict('records'))
        write_csv(output / 'summary.csv', summaries)
        write_csv(output / 'per_class.csv', per_class)
        write_csv(output / 'paired_deltas.csv', paired)
        print(month, 'complete; partial summary saved', flush=True)
    table = pd.DataFrame(summaries)
    report = '\n'.join(['TMA 原稿混合 epoch 协议；固定源模型，只训练分类头。',
        f'source_seed={source_seed}, repair_seed={repair_seed}; 协议={json.dumps(protocol)}',
        table[['month', 'method', 'macro_f1_present', 'collapse_f1', 'initial_collapse_count',
               'remaining_collapse_count', 'high_recall_group_f1', 'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        '所有方法使用同一静态 MLP 崩溃/高召回组；静态 RF 的此处 NFR 也相对 MLP。',
        '每月独立重置源头，预算按月计；未知查询扣预算但不进入闭集头。',
        'target-only 与 CARE 均30 epochs，但多重集大小不同，更新数不同；不能据此单独归因 CE/KD。',
        '保留未来时间测试，不是原 CESNET 同月查询排除协议；未验证五信号监测。',
        '所有目标月份已经查看，本轮是协议对齐；新种子不构成独立时间盲测。']) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('PAPER REPAIR COMPLETE:', output, flush=True)
    return output


def aggregate(root, source_seed, seeds):
    """Summarize repair randomness only; static rows are not independent source retrainings."""
    readers = [Reader(output_directory(root, source_seed) / f'repair_{s}') for s in seeds]
    signatures = [r.marker['signature'] for r in readers]
    for signature in signatures[1:]:
        if {k: v for k, v in signature.items() if k not in ('repair_seed', 'queries')} != {
                k: v for k, v in signatures[0].items() if k not in ('repair_seed', 'queries')}:
            raise ValueError('Cannot aggregate different source/protocol/environment runs')
    output = output_directory(root, source_seed) / ('aggregate_' + '_'.join(map(str, seeds)))
    signature = dict(inputs={str(s): sha(r.root / 'complete.json') for s, r in zip(seeds, readers)}, code=code_identity())
    if existing_output(output, signature):
        return output
    write_json(output / 'config.json', signature)
    table = pd.concat([pd.read_csv(r.file('summary.csv')) for r in readers], ignore_index=True)
    fields = ['macro_f1_present', 'collapse_f1', 'remaining_collapse_count', 'high_recall_group_f1',
              'negative_flip_rate_vs_static_mlp', 'new_collapses_among_historically_learned']
    result = table.groupby(['month', 'method'])[fields].agg(['mean', 'std', 'count'])
    result.columns = ['_'.join(c) for c in result.columns]
    result.reset_index().to_csv(output / 'summary_mean_sd.csv', index=False)
    pc = pd.concat([pd.read_csv(r.file('per_class.csv')) for r in readers], ignore_index=True)
    pc = pc[pc.collapse.astype(str).str.lower().eq('true')]
    means = pc.groupby(['month', 'method', 'device_id']).after_recall.agg(['mean', 'std', 'count']).reset_index()
    means['recovery_status_from_mean_recall'] = np.where(means['mean'] >= .30, 'recovered',
                                                        np.where(means['mean'] >= .05, 'partial', 'failed'))
    means.to_csv(output / 'collapse_recovery_from_mean_recall.csv', index=False)
    (output / 'README.txt').write_text('固定源模型，重复修复种子；std 是样本标准差，单次为空。\n'
                                      '无崩溃组月份的 collapse_f1 为空；count 是有效数，不是独立环境数。\n')
    complete(output, signature)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    p.add_argument('--source-seed', type=int, default=0)
    p.add_argument('--repair-seeds', default='0')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    if platform.system() != 'Linux':
        p.error('Run real experiments on the Linux server only; local tests use synthetic fixtures')
    if args.threads < 1 or args.source_seed < 0:
        p.error('threads must be positive and source-seed nonnegative')
    try:
        seeds = sorted(set(int(s) for s in args.repair_seeds.split(',')))
        if not seeds or min(seeds) < 0:
            raise ValueError()
    except ValueError:
        p.error('repair-seeds must be comma-separated nonnegative integers')
    if not args.audit_only and args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; activate traffic-ncde')
    root = args.root.resolve()
    # Validate inputs before creating output directories.
    context(root, args.source_seed)
    output = output_directory(root, args.source_seed)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another paper-protocol process is using this source output directory')
        audit(root, args.source_seed)
        if not args.audit_only:
            for seed in seeds:
                run(root, args.source_seed, seed, args.device, args.threads)
            print('AGGREGATE:', aggregate(root, args.source_seed, seeds), flush=True)


if __name__ == '__main__':
    main()
