"""Read-only paired-seed and query-coverage analysis; no training or inference imports."""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tma_analysis_v1'))
from analyze import Reader, sha, MONTHS

METHODS = ['static_mlp', 'static_rf', 'random_target', 'random_care',
           'badge_target', 'badge_care', 'margin_care']
# (larger is better, displayed multiplier, unit); delta always CARE minus target-only.
METRICS = {
    'macro_f1_present': (True, 100., 'percentage_points'),
    'collapse_f1': (True, 100., 'percentage_points'),
    'high_recall_group_f1': (True, 100., 'percentage_points'),
    'negative_flip_rate_vs_static_mlp': (False, 100., 'percentage_points'),
    'remaining_collapse_count': (False, 1., 'classes'),
    'new_collapses_among_historically_learned': (False, 1., 'classes'),
}
CASES = {'2020-03': [5, 22, 26, 37], '2020-04': [20]}
TOLERANCE = 1e-12  # Raw score/count units, before conversion to percentage points.


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def timestamps(values, month, query):
    matches = []
    start = pd.Timestamp(month + ('-01' if query else '-16'), tz='UTC')
    end = pd.Timestamp(month + '-15', tz='UTC') if query else pd.Timestamp(month + '-01', tz='UTC') + pd.offsets.MonthBegin()
    for unit in ['ns', 'us', 'ms', 's']:
        dt = pd.to_datetime(values, unit=unit, utc=True, errors='coerce')
        if len(dt) and ((dt >= start) & (dt < end)).all():
            matches.append(dt)
    if len(matches) != 1:
        raise ValueError('Ambiguous timestamp units or samples outside the fixed temporal split')
    return matches[0]


def truth(values):
    strings = values.astype(str).str.lower()
    if not strings.isin(['true', 'false']).all():
        raise ValueError('Invalid Boolean flags')
    return strings.eq('true').to_numpy()


def pair_row(meta, metric, care, target, larger=True, multiplier=100., unit='percentage_points'):
    valid = pd.notna(care) and pd.notna(target)
    if valid and not np.isfinite([care, target]).all():
        raise ValueError('Infinite metric')
    delta = float(care - target) if valid else np.nan
    benefit = delta if larger else -delta
    outcome = ('missing' if not valid else 'tie' if abs(delta) <= TOLERANCE else
               'win' if benefit > 0 else 'loss')
    return dict(**meta, metric=metric, care=care, target=target,
                delta=delta * multiplier, unit=unit, preferred='higher' if larger else 'lower', outcome=outcome)


def summarize_pairs(frame, group_columns):
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        valid = group[group.outcome != 'missing']
        row.update(n_total=len(group), n_valid=len(valid), missing=int((group.outcome == 'missing').sum()),
                   wins=int((group.outcome == 'win').sum()), ties=int((group.outcome == 'tie').sum()),
                   losses=int((group.outcome == 'loss').sum()),
                   mean_delta=valid.delta.mean(), sd_delta=valid.delta.std(ddof=1),
                   min_delta=valid.delta.min(), max_delta=valid.delta.max(),
                   unit=group.unit.iloc[0], preferred=group.preferred.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def validate_tables(summary, classes, class_ids, source_seed, repair_seed):
    expected = {(m, method) for m in MONTHS for method in METHODS}
    if len(summary) != len(expected) or summary.duplicated(['month', 'method']).any() or set(zip(summary.month, summary.method)) != expected:
        raise ValueError('Incomplete or duplicate monthly method rows')
    if not (summary.source_seed.eq(source_seed).all() and summary.repair_seed.eq(repair_seed).all()):
        raise ValueError('Summary seed identity mismatch')
    if classes.duplicated(['month', 'method', 'device_id']).any() or len(classes) != len(expected) * len(class_ids):
        raise ValueError('Incomplete or duplicate per-class rows')
    if not (classes.source_seed.eq(source_seed).all() and classes.repair_seed.eq(repair_seed).all()):
        raise ValueError('Per-class seed identity mismatch')
    if not summary.group_definition.eq('static_mlp_history_and_target_recall_v1').all():
        raise ValueError('Mixed diagnostic group definitions')
    baseline = classes[classes.method == 'static_mlp'].sort_values(['month', 'device_id'])
    for method in METHODS:
        rows = classes[classes.method == method].sort_values(['month', 'device_id'])
        for column in ['month', 'device_id', 'support', 'historical_support', 'historical_recall',
                       'before_recall', 'before_f1', 'collapse', 'high_recall_group']:
            a, b = rows[column].reset_index(drop=True), baseline[column].reset_index(drop=True)
            if not a.equals(b):
                raise ValueError('Methods do not share fixed static-MLP groups: ' + column)
    for row in summary.itertuples():
        subset = classes[(classes.month == row.month) & (classes.method == row.method)]
        if set(subset.device_id) != set(class_ids):
            raise ValueError('Per-class label universe mismatch')
        mask = truth(subset.collapse)
        if int(mask.sum()) != row.initial_collapse_count or int((mask & (subset.after_recall.to_numpy() < .1)).sum()) != row.remaining_collapse_count:
            raise ValueError('Collapse count inconsistent with per-class recalls')
        for field, selected in [('macro_f1_present', subset.support.to_numpy() > 0),
                                ('collapse_f1', mask), ('high_recall_group_f1', truth(subset.high_recall_group))]:
            expected_value = subset.after_f1.to_numpy()[selected].mean() if selected.any() else np.nan
            if not np.isclose(getattr(row, field), expected_value, atol=1e-8, rtol=1e-8, equal_nan=True):
                raise ValueError('Summary/per-class metric mismatch: ' + field)
    return baseline.reset_index(drop=True)


def selected_indices(reader, month, selector, pool, class_ids, budget):
    table = pd.read_csv(reader.file(f'{month}/{selector}_queries.csv'), dtype={'record_id': str})
    ids = [int(s) for s in table.record_id]
    lookup = {int(k): i for i, k in enumerate(pool['ids'])}
    if len(ids) != budget or len(set(ids)) != budget or not set(ids).issubset(lookup):
        raise ValueError('Query budget, uniqueness or membership mismatch')
    indices = np.array([lookup[k] for k in ids], dtype=np.int64)
    if not np.array_equal(pool['y'][indices], table.device_id.to_numpy()):
        raise ValueError('Query label differs from cached row')
    if not np.array_equal(truth(table.known_class), np.isin(pool['y'][indices], class_ids)):
        raise ValueError('Known-label flags inconsistent')
    return indices


def coverage_rows(month, seed, selector, pool, dates, indices, future, future_dates, devices, class_table):
    selected = np.zeros(len(pool['y']), dtype=bool)
    selected[indices] = True
    covers, daily, future_daily = [], [], []
    for device in devices:
        mask = pool['y'] == device
        paid = mask & selected
        fm = future['y'] == device
        available_days = set(dates[mask].day)
        queried_days = set(dates[paid].day)
        meta = dict(month=month, repair_seed=seed, selector=selector, device_id=device)
        row = dict(**meta, pool_rows=int(mask.sum()), queried_rows=int(paid.sum()),
                   label_budget=len(indices), expected_queries_under_uniform=len(indices) * mask.mean(),
                   selection_fraction=float(paid.sum() / mask.sum()) if mask.any() else np.nan,
                   available_days=len(available_days), queried_days=len(queried_days),
                   day_coverage=float(len(queried_days) / len(available_days)) if available_days else np.nan,
                   pool_rows_days_8_14=int((mask & (dates.day >= 8)).sum()),
                   queried_rows_days_8_14=int((paid & (dates.day >= 8)).sum()),
                   first_query_utc=dates[paid].min().isoformat() if paid.any() else '',
                   last_query_utc=dates[paid].max().isoformat() if paid.any() else '',
                   eval_rows=int(fm.sum()),
                   first_eval_utc=future_dates[fm].min().isoformat() if fm.any() else '',
                   last_eval_utc=future_dates[fm].max().isoformat() if fm.any() else '')
        row['gap_last_query_to_first_eval_days'] = ((future_dates[fm].min() - dates[paid].max()).total_seconds() / 86400
                                                   if paid.any() and fm.any() else np.nan)
        for variant in ['target', 'care']:
            found = class_table[(class_table.month == month) & (class_table.method == selector + '_' + variant)
                                & (class_table.device_id == device)]
            if len(found) != (0 if selector == 'margin' and variant == 'target' else 1):
                raise ValueError('Missing device result')
            for field in ['after_recall', 'after_f1']:
                row[variant + '_' + field] = found.iloc[0][field] if len(found) else np.nan
        covers.append(row)
        for day in range(1, 15):
            daily.append(dict(**meta, date=f'{month}-{day:02d}',
                              pool_rows=int((mask & (dates.day == day)).sum()),
                              queried_rows=int((paid & (dates.day == day)).sum())))
        # Future coverage is seed/selector independent; emit only once per device outside the caller.
        for day in range(16, pd.Timestamp(month + '-01').days_in_month + 1):
            future_daily.append(dict(month=month, device_id=device, date=f'{month}-{day:02d}',
                                     eval_rows=int((fm & (future_dates.day == day)).sum())))
    return covers, daily, future_daily


def analyze(root, source_seed=0, seeds=(0, 1, 2, 3, 4)):
    root = root.resolve()
    seeds = sorted(set(seeds))
    if not seeds or min(seeds) < 0 or source_seed < 0:
        raise ValueError('Require nonnegative source/repair seeds')
    cache, pilot = Reader(root / 'cache'), Reader(root / f'seed_{source_seed}')
    class_ids = json.loads(pilot.file('run_info.json').read_text())['classes']
    readers = {f'repair_{s}': Reader(root / 'paper_protocol_v1' / f'source_{source_seed}' / f'repair_{s}') for s in seeds}
    if pilot.marker['signature']['cache_sha256'] != sha(cache.root / 'complete.json'):
        raise ValueError('Pilot/cache identity mismatch')
    signatures = []
    for seed in seeds:
        sig = readers[f'repair_{seed}'].marker['signature']
        if sig['source_seed'] != source_seed or sig['repair_seed'] != seed or sig['source_marker'] != sha(pilot.root / 'complete.json') or sig['cache_marker'] != sha(cache.root / 'complete.json'):
            raise ValueError('Repair/source/cache identity mismatch')
        signatures.append({k: v for k, v in sig.items() if k not in ('repair_seed', 'queries')})
    if any(sig != signatures[0] for sig in signatures[1:]):
        raise ValueError('Cannot pair different protocols, source models or environments')
    all_readers = dict(cache=cache, pilot=pilot, **readers)
    signature = dict(schema=1, inputs={key: sha(r.root / 'complete.json') for key, r in all_readers.items()},
                     source_seed=source_seed, repair_seeds=seeds, cases=CASES, tolerance=TOLERANCE,
                     code={p.name: sha(p) for p in [Path(__file__), HERE.parent / 'tma_analysis_v1/analyze.py']},
                     analysis_versions=dict(numpy=np.__version__, pandas=pd.__version__))
    output = root / 'paired_analysis_v1' / f'source_{source_seed}' / ('repairs_' + '_'.join(map(str, seeds)))
    for path in [output, output.parent, output.parent.parent]:
        if path.is_symlink():
            raise ValueError('Output directories must not be symlinks')
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / 'config.json'
        if config.exists() and json.loads(config.read_text()) != signature:
            raise ValueError('Inputs/code changed; preserve this report and use a new analysis version')
        if not config.exists() and any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError('Unrecognized nonempty output directory')
        if (output / 'complete.json').exists():
            saved = Reader(output)
            if saved.marker['signature'] != signature:
                raise ValueError('Completed report signature changed')
            for name in saved.marker['artifacts']:
                saved.file(name)
            inputs = json.loads(saved.file('inputs.json').read_text())
            for role, files in inputs.items():
                for name, expected in files.items():
                    if all_readers[role].marker['artifacts'].get(name) != expected:
                        raise ValueError('Consumed input manifest changed')
                    all_readers[role].file(name)
            print((output / 'REPORT.txt').read_text(), flush=True)
            print('PAIRED ANALYSIS COMPLETE (verified existing report):', output, flush=True)
            return output
        write_json(config, signature)
        pairs, device_pairs, covers, daily, future_daily = [], [], [], [], []
        pools, futures, qdates, edates = {}, {}, {}, {}
        for month in MONTHS:
            pools[month] = cache.arrays(month + '_query.npz', ['ids', 'y', 'times'])
            futures[month] = cache.arrays(month + '_eval.npz', ['ids', 'y', 'times'])
            for part in [pools[month], futures[month]]:
                if len(np.unique(part['ids'])) != len(part['ids']) or not (len(part['ids']) == len(part['y']) == len(part['times'])):
                    raise ValueError('Duplicate/unaligned cached rows')
            if np.intersect1d(pools[month]['ids'], futures[month]['ids']).size:
                raise ValueError('Query/evaluation identity overlap')
            qdates[month] = timestamps(pools[month]['times'], month, True)
            edates[month] = timestamps(futures[month]['times'], month, False)
        fixed_baseline = None
        budget = pilot.marker['signature']['budget']
        for seed in seeds:
            print(f'Reading repair seed {seed}: metrics, query IDs and cached dates (no training)', flush=True)
            reader = readers[f'repair_{seed}']
            summary = pd.read_csv(reader.file('summary.csv'))
            classes = pd.read_csv(reader.file('per_class.csv'))
            baseline = validate_tables(summary, classes, class_ids, source_seed, seed)
            group_cols = ['month', 'device_id', 'support', 'historical_support', 'historical_recall',
                          'before_recall', 'before_f1', 'collapse', 'high_recall_group']
            if fixed_baseline is not None and not baseline[group_cols].equals(fixed_baseline):
                raise ValueError('Diagnostic groups differ between repair seeds')
            fixed_baseline = baseline[group_cols]
            for month in MONTHS:
                table = summary[summary.month == month].set_index('method')
                for c in class_ids:
                    expected_support = int((futures[month]['y'] == c).sum())
                    found = baseline[(baseline.month == month) & (baseline.device_id == c)].iloc[0]
                    if found.support != expected_support:
                        raise ValueError('Per-class support differs from evaluation cache')
                for selector in ['random', 'badge', 'margin']:
                    indices = selected_indices(reader, month, selector, pools[month], class_ids, budget)
                    known = int(np.isin(pools[month]['y'][indices], class_ids).sum())
                    variants = ['care'] if selector == 'margin' else ['target', 'care']
                    for variant in variants:
                        record = table.loc[selector + '_' + variant]
                        if record.label_queries != budget or record.usable_known_labels != known:
                            raise ValueError('Paired method query costs differ from saved selection')
                    if selector != 'margin':
                        care, target = [table.loc[selector + '_' + v] for v in ['care', 'target']]
                        meta = dict(source_seed=source_seed, repair_seed=seed, month=month, selector=selector)
                        for metric, (larger, multiplier, unit) in METRICS.items():
                            pairs.append(pair_row(meta, metric, care[metric], target[metric], larger, multiplier, unit))
                        for device in CASES.get(month, []):
                            subset = classes[(classes.month == month) & (classes.device_id == device)].set_index('method')
                            for metric in ['after_recall', 'after_f1']:
                                device_pairs.append(pair_row(dict(**meta, device_id=device), metric,
                                    subset.loc[selector + '_care', metric], subset.loc[selector + '_target', metric]))
                    if month in CASES:
                        a, b, c = coverage_rows(month, seed, selector, pools[month], qdates[month], indices,
                                               futures[month], edates[month], CASES[month], classes)
                        covers.extend(a); daily.extend(b)
                        if seed == seeds[0] and selector == 'random':
                            future_daily.extend(c)
        frames = dict(paired_by_seed=pd.DataFrame(pairs), device_pairs_by_seed=pd.DataFrame(device_pairs),
                      query_coverage_by_seed=pd.DataFrame(covers), query_dates=pd.DataFrame(daily),
                      eval_dates=pd.DataFrame(future_daily))
        frames['paired_wins'] = summarize_pairs(frames['paired_by_seed'], ['month', 'selector', 'metric'])
        frames['device_pair_wins'] = summarize_pairs(frames['device_pairs_by_seed'], ['month', 'selector', 'device_id', 'metric'])
        recovery = []
        for (month, selector, device), group in frames['query_coverage_by_seed'].groupby(['month', 'selector', 'device_id']):
            for variant in ['target', 'care']:
                values = group[variant + '_after_recall'].dropna()
                recovery.append(dict(month=month, selector=selector, device_id=device, variant=variant,
                                     n_valid=len(values), recall_mean=values.mean(), recall_sd=values.std(ddof=1),
                                     exits_collapse_ge_0_1=int((values >= .1).sum()),
                                     recovered_ge_0_3=int((values >= .3).sum()),
                                     failed_lt_0_05=int((values < .05).sum())))
        frames['device_recovery_counts'] = pd.DataFrame(recovery)
        for name, frame in frames.items():
            frame.to_csv(output / (name + '.csv'), index=False)
        wins = frames['paired_wins']
        key_metrics = ['macro_f1_present', 'collapse_f1', 'negative_flip_rate_vs_static_mlp',
                       'remaining_collapse_count', 'new_collapses_among_historically_learned']
        lines = [f'TMA 逐种子配对与查询覆盖；source={source_seed}，repair seeds={seeds}；未训练/未推理。',
            '差值始终为 CARE - target-only。F1/召回/NFR 用百分点；类数用类。',
            'win 指 CARE 更好：F1/召回越大越好，NFR/剩余崩溃/新增崩溃越小越好。',
            'sd_delta 是逐种子配对差的样本标准差，不是两个方法标准差相减。缺失组不计胜负或平局。',
            '\n各月配对差与胜负：', wins[wins.metric.isin(key_metrics)].to_string(index=False, float_format=lambda x: f'{x:.4f}'),
            '\n逐设备恢复次数：', frames['device_recovery_counts'].to_string(index=False, float_format=lambda x: f'{x:.4f}'),
            '\n逐种子查询覆盖与未来召回（均为缓存样本）：']
        coverage = frames['query_coverage_by_seed']
        for row in coverage.itertuples():
            dates = frames['query_dates']
            picked = dates[(dates.month == row.month) & (dates.repair_seed == row.repair_seed) &
                           (dates.selector == row.selector) & (dates.device_id == row.device_id) & (dates.queried_rows > 0)]
            date_text = ', '.join(f'{r.date}:{r.queried_rows}' for r in picked.itertuples()) or '无'
            lines.append(f'{row.month} device={row.device_id} seed={row.repair_seed} {row.selector}: '
                f'候选={row.pool_rows}, 查询={row.queried_rows}, 日期覆盖={row.queried_days}/{row.available_days}, '
                f'8--14日查询={row.queried_rows_days_8_14}; target召回={row.target_after_recall:.4f}, '
                f'care召回={row.care_after_recall:.4f}; 查询日期UTC={date_text}')
        lines += ['\n解释边界：',
            '同一选择器内 target/CARE 共用已保存查询；Margin 无 target 对照，只报告覆盖与恢复。',
            '查询池真实标签仅用于本次离线诊断；覆盖高低不能单独证明代表性或失败原因。',
            '日期计数是缓存样本，零行不等于设备离线；三月多个设备可能受同一环境变化影响。',
            '固定源模型、固定时间切分；五次修复不是五个独立环境。胜负计数不等于显著性检验。',
            'target-only 和 CARE 的更新次数可能不同，本报告不单独归因 CE/KD。',
            '所有月份已被查看；本次只整理既有证据，不是新的独立验证。']
        (output / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
        write_json(output / 'inputs.json', {key: reader.consumed for key, reader in all_readers.items()})
        artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('.lock', 'complete.json')}
        write_json(output / 'complete.json', dict(signature=signature, artifacts=artifacts))
        print((output / 'REPORT.txt').read_text(), flush=True)
        print('PAIRED ANALYSIS COMPLETE:', output, flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Run real-data analysis on the Linux server; local tests use synthetic fixtures only')
    try:
        seeds = sorted(set(int(s) for s in args.repair_seeds.split(',')))
        analyze(args.root, args.source_seed, seeds)
    except (ValueError, FileNotFoundError, BlockingIOError) as error:
        parser.exit(1, f'Analysis stopped: {error}\n')


if __name__ == '__main__':
    main()
