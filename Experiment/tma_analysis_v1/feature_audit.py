"""Post-hoc audit of cached feature collisions, time coverage and queried patterns."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from analyze import Reader, MONTHS, sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tma_pilot_v1'))
from data import FEATURES, Preprocessor


def keys(x):
    """Exact float32 feature equivalence, with canonical NaNs and signed zeros."""
    missing = ~np.isfinite(x)
    values = np.where(missing, 0, x).astype(np.float32)
    values[values == 0] = 0.
    canonical = np.ascontiguousarray(np.concatenate([values, missing.astype(np.float32)], axis=1))
    return canonical.view(np.dtype((np.void, canonical.shape[1] * canonical.dtype.itemsize))).ravel()


def collision_stats(x, y):
    k = keys(x)
    _, group = np.unique(k, return_inverse=True)
    if not len(group):
        return dict(rows=0, unique_vectors=0, conflicting_rows=0, empirical_accuracy_ceiling=None), np.array([], bool)
    stride = int(y.max()) + 1
    pairs, counts = np.unique(group * stride + y, return_counts=True)
    pair_group = pairs // stride
    majority = np.zeros(int(group.max()) + 1, dtype=np.int64)
    np.maximum.at(majority, pair_group, counts)
    mixed_groups = np.bincount(pair_group, minlength=len(majority)) > 1
    mixed = mixed_groups[group]
    return dict(rows=len(y), unique_vectors=len(majority), conflicting_rows=int(mixed.sum()),
                empirical_accuracy_ceiling=float(majority.sum() / len(y))), mixed


def timestamps(values):
    # pandas versions can store datetime int64 in ns or us. Infer from the
    # experiment's fixed calendar window, and require an unambiguous answer.
    matches = []
    for unit in ['ns', 'us', 'ms', 's']:
        dt = pd.to_datetime(values, unit=unit, utc=True, errors='coerce')
        if len(dt) and ((dt >= '2019-11-01') & (dt < '2020-05-01')).all():
            matches.append((unit, dt))
    if len(matches) != 1:
        raise ValueError('Ambiguous/out-of-window cached timestamp units')
    return matches[0]


def finite_summary(values):
    finite = values[np.isfinite(values)]
    return dict(missing=float((~np.isfinite(values)).mean()),
                min=float(finite.min()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None)


def load_role(cache, role, prep, classes):
    part = cache.arrays(role + '.npz', ['ids', 'y', 'x', 'times'])
    if part['x'].shape != (len(part['y']), len(FEATURES)):
        raise ValueError('Unexpected feature shape')
    part['transformed'] = prep.transform(part['x'])
    part['raw_keys'], part['transformed_keys'] = keys(part['x']), keys(part['transformed'])
    part['time_unit'], part['dt'] = timestamps(part['times'])
    known = np.isin(part['y'], classes)
    bounds = []
    for representation, value in [('raw16', part['x']), ('transformed32', part['transformed'])]:
        summary, conflict = collision_stats(value[known], part['y'][known])
        mask = np.zeros(len(known), dtype=bool); mask[known] = conflict
        part[representation + '_conflicts'] = mask
        bounds.append(dict(role=role, representation=representation, **summary))
    return part, bounds


def audit(root, seed=0):
    reader, cache = Reader(root / f'seed_{seed}'), Reader(root / 'cache')
    if reader.marker['signature']['cache_sha256'] != sha(root / 'cache/complete.json'):
        raise ValueError('Seed and cache do not match')
    config = json.loads((root / 'cache/config.json').read_text())
    if config != cache.marker['signature'] or config['features'] != FEATURES:
        raise ValueError('Cache feature order/configuration changed')
    classes = json.loads(reader.file('run_info.json').read_text())['classes']
    rows = pd.read_csv(reader.file('per_class.csv'))
    failed = rows[(rows.method == 'static_mlp') & rows.collapse.astype(str).str.lower().eq('true')]
    cases = sorted(set(zip(failed.month, failed.device_id.astype(int))))
    prep = Preprocessor()
    with np.load(reader.file('preprocessor.npz'), allow_pickle=False) as f:
        prep.median, prep.mean, prep.std = f['median'], f['mean'], f['std']
    history, bounds = load_role(cache, 'history', prep, classes)
    profiles, feature_rows, selected_rows, dominant_rows = [], [], [], []
    lines = [f'TMA 缓存特征诊断 seed={seed}（离线使用真实标签，不训练）',
             'raw16 指缓存中的 16 个统计特征，不是原始网络流的所有字段。',
             '相同向量按 float32 精确比较，缺失值与正负零统一；不使用近似聚类。']
    for month in MONTHS:
        target, extra = load_role(cache, month + '_eval', prep, classes); bounds.extend(extra)
        devices = [c for m, c in cases if m == month]
        if not devices:
            continue
        query, extra = load_role(cache, month + '_query', prep, classes); bounds.extend(extra)
        lookup = {int(k): i for i, k in enumerate(query['ids'])}
        selected = {}
        for selector in ['random', 'margin', 'badge']:
            table = pd.read_csv(reader.file(month + '/' + selector + '_queries.csv'), dtype={'record_id': str})
            if table.record_id.duplicated().any():
                raise ValueError('Duplicate paid query')
            indices = np.array([lookup[int(k)] for k in table.record_id], dtype=np.int64)
            if not np.array_equal(query['y'][indices], table.device_id.to_numpy()):
                raise ValueError('Query label differs from cache')
            selected[selector] = indices
        for c in devices:
            lines.append(f'\n{month} 设备 {c}')
            e = target['y'] == c
            q = query['y'] == c
            for role, part in [('history', history), ('query', query), ('eval', target)]:
                mask = part['y'] == c
                if not mask.any():
                    lines.append(role + ': 没有该设备样本')
                    continue
                unique, counts = np.unique(part['raw_keys'][mask], return_counts=True)
                daily = pd.Series(part['dt'][mask].strftime('%Y-%m-%d')).value_counts().sort_index().to_dict()
                profile = dict(month=month, device_id=c, role=role, rows=int(mask.sum()),
                    unique_raw16=len(unique), dominant_raw16_fraction=float(counts.max() / mask.sum()),
                    unique_transformed32=len(np.unique(part['transformed_keys'][mask])),
                    conflicting_raw16_rows=int(part['raw16_conflicts'][mask].sum()),
                    conflicting_transformed32_rows=int(part['transformed32_conflicts'][mask].sum()),
                    first=str(part['dt'][mask].min()), last=str(part['dt'][mask].max()),
                    daily_sample_counts=json.dumps({k: int(v) for k, v in daily.items()}))
                profiles.append(profile)
                lines.append(json.dumps(profile, ensure_ascii=False))
                for j, name in enumerate(FEATURES):
                    feature_rows.append(dict(month=month, device_id=c, role=role, feature=name,
                                             **finite_summary(part['x'][mask, j])))
            for selector, indices in selected.items():
                own = indices[query['y'][indices] == c]
                record = dict(month=month, device_id=c, selector=selector, queried=len(own),
                    unique_queried_raw16=len(np.unique(query['raw_keys'][own])),
                    target_exact_match_in_same_class_pool=float(np.isin(target['raw_keys'][e], query['raw_keys'][q]).mean()),
                    target_exact_match_in_same_class_queries=float(np.isin(target['raw_keys'][e], query['raw_keys'][own]).mean()))
                selected_rows.append(record)
                lines.append(json.dumps(record, ensure_ascii=False))
            unique, counts = np.unique(target['raw_keys'][e], return_counts=True)
            # Show the largest pattern, including other device labels carrying it.
            mode_key = unique[np.argmax(counts)]
            matching = target['raw_keys'] == mode_key
            labels, n = np.unique(target['y'][matching], return_counts=True)
            vector = target['x'][np.flatnonzero(matching)[0]]
            record = dict(month=month, device_id=c, target_mode_fraction=float(counts.max() / e.sum()),
                labels_at_same_raw16_vector={int(label): int(count) for label, count in zip(labels, n)},
                vector={name: float(v) if np.isfinite(v) else None for name, v in zip(FEATURES, vector)})
            dominant_rows.append(record)
            lines.append('目标期最常见特征向量：' + json.dumps(record, ensure_ascii=False))
    lines += ['\n已知类样本中的特征冲突统计：', pd.DataFrame(bounds).to_string(index=False),
              '\n解释边界：',
              'empirical_accuracy_ceiling 是在这批样本上利用真实标签计算的同向量多数类上限，不是可部署模型的准确率。',
              '不同设备若具有完全相同的输入，任何只使用这些输入的确定性分类器都无法逐条区分它们。',
              'raw16 冲突只说明本轮选用特征不足，不能推断完整流量不可区分；端口和域名等未进入本轮输入。',
              'raw16 可区分但 transformed32 合并提示检查标准化截断；两者均可区分仍不能排除编码器或标签不足问题。',
              '精确匹配率低不等于不可泛化，也不能据此认定缺少代表性样本；连续特征本来就可能不重复。',
              '日期计数来自抽样缓存，不是该日全量流量，也不能证明设备离线原因或固件更新。',
              '当前已查看所有测试月份，后续方法改动属于探索，正式有效性需额外独立时间段/数据验证。']
    out = root / 'feature_audit_v1' / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(lines) + '\n'
    (out / 'REPORT.txt').write_text(report)
    for name, records in [('profiles', profiles), ('feature_ranges', feature_rows), ('queried_patterns', selected_rows), ('collision_bounds', bounds)]:
        pd.DataFrame(records).to_csv(out / (name + '.csv'), index=False)
    (out / 'dominant_vectors.json').write_text(json.dumps(dominant_rows, ensure_ascii=False, indent=2) + '\n')
    (out / 'inputs.json').write_text(json.dumps(dict(seed=reader.consumed, cache=cache.consumed,
        seed_complete_sha256=sha(reader.root / 'complete.json'), cache_complete_sha256=sha(cache.root / 'complete.json'),
        script_sha256=sha(Path(__file__)), helper_sha256=sha(Path(__file__).with_name('analyze.py'))), indent=2) + '\n')
    print(report)
    print('请返回报告：', out / 'REPORT.txt')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    audit(args.root, args.seed)
