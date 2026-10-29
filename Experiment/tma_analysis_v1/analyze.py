"""Read-only post-hoc diagnosis of completed TMA pilots; no training or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ['2020-01', '2020-02', '2020-03', '2020-04']
METHODS = ['static_mlp', 'static_rf', 'random_target', 'random_replay', 'random_care',
           'margin_care', 'badge_target', 'badge_care']
PAIRS = [('random_target', 'static_mlp'), ('random_replay', 'random_target'),
         ('random_care', 'random_replay'), ('badge_care', 'badge_target'),
         ('badge_care', 'random_care'), ('margin_care', 'random_care')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Reader:
    def __init__(self, root):
        self.root = root
        self.marker = json.loads((root / 'complete.json').read_text())
        self.consumed = {}

    def file(self, name):
        expected = self.marker['artifacts'].get(name)
        path = self.root / name
        if expected is None or sha(path) != expected:
            raise ValueError('Missing or changed completed artifact: ' + str(path))
        self.consumed[name] = expected
        return path

    def arrays(self, name, keys):
        with np.load(self.file(name), allow_pickle=False) as f:
            return {k: f[k] for k in keys}


def confusion(labels, pred, classes):
    if len(labels) != len(pred) or not np.isin(pred, np.arange(classes)).all():
        raise ValueError('Invalid prediction shape/labels')
    keep = labels >= 0
    return np.bincount(labels[keep] * classes + pred[keep], minlength=classes ** 2).reshape(classes, classes)


def class_stats(cm):
    support = cm.sum(1)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.full(len(cm), np.nan), where=support > 0)
    denom = support + cm.sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(cm)), where=denom > 0)
    return support, recall, f1


def near(actual, expected, context):
    if not np.isclose(actual, expected, atol=1e-8, rtol=1e-8, equal_nan=True):
        raise ValueError(f'Metric mismatch {context}: {actual} vs {expected}')


def paired_deltas(summary):
    rows = []
    for month in MONTHS:
        table = summary[summary.month == month].set_index('method')
        for left, right in PAIRS:
            row = dict(month=month, comparison=left + ' minus ' + right)
            for metric in ['macro_f1_present', 'macro_recall_present', 'accuracy_known', 'negative_flip_rate']:
                row[metric + '_pp'] = 100 * float(table.loc[left, metric] - table.loc[right, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def query_counts(reader, cache, month, classes):
    pool = cache.arrays(month + '_query.npz', ['ids', 'y'])
    if len(np.unique(pool['ids'])) != len(pool['ids']):
        raise ValueError('Duplicate query pool IDs')
    lookup = dict(zip(pool['ids'].tolist(), pool['y'].tolist()))
    rows = []
    for selector in ['random', 'margin', 'badge']:
        selected = pd.read_csv(reader.file(month + '/' + selector + '_queries.csv'), dtype={'record_id': str})
        ids = [int(s) for s in selected.record_id]
        if len(set(ids)) != len(ids):
            raise ValueError('Repeated paid query')
        for i, label, flag in zip(ids, selected.device_id, selected.known_class):
            if i not in lookup or lookup[i] != label:
                raise ValueError('Query identity/label differs from cache')
            if str(flag).lower() != str(int(label) in classes).lower():
                raise ValueError('Incorrect known-class query flag')
        for c in sorted(set(pool['y'].tolist()) | set(classes)):
            n = int((pool['y'] == c).sum())
            count = int((selected.device_id == c).sum())
            rows.append(dict(month=month, selector=selector, device_id=int(c), known=c in classes,
                candidates=n, candidate_total=len(pool['ids']), queried=count, query_total=len(ids),
                expected_under_uniform_query=len(ids) * n / len(pool['ids']) if len(pool['ids']) else 0.))
    return rows


def analyze(root, seed=0):
    reader, cache = Reader(root / f'seed_{seed}'), Reader(root / 'cache')
    if reader.marker['signature']['cache_sha256'] != sha(root / 'cache/complete.json'):
        raise ValueError('Seed was generated using a different sample cache')
    info = json.loads(reader.file('run_info.json').read_text())
    classes = info['classes']
    mapping = {int(c): i for i, c in enumerate(classes)}
    summary = pd.read_csv(reader.file('summary.csv'))
    per_class = pd.read_csv(reader.file('per_class.csv'))
    if set(zip(summary.month, summary.method)) != {(m, s) for m in MONTHS for s in METHODS} or len(summary) != 32:
        raise ValueError('Incomplete or repeated month/method rows')
    if not (summary.seed == seed).all() or not (per_class.seed == seed).all():
        raise ValueError('Mixed seeds in tables')
    historical = reader.arrays('history_predictions.npz', ['labels', 'mlp', 'rf'])
    history_stats = {key: class_stats(confusion(historical['labels'], historical[key], len(classes)))
                     for key in ['mlp', 'rf']}
    diagnostics, queries, shifts = [], [], []
    shift_path = reader.file('conditional_shift.csv') if 'conditional_shift.csv' in reader.marker['artifacts'] else None
    shift_table = pd.read_csv(shift_path) if shift_path else pd.DataFrame()
    lines = [f'TMA 第一轮结果诊断，seed={seed}（仅离线统计）',
             f'训练期实际类别：{classes}',
             '检查范围：已消费文件校验和、测试预测/混淆矩阵/主要指标、查询身份和标签预算。',
             '\n各月各方法：所有分数为 0--1；RF 崩溃组以 RF 自身历史表现定义。']
    display = ['month', 'method', 'macro_f1_present', 'macro_recall_present', 'accuracy_known',
               'collapse_count', 'recovered_to_recall_0_1', 'stable_f1', 'negative_flip_rate']
    lines.append(summary[display].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    for month in MONTHS:
        qrows = query_counts(reader, cache, month, classes)
        queries.extend(qrows)
        arrays = reader.arrays(month + '/predictions.npz', ['ids', 'labels', 'device_ids'] + METHODS)
        target = cache.arrays(month + '_eval.npz', ['ids', 'y'])
        if not np.array_equal(arrays['ids'], target['ids']) or not np.array_equal(arrays['device_ids'], target['y']):
            raise ValueError('Evaluation rows differ from cache')
        truth = np.array([mapping.get(int(c), -1) for c in target['y']])
        if not np.array_equal(truth, arrays['labels']):
            raise ValueError('Evaluation class mapping differs from source vocabulary')
        for method in METHODS:
            key = 'rf' if method == 'static_rf' else 'mlp'
            hs, hr, _ = history_stats[key]
            cm = confusion(truth, arrays[method], len(classes))
            if not np.array_equal(cm, np.load(reader.file(month + '/' + method + '_confusion.npy'), allow_pickle=False)):
                raise ValueError('Saved confusion matrix disagrees with predictions')
            support, recall, f1 = class_stats(cm)
            baseline = arrays['static_' + key]
            _, before, before_f1 = class_stats(confusion(truth, baseline, len(classes)))
            learned = (hs >= 50) & (hr >= .8) & (support >= 50)
            collapse = learned & (before < .1)
            stable = learned & (before >= .8)
            row = summary[(summary.month == month) & (summary.method == method)].iloc[0]
            near(row.known_eval_rows, int((truth >= 0).sum()), method + ' known rows')
            near(row.unseen_eval_rows, int((truth < 0).sum()), method + ' unknown rows')
            near(row.macro_f1_present, f1[support > 0].mean(), method + ' F1')
            near(row.macro_recall_present, recall[support > 0].mean(), method + ' recall')
            near(row.accuracy_known, float(np.trace(cm) / cm.sum()) if cm.sum() else np.nan, method + ' accuracy')
            near(row.collapse_count, collapse.sum(), method + ' collapse count')
            near(row.recovered_to_recall_0_1, (collapse & (recall >= .1)).sum(), method + ' recovery count')
            near(row.recovered_to_recall_0_5, (collapse & (recall >= .5)).sum(), method + ' recovery .5')
            near(row.collapse_f1, f1[collapse].mean() if collapse.any() else np.nan, method + ' collapse F1')
            near(row.stable_f1, f1[stable].mean() if stable.any() else np.nan, method + ' stable F1')
            near(row.stable_delta_f1, (f1 - before_f1)[stable].mean() if stable.any() else np.nan, method + ' stable delta')
            near(row.new_collapses, (learned & (before >= .1) & (recall < .1)).sum(), method + ' new collapses')
            correct = (truth >= 0) & (baseline == truth)
            near(row.negative_flip_rate, float((arrays[method][correct] != truth[correct]).mean()) if correct.any() else np.nan,
                 method + ' negative flips')
            if not method.startswith('static_'):
                selected = [r for r in qrows if r['selector'] == method.split('_')[0]]
                near(row.label_queries, sum(r['queried'] for r in selected), method + ' paid queries')
                near(row.usable_known_labels, sum(r['queried'] for r in selected if r['known']), method + ' usable queries')
            rows = per_class[(per_class.month == month) & (per_class.method == method)].set_index('device_id')
            if len(rows) != len(classes) or set(rows.index) != set(classes):
                raise ValueError('Incomplete per-class table')
            for i, c in enumerate(classes):
                near(rows.loc[c, 'support'], support[i], method + ' class support')
                near(rows.loc[c, 'after_recall'], recall[i], method + ' class recall')
                near(rows.loc[c, 'historical_recall'], hr[i], method + ' historical recall')
                near(rows.loc[c, 'historical_support'], hs[i], method + ' historical support')
                near(rows.loc[c, 'before_recall'], before[i], method + ' before recall')
                near(rows.loc[c, 'after_f1'], f1[i] if support[i] else np.nan, method + ' class F1')
                if not collapse[i]:
                    continue
                wrong = [(classes[j], int(cm[i, j])) for j in np.argsort(-cm[i]) if j != i and cm[i, j] > 0][:3]
                diagnostics.append(dict(month=month, method=method, device_id=c, historical_support=int(hs[i]),
                    historical_recall=float(hr[i]), target_support=int(support[i]), before_recall=float(before[i]),
                    after_recall=float(recall[i]), after_f1=float(f1[i]),
                    top_wrong_devices='; '.join(f'{d}:{n}' for d, n in wrong)))
        collapsed_ids = [r['device_id'] for r in diagnostics if r['month'] == month and r['method'] == 'static_mlp']
        lines.append(f'\n{month} MLP 崩溃设备：{collapsed_ids}')
        if collapsed_ids:
            cols = ['selector', 'device_id', 'candidates', 'queried', 'expected_under_uniform_query']
            frame = pd.DataFrame(qrows)
            lines.append(frame[frame.device_id.isin(collapsed_ids)][cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
            if not shift_table.empty:
                for c in collapsed_ids:
                    subset = shift_table[(shift_table.month == month) & (shift_table.device_id == c)]
                    shifts.extend(subset.sort_values('ks_effect', ascending=False).head(3).to_dict('records'))
    deltas = paired_deltas(summary)
    lines += ['\n成对差值（左方法减右方法，单位：百分点）：',
              deltas.to_string(index=False, float_format=lambda x: f'{x:+.2f}'),
              '\n崩溃类详情（top_wrong_devices 为错误预测的设备 ID:条数）：']
    lines.append(pd.DataFrame(diagnostics).to_string(index=False, float_format=lambda x: f'{x:.4f}') if diagnostics else '无满足预设条件的崩溃类。')
    if shifts:
        lines += ['\n崩溃类最大的三个输入特征 KS 效应量（非因果或显著性结论）：',
                  pd.DataFrame(shifts).to_string(index=False, float_format=lambda x: f'{x:.3f}')]
    lines += ['\n解释边界：',
        '单种子是可行性结果；跨月同设备的崩溃不能当成独立设备证据。',
        'BADGE care-target 差异同时包含参考 CE 与 KD，不能单独归因于蒸馏。',
        'random care-replay 才在相同查询下单独比较增加 KD 的影响。',
        '查询候选中缺席、候选存在却没被查询、已有查询仍未恢复是不同问题。',
        '历史召回 >=.8 不保证历史 F1 高；稳定组按召回定义，须结合 F1 和误分类去向。',
        '宏指标随当月出现类别集合变化，不能把跨月总体分数差直接解释为漂移。',
        '输入分布变化和分类性能下降可提供线索，不能单独证明 P(Y|X) 改变或具体原因。',
        '本报告使用未来真标签事后诊断，不可将其当成在线采样信息或正式测试上的调参依据。']
    output = root / 'analysis_v1' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines) + '\n'
    (output / 'REPORT.txt').write_text(text)
    deltas.to_csv(output / 'paired_deltas.csv', index=False)
    pd.DataFrame(queries).to_csv(output / 'query_coverage.csv', index=False)
    pd.DataFrame(diagnostics, columns=['month', 'method', 'device_id', 'historical_support', 'historical_recall',
        'target_support', 'before_recall', 'after_recall', 'after_f1', 'top_wrong_devices']).to_csv(output / 'collapse_details.csv', index=False)
    (output / 'inputs.json').write_text(json.dumps(dict(seed=reader.consumed, cache=cache.consumed,
        seed_complete_sha256=sha(reader.root / 'complete.json'), cache_complete_sha256=sha(cache.root / 'complete.json'),
        analyzer_sha256=sha(Path(__file__))), indent=2) + '\n')
    print(text)
    print('诊断完成。请返回报告：', output / 'REPORT.txt')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    analyze(args.root, args.seed)
