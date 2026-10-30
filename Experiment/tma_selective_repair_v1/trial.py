"""Exploratory classwise CARE gate on completed paper-protocol TMA runs.

The gate is fitted from paid query labels only. Future evaluation predictions
and labels are opened after gate.json has been written for that month.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT / 'tma_paper_v1'))
from run import (PROTOCOL, check_predictions, complete, context, existing_output,
                 fit_paper, metrics, output_directory, verify_used_inputs)
from analyze import Reader
from controls import reference_data
from data import FEATURES, MONTHS, Preprocessor, sha, write_csv, write_json
from pilot import Model, choose_reference, forward_batches, mapped, stats
from gate import fold_assignments, learn_gate, route

FOLDS = 5
GATE_RULE = dict(min_support=10, min_discordant=3, probability_threshold=.9)


def code_identity():
    return {p.name: sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'selective_repair_v1'
    source = base / f'source_{source_seed}'
    result = source / f'repair_{repair_seed}'
    if any(p.is_symlink() for p in (base, source, result)):
        raise ValueError('Output directories must not be symlinks')
    return result


def load_source(reader, classes):
    checkpoint = torch.load(reader.file('source.pt'), map_location='cpu', weights_only=True)
    if checkpoint['classes'] != classes or checkpoint['input_features'] != FEATURES:
        raise ValueError('Source class/feature mapping changed')
    model = Model(2 * len(FEATURES), len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    prep = Preprocessor()
    with np.load(reader.file('preprocessor.npz'), allow_pickle=False) as f:
        prep.median, prep.mean, prep.std = f['median'], f['mean'], f['std']
    return model, prep


def load_references(reader, cache, paper, classes, source_seed, repair_seed):
    if repair_seed == source_seed:
        rx, ry = reference_data(reader, cache, classes)
        origins = pd.read_csv(reader.file('reference_ids.csv'), dtype={'record_id': str})
    else:
        train, ref = [cache.arrays(role + '.npz', ['ids', 'x', 'y'])
                      for role in ('train', 'reference')]
        samples, records = choose_reference(train, ref, classes, repair_seed)
        rx, ry = samples['x'], mapped(samples['y'], classes)
        origins = pd.DataFrame(records)
    saved = pd.read_csv(paper.file('reference_ids.csv'), dtype={'record_id': str})
    if not origins.reset_index(drop=True).equals(saved.reset_index(drop=True)):
        raise ValueError('Reconstructed references differ from paper-protocol run')
    if (ry < 0).any() or not np.all(np.bincount(ry, minlength=len(classes)) == 5):
        raise ValueError('Expected five historical references per known class')
    return rx, ry


def paid_query_indices(paper, month, query, classes, budget):
    table = pd.read_csv(paper.file(month + '/badge_queries.csv'), dtype={'record_id': str})
    lookup = {str(k): i for i, k in enumerate(query['ids'])}
    if len(lookup) != len(query['ids']) or len(table) != budget:
        raise ValueError('Query identities or paid-label budget changed')
    try:
        indices = np.asarray([lookup[k] for k in table.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Paid query is absent from query cache') from exc
    if len(np.unique(indices)) != budget:
        raise ValueError('Repeated paid query')
    paid_y = mapped(query['y'][indices], classes)
    flags = table.known_class.astype(str).str.lower().to_numpy()
    if not np.array_equal(query['y'][indices], table.device_id.to_numpy()) or not np.array_equal(
            paid_y >= 0, flags == 'true'):
        raise ValueError('Paid query label or known-class flag differs from paper run')
    return indices, paid_y


def fit_gate(model, query_features, source_query_pred, indices, paid_y,
             reference_features, reference_y, teacher, repair_seed, device):
    folds = fold_assignments(len(indices), FOLDS, 20261030 + repair_seed)
    oof_pred = np.full(len(indices), -1, dtype=np.int64)
    traces = []
    for fold in range(FOLDS):
        train = (folds != fold) & (paid_y >= 0)
        held = folds == fold
        head, trace = fit_paper(model.head, query_features[indices[train]], paid_y[train],
                                reference_features, reference_y, teacher, True,
                                repair_seed + 1000 + fold, device, PROTOCOL)
        oof_pred[held] = forward_batches(head, query_features[indices[held]].numpy(),
                                         device).argmax(1).numpy()
        traces.append(dict(fold=fold, paid_train=int((folds != fold).sum()),
                           known_train=int(train.sum()), held_out=int(held.sum()), **trace))
    if (oof_pred < 0).any():
        raise ValueError('Missing out-of-fold prediction')
    gate, rows = learn_gate(paid_y, source_query_pred[indices], oof_pred,
                            len(model.head.weight), **GATE_RULE)
    return gate, rows, folds, oof_pred, traces


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8):
    reader, cache, classes, _ = context(root, source_seed)
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    paper_signature = paper.marker['signature']
    if (paper_signature['source_seed'] != source_seed or paper_signature['repair_seed'] != repair_seed
            or paper_signature['protocol'] != PROTOCOL
            or paper_signature['source_marker'] != sha(reader.root / 'complete.json')
            or paper_signature['cache_marker'] != sha(cache.root / 'complete.json')):
        raise ValueError('Completed paper run does not match fixed source/cache/protocol')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
                     source_marker=sha(reader.root / 'complete.json'),
                     cache_marker=sha(cache.root / 'complete.json'),
                     paper_marker=sha(paper.root / 'complete.json'),
                     code=code_identity(), folds=FOLDS, gate_rule=GATE_RULE,
                     selector='badge', protocol=PROTOCOL, device=device, threads=threads)
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper)
        print('SELECTIVE REPAIR COMPLETE (verified existing run):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, prep = load_source(reader, classes)
    rx, ry = load_references(reader, cache, paper, classes, source_seed, repair_seed)
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    history = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(history['labels'], history['mlp'], len(classes))
    budget = reader.marker['signature']['budget']
    summaries, per_class, gates = [], [], []
    for month in MONTHS:
        part = output / month
        month_signature = dict(signature, month=month)
        if existing_output(part, month_signature):
            verify_used_inputs(part, pilot=reader, cache=cache, paper=paper)
            print(month, 'verified completed month reused', flush=True)
        else:
            write_json(part / 'config.json', month_signature)
            print(month, 'cross-fitting five CARE heads on paid BADGE labels', flush=True)
            # Query-pool y contains all labels, but only indexed paid labels are read.
            query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
            indices, paid_y = paid_query_indices(paper, month, query, classes, budget)
            qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
            source_q = forward_batches(model.head, qfeat.numpy(), device).argmax(1).numpy()
            gate, rows, folds, oof, traces = fit_gate(
                model, qfeat, source_q, indices, paid_y, rfeat, ry, teacher, repair_seed, device)
            write_json(part / 'gate.json', dict(month=month, classes=classes,
                       use_repair=gate.tolist(), rule=GATE_RULE,
                       paid_query_count=len(indices), usable_known_count=int((paid_y >= 0).sum()),
                       folds=FOLDS, rows=rows))
            np.savez_compressed(part / 'oof_predictions.npz',
                                paid_record_ids=query['ids'][indices], paid_labels=paid_y,
                                source_pred=source_q[indices], care_oof_pred=oof,
                                fold=folds)
            write_json(part / 'oof_training.json', traces)
            # Only now open future predictions and evaluation labels.
            frozen = json.loads((part / 'gate.json').read_text())
            gate = np.asarray(frozen['use_repair'], dtype=bool)
            future = paper.arrays(month + '/predictions.npz',
                                  ['ids', 'labels', 'device_ids', 'static_mlp', 'badge_care'])
            target = cache.arrays(month + '_eval.npz', ['ids', 'y'])
            y = check_predictions(future, target, classes)
            gated = route(future['static_mlp'], future['badge_care'], gate)
            np.savez_compressed(part / 'predictions.npz', ids=future['ids'], labels=y,
                                device_ids=future['device_ids'], static_mlp=future['static_mlp'],
                                badge_care=future['badge_care'], gated_care=gated)
            month_summary, month_classes = [], []
            for method, pred in [('static_mlp', future['static_mlp']),
                                 ('badge_care', future['badge_care']), ('gated_care', gated)]:
                score, class_rows, cm = metrics(y, future['static_mlp'], pred, historical, classes)
                meta = dict(source_seed=source_seed, repair_seed=repair_seed, month=month,
                            method=method, label_queries=0 if method == 'static_mlp' else budget,
                            usable_known_labels=0 if method == 'static_mlp' else int((paid_y >= 0).sum()))
                month_summary.append(dict(**meta, **score))
                month_classes.extend(dict(**meta, **row) for row in class_rows)
                np.save(part / (method + '_confusion.npy'), cm)
            write_csv(part / 'summary.csv', month_summary)
            write_csv(part / 'per_class.csv', month_classes)
            write_csv(part / 'gate_classes.csv', [dict(month=month, source_seed=source_seed,
                      repair_seed=repair_seed, device_id=classes[row['source_predicted_class']], **row)
                      for row in rows])
            write_json(part / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
                                                  paper=paper.consumed))
            complete(part, month_signature)
            print(month, 'complete; selected source-predicted classes =',
                  [classes[i] for i in np.flatnonzero(gate)], flush=True)
        summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
        per_class.extend(pd.read_csv(part / 'per_class.csv').to_dict('records'))
        gates.extend(pd.read_csv(part / 'gate_classes.csv').to_dict('records'))
        write_csv(output / 'summary.csv', summaries)
        write_csv(output / 'per_class.csv', per_class)
        write_csv(output / 'gate_classes.csv', gates)
    frame = pd.DataFrame(summaries)
    report = '\n'.join([
        'TMA 选择性 CARE 探索试验；固定源模型、BADGE 查询和原稿 CARE 头。',
        f'source_seed={source_seed}, repair_seed={repair_seed}; paid labels/month={budget}; OOF folds={FOLDS}',
        '每个源预测类别按已付费查询的折外净纠错证据决定是否启用 CARE；无额外标签。',
        'gate.json 在读取未来测试预测/标签之前写出；折外 CARE 仅用 80% 付费标签训练。',
        frame[['month', 'method', 'macro_f1_present', 'collapse_f1',
               'remaining_collapse_count', 'high_recall_group_f1',
               'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        '这是一轮已看过目标月份上的探索，不是独立时间盲测，也尚非新方法有效性结论。',
        'BADGE 付费样本并非随机总体，门控的类内净收益估计可能有选择偏差。',
        '源预测类别门控未必能修复源模型从不预测的崩溃类；留意 macro-F1 与负翻转的取舍。',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
                                          paper=paper.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('SELECTIVE REPAIR COMPLETE:', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Real experiments run on the Linux server; local tests use synthetic fixtures')
    if args.source_seed < 0 or args.repair_seed < 0 or args.threads < 1:
        parser.error('seeds must be nonnegative and threads positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; activate traffic-ncde')
    import fcntl
    root = args.root.resolve()
    context(root, args.source_seed)
    output = output_directory_for(root, args.source_seed, args.repair_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another selective-repair process is using this source output directory')
        run(root, args.source_seed, args.repair_seed, args.device, args.threads)


if __name__ == '__main__':
    main()
