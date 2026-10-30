"""Matched-step TMA ablation of reference CE and knowledge distillation.

Exploratory March/April comparison. All four arms have the CARE multiset,
shuffle stream, reference draws and optimizer-step count. Only the two loss
components vary; the full arm is checked against completed paper CARE heads.
"""
import argparse
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT / 'tma_selective_repair_v1'))
import trial
from trial import (Reader, PROTOCOL, check_predictions,
                   complete, context, existing_output, forward_batches,
                   load_references, load_source, mapped, metrics, output_directory,
                   sha, stats, verify_used_inputs, write_csv, write_json)
from engine import fit_variant

MONTHS = ['2020-03', '2020-04']
SELECTORS = ['badge', 'random']
ARMS = [('matched_target', False, False), ('replay_only', True, False),
        ('kd_only', False, True), ('matched_care', True, True)]


def identity():
    result = {p.name: sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}
    result['selective_trial.py'] = sha(EXPERIMENT / 'tma_selective_repair_v1/trial.py')
    return result


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'mechanism_v1'
    source = base / f'source_{source_seed}'
    result = source / f'repair_{repair_seed}'
    if any(p.is_symlink() for p in (base, source, result)):
        raise ValueError('Output directories must not be symlinks')
    return result


def paid_queries(paper, month, selector, query, classes, budget):
    table = pd.read_csv(paper.file(month + '/' + selector + '_queries.csv'),
                        dtype={'record_id': str})
    lookup = {str(k): i for i, k in enumerate(query['ids'])}
    if len(lookup) != len(query['ids']) or len(table) != budget:
        raise ValueError('Query identities or paid-label budget changed')
    try:
        indices = np.asarray([lookup[k] for k in table.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Paid query absent from query cache') from exc
    if len(np.unique(indices)) != budget:
        raise ValueError('Repeated paid query')
    labels = mapped(query['y'][indices], classes)
    flags = table.known_class.astype(str).str.lower().to_numpy()
    if not np.array_equal(query['y'][indices], table.device_id.to_numpy()) or not np.array_equal(
            labels >= 0, flags == 'true'):
        raise ValueError('Saved paid label differs from query cache')
    return indices, labels


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8,
        months=None, selectors=None):
    months = list(MONTHS if months is None else months)
    selectors = list(SELECTORS if selectors is None else selectors)
    if not months or not set(months).issubset(trial.MONTHS) or not selectors or not set(selectors).issubset(SELECTORS):
        raise ValueError('Unsupported months or selectors')
    reader, cache, classes, _ = context(root, source_seed)
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    ps = paper.marker['signature']
    if (ps['source_seed'] != source_seed or ps['repair_seed'] != repair_seed
            or ps['protocol'] != PROTOCOL or ps['device'] != device or ps['threads'] != threads
            or ps['source_marker'] != sha(reader.root / 'complete.json')
            or ps['cache_marker'] != sha(cache.root / 'complete.json')):
        raise ValueError('Paper run differs from requested source/cache/protocol/environment')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
                     source_marker=sha(reader.root / 'complete.json'),
                     cache_marker=sha(cache.root / 'complete.json'),
                     paper_marker=sha(paper.root / 'complete.json'),
                     code=identity(), protocol=PROTOCOL, device=device, threads=threads,
                     months=months, selectors=selectors, arms=[list(a) for a in ARMS])
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper)
        print('MATCHED ABLATION COMPLETE (verified existing run):', output, flush=True)
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
    summaries, per_class = [], []
    for month in months:
        query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
        qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
        for selector in selectors:
            part = output / month / selector
            part_signature = dict(signature, month=month, selector=selector)
            if existing_output(part, part_signature):
                verify_used_inputs(part, pilot=reader, cache=cache, paper=paper)
                print(month, selector, 'verified completed arm set reused', flush=True)
            else:
                write_json(part / 'config.json', part_signature)
                indices, labels = paid_queries(paper, month, selector, query, classes, budget)
                usable = labels >= 0
                print(month, selector, 'known paid labels', int(usable.sum()),
                      'matched CARE steps; training four arms', flush=True)
                heads, traces = {}, []
                for name, replay_ce, distill in ARMS:
                    head, trace = fit_variant(model.head, qfeat[indices[usable]], labels[usable],
                        rfeat, ry, teacher, replay_ce, distill, repair_seed, device, PROTOCOL)
                    heads[name] = head
                    traces.append(dict(method=name, **trace))
                    torch.save(head.state_dict(), part / (name + '_head.pt'))
                    print(month, selector, name, 'complete', flush=True)
                # The full arm must reproduce the original CARE artifact, not
                # merely appear similar by its evaluation score.
                saved = torch.load(paper.file(month + '/' + selector + '_care_head.pt'),
                                   map_location='cpu', weights_only=True)
                computed = heads['matched_care'].state_dict()
                if any(not torch.equal(computed[k], saved[k]) for k in saved):
                    raise ValueError('Matched full CARE head differs from paper artifact')
                write_json(part / 'training.json', traces)
                # Future evaluation is used only after all ablation heads are fixed.
                future = paper.arrays(month + '/predictions.npz',
                    ['ids', 'labels', 'device_ids', 'static_mlp',
                     selector + '_target', selector + '_care'])
                target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
                y = check_predictions(future, target, classes)
                efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
                source_pred = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
                if not np.array_equal(source_pred, future['static_mlp']):
                    raise ValueError('Reloaded source predictions differ from paper run')
                predictions = dict(ids=future['ids'], labels=y, device_ids=future['device_ids'],
                                   static_mlp=source_pred,
                                   paper_target=future[selector + '_target'],
                                   paper_care=future[selector + '_care'])
                for name, head in heads.items():
                    predictions[name] = forward_batches(head, efeat.numpy(), device).argmax(1).numpy()
                if not np.array_equal(predictions['matched_care'], predictions['paper_care']):
                    raise ValueError('Recomputed CARE predictions differ from paper run')
                np.savez_compressed(part / 'predictions.npz', **predictions)
                rows_out, class_out = [], []
                for method in ['static_mlp', 'paper_target', 'paper_care'] + [a[0] for a in ARMS]:
                    score, rows, cm = metrics(y, source_pred, predictions[method], historical, classes)
                    meta = dict(source_seed=source_seed, repair_seed=repair_seed,
                                month=month, selector=selector, method=method,
                                label_queries=0 if method == 'static_mlp' else budget,
                                usable_known_labels=0 if method == 'static_mlp' else int(usable.sum()))
                    rows_out.append(dict(**meta, **score))
                    class_out.extend(dict(**meta, **row) for row in rows)
                    np.save(part / (method + '_confusion.npy'), cm)
                write_csv(part / 'summary.csv', rows_out)
                write_csv(part / 'per_class.csv', class_out)
                write_json(part / 'inputs.json', dict(pilot=reader.consumed,
                              cache=cache.consumed, paper=paper.consumed))
                complete(part, part_signature)
            summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
            per_class.extend(pd.read_csv(part / 'per_class.csv').to_dict('records'))
            write_csv(output / 'summary.csv', summaries)
            write_csv(output / 'per_class.csv', per_class)
    frame = pd.DataFrame(summaries)
    focus = pd.DataFrame(per_class)
    focus = focus[((focus.month == '2020-03') & focus.device_id.isin([5, 22, 26, 37]))
                  | ((focus.month == '2020-04') & (focus.device_id == 20))]
    report = '\n'.join([
        'TMA March/April matched-step loss ablation; exploratory, not an independent test.',
        'All matched arms use CARE target-repeat + reference multiset, epoch count, batch stream and optimizer steps.',
        'Target CE contribution is unchanged when reference CE is masked; KD draws are consumed in every arm.',
        'matched_care state_dict and future predictions were checked against the completed paper artifact.',
        frame[['month', 'selector', 'method', 'macro_f1_present', 'collapse_f1',
               'remaining_collapse_count', 'high_recall_group_f1',
               'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        'Selected collapsed-device results (all methods; for mechanism diagnosis only):',
        focus[['month', 'selector', 'device_id', 'method', 'support',
               'after_recall', 'after_f1', 'predicted_count']].to_string(index=False),
        'The original paper_target has fewer optimizer steps; use matched_target for loss-component comparisons.',
        'Target months and devices were chosen after seeing results; do not use these comparisons as a blind evaluation.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed,
                                          cache=cache.consumed, paper=paper.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('MATCHED ABLATION COMPLETE:', output, flush=True)
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
        parser.error('Real experiments run on Linux; local tests use synthetic fixtures')
    if min(args.source_seed, args.repair_seed) < 0 or args.threads < 1:
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
            parser.error('Another matched-ablation process is using this source output directory')
        run(root, args.source_seed, args.repair_seed, args.device, args.threads)


if __name__ == '__main__':
    main()
