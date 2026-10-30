"""Exploratory class-conditional CARE preservation on the TMA time windows."""
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
sys.path.insert(0, str(EXPERIMENT / 'tma_mechanism_v1'))
from trial import (Reader, PROTOCOL, MONTHS, check_predictions, complete, context,
                   existing_output, forward_batches, load_references, load_source,
                   metrics, output_directory, sha, stats, verify_used_inputs,
                   write_csv, write_json)
from ablate import paid_queries
from adaptive import drift_flags, fit_adaptive

SELECTORS = ['random', 'badge']
RULE = dict(min_paid=5, error_probability=.9)


def identity():
    result = {p.name: sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}
    for folder, filename in [('tma_selective_repair_v1', 'trial.py'),
                             ('tma_mechanism_v1', 'ablate.py')]:
        result[folder + '/' + filename] = sha(EXPERIMENT / folder / filename)
    return result


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'adaptive_preservation_v1'
    source = base / f'source_{source_seed}'
    result = source / f'repair_{repair_seed}'
    if any(p.is_symlink() for p in (base, source, result)):
        raise ValueError('Output directories must not be symlinks')
    return result


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8,
        months=None, selectors=None):
    months = list(MONTHS if months is None else months)
    selectors = list(SELECTORS if selectors is None else selectors)
    if not months or not set(months).issubset(MONTHS) or not selectors or not set(selectors).issubset(SELECTORS):
        raise ValueError('Unsupported month or selector')
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
        paper_marker=sha(paper.root / 'complete.json'), code=identity(),
        protocol=PROTOCOL, device=device, threads=threads,
        months=months, selectors=selectors, rule=RULE)
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper)
        print('ADAPTIVE PRESERVATION COMPLETE (verified existing run):', output, flush=True)
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
    summaries, per_class, decisions = [], [], []
    for month in months:
        query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
        qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
        source_q = forward_batches(model.head, qfeat.numpy(), device).argmax(1).numpy()
        for selector in selectors:
            part = output / month / selector
            part_signature = dict(signature, month=month, selector=selector)
            if existing_output(part, part_signature):
                verify_used_inputs(part, pilot=reader, cache=cache, paper=paper)
                print(month, selector, 'verified completed arm reused', flush=True)
            else:
                write_json(part / 'config.json', part_signature)
                indices, labels = paid_queries(paper, month, selector, query, classes, budget)
                relax, rows = drift_flags(labels, source_q[indices], historical,
                                          len(classes), **RULE)
                write_json(part / 'decision.json', dict(month=month, selector=selector,
                    classes=classes, paid_label_count=budget,
                    usable_known_labels=int((labels >= 0).sum()),
                    relax_preservation=relax.tolist(), rule=RULE, rows=rows))
                write_csv(part / 'decision_classes.csv', [dict(month=month, selector=selector,
                    device_id=classes[row['class_index']], **row) for row in rows])
                print(month, selector, 'relaxed classes',
                      [classes[i] for i in np.flatnonzero(relax)], flush=True)
                usable = labels >= 0
                head, trace = fit_adaptive(model.head, qfeat[indices[usable]], labels[usable],
                    rfeat, ry, teacher, relax, repair_seed, device, PROTOCOL)
                torch.save(head.state_dict(), part / 'adaptive_head.pt')
                write_json(part / 'training.json', trace)
                reloaded = type(model.head)(model.head.in_features, model.head.out_features)
                reloaded.load_state_dict(torch.load(part / 'adaptive_head.pt',
                                                    map_location='cpu', weights_only=True))
                # Future labels and predictions are opened only after the
                # paid-label decision and adaptive head are saved.
                future = paper.arrays(month + '/predictions.npz',
                    ['ids', 'labels', 'device_ids', 'static_mlp',
                     selector + '_target', selector + '_care'])
                target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
                y = check_predictions(future, target, classes)
                efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
                source_pred = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
                if not np.array_equal(source_pred, future['static_mlp']):
                    raise ValueError('Reloaded source predictions differ from paper run')
                adapted = forward_batches(reloaded, efeat.numpy(), device).argmax(1).numpy()
                predictions = dict(ids=future['ids'], labels=y, device_ids=future['device_ids'],
                    static_mlp=source_pred, paper_target=future[selector + '_target'],
                    paper_care=future[selector + '_care'], adaptive=adapted)
                if not relax.any() and not np.array_equal(adapted, predictions['paper_care']):
                    raise ValueError('No-relaxation arm differs from original CARE')
                np.savez_compressed(part / 'predictions.npz', **predictions)
                rows_out, class_out = [], []
                for method in ['static_mlp', 'paper_target', 'paper_care', 'adaptive']:
                    score, class_rows, cm = metrics(y, source_pred,
                                                     predictions[method], historical, classes)
                    meta = dict(source_seed=source_seed, repair_seed=repair_seed,
                        month=month, selector=selector, method=method,
                        label_queries=0 if method == 'static_mlp' else budget,
                        usable_known_labels=0 if method == 'static_mlp' else int(usable.sum()),
                        relaxed_class_count=int(relax.sum()))
                    rows_out.append(dict(**meta, **score))
                    class_out.extend(dict(**meta, **row) for row in class_rows)
                    np.save(part / (method + '_confusion.npy'), cm)
                write_csv(part / 'summary.csv', rows_out)
                write_csv(part / 'per_class.csv', class_out)
                write_json(part / 'inputs.json', dict(pilot=reader.consumed,
                    cache=cache.consumed, paper=paper.consumed))
                complete(part, part_signature)
            summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
            per_class.extend(pd.read_csv(part / 'per_class.csv').to_dict('records'))
            decisions.extend(pd.read_csv(part / 'decision_classes.csv').to_dict('records'))
            write_csv(output / 'summary.csv', summaries)
            write_csv(output / 'per_class.csv', per_class)
            write_csv(output / 'decision_classes.csv', decisions)
    frame = pd.DataFrame(summaries)
    decision_frame = pd.DataFrame(decisions)
    selected = decision_frame[decision_frame.relax_preservation.astype(str).str.lower().eq('true')]
    focus = pd.DataFrame(per_class)
    focus = focus[((focus.month == '2020-03') & focus.device_id.isin([5, 22, 26, 37]))
                  | ((focus.month == '2020-04') & (focus.device_id == 20))]
    report = '\n'.join([
        'TMA paid-label class-conditional CARE preservation; exploratory, all four future months reported.',
        'Random/BADGE labels are reused within each selector; no added labeling cost.',
        'A historically learned class is relaxed if >=5 paid true-class samples and posterior P(source error>.5)>.9.',
        'For relaxed classes, historical reference CE is masked and KD operates only over unrelaxed logits.',
        'Decision and head are saved before future evaluation labels/predictions are opened.',
        'Flagged classes by month/selector:',
        selected.groupby(['month', 'selector']).device_id.apply(list).to_string() if len(selected) else 'none',
        frame[['month', 'selector', 'method', 'macro_f1_present', 'collapse_f1',
               'remaining_collapse_count', 'high_recall_group_f1',
               'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        'Selected collapse-class results:',
        focus[['month', 'selector', 'device_id', 'method', 'support',
               'after_recall', 'after_f1', 'predicted_count']].to_string(index=False),
        'All months have already been examined; this is exploratory method development, not a blind evaluation.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed,
        cache=cache.consumed, paper=paper.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('ADAPTIVE PRESERVATION COMPLETE:', output, flush=True)
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
            parser.error('Another adaptive process is using this source output directory')
        run(root, args.source_seed, args.repair_seed, args.device, args.threads)


if __name__ == '__main__':
    main()
