"""Exploratory half-uniform, half-BADGE queries with adaptive CARE repair."""
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
sys.path.insert(0, str(EXPERIMENT / 'tma_adaptive_v1'))
from trial import (Reader, PROTOCOL, MONTHS, check_predictions, complete, context,
                   existing_output, fit_paper, forward_batches, load_references,
                   load_source, mapped, metrics, output_directory, sha, stats,
                   verify_used_inputs, write_csv, write_json)
from controls import hybrid_selection
from adaptive import drift_flags, fit_adaptive

RULE = dict(min_paid=5, error_probability=.9)


def identity():
    result = {p.name: sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}
    for folder, filename in [('tma_selective_repair_v1', 'trial.py'),
                             ('tma_adaptive_v1', 'adaptive.py'),
                             ('tma_controls_v1', 'controls.py')]:
        result[folder + '/' + filename] = sha(EXPERIMENT / folder / filename)
    return result


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'hybrid_adaptive_v1'
    source = base / f'source_{source_seed}'
    result = source / f'repair_{repair_seed}'
    if any(p.is_symlink() for p in (base, source, result)):
        raise ValueError('Output directories must not be symlinks')
    return result


def hybrid_indices(paper, month, query, features, logits, budget, seed):
    original = pd.read_csv(paper.file(month + '/random_queries.csv'),
                           dtype={'record_id': str}, usecols=['record_id'])
    lookup = {str(k): i for i, k in enumerate(query['ids'])}
    if len(lookup) != len(query['ids']) or len(original) != budget:
        raise ValueError('Random query identities/budget changed')
    try:
        random_indices = np.asarray([lookup[k] for k in original.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Saved random query absent from pool') from exc
    if len(np.unique(random_indices)) != budget:
        raise ValueError('Repeated original random query')
    selected = hybrid_selection(features, logits, random_indices, seed)
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise ValueError('Hybrid paid-query budget/uniqueness mismatch')
    return selected


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8, months=None):
    months = list(MONTHS if months is None else months)
    if not months or not set(months).issubset(MONTHS):
        raise ValueError('Unsupported month')
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
        protocol=PROTOCOL, device=device, threads=threads, months=months,
        selector='random_half_then_badge_on_remaining', rule=RULE)
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper)
        print('HYBRID ADAPTIVE COMPLETE (verified existing run):', output, flush=True)
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
    summaries, per_class, decisions, coverage = [], [], [], []
    for month in months:
        part = output / month
        part_signature = dict(signature, month=month)
        if existing_output(part, part_signature):
            verify_used_inputs(part, pilot=reader, cache=cache, paper=paper)
            print(month, 'verified completed month reused', flush=True)
        else:
            write_json(part / 'config.json', part_signature)
            query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
            qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
            qlogits = forward_batches(model.head, qfeat.numpy(), device)
            selected = hybrid_indices(paper, month, query, qfeat, qlogits, budget, repair_seed)
            paid_y = mapped(query['y'][selected], classes)
            source_q = qlogits.argmax(1).numpy()
            write_csv(part / 'hybrid_queries.csv', [dict(record_id=str(query['ids'][i]),
                device_id=int(query['y'][i]), known_class=bool(paid_y[j] >= 0))
                for j, i in enumerate(selected)])
            relax, rows = drift_flags(paid_y, source_q[selected], historical,
                                      len(classes), **RULE)
            write_json(part / 'decision.json', dict(month=month, classes=classes,
                paid_label_count=budget, usable_known_labels=int((paid_y >= 0).sum()),
                relax_preservation=relax.tolist(), rule=RULE, rows=rows))
            write_csv(part / 'decision_classes.csv', [dict(month=month,
                device_id=classes[row['class_index']], **row) for row in rows])
            print(month, 'hybrid paid', budget, 'known', int((paid_y >= 0).sum()),
                  'relaxed', [classes[i] for i in np.flatnonzero(relax)], flush=True)
            usable = paid_y >= 0
            target_x, target_y = qfeat[selected[usable]], paid_y[usable]
            heads, traces = {}, {}
            for method, care in [('hybrid_target', False), ('hybrid_care', True)]:
                head, trace = fit_paper(model.head, target_x, target_y, rfeat, ry,
                    teacher, care, repair_seed, device, PROTOCOL)
                heads[method], traces[method] = head, trace
                torch.save(head.state_dict(), part / (method + '_head.pt'))
                print(month, method, 'complete', flush=True)
            head, trace = fit_adaptive(model.head, target_x, target_y, rfeat, ry,
                                       teacher, relax, repair_seed, device, PROTOCOL)
            heads['hybrid_adaptive'], traces['hybrid_adaptive'] = head, trace
            torch.save(head.state_dict(), part / 'hybrid_adaptive_head.pt')
            write_json(part / 'training.json', traces)
            if not relax.any() and any(not torch.equal(head.state_dict()[k],
                    heads['hybrid_care'].state_dict()[k]) for k in head.state_dict()):
                raise ValueError('No-relaxation adaptive head differs from hybrid CARE')
            # Evaluation labels are opened only after query choice, decision,
            # and all three heads have been fixed and saved.
            future = paper.arrays(month + '/predictions.npz',
                                  ['ids', 'labels', 'device_ids', 'static_mlp'])
            target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
            y = check_predictions(future, target, classes)
            efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
            source_pred = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
            if not np.array_equal(source_pred, future['static_mlp']):
                raise ValueError('Reloaded source predictions differ from paper run')
            predictions = dict(ids=future['ids'], labels=y, device_ids=future['device_ids'],
                               static_mlp=source_pred)
            for method, head in heads.items():
                predictions[method] = forward_batches(head, efeat.numpy(), device).argmax(1).numpy()
            np.savez_compressed(part / 'predictions.npz', **predictions)
            rows_out, class_out = [], []
            for method in ['static_mlp', 'hybrid_target', 'hybrid_care', 'hybrid_adaptive']:
                score, class_rows, cm = metrics(y, source_pred,
                                                 predictions[method], historical, classes)
                meta = dict(source_seed=source_seed, repair_seed=repair_seed,
                    month=month, method=method,
                    label_queries=0 if method == 'static_mlp' else budget,
                    usable_known_labels=0 if method == 'static_mlp' else int(usable.sum()),
                    relaxed_class_count=int(relax.sum()))
                rows_out.append(dict(**meta, **score))
                class_out.extend(dict(**meta, **row) for row in class_rows)
                np.save(part / (method + '_confusion.npy'), cm)
            write_csv(part / 'summary.csv', rows_out)
            write_csv(part / 'per_class.csv', class_out)
            # Retrospective coverage uses unqueried pool labels only after
            # selection, decisions, heads, and evaluation are already fixed.
            write_csv(part / 'query_coverage.csv', [dict(month=month, device_id=c,
                candidate_pool=int((query['y'] == c).sum()),
                paid_true_class_count=int((query['y'][selected] == c).sum()))
                for c in classes])
            write_json(part / 'inputs.json', dict(pilot=reader.consumed,
                cache=cache.consumed, paper=paper.consumed))
            complete(part, part_signature)
        summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
        per_class.extend(pd.read_csv(part / 'per_class.csv').to_dict('records'))
        decisions.extend(pd.read_csv(part / 'decision_classes.csv').to_dict('records'))
        coverage.extend(pd.read_csv(part / 'query_coverage.csv').to_dict('records'))
        write_csv(output / 'summary.csv', summaries)
        write_csv(output / 'per_class.csv', per_class)
        write_csv(output / 'decision_classes.csv', decisions)
        write_csv(output / 'query_coverage.csv', coverage)
    frame = pd.DataFrame(summaries)
    decision_frame = pd.DataFrame(decisions)
    flagged = decision_frame[decision_frame.relax_preservation.astype(str).str.lower().eq('true')]
    focus = pd.DataFrame(per_class)
    focus = focus[((focus.month == '2020-03') & focus.device_id.isin([5, 22, 26, 37]))
                  | ((focus.month == '2020-04') & (focus.device_id == 20))]
    focus_coverage = pd.DataFrame(coverage)
    focus_coverage = focus_coverage[((focus_coverage.month == '2020-03') &
        focus_coverage.device_id.isin([5, 22, 26, 37])) |
        ((focus_coverage.month == '2020-04') & (focus_coverage.device_id == 20))]
    report = '\n'.join([
        'TMA half-random half-BADGE query + class-conditional CARE; exploratory on seen months.',
        'Within hybrid, target/CARE/adaptive share exactly the same 1000 paid query labels.',
        'Hybrid chooses 500 saved uniform queries, then 500 BADGE queries from the remaining pool.',
        'No future labels enter query selection, adaptive decisions, or head training.',
        'Relaxed classes:',
        flagged.groupby('month').device_id.apply(list).to_string() if len(flagged) else 'none',
        frame[['month', 'method', 'macro_f1_present', 'collapse_f1',
               'remaining_collapse_count', 'high_recall_group_f1',
               'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        'Focus-class hybrid query coverage:',
        focus_coverage.to_string(index=False),
        'Focus-class future results:',
        focus[['month', 'device_id', 'method', 'support', 'after_recall',
               'after_f1', 'predicted_count']].to_string(index=False),
        'Compare hybrid_adaptive against hybrid_care to isolate training with the same labels.',
        'Comparisons with original random/BADGE methods change query identities as well as training.',
        'All months have been examined; this is not an independent validation or novelty claim.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed,
        cache=cache.consumed, paper=paper.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('HYBRID ADAPTIVE COMPLETE:', output, flush=True)
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
            parser.error('Another hybrid-adaptive process is using this source output directory')
        run(root, args.source_seed, args.repair_seed, args.device, args.threads)


if __name__ == '__main__':
    main()
