"""Exploratory quota-limited victim retrieval followed by BADGE and CARE.

The first 500 paid labels are the fixed uniform anchor. New labels enter only
after their query is chosen. A per-class true-label quota stops retrieval, so
the remaining budget stays with BADGE. Future evaluation data are opened only
after final selections, heads, and decisions have been saved.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ('tma_victim_retrieval_v1', 'tma_selective_repair_v1',
               'tma_paper_v1', 'tma_analysis_v1', 'tma_pilot_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader
from data import MONTHS, sha, write_csv, write_json
from pilot import forward_batches, mapped, select, stats
from retrieve import candidate_queues, nominate_classes, normalized_features
from run import (PROTOCOL, check_predictions, complete, context, existing_output,
                 fit_paper, metrics, output_directory, verify_used_inputs)
from trial import load_references, load_source

UNIFORM_ANCHOR = 500
LABEL_QUOTA_PER_CLASS = 15
MAX_FOCUS_CLASSES = 10
MAX_ATTEMPTS_PER_CLASS = 25
MAX_TARGETED_QUERIES = 150
METHODS = ('quota_target', 'quota_care')


def identity():
    files = [p for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    files += [EXPERIMENT / 'tma_victim_retrieval_v1/retrieve.py',
              EXPERIMENT / 'tma_selective_repair_v1/trial.py',
              EXPERIMENT / 'tma_paper_v1/run.py']
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in files}


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'quota_query_v1'
    source = base / f'source_{source_seed}'
    result = source / f'repair_{repair_seed}'
    if any(path.is_symlink() for path in (base, source, result)):
        raise ValueError('Output directories must not be symlinks')
    return result


def paid_indices(reader, name, query):
    table = pd.read_csv(reader.file(name), dtype={'record_id': str},
                        usecols=['record_id'])
    lookup = {str(key): i for i, key in enumerate(query['ids'])}
    if len(lookup) != len(query['ids']):
        raise ValueError('Duplicate candidate-pool IDs')
    try:
        indices = np.asarray([lookup[key] for key in table.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Saved paid query absent from candidate pool') from exc
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Duplicate saved paid query')
    return indices


def choose_focus(anchor_y, anchor_pred, pool_pred, classes):
    all_eligible, rows = nominate_classes(anchor_y, anchor_pred, pool_pred,
                                           classes, max_focus=len(classes))
    rare = [row for row in all_eligible
            if row['anchor_support'] < LABEL_QUOTA_PER_CLASS]
    focus = rare[:MAX_FOCUS_CLASSES]
    focus_indices = {row['class_index'] for row in focus}
    for row in rows:
        row['selected_for_quota'] = row['class_index'] in focus_indices
    return focus, rows


def quota_select(queues, anchor, pool_labels, classes, feature_tensor,
                 source_logits, seed, total_budget=1000):
    """Simulate sequentially purchased labels; never inspect unqueried labels."""
    n_pool = len(pool_labels)
    if len(anchor) != UNIFORM_ANCHOR or total_budget != 1000:
        raise ValueError('This fixed pilot requires 500 + 500 labels')
    used = np.zeros(n_pool, dtype=bool)
    used[anchor] = True
    anchor_y = mapped(pool_labels[anchor], classes)
    count = np.bincount(anchor_y[anchor_y >= 0], minlength=len(classes))
    attempts = {c: 0 for c in queues}
    found = {c: 0 for c in queues}
    pointers = {c: 0 for c in queues}
    targeted, log = [], []
    while len(targeted) < MAX_TARGETED_QUERIES:
        progress = False
        for c, queue in queues.items():
            if (count[c] >= LABEL_QUOTA_PER_CLASS or
                    attempts[c] >= MAX_ATTEMPTS_PER_CLASS or
                    len(targeted) >= MAX_TARGETED_QUERIES):
                continue
            while pointers[c] < len(queue) and used[queue[pointers[c]]]:
                pointers[c] += 1
            if pointers[c] == len(queue):
                continue
            index = int(queue[pointers[c]])
            pointers[c] += 1
            used[index] = True
            targeted.append(index)
            attempts[c] += 1
            # The label is revealed after this instance is selected and paid.
            true_class = int(mapped(pool_labels[index:index + 1], classes)[0])
            if true_class >= 0:
                count[true_class] += 1
            if true_class == c:
                found[c] += 1
            log.append(dict(query_order=len(anchor) + len(targeted) - 1,
                            nominated_device_id=int(classes[c]),
                            record_index=index, paid_device_id=int(pool_labels[index]),
                            hit=bool(true_class == c),
                            cumulative_nominated_labels=int(count[c])))
            progress = True
        if not progress:
            break
    available = np.flatnonzero(~used)
    badge_budget = total_budget - len(anchor) - len(targeted)
    if badge_budget < 0 or len(available) < badge_budget:
        raise ValueError('Invalid remaining BADGE budget')
    badge_local = select('badge', feature_tensor[available],
                         source_logits[available], badge_budget, seed)
    badge = available[badge_local]
    selected = np.r_[anchor, np.asarray(targeted, dtype=np.int64), badge]
    if len(selected) != total_budget or len(np.unique(selected)) != total_budget:
        raise ValueError('Quota query budget or uniqueness failed')
    info = dict(targeted_queries=len(targeted), badge_queries=len(badge),
                attempts_per_focus={str(classes[c]): int(attempts[c]) for c in queues},
                true_hits_per_focus={str(classes[c]): int(found[c]) for c in queues},
                paid_log=log)
    return selected, info


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8):
    reader, cache, classes, _ = context(root, source_seed)
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                    f'repair_{repair_seed}')
    for name, bundle in (('paper', paper), ('hybrid', hybrid)):
        info = bundle.marker['signature']
        if (info['source_seed'] != source_seed or info['repair_seed'] != repair_seed or
                info['source_marker'] != sha(reader.root / 'complete.json') or
                info['cache_marker'] != sha(cache.root / 'complete.json') or
                info['protocol'] != PROTOCOL):
            raise ValueError(name + ' completed run differs from fixed source/cache/protocol')
    if hybrid.marker['signature']['paper_marker'] != sha(paper.root / 'complete.json'):
        raise ValueError('Hybrid and paper runs are not paired')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        protocol=PROTOCOL, code=identity(), device=device, threads=threads,
        uniform_anchor=UNIFORM_ANCHOR, label_quota_per_class=LABEL_QUOTA_PER_CLASS,
        max_focus_classes=MAX_FOCUS_CLASSES,
        max_attempts_per_class=MAX_ATTEMPTS_PER_CLASS,
        max_targeted_queries=MAX_TARGETED_QUERIES)
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper,
                           hybrid=hybrid)
        print('QUOTA QUERY COMPLETE (verified existing run):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, prep = load_source(reader, classes)
    rx, ry = load_references(reader, cache, paper, classes, source_seed, repair_seed)
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    h = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(h['labels'], h['mlp'], len(classes))
    summaries, coverage, focuses, query_stats = [], [], [], []
    for month in MONTHS:
        part = output / month
        month_signature = dict(signature, month=month)
        if existing_output(part, month_signature):
            verify_used_inputs(part, pilot=reader, cache=cache, paper=paper,
                               hybrid=hybrid)
            print(month, 'verified completed month reused', flush=True)
        else:
            write_json(part / 'config.json', month_signature)
            query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
            random_all = paid_indices(paper, month + '/random_queries.csv', query)
            hybrid_all = paid_indices(hybrid, month + '/hybrid_queries.csv', query)
            anchor = random_all[:UNIFORM_ANCHOR]
            if (len(random_all) != 1000 or len(hybrid_all) != 1000 or
                    not np.array_equal(hybrid_all[:UNIFORM_ANCHOR], anchor)):
                raise ValueError('Fixed baseline query budgets or prefix changed')
            qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
            qlogits = forward_batches(model.head, qfeat.numpy(), device)
            pool_pred = qlogits.argmax(1).numpy()
            anchor_y = mapped(query['y'][anchor], classes)
            focus, nomination = choose_focus(anchor_y, pool_pred[anchor],
                                              pool_pred, classes)
            z = normalized_features(qfeat.numpy())
            queues = candidate_queues(z, pool_pred, anchor, anchor_y,
                                      focus, 'nearest', repair_seed)
            selected, info = quota_select(queues, anchor, query['y'], classes,
                                          qfeat, qlogits, repair_seed)
            if info['targeted_queries'] == 0 and not np.array_equal(
                    selected, hybrid_all):
                raise ValueError('Zero-target query differs from fixed hybrid baseline')
            # Complete query identities and all paid labels are saved before
            # any future evaluation prediction or label is opened.
            write_json(part / 'selection.json', dict(month=month, classes=classes,
                focus=focus, nomination=nomination,
                query_record_ids=[str(query['ids'][i]) for i in selected],
                paid_device_ids=[int(query['y'][i]) for i in selected],
                **info))
            frozen = json.loads((part / 'selection.json').read_text())
            if (frozen['query_record_ids'] !=
                    [str(query['ids'][i]) for i in selected] or
                    frozen['paid_device_ids'] !=
                    [int(query['y'][i]) for i in selected]):
                raise ValueError('Saved paid-query selection changed')
            print(month, 'quota queries', info['targeted_queries'],
                  'BADGE queries', info['badge_queries'],
                  'focus', [r['device_id'] for r in focus], flush=True)
            paid_y = mapped(query['y'][selected], classes)
            usable = paid_y >= 0
            heads = {}
            traces = {}
            for method, care in (('quota_target', False), ('quota_care', True)):
                head, trace = fit_paper(model.head, qfeat[selected[usable]],
                    paid_y[usable], rfeat, ry, teacher, care, repair_seed,
                    device, PROTOCOL)
                heads[method], traces[method] = head, trace
                torch.save(head.state_dict(), part / (method + '_head.pt'))
                print(month, method, 'complete', flush=True)
            write_json(part / 'training.json', traces)
            # Query and trained heads are fixed before future evaluation.
            future = hybrid.arrays(month + '/predictions.npz',
                ['ids', 'labels', 'device_ids', 'static_mlp',
                 'hybrid_target', 'hybrid_care'])
            target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
            y = check_predictions(future, target, classes)
            efeat = forward_batches(model.encoder,
                                    prep.transform(target['x']), device)
            source_pred = forward_batches(model.head,
                                          efeat.numpy(), device).argmax(1).numpy()
            if not np.array_equal(source_pred, future['static_mlp']):
                raise ValueError('Reloaded source differs from completed hybrid run')
            predictions = dict(static_mlp=source_pred,
                hybrid_target=future['hybrid_target'],
                hybrid_care=future['hybrid_care'])
            for method, head in heads.items():
                predictions[method] = forward_batches(head,
                    efeat.numpy(), device).argmax(1).numpy()
            np.savez_compressed(part / 'predictions.npz', ids=future['ids'],
                                labels=y, device_ids=future['device_ids'],
                                **predictions)
            summary_rows, class_rows = [], []
            for method, pred in predictions.items():
                score, rows, _ = metrics(y, source_pred, pred,
                                          historical, classes)
                summary_rows.append(dict(month=month, method=method,
                    label_queries=0 if method == 'static_mlp' else 1000,
                    **score))
                class_rows.extend(dict(month=month, method=method, **row)
                                  for row in rows)
            write_csv(part / 'summary.csv', summary_rows)
            write_csv(part / 'per_class.csv', class_rows)
            write_csv(part / 'coverage.csv', [dict(month=month,
                device_id=int(c), anchor_true_count=int((query['y'][anchor] == c).sum()),
                hybrid_true_count=int((query['y'][hybrid_all] == c).sum()),
                quota_true_count=int((query['y'][selected] == c).sum()),
                candidate_pool_count=int((query['y'] == c).sum()))
                for c in classes])
            write_json(part / 'inputs.json', dict(pilot=reader.consumed,
                cache=cache.consumed, paper=paper.consumed,
                hybrid=hybrid.consumed))
            complete(part, month_signature)
        summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
        coverage.extend(pd.read_csv(part / 'coverage.csv').to_dict('records'))
        saved = json.loads((part / 'selection.json').read_text())
        focuses.extend(dict(month=month, **row) for row in saved['focus'])
        query_stats.append(dict(month=month,
            targeted_queries=saved['targeted_queries'],
            badge_queries=saved['badge_queries'],
            focus_classes=[row['device_id'] for row in saved['focus']]))
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'coverage.csv', coverage)
    write_csv(output / 'focus_classes.csv', focuses)
    write_csv(output / 'query_stats.csv', query_stats)
    frame = pd.DataFrame(summaries)
    paired = []
    for month in MONTHS:
        subset = frame[frame.month == month].set_index('method')
        for current, baseline in (('quota_target', 'hybrid_target'),
                                  ('quota_care', 'hybrid_care')):
            paired.append(dict(month=month, comparison=current + ' minus ' + baseline,
                macro_f1_delta_pp=100 * (subset.loc[current, 'macro_f1_present'] -
                                         subset.loc[baseline, 'macro_f1_present']),
                collapse_f1_delta_pp=100 * (subset.loc[current, 'collapse_f1'] -
                                            subset.loc[baseline, 'collapse_f1']),
                negative_flip_delta_pp=100 * (
                    subset.loc[current, 'negative_flip_rate_vs_static_mlp'] -
                    subset.loc[baseline, 'negative_flip_rate_vs_static_mlp'])))
    write_csv(output / 'paired_comparison.csv', paired)
    focus_cov = pd.DataFrame(coverage)
    focus_cov = focus_cov[((focus_cov.month == '2020-03') &
        focus_cov.device_id.isin([5, 22, 26, 37])) |
        ((focus_cov.month == '2020-04') & (focus_cov.device_id == 20))]
    per_class = pd.concat([pd.read_csv(output / month / 'per_class.csv')
                           for month in MONTHS], ignore_index=True)
    focus_results = per_class[((per_class.month == '2020-03') &
        per_class.device_id.isin([5, 22, 26, 37])) |
        ((per_class.month == '2020-04') & (per_class.device_id == 20))]
    report = '\n'.join([
        'TMA exploratory quota-limited victim retrieval + BADGE; fixed 1000 paid labels/month.',
        'First 500 labels equal the completed hybrid uniform anchor.',
        f'Up to {MAX_FOCUS_CLASSES} rare underpredicted classes; stop at {LABEL_QUOTA_PER_CLASS} true paid labels/class, {MAX_ATTEMPTS_PER_CLASS} requests/class, {MAX_TARGETED_QUERIES} targeted requests/month.',
        'Only labels of selected queries enter sequential quota decisions; no future evaluation labels enter selection/training.',
        'Selection and both heads are saved before future evaluation is opened.',
        'Query allocation:',
        pd.DataFrame(query_stats).to_string(index=False),
        'Focus-class paid query coverage:',
        focus_cov.to_string(index=False),
        'Whole-model future results:',
        frame[['month', 'method', 'macro_f1_present', 'collapse_f1',
               'high_recall_group_f1', 'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        'Matched-budget differences in percentage points:',
        pd.DataFrame(paired).to_string(index=False),
        'Focus-class future results:',
        focus_results[['month', 'device_id', 'method', 'support',
                       'after_recall', 'after_f1', 'predicted_count']].to_string(index=False),
        'Compare quota_care vs hybrid_care and quota_target vs hybrid_target: same label budget, different identities.',
        'All target months have already been examined; this is exploratory, not independent validation.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed,
        cache=cache.consumed, paper=paper.consumed, hybrid=hybrid.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('QUOTA QUERY COMPLETE:', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Real TMA runs require the Linux data server')
    if min(args.source_seed, args.repair_seed) < 0 or args.threads < 1:
        parser.error('Invalid seed or thread count')
    run(args.root.resolve(strict=True), args.source_seed, args.repair_seed,
        args.device, args.threads)


if __name__ == '__main__':
    main()
