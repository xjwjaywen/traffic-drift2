"""Label-free second-stage retrieval for underpredicted TMA device classes.

The 500 random-anchor labels are reused from completed runs. Candidate-pool
labels are accessed only after every 500-query selection is saved. No heads
are trained and no future evaluation period is read.
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
for folder in ('tma_pilot_v1', 'tma_analysis_v1', 'tma_paper_v1',
               'tma_selective_repair_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader
from data import MONTHS, sha, write_csv, write_json
from pilot import forward_batches, mapped
from run import (PROTOCOL, complete, context, existing_output, output_directory,
                 verify_used_inputs)
from trial import load_source

MAX_FOCUS_CLASSES = 5
MAX_ABSORBERS = 2
MIN_ANCHOR_SUPPORT = 2
VARIANTS = ('nearest', 'contrastive', 'region_random')
BASELINES = ('uniform1000', 'hybrid')


def identity():
    files = [p for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    files += [EXPERIMENT / 'tma_selective_repair_v1/trial.py',
              EXPERIMENT / 'tma_paper_v1/run.py']
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in files}


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'victim_retrieval_v1'
    source = base / f'source_{source_seed}'
    output = source / f'repair_{repair_seed}'
    if any(path.is_symlink() for path in (base, source, output)):
        raise ValueError('Output directories must not be symlinks')
    return output


def indices_from_file(reader, name, query):
    saved = pd.read_csv(reader.file(name), dtype={'record_id': str},
                        usecols=['record_id'])
    lookup = {str(record_id): i for i, record_id in enumerate(query['ids'])}
    if len(lookup) != len(query['ids']):
        raise ValueError('Duplicate candidate-pool IDs')
    try:
        ids = np.asarray([lookup[key] for key in saved.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Saved query absent from candidate pool') from exc
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Repeated saved query')
    return ids


def nominate_classes(anchor_y, anchor_pred, pool_pred, classes,
                     max_focus=MAX_FOCUS_CLASSES):
    """Rank missing class mass from paid uniform labels and unlabeled outputs."""
    anchor_y = np.asarray(anchor_y, dtype=np.int64)
    anchor_pred = np.asarray(anchor_pred, dtype=np.int64)
    pool_pred = np.asarray(pool_pred, dtype=np.int64)
    n_classes = len(classes)
    if not len(anchor_y) or anchor_y.shape != anchor_pred.shape or not np.isin(
            np.r_[anchor_pred, pool_pred], np.arange(n_classes)).all():
        raise ValueError('Invalid anchor or source predictions')
    source_share = np.bincount(pool_pred, minlength=n_classes) / len(pool_pred)
    rows = []
    for c in range(n_classes):
        mask = anchor_y == c
        support = int(mask.sum())
        wrong = int((anchor_pred[mask] != c).sum())
        absorber_counts = np.bincount(anchor_pred[mask & (anchor_pred != c)],
                                      minlength=n_classes)
        absorbers = sorted(np.flatnonzero(absorber_counts),
                           key=lambda a: (-absorber_counts[a], a))[:MAX_ABSORBERS]
        estimated_share = (support + 1) / (len(anchor_y) + 2)
        deficit = max(0., estimated_share - source_share[c])
        eligible = support >= MIN_ANCHOR_SUPPORT and wrong >= 1 and deficit > 0
        rows.append(dict(class_index=c, device_id=int(classes[c]),
            anchor_support=support, anchor_source_errors=wrong,
            estimated_true_share=float(estimated_share),
            source_predicted_share=float(source_share[c]),
            estimated_missing_mass=float(deficit),
            absorbers=[int(a) for a in absorbers],
            absorber_device_ids=[int(classes[a]) for a in absorbers],
            eligible=bool(eligible)))
    focus = sorted((row for row in rows if row['eligible']),
                   key=lambda row: (-row['estimated_missing_mass'],
                                    -row['anchor_source_errors'], row['class_index']))[:max_focus]
    selected_classes = {row['class_index'] for row in focus}
    for row in rows:
        row['selected_focus'] = row['class_index'] in selected_classes
    return focus, rows


def normalized_features(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Invalid encoder features')
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    z = (features - center) / np.maximum(scale, 1e-5)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return (z / np.maximum(norm, 1e-8)).astype(np.float32)


def candidate_queues(z, pool_pred, anchor, anchor_y, focus, variant, seed):
    anchor_mask = np.zeros(len(pool_pred), dtype=bool)
    anchor_mask[anchor] = True
    queues = {}
    rng = np.random.default_rng(seed)
    for row in focus:
        c = row['class_index']
        absorbers = row['absorbers']
        region = np.flatnonzero(~anchor_mask & np.isin(pool_pred, absorbers))
        if not len(region):
            queues[c] = np.empty(0, dtype=np.int64)
            continue
        if variant == 'region_random':
            queues[c] = rng.permutation(region)
            continue
        seeds = anchor[(anchor_y == c) & np.isin(pool_pred[anchor], absorbers)]
        if not len(seeds):
            raise ValueError('Selected class has no paid victim seed in absorber region')
        positive = (z[region] @ z[seeds].T).max(axis=1)
        if variant == 'nearest':
            score = positive
        elif variant == 'contrastive':
            # A true absorber anchor is a negative example for its own region.
            score = positive.copy()
            for a in absorbers:
                in_region = pool_pred[region] == a
                negatives = anchor[(anchor_y == a) & (pool_pred[anchor] == a)]
                if in_region.any() and len(negatives):
                    score[in_region] -= (z[region[in_region]] @
                                         z[negatives].T).max(axis=1)
        else:
            raise ValueError('Unknown retrieval variant')
        queues[c] = region[np.argsort(-score, kind='stable')]
    return queues


def select_from_queues(queues, anchor, n_pool, budget, seed):
    """Allocate equal quotas; unfillable slots become uniform exploration."""
    if budget < 0 or len(anchor) + budget > n_pool:
        raise ValueError('Invalid active budget')
    used = np.zeros(n_pool, dtype=bool)
    used[anchor] = True
    focus = list(queues)
    chosen = []
    per_focus = {c: 0 for c in focus}
    if focus:
        quotas = {c: budget // len(focus) + (i < budget % len(focus))
                  for i, c in enumerate(focus)}
        pointers = {c: 0 for c in focus}
        for _ in range(max(quotas.values())):
            for c in focus:
                if per_focus[c] >= quotas[c]:
                    continue
                queue = queues[c]
                while pointers[c] < len(queue) and used[queue[pointers[c]]]:
                    pointers[c] += 1
                if pointers[c] < len(queue):
                    item = int(queue[pointers[c]])
                    chosen.append(item)
                    used[item] = True
                    per_focus[c] += 1
                    pointers[c] += 1
    fallback = budget - len(chosen)
    if fallback:
        available = np.flatnonzero(~used)
        chosen.extend(np.random.default_rng(seed + 100000).choice(
            available, fallback, replace=False).tolist())
    result = np.r_[anchor, np.asarray(chosen, dtype=np.int64)]
    if len(result) != len(anchor) + budget or len(np.unique(result)) != len(result):
        raise ValueError('Active query budget or uniqueness failed')
    return result, dict(fallback_uniform=int(fallback),
                        assigned_per_focus={str(k): int(v) for k, v in per_focus.items()})


def coverage_rows(month, classes, all_labels, anchor, selections, budget):
    rows = []
    for name, selected in selections.items():
        if len(selected) != budget or not np.array_equal(selected[:len(anchor)], anchor):
            raise ValueError('Baseline or retrieval prefix/budget mismatch')
        for c in classes:
            rows.append(dict(month=month, method=name, device_id=int(c),
                anchor_true_count=int((all_labels[anchor] == c).sum()),
                active_true_count=int((all_labels[selected[len(anchor):]] == c).sum()),
                paid_true_count=int((all_labels[selected] == c).sum()),
                candidate_pool_count=int((all_labels == c).sum())))
    return rows


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
            raise ValueError(name + ' run differs from fixed source/cache/protocol')
    if hybrid.marker['signature']['paper_marker'] != sha(paper.root / 'complete.json'):
        raise ValueError('Hybrid and paper runs are not paired')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        methods=VARIANTS, max_focus=MAX_FOCUS_CLASSES,
        max_absorbers=MAX_ABSORBERS, min_anchor_support=MIN_ANCHOR_SUPPORT,
        device=device, threads=threads, code=identity())
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper,
                           hybrid=hybrid)
        print('VICTIM RETRIEVAL COMPLETE (verified existing run):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    model, prep = load_source(reader, classes)
    budget = reader.marker['signature']['budget']
    if budget != 1000:
        raise ValueError('This fixed pilot expects 1000 paid labels')
    all_coverage, all_focus = [], []
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
            random_all = indices_from_file(paper, month + '/random_queries.csv', query)
            hybrid_all = indices_from_file(hybrid, month + '/hybrid_queries.csv', query)
            anchor = random_all[:budget // 2]
            if (len(random_all) != budget or len(hybrid_all) != budget or
                    not np.array_equal(hybrid_all[:len(anchor)], anchor)):
                raise ValueError('Baseline query budgets or random prefix differ')
            anchor_y = mapped(query['y'][anchor], classes)
            qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
            qlogits = forward_batches(model.head, qfeat.numpy(), device)
            pool_pred = qlogits.argmax(1).numpy()
            focus, nomination = nominate_classes(anchor_y, pool_pred[anchor],
                                                  pool_pred, classes)
            z = normalized_features(qfeat.numpy())
            selections = dict(uniform1000=random_all, hybrid=hybrid_all)
            diagnostics = {}
            for variant in VARIANTS:
                queues = candidate_queues(z, pool_pred, anchor, anchor_y, focus,
                                          variant, repair_seed)
                selected, info = select_from_queues(queues, anchor, len(query['ids']),
                                                    budget - len(anchor), repair_seed)
                selections[variant] = selected
                diagnostics[variant] = info
            # Frozen before opening any unqueried pool labels for coverage.
            write_json(part / 'selection.json', dict(month=month, classes=classes,
                anchor_record_ids=[str(query['ids'][i]) for i in anchor],
                focus=focus, nomination=nomination, diagnostics=diagnostics,
                selections={name: [str(query['ids'][i]) for i in selected]
                            for name, selected in selections.items()}))
            print(month, 'selected focus classes =',
                  [row['device_id'] for row in focus], flush=True)
            frozen = json.loads((part / 'selection.json').read_text())
            if (set(frozen['selections']) != set(selections) or any(
                    frozen['selections'][name] !=
                    [str(query['ids'][i]) for i in selected]
                    for name, selected in selections.items())):
                raise ValueError('Frozen selection list changed')
            rows = coverage_rows(month, classes, query['y'], anchor,
                                 selections, budget)
            write_csv(part / 'coverage.csv', rows)
            write_json(part / 'inputs.json', dict(pilot=reader.consumed,
                cache=cache.consumed, paper=paper.consumed, hybrid=hybrid.consumed))
            complete(part, month_signature)
        all_coverage.extend(pd.read_csv(part / 'coverage.csv').to_dict('records'))
        selection = json.loads((part / 'selection.json').read_text())
        all_focus.extend(dict(month=month, **row) for row in selection['focus'])
    write_csv(output / 'coverage.csv', all_coverage)
    write_csv(output / 'focus_classes.csv', all_focus)
    frame = pd.DataFrame(all_coverage)
    focus = frame[((frame.month == '2020-03') & frame.device_id.isin([5, 22, 26, 37])) |
                  ((frame.month == '2020-04') & (frame.device_id == 20))]
    wider = focus.pivot_table(index=['month', 'device_id', 'anchor_true_count'],
        columns='method', values='paid_true_count', aggfunc='first').reset_index()
    focus_table = pd.DataFrame(all_focus)
    coverage_10 = frame.assign(reached_10=frame.paid_true_count >= 10).groupby(
        ['month', 'method']).reached_10.sum().reset_index()
    report = '\n'.join([
        'TMA victim-neighbor retrieval pilot; no training, no future evaluation data.',
        'Every method uses the identical 500 uniform labels plus 500 extra queries.',
        'Focus classes are nominated from the uniform paid labels and unlabeled source-prediction deficit.',
        'No known collapse class IDs or unqueried labels enter selection; all class coverage is retrospective.',
        'nearest / contrastive / region_random share the same nominated absorber regions.',
        'Nominated classes and evidence:',
        focus_table[['month', 'device_id', 'anchor_support', 'anchor_source_errors',
                     'estimated_missing_mass', 'absorber_device_ids']].to_string(index=False)
            if len(focus_table) else 'none',
        'Focus class total paid-label counts:',
        wider.to_string(index=False),
        'Number of known classes reaching >=10 true paid labels:',
        coverage_10.to_string(index=False),
        'Compare absolute gains and losses against uniform1000 and hybrid; do not choose a winner from these seen months.',
        'If focus classes were not nominated or no candidate enrichment appears, this approach is not promising.',
        'Previously examined months; exploratory query-coverage diagnostic only.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
        paper=paper.consumed, hybrid=hybrid.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('VICTIM RETRIEVAL COMPLETE:', output, flush=True)
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
        parser.error('The TMA query pilot runs on the Linux data server')
    if min(args.source_seed, args.repair_seed) < 0 or args.threads < 1:
        parser.error('Invalid seed or thread count')
    run(args.root.resolve(strict=True), args.source_seed, args.repair_seed,
        args.device, args.threads)


if __name__ == '__main__':
    main()
