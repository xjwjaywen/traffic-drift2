"""Fixed-label nonlinear controls on the frozen TMA source representation."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ('tma_pilot_v1', 'tma_analysis_v1', 'tma_paper_v1',
               'tma_selective_repair_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader
from data import MONTHS, sha, finish, validate_complete, write_csv, write_json
from pilot import forward_batches, mapped, stats
from run import (PROTOCOL, check_predictions, context, metrics, output_directory,
                 verify_used_inputs)
from trial import load_references, load_source

RF = dict(n_estimators=200, min_samples_leaf=2,
          class_weight='balanced_subsample', max_features='sqrt')
FOCUS = {'2020-03': (5, 22, 26, 37), '2020-04': (20,)}


def paid_queries(hybrid, month, query, classes, budget):
    table = pd.read_csv(hybrid.file(month + '/hybrid_queries.csv'),
                        dtype={'record_id': str})
    lookup = {str(key): i for i, key in enumerate(query['ids'])}
    if len(table) != budget or len(lookup) != len(query['ids']):
        raise ValueError('Hybrid paid budget or query IDs changed')
    try:
        selected = np.asarray([lookup[key] for key in table.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Paid query absent from cache') from exc
    if len(np.unique(selected)) != budget:
        raise ValueError('Duplicate paid query')
    labels = table.device_id.to_numpy(dtype=np.int64)
    if not np.array_equal(query['y'][selected], labels):
        raise ValueError('Saved paid label differs from cache')
    return selected, mapped(labels, classes)


def fit_rf(target_x, target_y, reference_x, reference_y, seed, threads):
    known = target_y >= 0
    if len(reference_y) == 0 or (reference_y < 0).any():
        raise ValueError('Historical reference must cover known classes')
    x = np.concatenate((target_x[known], reference_x))
    y = np.concatenate((target_y[known], reference_y))
    model = RandomForestClassifier(**RF, random_state=seed, n_jobs=threads)
    model.fit(x, y)
    if not np.array_equal(model.classes_, np.arange(int(reference_y.max()) + 1)):
        raise ValueError('RF output classes do not match the fixed source head')
    return model, int(known.sum())


def run(root, source_seed=0, repair_seeds=(0, 1, 2, 3, 4), device='cuda', threads=8):
    reader, cache, classes, _ = context(root, source_seed)
    output = root.resolve() / 'nonlinear_head_control_v1' / f'source_{source_seed}'
    if any(p.is_symlink() for p in (output.parent, output)):
        raise ValueError('Output directory must not be a symlink')
    repair_seeds = tuple(sorted(set(repair_seeds)))
    upstream = {}
    papers, hybrids = {}, {}
    for repair_seed in repair_seeds:
        paper_root = output_directory(root, source_seed) / f'repair_{repair_seed}'
        hybrid_root = (root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                       f'repair_{repair_seed}')
        papers[repair_seed], hybrids[repair_seed] = Reader(paper_root), Reader(hybrid_root)
        upstream[str(repair_seed)] = dict(paper=sha(paper_root / 'complete.json'),
                                          hybrid=sha(hybrid_root / 'complete.json'))
    readers = dict(pilot=reader, cache=cache)
    readers.update({f'paper_{seed}': papers[seed] for seed in repair_seeds})
    readers.update({f'hybrid_{seed}': hybrids[seed] for seed in repair_seeds})
    signature = dict(schema=1, source_seed=source_seed, repair_seeds=list(repair_seeds),
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        upstream_markers=upstream, code_sha256=sha(HERE / 'experiment.py'), protocol=PROTOCOL,
        forest=RF, months=MONTHS, focus={k: list(v) for k, v in FOCUS.items()},
        device=device, threads=threads,
        note='Fixed 1000 paid hybrid labels and same historical references as CARE; exploratory.')
    if output.exists() and any(output.iterdir()) and not (output / 'config.json').exists():
        raise ValueError('Refusing unrecognized nonempty output directory')
    output.mkdir(parents=True, exist_ok=True)
    if validate_complete(output, signature):
        verify_used_inputs(output, **readers)
        print('NONLINEAR HEAD CONTROL COMPLETE (verified):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    model, prep = load_source(reader, classes)
    h = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(h['labels'], h['mlp'], len(classes))
    rows, class_rows = [], []
    for month in MONTHS:
        query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
        raw_query = prep.transform(query['x'])
        frozen_query = forward_batches(model.encoder, raw_query, device).numpy()
        # Reading test inputs does not reveal their labels. No training uses these rows.
        eval_x = cache.arrays(month + '_eval.npz', ['x'])['x']
        raw_eval = prep.transform(eval_x)
        frozen_eval = forward_batches(model.encoder, raw_eval, device).numpy()
        for repair_seed in repair_seeds:
            part = output / month / f'repair_{repair_seed}'
            part.mkdir(parents=True, exist_ok=True)
            paper, hybrid = papers[repair_seed], hybrids[repair_seed]
            for name, bundle in (('paper', paper), ('hybrid', hybrid)):
                s = bundle.marker['signature']
                if (s['source_seed'] != source_seed or s['repair_seed'] != repair_seed or
                    s['source_marker'] != signature['source_marker'] or
                    s['cache_marker'] != signature['cache_marker'] or
                    s['protocol'] != PROTOCOL):
                    raise ValueError(name + ' run differs from requested source or protocol')
            if hybrid.marker['signature']['paper_marker'] != sha(paper.root / 'complete.json'):
                raise ValueError('Hybrid queries are not paired with paper run')
            selected, paid_y = paid_queries(hybrid, month, query, classes,
                                            reader.marker['signature']['budget'])
            rx, ry = load_references(reader, cache, paper, classes,
                                     source_seed, repair_seed)
            raw_ref = prep.transform(rx)
            frozen_ref = forward_batches(model.encoder, raw_ref, device).numpy()
            predictions = {}
            for name, train_x, ref_x, test_x in (
                    ('raw_rf', raw_query, raw_ref, raw_eval),
                    ('frozen_rf', frozen_query, frozen_ref, frozen_eval)):
                forest, known = fit_rf(train_x[selected], paid_y, ref_x, ry,
                                       repair_seed, threads)
                predictions[name] = forest.predict(test_x).astype(np.int64)
                print(month, repair_seed, name, 'fit using', known,
                      'known paid labels +', len(ry), 'historical references', flush=True)
            np.savez_compressed(part / 'predictions.npz', **predictions)
            write_json(part / 'fit.json', dict(month=month, repair_seed=repair_seed,
                paid_queries=len(selected), known_paid=int((paid_y >= 0).sum()),
                reference_rows=len(ry), rf=RF,
                note='Predictions saved before future evaluation labels were opened.'))
            # Open future labels and existing paired model predictions only now.
            target = cache.arrays(month + '_eval.npz', ['ids', 'y'])
            base = hybrid.arrays(month + '/predictions.npz',
                ['ids', 'labels', 'device_ids', 'static_mlp', 'hybrid_target', 'hybrid_care'])
            y = check_predictions(base, target, classes)
            for name in ('static_mlp', 'hybrid_target', 'hybrid_care',
                         'raw_rf', 'frozen_rf'):
                pred = base[name] if name in base else predictions[name]
                summary, per_class, _ = metrics(y, base['static_mlp'], pred,
                                                historical, classes)
                rows.append(dict(month=month, repair_seed=repair_seed,
                                 method=name, **summary))
                for device_id in FOCUS.get(month, ()):
                    found = next((r for r in per_class if r['device_id'] == device_id), None)
                    if found is None:
                        raise ValueError('Focus class missing from class report')
                    class_rows.append(dict(month=month, repair_seed=repair_seed,
                                           method=name, **found))
    write_csv(output / 'summary.csv', rows)
    write_csv(output / 'focus_classes.csv', class_rows)
    table = pd.DataFrame(rows)
    group = table.groupby(['month', 'method'])[['macro_f1_present',
        'collapse_f1', 'high_recall_group_f1',
        'negative_flip_rate_vs_static_mlp']].mean().reset_index()
    focus = pd.DataFrame(class_rows).groupby(['month', 'device_id', 'method'])[
        'after_f1'].mean().reset_index()
    (output / 'REPORT.txt').write_text(
        'Exploratory fixed-label nonlinear-head controls on raw and frozen features.\n'
        'Updated methods share the same 1000 paid hybrid labels per month and repair seed.\n'
        'RF controls also use the same 5/class historical reference rows as CARE.\n'
        'RF changes architecture and objective; differences do not isolate only nonlinearity.\n'
        'No future evaluation labels enter fitting. All target months have been examined before.\n'
        'The full-query-label oracle report is NOT a comparable 1000-label method result.\n\n'
        'Across-seed whole-model means:\n' + group.to_string(index=False) +
        '\n\nAcross-seed focus-class F1:\n' + focus.to_string(index=False) + '\n')
    write_json(output / 'inputs.json', {role: bundle.consumed for role, bundle in readers.items()})
    files = [str(p.relative_to(output)) for p in output.rglob('*')
             if p.is_file() and p.name not in ('config.json', 'complete.json')]
    finish(output, signature, files)
    print('NONLINEAR HEAD CONTROL COMPLETE:', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.repair_seeds.split(','))
    run(args.root, args.source_seed, seeds, args.device, args.threads)
