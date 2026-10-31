"""Fixed-label TMA test of a small class-balanced memory of paid past queries.

Each month restarts from the fixed source head.  Two newly trained heads add
equal numbers of rows to the current paid queries: one draws only from previous
months' paid queries, the other repeats current paid queries.  Both retain the
original CARE historical references, KD, epochs, and optimizer settings.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ('tma_selective_repair_v1', 'tma_paper_v1',
               'tma_analysis_v1', 'tma_pilot_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))

from audit import load_paid  # noqa: E402
from trial import load_references, load_source  # noqa: E402
from run import (PROTOCOL, check_predictions, complete, context,
                 existing_output, fit_paper, metrics, output_directory)  # noqa: E402
from analyze import Reader  # noqa: E402
from pilot import forward_batches, stats  # noqa: E402
from data import MONTHS, sha, write_csv, write_json  # noqa: E402

MEMORY_PER_CLASS = 10
METHODS = ('current_repeat_control', 'paid_temporal_memory')


def choose_paid_memory(prior, classes, cap, seed):
    """Take up to cap paid rows/class, newest month first, randomly within month."""
    if cap < 1 or not prior:
        raise ValueError('Expected prior paid months and positive class cap')
    rng = np.random.default_rng(seed)
    chosen = []
    counts = np.zeros(len(classes), dtype=int)
    for month, part in reversed(prior):
        labels = part['labels']
        for c in range(len(classes)):
            available = np.flatnonzero(labels == c)
            if not len(available) or counts[c] >= cap:
                continue
            draw = rng.permutation(available)[:cap - counts[c]]
            chosen.extend((month, int(i), c) for i in draw)
            counts[c] += len(draw)
    return chosen


def output_for(root, source_seed, repair_seed, month):
    base = root / 'temporal_memory_trial_v1'
    source = base / f'source_{source_seed}'
    seed = source / f'repair_{repair_seed}'
    result = seed / month
    if any(p.is_symlink() for p in (base, source, seed, result)):
        raise ValueError('Output directory cannot be a symlink')
    return result


def run_seed(root, source_seed, repair_seed, device, threads):
    reader, cache, classes, _ = context(root, source_seed)
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                    f'repair_{repair_seed}')
    ps, hs = paper.marker['signature'], hybrid.marker['signature']
    if (ps['source_seed'] != source_seed or ps['repair_seed'] != repair_seed
            or hs['source_seed'] != source_seed or hs['repair_seed'] != repair_seed
            or ps['protocol'] != PROTOCOL or hs['protocol'] != PROTOCOL
            or hs['paper_marker'] != sha(paper.root / 'complete.json')
            or ps['source_marker'] != sha(reader.root / 'complete.json')
            or ps['cache_marker'] != sha(cache.root / 'complete.json')
            or hs['source_marker'] != ps['source_marker']
            or hs['cache_marker'] != ps['cache_marker']
            or ps['device'] != device or hs['device'] != device
            or ps['threads'] != threads or hs['threads'] != threads
            or hs['months'] != list(MONTHS)):
        raise ValueError('Existing source, labels, or protocol do not match')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        months=list(MONTHS), memory_per_class=MEMORY_PER_CLASS,
        device=device, threads=threads, protocol=PROTOCOL,
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        code={p.name: sha(p) for p in (HERE / 'experiment.py', HERE / 'audit.py')})
    torch.set_num_threads(threads)
    torch.manual_seed(repair_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, prep = load_source(reader, classes)
    rx, ry = load_references(reader, cache, paper, classes, source_seed, repair_seed)
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    history = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(history['labels'], history['mlp'], len(classes))
    paid = {}
    results = []
    for month_idx, month in enumerate(MONTHS):
        # Load only the current and already completed earlier paid queries.
        paid[month] = load_paid(cache, paper, hybrid, month, model, prep,
                                device, classes)
        if month_idx == 0:
            continue
        out = output_for(root, source_seed, repair_seed, month)
        month_signature = dict(signature, month=month)
        if existing_output(out, month_signature):
            results.append(pd.read_csv(out / 'metrics.csv'))
            print(month, 'seed', repair_seed, 'verified trial reused', flush=True)
            continue
        write_json(out / 'config.json', month_signature)
        current = paid[month]
        current_known = np.flatnonzero(current['labels'] >= 0)
        if not len(current_known):
            raise ValueError('No known current paid labels')
        prior = [(m, paid[m]) for m in MONTHS[:month_idx]]
        chosen = choose_paid_memory(prior, classes, MEMORY_PER_CLASS,
                                    20261031 + repair_seed * 100 + month_idx)
        if not chosen:
            raise ValueError('No eligible paid historical memory')
        mem_x = np.stack([paid[m]['features'][i] for m, i, _ in chosen])
        mem_y = np.asarray([c for _, _, c in chosen], dtype=np.int64)
        write_csv(out / 'memory_paid_ids.csv', [dict(source_month=m,
            record_id=str(paid[m]['ids'][i]), device_id=int(classes[c]))
            for m, i, c in chosen])
        rng = np.random.default_rng(20261031 + repair_seed * 100 + month_idx)
        repeated = rng.choice(current_known, size=len(chosen), replace=True)
        write_csv(out / 'current_repeat_ids.csv', [dict(record_id=str(current['ids'][i]),
            device_id=int(classes[current['labels'][i]])) for i in repeated])
        cx = current['features'][current_known]
        cy = current['labels'][current_known]
        training = {
            'current_repeat_control': (np.concatenate([cx, current['features'][repeated]]),
                                       np.concatenate([cy, current['labels'][repeated]])),
            'paid_temporal_memory': (np.concatenate([cx, mem_x]),
                                     np.concatenate([cy, mem_y])),
        }
        heads, traces = {}, {}
        for method in METHODS:
            x, y = training[method]
            head, trace = fit_paper(model.head, torch.from_numpy(x), y,
                rfeat, ry, teacher, True, repair_seed, device, PROTOCOL)
            heads[method], traces[method] = head, trace
            torch.save(head.state_dict(), out / (method + '_head.pt'))
            print(month, 'seed', repair_seed, method, 'head saved', flush=True)
        if traces[METHODS[0]]['optimizer_steps'] != traces[METHODS[1]]['optimizer_steps']:
            raise ValueError('Compute-matched head schedules differ')
        write_json(out / 'training.json', traces)

        # All selection and fitted heads are saved before opening test labels.
        future = hybrid.arrays(month + '/predictions.npz',
            ['ids', 'labels', 'device_ids', 'static_mlp', 'hybrid_care'])
        target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
        y = check_predictions(future, target, classes)
        efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
        source = future['static_mlp']
        source_check = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
        if not np.array_equal(source, source_check):
            raise ValueError('Reloaded source differs from fixed hybrid source')
        rows, class_rows = [], []
        for method, pred in [('hybrid_care', future['hybrid_care'])] + [
                (name, forward_batches(heads[name], efeat.numpy(), device).argmax(1).numpy())
                for name in METHODS]:
            score, per_class, _ = metrics(y, source, pred, historical, classes)
            rows.append(dict(source_seed=source_seed, repair_seed=repair_seed,
                month=month, method=method, paid_current=1000,
                available_previous_paid=sum(len(paid[m]['ids']) for m in MONTHS[:month_idx]),
                memory_rows=len(chosen) if method == 'paid_temporal_memory' else 0,
                extra_current_rows=len(chosen) if method == 'current_repeat_control' else 0,
                **score))
            class_rows.extend(dict(repair_seed=repair_seed, month=month,
                                   method=method, **row) for row in per_class)
        write_csv(out / 'metrics.csv', rows)
        write_csv(out / 'per_class.csv', class_rows)
        complete(out, month_signature)
        results.append(pd.DataFrame(rows))
        print(month, 'seed', repair_seed, 'trial complete', flush=True)
    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    try:
        seeds = [int(s) for s in args.repair_seeds.split(',')]
    except ValueError:
        parser.error('Repair seeds must be comma-separated integers')
    if (not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds)
            or args.source_seed < 0 or args.threads < 1):
        parser.error('Invalid seeds or thread count')
    root = args.root.resolve(strict=True)
    results = pd.concat([run_seed(root, args.source_seed, seed, args.device,
                                  args.threads) for seed in seeds], ignore_index=True)
    out = (root / 'temporal_memory_trial_v1' / f'source_{args.source_seed}' /
           ('aggregate_' + '_'.join(map(str, seeds))))
    if out.is_symlink():
        raise ValueError('Aggregate output cannot be a symlink')
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / 'by_seed.csv', index=False)
    columns = ['macro_f1_present', 'collapse_f1', 'remaining_collapse_count',
               'high_recall_group_f1', 'negative_flip_rate_vs_static_mlp']
    means = results.groupby(['month', 'method'])[columns].agg(['mean', 'std', 'count'])
    means.columns = ['_'.join(c) for c in means.columns]
    means = means.reset_index()
    means.to_csv(out / 'mean_sd.csv', index=False)
    focus_frames = [pd.read_csv(output_for(root, args.source_seed, seed, month) /
                                'per_class.csv') for seed in seeds for month in MONTHS[1:]]
    focus = pd.concat(focus_frames, ignore_index=True)
    focus = focus[((focus.month == '2020-03') & focus.device_id.isin([5, 22, 26, 37]))
                  | ((focus.month == '2020-04') & (focus.device_id == 20))]
    focus = focus.groupby(['month', 'device_id', 'method'])[['after_f1', 'after_recall']].agg(
        ['mean', 'std', 'count'])
    focus.columns = ['_'.join(c) for c in focus.columns]
    focus = focus.reset_index()
    focus.to_csv(out / 'focus_class_mean_sd.csv', index=False)
    pivot = results.pivot(index=['repair_seed', 'month'], columns='method',
        values=['macro_f1_present', 'collapse_f1',
                'negative_flip_rate_vs_static_mlp'])
    comparisons = []
    for baseline in ('hybrid_care', 'current_repeat_control'):
        for metric in ('macro_f1_present', 'collapse_f1',
                       'negative_flip_rate_vs_static_mlp'):
            delta = (pivot[(metric, 'paid_temporal_memory')] - pivot[(metric, baseline)])
            for month in MONTHS[1:]:
                values = delta.xs(month, level='month').dropna()
                comparisons.append(dict(month=month, comparison='memory_minus_' + baseline,
                    metric=metric, mean_pp=100 * float(values.mean()) if len(values) else np.nan,
                    std_pp=100 * float(values.std()) if len(values) > 1 else np.nan,
                    positive_seeds=int((values > 0).sum()), count=len(values)))
    pd.DataFrame(comparisons).to_csv(out / 'paired_deltas.csv', index=False)
    report = '\n'.join([
        'Exploratory chronological paid-memory training control on fixed hybrid labels.',
        f'Source seed {args.source_seed}; repair seeds {seeds}; per-class memory cap {MEMORY_PER_CLASS}.',
        'Only paid queries from strictly earlier months enter memory; current query IDs match hybrid CARE.',
        'Both new heads use equal training rows, original CARE references/KD, epochs and optimizer schedule.',
        'Current-repeat control adds duplicated current labels; temporal memory adds old paid labels.',
        'Past paid labels have zero new annotation cost but add stored data and training rows.',
        'Monthly results (mean, sample SD, count):',
        means.to_string(index=False),
        'Memory minus baselines (percentage points, by repair seed):',
        pd.DataFrame(comparisons).to_string(index=False),
        'Previously identified difficult devices (mean, sample SD, count):',
        focus.to_string(index=False),
        'All four months were previously inspected; this is exploratory, not independent validation.',
    ]) + '\n'
    (out / 'REPORT.txt').write_text(report)
    print(report, flush=True)
    print('TEMPORAL MEMORY TRIAL COMPLETE:', out, flush=True)


if __name__ == '__main__':
    main()
