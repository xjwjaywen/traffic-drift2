"""Audit whether previously paid TMA labels transfer to later months.

Only completed query selections and their paid labels enter this audit.  The
current month's first 500 uniformly selected paid labels are an assessment
anchor.  No *_eval.npz or saved evaluation predictions are opened.
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

from trial import load_source  # noqa: E402
from run import (PROTOCOL, complete, context, existing_output,
                 output_directory)  # noqa: E402
from analyze import Reader  # noqa: E402
from pilot import forward_batches, mapped  # noqa: E402
from data import MONTHS, sha, write_csv, write_json  # noqa: E402

FOCUS = {'2020-03': (5, 22, 26, 37), '2020-04': (20,)}


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Features must be a finite matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def transfer_rows(history_x, history_y, anchor_x, anchor_y, classes):
    """Nearest old paid examples and old class centroids on current paid anchor.

    Returns one row per known class.  This is a diagnostic, not a classifier
    selection rule: low current support produces unstable classwise estimates.
    """
    history_y, anchor_y = np.asarray(history_y), np.asarray(anchor_y)
    if (len(history_x) != len(history_y) or len(anchor_x) != len(anchor_y)
            or len(history_x) == 0 or history_x.shape[1] != anchor_x.shape[1]
            or np.any((history_y < 0) | (history_y >= len(classes)))
            or np.any((anchor_y < -1) | (anchor_y >= len(classes)))):
        raise ValueError('Invalid history or anchor')
    hx, ax = normalize(history_x), normalize(anchor_x)
    known = anchor_y >= 0
    sim = ax[known] @ hx.T
    ay = anchor_y[known]
    top = np.argmax(sim, axis=1) if len(ay) else np.empty(0, dtype=int)
    k = min(5, len(history_y))
    topk = np.argsort(-sim, axis=1)[:, :k] if len(ay) else np.empty((0, k), dtype=int)
    present = np.unique(history_y)
    centroids = normalize(np.vstack([hx[history_y == c].mean(axis=0) for c in present]))
    cent_pred = present[np.argmax(ax[known] @ centroids.T, axis=1)] if len(ay) else np.empty(0, dtype=int)
    rows = []
    for c, device_id in enumerate(classes):
        mask = ay == c
        n = int(mask.sum())
        old = int((history_y == c).sum())
        if n and old:
            same = sim[mask][:, history_y == c].max(axis=1)
            other = sim[mask][:, history_y != c]
            margin = float(np.median(same - other.max(axis=1))) if other.shape[1] else np.nan
            top1 = float((history_y[top[mask]] == c).mean())
            top5 = float((history_y[topk[mask]] == c).any(axis=1).mean())
            prototype = float((cent_pred[mask] == c).mean())
            chance_top1 = old / len(history_y)
            misses = 1.0
            for j in range(k):
                misses *= (len(history_y) - old - j) / (len(history_y) - j)
            chance_top5 = 1.0 - max(0.0, misses)
        else:
            margin = top1 = top5 = prototype = np.nan
            chance_top1 = chance_top5 = np.nan
        rows.append(dict(device_id=int(device_id), anchor_support=n,
                         previous_paid_support=old, top1_old_neighbor_rate=top1,
                         top5_contains_true_rate=top5,
                         random_memory_top1_rate=chance_top1,
                         random_memory_top5_rate=chance_top5,
                         top5_over_random_memory=top5 - chance_top5,
                         old_prototype_recall=prototype,
                         median_true_minus_other_similarity=margin))
    return rows


def load_paid(cache, paper, hybrid, month, model, prep, device, classes):
    table = pd.read_csv(hybrid.file(month + '/hybrid_queries.csv'),
                        dtype={'record_id': str})
    random = pd.read_csv(paper.file(month + '/random_queries.csv'),
                         dtype={'record_id': str})
    if len(table) != 1000 or len(random) != 1000 or not table.record_id.is_unique:
        raise ValueError('Expected 1000 unique paid hybrid/random queries')
    if table.record_id.iloc[:500].tolist() != random.record_id.iloc[:500].tolist():
        raise ValueError('Hybrid random anchor is not the original fixed prefix')
    pool = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
    lookup = {str(key): i for i, key in enumerate(pool['ids'])}
    if len(lookup) != len(pool['ids']):
        raise ValueError('Duplicate query-pool identities')
    try:
        selected = np.asarray([lookup[k] for k in table.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Paid query absent from cached pool') from exc
    paid_device = pool['y'][selected]
    if not np.array_equal(paid_device, table.device_id.to_numpy()):
        raise ValueError('Paid label differs from completed hybrid record')
    labels = mapped(paid_device, classes)
    if not np.array_equal(labels >= 0,
                          table.known_class.astype(str).str.lower().eq('true').to_numpy()):
        raise ValueError('Known-class flags differ from fixed source classes')
    features = forward_batches(model.encoder, prep.transform(pool['x'][selected]), device).numpy()
    if not np.isfinite(features).all():
        raise ValueError('Non-finite encoded paid features')
    return dict(features=features, labels=labels, ids=table.record_id.to_numpy())


def run(root, source_seed, repair_seeds, device, threads):
    reader, cache, classes, _ = context(root, source_seed)
    source_marker = sha(reader.root / 'complete.json')
    cache_marker = sha(cache.root / 'complete.json')
    inputs = {}
    pairs = {}
    for seed in repair_seeds:
        paper = Reader(output_directory(root, source_seed) / f'repair_{seed}')
        hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' / f'repair_{seed}')
        ps, hs = paper.marker['signature'], hybrid.marker['signature']
        if (ps['source_seed'] != source_seed or ps['repair_seed'] != seed
                or hs['source_seed'] != source_seed or hs['repair_seed'] != seed
                or ps['protocol'] != PROTOCOL or hs['protocol'] != PROTOCOL
                or ps['source_marker'] != source_marker or ps['cache_marker'] != cache_marker
                or hs['paper_marker'] != sha(paper.root / 'complete.json')
                or hs['source_marker'] != source_marker or hs['cache_marker'] != cache_marker
                or ps['device'] != device or hs['device'] != device
                or ps['threads'] != threads or hs['threads'] != threads
                or hs['months'] != list(MONTHS)):
            raise ValueError(f'Completed input provenance mismatch for seed {seed}')
        pairs[seed] = paper, hybrid
        inputs[str(seed)] = dict(paper=sha(paper.root / 'complete.json'),
                                 hybrid=sha(hybrid.root / 'complete.json'))
    signature = dict(schema=1, source_seed=source_seed, repair_seeds=repair_seeds,
                     months=list(MONTHS), device=device, threads=threads,
                     protocol=PROTOCOL, source_marker=source_marker,
                     cache_marker=cache_marker, inputs=inputs,
                     code_sha256=sha(HERE / 'audit.py'))
    base = root / 'temporal_memory_audit_v1'
    source = base / f'source_{source_seed}'
    out = source / ('aggregate_' + '_'.join(map(str, repair_seeds)))
    if any(path.is_symlink() for path in (base, source, out)):
        raise ValueError('Audit output cannot be a symlink')
    if existing_output(out, signature):
        print((out / 'REPORT.txt').read_text(), flush=True)
        print('TEMPORAL MEMORY AUDIT COMPLETE (verified):', out, flush=True)
        return out
    write_json(out / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, prep = load_source(reader, classes)
    count_rows, transfer, overview = [], [], []
    for seed in repair_seeds:
        paper, hybrid = pairs[seed]
        paid = {month: load_paid(cache, paper, hybrid, month, model, prep,
                                 device, classes) for month in MONTHS}
        for index, month in enumerate(MONTHS):
            current = paid[month]
            prior = [paid[m] for m in MONTHS[:index]]
            last = prior[-1]['labels'] if prior else np.empty(0, dtype=int)
            old_y = np.concatenate([p['labels'] for p in prior]) if prior else np.empty(0, dtype=int)
            for c, device_id in enumerate(classes):
                count_rows.append(dict(repair_seed=seed, month=month,
                    device_id=int(device_id), current_random_500=int((current['labels'][:500] == c).sum()),
                    current_paid_1000=int((current['labels'] == c).sum()),
                    previous_month_paid=int((last == c).sum()),
                    all_previous_paid=int((old_y == c).sum())))
            if not prior:
                continue
            old_x = np.concatenate([p['features'][p['labels'] >= 0] for p in prior])
            old_y = old_y[old_y >= 0]
            anchor_x, anchor_y = current['features'][:500], current['labels'][:500]
            rows = transfer_rows(old_x, old_y, anchor_x, anchor_y, classes)
            transfer.extend(dict(repair_seed=seed, month=month, **row) for row in rows)
            represented = [r for r in rows if r['anchor_support'] > 0 and r['previous_paid_support'] > 0]
            support = sum(r['anchor_support'] for r in represented)
            overview.append(dict(repair_seed=seed, month=month,
                known_anchor=int((anchor_y >= 0).sum()), old_paid=len(old_y),
                represented_anchor_classes=len(represented),
                represented_anchor_rows=support,
                represented_class_macro_top1=float(np.mean([r['top1_old_neighbor_rate'] for r in represented])) if represented else np.nan,
                represented_class_macro_top5=float(np.mean([r['top5_contains_true_rate'] for r in represented])) if represented else np.nan))
        print('seed', seed, 'paid-label feature audit complete', flush=True)
    write_csv(out / 'paid_class_counts.csv', count_rows)
    write_csv(out / 'anchor_transfer.csv', transfer)
    write_csv(out / 'overview.csv', overview)
    counts = pd.DataFrame(count_rows)
    transfers = pd.DataFrame(transfer)
    focus = counts[counts.apply(lambda row: row.device_id in FOCUS.get(row.month, ()), axis=1)]
    focus = focus.merge(transfers, on=['repair_seed', 'month', 'device_id'], how='left')
    cols = ['repair_seed', 'month', 'device_id', 'current_random_500',
            'current_paid_1000', 'previous_month_paid', 'all_previous_paid',
            'top1_old_neighbor_rate', 'top5_contains_true_rate',
            'random_memory_top5_rate', 'top5_over_random_memory',
            'median_true_minus_other_similarity']
    report = '\n'.join([
        'TMA chronological paid-label memory feasibility audit (exploratory).',
        f'Fixed source seed {source_seed}; repair seeds {repair_seeds}; 1000 paid labels/month.',
        'Each month uses the completed hybrid query set; the first 500 paid labels are uniform anchors.',
        'Old memory contains only paid queries from strictly earlier months; no future evaluation file is read.',
        'Cosine-neighbor rates on the current uniform paid anchor are diagnostics, not future-test scores.',
        'For rare classes with only a few anchor labels, these rates are very uncertain.',
        'Focus-class counts and transfer diagnostics:', focus[cols].to_string(index=False),
        'All-class overview:', pd.DataFrame(overview).to_string(index=False),
        'All four months have been inspected during prior work: no independent validation claim.',
    ]) + '\n'
    (out / 'REPORT.txt').write_text(report)
    complete(out, signature)
    print(report, flush=True)
    print('TEMPORAL MEMORY AUDIT COMPLETE:', out, flush=True)
    return out


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
    run(args.root.resolve(strict=True), args.source_seed, seeds,
        args.device, args.threads)


if __name__ == '__main__':
    main()
