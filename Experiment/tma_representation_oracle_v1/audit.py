"""Retrospective, fully labeled query-pool audit of TMA class information.

This deliberately spends far more than the paper's 1000-label budget. It is an
abundant-label mechanism check, never a deployable CARE variant or main result.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ('tma_pilot_v1', 'tma_analysis_v1', 'tma_paper_v1',
               'tma_selective_repair_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))
from data import sha, validate_complete, write_csv, write_json, finish
from pilot import forward_batches
from run import context
from trial import load_source

FOCUS = {'2020-03': (5, 22, 26, 37), '2020-04': (20,)}
SEED = 20261031
NEGATIVE_RATIO = 20


def chronological_split(times, month):
    """Use days 1-10 to fit and days 11-14 for query-side checking."""
    dates = pd.to_datetime(times, utc=True)
    start = pd.Timestamp(month + '-01', tz='UTC')
    boundary = start + pd.Timedelta(days=10)
    end = start + pd.Timedelta(days=14)
    if not ((dates >= start) & (dates < end)).all():
        raise ValueError('Query cache contains a timestamp outside days 1-14')
    train = np.asarray(dates < boundary)
    hold = ~train
    if not train.any() or not hold.any():
        raise ValueError('Chronological query split is empty')
    return train, hold


def sampled_train_indices(labels, train_mask, device_id):
    """Fixed class-balanced subset; identical rows for raw and frozen probes."""
    positive = np.flatnonzero(train_mask & (labels == device_id))
    negative = np.flatnonzero(train_mask & (labels != device_id))
    if len(positive) < 10 or len(negative) < 10:
        return None
    rng = np.random.default_rng(SEED + int(device_id))
    negative = rng.choice(negative, size=min(len(negative),
                                            NEGATIVE_RATIO * len(positive)),
                          replace=False)
    return np.sort(np.concatenate((positive, negative)))


def make_probe(kind):
    if kind == 'linear':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=SEED))
    if kind == 'nonlinear':
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0, random_state=SEED)
    raise ValueError(kind)


def score(y, values):
    if len(np.unique(y)) != 2:
        return dict(ap=None, auroc=None, prevalence=float(y.mean()))
    return dict(ap=float(average_precision_score(y, values)),
                auroc=float(roc_auc_score(y, values)),
                prevalence=float(y.mean()))


def run(root, source_seed=0, device='cuda', threads=8):
    reader, cache, classes, source_signature = context(root, source_seed)
    output = root.resolve() / 'representation_oracle_v1' / f'source_{source_seed}'
    if any(path.is_symlink() for path in (output.parent, output)):
        raise ValueError('Output directory must not be a symlink')
    signature = dict(schema=1, source_seed=source_seed, source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'), source_signature=source_signature,
        code_sha256=sha(HERE / 'audit.py'), focus={k: list(v) for k, v in FOCUS.items()},
        train_days='1-10', holdout_days='11-14', eval_days='16-end',
        negative_ratio=NEGATIVE_RATIO, seed=SEED, device=device, threads=threads,
        note='Retrospective full-query-label oracle; exceeds the 1000-label budget.')
    if output.exists() and any(output.iterdir()) and not (output / 'config.json').exists():
        raise ValueError('Refusing unrecognized nonempty output directory')
    output.mkdir(parents=True, exist_ok=True)
    if validate_complete(output, signature):
        print('REPRESENTATION ORACLE AUDIT COMPLETE (verified):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    model, prep = load_source(reader, classes)
    class_indices = {c: i for i, c in enumerate(classes)}
    rows = []
    for month, devices in FOCUS.items():
        query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y', 'times'])
        train_mask, hold_mask = chronological_split(query['times'], month)
        raw_query = prep.transform(query['x'])
        frozen_query = forward_batches(model.encoder, raw_query, device).numpy()
        source_query = softmax(forward_batches(model.head, frozen_query, device).numpy(), axis=1)
        fitted = {}
        for device_id in devices:
            if device_id not in class_indices:
                raise ValueError(f'Unknown source device {device_id}')
            chosen = sampled_train_indices(query['y'], train_mask, device_id)
            if chosen is None:
                rows.append(dict(month=month, device_id=device_id, method='insufficient_train_positives',
                                 train_positives=int((train_mask & (query['y'] == device_id)).sum())))
                continue
            y_train = (query['y'][chosen] == device_id).astype(int)
            y_hold = (query['y'][hold_mask] == device_id).astype(int)
            reference = source_query[hold_mask, class_indices[device_id]]
            rows.append(dict(month=month, device_id=device_id, method='source',
                train_positives=int(y_train.sum()), train_negatives=int((1-y_train).sum()),
                hold_positives=int(y_hold.sum()), hold_rows=int(len(y_hold)),
                **{f'hold_{k}': v for k, v in score(y_hold, reference).items()}))
            for space, x in (('raw', raw_query), ('frozen', frozen_query)):
                for kind in ('linear', 'nonlinear'):
                    name = f'{space}_{kind}'
                    probe = make_probe(kind)
                    probe.fit(x[chosen], y_train)
                    fitted[device_id, name] = probe
                    predicted = probe.predict_proba(x[hold_mask])[:, 1]
                    rows.append(dict(month=month, device_id=device_id, method=name,
                        train_positives=int(y_train.sum()), train_negatives=int((1-y_train).sum()),
                        hold_positives=int(y_hold.sum()), hold_rows=int(len(y_hold)),
                        **{f'hold_{k}': v for k, v in score(y_hold, predicted).items()}))
            print(month, device_id, 'query-only probes fitted; opening future evaluation', flush=True)
        # Candidate models and settings are fixed before the future labels are opened.
        target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
        raw_eval = prep.transform(target['x'])
        frozen_eval = forward_batches(model.encoder, raw_eval, device).numpy()
        source_eval = softmax(forward_batches(model.head, frozen_eval, device).numpy(), axis=1)
        for row in rows:
            if row['month'] != month or row['method'] == 'insufficient_train_positives':
                continue
            device_id, name = row['device_id'], row['method']
            y_eval = (target['y'] == device_id).astype(int)
            if name == 'source':
                predicted = source_eval[:, class_indices[device_id]]
            else:
                x = raw_eval if name.startswith('raw_') else frozen_eval
                predicted = fitted[device_id, name].predict_proba(x)[:, 1]
            row.update(eval_positives=int(y_eval.sum()), eval_rows=int(len(y_eval)),
                       **{f'eval_{k}': v for k, v in score(y_eval, predicted).items()})
        print(month, 'complete', flush=True)
    write_csv(output / 'scores.csv', rows)
    (output / 'REPORT.txt').write_text(
        'Retrospective full-query-label information audit, NOT a 1000-label method result.\n'
        'Focus devices were chosen after looking at TMA outcomes; months already examined.\n'
        'Days 1-10 fit binary probes; days 11-14 provide query-side checks; days 16-end evaluation.\n'
        'All raw/frozen probes use the same sampled training IDs and fixed hyperparameters.\n'
        'Compare AP with eval prevalence; high frozen AP suggests information is present, while a raw/frozen gap suggests possible representation loss.\n'
        'A low AP in both spaces does not prove intrinsic impossibility. High binary AP does not guarantee multiclass repair.\n'
        'Binary AP does not equal multiclass F1. Raw/frozen gaps are suggestive, not causal proof.\n'
        'No inference or paper claim may use query-pool labels beyond the paid budget.\n\n' +
        pd.DataFrame(rows).to_string(index=False) + '\n')
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed))
    finish(output, signature, ['scores.csv', 'REPORT.txt', 'inputs.json'])
    print('REPRESENTATION ORACLE AUDIT COMPLETE:', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    run(args.root, args.source_seed, args.device, args.threads)
