"""Read-only exact-feature overlap check for the retrospective TMA probe."""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'tma_analysis_v1'))
from analyze import Reader

FOCUS = {'2020-03': (5, 22, 26, 37), '2020-04': (20,)}


def keys(x):
    """Exact bytes of the 16 float32 model inputs, excluding time/device IDs."""
    x = np.ascontiguousarray(x, dtype='<f4')
    if x.ndim != 2 or x.shape[1] != 16:
        raise ValueError('Expected the 16 original flow statistics')
    return x.view(np.dtype(f'V{x.shape[1] * 4}')).ravel()


def match_counts(target_keys, target_labels, device_id, known):
    positive = np.flatnonzero(target_labels == device_id)
    any_match = same_class = pure_same_class = 0
    for i in positive:
        labels = known.get(target_keys[i].tobytes())
        if labels is not None:
            any_match += 1
            same_class += int(device_id in labels)
            pure_same_class += int(labels == {device_id})
    return dict(positives=len(positive), exact_match_any=any_match,
                exact_match_same_class=same_class,
                exact_match_pure_same_class=pure_same_class,
                exact_match_fraction=any_match / len(positive) if len(positive) else None)


def run(root):
    cache = Reader(root / 'cache')
    rows = []
    for month, devices in FOCUS.items():
        query = cache.arrays(month + '_query.npz', ['x', 'y', 'times'])
        eval_set = cache.arrays(month + '_eval.npz', ['x', 'y'])
        dates = pd.to_datetime(query['times'], utc=True)
        start = pd.Timestamp(month + '-01', tz='UTC')
        fit = np.asarray((dates >= start) & (dates < start + pd.Timedelta(days=10)))
        hold = np.asarray((dates >= start + pd.Timedelta(days=10)) &
                          (dates < start + pd.Timedelta(days=14)))
        if not (fit | hold).all() or not fit.any() or not hold.any():
            raise ValueError('Unexpected query dates')
        query_keys, eval_keys = keys(query['x']), keys(eval_set['x'])
        known = defaultdict(set)
        for key, label in zip(query_keys[fit], query['y'][fit]):
            known[key.tobytes()].add(int(label))
        for device_id in devices:
            for period, mask in (('query_days_11_14', hold), ('future_days_16_end', None)):
                if mask is None:
                    result = match_counts(eval_keys, eval_set['y'], device_id, known)
                else:
                    result = match_counts(query_keys[mask], query['y'][mask], device_id, known)
                rows.append(dict(month=month, device_id=device_id, period=period,
                                 fit_rows=int(fit.sum()), fit_unique_features=len(known), **result))
    print('Exact 16-feature overlap with fully labeled query days 1-10; diagnostic only.')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    args = parser.parse_args()
    run(args.root)
