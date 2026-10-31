"""Read-only, label-oracle diagnosis of frozen RF versus hybrid CARE errors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'tma_analysis_v1'))
from analyze import Reader


def macro_f1_present(y, pred, n_classes):
    keep = y >= 0
    matrix = np.bincount(y[keep] * n_classes + pred[keep],
                         minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(n_classes),
                   where=support + predicted > 0)
    return float(f1[support > 0].mean()) if (support > 0).any() else np.nan


def diagnose(y, source, care, rf, classes, paid_support, month, seed):
    y, source, care, rf = [np.asarray(a) for a in (y, source, care, rf)]
    n = len(classes)
    if not (len(y) == len(source) == len(care) == len(rf)):
        raise ValueError('Prediction/label lengths differ')
    if not np.isin(np.r_[source, care, rf], np.arange(n)).all():
        raise ValueError('Prediction class index outside source head')
    known = y >= 0
    disagree = known & (rf != care)
    source_correct = known & (source == y)
    rows, use_rf = [], np.zeros(n, dtype=bool)
    for c, device_id in enumerate(classes):
        mask = disagree & (rf == c)
        rf_win = int((rf[mask] == y[mask]).sum())
        care_win = int((care[mask] == y[mask]).sum())
        use_rf[c] = rf_win > care_win
        rows.append(dict(month=month, repair_seed=seed, rf_predicted_device=device_id,
            paid_true_support=int(paid_support.get(device_id, 0)),
            disagreement_rows=int(mask.sum()), rf_correct=rf_win,
            care_correct=care_win, net_rf_correct=rf_win - care_win,
            oracle_choose_rf=bool(use_rf[c])))
    routed = np.where(use_rf[rf], rf, care)
    score = dict(month=month, repair_seed=seed,
        known_rows=int(known.sum()), disagreement_rows=int(disagree.sum()),
        rf_only_correct=int((disagree & (rf == y)).sum()),
        care_only_correct=int((disagree & (care == y)).sum()),
        added_source_flips=int((source_correct & (rf != y) & (care == y)).sum()),
        avoided_source_flips=int((source_correct & (rf == y) & (care != y)).sum()),
        care_macro_f1=macro_f1_present(y, care, n),
        rf_macro_f1=macro_f1_present(y, rf, n),
        oracle_routed_macro_f1=macro_f1_present(y, routed, n),
        oracle_rf_class_count=int(use_rf.sum()))
    score['oracle_minus_care_pp'] = 100 * (score['oracle_routed_macro_f1'] -
                                           score['care_macro_f1'])
    return score, rows


def run(root, source_seed=0):
    result = Reader(root / 'nonlinear_head_control_v1' / f'source_{source_seed}')
    signature = result.marker['signature']
    if signature['source_seed'] != source_seed:
        raise ValueError('Nonlinear control source seed changed')
    cache = Reader(root / 'cache')
    scores, class_rows = [], []
    for month in signature['months']:
        target_ids = cache.arrays(month + '_eval.npz', ['ids'])['ids']
        for seed in signature['repair_seeds']:
            hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                            f'repair_{seed}')
            if (hybrid.marker['signature']['repair_seed'] != seed or
                hybrid.marker['signature']['source_seed'] != source_seed):
                raise ValueError('Hybrid run source or repair seed changed')
            classes = json.loads(hybrid.file(month + '/decision.json').read_text())['classes']
            base = hybrid.arrays(month + '/predictions.npz',
                ['ids', 'labels', 'static_mlp', 'hybrid_care'])
            if not np.array_equal(base['ids'], target_ids):
                raise ValueError('Hybrid predictions and target IDs differ')
            with np.load(result.file(month + f'/repair_{seed}/predictions.npz'),
                         allow_pickle=False) as file:
                rf = file['frozen_rf']
            paid = pd.read_csv(hybrid.file(month + '/hybrid_queries.csv'))
            support = paid.device_id.value_counts().to_dict()
            score, rows = diagnose(base['labels'], base['static_mlp'],
                base['hybrid_care'], rf, classes, support, month, seed)
            scores.append(score)
            class_rows.extend(rows)
    return pd.DataFrame(scores), pd.DataFrame(class_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    args = parser.parse_args()
    scores, class_rows = run(args.root, args.source_seed)
    output = args.root.resolve() / 'rf_complementarity_v1' / f'source_{args.source_seed}'
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error('Output directory must be new and empty: ' + str(output))
    output.mkdir(parents=True)
    scores.to_csv(output / 'scores.csv', index=False)
    class_rows.to_csv(output / 'rf_predicted_class_disagreements.csv', index=False)
    means = scores.groupby('month')[['rf_only_correct', 'care_only_correct',
        'added_source_flips', 'avoided_source_flips',
        'oracle_minus_care_pp']].mean().reset_index()
    focus = class_rows[((class_rows.month == '2020-03') &
                        class_rows.rf_predicted_device.isin([5, 22, 26, 37])) |
                       ((class_rows.month == '2020-04') &
                        (class_rows.rf_predicted_device == 20))]
    focus = focus.groupby(['month', 'rf_predicted_device'])[[
        'paid_true_support', 'disagreement_rows', 'net_rf_correct']].mean().reset_index()
    report = '\n'.join([
        'Read-only frozen-RF versus CARE complementarity audit.',
        'Future labels are used to evaluate and to choose the oracle route; oracle scores are contaminated.',
        'No routed score here is a valid deployable method result.',
        'Across-seed means:', means.to_string(index=False),
        'Focus RF-predicted classes:', focus.to_string(index=False),
        'See per-seed and all-class CSVs before making a routing hypothesis.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    print(report, flush=True)
    print('RF COMPLEMENTARITY AUDIT COMPLETE:', output, flush=True)


if __name__ == '__main__':
    main()
