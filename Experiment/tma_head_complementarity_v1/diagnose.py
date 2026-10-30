"""Read-only, label-oracle diagnosis of target-only versus CARE complementarity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tma_analysis_v1'))
from analyze import Reader

METHODS = ('hybrid_target', 'hybrid_care')


def macro_f1_present(y, pred, n_classes):
    known = y >= 0
    if not known.any():
        return np.nan
    matrix = np.bincount(y[known] * n_classes + pred[known],
                         minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix)
    f1 = np.divide(2 * tp, support + predicted,
                   out=np.zeros(n_classes, dtype=float), where=(support + predicted) > 0)
    return float(f1[support > 0].mean())


def diagnose_month(y, target, care, classes, seed, month):
    y, target, care = [np.asarray(a) for a in (y, target, care)]
    n_classes = len(classes)
    if not (len(y) == len(target) == len(care)) or not np.isin(
            np.r_[target, care], np.arange(n_classes)).all():
        raise ValueError('Prediction lengths or class indices invalid')
    disagreement = (target != care) & (y >= 0)
    rows = []
    choose_target = np.zeros(n_classes, dtype=bool)
    for c in range(n_classes):
        mask = disagreement & (target == c)
        t_win = int((target[mask] == y[mask]).sum())
        c_win = int((care[mask] == y[mask]).sum())
        choose_target[c] = t_win > c_win
        rows.append(dict(repair_seed=seed, month=month, target_predicted_device=classes[c],
                         disagreement_rows=int(mask.sum()), target_correct=t_win,
                         care_correct=c_win, net_target_correct=t_win - c_win,
                         oracle_choose_target=bool(choose_target[c])))
    routed = np.where(choose_target[target], target, care)
    score = dict(repair_seed=seed, month=month, known_rows=int((y >= 0).sum()),
                 disagreement_rows=int(disagreement.sum()),
                 target_macro_f1=macro_f1_present(y, target, n_classes),
                 care_macro_f1=macro_f1_present(y, care, n_classes),
                 oracle_routed_macro_f1=macro_f1_present(y, routed, n_classes),
                 oracle_target_class_count=int(choose_target.sum()))
    score['oracle_minus_care_pp'] = 100 * (score['oracle_routed_macro_f1'] -
                                          score['care_macro_f1'])
    return score, rows


def run(root, source_seed=0, seeds=(0, 1, 2, 3, 4)):
    scores, class_rows = [], []
    for seed in seeds:
        result = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' / f'repair_{seed}')
        signature = result.marker['signature']
        if signature['source_seed'] != source_seed or signature['repair_seed'] != seed:
            raise ValueError('Wrong completed hybrid run: ' + str(result.root))
        for month in signature['months']:
            classes = json.loads(result.file(month + '/decision.json').read_text())['classes']
            pred = result.arrays(month + '/predictions.npz',
                                 ['labels', 'hybrid_target', 'hybrid_care'])
            score, rows = diagnose_month(pred['labels'], pred['hybrid_target'],
                                         pred['hybrid_care'], classes, seed, month)
            scores.append(score)
            class_rows.extend(rows)
    return pd.DataFrame(scores), pd.DataFrame(class_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        seeds = sorted(set(int(item) for item in args.repair_seeds.split(',')))
        if not seeds or min(seeds) < 0 or args.source_seed < 0:
            raise ValueError()
    except ValueError:
        parser.error('repair-seeds must be comma-separated nonnegative integers')
    root = args.root.resolve()
    output = args.output or root / 'head_complementarity_v1' / f'source_{args.source_seed}'
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error('Output directory must be new and empty: ' + str(output))
    scores, classes = run(root, args.source_seed, seeds)
    output.mkdir(parents=True)
    scores.to_csv(output / 'oracle_scores.csv', index=False)
    classes.to_csv(output / 'predicted_class_disagreements.csv', index=False)
    summary = scores.groupby('month')[['target_macro_f1', 'care_macro_f1',
                                       'oracle_routed_macro_f1', 'oracle_minus_care_pp']].agg(
                                           ['mean', 'std', 'count'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index().to_csv(output / 'oracle_mean_sd.csv', index=False)
    useful = classes[(classes.disagreement_rows >= 20) & (classes.net_target_correct > 0)]
    report = '\n'.join([
        'TMA target-only vs CARE complementarity; read-only LABEL-ORACLE diagnostic.',
        'Future evaluation labels choose target or CARE for each target-predicted class.',
        'This routed score is deliberately contaminated by test labels and is NOT a deployable result.',
        summary[['oracle_minus_care_pp_mean', 'oracle_minus_care_pp_std']].to_string(),
        'Target-predicted classes with >=20 disagreements and positive target net corrections:',
        useful[['repair_seed', 'month', 'target_predicted_device', 'disagreement_rows',
                'target_correct', 'care_correct', 'net_target_correct']].to_string(index=False)
        if len(useful) else 'none',
        'The oracle maximizes classwise correct counts, not macro-F1; its gap is only diagnostic.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    print(report, flush=True)
    print('DIAGNOSIS COMPLETE:', output, flush=True)


if __name__ == '__main__':
    main()
