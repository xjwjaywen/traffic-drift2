"""Can a small, representative paid-label audit predict which repair helps?

This is a diagnostic, not a deployable class router. All decisions use the
uniform half of an existing hybrid query for assessment; the other half may
train heads. Future evaluation labels are opened only after decisions are saved.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ('tma_pilot_v1', 'tma_analysis_v1', 'tma_paper_v1',
               'tma_selective_repair_v1', 'tma_hybrid_adaptive_v1'):
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader, class_stats, confusion
from data import MONTHS, sha, write_csv, write_json
from gate import fold_assignments
from pilot import forward_batches, mapped, stats
from run import (PROTOCOL, check_predictions, complete, context, existing_output,
                 fit_paper, metrics, output_directory, verify_used_inputs)
from trial import load_references, load_source

METHODS = ('source', 'bias_only', 'target_only', 'care')
FOLDS = 5
MIN_CLASS_SUPPORT = 10
MIN_F1_GAIN = .05
BIAS_L2 = .1


def code_identity():
    files = [p for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    files += [EXPERIMENT / 'tma_selective_repair_v1' / name
              for name in ('trial.py', 'gate.py')]
    files += [EXPERIMENT / 'tma_paper_v1' / name
              for name in ('run.py', 'paper_engine.py')]
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in files}


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'repairability_audit_v1'
    source = base / f'source_{source_seed}'
    result = source / f'repair_{repair_seed}'
    if any(path.is_symlink() for path in (base, source, result)):
        raise ValueError('Output directories must not be symlinks')
    return result


def paid_indices(hybrid, paper, month, query, budget):
    paid = pd.read_csv(hybrid.file(month + '/hybrid_queries.csv'),
                       dtype={'record_id': str})
    uniform = pd.read_csv(paper.file(month + '/random_queries.csv'),
                          dtype={'record_id': str})
    half = budget // 2
    if len(paid) != budget or len(uniform) != budget or half < FOLDS:
        raise ValueError('Saved query budget changed or insufficient')
    if paid.record_id.iloc[:half].tolist() != uniform.record_id.iloc[:half].tolist():
        raise ValueError('Hybrid prefix is not the original uniform anchor')
    lookup = {str(key): i for i, key in enumerate(query['ids'])}
    if len(lookup) != len(query['ids']):
        raise ValueError('Duplicate query-pool identities')
    try:
        selected = np.asarray([lookup[key] for key in paid.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Paid query absent from cache') from exc
    if len(np.unique(selected)) != budget:
        raise ValueError('Repeated paid query')
    labels = query['y'][selected]  # Only paid labels are used.
    if not np.array_equal(labels, paid.device_id.to_numpy()):
        raise ValueError('Saved paid labels differ from cache')
    return selected, labels, half


def fit_bias(logits, labels, n_classes, l2=BIAS_L2):
    """Regularized intercept-only calibration on known uniform labels."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    keep = (labels >= 0) & (labels < n_classes)
    if logits.ndim != 2 or logits.shape != (len(labels), n_classes) or not keep.any():
        raise ValueError('Invalid or empty bias-calibration data')
    if not np.isfinite(logits).all() or l2 <= 0:
        raise ValueError('Non-finite logits or invalid bias penalty')
    x, y = logits[keep], labels[keep]

    def objective(bias):
        scores = x + bias
        probabilities = softmax(scores, axis=1)
        loss = (logsumexp(scores, axis=1) - scores[np.arange(len(y)), y]).mean()
        loss += .5 * l2 * np.dot(bias, bias)
        gradient = probabilities.mean(axis=0)
        gradient -= np.bincount(y, minlength=n_classes) / len(y)
        gradient += l2 * bias
        return loss, gradient

    result = minimize(objective, np.zeros(n_classes), jac=True, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Bias-only optimization failed: ' + result.message)
    return result.x


def class_scores(labels, predictions, n_classes):
    result = {}
    for name in METHODS:
        matrix = confusion(labels, predictions[name], n_classes)
        support, recall, f1 = class_stats(matrix)
        f1 = np.where(support > 0, f1, np.nan)
        result[name] = dict(support=support, recall=recall, f1=f1)
    return result


def choose_classes(oof_scores, classes):
    """True-class choice is diagnostic only; true class is unknown at inference."""
    rows = []
    for index, device_id in enumerate(classes):
        support = int(oof_scores['care']['support'][index])
        if support < MIN_CLASS_SUPPORT:
            choice, reason = 'care', 'insufficient_uniform_labels'
        else:
            values = {name: float(oof_scores[name]['f1'][index]) for name in METHODS}
            best = max(METHODS, key=lambda name: (values[name], name == 'care'))
            if values[best] >= values['care'] + MIN_F1_GAIN:
                choice, reason = best, 'paid_oof_f1_gain'
            else:
                choice, reason = 'care', 'no_clear_gain_over_care'
        rows.append(dict(device_id=int(device_id), class_index=index,
                         uniform_true_support=support, chosen_method=choice,
                         decision_reason=reason, eligible=support >= MIN_CLASS_SUPPORT,
                         oof_chosen_f1=float(oof_scores[choice]['f1'][index])
                         if support else None,
                         oof_care_f1=float(oof_scores['care']['f1'][index])
                         if support else None))
    return rows


def audit_one_month(root, reader, cache, paper, hybrid, classes, model, prep,
                    rfeat, ry, teacher, historical, month, repair_seed,
                    device, output, signature):
    part = output / month
    month_signature = dict(signature, month=month)
    if existing_output(part, month_signature):
        verify_used_inputs(part, pilot=reader, cache=cache, paper=paper, hybrid=hybrid)
        print(month, 'verified completed month reused', flush=True)
        return
    write_json(part / 'config.json', month_signature)
    query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
    budget = reader.marker['signature']['budget']
    selected, paid_device_ids, half = paid_indices(hybrid, paper, month, query, budget)
    paid_y = mapped(paid_device_ids, classes)
    qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
    qlogits = forward_batches(model.head, qfeat.numpy(), device).numpy()
    folds = fold_assignments(half, FOLDS, 20261030 + repair_seed)
    anchor_y = paid_y[:half]
    oof = {name: np.full(half, -1, dtype=np.int64) for name in METHODS}
    oof['source'] = qlogits[selected[:half]].argmax(1)
    traces = []
    for fold in range(FOLDS):
        held = folds == fold
        train = np.ones(budget, dtype=bool)
        train[:half][held] = False
        known_train = train & (paid_y >= 0)
        bias = fit_bias(qlogits[selected[:half][~held]], anchor_y[~held], len(classes))
        oof['bias_only'][held] = (qlogits[selected[:half][held]] + bias).argmax(1)
        for name, care in (('target_only', False), ('care', True)):
            head, trace = fit_paper(model.head, qfeat[selected[known_train]],
                                    paid_y[known_train], rfeat, ry, teacher,
                                    care, repair_seed + 1000 + fold, device, PROTOCOL)
            oof[name][held] = forward_batches(head,
                qfeat[selected[:half][held]].numpy(), device).argmax(1).numpy()
            traces.append(dict(fold=fold, method=name, uniform_held=int(held.sum()),
                               paid_train=int(train.sum()), known_train=int(known_train.sum()),
                               **trace))
    if any(np.any(pred < 0) for pred in oof.values()):
        raise ValueError('Missing out-of-fold prediction')
    scores = class_scores(anchor_y, oof, len(classes))
    choices = choose_classes(scores, classes)
    bias = fit_bias(qlogits[selected[:half]], anchor_y, len(classes))
    np.save(part / 'bias_only.npy', bias)
    np.savez_compressed(part / 'oof_predictions.npz', ids=query['ids'][selected[:half]],
                        labels=anchor_y, fold=folds, **oof)
    write_json(part / 'oof_training.json', traces)
    write_csv(part / 'choices.csv', choices)
    write_json(part / 'decision.json', dict(month=month, methods=METHODS,
        uniform_anchor=half, active_training_labels=budget - half,
        known_paid_labels=int((paid_y >= 0).sum()), folds=FOLDS,
        min_class_support=MIN_CLASS_SUPPORT, min_f1_gain=MIN_F1_GAIN,
        bias_l2=BIAS_L2, class_choices=choices,
        note='True-class choices are diagnostic; true class is unavailable for routing at inference.'))
    print(month, 'paid-label decisions frozen; reading future evaluation now', flush=True)

    # These future labels and predictions are opened only after decision.json.
    frozen = json.loads((part / 'decision.json').read_text())
    if frozen['class_choices'] != choices:
        raise ValueError('Saved decisions changed')
    future = hybrid.arrays(month + '/predictions.npz',
        ['ids', 'labels', 'device_ids', 'static_mlp', 'hybrid_target', 'hybrid_care'])
    target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
    y = check_predictions(future, target, classes)
    eval_features = forward_batches(model.encoder, prep.transform(target['x']), device)
    eval_logits = forward_batches(model.head, eval_features.numpy(), device).numpy()
    eval_predictions = dict(source=future['static_mlp'],
        bias_only=(eval_logits + bias).argmax(1),
        target_only=future['hybrid_target'], care=future['hybrid_care'])
    if not np.array_equal(eval_predictions['source'], eval_logits.argmax(1)):
        raise ValueError('Reloaded source prediction differs from saved hybrid run')
    eval_scores = class_scores(y, eval_predictions, len(classes))
    oof_rows, eval_rows, outcome_rows = [], [], []
    for index, device_id in enumerate(classes):
        choice = choices[index]
        future_support = int(eval_scores['care']['support'][index])
        best_eval = max(METHODS, key=lambda name: (
            float(eval_scores[name]['f1'][index]) if future_support else -1,
            name == 'care'))
        selected_name = choice['chosen_method']
        outcome_rows.append(dict(month=month, device_id=int(device_id),
            uniform_true_support=choice['uniform_true_support'],
            future_true_support=future_support, eligible=choice['eligible'],
            chosen_method=selected_name, decision_reason=choice['decision_reason'],
            future_best_method=best_eval if future_support else None,
            future_chosen_f1=float(eval_scores[selected_name]['f1'][index])
                if future_support else None,
            future_care_f1=float(eval_scores['care']['f1'][index])
                if future_support else None,
            future_oracle_f1=float(eval_scores[best_eval]['f1'][index])
                if future_support else None,
            future_regret_pp=100 * (float(eval_scores[best_eval]['f1'][index]) -
                float(eval_scores[selected_name]['f1'][index])) if future_support else None,
            future_gain_over_care_pp=100 * (float(eval_scores[selected_name]['f1'][index]) -
                float(eval_scores['care']['f1'][index])) if future_support else None))
        for name in METHODS:
            oof_rows.append(dict(month=month, device_id=int(device_id), method=name,
                true_support=int(scores[name]['support'][index]),
                recall=float(scores[name]['recall'][index])
                    if scores[name]['support'][index] else None,
                f1=float(scores[name]['f1'][index])
                    if scores[name]['support'][index] else None))
            eval_rows.append(dict(month=month, device_id=int(device_id), method=name,
                true_support=future_support,
                recall=float(eval_scores[name]['recall'][index]) if future_support else None,
                f1=float(eval_scores[name]['f1'][index]) if future_support else None))
    write_csv(part / 'oof_class_scores.csv', oof_rows)
    write_csv(part / 'future_class_scores.csv', eval_rows)
    write_csv(part / 'choice_outcomes.csv', outcome_rows)
    oof_summary = []
    for name in METHODS:
        score, _, _ = metrics(anchor_y, oof['source'], oof[name],
                               historical, classes)
        oof_summary.append(dict(month=month, method=name, **score))
    write_csv(part / 'oof_global.csv', oof_summary)
    future_summary = []
    for name in METHODS:
        score, _, _ = metrics(y, eval_predictions['source'],
                               eval_predictions[name], historical, classes)
        future_summary.append(dict(month=month, method=name, **score))
    write_csv(part / 'future_global.csv', future_summary)
    np.savez_compressed(part / 'future_predictions.npz', ids=future['ids'],
                        labels=y, **eval_predictions)
    write_json(part / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
               paper=paper.consumed, hybrid=hybrid.consumed))
    complete(part, month_signature)
    print(month, 'complete', flush=True)


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8,
        months=None):
    months = list(MONTHS if months is None else months)
    if not months or not set(months).issubset(MONTHS):
        raise ValueError('Unsupported month')
    reader, cache, classes, _ = context(root, source_seed)
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' /
                    f'repair_{repair_seed}')
    for name, bundle in (('paper', paper), ('hybrid', hybrid)):
        signature = bundle.marker['signature']
        if (signature['source_seed'] != source_seed or
                signature['repair_seed'] != repair_seed or
                signature['source_marker'] != sha(reader.root / 'complete.json') or
                signature['cache_marker'] != sha(cache.root / 'complete.json')):
            raise ValueError(name + ' run does not match source/cache/seeds')
    if (paper.marker['signature']['protocol'] != PROTOCOL or
            hybrid.marker['signature']['protocol'] != PROTOCOL or
            hybrid.marker['signature']['paper_marker'] !=
            sha(paper.root / 'complete.json') or
            not set(months).issubset(hybrid.marker['signature']['months'])):
        raise ValueError('Existing head protocol or completed months differ')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'),
        code=code_identity(), months=months, protocol=PROTOCOL,
        folds=FOLDS, min_class_support=MIN_CLASS_SUPPORT,
        min_f1_gain=MIN_F1_GAIN, bias_l2=BIAS_L2,
        device=device, threads=threads)
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper,
                           hybrid=hybrid)
        print('REPAIRABILITY AUDIT COMPLETE (verified existing run):', output, flush=True)
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
    for month in months:
        audit_one_month(root, reader, cache, paper, hybrid, classes,
            model, prep, rfeat, ry, teacher, historical, month, repair_seed,
            device, output, signature)
    outcomes = pd.concat([pd.read_csv(output / month / 'choice_outcomes.csv')
                          for month in months], ignore_index=True)
    oof_global = pd.concat([pd.read_csv(output / month / 'oof_global.csv')
                            for month in months], ignore_index=True)
    scores = pd.concat([pd.read_csv(output / month / 'future_global.csv')
                        for month in months], ignore_index=True)
    outcomes.to_csv(output / 'choice_outcomes.csv', index=False)
    oof_global.to_csv(output / 'oof_global.csv', index=False)
    scores.to_csv(output / 'future_global.csv', index=False)
    eligible = outcomes[outcomes.eligible.astype(str).str.lower().eq('true') &
                        outcomes.future_true_support.gt(0)]
    focus = outcomes[((outcomes.month == '2020-03') &
                      outcomes.device_id.isin([5, 22, 26, 37])) |
                     ((outcomes.month == '2020-04') & (outcomes.device_id == 20))]
    report = '\n'.join([
        'TMA repairability feasibility audit; existing hybrid queries, fixed source, 1000 paid labels/month.',
        'The first 500 uniform labels are the only assessment sample; active 500 train heads only.',
        'Five-fold OOF estimates: each held-out uniform label is excluded from both head and bias training.',
        'Candidate whole-model interventions: source, bias-only, target-only head, CARE head.',
        'Class choices use true class and are DIAGNOSTIC ONLY, not deployable routing performance.',
        f'Eligible class requires >= {MIN_CLASS_SUPPORT} uniform true labels; choose non-CARE only for OOF F1 gain >= {MIN_F1_GAIN:.2f}.',
        'Decision file is written before future evaluation labels/predictions are opened.',
        'Eligible classes with future support: ' + str(len(eligible)) + '/' + str(len(outcomes)),
        'Eligible decision matches future best: ' +
            (str(int((eligible.chosen_method == eligible.future_best_method).sum())) + '/' +
             str(len(eligible)) if len(eligible) else 'none'),
        'Eligible mean future regret vs label-oracle best (pp): ' +
            (f'{eligible.future_regret_pp.mean():.3f}' if len(eligible) else 'n/a'),
        'Focus collapse-class choices:',
        focus[['month', 'device_id', 'uniform_true_support', 'chosen_method',
               'future_best_method', 'future_regret_pp',
               'future_gain_over_care_pp']].to_string(index=False),
        'Uniform-anchor out-of-fold scores:',
        oof_global[['month', 'method', 'macro_f1_present',
                    'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        'Whole-model future scores (NOT class-choice routing scores):',
        scores[['month', 'method', 'macro_f1_present', 'collapse_f1',
                'negative_flip_rate_vs_static_mlp']].to_string(index=False),
        'Bias-only is regularized supervised logit calibration, not a proof of pure label shift.',
        'OOF heads train on about 900 labels versus 1000 for final heads; the mismatch is reported in traces.',
        'This does not test encoder adaptation. All target months have been seen before: exploratory only.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
               paper=paper.consumed, hybrid=hybrid.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('REPAIRABILITY AUDIT COMPLETE:', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--months', default=','.join(MONTHS))
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('The real-data audit must run on the Linux TMA server')
    if args.source_seed < 0 or args.repair_seed < 0 or args.threads < 1:
        parser.error('Seeds must be nonnegative and threads positive')
    months = [item.strip() for item in args.months.split(',')]
    run(args.root.resolve(strict=True), args.source_seed, args.repair_seed,
        args.device, args.threads, months)


if __name__ == '__main__':
    main()
