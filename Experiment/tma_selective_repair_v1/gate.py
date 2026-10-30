"""Label-budget-preserving, cross-fitted gate for a repaired classifier.

The gate is keyed by the *source prediction*, which is observable at deployment.
Only paid query labels and out-of-fold repair predictions enter its decision.
"""
from math import comb

import numpy as np


def fold_assignments(n, folds, seed):
    if folds < 2 or n < folds:
        raise ValueError('Need at least two nonempty folds')
    assignment = np.empty(n, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(n)
    assignment[order] = np.arange(n) % folds
    return assignment


def probability_of_net_gain(gains, losses):
    """P(p_gain > .5) under a uniform Beta prior on discordant outcomes."""
    if gains < 0 or losses < 0:
        raise ValueError('Counts must be nonnegative')
    trials = int(gains + losses + 1)
    return sum(comb(trials, k) for k in range(int(gains) + 1)) / (2 ** trials)


def learn_gate(labels, source_pred, out_of_fold_repair_pred, n_classes,
               min_support=10, min_discordant=3, probability_threshold=.9):
    """Choose source or CARE for each source-predicted class using paid labels.

    Unknown labels are -1 and count against the query budget but provide no
    closed-set correctness observation. Nothing in this function sees future
    evaluation labels or predictions.
    """
    labels = np.asarray(labels, dtype=np.int64)
    source_pred = np.asarray(source_pred, dtype=np.int64)
    out_of_fold_repair_pred = np.asarray(out_of_fold_repair_pred, dtype=np.int64)
    if labels.shape != source_pred.shape or labels.shape != out_of_fold_repair_pred.shape:
        raise ValueError('Paid-label and prediction arrays must align')
    if np.any((source_pred < 0) | (source_pred >= n_classes)) or np.any(
            (out_of_fold_repair_pred < 0) | (out_of_fold_repair_pred >= n_classes)):
        raise ValueError('Predictions outside fixed source class space')
    if min_support < 1 or min_discordant < 1 or not .5 < probability_threshold < 1:
        raise ValueError('Invalid fixed gate parameters')
    known = (labels >= 0) & (labels < n_classes)
    use_repair = np.zeros(n_classes, dtype=bool)
    rows = []
    for c in range(n_classes):
        mask = known & (source_pred == c)
        support = int(mask.sum())
        source_correct = source_pred[mask] == labels[mask]
        repair_correct = out_of_fold_repair_pred[mask] == labels[mask]
        gains = int((~source_correct & repair_correct).sum())
        losses = int((source_correct & ~repair_correct).sum())
        probability = probability_of_net_gain(gains, losses)
        selected = (support >= min_support and gains + losses >= min_discordant
                    and probability > probability_threshold)
        use_repair[c] = selected
        rows.append(dict(source_predicted_class=c, paid_known_support=support,
                         gain_count=gains, loss_count=losses,
                         posterior_net_gain_probability=probability,
                         use_repair=bool(selected)))
    return use_repair, rows


def route(source_pred, repaired_pred, use_repair):
    source_pred = np.asarray(source_pred, dtype=np.int64)
    repaired_pred = np.asarray(repaired_pred, dtype=np.int64)
    use_repair = np.asarray(use_repair, dtype=bool)
    if source_pred.shape != repaired_pred.shape:
        raise ValueError('Prediction arrays must align')
    if np.any((source_pred < 0) | (source_pred >= len(use_repair))):
        raise ValueError('Source prediction outside gate class space')
    return np.where(use_repair[source_pred], repaired_pred, source_pred)
