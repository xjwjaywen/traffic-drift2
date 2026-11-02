# Donor-level repair under temporal shift: feasibility before method design

Status: research question and source-only coverage audit. No new method has
been validated. The final UNSW window remains unopened.

## What the home07 control established

The 4-of-5 RF agreement route improved one examined seed mainly by refusing
class-14-to-9 overrides. Its macro-F1 change was within the range of random
selection that kept the same number of overrides from each source-predicted
class. Thus row-level agreement has not demonstrated useful ranking within a
donor class. The earlier query pool had 0.57% source-predicted class 14,
versus 18.64% in the development sample. Those are source-prediction shares,
not true-class shares. With only 1,000 paid labels, a rapidly growing donor
may have too few earlier labels to evaluate its proposed overrides.

## Related work and novelty boundary

- BEAM-F directly optimizes multiclass macro F-measure. Merely changing the
  objective from accuracy to macro F1 is established work:
  https://research.ibm.com/publications/optimizing-the-multiclass-f-measure-via-biconcave-programming
- MILD studies two-stage expert deferral when one expert is favored by class
  imbalance. Routing a source model to a challenger is not itself new:
  https://proceedings.mlr.press/v306/cortes26b.html
- High-probability risk control under covariate shift weights calibration
  losses using unlabeled target covariates. A simple shift-weighted gate is
  therefore insufficient as a novelty claim:
  https://proceedings.mlr.press/v266/almeida25a.html
- Policy learning with abstention links returning to a baseline with safe
  improvement and robustness to small shifts:
  https://proceedings.mlr.press/v300/sawarni26a.html
- Overlap-aware disagreement discrepancy uses domain overlap to bound target
  error under assumptions:
  https://proceedings.mlr.press/v286/mishra25a.html
- Without restrictions on the future conditional label distribution,
  labeled earlier data plus unlabeled future data cannot generally identify
  future accuracy, much less guarantee macro-F1 improvement:
  https://arxiv.org/abs/2201.04234

Potentially distinct problem, still unproven: choose a small set of
**source-predicted donor groups** to repair using a nonlinear challenger,
with only sparse paid labels, observable unlabeled future group mass, a
baseline negative-flip constraint, and a nondecomposable macro-F1 objective.
This combination needs a concrete uncertainty set, an algorithm, a proof
under stated temporal assumptions, and independent improvements over the
above methods. A 4-of-5 vote or an arbitrary share-ratio threshold is not
such a contribution.

One mathematical direction is to let `G_s = {x: h_source(x) = s}` and select
group gates `g_s`. Paid labels estimate the conditional confusion outcomes
of challenger proposals in `G_s`. Unlabeled deployment data estimate each
group's mass. A proposed rule would maximize the worst plausible change in
macro F1 over simultaneous uncertainty sets for those outcomes, subject to
a bound on new errors among source-correct rows. If a donor has very few
paid rows, its uncertainty set should be wide and the rule should usually
retain the source prediction. This remains a hypothesis: the uncertainty
set must explicitly allow bounded conditional drift, and its radius cannot
be justified merely from the observed unlabeled share change.

## Feasibility audit before implementing that rule

`donor_coverage_audit.py` examines the fixed 11 homes and five overlapping
paid draws per home. For each source-predicted donor class it records:

1. Its share in the full earlier query reservoir and in the curated
   development window, using only frozen source predictions.
2. The number of already paid rows predicted as that donor, and source
   correct/error counts on those rows.
3. A descriptive flag for at least 100 later rows, at least 4-fold share
   growth, and fewer than 10 paid donor rows. These cutoffs are **audit
   summaries**, not method hyperparameters.

The script reads no unpaid query label, fits no challenger, and does not
inspect the final window. Five seeds within a home are repeated draws; homes
are the independent units. Run from the repository root on the server:

```bash
python -u Experiment/unsw_home_temporal_v1/donor_coverage_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

If rapidly growing donors usually have fewer than ten paid rows across
homes, a donor-level safety certificate based on those labels will likely
be vacuous. Then the next research question is how to allocate paid queries
to emerging groups without sacrificing broad class coverage. If support is
often adequate, test a fully specified group-level rule against always-source,
full RF, random matched donor groups, and simpler paid-support/share guards
across homes before using the untouched final window.
