# Closed-set delayed-label repair: falsifiable protocol draft

Status: exploratory research for a possible **separate paper**, not a CARE
manuscript experiment or a validated method. The untouched UNSW final windows
must remain unopened. The development windows have already been examined.

## Population and chronology

- The primary population is the public, official 24-device inventory for each
  home, matching the existing closed-set benchmark. Inventory membership is
  supplied by dataset curation; it is not a feature available to the model.
  An all-flow/open-set experiment is a different task and must be reported
  separately.
- Source model training uses only source days. The ordered query and examined
  development days form a research replay. Each day is predicted before that
  day's requested labels are exposed. Labels from day `t` may affect only
  predictions on later days. Archive day, original row ID, model version,
  query index, and prediction hashes must be saved before label lookup.
- The current archive has 5–9 query days and 4–6 examined development days
  per primary home. A day-capacity audit is required before fixing block size
  and label budget. The final windows remain sealed until method, baselines,
  budgets, and implementation are fixed. Homes, not overlapping seeds, are
  independent units.
- Labels spent on model training and on performance auditing share a fixed
  total budget. Reusing an audit label to fit the **same candidate whose
  audited result chooses the route** creates a selection bias. For the first
  route test, train a challenger before day `t`, audit that frozen challenger
  with day-`t` labels, and apply the route on day `t+1` with the challenger
  still frozen. Later labels can train a new, separately audited version.

## A paired macro-F1 identity

For a fixed evaluation block and class `c`, write `S_c` for its unknown true
support; `P0_c` and `P1_c` for the number of predictions of `c` by the
source and candidate; `T0_c` for source true positives; and

`D_c = sum_i 1{y_i=c} [1{h1(x_i)=c} - 1{h0(x_i)=c}]`.

Then `P0_c`, `P1_c`, and `P1_c-P0_c` are **known exactly without labels**,
while `D_c` is zero on all source/candidate agreement rows. The paired class
F1 change is exactly

`2(T0_c+D_c)/(S_c+P1_c) - 2T0_c/(S_c+P0_c)`.

Use zero when a denominator is zero and average the change over the fixed
24-class inventory. This identity suggests spending some audit labels on
disagreements, while retaining a probability sample of all rows to estimate
`S_c` and `T0_c`. Known prediction counts can replace the noisy estimated
prediction marginals of a full confusion-matrix plug-in estimator. The
inclusion probabilities must remain known and positive for every stratum.
The identity itself is elementary and **is not a novelty claim**.

The first falsification control should cross two label designs (uniform and
agreement/disagreement stratified) with two estimators (sample-estimated and
exact prediction counts), using identical paid samples within each design.
Report per-home paired error and interval behavior; ordinary row bootstrap
is descriptive only. If the exact-count estimator does not materially improve
decision accuracy or label efficiency, stop this branch of the method.

**Completed exploratory result.** With 1,000 extra labels and 55 overlapping
home/seed repeats, the numbers of bootstrap intervals excluding zero were:
uniform with sampled prediction counts 6, uniform with exact counts 8,
disagreement-stratified with sampled counts 12, and disagreement-stratified
with exact counts 15. No resolved direction disagreed with the full
development comparison in this particular replay, but the interval was
descriptive and still contained the full difference only 47–49 times out of
55, depending on the control. The stratified exact-count version left 40/55
decisions unresolved. Five seeds within a home overlap, so they do not supply
55 independent datasets. The extra 2–3 resolutions from exact prediction
counts are too small to justify a method claim.
Accordingly, stop tuning this estimator on the examined development window.
The remaining hypothesis concerns **transport of a paired audit across
time**, not a better calculation of a same-window F1 score.

```bash
python -u Experiment/unsw_home_temporal_v1/f1_exact_count_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_f1_exact_count_audit_v1/REPORT.txt
```

## What a deployable method would still need

At the end of day `t`, an audit estimates a **current-day** gain for a frozen
challenger. A route chosen for day `t+1` needs an explicit assumption or
uncertainty set for changes in `P(Y|X)` between days. Exact prediction
counts on `t+1` reveal class and donor masses but cannot identify whether
the candidate is right. A safety claim is conditional on both audit
uncertainty coverage and a justified temporal-change bound. If the bound is
too wide, the method returns the source model. Negative flips and class
recall must be reported alongside pooled macro F1.

A candidate transport model partitions flows by the **pair** of frozen
source and challenger predictions `z=(h0(x),h1(x))`. Let `q[t,z]` be the
observed fraction of flows in that group and `theta[t,z,c]` the unknown
conditional true-class fraction. Every source/candidate confusion entry is
a sum of `q[t,z]*theta[t,z,c]` over relevant groups. The next day's `q`
is known from unlabeled predictions; its `theta` is not. An audit on day
`t` can construct a confidence set for `theta[t,z]`. Applying it on day
`t+1` additionally requires a **stated** bound such as
`TV(theta[t+1,z],theta[t,z]) <= rho[z]` for each group. A route would be
used only when the worst-case next-day macro-F1 change is positive and a
negative-flip constraint holds across this set. This is a research
hypothesis, not a validated guarantee: `rho[z]` cannot be inferred from
unlabeled group mass alone, and sparse groups may make the decision set
vacuous. Test against source-only, always-CARE, ordinary active testing,
and simpler per-group paid-support gates before claiming value.

This setting overlaps with [active F-measure estimation](https://papers.neurips.cc/paper_files/paper/2010/hash/d7a728a67d909e714c0774e22cb806f2-Abstract.html),
[active testing](https://proceedings.mlr.press/v139/kossen21a.html),
[delayed-label stream evaluation](https://link.springer.com/article/10.1007/s10618-019-00654-y),
and [sequential model confidence sets](https://arxiv.org/abs/2404.18678).
More direct overlaps are [discordant-pair validation of an updated model](https://pmc.ncbi.nlm.nih.gov/articles/PMC10684559/),
[active F1 model evaluation](https://www.nature.com/articles/s41598-024-58633-3.pdf),
and [paired multiclass F1 inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11483486/).
The published discordant-pair method is binary and assumes a baseline
performance and prevalence that transfer to the new period; its authors
identify drift as a limitation. That is a possible research gap, but merely
extending a sampling rule to 24 classes or counting prediction marginals
exactly is not enough to establish novelty or practical value.
A contribution would require an effective method for **paired macro-F1
repair under delayed labels and temporal group shift**, with stronger
evidence than these controls. The novelty boundary must be checked again
against full papers before any paper claim.

## Capacity audit before model training

The next script reads only query and already examined development periods,
counting daily official-device support; it does not fit a model or touch final
labels/features. Daily class shares are an oracle feasibility diagnostic and
must never enter a query or routing rule.

```bash
python -u Experiment/unsw_home_temporal_v1/online_day_feasibility.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_online_day_feasibility_v1/REPORT.txt
```

**Completed capacity result.** Every examined development day has at least
3,001 curated flows, so a 1,000-label daily sample is physically possible.
That would, however, cost 4,000–6,000 labels per home just for development,
well above the earlier 1,000-label **total** repair budget. Many daily
classes are too rare for stable F1 auditing: for example home12 has a median
of 11 present classes per day with fewer than five *expected* positives in
1,000 uniform curated labels. Home07 has only eight or nine active classes
per day, despite having all 24 in its source period. Pooling days can improve
support but reduces the number of sequential decisions. No budget or block
size should be called final until this tradeoff is measured.

The existing CARE heads use 1,000 labels drawn across the **entire query
window**, so evaluating them on earlier query days would leak future labels.
For a first temporal transport diagnostic, use only the already examined
development days. CARE and source predictions for those days were frozen
before their labels were read. Compare each day's full-oracle CARE gain and
its 1,000-label estimate with the following day's full gain. Also measure
how much next-day prediction-pair mass lies in groups barely covered by the
earlier paid sample. This diagnoses whether yesterday's information can
support tomorrow's route; it does not train or validate a new method.

```bash
python -u Experiment/unsw_home_temporal_v1/online_transport_feasibility.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_online_transport_feasibility_v1/REPORT.txt
```

**Completed transport diagnostic.** Across 11 homes, the previous day's
full-label gain had the same sign as the next day's gain in 80.48% of the
overlapping home/seed transitions on average; yesterday's 1,000-label
estimate matched in 69.09%. The prediction-pair mass changed by mean total
variation 0.145, and 8.78% of the next day's mass lay in pair groups with
fewer than five paid labels the previous day. Home07 was extreme: pair-mass
TV 0.356, 28.60% low-coverage mass, and 13.03% mass in pairs wholly absent
the previous day. Its sign-match rate was nevertheless 96%, which could
simply reflect CARE being consistently worse there. Sign agreement is not
itself a gain from routing.

Before designing a new route, score the predeclared, elementary rule
"use CARE tomorrow iff today's paid audit estimates positive gain" against
always-source and always-CARE. The full-today oracle and tomorrow oracle
are ceilings, never deployable results. This script reads only the saved
diagnostic CSV, not the archive or final window:

```bash
python Experiment/unsw_home_temporal_v1/online_transport_decision_review.py
cat /data/xjw/tma-care-data/unsw_online_transport_decision_review_v1/REPORT.txt
```

**Completed simple-route review.** The after-the-fact rule using yesterday's
1,000-label sign yielded mean next-day macro-F1 gain +0.565 percentage points
over source across 11 homes, versus -0.228 for always-CARE. It improved 7/11
homes over source. Full-yesterday-label routing (an oracle unavailable at this
budget) yielded +0.909, while a tomorrow-label oracle yielded +1.124.
Home07's high sign persistence was indeed mostly a repeated warning against
CARE: it was better on none of the examined next days, and the paid-sign
route's gain was -0.016. These are post-hoc exploratory comparisons, and the
daily sign rule spends 3,000–5,000 **additional** labels per seed across the
development transitions. It must be compared against a much cheaper rule:
use only the first day's 1,000 audit labels to select one route for all
following days. Cumulative and two-day-smoothed paid signs serve as
equal-budget controls for daily switching.

```bash
python Experiment/unsw_home_temporal_v1/online_transport_budget_review.py
cat /data/xjw/tma-care-data/unsw_online_transport_budget_review_v1/REPORT.txt
```

**Completed budget comparison.** The daily paid-sign route gained +0.565
macro-F1 percentage points over source, versus +0.241 for the first-day-fixed
route, averaged over homes. Daily switching beat the fixed choice in 6/11
homes, but cost mean 4,818 rather than 1,000 additional labels per seed.
The cumulative and two-day paid-sign controls gained +0.542 and +0.553,
respectively, very close to the daily rule at the same cost. Further
threshold tuning of a **frozen** CARE switch is not a promising main method.

The remaining reason to buy new labels is to **change the candidate model**.
As a standard feasibility control, refit a fresh ordinary CARE head from the
frozen source head after each examined development day, using the original
1,000 paid query labels plus exactly 1,000 selected curated labels from each
elapsed development day. The source encoder, historical references, 500
optimizer steps, and source model remain fixed. Predict the next day and
freeze prediction files before loading full labels for evaluation. Compare
the updated head with the original fixed CARE and source on the same next-day
rows. This is not a new method and costs 3,000–5,000 extra development labels
per home/seed. First run one home as a pipeline check; only then run all
homes. No final-window data is used.

```bash
python -u Experiment/unsw_home_temporal_v1/online_care_update.py predict \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip --home home01
python -u Experiment/unsw_home_temporal_v1/online_care_update.py evaluate \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip --home home01
cat /data/xjw/tma-care-data/unsw_online_care_update_v1/seed_0/evaluation_home01/REPORT.txt
```

**Seed-0 all-home control.** On pooled next-day rows, updated CARE exceeded
fixed CARE in 9/11 homes. The across-home fixed-24-class macro F1 was 0.50368
for source, 0.50432 for fixed CARE, and 0.51121 for updated CARE. The update
therefore gained about 0.69 percentage points over fixed CARE on average.
It still trailed source in home05, home06, home07, home11, and home12. Mean
negative-flip rates were 0.04968 for fixed CARE and 0.05005 for updated CARE;
home07 and home08 increased sharply. This is one paid-label seed after prior
development inspection, so it is feasibility evidence only. Updated CARE
also spends 3,000–5,000 additional labels per home and has no same-budget
new-method comparison yet.

Finish all five paid-label seeds and aggregate the paired changes within
each home. Do not treat seeds as independent homes. The aggregate script
reads only completed evaluation CSVs and does not open the archive:

```bash
for seed in 1 2 3 4; do
  python -u Experiment/unsw_home_temporal_v1/online_care_update.py predict \
    /data/xjw/tma-care-data/unsw_home_temporal/all.zip --seed "$seed" || break
  python -u Experiment/unsw_home_temporal_v1/online_care_update.py evaluate \
    /data/xjw/tma-care-data/unsw_home_temporal/all.zip --seed "$seed" || break
done
python Experiment/unsw_home_temporal_v1/online_care_update_aggregate.py
cat /data/xjw/tma-care-data/unsw_online_care_update_v1/aggregate_0_1_2_3_4/REPORT.txt
```

**Completed five-seed update control.** The mean pooled fixed-24-class
macro-F1 difference, updated CARE minus fixed CARE, was +0.909 percentage
points across the 11 independent homes. It was positive on average in 11/11
homes; updated CARE exceeded source in only 6/11. Mean negative-flip rate
versus source fell from 0.0551 to 0.0509, although home05 and home08 worsened.
This is strong evidence that adding recent paid labels can improve a fixed
CARE head in this replay. It does not identify which CARE loss term matters,
nor does it establish a new method. The five overlapping paid-label seeds
reduce sampling sensitivity but do not create 55 independent test units.

Next, use exactly the same frozen daily paid identities and original query
labels to train a fresh target-only head from the same source head. Keep the
500 optimizer steps, target minibatch stream, and next-day prediction rows
identical. Remove only historical replay CE and KD. Freeze all target-only
predictions before decoding full development labels. Evaluate target-only,
fixed target, and source, then compare the paired updated-target and
updated-CARE scores by home. This checks whether the apparent benefit comes
from continuing adaptation or from CARE's historical constraints. Since
the target-only control uses fewer loss evaluations per step, runtime is not
matched; the comparison isolates the losses under a fixed optimizer-step
budget, not compute cost.

```bash
for seed in 0 1 2 3 4; do
  python -u Experiment/unsw_home_temporal_v1/online_target_update.py predict \
    /data/xjw/tma-care-data/unsw_home_temporal/all.zip --seed "$seed" || break
done
for seed in 0 1 2 3 4; do
  python -u Experiment/unsw_home_temporal_v1/online_target_update.py evaluate \
    /data/xjw/tma-care-data/unsw_home_temporal/all.zip --seed "$seed" || break
done
python Experiment/unsw_home_temporal_v1/online_update_compare.py
cat /data/xjw/tma-care-data/unsw_online_update_compare_v1/REPORT.txt
```

**Completed paired update comparison.** Across the 11 homes, updated CARE
exceeded updated target-only by only +0.125 macro-F1 percentage points on
average and won in 5/11 homes. Home07 alone contributed +6.692 points to
that mean; excluding it, updated CARE trailed target-only by about 0.532
points across the other ten homes. Home07 is not a success for either update:
both remain below source by 5.614 and 12.306 points, respectively. In
contrast, CARE's negative-flip rate was lower than target-only in 10/11
homes, by 3.954 percentage points on average among source-correct rows.
Home08 and home09 show a substantial F1 advantage for target-only while CARE
preserves more source-correct predictions. The fixed CARE versus fixed target
F1 advantage averaged +2.612 points; after repeated paid updates the
advantage nearly vanished. These numbers identify a stability/plasticity
tradeoff, not a generally better F1 learner or a novel method.

Literature check (2026-11-02): positive-congruent/focal-distillation training
already targets negative flips, backward-compatible weight interpolation
already trades new accuracy against prediction regressions, and GEM/mGEM
already expresses old-task preservation through gradient constraints. Thus
simply adding a negative-flip penalty, interpolating heads, or projecting a
gradient is not a defensible novelty claim. Relevant primary sources:
Yan et al., CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html;
Schumann et al., ICML 2023, https://arxiv.org/abs/2301.10546;
Lopez-Paz and Ranzato, NeurIPS 2017, https://arxiv.org/abs/1706.08840;
Fine-Grained Gradient Restriction, arXiv 2024,
https://arxiv.org/abs/2410.00868.

The useful unresolved question is whether a *traffic-drift-specific* update
can exceed the F1/negative-flip frontier of these established baselines
under the same labels and chronological feedback. The current two endpoints
alone cannot answer this. A next experimental gate should compare at least
one standard compatibility baseline (focal distillation or weight
interpolation) and include the source model as a fallback. Do not open the
final window until a rule and all controls are frozen from development.

**Fixed-grid BCWI control prepared.** Schumann et al.'s original formula is
`mixed = alpha * old + (1 - alpha) * new`. Here `old` is the frozen source
linear head and `new` is the daily target-only head refitted from the same
paid labels as the completed run. Both share the frozen encoder and source
initialization. Evaluate the prespecified old-weight grid 0.25, 0.50, 0.75
without choosing an alpha from development outcomes. Alpha 0 and 1 must
exactly reproduce the frozen target-only and source predictions before any
full development label is read. The script reports all grid points, source,
updated target, and updated CARE. Its negative-flip rate divides by the
number of source-correct rows, matching our prior reports; the original BCWI
paper divides by all rows. The ordering of methods within a home is the same
under either denominator. This is an established baseline, not our method.

Run the `predict` stage for all seeds before `evaluate`:

```bash
for seed in 0 1 2 3 4; do
  python -u Experiment/unsw_home_temporal_v1/online_bcwi_control.py predict \
    /data/xjw/tma-care-data/unsw_home_temporal/all.zip --seed "$seed" || break
done
for seed in 0 1 2 3 4; do
  python -u Experiment/unsw_home_temporal_v1/online_bcwi_control.py evaluate \
    /data/xjw/tma-care-data/unsw_home_temporal/all.zip --seed "$seed" || break
done
python Experiment/unsw_home_temporal_v1/online_bcwi_control.py aggregate
cat /data/xjw/tma-care-data/unsw_bcwi_v1/aggregate_0_1_2_3_4/REPORT.txt
```
