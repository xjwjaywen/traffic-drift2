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

This setting overlaps with [active F-measure estimation](https://papers.neurips.cc/paper_files/paper/2010/hash/d7a728a67d909e714c0774e22cb806f2-Abstract.html),
[active testing](https://proceedings.mlr.press/v139/kossen21a.html),
[delayed-label stream evaluation](https://link.springer.com/article/10.1007/s10618-019-00654-y),
and [sequential model confidence sets](https://arxiv.org/abs/2404.18678).
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
