# Online macro-F1 repair with delayed labels: research proposal

Status: a **candidate research problem**, not a validated algorithm or a
novelty claim. This is separate from the current CARE manuscript. No final
UNSW window has been opened.

## Why the static protocol is insufficient

Across 11 UNSW homes, sparse paid coverage of rapidly growing
source-predicted groups was concentrated in home07: 51.30% of its curated
development predictions lay in groups with at least four-fold share growth
and fewer than ten earlier paid donor rows. The corresponding share was at
most 4.52% in any other home. A fixed early-label gate is therefore neither
a reliable home07 rescue nor a general cross-home method. In home07, a donor
with zero or three early labels can become a large part of the next period.
Unlabeled mass reveals that it grew; it does not reveal whether the
challenger is right. Under unrestricted changes to `P(Y|X)`, no label-free
rule can guarantee future macro-F1 improvement.

The next protocol must allow labels **after** predicting on a new block.
Each prediction is scored before its label is requested or revealed. A label
may affect only later blocks. A fixed global label budget is shared by all
methods, including any pilot or audit labels.

## Proposed mathematical target

At block `t`, let `h0` be the frozen source model and `h1,t` a challenger
trained only on labels obtained before `t`. Define a small, predeclared
family of donor-group routes `G`; every `g in G` either uses `h1,t(x)` for
some source-predicted groups or returns `h0(x)`. The source-only route is
always in `G`.

Before obtaining block-`t` labels, partition the unlabeled block by
observable source/challenger predictions. Allocate the whole current-block
budget using only earlier labels and current unlabeled group masses; retain
a uniform floor for broad coverage. Then draw a randomized, stratified
sample with known inclusion probability `p_i > 0`. A separate adaptive
within-block design would require a different sequential estimator and is
not covered by the formula below. Avoid a deterministic BADGE-style audit
sample because its unweighted class frequencies are not an unbiased estimate
of the block.

For each route `g`, estimate its finite-block confusion matrix using the
same paid labels and inverse inclusion probabilities:

`C_hat_g[a,b] = (1/N_t) sum_(i labeled) 1{y_i=a, g(x_i)=b} / p_i`.

This is unbiased for the **current block** under the stated sampling design;
it does not by itself estimate the next block. Compute macro F1 from the
matrix as the mean over a fixed 24-class inventory of
`2 C[c,c] / (sum_b C[c,b] + sum_a C[a,c])`, with zero for a zero denominator.
Using a fixed inventory avoids choosing the evaluated class set using future
labels. Similarly estimate the rate of source-correct predictions that `g`
would turn into errors.

Build simultaneous uncertainty sets for the current-block confusion
matrices using the known stratified sampling design. Add a separately
specified temporal uncertainty radius `rho_t` for the possible change in
confusion outcomes before block `t+1`. The radius must be fixed from
earlier blocks or declared as a sensitivity parameter; it cannot be
certified from the unlabeled next block alone. Select a route only if its
**worst plausible** next-block macro-F1 change versus `h0` is positive and
its **worst plausible** negative-flip rate stays under a fixed bound.
Otherwise return `h0`. This is an explicit abstention rule, not a claim of
unconditional safety.

If the simultaneous coverage statement and temporal radius assumption
hold, inclusion of the true confusion matrices in the uncertainty set gives
the elementary safety implication: a route with positive lower-bound gain
has nonnegative actual gain on the next block. The hard research work is to
derive a nonvacuous finite-sample set for adaptive, imbalanced sampling and
to optimize the nondecomposable macro-F1 bound efficiently. If no route
passes, the method should honestly return the source model.

## What is already known

- Active estimation of F-measures with a label budget predates this proposal:
  https://papers.neurips.cc/paper_files/paper/2010/hash/d7a728a67d909e714c0774e22cb806f2-Abstract.html
- Active testing corrects the selection bias created by targeted labeling:
  https://proceedings.mlr.press/v139/kossen21a.html
- Limited-label selection among pretrained models has recent strong methods:
  https://proceedings.mlr.press/v258/okanovic25a.html
- Drift-triggered dynamic labeling budgets and online network-traffic active
  learning have been studied:
  https://doi.org/10.1016/j.ins.2023.119821 and
  https://doi.org/10.1016/j.engappai.2022.105607
- Returning to a baseline with a safety objective is related to policy
  learning with abstention:
  https://proceedings.mlr.press/v300/sawarni26a.html
- Multiclass confusion-matrix constraints and class-proportion-constrained
  optimal transport are established:
  https://www.jmlr.org/papers/v25/22-1137.html and
  https://proceedings.mlr.press/v139/tai21a.html

Therefore the possible contribution is **not** active F1 sampling, a drift
alarm, an OT proportion constraint, or a source fallback by itself. It
would require a concrete and useful coupling of delayed active audits,
macro-F1 improvement over a preserved source, uncertainty about changing
class/group mass, and a finite-sample decision rule. Novelty remains to be
checked against the full papers and confirmed by theory and experiments.

## Falsification order

The first uniform replay is complete. Out of 55 overlapping home/seed
replications, a 1,000-label audit produced a row-bootstrap interval excluding
zero for CARE in 18 and RF in 21. The interval contained the full development
macro-F1 difference in 47/55 CARE and 48/55 RF comparisons, below the
nominal 95% expectation; this is a descriptive check, not a coverage proof.
The random audits observed only 20.89 of the 24 evaluated classes on average.
That first replay sampled only rows selected through the official inventory;
this matches the existing curated benchmark, but assumes its membership is
provided before querying and is optimistic for an all-flow deployment.
The next stratified control samples all development rows and charges a label
even when a row falls outside the public 24-class inventory. It asks whether
the information shortage is partly an avoidable sampling problem. Its
interval method has the same limitations.

The stratified control is now complete. Of 55 overlapping home/seed repeats,
it resolved 17 decisions versus 12 for uniform sampling, with one versus two
resolved decisions in the wrong direction. Its descriptive bootstrap interval
contained the full development difference 48/55 times versus 47/55 for
uniform sampling. It observed 21.53 of 24 classes per draw versus 19.40,
and 693 versus 676 curated rows per 1,000 paid labels. These are small,
exploratory gains, not a safety result or evidence of a new sampling method.
In home04 the stratified draw contained 264 curated rows on average versus
87 under uniform sampling, so part of its gain comes from enriching official
device traffic among all flows. That changes the population relative to the
existing **curated closed-set** protocol, where the public inventory defines
the candidate pool in advance. Do not combine these all-flow audit numbers
with the earlier closed-set model comparison as if they shared a deployment
task. The four to six examined development days per home are also too few to
claim a robust sequential adaptation benchmark on their own. Any online
method needs a separately frozen stream protocol and independent validation.

The label bottleneck is structural for macro F1. If a class occupies fraction
`q` of the audit population, a uniform `B`-label draw sees no positives with
probability approximately `(1-q)^B`. At `q=0.001` and `B=1000`, this is about
36.8%; even ten *expected* positives require 10,000 uniform labels. A rare
class can still contribute 1/24 of fixed-inventory macro F1. Prediction-based
strata may help when the candidate recognizes that class, but cannot assure
coverage if both source and challenger never predict it. This is a reason to
study the information assumptions and label budget before promising a
finite-sample improvement guarantee.

1. **Information feasibility.** On already examined development periods,
   simulate a current-window audit with a fixed 1,000-label *additional*
   budget. Estimate whether source-versus-CARE/RF macro-F1 differences can
   even be resolved with simultaneous intervals. Do not call those labels
   part of the earlier 1,000-label model-training budget. The first replay
   uses ordinary paired row-bootstrap intervals only as a diagnostic; they
   are not simultaneous, temporal, or valid deployment guarantees. Run:

   ```bash
   python -u Experiment/unsw_home_temporal_v1/f1_audit_feasibility.py \
     /data/xjw/tma-care-data/unsw_home_temporal/all.zip
   cat /data/xjw/tma-care-data/unsw_f1_audit_feasibility_v1/REPORT.txt
   ```
2. **Sampling value.** Compare uniform, active testing/F1 estimation,
   BADGE, class quota, and the proposed decision-width allocation with
   exactly equal total query counts and known selection probabilities. The
   first limited-label control compares uniform sampling with a fixed
   prediction-pair stratification; it is a standard sampling idea and is
   **not** a novel method. Run:

   ```bash
   python -u Experiment/unsw_home_temporal_v1/f1_stratified_audit.py \
     /data/xjw/tma-care-data/unsw_home_temporal/all.zip
   cat /data/xjw/tma-care-data/unsw_f1_stratified_audit_v1/REPORT.txt
   ```
3. **Decision value.** Compare source-only, CARE, RF, simpler per-donor
   paid-support gates, and a robust macro-F1 route. Include negative flips,
   zero-recall classes, class F1, query cost, and block latency.
4. **Chronology.** Predictions for block `t` are saved before any `t` label
   is read. A block's paid labels may update only blocks `t+1` onward.
   Homes, not overlapping seeds, are independent units.
5. **Stop rule.** If the 1,000-label intervals are mostly vacuous or the
   proposed query allocation does not beat established active-testing
   baselines across homes, stop this direction. Do not open the untouched
   final window for a home07-specific rescue.

The present UNSW source/query/development split was not designed for this
online information pattern. A prequential replay on development data can
test feasibility, but it is exploratory because development outcomes have
already influenced the research direction. A separate independent stream
or a fully frozen final protocol would be required for a method claim.
