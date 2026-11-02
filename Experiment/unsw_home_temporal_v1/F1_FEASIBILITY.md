# Can 1,000 paid labels support an F1-aware repair decision?

Status: exploratory feasibility audit, not a new method result. The home07
class-9 focus was chosen after development outcomes were examined. The final
window remains unopened.

The archive is not required to be in chronological row order. When mapping
paid query row IDs to dates, the program decodes timestamps only for those
selected IDs; other archive rows are streamed past without parsing any field.

## Why test this before building an optimizer

The paid-label gate accepted RF proposals to class 9 in four of five
overlapping seeds. On development data, that route raised mean macro F1 by
only 0.21 percentage points: class 9 improved, while classes 6 and 10 lost
F1. Counting corrected minus damaged rows optimizes an accuracy-like
quantity, not macro F1. For class `c`, `F1_c = 2 TP_c / (N_c + P_c)`; a
single change from source prediction `s` to RF prediction `t` changes the
denominators for both classes. The effects of several changes interact.

`f1_feasibility.py` reproduces the exact five-fold RF predictions from the
prior paid-only audit and checks them against its frozen gate counts. It
evaluates one fixed action, all RF-to-class-9 proposals, on each 1,000-row
paid draw. It does **not** select a route using these F1 estimates. It
reports paired macro-F1 differences, proposed donor edges, and two
descriptive bootstrap intervals: resampling individual paid rows and
resampling whole query days. Seven query days are too few for a reliable
day-level interval; the row interval ignores temporal dependence. Neither
interval includes model-fitting uncertainty or certifies a future window.

Run from the repository root on the server:

```bash
python -u Experiment/unsw_home_temporal_v1/f1_feasibility.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

The output is `/data/xjw/tma-care-data/unsw_f1_feasibility_v1/home07/REPORT.txt`,
with `summary.csv` and `edges.csv` for inspection.

## Completed home07 audit

The fixed all-RF-to-class-9 action increased paid-query OOF macro F1 by
3.67, 2.66, 4.24, 3.68, and 0.71 percentage points across the five
overlapping seeds. The row-bootstrap lower endpoint was above zero for
seeds 0, 2, and 3; the seven-day block-bootstrap lower endpoint was above
zero for seeds 1, 2, and 3. Neither set of intervals is uniformly positive.
There were only 19--38 proposed overrides per paid draw, making donor-edge
estimates especially uncertain.

One concrete objective mismatch appears in seed 1: source-6 to RF-9 had
one RF accuracy win and three source wins, yet its stand-alone paid macro-F1
change was slightly positive (+0.07 percentage points). Macro F1 cannot be
replaced by a simple tally of corrected minus damaged rows. However, the
previous home07 development result from a paid-gated route improved mean
macro F1 by only +0.21 percentage points. Those two comparisons differ in
both policy (fixed all-proposals versus seed-wise gate) and RF training size
(800 versus 1,000 paid rows), as well as time period. The next read-only
audit, `rf_transport_audit.py`, compares the same all-proposals route and
its donor edges on query versus development to separate what can be seen
without opening the final window. It still cannot causally isolate time
drift from the RF training-size change.

```bash
python -u Experiment/unsw_home_temporal_v1/rf_transport_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

This writes `/data/xjw/tma-care-data/unsw_rf_transport_audit_v1/home07/REPORT.txt`.

## Query-to-development transport result

Using the same all-RF-to-class-9 action, development macro-F1 changes were
+0.08, +0.49, -0.01, +0.50, and -0.45 percentage points, far below the
paid OOF changes. In seed 4, the paid OOF models proposed just one
source-class-14 to class-9 change, but the full RF proposed 3,233 on
development. Only about 0.12% of those development proposals were true
class 9. The 800-label fold models and the full 1,000-label RF differ in
both training size and random state, so this observation alone cannot tell
whether refitting or the time shift caused the discrepancy.

`rf_refit_stability.py` sends each existing 800-label fold RF to both its
held-out paid rows and a fixed uniform sample of 20,000 curated development
rows. It also compares the already frozen full RF on those same development
rows. This identifies whether the seed-4 class-14 proposal explosion is
specific to the full refit or appears in the fold models too. A 4-of-5
fold-model consensus route is reported as an exploratory stability control;
the fold models share training rows, so the vote count is not a statistical
confidence bound. Development outcomes were already viewed and this control
is not independently validated.

```bash
python -u Experiment/unsw_home_temporal_v1/rf_refit_stability.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

The corrected run writes `/data/xjw/tma-care-data/unsw_rf_refit_stability_v2/home07/REPORT.txt`.
Compare proposal **rates** and donor precision because the paid fold and
development samples have different sizes. Even if the instability can be
reduced, do not set a route or threshold using the already examined
development outcomes and present it as independent validation.

The initial 20,000-row development audit reported that seed-4 fold models
3 and 4 each proposed 1,201 class-14 to class-9 changes. That comparison
used a different feature-extraction batch order from the frozen full RF;
its fold counts and consensus result are **superseded**. Under the original
full-query feature order, a targeted rerun found seven such proposals for
fold 3 and 1,201 for fold 4. The full RF still proposed 1,201 and exactly
matched the frozen predictions. This establishes strong sensitivity to
the paid training subset but does not validate a 4-of-5 voting policy. Run
`rf_refit_stability.py` again; its corrected output goes to
`unsw_rf_refit_stability_v2` and checks all five reproduced full RFs.

The remaining question is whether this region was present in the earlier
unlabeled query pool or appeared mainly in development. `rf_query_coverage.py`
reproduces seed-4 folds 3 and 4 and the full RF, checks the full RF against
frozen predictions, and compares class-14-to-9 proposal rates on the unpaid
query pool versus the same development sample. It reads no unpaid query
label; class membership of development rows is used only to retain the
already curated evaluation population. Query-to-development rate changes
describe feature coverage and cannot by themselves identify a causal drift
mechanism.

In the completed query-coverage audit, the frozen source predicted class
14 on 135/23,759 unpaid query-pool rows (0.57%) versus 3,727/20,000
development-sample rows (18.64%). This is about a 33-fold change in the
source-predicted class share, not a measured change in true class-14
prevalence. The matched full RF proposed 14-to-9 on all 135 early rows
versus 1,201 later rows; the corresponding population proposal rates are
0.57% and 6.01%. Fold 3 proposed 105 versus seven, while fold 4 proposed
135 versus 1,201. Thus a large source-prediction-distribution change and
RF fit instability are both present in this routing problem. No
unpaid query correctness can be inferred from this coverage audit.

```bash
python -u Experiment/unsw_home_temporal_v1/rf_query_coverage.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

## Count-matched stability falsification

The corrected development audit gave a positive signal for 4-of-5 fold
consensus, but that route is a separate 800-label ensemble: it may propose
class 9 even when the full 1,000-label RF does not. It also accepts far
fewer changes. For seed 4, the full route changed development macro F1 by
-0.45 percentage points with an 8.39% negative-flip rate; consensus changed
macro F1 by +0.95 points with a 0.26% negative-flip rate. Across the other
four overlapping seeds, consensus was better twice and worse twice. In
seed 4, corrected fold models 2 and 4 each proposed 1,201 class-14-to-9
changes on the development sample, while folds 0, 1, and 3 proposed 0, 7,
and 7. `rf_stability_matched_control.py` tests the narrower action
`full RF proposes class 9 AND at least four fold RFs agree`. It compares
this intersection with two random controls using the same full-RF proposal
set: one keeps the same total number of proposals, and one additionally
keeps the same number from each source-predicted donor class. Both random
controls use 200 fixed draws per seed; their intervals describe only
random assignment on the already examined development sample. The script
checks every fold's proposal counts against the corrected v2 audit.

```bash
python -u Experiment/unsw_home_temporal_v1/rf_stability_matched_control.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

Output: `/data/xjw/tma-care-data/unsw_rf_stability_matched_v1/home07/REPORT.txt`.
If the intersection performs like count-matched random selection, fold
agreement supplies no evidence of identifying better individual overrides
in this setting. Even if it wins, the class, threshold, and home were chosen
after development inspection, so this remains a mechanism probe and must
not be reported as an independent method result.

### Completed matched-control result and decision

For every one of the five overlapping home07 seeds, the intersection's
development macro-F1 change fell inside the 2.5--97.5% range of the
donor-count-matched random control. Its class-9 F1 change did too. In
seed 4, the intersection changed macro F1 by +0.953 percentage points;
random selection with the same number of accepted proposals from each
source-predicted class averaged +0.962 points. Random selection matching
only the total number averaged +0.040 points. The intersection kept just
7 of the full RF's 1,201 class-14-to-9 proposals, none of which was truly
class 9. The donor-count-matched random control copies the intersection's
per-donor counts, so it is a diagnostic control, not a deployable policy.

These results support a **donor-allocation** explanation for the apparent
benefit: avoiding most proposals from the dangerous source-predicted class
matters, while fold agreement has not demonstrated a reliable advantage
for ranking individual proposals within a donor class. The intersection's
seed-4 negative-flip rate was slightly below the donor-matched random range,
but this does not rescue the row-level macro-F1 claim. Stop developing the
4-of-5 row-voting rule as a stand-alone method. Do not open the final window
for this home07-specific rule.

If work continues on selective repair, formulate a class-agnostic donor-level
decision using only paid labels and observable unlabeled mass/coverage, then
test its ability to protect macro F1 across independent homes and periods.
The source-predicted class share can change sharply (0.57% to 18.64% for
class 14 here), so paid-query accuracy alone is not sufficient to transport
a routing decision. Any guarantee must state an explicit temporal-shift
assumption; no unsupported guarantee follows from these audits.

## Potential research problem, if data support it

Let `h0` be a frozen source classifier, `h1` a nonlinear challenger, and
`g(x)` decide whether to accept `h1(x)`. A candidate problem is to maximize
the *worst plausible* improvement in multiclass macro F1 while bounding
the rate of changes from a source-correct prediction to an error. The
plausible distributions would be built from paid-label uncertainty plus an
explicit, testable bound on temporal shift. Without such a shift assumption
or fresh feedback, no method can guarantee improvement under arbitrary
future concept drift.

Direct macro-F1 optimization and imbalanced-expert routing are established
research. A contribution would require a distinct estimator or tractable
decision rule for sparse paid labels and temporal shift, a defensible
guarantee under stated assumptions, and gains across independent homes
against those established baselines. A class-9-specific threshold or a
relabeling of the existing gate is not enough.

Relevant prior work:

- Multiclass macro-F1 optimization (BEAM-F): https://research.ibm.com/publications/optimizing-the-multiclass-f-measure-via-biconcave-programming
- Imbalanced-expert deferral (MILD): https://proceedings.mlr.press/v306/cortes26b.html
- IoT label shift: https://www2.ee.unsw.edu.au/~hhabibi/pubs/conf/25cpsiotsec-labelshift.pdf
