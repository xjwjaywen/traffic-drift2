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

This writes `/data/xjw/tma-care-data/unsw_rf_refit_stability_v1/home07/REPORT.txt`.
Compare proposal **rates** and donor precision because the paid fold and
development samples have different sizes. Even if the instability can be
reduced, do not set a route or threshold using the already examined
development outcomes and present it as independent validation.

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
