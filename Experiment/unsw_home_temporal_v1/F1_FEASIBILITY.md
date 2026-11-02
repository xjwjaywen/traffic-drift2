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

The central diagnostic is whether even the optimistic row interval has a
positive lower endpoint. If it does not, this audit does not establish
positive macro-F1 gain for this fixed action even under an iid-query
interpretation, much less under future drift. If it does, compare the
day-block interval and donor-edge support before designing a policy. Do not
pick a score threshold on the same development outcomes.

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
