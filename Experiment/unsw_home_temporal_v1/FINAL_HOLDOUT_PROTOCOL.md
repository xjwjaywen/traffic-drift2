# Frozen 11-home IoT final-window evaluation

This protocol was written after the development results had been examined and
before any final-window device label is decoded. It is an external holdout of
the existing **one-shot, 1,000-label head-repair port**, not an evaluation of
the proposed monitoring pipeline or a new IoT method. The archived final
windows in `metadata/windows.csv` and the 11 primary homes are unchanged.

## Fixed models and information boundary

- Reuse each home's original source checkpoint, source reservoir, query pool,
  five source references per type, and the five original paid 1,000-label
  query sets. Do not add final-window labels to training or selection.
- Refit target-only and CARE heads with the exact optimizer configuration in
  `predict.py`. Require their entire development-window predictions to match
  the previously frozen predictions before making final predictions. Abort on
  a mismatch; never silently change seeds, schedule, or device to improve it.
- Compare `static_mlp`, `target`, `care`, and a fixed old/new head-weight
  interpolation `bcwi_050 = 0.5 * source + 0.5 * target`. The interpolation is
  an existing-method control chosen after development inspection; it is not
  a CARE contribution. Report all four, even if one is unfavorable.
- The prediction stage parses final-window timestamps and 28 flow features
  only. It does not use the `device`/MAC column. The evaluation stage checks
  the completed prediction hashes for **all 11 homes and five seeds** before
  decoding any final-window MAC into its official device-type label.
- Evaluation uses the published 24-type MAC inventory to retain its curated
  closed-set rows. This idealized membership is not a deployable filter. The
  archive is mixed-protocol; results must not be called encrypted-only IoT.

## Prespecified outcomes

The primary outcome is each home's macro-F1 over official device types
present in its final window, with an equal-home mean and paired home-level
CARE-minus-static differences. The five paid draws are repeated measurements
inside a home, not 55 independent homes. Also report per-home negative-flip
rate conditional on a correct static prediction, the number of homes with a
positive CARE difference, and seed variation. For the equal-home mean paired
CARE-minus-static difference, use a descriptive 95% percentile bootstrap of
11 homes (10,000 resamples, RNG seed 0); do not resample the five seeds as
independent households.

For every home/type pair, report final support, predicted count, recall and
F1 under every method. The following groups are defined once from the static
predictions and labels, never selected by the challenger outcome:

- **Final low recall:** final support >= 50 and static final recall < 0.1.
  Report mean F1 and the number crossing recall 0.1/0.5 under each method.
- **Incident collapse:** the same final low-recall condition plus development
  support >= 50 and static development recall >= 0.8. This is a narrower
  temporal descriptor; report `NA` if the set is empty.
- **Stable recognition:** development support >= 50, static development
  recall >= 0.8, final support >= 50, and static final recall >= 0.8. Report
  F1 change and count of losses exceeding five F1 percentage points.

As secondary diagnostics, calculate (i) total-variation distance between
development and final 24-type class shares, (ii) the largest final class
share as a simple imbalance indicator, and (iii) the mean number of already
paid positive labels among final low-recall types. Report each indicator's
Spearman correlation with the 11 home-level CARE-minus-static F1 differences,
including the number of eligible homes; omit a correlation when fewer than
five homes have that indicator. These are post-hoc *descriptors* because final
labels define the first two indicators and the low-recall subset. They are
not evidence that a particular shift mechanism caused a CARE gain or loss.
No threshold, route, or method choice will be fitted to final labels. We will
retain both positive and negative homes in the paper and explicitly identify
the data as a curated, mixed-protocol IoT stress test.

Run `final_predict.py` first, then `final_evaluate.py`. Both default to new
output directories; neither modifies the original development artifacts:

```bash
python -u Experiment/unsw_home_temporal_v1/final_predict.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
python -u Experiment/unsw_home_temporal_v1/final_evaluate.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_iot_final_holdout_v1/evaluation/REPORT.txt
```
