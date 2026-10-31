# UNSW 12-home temporal drift audit

The [official IPFIX dataset](https://iotanalytics.unsw.edu.au/homedataset.html)
contains 12 homes observed over 47 days. Its [source paper](https://arxiv.org/abs/2301.06695)
reports temporal degradation of device identification within homes. This is a
closer independent setting for traffic drift than the CPS label-shift dataset.

Download the `all.zip` archive outside the repository, then audit timestamps:

```bash
mkdir -p /data/xjw/tma-care-data/unsw_home_temporal
curl -fL --retry 3 \
  https://iotanalytics.unsw.edu.au/homedataset/all.zip \
  -o /data/xjw/tma-care-data/unsw_home_temporal/all.zip
python -u Experiment/unsw_home_temporal_v1/audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_home_temporal_audit_v1/REPORT.txt
```

The audit decodes timestamps only. It checks the 28 flow-statistic and protocol
columns used by the source paper; MAC addresses, IP addresses, ports, and device
labels are excluded from model inputs. The chronological boundaries were fixed
after reviewing timestamp coverage. Device labels from the final test window
must remain unopened until the method and comparisons are frozen.

The timestamp audit has now been reviewed; the exact per-home windows are in
[`metadata/windows.csv`](metadata/windows.csv). The public archive contains
32,117,175 rows before filtering to the official IoT inventory. The published
`devices.xlsx` was downloaded separately (SHA256
`b1bf51fdc8b5f721c5c7360ec4a00c98e2f83748033eb7fe3afd7957536fdc4f`)
and its 24-type/home mapping is in
[`metadata/official_devices.csv`](metadata/official_devices.csv). The source
audit counts device labels only inside the frozen source windows:

```bash
python -u Experiment/unsw_home_temporal_v1/source_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_home_source_audit_v1/REPORT.txt
```

See [`PROTOCOL.md`](PROTOCOL.md) for the evaluation boundary and the important
distinction between a curated 24-device benchmark and a mixed-traffic deployment
stream.

Prepare the fixed source reservoir, unlabeled query candidate reservoir, and
development features. The preparation stage stores query labels for later
paid selection, but does not use them to select rows. It does not decode any
development label or parse any final-window row. Run one home first if desired;
rerunning a completed home verifies its artifacts and skips the scan.

```bash
python -u Experiment/unsw_home_temporal_v1/prepare.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip \
  --home home01
python -u Experiment/unsw_home_temporal_v1/prepare.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
```

The prepared files are written outside the repository under
`/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1`. Do not load the
final-window labels during development.

Freeze the first standard-control predictions for the 11 closed-set homes:

```bash
python -u Experiment/unsw_home_temporal_v1/predict.py
```

This fits a new tabular source MLP and, per repair seed, a target-only head,
CARE head, frozen-feature random forest, and cosine 1-NN. It indexes only the
1,000 paid labels selected uniformly for that seed. It never opens the raw
archive or development/final labels. The output is under
`/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1`. Completed homes
and seeds are verified on rerun. If a run stops inside a seed, remove only that
incomplete `seed_N` directory before rerunning.

After predictions for **all 11 homes and five seeds** are complete, evaluate
the development window:

```bash
python -u Experiment/unsw_home_temporal_v1/evaluate.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_home_temporal_development_v1/REPORT.txt
```

The evaluator checks all frozen predictions before decoding any development
label. It excludes flows outside the official 24-device inventory from the
reported closed-set metrics. It never decodes final-window labels or features.
These are standard controls, not a new method or a test-set result.

For a read-only diagnosis after the development comparison, quantify paired
source-to-CARE repairs and harms by true device type:

```bash
python -u Experiment/unsw_home_temporal_v1/interference_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_home_temporal_interference_v1/REPORT.txt
```

This audit only indexes already paid query labels and reopens development
labels. Its per-class scores and routes cannot be used as a final-test method
without a new target-label-free rule fixed in advance.

The read-only audit revealed that a class's F1 can rise even when CARE loses
true positives, because its false-positive count falls. A small follow-up
**standard control** tests whether simple prior calibration already explains
part of this effect. It was specified after viewing development outcomes, so
its results are exploratory. The fixed rule uses the same 1,000 paid query
labels, the frozen source model, a pseudocount of one, and half-strength
log-ratio correction against source prediction frequencies in the unlabeled
query pool. It has separate prediction and evaluation stages:

```bash
python -u Experiment/unsw_home_temporal_v1/prior_control.py predict
python -u Experiment/unsw_home_temporal_v1/prior_control.py evaluate
cat /data/xjw/tma-care-data/unsw_home_prior_control_v1/development/REPORT.txt
```

The evaluator refuses to read development labels until predictions for all
11 primary homes and five seeds are complete. It never opens final-window
labels. This is a test of an existing calibration idea, not a novelty claim.

To test whether the paid query labels can *predict* which home benefits from
that correction, run the diagnostic transport audit:

```bash
python -u Experiment/unsw_home_temporal_v1/prior_transport_audit.py
cat /data/xjw/tma-care-data/unsw_home_prior_transport_v1/REPORT.txt
```

Within each selected 1,000-label query set, it fits the fixed correction on
the earlier 700 rows by archive order and evaluates it on the later 300. It
then checks a fixed sign rule and a leave-one-home-out logistic rule against
the already measured development outcomes. The logistic rule learns from the
development outcomes of the other ten homes, never from the held-out home's
development labels. This is a small-sample feasibility audit, not a method
validation; development outcomes had already been examined when it was
designed. It reads no final-window data.

## Source-only temporal-invariance feasibility probe

The following diagnostic uses only the existing labeled source reservoir.
It recovers only the retained source rows' dates from the verified archive,
skipping labels, features, and timestamps on all other rows, then splits
observed source dates into three consecutive blocks, fits projections and the
same classifier on blocks 1–2, and evaluates block 3. Archive bytes for
intervening rows may be traversed to reach retained source IDs; no later
labels, features, or timestamps are decoded. The temporal projection adds a class-balanced
within-class mean-drift penalty to ordinary Fisher separation. The ordinary
Fisher projection, temporal projection, and full feature view use the same
classifier; projection comparisons have identical dimensions. Classes with
fewer than 30 rows in either fit block do not define projection directions,
but remain in the classifier and evaluation. Inspect their block counts and
per-class recall; aggregate macro F1 alone is insufficient.

```bash
python -u Experiment/unsw_home_temporal_v1/invariant_probe.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip --home home01
python -u Experiment/unsw_home_temporal_v1/invariant_probe.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_invariant_probe_v1/REPORT.txt
```

Outputs include `block_counts.csv`, `per_class.csv`, and `summary.csv` per home.
This was designed after development outcomes were seen. Even a positive
source-block result would only establish a necessary condition for the
invariance idea, not a target-month result or a new method.

To check whether the small average difference comes from a few class/home
pairs, summarize the already generated per-class CSVs without reading the
archive or later-window data:

```bash
python Experiment/unsw_home_temporal_v1/invariant_probe_review.py
cat /data/xjw/tma-care-data/unsw_invariant_probe_v1/REPORT_classes.txt
```

The review reports class support, projection eligibility, paired F1 changes,
and each class's contribution to its home's macro-F1 difference. It is
descriptive and cannot select a class-specific target method after the fact.

The class review showed no device type improving in at least eight homes.
Several of the largest class/home F1 changes occurred where the first source
day block had no retained rows for that class. Before pursuing a prior or
conditional-shift model, audit both mechanisms using source labels only:

```bash
python -u Experiment/unsw_home_temporal_v1/source_mechanism_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip --home home01
python -u Experiment/unsw_home_temporal_v1/source_mechanism_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_source_mechanism_v1/REPORT.txt
```

The audit reweights the existing uniform within-class source reservoir by
each class's reviewed full source count to estimate class shares by source
day block. It also reports a class-conditional per-feature KS effect size
between adjacent blocks, when both have at least 100 retained rows, and a
within-block day-split reference when possible. These are descriptive
estimates, not a formal hypothesis test or proof that a class's feature
distribution will remain stable in later windows. No target-period label,
feature, or timestamp is decoded.

The source audit found many late-appearing classes, while adjacent source
blocks had similar average class-conditional KS to the within-block day
reference. This does not establish that class conditionals stay stable in the
query period. The next audit uses only each seed's already paid 1,000 uniform
query labels, never pooling seeds into a larger label budget:

```bash
python -u Experiment/unsw_home_temporal_v1/paid_shift_audit.py --home home01
python -u Experiment/unsw_home_temporal_v1/paid_shift_audit.py
cat /data/xjw/tma-care-data/unsw_paid_shift_v1/REPORT.txt
```

It compares query class shares with weighted late-source shares and reports
Wilson intervals for the paid-label estimate. For classes with at least 30
matched paid/source rows, it compares source-to-query feature KS with a
same-size source-to-source reference. This remains exploratory: class-level
query labels are sparse, seed results are not independent, and no development
or final-window label or feature is read. The audit is a diagnostic of whether
prior-only methods deserve further testing, not a new adaptation method.

After the paid-shift and prior-control development reports exist, join their
fixed outputs to check whether query class proportions persist into the
development window and whether the class-conditional KS signal tracks where
prior correction harms class F1:

```bash
python -u Experiment/unsw_home_temporal_v1/shift_transport_audit.py
cat /data/xjw/tma-care-data/unsw_shift_transport_audit_v1/REPORT.txt
```

This is a read-only, post-hoc audit. It opens only existing CSV artifacts, not
the archive or final window. `by_home.csv` treats homes as the independent
units; `lost_classes.csv` lists present classes whose recall fell to zero.
Development outcomes were already viewed before this audit was designed, so
its associations cannot establish a new adaptive method or independent gain.

For the home07 class-collapse mechanism, compare the frozen hard-count prior
control with a soft-probability-count denominator. Both use the same 1,000
paid labels per seed and the same bias weight; this is a standard baseline
check, not a proposed new method:

```bash
python -u Experiment/unsw_home_temporal_v1/soft_count_probe.py --home home07
```

The probe verifies frozen artifacts, recomputes unlabeled query counts with
the fixed source model, and evaluates on the already examined development
window only. It prints whole-model and classes 6, 8, 9, and 10 results.
