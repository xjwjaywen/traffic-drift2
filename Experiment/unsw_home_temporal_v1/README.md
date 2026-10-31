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
