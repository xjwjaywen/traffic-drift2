# Tests

Protection-gap diagnostic checks can also run independently:

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_kbs_protection_gap_diagnosis.py'
bash Experiment/core_code/scripts/tests/test_run_kbs_protection_gap_diagnosis.sh
```

They compare analytical CE gradients (including bias) against independent autograd,
hand-count training overlap, persistence, class coverage and feature diversity, and
recompute unified metrics from both completed synthetic studies. Diagnosis blocks
all fitting/acquisition/full-pool prediction and never loads reference features;
every input hash/mtime is preserved. Empty denominators, prefix differences, all-class
rows, source completion, changed seeds/outputs and rehashed stale metrics are checked.
Resume loads no arrays. Separate CPU CLI invocations and launcher tests cover quoted
paths, isolated seeds/output, hidden CUDA and exit propagation. These are synthetic
correctness tests, not evidence for the research hypothesis. See
`Experiment/core_code/PROTECTION_GAP_DIAGNOSIS.md`.

Run from the repository root in the existing experiment environment.

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_kbs*.py'
bash Experiment/core_code/scripts/tests/test_run_kbs_supplement.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_acquisition_pilot.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_acquisition_diagnosis.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_pool_control.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_relation_audit.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_badge_controls.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_repair_aware.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_repair_diagnosis.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_oracle_protection.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_protection_budget.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_learned_protection.sh
bash Experiment/core_code/scripts/tests/test_run_runtime_benchmark_m12.sh
bash -n Experiment/core_code/scripts/run_kbs_supplement.sh
```

The Python tests use synthetic tensors in temporary directories, never CESNET paper results:
- Five primary configurations and eight unique sensitivity configurations, with two reusable configurations.
- Identical optimizer step counts and target/reference sampling streams across loss ablations.
- FT+KD independence from reference labels; unchanged source head; seed repeatability.
- Balanced nested reference selection and failure on missing class support.
- Global confusion-derived per-class F1, paired query-exclusion, supported-class accounting.
- Frozen feature extraction via a mocked loader, cache reuse, raw-input metadata invalidation, period mismatch.
- The existing BADGE selector on a small synthetic pool.
- Saved predictions/head/metrics, common evaluation mask, resumability and incompatible protocol rejection.
- Completion checks reject missing/modified artifacts.
- BADGE follow-up adds exactly five fits to 25 completed synthetic primary runs; repeated continuation adds zero fits and preserves every old file's hash and modification time.
- New BADGE runs retain the KD term, disable only reference CE, and reuse the original engine hash and query selections.
- Paired reporting checks IDs/order, training streams and prediction/evaluation alignment; reports only matched seeds, sample SD and the five-seed requirement.
- Missing baselines, modified selections, changed study/extension protocols and mismatched reference IDs are rejected.
- Class audit covers all six configurations, reconciles common-split class metrics with saved run summaries, preserves every input hash/mtime, and generates deterministic reports.
- Hand-calculated fixtures verify new/residual collapse counts, zero-support exclusions, seed recurrence and worst-class selection. Missing runs, modified files, duplicate classes and metric disagreements fail before writing a report.
- A completed 30-run synthetic study adds exactly 18 sensitivity fits and none on resume. Matched summaries select seeds 0–2 for all eight configurations (24 rows), excluding the two baselines' extra seeds that remain in the legacy 28-row output. Missing requested seeds fail reporting.

The launcher tests verify path quoting, GPU selection, argument forwarding, and that a failed preparation propagates its exit code and prevents training. The runtime-benchmark shell tests cover the unchanged earlier launcher.
They also cover `badge-kd`, `badge-kd-plan`, `badge-kd-summarize` dispatch and propagation of a failed follow-up run.
`class-audit` and `stage2` tests check independent audit/training seed arguments,
audit → sensitivity preparation → sensitivity run → matched reporting order, quoted paths, and
that failed analysis prevents any training. `python -S scripts/kbs_class_audit.py --help`
from `Experiment/core_code` checks the audit's standard-library-only entry.

Real CUDA/CESNET integration must be checked on the server using `preflight` and the primary run. No synthetic score is scientific evidence.

BADGE component-control tests complete 30 synthetic primary/KD runs with exactly
10 new fits, then zero on resume without loading any feature cache. They preserve
every original file's hash/mtime and the source head, verify the four independent
loss switches, sample streams, exact Margin/BADGE exclusion masks and query truth,
and recompute metrics from saved predictions. Rehashed semantic inconsistencies in
IDs, losses, masks and metrics are rejected, as are corrupt files, missing KD
baselines and changed settings/extension identities. Reports use only seeds complete
in all four cells, including empty/one-seed cases, and include all classes and paired
differences. A hand calculation checks sample SD. Separate CLI summary works without
a feature cache. Launcher tests check quoting, GPU, argument forwarding, isolated
seed/step settings and failure propagation. These tests do not measure real CESNET
performance, GPU runtime or KBS suitability.

Acquisition pilot tests cover reference IDs matched to the original five-per-class
sampling, predicted-frequency risk arithmetic, confident-error discovery on known
synthetic geometry, zero-score fallback and duplicated features, identical exploration
and reproducibility, hand-calculated retrieval/coverage metrics, target labels replaced
by an opaque object during selection, all-seed completion before any label evaluation,
paired CSV statistics, query-union exclusion, process-separated CLI execution,
input hash/mtime preservation, no-op resume, changed protocols and corrupt artifacts.
The new launcher test checks quoting, seed/GPU forwarding and that failed selection
prevents the label-evaluation process. The pilot uses no CESNET import or model fitting.

Acquisition diagnosis tests check the four head/prototype correctness cells with
hand-calculated counts, empty/unsupported cohorts, candidate exclusions, phase
partition arithmetic, and budget-limited scouting. A synthetic completed pilot is
diagnosed end-to-end: prototype construction rows are excluded from reference
evaluation, original acquisition metrics reconcile, phase counts sum to full selection,
all classes survive reporting, and original hashes/mtimes are preserved. Resume loads
no tensors and preserves derived results. Missing original evaluation, changed source
identities, invalid seeds/output paths and corrupted derived files fail. The diagnosis
launcher forwards quoted paths, GPU/seeds/options and nonzero exit status.

Fixed-pool control tests independently calculate sequential sampling probabilities
for uniform, score-only, distance-only and score-times-distance policies. They verify
original hybrid query order, identical/zero-candidate/exhausted pools, uniqueness and
shared exploration/fallback, hand-counted phase metrics and sample SD, and target labels
replaced by an opaque object during selection. End-to-end synthetic tests reconcile
original metrics, sum phases, include every class, check conditional uniform expectations,
and exclude the union of all six original methods plus three controls. Original inputs
retain their hashes/mtimes; no-op resume loads no tensors or modifies result files.
Missing selection completion, corrupted pool/source files and changed runtime/protocols
are rejected. Separate CLI subprocesses and launcher failure propagation enforce
selection before evaluation; independent POOL_SEEDS prevent old seed variables leaking in.

Relation audit tests use completed synthetic primary/BADGE-KD fixtures. Independent
hand calculations check the failure-to-correction direction, self-loops, outdegree
control, positive/negative/wrong-to-different-wrong flips, achievable accuracy/recall
bounds and unsupported classes. Recomputed class metrics agree with the original
engine. End-to-end tests reconcile common metrics and exact query unions, emit all
classes/observed transitions, preserve input hashes/mtimes, and resume without array
loads. A guarded npz interface makes target truth and repaired predictions unreadable
during relation freezing; missing any frozen seed blocks evaluation before any label
load. Mask/query-truth mismatches fail even after recomputing the completion hashes.
Source/derived corruption and incompatible output/seed settings fail. Separate CLI
subprocesses work without a feature cache; importing the production module does not
import PyTorch. Launcher tests verify quoting, independent RELATION_SEEDS, separate
freeze/evaluate processes, argument forwarding and failure propagation.

Repair-aware acquisition tests independently check reverse error-edge weighting,
uniform/weighted sampling, empty/small-pool fallback, uniqueness, deterministic
zero-relation equivalence, query budgets, per-class negative flips and sample SD.
End-to-end synthetic runs enforce scout-only / frozen-query-only label access using
a guarded label object; replacing every unscouted label leaves acquisition unchanged.
They verify fresh-source initialization for all five fits per seed, paired training
streams, recomputed common-union metrics, source hash/mtime preservation and no-op
resume without loading tensors. Missing earlier stages block before label access;
corrupted inputs/outputs and changed protocols fail. Separate CLI stages and launcher
tests cover paths with spaces, GPU forwarding, isolated seed/output variables and
failure propagation before training/evaluation. See `Experiment/core_code/REPAIR_AWARE_PILOT.md`.

Repair diagnosis tests independently hand-calculate coverage, random enrichment,
damage persistence, relation cohorts, reverse transition counts and exposure-adjusted
absorber risk. They check zero denominators, unsupported classes, metric-specific
valid seed counts, query exclusion and count partitions. Completed synthetic pilots
are diagnosed using only saved labels/logits/predictions while fitting, inference,
device selection and feature access are blocked. Recomputed metrics must match the
original evaluation; every input hash/mtime is preserved, and resume loads no arrays.
Missing stages, changed identities, corrupt outputs and rehashed semantic metric
mismatches fail. Separate CPU CLI processes and the launcher test check quoted paths,
ignored GPU/seed variables, CPU visibility and nonzero exit propagation. See
`Experiment/core_code/REPAIR_DIAGNOSIS.md`.

Oracle protection tests hand-check uniform sampling from source-correct and
source-correct/probe-wrong pools, identical-pool equivalence, exact six-arm query
unions, source-truth privilege accounting and insufficient-pool failure without
fallback. A completed three-seed synthetic pilot adds exactly six fits with fresh
source initialization, original reference rows and paired training streams; training
can read only frozen query labels. Selection cannot read final repair predictions.
Old baseline metrics reconcile on their original masks; all six arms then recompute
on the enlarged common exclusion. All original hashes/mtimes remain unchanged;
resume adds zero fits and loads no arrays. Missing stages, rehashed ineligible queries,
corrupt artifacts, changed settings and rehashed old metric discrepancies fail.
Separate CLI stages and launcher tests cover quoted paths, independent oracle seeds,
inherited runtime, GPU visibility and failure propagation. Synthetic protocol changes
are confined to temporary fixtures; production hyperparameters are inherited unchanged.
See `Experiment/core_code/ORACLE_PROTECTION_CONTROL.md`.

Protection-budget tests independently check nested subsets, BADGE rank-order fill,
overlap removal, exact reused endpoint ordering and retention of the full old
evaluation exclusion. Three completed synthetic oracle seeds add exactly 12 fits,
then zero on resume; source hashes/mtimes, fresh source initialization and paired
training streams are preserved. Selection cannot load tensors; training labels are
restricted to frozen query IDs. All old metrics reconcile on the unchanged mask,
and every allocation, pair and class appears in the report. Missing source/selection/
training stages, rehashed mask/label/metric inconsistencies, incompatible identities
and corrupt results fail. Separate CLI stages and launcher tests check quoted paths,
independent budget seed/output variables, inherited runtime and failure propagation.
See `Experiment/core_code/PROTECTION_BUDGET_CONTROL.md`.

Learned-protection tests check the logistic objective's first-order condition,
ranking direction, constant signals, insufficient-support/solver fallback, independent
signal arithmetic and saved-probe alignment. Hand fixtures verify the paid 80%/15%/5%
partition, top-ranked selection, random ties and empty/small-pool fill. Guarded label
objects allow only the 15% fit queries during acquisition and frozen full queries
during training; replacing every other label leaves acquisition unchanged. Completed
synthetic pilots add exactly 12 paired fresh-source fits, preserve all source hashes/
mtimes, recompute expanded-mask metrics and resume without tensor loads. Missing stages,
changed identities, rehashed query/model inconsistencies and stale source metrics fail.
Separate CLI processes and launcher tests cover quoted paths, isolated learned-policy
seeds/output, inherited runtime and fail-stop behavior. See
`Experiment/core_code/LEARNED_PROTECTION_PILOT.md`.
