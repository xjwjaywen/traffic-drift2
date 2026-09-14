# Tests

Run from the repository root in the existing experiment environment.

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_kbs*.py'
bash Experiment/core_code/scripts/tests/test_run_kbs_supplement.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_acquisition_pilot.sh
bash Experiment/core_code/scripts/tests/test_run_kbs_acquisition_diagnosis.sh
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
