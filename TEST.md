# Tests

Run from the repository root in the existing experiment environment.

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_kbs*.py'
bash Experiment/core_code/scripts/tests/test_run_kbs_supplement.sh
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

The launcher tests verify path quoting, GPU selection, argument forwarding, and that a failed preparation propagates its exit code and prevents training. The runtime-benchmark shell tests cover the unchanged earlier launcher.
They also cover `badge-kd`, `badge-kd-plan`, `badge-kd-summarize` dispatch and propagation of a failed follow-up run.

Real CUDA/CESNET integration must be checked on the server using `preflight` and the primary run. No synthetic score is scientific evidence.
