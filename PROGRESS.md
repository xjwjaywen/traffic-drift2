# Progress

## 2026-09-14 — Saved class audit and matched sensitivity reports

Implementation commit: `44e1ce3` (based on main `4f113e9`).

- Added a standard-library class audit of all six completed configurations: common-split per-class support, recall/F1 changes, new/residual collapse, worst noncollapse classes, query/reference counts and seed recurrence. It checks consumed small-file hashes and reconciles class metrics with run summaries before writing derived CSVs and a Chinese report.
- Added `stage2` to run the audit first, then the existing Margin sensitivity grid. Audit failures prevent training. Existing numerical engines, selector and BADGE follow-up identities are unchanged.
- End-to-end testing exposed an aggregation issue: the legacy sensitivity report includes all five saved baseline seeds but only three seeds for new settings, yielding 28 rows. The new `sensitivity-report` emits separate matched tables using the same requested seeds for all eight settings (24 rows by default), without editing the hash-pinned engine or discarding original runs.
- Validation: 24 Python synthetic tests, shell launch/order/failure tests, Python 3.10 syntax checks, and a complete class-audit run with Python site packages disabled. Hand-calculated fixtures cover support-zero exclusions and repeated class failures. A completed 30-run synthetic study adds exactly 18 sensitivity fits, then zero on resume; original run files retain hashes and modification times.
- Actual class-level findings and sensitivity scores await the user's server run. Synthetic fixtures are code checks, not paper evidence. Keep exploratory parameter scans distinct from parameter selection on an independent calibration period.

## 2026-09-14 — Paired BADGE reference-CE follow-up

Implementation commit: `fbe5fdd` (based on main `fdbcd18`).

- Added `badge-kd` after the server's 25-run primary study, to compare BADGE FT+KD with the completed full BADGE configuration using seeds 0–4. This is an exploratory follow-up motivated by the Margin ablation, not an additional independent source-model or time-period replication.
- Editing the original runner would change its implementation hash and invalidate resumability. The new entry registers only configuration/reporting callbacks; the numerical engine and selector files remain byte-for-byte unchanged. The extension records its own identity and checks the original hashes normally.
- Require verified baselines and existing selections before training. Match query/reference IDs and order, training streams, source predictions and evaluation masks before emitting independent paired tables. The 25 original runs and primary tables remain unchanged.
- Validation: all 18 Python synthetic tests, updated launcher dispatch/failure tests, original runtime-benchmark shell tests, Python 3.10 syntax checks and unchanged v1 implementation hash. A synthetic 25-run study adds exactly five fits, and a repeated follow-up adds zero; every original file retains its hash and modification time.
- Real BADGE FT+KD scores remain pending the user's CUDA server run. Synthetic test scores are not experimental evidence.

## 2026-09-14 — Controlled KBS supplements

Implementation commit: `39413d3` (based on main `63992e7`).

- Added a server runner for fixed-step replay/KD ablations, full-class preservation analysis, and optional sensitivity scans.
- Existing scripts varied target repetition and optimizer update counts across ablations. The new protocol uses independent, fixed target/reference sampling streams and separately switches reference CE and KD. It is explicitly separate from archived main-table runs.
- Added snapshot-qualified sample IDs, source/cache checksums, saved repaired heads, all-sample predictions, paired and common evaluation masks, per-class metrics, and completion hashes for resumability.
- A shared output layout lets sensitivity scans reuse two primary configurations without duplicating runs.
- Validation: 12 Python synthetic tests, new launcher shell tests, earlier runtime benchmark shell tests, Python 3.10 syntax parsing, and compatibility with the stored 178-class / 256-feature source checkpoint.
- Synthetic tests are implementation checks only. Real CUDA/CESNET experiment results must come from the server run.
- Keep completed results immutable. Changed data/protocol/code should use a new output directory; do not merge different protocols by filename alone.
