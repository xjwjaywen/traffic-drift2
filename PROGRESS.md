# Progress

## 2026-09-14 — Controlled KBS supplements

Implementation commit: `39413d3` (based on main `63992e7`).

- Added a server runner for fixed-step replay/KD ablations, full-class preservation analysis, and optional sensitivity scans.
- Existing scripts varied target repetition and optimizer update counts across ablations. The new protocol uses independent, fixed target/reference sampling streams and separately switches reference CE and KD. It is explicitly separate from archived main-table runs.
- Added snapshot-qualified sample IDs, source/cache checksums, saved repaired heads, all-sample predictions, paired and common evaluation masks, per-class metrics, and completion hashes for resumability.
- A shared output layout lets sensitivity scans reuse two primary configurations without duplicating runs.
- Validation: 12 Python synthetic tests, new launcher shell tests, earlier runtime benchmark shell tests, Python 3.10 syntax parsing, and compatibility with the stored 178-class / 256-feature source checkpoint.
- Synthetic tests are implementation checks only. Real CUDA/CESNET experiment results must come from the server run.
- Keep completed results immutable. Changed data/protocol/code should use a new output directory; do not merge different protocols by filename alone.
