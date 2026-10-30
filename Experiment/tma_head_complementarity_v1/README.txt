TMA target-only / CARE complementary prediction diagnosis

The earlier gate used the source model's predicted class. A collapsed class that
the source model never predicts cannot activate that gate. This diagnostic asks
whether the target-only head's predicted class is a more useful routing key.

No new training or labels. It reads the completed hybrid repair seeds 0--4 and
the already-seen future evaluation predictions/labels. For each seed, month and
target-predicted class, it counts target-only versus CARE correct predictions
among rows on which the two disagree. A label-oracle then routes by that class.

IMPORTANT: Evaluation labels are used to decide the route. The routed score is
deliberately contaminated and must never be reported as a valid method result.
This only tests whether complementarity is large enough to justify developing
an out-of-fold gate using paid query labels, with independent validation later.

On the server, after pulling codex/tma-selective-repair:
  python -u Experiment/tma_head_complementarity_v1/diagnose.py
  cat /data/xjw/tma-care-data/pilot_v1/head_complementarity_v1/source_0/REPORT.txt

Outputs include oracle_scores.csv, oracle_mean_sd.csv and
predicted_class_disagreements.csv. The command is read-only for existing
experiment outputs and writes to a new diagnosis directory.
