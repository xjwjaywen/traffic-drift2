TMA source/target disagreement query pilot

Motivation
The completed hybrid experiment shows that 500 uniform + 500 BADGE queries
improve drift-month CARE over pure BADGE, but the fixed mixture is not a new
method. A read-only head-complementarity diagnosis found some predictions where
target-only repairs classes that CARE suppresses. This pilot tests whether
queries in source/target disagreement regions supply more useful labels.

Protocol
1. Take the same first 500 saved uniform queries as hybrid_adaptive_v1.
2. Train a provisional target-only head on only these 500 paid labels.
3. Predict the remaining unlabeled pool with source and provisional heads.
4. Use BADGE on their disagreement set for the remaining 500 labels. If fewer
   than 500 disagreements exist, use all of them and fill with source BADGE.
5. Retrain target-only and CARE from the fixed source on all 1000 paid labels.

The provisional model adds computation, not labels. The final comparison with
hybrid_care shares the first 500 labels, label budget, source, references,
optimizer schedule and evaluation. The second 500 query identities differ.
No future evaluation label enters selection or training.

On the Linux server with completed paper and hybrid repair seed 0:
  cd /data/xjw/traffic-drift2
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_conflict_query_v1/experiment.py
  cat /data/xjw/tma-care-data/pilot_v1/conflict_query_v1/source_0/repair_0/REPORT.txt

Outputs are under conflict_query_v1 and do not overwrite the old experiments.
Each month has a completion marker; repeating the command resumes safely.
If repair 0 looks promising, repeat with --repair-seed 1 through 4 and aggregate
without choosing favorable seeds. The current four months have been inspected
many times, so this remains exploratory and cannot establish generalization.

This is an experimental hypothesis, not a claim of methodological novelty.
Source/target disagreement and mixed querying have related prior work and must
be compared and cited before a paper claim.
