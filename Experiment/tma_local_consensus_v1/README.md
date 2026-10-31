# Local consensus feasibility control

This experiment tests one fixed rule on the already inspected TMA months:
when cosine 1-NN and CARE disagree, use 1-NN only if the three nearest *paid
target* anchors unanimously have its predicted class. Otherwise retain CARE.
The closest historical reference alone cannot trigger a correction. No
additional labels are bought. A matched random control changes the same
number of disagreements within each CARE-to-1-NN predicted-class pair,
without using true labels.

```bash
cd /data/xjw/traffic-drift2
python -u Experiment/tma_local_consensus_v1/experiment.py
cat /data/xjw/tma-care-data/pilot_v1/local_consensus_v1/source_0/REPORT.txt
```

The predictions and random-control choices are written before the script
opens future evaluation labels. This is a simple locality control, not a
novel method. The months and classes were selected after previous TMA
results. Do not tune the neighbor count or gate on these same outcomes and
then report a confirmatory result. A method claim would additionally need
comparison to standard dynamic selection and risk-control methods, plus
independent evaluation.

To check whether the consensus rule adds information beyond the predicted
class transition, run the post-hoc matched randomization audit. It uses
future labels for evaluation and has no confirmatory status:

```bash
python -u Experiment/tma_local_consensus_v1/randomization_audit.py
cat /data/xjw/tma-care-data/pilot_v1/local_consensus_v1/source_0/randomization_audit/REPORT.txt
```
