# Paid-label local geometry feasibility probe

Run on the existing TMA server artifacts:

```bash
cd /data/xjw/traffic-drift2
python -u Experiment/tma_local_geometry_v1/probe.py
cat /data/xjw/tma-care-data/pilot_v1/local_geometry_probe_v1/source_0/REPORT.txt
```

This probe tests one narrowly stated premise: do the few *already paid*
positive examples of each previously examined collapsed class rank future
same-class examples highly in the frozen source representation? It compares
the source classifier's AP with two fixed cosine-neighborhood scores and
reports positive-label support. It fits no new classifier and buys no labels.

The future month labels are used to evaluate AP. All focus classes and months
were chosen after earlier results, so this is exploratory. High AP only says
that a binary ranking signal exists. It does not provide a calibrated decision
threshold, multiclass F1, or evidence for a deployable graph method. Low AP
would argue against pursuing target-anchor graph propagation with this budget.

Related prior art: Poisson learning for very low label rates
(https://proceedings.mlr.press/v119/calder20a.html), graph label propagation
under class imbalance (https://proceedings.neurips.cc/paper/2021/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html),
and deep-feature kNN adaptation to concept drift
(https://arxiv.org/abs/2107.14759). A positive probe is a reason to test
these controls, not a novelty claim.
