# Matched-label nearest-anchor control

Run on the server after the paid-label geometry probe and the frozen RF
control have completed:

```bash
cd /data/xjw/traffic-drift2
python -u Experiment/tma_nearest_anchor_v1/experiment.py
cat /data/xjw/tma-care-data/pilot_v1/nearest_anchor_v1/source_0/REPORT.txt
```

This standard cosine 1-nearest-neighbor classifier uses exactly the existing
1000 hybrid paid labels per month/seed plus the same five historical reference
features per source class used by CARE and frozen RF. It predicts future rows
using the frozen source encoder. The output compares static MLP, hybrid CARE,
frozen RF and this control using the same TMA metric definition. No additional
labels are purchased. The nearest-anchor predictions are written before
future evaluation labels are opened by this script.

The two target months, focus classes and frozen representation have already
been inspected repeatedly. This is mechanism exploration, not a blind test.
Nearest-neighbor adaptation to concept drift is prior work
(https://arxiv.org/abs/2107.14759); no novelty claim follows from a good score.
