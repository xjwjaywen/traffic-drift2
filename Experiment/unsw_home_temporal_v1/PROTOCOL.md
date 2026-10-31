# UNSW 12-home time and scope protocol (pre-model)

This is a new evaluation on the [public 12-home IPFIX archive](https://iotanalytics.unsw.edu.au/homedataset.html),
not a reproduction of the sample filtering in the [2023 source paper](https://arxiv.org/abs/2301.06695).
The archive SHA256 is `c267ba1e69f2bfc119f94022ef5d66476d43e8e49f0d120b4c8342c8355dd87c`.
The public archive has 32,117,175 CSV rows before class filtering, whereas
the source paper reports about 6.3 million experimental flows. Do not equate
these counts.

## Time and information boundary

For each home, sort its distinct observed calendar dates. Split the dates at
`floor(0.50 D)`, `floor(0.70 D)`, and `floor(0.85 D)` into source, adaptation
query, development evaluation, and final test. Dates and row counts are fixed
in [`metadata/windows.csv`](metadata/windows.csv). No row from a later date
enters an earlier stage. The final-window device labels stay unopened until a
new method, all baselines, hyperparameters, and analysis code have been fixed
using development windows. The timestamp audit read no device-label values;
the subsequent source audit decoded labels only from the source windows.

The official [`devices.xlsx`](https://iotanalytics.unsw.edu.au/homedataset/devices.xlsx)
maps 24 IoT device types to MAC addresses in each home. The public CSVs also
contain many other MAC addresses. The *curated closed-set benchmark* consists
of flows whose device MAC appears in that home's official 24-device inventory.
This membership is supplied by dataset curation, not inferred by the model.
The model input excludes MAC addresses, IP addresses, ports, the `device`
column, and timestamps. Candidate-pool membership is therefore an idealized
benchmark assumption and must be disclosed; it is not a deployable IoT
filter. An all-flow/open-set stress test would require a separate protocol.

Primary units are homes 01–09, 11, and 12. The source period contains all 24
official types in those 11 homes. Home 10 has only 22 official types in its
source window and is reserved for a separately reported open-vocabulary
stress test. This decision uses source labels only.

## Features and comparison

Use the source paper's 11 activity statistics in each direction and the six
protocol indicators HTTP, HTTPS, DNS, NTP, TCP, and UDP, totaling 28 features.
The exact names are checked in `audit.py`. For the first baseline, retain a
uniform reservoir of at most 5,000 source rows per official class per home,
using source seed 0; retain every row of smaller classes. Fit signed-log
standardization on that reservoir only. Train the existing 256/128 tabular
MLP for 30 epochs. This is a new tabular port, not the manuscript's source
CNN. Keep a label-independent uniform reservoir of at most 100,000 curated
query-window rows per home. For repair, select 1,000 rows from that pool
uniformly without replacement
for each of repair seeds 0–4. This choice avoids query-policy confounding in
the first comparison. Select five historical references per source class from
the same reservoir. Fit target-only and CARE heads for 500 fixed optimizer
steps using the existing pilot implementation. Fit a 200-tree frozen-feature
random forest and cosine 1-NN on the same paid labels and references. No
method may access query labels outside its selected 1,000 rows. Development
and final labels cannot inform query selection or training.

The first comparison is static source MLP, target-only linear head, CARE,
frozen-feature random forest, and frozen-feature cosine 1-NN under the same
budget. Report macro F1 over source-vocabulary types present in the
evaluated window, per-class F1/recall/support, zero-recall class count, and
negative-flip rate relative to the static source MLP. Treat homes as
independent units; paired repair seeds are repeats, not independent homes.
Show the macro-F1/negative-flip tradeoff rather than choosing a method from
one metric. A new method must beat relevant standard controls on development
before opening the final window.

The detailed implementation and its hash must be fixed before any development
labels are decoded. This document does not authorize a final-test run.
