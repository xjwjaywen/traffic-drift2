# UNSW 12-home temporal drift audit

The [official IPFIX dataset](https://iotanalytics.unsw.edu.au/homedataset.html)
contains 12 homes observed over 47 days. Its [source paper](https://arxiv.org/abs/2301.06695)
reports temporal degradation of device identification within homes. This is a
closer independent setting for traffic drift than the CPS label-shift dataset.

Download the `all.zip` archive outside the repository, then audit timestamps:

```bash
mkdir -p /data/xjw/tma-care-data/unsw_home_temporal
curl -fL --retry 3 \
  https://iotanalytics.unsw.edu.au/homedataset/all.zip \
  -o /data/xjw/tma-care-data/unsw_home_temporal/all.zip
python -u Experiment/unsw_home_temporal_v1/audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_home_temporal_audit_v1/REPORT.txt
```

The audit decodes timestamps only. It checks the 28 flow-statistic and protocol
columns used by the source paper; MAC addresses, IP addresses, ports, and device
labels are excluded from model inputs. The chronological boundaries were fixed
after reviewing timestamp coverage. Device labels from the final test window
must remain unopened until the method and comparisons are frozen.

The timestamp audit has now been reviewed; the exact per-home windows are in
[`metadata/windows.csv`](metadata/windows.csv). The public archive contains
32,117,175 rows before filtering to the official IoT inventory. The published
`devices.xlsx` was downloaded separately (SHA256
`b1bf51fdc8b5f721c5c7360ec4a00c98e2f83748033eb7fe3afd7957536fdc4f`)
and its 24-type/home mapping is in
[`metadata/official_devices.csv`](metadata/official_devices.csv). The source
audit counts device labels only inside the frozen source windows:

```bash
python -u Experiment/unsw_home_temporal_v1/source_audit.py \
  /data/xjw/tma-care-data/unsw_home_temporal/all.zip
cat /data/xjw/tma-care-data/unsw_home_source_audit_v1/REPORT.txt
```

See [`PROTOCOL.md`](PROTOCOL.md) for the evaluation boundary and the important
distinction between a curated 24-device benchmark and a mixed-traffic deployment
stream.
