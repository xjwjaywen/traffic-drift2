Independent IoT data feasibility check: UNSW-IoTraffic flows

Dataset: https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b
Publisher describes 27 device-specific flow CSVs, 4.94M flows and about 203 days.
Only flows.zip (~95.5 MB compressed) is needed for this audit; do not download PCAPs.
The dataset is independent of the TMA experiment and has different flow features.

On the server, first obtain flows.zip from the official Dryad page and put it at:
  /data/xjw/tma-care-data/unsw_iotraffic/flows.zip
Direct anonymous file downloads returned HTTP 403 in the development environment;
if the server also gets 403, use the official Dryad browser download and copy the file.

Then run:
  cd /data/xjw/traffic-drift2
  python -u Experiment/unsw_iotraffic_v1/audit.py \
    /data/xjw/tma-care-data/unsw_iotraffic/flows.zip

The script streams the ZIP without extraction and writes an independent audit next
to the ZIP in unsw_iotraffic_audit_v1/. Return its REPORT.txt and, if needed,
month_overview.csv and devices.csv. It only reads timestamps, protocol numbers,
file identities and column names. It does not fit or evaluate a classifier.

After seeing date/device coverage, freeze source/query/evaluation windows and
feature mapping before any predictive results. The TMA 500/500 mixture and CARE
head will then be compared against uniform, BADGE and target-only controls under
the same label budget. No outcome from this dataset should tune the method.
