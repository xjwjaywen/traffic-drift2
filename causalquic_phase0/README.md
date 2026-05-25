# CausalQUIC-Bench Phase-0

This folder contains the early-stop validation code for the CausalQUIC-Bench idea.
The goal is not to build the final paper system yet. It answers one question:

> Do QUIC temporal drift windows contain source-separable, calibrated, operationally useful signals beyond always masking handshake features?

## Phase-0 Gates

Pass Phase-0 only if the run produces evidence for all of the following:

1. At least 30 high-confidence non-Google drift windows.
2. At least 3 probe-separable source classes, not just variants of the same handshake artifact.
3. Event/changepoint alignment is stronger than placebo and stable under rollout lag.
4. Label calibration, policy training, and policy evaluation can be split by service/provider/month.
5. Source-aware actions improve actual recovery metrics on held-out windows.
6. Always no-handshake masking does not Pareto dominate the source-aware policy.
7. Unknown is treated as a conservative class instead of forcing attribution.

## Install

```bash
cd causalquic_phase0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test

The smoke test generates a tiny synthetic QUIC-like dataset and runs the pipeline.

```bash
python scripts/make_synthetic_quic.py --out data/synthetic_quic.csv
python scripts/run_phase0.py \
  --input-glob "data/synthetic_quic.csv" \
  --out-dir outputs/smoke \
  --time-col time \
  --service-col service \
  --target-col service \
  --bin-size 1D \
  --min-flows-per-bin 20
```

Expected outputs:

- `outputs/smoke/window_metrics.csv`
- `outputs/smoke/candidate_windows.csv`
- `outputs/smoke/action_policy_eval.csv`
- `outputs/smoke/gate_report.md`

## Run On QUICEXT-25-Like Flow Records

Place CSV or Parquet files on the server, then run:

```bash
python scripts/run_phase0.py \
  --input-glob "/path/to/quicext25/**/*.parquet" \
  --out-dir outputs/quicext25_phase0 \
  --time-col <TIME_COLUMN> \
  --service-col <SERVICE_OR_APP_COLUMN> \
  --target-col <SERVICE_OR_APP_COLUMN> \
  --provider-col DST_ASN \
  --country-col DST_COUNTRY \
  --bin-size 1D \
  --sample-rows 2000000
```

If the dataset uses CSV:

```bash
python scripts/run_phase0.py \
  --input-glob "/path/to/quicext25/**/*.csv" \
  --out-dir outputs/quicext25_phase0 \
  --time-col <TIME_COLUMN> \
  --service-col <SERVICE_OR_APP_COLUMN> \
  --target-col <SERVICE_OR_APP_COLUMN>
```

## Optional External Events

If CT-log, provider, outage, or release events are available, add a CSV with:

```text
service,time,source,event_key,confidence
youtube,2025-03-10,certificate,ct_san_chain_change,0.9
```

and pass:

```bash
--events-csv /path/to/events.csv
```

The current script uses events only for alignment diagnostics. It does not treat them as ground truth.

## Interpretation

High-confidence source labels are still weak labels. They should be described as
`confidence-scored auditable event attributions`, not causal ground truth.

If source probes cannot separate certificate/protocol/provider/network shifts, downgrade the project to a QUIC handshake-shortcut measurement artifact.
