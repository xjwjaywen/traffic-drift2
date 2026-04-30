# WST Validation: Wavelet Scattering Transform for Encrypted Traffic

48-hour minimum viability experiment for the WST direction (following Xu Shijie's path-signature recipe).

## Hypotheses

| Hypothesis | Test | Cost |
|-----------|------|------|
| H1 | Kymatio WST runs on PPI shape (3, 30) | 2h ✅ DONE |
| H2 | WST features ≥ raw PPI baseline (i.i.d.) | 4h |
| H3 | WST features more drift-robust than PPI | 8h |
| H4 | Stability bound non-vacuous on real ε | 12h |

## Run

### H2: i.i.d. comparison

```bash
cd /data/xjw/traffic-drift2/Experiment/core_code

# QUIC22
python ../../WST_validation/scripts/h2_iid_compare.py \
    --config configs/eval_quic22.yaml \
    --epochs 30 --max-samples 20000

# TLS22
python ../../WST_validation/scripts/h2_iid_compare.py \
    --config configs/eval_tls22.yaml \
    --epochs 30 --max-samples 20000
```

### H3: drift comparison

```bash
# QUIC22
python ../../WST_validation/scripts/h3_drift_compare.py \
    --config configs/eval_quic22.yaml \
    --max-train 20000 --max-test-per-period 5000 \
    --clf mlp

# TLS22 (longer drift, more interesting)
python ../../WST_validation/scripts/h3_drift_compare.py \
    --config configs/eval_tls22.yaml \
    --max-train 20000 --max-test-per-period 5000 \
    --clf mlp
```

## Decision matrix

| H2 | H3 | Action |
|----|----|--------|
| PASS | PASS | Green light — start full WST project |
| PASS | PARTIAL | Yellow — drop drift-robustness as main angle |
| PASS | FAIL | WST has no advantage over PPI on drift |
| FAIL | — | Direction dies, pivot to Plan B (Graph Scattering) |

## File structure

```
WST_validation/
├── methods/
│   └── wst_extractor.py         # Kymatio wrapper for PPI
├── scripts/
│   ├── h2_iid_compare.py        # H2: i.i.d. accuracy comparison
│   └── h3_drift_compare.py      # H3: drift-robustness comparison
├── outputs/                      # JSON results
└── README.md
```

## Notes

- Kymatio 0.3.0 has a broken 3D module under scipy>=1.16; we import 1D module directly to bypass.
- PPI length 30 is padded to 32 (next power of 2).
- WST hyperparameters: J=3 octaves, Q=4 wavelets per octave, multi-channel concat.
