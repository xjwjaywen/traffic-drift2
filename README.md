# CARE: Collapse-Aware Active Repair for Encrypted Traffic Classification

Encrypted traffic classifiers degrade over time due to concept drift. We discover that this degradation follows a systematic **absorber-collapse pattern**: a few dominant classes absorb predictions from victim classes, causing them to collapse to near-zero recall.

We propose **CARE**, a targeted repair framework that detects collapse candidates without labels and repairs them with 1,000 target labels plus source-period replay samples via margin selection, head-only fine-tuning, and knowledge distillation.

## Key Results (CESNET-TLS-Year22, M-2022-12)

| Method | Macro-F1 | Collapse-Class F1 | Collapsed Classes |
|--------|----------|-------------------|-------------------|
| Static (no adaptation) | 0.629 | 0.028 | 12/178 |
| SAR (best TTA) | 0.629 | 0.026 | 11/178 |
| **CARE-BADGE (ours, best)** | **0.700±0.003** | **0.333±0.040** | **4.0/178** |
| CARE-Margin (ours, lightweight) | 0.686±0.001 | 0.256±0.009 | 5.8/178 |

All 5 TTA methods fail on class collapse (F1 < 0.03). CARE is a framework: BADGE is the strongest selector (~20min); margin is the low-latency alternative (<1s). Both use all-class replay with 1,000 target labels.

## Repository Structure

```
├── Publication/              # Paper draft and figures
│   ├── paper/main_v3.tex     # LaTeX source (CARE paper, latest)
│   └── figures/              # 5 publication figures
│
├── Experiment/core_code/     # Main experiment codebase
│   ├── train.py              # Model training (joint CLS + SSL)
│   ├── evaluate_tta.py       # TTA baseline evaluation
│   ├── tta_tc/               # Core library
│   │   ├── models/           # CNN/Transformer encoders, Y-shaped model
│   │   ├── baselines/        # TTA baselines (Tent, EATA, SAR, etc.)
│   │   ├── ssl_tasks/        # MPFP, POP, FSR self-supervised tasks
│   │   ├── tta/              # Adaptation engines
│   │   ├── data/             # CESNET DataZoo loaders
│   │   └── utils/            # Metrics, config
│   ├── scripts/
│   │   ├── collapse_active_maintenance_tls22.py  # ★ CARE main method
│   │   ├── eval_baselines_with_groups.py          # Baseline group metrics
│   │   ├── make_paper_figures.py                  # Paper figure generation
│   │   ├── extract_collapse_active_classes.py     # Class group extraction
│   │   └── ...               # Analysis and diagnostic scripts
│   ├── configs/              # YAML experiment configs
│   └── outputs/              # Experiment results (CSV/JSON)
│
├── 00_项目总览.md             # Project overview (Chinese)
├── 01_文献综述.md             # Literature survey
├── 02_技术方案.md             # Technical design (original TTA-TC)
└── 03_实验指南.md             # Experiment guide
```

## Quick Start

### Prerequisites
```bash
conda create -n traffic-ncde python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install cesnet-datazoo scikit-learn pandas pyyaml tqdm matplotlib
```

### Train source model
```bash
cd Experiment/core_code
python train.py --config configs/train_tls22_cnn.yaml
```

### Evaluate TTA baselines (with collapse/stable group metrics)
```bash
python scripts/eval_baselines_with_groups.py \
    --config configs/eval_tls22.yaml \
    --checkpoint outputs/tls22_cnn/best_model.pt \
    --test-period M-2022-12 \
    --output-dir outputs/baselines_group_metrics_M12
```

### Run CARE repair (recommended: all-class replay)
```bash
# Margin selector (lightweight, <1s): macro 0.686
bash scripts/run_primary_experiments.sh p1

# BADGE selector (strongest, ~20min): macro 0.700
bash scripts/run_primary_experiments.sh p4
```

### Run CARE repair (diagnostic/oracle mode)
```bash
# Uses hardcoded oracle collapse/absorber class lists — for ablation only
python scripts/collapse_active_maintenance_tls22.py \
    --config configs/eval_tls22.yaml \
    --checkpoint outputs/tls22_cnn/best_model.pt \
    --reference-period M-2022-4 \
    --target-period M-2022-12 \
    --strategies "random,absorber_margin" \
    --budgets "200,500,1000" \
    --replay-mode all --replay-per-class 5 --target-repeat 2 \
    --replay-distill-weight 0.5 --distill-temperature 2.0 \
    --output-dir outputs/care_tls22_M12
```

### Generate paper figures
```bash
python scripts/make_paper_figures.py --output-dir ../../Publication/figures
```

## Datasets

| Dataset | Protocol | Classes | Duration | Source |
|---------|----------|---------|----------|--------|
| CESNET-TLS-Year22 | TLS | 178 | 12 months | [Zenodo](https://zenodo.org) |
| CESNET-QUICEXT-25 | QUIC | 44 | 12 months | [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo) |

## Research Note

This project explored multiple directions before converging on CARE:
- SSL-based TTA (MPFP/POP/FSR reconstruction) — gradients not aligned with classification
- Kalman prototype tracking — marginal improvement (+0.3% AURC)
- Graph-based label propagation — insufficient same-class neighbors per batch
- IRM invariant training — hurt majority classes

These negative results informed the CARE design: the problem is **class-level collapse**, not global distribution shift, so the solution must be **class-targeted**.
