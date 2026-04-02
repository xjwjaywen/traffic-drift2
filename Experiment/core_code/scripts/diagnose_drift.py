"""
Diagnose per-position drift in CESNET datasets.

Compares packet size/direction/IPT distributions between training period
and each test period. Outputs z-scores and KS statistics per position.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
import argparse
from scipy import stats


def collect_position_stats(loader, max_batches=200):
    """Collect per-position statistics from a data loader."""
    all_sizes = []
    all_dirs = []
    all_ipts = []
    count = 0
    for batch in loader:
        ppi = batch["ppi"]  # (B, 3, 30)
        all_sizes.append(ppi[:, 0, :].numpy())
        all_dirs.append(ppi[:, 1, :].numpy())
        all_ipts.append(ppi[:, 2, :].numpy())
        count += 1
        if count >= max_batches:
            break
    return {
        "sizes": np.concatenate(all_sizes, axis=0),
        "dirs": np.concatenate(all_dirs, axis=0),
        "ipts": np.concatenate(all_ipts, axis=0),
    }


def analyze_drift(source_stats, test_stats, period_name):
    """Compare source and test period distributions per position."""
    print(f"\n{'='*70}")
    print(f"Period: {period_name}")
    print(f"  Source samples: {source_stats['sizes'].shape[0]}, "
          f"Test samples: {test_stats['sizes'].shape[0]}")

    for channel_name, key in [("Packet Size", "sizes"),
                               ("Direction", "dirs"),
                               ("IPT", "ipts")]:
        src = source_stats[key]
        tst = test_stats[key]

        src_mean = src.mean(axis=0)
        tst_mean = tst.mean(axis=0)
        src_std = src.std(axis=0) + 1e-8

        z_scores = (tst_mean - src_mean) / src_std

        # KS test per position
        ks_stats = []
        ks_pvals = []
        for p in range(30):
            ks, pval = stats.ks_2samp(src[:, p], tst[:, p])
            ks_stats.append(ks)
            ks_pvals.append(pval)
        ks_stats = np.array(ks_stats)
        ks_pvals = np.array(ks_pvals)

        drifted_z = np.where(np.abs(z_scores) > 2.0)[0]
        drifted_ks = np.where((ks_stats > 0.05) & (ks_pvals < 0.001))[0]

        print(f"\n  --- {channel_name} ---")
        print(f"  Z-score drifted (|z|>2): {drifted_z.tolist()}")
        print(f"  KS drifted (D>0.05, p<0.001): {drifted_ks.tolist()}")

        top_z = np.argsort(np.abs(z_scores))[::-1][:10]
        print(f"  Top 10 by |z-score|:")
        print(f"  {'Pos':>4} {'Src Mean':>10} {'Tst Mean':>10} {'Z-score':>10} {'KS-stat':>10}")
        for p in top_z:
            print(f"  {p:>4} {src_mean[p]:>10.4f} {tst_mean[p]:>10.4f} "
                  f"{z_scores[p]:>10.4f} {ks_stats[p]:>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    from tta_tc.data.cesnet_loader import build_dataloaders, build_sequential_test_loaders

    print("Loading source period data...")
    _, val_loader, _, _ = build_dataloaders(data_cfg)
    source_stats = collect_position_stats(val_loader)
    print(f"Source period: {data_cfg['train_period']}, samples: {source_stats['sizes'].shape[0]}")

    print("Loading test period data...")
    test_loaders, _ = build_sequential_test_loaders(data_cfg)

    for period_name, loader in test_loaders:
        test_stats = collect_position_stats(loader)
        analyze_drift(source_stats, test_stats, period_name)


if __name__ == "__main__":
    main()
