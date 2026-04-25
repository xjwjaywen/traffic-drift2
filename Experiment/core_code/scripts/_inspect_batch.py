"""Inspect cesnet-datazoo batch format and test our wrapper."""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from tta_tc.data.cesnet_loader import build_dataloaders
    import torch

    cfg = {
        "dataset": "quic22",
        "data_dir": "./data",
        "size": "XS",
        "train_period": "W-2022-44",
        "test_period": "W-2022-45",
        "batch_size": 256,
        "num_workers": 4,
        "val_ratio": 0.2,
        "use_flow_stats": False,
    }

    print("Building dataloaders...")
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)
    print(f"num_classes: {num_classes}")
    print(f"train batches: {len(train_loader)}")

    print("\nChecking first train batch...")
    for batch in train_loader:
        print(f"  ppi shape:   {batch['ppi'].shape}  dtype={batch['ppi'].dtype}")
        print(f"  label shape: {batch['label'].shape}  dtype={batch['label'].dtype}")
        if 'flow_stats' in batch:
            print(f"  flow_stats:  {batch['flow_stats'].shape}")
        print(f"  label range: min={batch['label'].min()}, max={batch['label'].max()}")
        print("  SUCCESS - batch format is correct!")
        break
