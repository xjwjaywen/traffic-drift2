"""
CESNET DataZoo data loading utilities.

Loads CESNET-QUIC22 and CESNET-TLS-Year22 datasets with temporal splits.
Converts cesnet-datazoo format to PyTorch tensors.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CESNETFlowDataset(Dataset):
    """
    Wrapper around cesnet-datazoo data arrays.

    Expects pre-loaded numpy arrays from cesnet-datazoo.
    PPI format: 3 sequences × 30 packets (sizes, directions, IPTs).
    """

    def __init__(self, ppi_features, flow_stats=None, labels=None):
        """
        Args:
            ppi_features: np.ndarray of shape (N, 3, 30) or (N, 90)
            flow_stats: np.ndarray of shape (N, D) or None
            labels: np.ndarray of shape (N,) or None (for test-time)
        """
        if ppi_features.ndim == 2 and ppi_features.shape[1] == 90:
            # Reshape flat PPI to (N, 3, 30)
            ppi_features = ppi_features.reshape(-1, 3, 30)

        self.ppi = torch.tensor(ppi_features, dtype=torch.float32)
        self.flow_stats = (
            torch.tensor(flow_stats, dtype=torch.float32)
            if flow_stats is not None else None
        )
        self.labels = (
            torch.tensor(labels, dtype=torch.long)
            if labels is not None else None
        )

    def __len__(self):
        return len(self.ppi)

    def __getitem__(self, idx):
        item = {"ppi": self.ppi[idx]}
        if self.flow_stats is not None:
            item["flow_stats"] = self.flow_stats[idx]
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def build_dataloaders(cfg: dict):
    """
    Build train/val/test dataloaders from CESNET datasets.

    Uses cesnet-datazoo API for temporal splitting.

    Args:
        cfg: dict with keys:
            dataset: "quic22" or "tls22"
            data_dir: path to data
            size: "XS", "S", "M", "L"
            train_period: e.g. "W-2022-44"
            test_period: e.g. "W-2022-45"
            batch_size: int
            num_workers: int

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    from cesnet_datazoo.config import DatasetConfig, AppSelection

    dataset_name = cfg.get("dataset", "quic22")
    data_dir = cfg.get("data_dir", "./data")
    size = cfg.get("size", "S")
    train_period = cfg["train_period"]
    test_period = cfg["test_period"]
    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)
    val_ratio = cfg.get("val_ratio", 0.2)

    # Import dataset class
    if dataset_name == "quic22":
        from cesnet_datazoo.datasets import CESNET_QUIC22
        dataset = CESNET_QUIC22(data_dir, size=size)
    elif dataset_name == "tls22":
        from cesnet_datazoo.datasets import CESNET_TLS22
        dataset = CESNET_TLS22(data_dir, size=size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Configure temporal split
    config = DatasetConfig(
        dataset=dataset,
        apps_selection=AppSelection.ALL_KNOWN,
        train_period_name=train_period,
        test_period_name=test_period,
        val_known_size=val_ratio,
        min_train_samples_per_app=100,
    )
    dataset.set_dataset_config_and_initialize(config)

    # Get dataloaders from cesnet-datazoo
    train_loader = dataset.get_train_dataloader()
    val_loader = dataset.get_val_dataloader()
    test_loader = dataset.get_test_dataloader()

    # Get number of classes
    num_classes = len(dataset.get_known_apps())

    return train_loader, val_loader, test_loader, num_classes


def build_sequential_test_loaders(cfg: dict):
    """
    Build sequential test loaders for continual TTA evaluation.

    Returns a list of (period_name, test_loader) tuples.
    """
    from cesnet_datazoo.config import DatasetConfig, AppSelection

    dataset_name = cfg.get("dataset", "quic22")
    data_dir = cfg.get("data_dir", "./data")
    size = cfg.get("size", "S")
    train_period = cfg["train_period"]
    test_periods = cfg["test_periods"]  # list of period names
    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)

    if dataset_name == "quic22":
        from cesnet_datazoo.datasets import CESNET_QUIC22
        DatasetClass = CESNET_QUIC22
    elif dataset_name == "tls22":
        from cesnet_datazoo.datasets import CESNET_TLS22
        DatasetClass = CESNET_TLS22
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loaders = []
    for period in test_periods:
        dataset = DatasetClass(data_dir, size=size)
        config = DatasetConfig(
            dataset=dataset,
            apps_selection=AppSelection.ALL_KNOWN,
            train_period_name=train_period,
            test_period_name=period,
            min_train_samples_per_app=100,
        )
        dataset.set_dataset_config_and_initialize(config)
        loaders.append((period, dataset.get_test_dataloader()))

    num_classes = len(dataset.get_known_apps())
    return loaders, num_classes
