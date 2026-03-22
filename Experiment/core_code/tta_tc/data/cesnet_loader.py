"""
CESNET DataZoo data loading utilities.

Loads CESNET-QUIC22 and CESNET-TLS-Year22 datasets with temporal splits.
Converts cesnet-datazoo batch format to PyTorch tensors expected by TTA-TC.

cesnet-datazoo batches come as pandas DataFrames with columns:
  - "PPI"  : per-row numpy array of shape (3, 30) — sizes, directions, IPTs
  - "APP"  : integer application class label
  - flow stat columns (BYTES, DURATION, etc.) if configured
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# cesnet-datazoo column name constants
_PPI_COLS = ["PPI", "ppi"]
_APP_COLS = ["APP", "app", "label", "labels", "y"]
# Flow-level stat columns used for FSR target (6 features)
FLOW_STAT_COLS = ["BYTES", "DURATION", "PACKETS", "BYTES_REV", "PACKETS_REV", "DST_ASN"]


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


def _convert_batch(batch, use_flow_stats=False):
    """
    Convert a cesnet-datazoo batch to our expected dict format:
        {"ppi": Tensor(B,3,30), "label": Tensor(B,), "flow_stats": Tensor(B,D) or absent}

    cesnet-datazoo v0.2.x yields a tuple of 4 elements per batch:
        (empty_df, ppi_array(B,3,30), flow_stats_array(B,D), labels_array(B,))

    Older or future versions may yield dicts or DataFrames; all are handled.
    """
    import pandas as pd

    # --- Primary: tuple format (cesnet-datazoo v0.2.x) ---
    if isinstance(batch, (tuple, list)):
        # Find numpy arrays in the tuple
        arrays = [(i, v) for i, v in enumerate(batch) if isinstance(v, np.ndarray)]
        if len(arrays) >= 2:
            # Identify PPI (3-channel, seq_len=30), flow_stats, labels by shape
            ppi_arr = None
            fs_arr = None
            lbl_arr = None
            for _, arr in arrays:
                if arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] == 30:
                    ppi_arr = arr
                elif arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] == 30:
                    ppi_arr = arr.transpose(0, 2, 1)  # -> (B, 3, 30)
                elif arr.ndim == 1 and (arr.dtype == np.int64 or arr.dtype == np.int32):
                    lbl_arr = arr
                elif arr.ndim == 2:
                    fs_arr = arr   # flow stats

            # v0.2.x tuple order: (df, ppi, flow_stats, labels)
            # Fallback: assign by index if pattern matching failed
            if ppi_arr is None and len(arrays) >= 1:
                ppi_arr = arrays[0][1]
            if lbl_arr is None and len(arrays) >= 2:
                lbl_arr = arrays[-1][1]  # labels are last
            if fs_arr is None and len(arrays) >= 3:
                fs_arr = arrays[1][1]    # flow_stats are second

            result = {
                "ppi": torch.tensor(ppi_arr, dtype=torch.float32),
                "label": torch.tensor(lbl_arr.astype(np.int64), dtype=torch.long),
            }
            if use_flow_stats and fs_arr is not None:
                result["flow_stats"] = torch.tensor(fs_arr, dtype=torch.float32)
            return result

    # --- DataFrame path (older cesnet-datazoo format) ---
    if isinstance(batch, pd.DataFrame):
        ppi_col = next((c for c in _PPI_COLS if c in batch.columns), None)
        app_col = next((c for c in _APP_COLS if c in batch.columns), None)

        if ppi_col is None or app_col is None:
            raise KeyError(
                f"Cannot find PPI/label columns. Available: {list(batch.columns)}"
            )

        raw = np.stack(batch[ppi_col].values)   # (B, 3, 30) or (B, 30, 3)
        if raw.ndim == 3 and raw.shape[1] == 30 and raw.shape[2] == 3:
            raw = raw.transpose(0, 2, 1)
        ppi = torch.tensor(raw, dtype=torch.float32)
        labels = torch.tensor(batch[app_col].values.astype(np.int64), dtype=torch.long)

        result = {"ppi": ppi, "label": labels}
        if use_flow_stats:
            avail = [c for c in FLOW_STAT_COLS if c in batch.columns]
            if avail:
                result["flow_stats"] = torch.tensor(
                    batch[avail].values.astype(np.float32)
                )
        return result

    # --- Dict path ---
    if isinstance(batch, dict):
        ppi_key = next((k for k in _PPI_COLS if k in batch), None)
        app_key = next((k for k in _APP_COLS if k in batch), None)

        if ppi_key is None or app_key is None:
            raise KeyError(
                f"Cannot find PPI/label keys. Available: {list(batch.keys())}"
            )

        raw = batch[ppi_key]
        if isinstance(raw, np.ndarray):
            raw = torch.tensor(raw, dtype=torch.float32)
        if raw.ndim == 3 and raw.shape[1] == 30 and raw.shape[2] == 3:
            raw = raw.permute(0, 2, 1)

        lbl = batch[app_key]
        if isinstance(lbl, np.ndarray):
            lbl = torch.tensor(lbl.astype(np.int64), dtype=torch.long)

        result = {"ppi": raw, "label": lbl}
        if use_flow_stats:
            for fs_key in ["flow_stats", "FLOW_STATS"]:
                if fs_key in batch:
                    fs = batch[fs_key]
                    if isinstance(fs, np.ndarray):
                        fs = torch.tensor(fs, dtype=torch.float32)
                    result["flow_stats"] = fs
                    break
        return result

    raise TypeError(f"Unsupported batch type from cesnet-datazoo: {type(batch)}")


class _WrappedLoader:
    """Wraps a cesnet-datazoo DataLoader to yield our dict-format batches."""

    def __init__(self, loader, use_flow_stats=False):
        self.loader = loader
        self.use_flow_stats = use_flow_stats

    def __iter__(self):
        for batch in self.loader:
            yield _convert_batch(batch, self.use_flow_stats)

    def __len__(self):
        return len(self.loader)


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
            val_ratio: float (default 0.2)
            use_flow_stats: bool (default False)

    Returns:
        train_loader, val_loader, test_loader, num_classes
        (all yield batches with keys "ppi", "label", optionally "flow_stats")
    """
    from cesnet_datazoo.config import DatasetConfig, AppSelection

    dataset_name = cfg.get("dataset", "quic22")
    data_dir = cfg.get("data_dir", "./data")
    size = cfg.get("size", "S")
    train_period = cfg["train_period"]
    test_period = cfg["test_period"]
    use_flow_stats = cfg.get("use_flow_stats", False)

    # Import dataset class
    if dataset_name == "quic22":
        from cesnet_datazoo.datasets import CESNET_QUIC22
        dataset = CESNET_QUIC22(data_dir, size=size)
    elif dataset_name == "tls22":
        try:
            from cesnet_datazoo.datasets import CESNET_TLS_Year22
            dataset = CESNET_TLS_Year22(data_dir, size=size)
        except ImportError:
            from cesnet_datazoo.datasets import CESNET_TLS22
            dataset = CESNET_TLS22(data_dir, size=size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Configure temporal split
    # cesnet-datazoo >= 0.4: val_known_size is fraction, not count
    config_kwargs = dict(
        dataset=dataset,
        apps_selection=AppSelection.ALL_KNOWN,
        train_period_name=train_period,
        test_period_name=test_period,
        min_train_samples_per_app=100,
    )
    # val_known_size accepts int (num samples) or 'all'; we use 'all' and
    # rely on the val split that cesnet-datazoo creates by default.
    try:
        config = DatasetConfig(val_known_size="all", **config_kwargs)
    except (TypeError, Exception):
        config = DatasetConfig(**config_kwargs)

    dataset.set_dataset_config_and_initialize(config)

    # Get dataloaders from cesnet-datazoo and wrap them
    raw_train = dataset.get_train_dataloader()
    raw_val = dataset.get_val_dataloader()
    raw_test = dataset.get_test_dataloader()

    train_loader = _WrappedLoader(raw_train, use_flow_stats=use_flow_stats)
    val_loader = _WrappedLoader(raw_val, use_flow_stats=use_flow_stats)
    test_loader = _WrappedLoader(raw_test, use_flow_stats=use_flow_stats)

    # Get number of classes
    try:
        num_classes = len(dataset.get_known_apps())
    except AttributeError:
        num_classes = len(dataset.known_apps)

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
    use_flow_stats = cfg.get("use_flow_stats", False)

    if dataset_name == "quic22":
        from cesnet_datazoo.datasets import CESNET_QUIC22
        DatasetClass = CESNET_QUIC22
    elif dataset_name == "tls22":
        try:
            from cesnet_datazoo.datasets import CESNET_TLS_Year22
            DatasetClass = CESNET_TLS_Year22
        except ImportError:
            from cesnet_datazoo.datasets import CESNET_TLS22
            DatasetClass = CESNET_TLS22
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loaders = []
    last_dataset = None
    for period in test_periods:
        ds = DatasetClass(data_dir, size=size)
        config_kwargs = dict(
            dataset=ds,
            apps_selection=AppSelection.ALL_KNOWN,
            train_period_name=train_period,
            test_period_name=period,
            min_train_samples_per_app=100,
        )
        try:
            config = DatasetConfig(**config_kwargs)
        except TypeError:
            config = DatasetConfig(**config_kwargs)

        ds.set_dataset_config_and_initialize(config)
        raw_test = ds.get_test_dataloader()
        loaders.append((period, _WrappedLoader(raw_test, use_flow_stats=use_flow_stats)))
        last_dataset = ds

    try:
        num_classes = len(last_dataset.get_known_apps())
    except AttributeError:
        num_classes = len(last_dataset.known_apps)

    return loaders, num_classes
