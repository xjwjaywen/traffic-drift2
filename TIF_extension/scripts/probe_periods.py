"""
Step 0.1: Probe which CESNET periods are actually loadable.

Approach: try to construct a sequential test loader for each candidate period
and see which ones succeed. Print all attributes of the dataset object that
might list available periods.
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))


def inspect_dataset_object(dataset_class, data_dir):
    """Print all attributes / docstrings that might tell us about periods."""
    ds = dataset_class(data_dir, size="S")
    print(f"\n=== {dataset_class.__name__} object inspection ===")
    print(f"Type: {type(ds)}")
    print(f"\nNon-private attributes:")
    for attr in sorted(dir(ds)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(ds, attr)
            if callable(val):
                continue
            sval = repr(val)
            if len(sval) > 200:
                sval = sval[:200] + "..."
            print(f"  {attr}: {sval}")
        except Exception as e:
            print(f"  {attr}: <error: {e}>")
    print()


def probe_quic22(data_dir, candidates):
    print("=" * 76)
    print("QUIC22 period probe")
    print("=" * 76)
    from cesnet_datazoo.datasets import CESNET_QUIC22

    inspect_dataset_object(CESNET_QUIC22, data_dir)

    from tta_tc.data.cesnet_loader import build_sequential_test_loaders

    print("\n=== Probing candidates by attempting load ===")
    for period in candidates:
        try:
            cfg = {
                "dataset": "quic22",
                "data_dir": data_dir,
                "size": "S",
                "train_period": "W-2022-44",
                "test_periods": [period],
                "batch_size": 256,
                "num_workers": 2,
            }
            loaders, _ = build_sequential_test_loaders(cfg)
            # Try to actually pull one batch to confirm
            ok = False
            for batch in loaders[0][1]:
                ok = True
                break
            print(f"  {period}: {'OK' if ok else 'EMPTY'}")
        except Exception as e:
            print(f"  {period}: FAIL — {type(e).__name__}: {str(e)[:120]}")


def probe_tls22(data_dir, candidates):
    print("=" * 76)
    print("TLS22 period probe")
    print("=" * 76)
    try:
        from cesnet_datazoo.datasets import CESNET_TLS_Year22
        DatasetClass = CESNET_TLS_Year22
    except ImportError:
        from cesnet_datazoo.datasets import CESNET_TLS22
        DatasetClass = CESNET_TLS22

    inspect_dataset_object(DatasetClass, data_dir)

    from tta_tc.data.cesnet_loader import build_sequential_test_loaders

    print("\n=== Probing candidates by attempting load ===")
    for period in candidates:
        try:
            cfg = {
                "dataset": "tls22",
                "data_dir": data_dir,
                "size": "S",
                "train_period": "M-2022-3",
                "test_periods": [period],
                "batch_size": 256,
                "num_workers": 2,
            }
            loaders, _ = build_sequential_test_loaders(cfg)
            ok = False
            for batch in loaders[0][1]:
                ok = True
                break
            print(f"  {period}: {'OK' if ok else 'EMPTY'}")
        except Exception as e:
            print(f"  {period}: FAIL — {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-quic", default="./Experiment/core_code/data/quic22")
    parser.add_argument("--data-dir-tls", default="./Experiment/core_code/data/tls22")
    args = parser.parse_args()

    # Candidates: weeks before W-44 and after W-47
    quic_candidates = [
        "W-2022-40", "W-2022-41", "W-2022-42", "W-2022-43",
        "W-2022-44", "W-2022-45", "W-2022-46", "W-2022-47",
        "W-2022-48",
    ]
    # TLS: months before M-3 and after M-12
    tls_candidates = [
        "M-2022-1", "M-2022-2", "M-2022-3", "M-2022-4",
        "M-2022-12",
    ]

    import multiprocessing as _mp
    if sys.platform != "win32":
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

    probe_quic22(args.data_dir_quic, quic_candidates)
    probe_tls22(args.data_dir_tls, tls_candidates)
