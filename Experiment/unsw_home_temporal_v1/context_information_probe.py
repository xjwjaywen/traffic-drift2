"""Source-to-paid-query probe for extra information in unlabeled flow context.

This is a binary oracle-subset diagnostic for two preselected true classes,
not a deployable multiclass method. It reads timestamps and feature values
through the end of the query period, never MAC values or later-period rows.
The same frozen source rows train every control; paid labels evaluate only.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit import FEATURES, sha256
from predict import verify_prepared


HERE = Path(__file__).resolve().parent
CONTEXT_COLUMNS = (
    "outPacketCount", "inPacketCount", "outByteCount", "inByteCount",
    "http", "https", "dns", "ntp", "tcp", "udp",
)
WINDOW_MINUTES = 5


def verified(folder, name):
    marker = json.loads((folder / "complete.json").read_text())
    path = folder / name
    if sha256(path) != marker["artifacts"][name]:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def time_bin(stamp):
    if (len(stamp) < 16 or stamp[4:5] != b"-" or stamp[7:8] != b"-"
            or stamp[13:14] != b":" or not stamp[11:13].isdigit()
            or not stamp[14:16].isdigit()):
        raise ValueError(f"Unexpected timestamp: {stamp!r}")
    minute = int(stamp[14:16])
    if minute >= 60:
        raise ValueError(f"Unexpected minute: {stamp!r}")
    return stamp[:14] + f"{minute // WINDOW_MINUTES * WINDOW_MINUTES:02d}".encode()


def stream_context(archive, member, last_query_day, selected_ids):
    """Return leave-one-row-out context for selected rows, using all flows."""
    bins = defaultdict(lambda: [0, np.zeros(len(CONTEXT_COLUMNS), np.float64)])
    selected = {}
    with zipfile.ZipFile(archive) as files, files.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").strip().split(",")
        if header[0] != "start" or header[-1] != "device":
            raise ValueError("Unexpected archive schema")
        indices = [header.index(name) for name in CONTEXT_COLUMNS]
        for row_id, line in enumerate(stream):
            stamp = line.split(b",", 1)[0]
            if stamp[:10].decode("ascii") > last_query_day:
                break  # Do not parse later-period feature, time, or label.
            fields = line.rstrip(b"\r\n").split(b",")
            if len(fields) != len(header):
                raise ValueError(f"Malformed CSV row {row_id}")
            value = np.array([float(fields[i]) if fields[i] else 0.
                              for i in indices], dtype=np.float64)
            value[~np.isfinite(value)] = 0.
            value[:4] = np.log1p(np.maximum(value[:4], 0.))
            key = time_bin(stamp)
            bucket = bins[key]
            bucket[0] += 1
            bucket[1] += value
            if row_id in selected_ids:
                selected[row_id] = (key, value)
    if selected.keys() != selected_ids:
        raise ValueError(f"Missing selected row IDs: {len(selected_ids - selected.keys())}")
    context = {}
    for row_id, (key, own) in selected.items():
        count, total = bins[key]
        other_mean = (total - own) / (count - 1) if count > 1 else np.zeros_like(own)
        context[row_id] = np.r_[np.log1p(count - 1), other_mean]
    return context


def classifier(x, y):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1., class_weight="balanced", max_iter=700, random_state=0)).fit(x, y)


def scores(y, prob):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(average_precision_score(y, prob)), float(roc_auc_score(y, prob))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--class-a", type=int, default=6)
    parser.add_argument("--class-b", type=int, default=9)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    args = parser.parse_args()
    if not (0 <= args.class_a < 24 and 0 <= args.class_b < 24
            and args.class_a != args.class_b):
        parser.error("Select two distinct class IDs in 0..23")
    archive = args.archive.resolve(strict=True)
    expected = json.loads((HERE / "metadata/audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    prep = args.prepared.resolve(strict=True) / args.home
    base = args.baseline.resolve(strict=True) / args.home
    verify_prepared(prep)
    members = pd.read_csv(HERE / "metadata/members.csv").set_index("home")
    windows = pd.read_csv(HERE / "metadata/windows.csv").set_index(["home", "split"])
    member = str(members.loc[args.home, "member"])
    last_query_day = str(windows.loc[(args.home, "query"), "last_day"])

    with np.load(verified(prep, "source.npz"), allow_pickle=False) as data:
        source_ids = data["ids"]
        source_x = data["x"]
        source_y = data["y"]
    with np.load(verified(prep, "query.npz"), allow_pickle=False) as data:
        query_ids = data["ids"]
        query_x = data["x"]
    paid_labels = np.load(verified(prep, "query_labels.npy"), mmap_mode="r")
    source_keep = np.isin(source_y, [args.class_a, args.class_b])
    source_ids = source_ids[source_keep]
    source_x = source_x[source_keep]
    source_y = source_y[source_keep]
    query_selected = []
    for seed in range(5):
        chosen = np.load(verified(base / f"seed_{seed}",
                                  "paid_query_pool_indices.npy"))
        if len(chosen) != 1000 or len(np.unique(chosen)) != 1000:
            raise ValueError(f"Paid query indices differ: {seed}")
        keep = np.isin(paid_labels[chosen], [args.class_a, args.class_b])
        query_selected.append(chosen[keep])
    selected_ids = set(map(int, source_ids))
    selected_ids.update(map(int, query_ids[np.unique(np.concatenate(query_selected))]))
    context = stream_context(archive, member, last_query_day, selected_ids)
    source_context = np.stack([context[int(i)] for i in source_ids])
    source_input = np.nan_to_num(source_x, nan=0., posinf=0., neginf=0.)
    source_input = np.sign(source_input) * np.log1p(np.abs(source_input))
    y_train = (source_y == args.class_b).astype(int)
    models = {
        "flow": classifier(source_input, y_train),
        "context": classifier(source_context, y_train),
        "flow_plus_context": classifier(
            np.column_stack((source_input, source_context)), y_train),
    }
    rows = []
    for seed, chosen in enumerate(query_selected):
        y = (np.asarray(paid_labels[chosen]) == args.class_b).astype(int)
        x = np.nan_to_num(query_x[chosen], nan=0., posinf=0., neginf=0.)
        x = np.sign(x) * np.log1p(np.abs(x))
        c = np.stack([context[int(i)] for i in query_ids[chosen]])
        for method, model in models.items():
            features = c if method == "context" else (
                x if method == "flow" else np.column_stack((x, c)))
            ap, auc = scores(y, model.predict_proba(features)[:, 1])
            rows.append(dict(seed=seed, method=method, paid_rows=len(y),
                             positives=int(y.sum()), prevalence=float(y.mean()),
                             ap=ap, auc=auc))
    result = pd.DataFrame(rows)
    print("Exploratory source-to-paid-query binary information probe.")
    print("Only the preselected true-class pair is evaluated: an oracle subset.")
    print("Same frozen source rows train each logistic control; no paid label trains it.")
    print("5-minute context excludes the candidate row and reads no MAC values.")
    print("Seeds overlap and are not independent; development/final data unopened.")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
