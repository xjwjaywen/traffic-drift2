"""Bounded, chronological TMA pilot cache. Only the CLI reads server data."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tma_server_v1'))
from stage0 import ForwardReader, sha, write_json

FEATURES = ['bytes_out', 'num_pkts_out', 'bytes_in', 'num_pkts_in',
            'f_ipt_mean', 'f_ipt_std', 'f_ipt_var', 'f_ipt_skew', 'f_ipt_kurtosis',
            'f_b_mean', 'f_b_std', 'f_b_var', 'f_b_skew', 'f_b_kurtosis', 'duration', 'pr']
MONTHS = ['2020-01', '2020-02', '2020-03', '2020-04']


def implementation():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))
            if not p.name.startswith('test_')}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def period_masks(dt):
    def between(a, b):
        return ((dt >= a) & (dt < b)).to_numpy()
    masks = {
        'train': between('2019-11-01', '2019-11-22'),
        'val': between('2019-11-23', '2019-12-01'),
        'reference': between('2019-12-01', '2019-12-15'),
        'history': between('2019-12-16', '2020-01-01'),
    }
    for month in MONTHS:
        end = (pd.Timestamp(month + '-01') + pd.offsets.MonthBegin()).strftime('%Y-%m-%d')
        masks[month + '_query'] = between(month + '-01', month + '-15')
        masks[month + '_eval'] = between(month + '-16', end)
    return masks


class BottomK:
    """Deterministic uniform hash sample of unique full-record fingerprints."""
    def __init__(self, cap):
        self.cap = cap
        self.data = None

    def add(self, ids, x, y, times):
        if not len(ids):
            return
        incoming = dict(ids=ids, x=x, y=y, times=times)
        if self.data is not None:
            incoming = {k: np.concatenate([self.data[k], v]) for k, v in incoming.items()}
        _, unique = np.unique(incoming['ids'], return_index=True)
        if len(unique) > self.cap:
            priority = incoming['ids'][unique] ^ np.uint64(20261029)
            unique = unique[np.argpartition(priority, self.cap - 1)[:self.cap]]
        self.data = {k: v[unique] for k, v in incoming.items()}


def validate_complete(root, signature):
    marker = root / 'complete.json'
    if not marker.exists():
        config = root / 'config.json'
        if config.exists() and json.loads(config.read_text()) != signature:
            raise ValueError('Configuration changed; use a new --output-root')
        return False
    info = json.loads(marker.read_text())
    if info['signature'] != signature:
        raise ValueError('Configuration/code/data changed; use a new --output-root')
    for name, expected in info['artifacts'].items():
        if sha(root / name) != expected:
            raise ValueError('Changed artifact: ' + str(root / name))
    return True


def finish(root, signature, names):
    write_json(root / 'complete.json', dict(signature=signature,
               artifacts={n: sha(root / n) for n in names}))


def prepare(archive, root, caps):
    archive_hash = sha(archive)
    manifest = json.loads(archive.with_name(archive.name + '.manifest.json').read_text())
    if manifest['sha256'] != archive_hash:
        raise ValueError('Archive checksum mismatch')
    signature = dict(version='tma-pilot-v1', archive_sha256=archive_hash,
                     caps=caps, features=FEATURES, implementation=implementation())
    root.mkdir(parents=True, exist_ok=True)
    if validate_complete(root, signature):
        print('Verified sample cache reused', flush=True)
        return
    write_json(root / 'config.json', signature)
    buckets, counts, exclusions, raw_rows = {}, Counter(), Counter(), 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith('.csv'):
                continue
            stream = io.BufferedReader(ForwardReader(tar.extractfile(member)))
            for df in pd.read_csv(stream, dtype=str, chunksize=100000):
                if not set(FEATURES + ['time_start', 'deviceId']).issubset(df.columns):
                    raise ValueError('Missing required fields')
                raw_rows += len(df)
                dt = pd.to_datetime(df.time_start, utc=True, errors='coerce')
                y = pd.to_numeric(df.deviceId, errors='coerce').to_numpy(dtype=float)
                pr = pd.to_numeric(df.pr, errors='coerce').to_numpy(dtype=float)
                good_label = np.isfinite(y) & (y == np.floor(y)) & (y >= 1) & (y <= 41)
                good_time = dt.notna().to_numpy()
                tcpudp = np.isin(pr, [6, 17])
                exclusions['invalid_time_or_label'] += int((~(good_label & good_time)).sum())
                exclusions['non_tcp_udp'] += int((~tcpudp).sum())
                exclusions['outside_nov_apr'] += int(((dt < '2019-11-01') | (dt >= '2020-05-01')).sum())
                valid = good_label & good_time & tcpudp
                # Every raw field participates, including time and device. Endpoint IDs
                # participate only in exact-record identity, never as model features.
                ids = pd.util.hash_pandas_object(df.fillna('<MISSING>'), index=False).to_numpy(dtype=np.uint64)
                x = df[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
                x[~np.isfinite(x)] = np.nan
                times = dt.astype('int64').to_numpy()
                assigned = np.zeros(len(df), dtype=bool)
                for role, mask in period_masks(dt).items():
                    mask &= valid
                    assigned |= mask
                    for c in np.unique(y[mask]).astype(int):
                        counts[role, int(c)] += int((mask & (y == c)).sum())
                    if role in ('train', 'reference'):
                        for c in np.unique(y[mask]).astype(int):
                            sel = mask & (y == c)
                            key = (role, int(c))
                            buckets.setdefault(key, BottomK(caps[role])).add(ids[sel], x[sel], y[sel], times[sel])
                    else:
                        buckets.setdefault((role, 0), BottomK(caps['query'] if role.endswith('_query') else caps['eval'])).add(
                            ids[mask], x[mask], y[mask], times[mask])
                exclusions['unassigned_including_purge_and_invalid'] += int((~assigned).sum())
                if raw_rows % 1000000 == 0:
                    print(f'Prepared {raw_rows:,} raw rows', flush=True)
    roles = sorted({key[0] for key, bucket in buckets.items() if bucket.data is not None})
    expected_roles = set(period_masks(pd.Series(pd.to_datetime(['2020-01-01'], utc=True))))
    if set(roles) != expected_roles:
        raise ValueError('Missing sample periods: ' + str(expected_roles - set(roles)))
    artifacts, selected_counts, all_ids = [], [], []
    for role in roles:
        parts = [b.data for (r, c), b in sorted(buckets.items()) if r == role and b.data is not None]
        arrays = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
        order = np.argsort(arrays['ids'])
        arrays = {k: v[order] for k, v in arrays.items()}
        arrays['y'] = arrays['y'].astype(np.int64)
        np.savez_compressed(root / (role + '.npz'), **arrays)
        artifacts.append(role + '.npz')
        all_ids.append(arrays['ids'])
        for c, n in zip(*np.unique(arrays['y'], return_counts=True)):
            selected_counts.append(dict(role=role, device_id=int(c), selected=int(n),
                                        raw_eligible=counts[role, int(c)]))
    combined = np.concatenate(all_ids)
    if len(np.unique(combined)) != len(combined):
        raise ValueError('Duplicate fingerprint across sample periods')
    write_csv(root / 'sample_counts.csv', selected_counts)
    write_json(root / 'preparation.json', dict(raw_rows=raw_rows, overlapping_exclusion_counts=dict(exclusions),
        features=FEATURES, note='Global hash samples for validation/history/query/eval; class-capped training/reference. '
        'Exact-record duplicates suppressed in selected samples via 64-bit fingerprints. '
        'Not a global duplicate census or proof of independent sessions; 24-hour temporal purge reduces boundary overlap. '
        'TCP/UDP device identification, not verified encrypted traffic.'))
    finish(root, signature, artifacts + ['sample_counts.csv', 'preparation.json'])
    print('Sample cache complete:', root, flush=True)


def load(root, role):
    with np.load(root / (role + '.npz'), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


class Preprocessor:
    def fit(self, x):
        z = np.sign(x) * np.log1p(np.abs(x))
        self.median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0
                                for col in z.T], dtype=np.float32)
        z = np.where(np.isfinite(z), z, self.median)
        self.mean = z.mean(0)
        self.std = np.maximum(z.std(0), 1e-4)
        return self

    def transform(self, x):
        missing = ~np.isfinite(x)
        z = np.sign(x) * np.log1p(np.abs(x))
        z = np.where(missing, self.median, z)
        return np.concatenate([np.clip((z - self.mean) / self.std, -10, 10), missing.astype(np.float32)], axis=1).astype(np.float32)
