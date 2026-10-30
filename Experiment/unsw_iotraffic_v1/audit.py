"""Read-only time and device coverage audit for the independent UNSW-IoTraffic flows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

REQUIRED = {'time', 'ipProto'}
NUMERIC_FEATURES = {
    'srcNumPackets', 'dstNumPackets', 'srcPayloadSize', 'dstPayloadSize',
    'srcAvgPayloadSize', 'dstAvgPayloadSize', 'srcMaxPayloadSize',
    'dstMaxPayloadSize', 'srcStdDevPayloadSize', 'dstStdDevPayloadSize',
    'flowDuration', 'srcAvgInterarrivalTime', 'dstAvgInterarrivalTime',
    'avgInterarrivalTime', 'srcStdDevInterarrivalTime',
    'dstStdDevInterarrivalTime', 'stdDevInterarrivalTime',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def flow_members(archive):
    members = [info for info in archive.infolist()
               if not info.is_dir() and info.filename.lower().endswith('_flows.csv')
               and not Path(info.filename).name.startswith('._')]
    devices = [Path(info.filename).name[:-len('_flows.csv')] for info in members]
    if not members or len(devices) != len(set(devices)):
        raise ValueError('Expected distinct device *_flows.csv files inside flows.zip')
    return sorted(zip(devices, members), key=lambda pair: pair[0])


def audit(zip_path, chunksize=100_000):
    if chunksize < 1:
        raise ValueError('chunksize must be positive')
    device_rows, month_counts, column_rows = [], [], []
    with zipfile.ZipFile(zip_path) as archive:
        for device, info in flow_members(archive):
            with archive.open(info) as stream:
                columns = pd.read_csv(stream, nrows=0).columns.tolist()
            if not REQUIRED.issubset(columns):
                raise ValueError(f'{info.filename}: missing {sorted(REQUIRED - set(columns))}')
            column_rows.append(dict(device=device, archive_member=info.filename,
                                    column_count=len(columns),
                                    feature_count=len(NUMERIC_FEATURES & set(columns)),
                                    missing_numeric_features='|'.join(sorted(NUMERIC_FEATURES - set(columns))),
                                    columns='|'.join(columns)))
            counts, days = Counter(), set()
            total = invalid = tcp_udp = 0
            first = last = None
            with archive.open(info) as stream:
                for chunk in pd.read_csv(stream, usecols=['time', 'ipProto'],
                                         chunksize=chunksize, low_memory=False):
                    total += len(chunk)
                    times = pd.to_datetime(pd.to_numeric(chunk.time, errors='coerce'),
                                           unit='s', utc=True, errors='coerce')
                    good = times.notna()
                    invalid += int((~good).sum())
                    valid = times[good]
                    if len(valid):
                        lo, hi = valid.min(), valid.max()
                        first = lo if first is None else min(first, lo)
                        last = hi if last is None else max(last, hi)
                        counts.update(valid.dt.strftime('%Y-%m').value_counts().to_dict())
                        days.update(valid.dt.strftime('%Y-%m-%d').unique().tolist())
                    tcp_udp += int(pd.to_numeric(chunk.ipProto, errors='coerce').isin([6, 17]).sum())
            device_rows.append(dict(device=device, rows=total, valid_time_rows=total - invalid,
                                    invalid_time_rows=invalid, tcp_udp_rows=tcp_udp,
                                    active_days=len(days),
                                    first_utc=first.isoformat() if first is not None else '',
                                    last_utc=last.isoformat() if last is not None else ''))
            month_counts.extend(dict(device=device, month=month, rows=count)
                                for month, count in sorted(counts.items()))
            print(device, 'rows', total, 'active_days', len(days), flush=True)
    return (pd.DataFrame(device_rows), pd.DataFrame(month_counts),
            pd.DataFrame(column_rows))


def write_report(zip_path, output, devices, months, columns):
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError('Output directory must be new and empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    devices.to_csv(output / 'devices.csv', index=False)
    months.to_csv(output / 'monthly_counts.csv', index=False)
    columns.to_csv(output / 'columns.csv', index=False)
    overview = months.groupby('month').agg(devices=('device', 'nunique'),
                                            rows=('rows', 'sum')).reset_index()
    (output / 'month_overview.csv').write_text(overview.to_csv(index=False))
    missing = set()
    for item in columns.missing_numeric_features:
        missing.update(name for name in item.split('|') if name)
    report = '\n'.join([
        'UNSW-IoTraffic independent-dataset coverage audit; no model training or outcome evaluation.',
        f'Archive: {zip_path}',
        f'Devices: {len(devices)}; rows: {int(devices.rows.sum())}; '
        f'invalid timestamps: {int(devices.invalid_time_rows.sum())}.',
        f'Common numeric features: {len(NUMERIC_FEATURES - missing)} '
        '(check columns.csv for per-device schema).',
        'Monthly coverage:',
        overview.to_string(index=False),
        'Do not select protocol/split dates based on future classifier scores.',
        'A valid chronological study needs shared device coverage in source and later windows.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    (output / 'manifest.json').write_text(json.dumps(dict(
        source_zip_sha256=digest(zip_path), code_sha256=digest(Path(__file__)),
        files={p.name: digest(p) for p in output.iterdir() if p.is_file()}),
        indent=2, sort_keys=True) + '\n')
    print(report, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('flows_zip', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--chunksize', type=int, default=100_000)
    args = parser.parse_args()
    path = args.flows_zip.resolve(strict=True)
    output = args.output or path.parent / 'unsw_iotraffic_audit_v1'
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error('Output directory must be new and empty: ' + str(output))
    devices, months, columns = audit(path, args.chunksize)
    write_report(path, output, devices, months, columns)


if __name__ == '__main__':
    main()
