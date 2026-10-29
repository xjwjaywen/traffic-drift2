"""Describe coverage from verified stage-0 tables; no downloads or model fitting."""
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path


MONTHS = ['2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04']
INPUTS = ['data_audit.json', 'monthly_class_counts.csv', 'weekly_class_counts.csv']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_counts(path, period_key):
    result = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            period, device, n = row[period_key], int(row['device_id']), int(row['rows'])
            if n <= 0 or not 1 <= device <= 41 or (period, device) in result:
                raise ValueError('Invalid or duplicate counts in ' + str(path))
            result[period, device] = n
    return result


def describe(monthly, weekly, min_support=50):
    """Source vocabulary uses November only; future coverage never selects it."""
    if min_support < 1:
        raise ValueError('min_support must be positive')
    source = sorted(c for (m, c), n in monthly.items() if m == MONTHS[0])
    if not source:
        raise ValueError('No November source records')
    # Keep all historically seen IDs, including rare ones. A support threshold is
    # descriptive here, not a claim that these classes are learned or evaluable.
    seen = set(source)
    universe = sorted({c for (m, c) in monthly if MONTHS[0] <= m <= MONTHS[-1]})
    weeks = []
    d = date(2019, 11, 1)
    d -= timedelta(days=d.weekday())
    while d < date(2020, 5, 1):
        weeks.append(d.strftime('%G-W%V'))
        d += timedelta(days=7)

    def entry(period, counts):
        present = {c for (p, c), n in counts.items() if p == period and n > 0}
        return dict(period=period, rows=sum(n for (p, c), n in counts.items() if p == period),
                    present_ids=sorted(present), absent_source_ids=sorted(seen - present),
                    present_source_below_support_ids=sorted(
                        c for c in seen & present if counts[period, c] < min_support),
                    present_source_at_support_ids=sorted(
                        c for c in seen & present if counts[period, c] >= min_support),
                    unseen_source_ids=sorted(present - seen))

    return dict(
        source_month=MONTHS[0], source_ids=source, descriptive_min_support=min_support,
        source_below_support_ids=[c for c in source if monthly[MONTHS[0], c] < min_support],
        monthly=[entry(m, monthly) for m in MONTHS],
        weekly=[entry(w, weekly) for w in weeks],
        device_month_counts=[dict(device_id=c, **{m: monthly.get((m, c), 0) for m in MONTHS})
                             for c in universe],
        outside_primary_months=[dict(month=m, device_id=c, rows=n)
                                for (m, c), n in sorted(monthly.items()) if m not in MONTHS],
        limitations=[
            'Counts describe observed coverage, not device removal, label-shift correction, concept drift or classifier collapse.',
            'Weekly tables include boundary days in October/May; they cannot isolate November-April exactly.',
            '1970 records must be excluded by timestamp in a later row-level experiment; original audit files are unchanged.',
            'Zero target support makes class recall undefined; never count an absent class as a collapse.',
            'Low support is descriptive, not an automatic exclusion from prediction or acquisition.',
            'Future-label summaries are offline diagnostics, never acquisition signals or tuning data.',
            'Support counts precede deduplication and train/query/test splitting; recompute support on actual held-out sets.',
            'Historically unseen IDs require separate open-set reporting, not known-class collapse accounting.',
        ])


def run(audit_dir, min_support=50):
    marker = json.loads((audit_dir / 'audit_complete.json').read_text())
    hashes = {n: digest(audit_dir / n) for n in INPUTS}
    if any(marker['artifacts'].get(n) != h for n, h in hashes.items()):
        raise ValueError('Audit artifacts changed; preserve originals and investigate before proceeding')
    audit = json.loads((audit_dir / INPUTS[0]).read_text())
    monthly = read_counts(audit_dir / INPUTS[1], 'month')
    weekly = read_counts(audit_dir / INPUTS[2], 'week')
    expected = audit['valid_time_label_rows']
    if sum(monthly.values()) != expected or sum(weekly.values()) != expected:
        raise ValueError('Count tables disagree with audit total')
    result = describe(monthly, weekly, min_support)
    result['audit_artifact_sha256'] = hashes
    result['schema_summary'] = {k: audit.get(k) for k in (
        'raw_rows', 'valid_time_label_rows', 'first_timestamp_utc', 'last_timestamp_utc',
        'columns', 'missing_expected_features', 'invalid_rows', 'nonfinite_feature_counts', 'protocol_counts')}
    out = audit_dir / 'coverage_v1'
    out.mkdir(exist_ok=True)
    (out / 'coverage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (out / 'device_month_matrix.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['device_id'] + MONTHS)
        writer.writeheader()
        writer.writerows(result['device_month_counts'])
    lines = ['TMA 覆盖诊断（只读既有审计统计，未训练）',
             '11 月来源类别：' + str(result['source_ids']),
             f'11 月不足 {min_support} 条的来源类别：' + str(result['source_below_support_ids']),
             '主实验月份以外（1970 异常与 5 月边界分别保留记录）：' +
             json.dumps(result['outside_primary_months'], ensure_ascii=False),
             '\n每周：记录数 / 出现类别数 / 来源类缺席 ID / 来源类少量 ID / 来源未见 ID']
    for item in result['weekly']:
        lines.append(f"{item['period']} {item['rows']:>9} / {len(item['present_ids']):2} / "
                     f"{item['absent_source_ids']} / {item['present_source_below_support_ids']} / {item['unseen_source_ids']}")
    lines += ['\n每设备每月计数：', 'device_id,' + ','.join(MONTHS)]
    for row in result['device_month_counts']:
        lines.append(','.join(str(row[k]) for k in ['device_id'] + MONTHS))
    lines += ['\n字段与数据质量：', json.dumps(result['schema_summary'], ensure_ascii=False, indent=2),
              '\n注意：缺席不等于崩溃；覆盖变化不证明概念漂移。',
              '首尾周跨月；本表不能代替按原始时间戳过滤、去重及划分后的检查。',
              '未来标签仅用于离线诊断，禁止用于采样决策或参数选择。']
    report = '\n'.join(lines) + '\n'
    (out / 'REPORT.txt').write_text(report)
    print(report)
    print('已保存至', out)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/data/xjw/tma-care-data'))
    parser.add_argument('--min-support', type=int, default=50)
    args = parser.parse_args()
    run(args.data_root / 'audit_v1', args.min_support)
