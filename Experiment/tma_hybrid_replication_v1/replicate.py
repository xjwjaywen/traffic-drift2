"""Repeat the frozen hybrid CARE experiment and summarize paired seed effects."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT / 'tma_analysis_v1'))
sys.path.insert(0, str(EXPERIMENT / 'tma_paper_v1'))
from analyze import Reader, sha
from run import complete, existing_output, output_directory, write_json

METRICS = ['macro_f1_present', 'collapse_f1', 'high_recall_group_f1',
           'negative_flip_rate_vs_static_mlp', 'remaining_collapse_count']
COMPARISONS = [
    ('hybrid_adaptive', 'hybrid_care', 'same_query_training'),
    ('hybrid_care', 'hybrid_target', 'same_query_training'),
    ('hybrid_care', 'badge_care', 'different_query_sets'),
    ('hybrid_care', 'random_care', 'different_query_sets'),
]


def result_dir(root, source_seed, seeds):
    base = root.resolve() / 'hybrid_replication_v1'
    source = base / f'source_{source_seed}'
    output = source / ('aggregate_' + '_'.join(map(str, seeds)))
    if any(p.is_symlink() for p in (base, source, output)):
        raise ValueError('Output directories must not be symlinks')
    return output


def checked_table(reader, name, seed, months, methods):
    frame = pd.read_csv(reader.file(name))
    if not {'month', 'method', 'source_seed', 'repair_seed'}.issubset(frame.columns):
        raise ValueError(f'Missing keys in {reader.root / name}')
    wanted = {(m, method) for m in months for method in methods}
    actual = list(zip(frame.month, frame.method))
    if len(actual) != len(set(actual)) or not wanted.issubset(set(actual)):
        raise ValueError(f'Incomplete or duplicate month/method rows in {reader.root / name}')
    if not (frame.repair_seed == seed).all():
        raise ValueError(f'Wrong repair seed in {reader.root / name}')
    return frame


def load_runs(root, source_seed, seeds):
    hybrid_frames, paper_frames, class_frames, decision_frames = [], [], [], []
    signatures, manifests = [], {}
    for seed in seeds:
        hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' / f'repair_{seed}')
        paper = Reader(output_directory(root, source_seed) / f'repair_{seed}')
        hs, ps = hybrid.marker['signature'], paper.marker['signature']
        if (hs['source_seed'] != source_seed or hs['repair_seed'] != seed
                or ps['source_seed'] != source_seed or ps['repair_seed'] != seed
                or hs['paper_marker'] != sha(paper.root / 'complete.json')
                or hs['protocol'] != ps['protocol']):
            raise ValueError(f'Hybrid/paper provenance mismatch at repair seed {seed}')
        signatures.append({k: v for k, v in hs.items() if k not in ('repair_seed', 'paper_marker')})
        months = hs['months']
        h = checked_table(hybrid, 'summary.csv', seed, months,
                          ['static_mlp', 'hybrid_target', 'hybrid_care', 'hybrid_adaptive'])
        p = checked_table(paper, 'summary.csv', seed, months, ['badge_care', 'random_care'])
        if not set(METRICS).issubset(h.columns) or not set(METRICS).issubset(p.columns):
            raise ValueError(f'Missing comparison metric at repair seed {seed}')
        if (h[h.method != 'static_mlp'].label_queries != 1000).any():
            raise ValueError(f'Hybrid query budget changed at repair seed {seed}')
        hybrid_frames.append(h)
        paper_frames.append(p[p.method.isin(['badge_care', 'random_care'])])
        pc = pd.read_csv(hybrid.file('per_class.csv'))
        dc = pd.read_csv(hybrid.file('decision_classes.csv'))
        pc['repair_seed'] = seed
        dc['repair_seed'] = seed
        class_frames.append(pc)
        decision_frames.append(dc)
        manifests[str(seed)] = dict(hybrid_complete=sha(hybrid.root / 'complete.json'),
                                    paper_complete=sha(paper.root / 'complete.json'))
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError('Hybrid source, code, protocol, or environment differs between seeds')
    return (pd.concat(hybrid_frames, ignore_index=True),
            pd.concat(paper_frames, ignore_index=True),
            pd.concat(class_frames, ignore_index=True),
            pd.concat(decision_frames, ignore_index=True), signatures[0], manifests)


def paired_rows(hybrid, paper, seeds, months):
    table = pd.concat([hybrid, paper], ignore_index=True)
    rows = []
    for seed in seeds:
        for month in months:
            sub = table[(table.repair_seed == seed) & (table.month == month)].set_index('method')
            for left, right, scope in COMPARISONS:
                row = dict(repair_seed=seed, month=month,
                           comparison=left + '_minus_' + right, scope=scope)
                for metric in METRICS:
                    scale = 1 if metric == 'remaining_collapse_count' else 100
                    row[metric + ('_delta' if scale == 1 else '_pp')] = scale * (
                        float(sub.loc[left, metric]) - float(sub.loc[right, metric]))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(hybrid, paper, classes, decisions, seeds, months):
    values = hybrid.groupby(['month', 'method'])[METRICS].agg(['mean', 'std', 'count'])
    values.columns = ['_'.join(col) for col in values.columns]
    summary = values.reset_index()
    paired = paired_rows(hybrid, paper, seeds, months)
    delta_fields = [col for col in paired if col.endswith(('_pp', '_delta'))]
    deltas = paired.groupby(['month', 'comparison', 'scope'])[delta_fields].agg(
        ['mean', 'std', 'count'])
    deltas.columns = ['_'.join(col) for col in deltas.columns]
    deltas = deltas.reset_index()
    macro = 'macro_f1_present_pp'
    wins = paired.groupby(['month', 'comparison'])[macro].agg(
        positive_count=lambda x: int((x > 0).sum()),
        nonpositive_count=lambda x: int((x <= 0).sum())).reset_index()
    deltas = deltas.merge(wins, on=['month', 'comparison'], validate='one_to_one')
    focus = classes[((classes.month == '2020-03') & classes.device_id.isin([5, 22, 26, 37]))
                    | ((classes.month == '2020-04') & (classes.device_id == 20))]
    focus = focus.groupby(['month', 'device_id', 'method'])[['after_recall', 'after_f1']].agg(
        ['mean', 'std', 'count'])
    focus.columns = ['_'.join(col) for col in focus.columns]
    focus = focus.reset_index()
    decisions = decisions.copy()
    decisions['relax_preservation'] = decisions.relax_preservation.astype(str).str.lower().eq('true')
    flags = decisions.groupby(['month', 'device_id']).relax_preservation.agg(
        ['sum', 'count']).reset_index().rename(columns={'sum': 'relaxed_seed_count',
                                                        'count': 'seed_count'})
    return summary, paired, deltas, focus, flags


def aggregate(root, source_seed, seeds):
    hybrid, paper, classes, decisions, signature, manifests = load_runs(root, source_seed, seeds)
    months = signature['months']
    output = result_dir(root, source_seed, seeds)
    aggregate_signature = dict(schema=1, source_seed=source_seed, seeds=seeds,
                               run_signature=signature, inputs=manifests,
                               code_sha256=sha(Path(__file__)))
    if existing_output(output, aggregate_signature):
        print('REPLICATION AGGREGATE COMPLETE (verified):', output, flush=True)
        return output
    write_json(output / 'config.json', aggregate_signature)
    summary, paired, deltas, focus, flags = summarize(
        hybrid, paper, classes, decisions, seeds, months)
    summary.to_csv(output / 'summary_mean_sd.csv', index=False)
    paired.to_csv(output / 'paired_by_seed.csv', index=False)
    deltas.to_csv(output / 'paired_mean_sd.csv', index=False)
    focus.to_csv(output / 'focus_class_mean_sd.csv', index=False)
    flags.to_csv(output / 'relaxation_frequency.csv', index=False)
    overview = deltas[['month', 'comparison', 'macro_f1_present_pp_mean',
                       'macro_f1_present_pp_std', 'positive_count',
                       'collapse_f1_pp_mean',
                       'negative_flip_rate_vs_static_mlp_pp_mean']]
    report = '\n'.join([
        f'Frozen hybrid CARE repair-seed replication; fixed source seed {source_seed}; repair seeds {seeds}.',
        'Each hybrid seed uses 1000 paid query labels; methods within that seed share the same queries.',
        'Macro-F1 positive_count counts seeds with a strictly positive paired difference.',
        'NaN collapse-F1 means that month has no eligible collapsed classes.',
        overview.to_string(index=False),
        'Class relaxation frequency (out of repair seeds):',
        flags[flags.relaxed_seed_count > 0].to_string(index=False) if flags.relaxed_seed_count.any() else 'none',
        'Fixed source/evaluation months: seed variation is not independent temporal or dataset validation.',
        'All four months were examined during method design; do not tune the frozen rule using this report.',
        'Different-query-set comparisons include query selection effects; they do not isolate training.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    complete(output, aggregate_signature)
    print(report, flush=True)
    print('REPLICATION AGGREGATE COMPLETE:', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seeds', default='0,1,2,3,4')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    try:
        seeds = sorted(set(int(s) for s in args.repair_seeds.split(',')))
        if not seeds or min(seeds) < 0 or args.source_seed < 0 or args.threads < 1:
            raise ValueError()
    except ValueError:
        parser.error('Provide nonnegative comma-separated repair seeds and positive threads')
    if not args.aggregate_only and platform.system() != 'Linux':
        parser.error('Real experiments run on Linux; use --aggregate-only for local analysis')
    root = args.root.resolve()
    if not args.aggregate_only:
        script = EXPERIMENT / 'tma_hybrid_adaptive_v1' / 'experiment.py'
        for seed in seeds:
            command = [sys.executable, '-u', str(script), '--root', str(root),
                       '--source-seed', str(args.source_seed), '--repair-seed', str(seed),
                       '--device', args.device, '--threads', str(args.threads)]
            print('RUNNING:', ' '.join(command), flush=True)
            subprocess.run(command, check=True)
    aggregate(root, args.source_seed, seeds)


if __name__ == '__main__':
    main()
