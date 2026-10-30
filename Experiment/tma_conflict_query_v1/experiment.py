"""Explore source/target disagreement queries under the same 1000-label TMA budget."""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT / 'tma_selective_repair_v1'))
from trial import (Reader, PROTOCOL, MONTHS, check_predictions, complete, context,
                   existing_output, fit_paper, forward_batches, load_references,
                   load_source, mapped, metrics, output_directory, sha, stats,
                   verify_used_inputs, write_csv, write_json)
from pilot import select


def identity():
    result = {p.name: sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}
    for folder, filename in [('tma_selective_repair_v1', 'trial.py'),
                             ('tma_pilot_v1', 'pilot.py'),
                             ('tma_paper_v1', 'run.py')]:
        result[folder + '/' + filename] = sha(EXPERIMENT / folder / filename)
    return result


def output_directory_for(root, source_seed, repair_seed):
    base = root.resolve() / 'conflict_query_v1'
    source = base / f'source_{source_seed}'
    output = source / f'repair_{repair_seed}'
    if any(p.is_symlink() for p in (base, source, output)):
        raise ValueError('Output directories must not be symlinks')
    return output


def random_anchor(paper, month, query_ids, budget):
    saved = pd.read_csv(paper.file(month + '/random_queries.csv'),
                        dtype={'record_id': str}, usecols=['record_id'])
    lookup = {str(value): i for i, value in enumerate(query_ids)}
    if len(lookup) != len(query_ids) or len(saved) != budget:
        raise ValueError('Random query identities or budget changed')
    try:
        anchor = np.asarray([lookup[value] for value in saved.record_id[:budget // 2]],
                            dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Saved uniform query not in pool') from exc
    if len(np.unique(anchor)) != len(anchor):
        raise ValueError('Repeated uniform anchor query')
    return anchor


def badge_from_candidates(features, logits, candidates, budget, seed):
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    if len(candidates) <= budget:
        return candidates
    if len(candidates) < 50:
        return np.random.default_rng(seed).choice(candidates, budget, replace=False)
    chosen = select('badge', features[candidates], logits[candidates], budget, seed)
    return candidates[chosen]


def conflict_selection(features, source_logits, provisional_logits, anchor, seed):
    """Select all possible disagreement BADGE queries, then fill from the rest."""
    n, budget = len(features), 2 * len(anchor)
    if n < budget or len(np.unique(anchor)) != len(anchor) or not np.isin(anchor, np.arange(n)).all():
        raise ValueError('Invalid anchor or insufficient pool')
    remainder = budget - len(anchor)
    source_pred = source_logits.argmax(1).numpy()
    adapted_pred = provisional_logits.argmax(1).numpy()
    available = np.setdiff1d(np.arange(n), anchor)
    disagree = available[source_pred[available] != adapted_pred[available]]
    active = badge_from_candidates(features, provisional_logits, disagree, remainder, seed)
    fill = remainder - len(active)
    if fill:
        other = np.setdiff1d(available, active)
        active = np.r_[active, badge_from_candidates(features, source_logits,
                                                      other, fill, seed)]
    selected = np.r_[anchor, active]
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise ValueError('Conflict query budget or uniqueness mismatch')
    return selected, dict(unlabeled_disagreements=int(len(disagree)),
                          selected_disagreements=int((source_pred[active] !=
                                                      adapted_pred[active]).sum()),
                          random_anchor=len(anchor), active_queries=len(active))


def run(root, source_seed=0, repair_seed=0, device='cuda', threads=8):
    reader, cache, classes, _ = context(root, source_seed)
    paper = Reader(output_directory(root, source_seed) / f'repair_{repair_seed}')
    ps = paper.marker['signature']
    if (ps['source_seed'] != source_seed or ps['repair_seed'] != repair_seed
            or ps['protocol'] != PROTOCOL or ps['device'] != device or ps['threads'] != threads
            or ps['source_marker'] != sha(reader.root / 'complete.json')
            or ps['cache_marker'] != sha(cache.root / 'complete.json')):
        raise ValueError('Paper run differs from source/cache/protocol/environment')
    hybrid = Reader(root / 'hybrid_adaptive_v1' / f'source_{source_seed}' / f'repair_{repair_seed}')
    hs = hybrid.marker['signature']
    if (hs['source_seed'] != source_seed or hs['repair_seed'] != repair_seed
            or hs['paper_marker'] != sha(paper.root / 'complete.json')
            or hs['protocol'] != PROTOCOL or hs['device'] != device or hs['threads'] != threads):
        raise ValueError('Completed hybrid baseline differs from requested run')
    signature = dict(schema=1, source_seed=source_seed, repair_seed=repair_seed,
        source_marker=sha(reader.root / 'complete.json'),
        cache_marker=sha(cache.root / 'complete.json'),
        paper_marker=sha(paper.root / 'complete.json'),
        hybrid_marker=sha(hybrid.root / 'complete.json'), code=identity(),
        protocol=PROTOCOL, device=device, threads=threads, months=MONTHS,
        selector='500_uniform_then_BADGE_on_source_target_disagreement')
    output = output_directory_for(root, source_seed, repair_seed)
    if existing_output(output, signature):
        verify_used_inputs(output, pilot=reader, cache=cache, paper=paper, hybrid=hybrid)
        print('CONFLICT QUERY COMPLETE (verified existing run):', output, flush=True)
        return output
    write_json(output / 'config.json', signature)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, prep = load_source(reader, classes)
    rx, ry = load_references(reader, cache, paper, classes, source_seed, repair_seed)
    rfeat = forward_batches(model.encoder, prep.transform(rx), device)
    teacher = forward_batches(model.head, rfeat.numpy(), device)
    history = reader.arrays('history_predictions.npz', ['labels', 'mlp'])
    historical = stats(history['labels'], history['mlp'], len(classes))
    budget = reader.marker['signature']['budget']
    summaries, per_class, coverage, selection_rows = [], [], [], []
    for month in MONTHS:
        part = output / month
        part_signature = dict(signature, month=month)
        if existing_output(part, part_signature):
            verify_used_inputs(part, pilot=reader, cache=cache, paper=paper, hybrid=hybrid)
            print(month, 'verified completed month reused', flush=True)
        else:
            write_json(part / 'config.json', part_signature)
            query = cache.arrays(month + '_query.npz', ['ids', 'x', 'y'])
            qfeat = forward_batches(model.encoder, prep.transform(query['x']), device)
            qlogits = forward_batches(model.head, qfeat.numpy(), device)
            anchor = random_anchor(paper, month, query['ids'], budget)
            anchor_y = mapped(query['y'][anchor], classes)
            usable_anchor = anchor_y >= 0
            provisional, provisional_trace = fit_paper(model.head,
                qfeat[anchor[usable_anchor]], anchor_y[usable_anchor], rfeat, ry,
                teacher, False, repair_seed, device, PROTOCOL)
            torch.save(provisional.state_dict(), part / 'provisional_target_head.pt')
            provisional_logits = forward_batches(provisional, qfeat.numpy(), device)
            selected, info = conflict_selection(qfeat, qlogits, provisional_logits,
                                                anchor, repair_seed)
            paid_y = mapped(query['y'][selected], classes)
            write_csv(part / 'queries.csv', [dict(record_id=str(query['ids'][i]),
                device_id=int(query['y'][i]), known_class=bool(paid_y[j] >= 0),
                stage='uniform_anchor' if j < len(anchor) else 'active')
                for j, i in enumerate(selected)])
            write_json(part / 'selection.json', dict(month=month, budget=budget,
                usable_anchor_labels=int(usable_anchor.sum()),
                usable_final_labels=int((paid_y >= 0).sum()),
                provisional_trace=provisional_trace, **info))
            print(month, 'selected', info, flush=True)
            usable = paid_y >= 0
            heads, traces = {}, {}
            for method, care in [('conflict_target', False), ('conflict_care', True)]:
                head, trace = fit_paper(model.head, qfeat[selected[usable]], paid_y[usable],
                    rfeat, ry, teacher, care, repair_seed, device, PROTOCOL)
                heads[method], traces[method] = head, trace
                torch.save(head.state_dict(), part / (method + '_head.pt'))
                print(month, method, 'complete', flush=True)
            write_json(part / 'training.json', traces)
            # Only now open future predictions and labels for evaluation.
            future = paper.arrays(month + '/predictions.npz',
                                  ['ids', 'labels', 'device_ids', 'static_mlp'])
            target = cache.arrays(month + '_eval.npz', ['ids', 'x', 'y'])
            y = check_predictions(future, target, classes)
            efeat = forward_batches(model.encoder, prep.transform(target['x']), device)
            source_pred = forward_batches(model.head, efeat.numpy(), device).argmax(1).numpy()
            if not np.array_equal(source_pred, future['static_mlp']):
                raise ValueError('Reloaded source predictions differ from paper run')
            predictions = dict(ids=future['ids'], labels=y, device_ids=future['device_ids'],
                               static_mlp=source_pred)
            for method, head in heads.items():
                predictions[method] = forward_batches(head, efeat.numpy(), device).argmax(1).numpy()
            np.savez_compressed(part / 'predictions.npz', **predictions)
            rows_out, class_out = [], []
            for method in ['static_mlp', 'conflict_target', 'conflict_care']:
                score, class_rows, cm = metrics(y, source_pred,
                                                 predictions[method], historical, classes)
                meta = dict(source_seed=source_seed, repair_seed=repair_seed,
                    month=month, method=method,
                    label_queries=0 if method == 'static_mlp' else budget,
                    usable_known_labels=0 if method == 'static_mlp' else int(usable.sum()))
                rows_out.append(dict(**meta, **score))
                class_out.extend(dict(**meta, **row) for row in class_rows)
                np.save(part / (method + '_confusion.npy'), cm)
            write_csv(part / 'summary.csv', rows_out)
            write_csv(part / 'per_class.csv', class_out)
            write_csv(part / 'query_coverage.csv', [dict(month=month, device_id=c,
                candidate_pool=int((query['y'] == c).sum()),
                paid_true_class_count=int((query['y'][selected] == c).sum()))
                for c in classes])
            write_json(part / 'inputs.json', dict(pilot=reader.consumed,
                cache=cache.consumed, paper=paper.consumed, hybrid=hybrid.consumed))
            complete(part, part_signature)
        summaries.extend(pd.read_csv(part / 'summary.csv').to_dict('records'))
        per_class.extend(pd.read_csv(part / 'per_class.csv').to_dict('records'))
        coverage.extend(pd.read_csv(part / 'query_coverage.csv').to_dict('records'))
        selection_rows.append(json.loads((part / 'selection.json').read_text()))
        write_csv(output / 'summary.csv', summaries)
        write_csv(output / 'per_class.csv', per_class)
        write_csv(output / 'query_coverage.csv', coverage)
    baseline = pd.read_csv(hybrid.file('summary.csv'))
    frame = pd.DataFrame(summaries)
    comparisons = []
    for month in MONTHS:
        current = frame[(frame.month == month) & (frame.method == 'conflict_care')].iloc[0]
        old = baseline[(baseline.month == month) & (baseline.method == 'hybrid_care')].iloc[0]
        comparisons.append(dict(month=month,
            conflict_macro_f1=current.macro_f1_present,
            hybrid_macro_f1=old.macro_f1_present,
            macro_f1_delta_pp=100 * (current.macro_f1_present - old.macro_f1_present),
            conflict_collapse_f1=current.collapse_f1,
            hybrid_collapse_f1=old.collapse_f1,
            conflict_negative_flip=current.negative_flip_rate_vs_static_mlp,
            hybrid_negative_flip=old.negative_flip_rate_vs_static_mlp))
    write_csv(output / 'paired_comparison.csv', comparisons)
    write_csv(output / 'selection_summary.csv', selection_rows)
    report = '\n'.join([
        'TMA exploratory conflict-aware query trial; fixed source and 1000 labels/month.',
        'First 500 labels are identical to the previous hybrid run; second 500 change.',
        'A provisional target head is trained on the first 500; BADGE favors its disagreements with the source.',
        'No future evaluation labels enter query selection or head training.',
        pd.DataFrame(comparisons).to_string(index=False),
        'Selection diagnostics:',
        pd.DataFrame(selection_rows)[['month', 'unlabeled_disagreements',
                                      'selected_disagreements', 'usable_final_labels']].to_string(index=False),
        'Comparisons with hybrid_care differ in active query identities, not label budget or CARE schedule.',
        'All target months were previously examined; this is exploratory, not independent validation.',
    ]) + '\n'
    (output / 'REPORT.txt').write_text(report)
    write_json(output / 'inputs.json', dict(pilot=reader.consumed, cache=cache.consumed,
                                            paper=paper.consumed, hybrid=hybrid.consumed))
    complete(output, signature)
    print(report, flush=True)
    print('CONFLICT QUERY COMPLETE:', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    parser.add_argument('--source-seed', type=int, default=0)
    parser.add_argument('--repair-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Real experiments run on Linux; local tests use synthetic fixtures')
    if min(args.source_seed, args.repair_seed) < 0 or args.threads < 1:
        parser.error('Seeds must be nonnegative and threads positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; activate traffic-ncde')
    root = args.root.resolve()
    context(root, args.source_seed)
    output = output_directory_for(root, args.source_seed, args.repair_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output.parent / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another conflict-query process uses this source output directory')
        run(root, args.source_seed, args.repair_seed, args.device, args.threads)


if __name__ == '__main__':
    main()
