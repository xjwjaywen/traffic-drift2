"""Chronological server pilot: static RF/MLP, matched head repair and CARE."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier

from data import (FEATURES, MONTHS, Preprocessor, finish, implementation, load,
                  prepare, sha, validate_complete, write_csv, write_json)
from existing_care import badge_selection, fit_controlled


class Model(nn.Module):
    def __init__(self, inputs, classes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, 256), nn.LayerNorm(256), nn.ReLU(),
                                     nn.Dropout(.1), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU())
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def mapped(y, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup.get(int(c), -1) for c in y], dtype=np.int64)


def stats(y, pred, classes):
    keep = y >= 0
    cm = np.bincount(y[keep] * classes + pred[keep], minlength=classes ** 2).reshape(classes, classes)
    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm).astype(float)
    recall = np.divide(tp, support, out=np.full(classes, np.nan), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=support + predicted > 0)
    return dict(cm=cm, support=support, predicted=predicted, recall=recall, f1=f1)


def evaluate(y, before, after, historical, classes, min_support=50):
    b, a = stats(y, before, len(classes)), stats(y, after, len(classes))
    sufficient = (b['support'] >= min_support) & (historical['support'] >= min_support)
    learned = sufficient & (historical['recall'] >= .8)
    collapsed = learned & (b['recall'] < .1)
    stable = learned & (b['recall'] >= .8)
    supported = a['support'] > 0
    correct = (y >= 0) & (before == y)
    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None
    summary = dict(known_eval_rows=int((y >= 0).sum()), unseen_eval_rows=int((y < 0).sum()),
        present_classes=int(supported.sum()), supported_classes=int((a['support'] >= min_support).sum()),
        accuracy_known=float((after[y >= 0] == y[y >= 0]).mean()) if (y >= 0).any() else None,
        macro_f1_present=mean(a['f1'], supported), macro_recall_present=mean(a['recall'], supported),
        macro_f1_support50=mean(a['f1'], a['support'] >= min_support),
        collapse_count=int(collapsed.sum()), collapse_f1=mean(a['f1'], collapsed),
        recovered_to_recall_0_1=int((collapsed & (a['recall'] >= .1)).sum()),
        recovered_to_recall_0_5=int((collapsed & (a['recall'] >= .5)).sum()),
        stable_f1=mean(a['f1'], stable), stable_delta_f1=mean(a['f1'] - b['f1'], stable),
        new_collapses=int((learned & (b['recall'] >= .1) & (a['recall'] < .1)).sum()),
        negative_flip_rate=float((after[correct] != y[correct]).mean()) if correct.any() else None,
        relaxed_source_0_5_collapse_count=int((sufficient & (historical['recall'] >= .5) & (b['recall'] < .1)).sum()))
    rows = []
    for i, device in enumerate(classes):
        rows.append(dict(device_id=device, historical_support=int(historical['support'][i]),
            historical_recall=float(historical['recall'][i]) if historical['support'][i] else None,
            support=int(a['support'][i]), predicted_count=int(a['predicted'][i]),
            before_recall=float(b['recall'][i]) if b['support'][i] else None,
            after_recall=float(a['recall'][i]) if a['support'][i] else None,
            before_f1=float(b['f1'][i]) if b['support'][i] else None,
            after_f1=float(a['f1'][i]) if a['support'][i] else None,
            eligible=bool(learned[i]), collapse=bool(collapsed[i]), stable=bool(stable[i])))
    return summary, rows, a['cm']


@torch.no_grad()
def forward_batches(module, x, device):
    module.to(device).eval()
    out = [module(torch.as_tensor(x[i:i + 4096], device=device)).cpu()
           for i in range(0, len(x), 4096)]
    module.cpu()
    if not out:
        raise ValueError('Empty inference pool')
    return torch.cat(out)


def train_source(x, y, vx, vy, classes, args, seed, root):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(x.shape[1], len(classes)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    best, state, history = -1, None, []
    for epoch in range(args.epochs):
        model.train().to(args.device)
        order = torch.randperm(len(x), generator=rng)
        loss_sum = 0.
        for ids in order.split(512):
            loss = nn.functional.cross_entropy(model(tx[ids].to(args.device)), ty[ids].to(args.device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite source loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(ids)
        pred = forward_batches(model, vx, args.device).argmax(1).numpy()
        v = stats(vy, pred, len(classes))
        valid = v['support'] >= 50
        if not valid.any():
            valid = v['support'] > 0
        score = float(v['recall'][valid].mean())
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / len(x), validation_macro_recall=score))
        print(f'Source seed={seed} epoch={epoch + 1}/{args.epochs} validation recall={score:.4f}', flush=True)
        if score > best:
            best, state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    model.cpu().eval()
    torch.save(dict(state_dict=state, classes=classes, input_features=FEATURES,
                    validation_best=best), root / 'source.pt')
    write_csv(root / 'source_training.csv', history)
    return model


def choose_reference(train, reference, classes, seed):
    rng = np.random.default_rng(seed + 10007)
    selected, origins = [], []
    for c in classes:
        candidates = np.flatnonzero(reference['y'] == c)
        n = min(5, len(candidates))
        for i in rng.choice(candidates, n, replace=False):
            selected.append((reference, int(i)))
            origins.append(dict(device_id=c, role='reference', record_id=str(reference['ids'][i])))
        candidates = np.flatnonzero(train['y'] == c)
        if len(candidates) < 5 - n:
            raise ValueError(f'Class {c} lacks five historical reference examples')
        for i in rng.choice(candidates, 5 - n, replace=False):
            selected.append((train, int(i)))
            origins.append(dict(device_id=c, role='train_fallback', record_id=str(train['ids'][i])))
    return dict(x=np.stack([part['x'][i] for part, i in selected]),
                y=np.array([part['y'][i] for part, i in selected])), origins


def select(kind, features, logits, budget, seed):
    budget = min(budget, len(features))
    if kind == 'random':
        return np.random.default_rng(seed).choice(len(features), budget, replace=False)
    if kind == 'margin':
        top = logits.softmax(1).topk(2).values.numpy()
        return np.argsort(top[:, 0] - top[:, 1], kind='stable')[:budget]
    if len(features) < 50:
        raise ValueError('Existing BADGE approximation requires at least 50 candidates')
    ids = badge_selection(features, logits, logits.shape[1], budget, seed).numpy()
    if len(np.unique(ids)) != budget:
        raise ValueError('BADGE returned repeated selections')
    return ids


def conditional_shift(history, target, classes, month):
    from scipy.stats import ks_2samp
    rows = []
    for c in classes:
        hx, tx = history['x'][history['y'] == c], target['x'][target['y'] == c]
        if min(len(hx), len(tx)) < 50:
            continue
        for j, name in enumerate(FEATURES):
            h, t = hx[:, j], tx[:, j]
            hf, tf = h[np.isfinite(h)], t[np.isfinite(t)]
            rows.append(dict(month=month, device_id=c, feature=name,
                historical_n=len(h), target_n=len(t), historical_missing=float((~np.isfinite(h)).mean()),
                target_missing=float((~np.isfinite(t)).mean()),
                ks_effect=float(ks_2samp(hf, tf, method='asymp').statistic) if min(len(hf), len(tf)) >= 50 else None))
    return rows


def run_seed(cache, root, args, seed):
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'pandas', 'torch', 'scikit-learn', 'scipy', 'joblib')}
    signature = dict(implementation=implementation(), cache_sha256=sha(cache / 'complete.json'),
                     seed=seed, epochs=args.epochs, repair_steps=args.steps, budget=args.budget,
                     rf_trees=args.rf_trees, device=args.device, threads=args.threads, versions=versions)
    root.mkdir(parents=True, exist_ok=True)
    if validate_complete(root, signature):
        print('Completed seed reused:', seed, flush=True)
        return
    write_json(root / 'config.json', signature)
    started = time.perf_counter()
    train, val, history, reference = [load(cache, r) for r in ('train', 'val', 'history', 'reference')]
    classes = sorted(int(c) for c in np.unique(train['y']))
    if len(classes) < 2:
        raise ValueError('Insufficient training classes')
    prep = Preprocessor().fit(train['x'])
    np.savez(root / 'preprocessor.npz', median=prep.median, mean=prep.mean, std=prep.std)
    x, vx, hx = [prep.transform(part['x']) for part in (train, val, history)]
    y, vy, hy = [mapped(part['y'], classes) for part in (train, val, history)]
    model = train_source(x, y, vx, vy, classes, args, seed, root)
    hfeat = forward_batches(model.encoder, hx, args.device)
    hpred = forward_batches(model.head, hfeat.numpy(), args.device).argmax(1).numpy()
    hist = stats(hy, hpred, len(classes))
    ref, origins = choose_reference(train, reference, classes, seed)
    write_csv(root / 'reference_ids.csv', origins)
    rfeat = forward_batches(model.encoder, prep.transform(ref['x']), args.device)
    rlabels = mapped(ref['y'], classes)
    teacher = forward_batches(model.head, rfeat.numpy(), args.device)
    print(f'Fit static RF seed={seed}', flush=True)
    rf = RandomForestClassifier(n_estimators=args.rf_trees, min_samples_leaf=2,
                                max_features='sqrt', n_jobs=args.threads, random_state=seed)
    rf.fit(x, y)
    import joblib
    joblib.dump(rf, root / 'random_forest.joblib')
    rf_history = rf.predict(hx)
    rf_hist = stats(hy, rf_history, len(classes))
    np.savez_compressed(root / 'history_predictions.npz', ids=history['ids'], device_ids=history['y'],
                        labels=hy, mlp=hpred, rf=rf_history)
    summaries, class_rows, shift_rows = [], [], []
    # Independent monthly repair trials: reset to the same source head each time.
    for month in MONTHS:
        query, target = load(cache, month + '_query'), load(cache, month + '_eval')
        qx, ex = prep.transform(query['x']), prep.transform(target['x'])
        qy, ey = mapped(query['y'], classes), mapped(target['y'], classes)
        qfeat = forward_batches(model.encoder, qx, args.device)
        efeat = forward_batches(model.encoder, ex, args.device)
        qlogits = forward_batches(model.head, qfeat.numpy(), args.device)
        before = forward_batches(model.head, efeat.numpy(), args.device).argmax(1).numpy()
        predictions = dict(ids=target['ids'], device_ids=target['y'], labels=ey, static_mlp=before)
        rf_pred = rf.predict(ex)
        predictions['static_rf'] = rf_pred
        month_dir = root / month
        month_dir.mkdir(exist_ok=True)
        def report(method, pred, baseline, historical, queries=0, usable=0):
            summary, rows, cm = evaluate(ey, baseline, pred, historical, classes)
            summary = dict(seed=seed, month=month, method=method, label_queries=queries,
                           usable_known_labels=usable, **summary)
            summaries.append(summary)
            class_rows.extend(dict(seed=seed, month=month, method=method, **r) for r in rows)
            np.save(month_dir / (method + '_confusion.npy'), cm)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        report('static_mlp', before, before, hist)
        report('static_rf', rf_pred, rf_pred, rf_hist)
        for selector, variants in [('random', ['target', 'replay', 'care']),
                                    ('margin', ['care']), ('badge', ['target', 'care'])]:
            print(f'{month} selecting {selector}, seed={seed}', flush=True)
            ids = select(selector, qfeat, qlogits, args.budget, seed)
            # No true query labels are accessed by select(). Unknown labels still
            # cost a query and are reported, but the fixed closed-set head cannot fit them.
            write_csv(month_dir / (selector + '_queries.csv'), [dict(
                record_id=str(query['ids'][i]), device_id=int(query['y'][i]),
                known_class=bool(qy[i] >= 0)) for i in ids])
            usable = ids[qy[ids] >= 0]
            for variant in variants:
                method = selector + '_' + variant
                if len(usable):
                    head, trace = fit_controlled(model.head, qfeat[usable], qy[usable], rfeat, rlabels, teacher,
                        replay_ce=variant != 'target', kd_weight=.5 if variant == 'care' else 0.,
                        temperature=2., target_weight=1., replay_weight=1., steps=args.steps,
                        batch_size=128, lr=1e-3, weight_decay=1e-4, seed=seed,
                        device=torch.device(args.device), log_every=0)
                else:
                    head, trace = copy.deepcopy(model.head), dict(skipped='No known queried labels')
                write_json(month_dir / (method + '_trace.json'), trace)
                torch.save(head.state_dict(), month_dir / (method + '_head.pt'))
                pred = forward_batches(head, efeat.numpy(), args.device).argmax(1).numpy()
                predictions[method] = pred
                report(method, pred, before, hist, len(ids), len(usable))
        np.savez_compressed(month_dir / 'predictions.npz', **predictions)
        shift_rows.extend(conditional_shift(history, target, classes, month))
        write_csv(root / 'summary.csv', summaries)
        write_csv(root / 'per_class.csv', class_rows)
        write_csv(root / 'conditional_shift.csv', shift_rows)
    write_json(root / 'run_info.json', dict(classes=classes, elapsed_seconds=time.perf_counter() - started,
        python=sys.version, versions=versions, device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
        note='Monthly trials reset source head. Budget per month, not shared across four months. '
        'CARE repair loss and repository BADGE approximation reused verbatim; tabular encoder newly trained. '
        'Not a reproduction of paper CNN or full diagnostic monitor. Unknown query labels consume budget. '
        'Closed-set metrics exclude unknown target labels; no open-set detection claim. '
        'Macros use currently present labels; consult per_class.csv before comparing months. '
        'Conditional KS effects are descriptive, not independent-sample significance tests.'))
    names = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name not in ('complete.json', 'config.json')]
    finish(root, signature, names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'all'])
    p.add_argument('--data-root', type=Path, default=Path('/data/xjw/tma-care-data'))
    p.add_argument('--output-root', type=Path, default=Path('/data/xjw/tma-care-data/pilot_v1'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', default='0')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--budget', type=int, default=1000)
    p.add_argument('--rf-trees', type=int, default=200)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--train-per-class', type=int, default=2000)
    p.add_argument('--query-cap', type=int, default=50000)
    p.add_argument('--eval-cap', type=int, default=200000)
    args = p.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Run on the Linux server; local machine is for synthetic tests only.')
    if min(args.epochs, args.steps, args.budget, args.rf_trees, args.threads,
           args.train_per_class, args.query_cap, args.eval_cap) < 1:
        p.error('All numeric limits must be positive')
    if args.command != 'prepare' and args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check traffic-ncde environment.')
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import fcntl
    with (args.output_root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = args.output_root / 'cache'
        caps = dict(train=args.train_per_class, reference=200, query=args.query_cap, eval=args.eval_cap)
        if args.command in ('prepare', 'all'):
            prepare(args.data_root / 'raw/features_nov-apr.tar.gz', cache, caps)
        if args.command in ('run', 'all'):
            config = json.loads((cache / 'config.json').read_text())
            if config['implementation'] != implementation() or config['caps'] != caps or not validate_complete(cache, config):
                raise ValueError('Cache configuration differs or preparation incomplete')
            seeds = sorted(set(int(s) for s in args.seeds.split(',')))
            for seed in seeds:
                run_seed(cache, args.output_root / f'seed_{seed}', args, seed)
            tables = []
            import pandas as pd
            for seed in seeds:
                tables.extend(pd.read_csv(args.output_root / f'seed_{seed}/summary.csv').to_dict('records'))
            write_csv(args.output_root / 'summary.csv', tables)
            print('PILOT COMPLETE. Return summary.csv, seed_0/per_class.csv and cache/sample_counts.csv.', flush=True)


if __name__ == '__main__':
    main()
