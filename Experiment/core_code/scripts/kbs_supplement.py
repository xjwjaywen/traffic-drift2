#!/usr/bin/env python3
"""Controlled CARE supplements. Real CESNET runs only; synthetic tests live in tests/."""
import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parents[1]
SCHEMA = "care-controlled-v1"
COLLAPSE = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
STABLE = [8, 15, 44, 57, 59, 62, 64, 76, 94, 98, 99, 107, 113, 119, 128, 130, 131, 132, 144, 145]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process owns {path}; use one writer per directory.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def finish(directory, signature, artifacts):
    atomic_json(Path(directory) / "complete.json", {
        "signature": signature,
        "artifacts": {name: file_sha(Path(directory) / name) for name in artifacts},
    })


def is_complete(directory, signature):
    directory = Path(directory)
    path = directory / "complete.json"
    if not path.exists():
        resolved = directory / "resolved_config.json"
        if resolved.exists() and json.loads(resolved.read_text()) != signature:
            raise ValueError(f"Incomplete run has a different configuration: {directory}. Use a new output directory.")
        return False
    info = json.loads(path.read_text())
    if info["signature"] != signature:
        raise ValueError(f"Configuration changed at {directory}. Use a new output directory.")
    for name, expected in info["artifacts"].items():
        if not (directory / name).is_file() or file_sha(directory / name) != expected:
            raise ValueError(f"Missing/modified artifact {directory / name}; recover it or use a new output directory.")
    return True


def runtime_info():
    versions = {}
    for name in ["torch", "numpy", "scikit-learn", "cesnet-datazoo", "PyYAML"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def git(*args):
        proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    return {
        "python": sys.version, "python_executable": sys.executable,
        "packages": versions, "hostname": platform.node(), "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": sys.argv,
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment. Check conda/PyTorch or explicitly use --device cpu.")
    return torch.device(name)


def int_list(value):
    return [int(v) for v in value.replace(",", " ").split()]


def implementation_sha():
    paths = [Path(__file__), CORE / "scripts/collapse_active_maintenance_tls22.py"]
    return digest({str(p.relative_to(CORE)): file_sha(p) for p in paths})


def loader_sha():
    paths = sorted((CORE / "tta_tc").rglob("*.py"))
    paths.append(CORE / "scripts/prototype_recalibration_tls22.py")
    return digest({str(p.relative_to(CORE)): file_sha(p) for p in paths})


def data_inventory(directory):
    # Metadata detects ordinary replacement without hashing multi-GB raw files on every resume.
    return [{"path": str(p.relative_to(directory)), "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(directory.rglob("*")) if p.is_file()]


def prepare_spec(args):
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    cfg = dict(config["data"])
    if cfg.get("dataset") != "tls22":
        raise ValueError("This supplement targets TLS-Year22; use the separate QUIC experiment protocol for QUIC.")
    cfg["data_dir"] = str(Path(args.data_dir or cfg["data_dir"]).expanduser().resolve())
    if not Path(cfg["data_dir"]).is_dir():
        raise ValueError(f"TLS data directory not found: {cfg['data_dir']}. Set DATA_DIR or --data-dir to the existing server dataset.")
    if not any(Path(cfg["data_dir"]).iterdir()):
        raise ValueError("The TLS data directory is empty; this script does not provision a new dataset.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Source checkpoint not found: {checkpoint}")
    return config, {
        "schema": SCHEMA, "data_config": cfg, "checkpoint_sha256": file_sha(checkpoint),
        "periods": {"reference": args.reference_period, "target": args.target_period},
        "loader_sha256": loader_sha(), "data_seed": 0,
    }


def preflight(args):
    device = device_for(args.device)
    config, spec = prepare_spec(args)
    import cesnet_datazoo  # noqa: F401 — fail here before costly feature collection
    print(json.dumps({"device": str(device), "checkpoint": str(Path(args.checkpoint).resolve()),
                      "data_dir": spec["data_config"]["data_dir"], "periods": spec["periods"],
                      "environment": runtime_info()}, ensure_ascii=False, indent=2))
    return config, spec


def prepare(args):
    config, spec = preflight(args)
    root = Path(args.cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".lock"):
        manifest_path = root / "manifest.json"
        current_inventory = data_inventory(Path(spec["data_config"]["data_dir"]))
        if manifest_path.exists():
            info = json.loads(manifest_path.read_text())
            if info["spec"] != spec or info["data_inventory"] != current_inventory:
                raise ValueError("Data/config/checkpoint/loader changed. Use a new CACHE_DIR and OUTPUT_DIR.")
            validate_cache_files(root, info)
            print(f"Reusing verified feature cache: {root}", flush=True)
            return
        # Lazy import: CLI plans and synthetic tests do not require CESNET.
        sys.path.insert(0, str(CORE / "scripts"))
        import prototype_recalibration_tls22 as proto
        device = device_for(args.device)
        seed_all(0)
        started = time.perf_counter()
        model, _, classes = proto.load_source_model(args.checkpoint, device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        state = model.cls_head.state_dict()
        if set(state) != {"fc.weight", "fc.bias"}:
            raise ValueError("Expected the paper's linear classification head (fc.weight/fc.bias).")
        torch.save({k.removeprefix("fc."): v.detach().cpu() for k, v in state.items()}, root / "head.pt")
        raw_hashes, counts = {}, {}
        for role, period in spec["periods"].items():
            seed_all(0)
            cfg = {"data": dict(spec["data_config"])}
            cfg["data"]["num_classes"] = classes
            loader, loader_classes = proto.make_test_loader(cfg, period)
            if loader_classes != classes:
                raise ValueError(f"{period}: loader classes {loader_classes} != checkpoint classes {classes}")
            all_x, all_logits, all_y = [], [], []
            raw_hash = hashlib.sha256()
            for batch_i, batch in enumerate(loader):
                # IDs are snapshot-qualified row positions; the input stream hash binds their order.
                for key in ["ppi", "flow_stats", "label"]:
                    if key in batch:
                        value = batch[key].detach().cpu().contiguous()
                        raw_hash.update(canonical([key, list(value.shape), str(value.dtype)]).encode())
                        raw_hash.update(value.numpy().tobytes())
                with torch.no_grad():
                    stats = batch.get("flow_stats")
                    logits, features = model(
                        batch["ppi"].to(device), stats.to(device) if stats is not None else None,
                        return_repr=True)
                all_x.append(features.detach().cpu())
                all_logits.append(logits.detach().cpu())
                all_y.append(batch["label"].detach().cpu().long())
                if batch_i % 100 == 0:
                    print(f"{role} {period}: collected {batch_i + 1} batches", flush=True)
            if not all_x:
                raise ValueError(f"No samples in {period}")
            payload = {"features": torch.cat(all_x), "logits": torch.cat(all_logits),
                       "labels": torch.cat(all_y)}
            counts[role] = len(payload["labels"])
            if not torch.isfinite(payload["features"]).all() or not torch.isfinite(payload["logits"]).all():
                raise ValueError(f"Non-finite source outputs in {period}")
            torch.save(payload, root / f"{role}.pt")
            raw_hashes[role] = raw_hash.hexdigest()
            del loader, payload, all_x, all_logits, all_y
        info = {
            "spec": spec, "checkpoint_sha256": spec["checkpoint_sha256"],
            "periods": spec["periods"], "num_classes": classes,
            "input_stream_sha256": raw_hashes, "sample_counts": counts,
            "sample_id_definition": "fingerprint:role:row_index (snapshot rows, not original network flow IDs)",
            "fingerprint": digest({"spec": spec, "inputs": raw_hashes}),
            "data_inventory": data_inventory(Path(spec["data_config"]["data_dir"])),
            "files": {n: file_sha(root / n) for n in ["head.pt", "reference.pt", "target.pt"]},
            "prepare_wall_s": time.perf_counter() - started, "environment": runtime_info(),
        }
        atomic_json(manifest_path, info)  # Written last: interrupted preparation is never treated as complete.
        print(f"Feature cache complete: {root}; counts={counts}", flush=True)


def validate_cache_files(root, info):
    for name, expected in info["files"].items():
        if not (root / name).is_file() or file_sha(root / name) != expected:
            raise ValueError(f"Missing/modified cache file {root / name}. Use a new cache directory.")


def load_cache(args):
    root = Path(args.cache_dir).resolve()
    info = json.loads((root / "manifest.json").read_text())
    if info["spec"]["schema"] != SCHEMA or info["spec"]["loader_sha256"] != loader_sha():
        raise ValueError("Cache schema or feature loader changed. Prepare a new cache.")
    if info["periods"] != {"reference": args.reference_period, "target": args.target_period}:
        raise ValueError("Requested periods do not match the feature cache.")
    validate_cache_files(root, info)
    if file_sha(args.checkpoint) != info["checkpoint_sha256"]:
        raise ValueError("The source checkpoint does not match the feature cache.")
    state = torch.load(root / "head.pt", map_location="cpu", weights_only=True)
    head = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
    head.load_state_dict(state)
    data = {"head": head, "manifest": info, "num_classes": info["num_classes"]}
    for role in ["reference", "target"]:
        payload = torch.load(root / f"{role}.pt", map_location="cpu", weights_only=True)
        payload["labels"] = payload["labels"].numpy()
        data[role] = payload
    return data


def make_specs(suite):
    def spec(name, selector="margin", replay_ce=True, kd_weight=.5, temperature=2., k=5):
        return {"name": name, "selector": selector, "replay_ce": replay_ce,
                "kd_weight": kd_weight, "temperature": temperature, "replay_per_class": k}
    base = spec("margin_full")
    no_kd = spec("margin_replay", kd_weight=0.)
    if suite == "primary":
        return [spec("margin_ft_only", replay_ce=False, kd_weight=0.), no_kd,
                spec("margin_kd", replay_ce=False), base, spec("badge_full", selector="badge")]
    return [no_kd, spec("margin_lambda_0p1", kd_weight=.1), base,
            spec("margin_lambda_1", kd_weight=1.),
            spec("margin_temperature_1", temperature=1.), spec("margin_temperature_4", temperature=4.),
            spec("margin_replay_k1", k=1), spec("margin_replay_k10", k=10)]


def spec_key(spec):
    return canonical({k: v for k, v in spec.items() if k != "name"})


def select_query(selector, features, logits, budget, seed):
    if not 0 < budget < len(logits):
        raise ValueError(f"Budget {budget} must be smaller than the target pool ({len(logits)}).")
    if selector == "margin":
        top = torch.topk(logits, 2, dim=1).values
        return torch.argsort(top[:, 0] - top[:, 1])[:budget].numpy()
    if selector == "badge":
        sys.path.insert(0, str(CORE / "scripts"))
        from collapse_active_maintenance_tls22 import badge_selection
        return badge_selection(features, logits, logits.shape[1], budget, seed).numpy()
    raise ValueError(selector)


def replay_indices(labels, classes, k, seed):
    gen = torch.Generator().manual_seed(seed + 10007)
    result = []
    for c in range(classes):
        candidates = np.flatnonzero(labels == c)
        if len(candidates) < k:
            raise ValueError(f"Reference class {c} has {len(candidates)} samples, fewer than k={k}.")
        result.extend(candidates[torch.randperm(len(candidates), generator=gen)[:k].numpy()])
    return np.asarray(result, dtype=np.int64)


def fit_controlled(source_head, target_x, target_y, ref_x, ref_y, teacher_logits, *,
                   replay_ce, kd_weight, temperature, target_weight, replay_weight,
                   steps, batch_size, lr, weight_decay, seed, device, log_every=200):
    head = copy.deepcopy(source_head).to(device)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    # Separate CPU generators make the target stream independent of which loss terms are enabled.
    target_idx = torch.randint(len(target_x), (steps, batch_size),
                               generator=torch.Generator().manual_seed(seed + 20011))
    ref_idx = torch.randint(len(ref_x), (steps, batch_size),
                            generator=torch.Generator().manual_seed(seed + 30011))
    trace = {
        "optimizer_steps": steps, "target_presentations": steps * batch_size,
        "reference_ce_presentations": steps * batch_size if replay_ce else 0,
        "reference_kd_presentations": steps * batch_size if kd_weight else 0,
        "target_stream_sha256": hashlib.sha256(target_idx.numpy().tobytes()).hexdigest(),
        "reference_stream_sha256": hashlib.sha256(ref_idx.numpy().tobytes()).hexdigest(),
    }
    x, y = target_x.to(device), torch.as_tensor(target_y, dtype=torch.long, device=device)
    rx, ry = ref_x.to(device), torch.as_tensor(ref_y, dtype=torch.long, device=device)
    teacher = teacher_logits.to(device)
    target_idx, ref_idx = target_idx.to(device), ref_idx.to(device)
    sums = {"target_ce": 0., "reference_ce": 0., "reference_kd": 0., "loss": 0.}
    started = time.perf_counter()
    for step in range(steps):
        tx, ridx = target_idx[step], ref_idx[step]
        ce = F.cross_entropy(head(x[tx]), y[tx])
        loss = target_weight * ce
        rce = kd = torch.zeros((), device=device)
        if replay_ce or kd_weight:
            student = head(rx[ridx])
            if replay_ce:
                rce = F.cross_entropy(student, ry[ridx])
                loss = loss + replay_weight * rce
            if kd_weight:
                kd = F.kl_div(F.log_softmax(student / temperature, dim=1),
                              F.softmax(teacher[ridx] / temperature, dim=1),
                              reduction="batchmean") * temperature ** 2
                loss = loss + kd_weight * kd
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for key, value in [("target_ce", ce), ("reference_ce", rce), ("reference_kd", kd), ("loss", loss)]:
            sums[key] += float(value.detach())
        if log_every and ((step + 1) % log_every == 0 or step + 1 == steps):
            print(f"  step {step + 1}/{steps} loss={float(loss.detach()):.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trace["train_wall_s"] = time.perf_counter() - started
    trace["mean_losses"] = {k: v / steps for k, v in sums.items()}
    return head.cpu().eval(), trace


def confusion(labels, pred, classes):
    return np.bincount(labels * classes + pred, minlength=classes * classes).reshape(classes, classes)


def class_stats(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    return precision, recall, f1, support, predicted


def compare_predictions(labels, before, after, mask, classes, collapse, stable, threshold, drop):
    if not mask.any():
        raise ValueError("The unqueried evaluation set is empty.")
    cms = {"before": confusion(labels[mask], before[mask], classes),
           "after": confusion(labels[mask], after[mask], classes)}
    bp, br, bf, support, bcount = class_stats(cms["before"])
    ap, ar, af, _, acount = class_stats(cms["after"])
    noncollapse = sorted(set(range(classes)) - set(collapse))
    rows = [{"class_id": c, "support": int(support[c]),
             "is_collapse_group": c in collapse, "is_stable_group": c in stable,
             "before_precision": bp[c], "after_precision": ap[c],
             "before_recall": br[c], "after_recall": ar[c],
             "before_f1": bf[c], "after_f1": af[c], "delta_f1": af[c] - bf[c],
             "before_predicted_count": int(bcount[c]), "after_predicted_count": int(acount[c])}
            for c in range(classes)]
    summary = {"eval_samples": int(mask.sum()), "num_classes": classes}
    for group, indices in [("overall", list(range(classes))), ("collapse", collapse),
                            ("noncollapse", noncollapse), ("stable", stable)]:
        summary[f"{group}_supported_classes"] = int((support[indices] > 0).sum())
        for when, values in [("before", bf), ("after", af), ("delta", af - bf)]:
            summary[f"{group}_macro_f1_{when}"] = float(values[indices].mean()) if indices else None
    supported_nc = [c for c in noncollapse if support[c] > 0]
    delta = af[supported_nc] - bf[supported_nc]
    summary.update({
        "noncollapse_degraded_count": int((delta < 0).sum()),
        "noncollapse_degraded_fraction": float((delta < 0).mean()) if len(delta) else None,
        "noncollapse_drop_gt_threshold_count": int((delta < -drop).sum()),
        "noncollapse_worst_delta_f1": float(delta.min()) if len(delta) else None,
        "noncollapse_new_collapses": sum(bool(br[c] >= threshold and ar[c] < threshold) for c in supported_nc),
        "collapse_residual_count": sum(bool(support[c] > 0 and ar[c] < threshold) for c in collapse),
    })
    return summary, rows, cms


@torch.no_grad()
def predict(head, features, device, batch_size=4096):
    head = head.to(device).eval()
    predictions = [head(part.to(device)).argmax(1).cpu().numpy()
                   for part in features.split(batch_size)]
    head.cpu()
    return np.concatenate(predictions)


@contextlib.contextmanager
def tee_log(path):
    original = sys.stdout
    class Tee:
        def write(self, text):
            original.write(text)
            stream.write(text)
            return len(text)
        def flush(self):
            original.flush()
            stream.flush()
    with open(path, "w") as stream:
        sys.stdout = Tee()
        try:
            yield
        finally:
            sys.stdout = original


def qualified_id(data, role, row):
    return f"{data['manifest']['fingerprint']}:{role}:{row}"


def queries_for_seed(out, data, budget, seed):
    result = {}
    for selector in ["margin", "badge"]:
        directory = out / "selections" / selector / f"seed_{seed}"
        signature = {"cache": data["manifest"]["fingerprint"], "implementation": implementation_sha(),
                     "selector": selector, "budget": budget, "seed": seed}
        if not is_complete(directory, signature):
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "resolved_config.json", signature)
            started = time.perf_counter()
            idx = np.asarray(select_query(selector, data["target"]["features"],
                                          data["target"]["logits"], budget, seed), dtype=np.int64)
            if len(np.unique(idx)) != budget or idx.min() < 0 or idx.max() >= len(data["target"]["labels"]):
                raise ValueError("Selector returned invalid query indices.")
            atomic_json(directory / "selection.json", {"row_indices": idx.tolist(),
                                                       "selection_wall_s": time.perf_counter() - started})
            finish(directory, signature, ["resolved_config.json", "selection.json"])
        result[selector] = json.loads((directory / "selection.json").read_text())
    return result


def run_study(args, data):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    classes = data["num_classes"]
    collapse, stable = int_list(args.collapse_classes), int_list(args.stable_classes)
    if any(c < 0 or c >= classes for c in collapse + stable) or not collapse:
        raise ValueError("Evaluation groups do not match the checkpoint class universe.")
    if len(set(collapse)) != len(collapse) or len(set(stable)) != len(stable):
        raise ValueError("Evaluation groups must not contain duplicate class IDs.")
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    # Weights match the nominal full-CARE target/reference mixture; fixed even when k changes.
    nominal_target, nominal_ref = args.budget * 2, classes * 5
    protocol = {
        "schema": SCHEMA, "cache": data["manifest"]["fingerprint"],
        "checkpoint_sha256": data["manifest"]["checkpoint_sha256"],
        "implementation_sha256": implementation_sha(),
        "budget": args.budget, "optimizer_steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "target_weight": nominal_target / (nominal_target + nominal_ref),
        "replay_weight": nominal_ref / (nominal_target + nominal_ref),
        "collapse_classes": collapse, "stable_classes": stable,
        "collapse_recall_threshold": args.collapse_recall_threshold,
        "f1_drop_threshold": args.f1_drop_threshold,
        "num_classes": classes,
        "evaluation": "paired query-exclusion plus common exclusion of Margin and BADGE queries",
        "sampling": "fixed-step independent target/reference uniform streams with replacement",
        "device": str(device), "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "threads": args.threads,
    }
    with file_lock(out / ".lock"):
        manifest_path = out / "study_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["protocol"] != protocol:
            raise ValueError("Study protocol changed. Use a new OUTPUT_DIR; existing results will not be mixed.")
        if not manifest_path.exists():
            atomic_json(manifest_path, {"protocol": protocol, "environment": runtime_info(),
                                        "source_manifest": data["manifest"]})
        for seed in int_list(args.seeds):
            seed_all(seed)
            choices = queries_for_seed(out, data, args.budget, seed)
            n = len(data["target"]["labels"])
            common_mask = np.ones(n, dtype=bool)
            for choice in choices.values():
                common_mask[choice["row_indices"]] = False
            if not common_mask.any():
                raise ValueError("Margin/BADGE query union leaves no common evaluation samples.")
            for spec in make_specs(args.suite):
                directory = out / "runs" / spec["name"] / f"seed_{seed}"
                signature = {"protocol": protocol, "spec": spec, "seed": seed}
                if is_complete(directory, signature):
                    print(f"Verified, skipping {spec['name']} seed {seed}", flush=True)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "resolved_config.json", signature)
                atomic_json(directory / "environment.json", runtime_info())
                with tee_log(directory / "run.log"):
                    print(f"Running {spec['name']} seed {seed}", flush=True)
                    started = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    idx = np.asarray(choices[spec["selector"]]["row_indices"], dtype=np.int64)
                    ridx = replay_indices(data["reference"]["labels"], classes, spec["replay_per_class"], seed)
                    head, trace = fit_controlled(
                        data["head"], data["target"]["features"][idx], data["target"]["labels"][idx],
                        data["reference"]["features"][ridx], data["reference"]["labels"][ridx],
                        data["reference"]["logits"][ridx],
                        replay_ce=spec["replay_ce"], kd_weight=spec["kd_weight"], temperature=spec["temperature"],
                        target_weight=protocol["target_weight"], replay_weight=protocol["replay_weight"],
                        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                        weight_decay=args.weight_decay, seed=seed, device=device, log_every=args.log_every)
                    torch.save({"cls_head_state_dict": {f"fc.{k}": v for k, v in head.state_dict().items()},
                                "signature": signature}, directory / "repaired_head.pt")
                    infer_start = time.perf_counter()
                    after = predict(head, data["target"]["features"], device)
                    before = data["target"]["logits"].argmax(1).numpy()
                    strict = np.ones(n, dtype=bool)
                    strict[idx] = False
                    np.savez_compressed(directory / "predictions.npz",
                                        row_id=np.arange(n, dtype=np.int64),
                                        y_true=data["target"]["labels"], static_pred=before, repaired_pred=after,
                                        queried=~strict, strict_eval=strict, common_eval=common_mask)
                    trace["inference_and_prediction_save_wall_s"] = time.perf_counter() - infer_start
                    metric_rows, per_class, confusion_files, worst = {}, [], {}, []
                    for split, mask in [("strict", strict), ("common", common_mask)]:
                        metrics, rows, cms = compare_predictions(
                            data["target"]["labels"], before, after, mask, classes, collapse, stable,
                            args.collapse_recall_threshold, args.f1_drop_threshold)
                        metric_rows.update({f"{split}_{k}": v for k, v in metrics.items()})
                        per_class.extend([{"split": split, **r} for r in rows])
                        confusion_files.update({f"{split}_{when}": cm for when, cm in cms.items()})
                        bad = sorted((r for r in rows if not r["is_collapse_group"] and r["support"] > 0),
                                     key=lambda r: r["delta_f1"])[:10]
                        worst.extend([{"split": split, **r} for r in bad])
                    write_csv(directory / "per_class_metrics.csv", per_class)
                    write_csv(directory / "worst_noncollapse_classes.csv", worst)
                    np.savez_compressed(directory / "confusion_matrices.npz", **confusion_files)
                    write_csv(directory / "query_ids.csv", [
                        {"sample_id": qualified_id(data, "target", int(i)), "row_index": int(i),
                         "label": int(data["target"]["labels"][i])} for i in idx])
                    write_csv(directory / "replay_ids.csv", [
                        {"sample_id": qualified_id(data, "reference", int(i)), "row_index": int(i),
                         "label": int(data["reference"]["labels"][i]),
                         "used_for_ce": spec["replay_ce"], "used_for_kd": spec["kd_weight"] > 0} for i in ridx])
                    trace["selection_wall_s_cached"] = choices[spec["selector"]]["selection_wall_s"]
                    trace["process_lifetime_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                        1024 ** 2 if sys.platform == "darwin" else 1024)
                    trace["gpu_peak_allocated_mib"] = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                                       if device.type == "cuda" else None)
                    trace["run_wall_s_excluding_cached_extraction_and_selection"] = time.perf_counter() - started
                    atomic_json(directory / "training_trace.json", trace)
                    atomic_json(directory / "metrics.json", {
                        "name": spec["name"], "seed": seed, **spec, **metric_rows,
                        "train_wall_s": trace["train_wall_s"], "budget": args.budget,
                        "optimizer_steps": args.steps})
                    print(f"  strict overall={metric_rows['strict_overall_macro_f1_after']:.6f} "
                          f"collapse={metric_rows['strict_collapse_macro_f1_after']:.6f} "
                          f"noncollapse delta={metric_rows['strict_noncollapse_macro_f1_delta']:+.6f}", flush=True)
                finish(directory, signature, [
                    "resolved_config.json", "repaired_head.pt", "predictions.npz", "query_ids.csv",
                    "replay_ids.csv", "per_class_metrics.csv", "worst_noncollapse_classes.csv",
                    "confusion_matrices.npz", "training_trace.json", "metrics.json", "run.log", "environment.json"])
            summarize(out, args.suite)
        print(f"Results: {out / (args.suite + '_summary.csv')}", flush=True)


def summarize(out, suite):
    out = Path(out)
    rows, summaries = [], []
    study = json.loads((out / "study_manifest.json").read_text())["protocol"]
    for spec in make_specs(suite):
        selected = []
        for run in sorted((out / "runs" / spec["name"]).glob("seed_*")):
            if not (run / "complete.json").exists():
                continue
            signature = {"protocol": study, "spec": spec, "seed": int(run.name.removeprefix("seed_"))}
            if is_complete(run, signature):
                selected.append(json.loads((run / "metrics.json").read_text()))
        rows.extend(selected)
        item = {**spec, "n_seeds": len(selected), "recommended_n_seeds": 5 if suite == "primary" else 3,
                "meets_recommended_seed_count": len(selected) >= (5 if suite == "primary" else 3)}
        if selected:
            for key in selected[0]:
                if not key.startswith(("strict_", "common_")) and key != "train_wall_s":
                    continue
                values = [float(r[key]) for r in selected if r.get(key) is not None]
                if values:
                    item[key + "_mean"] = statistics.mean(values)
                    item[key + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(item)
    write_csv(out / f"{suite}_results_by_seed.csv", rows)
    write_csv(out / f"{suite}_summary.csv", summaries)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ["preflight", "prepare", "run", "summarize", "plan"]:
        q = sub.add_parser(command)
        q.add_argument("--config", default=str(CORE / "configs/eval_tls22.yaml"))
        q.add_argument("--checkpoint", default=str(CORE / "outputs/tls22_cnn/best_model.pt"))
        q.add_argument("--data-dir")
        q.add_argument("--cache-dir", default=str(CORE / "outputs/kbs_supplement_v1/cache"))
        q.add_argument("--output-dir", default=str(CORE / "outputs/kbs_supplement_v1"))
        q.add_argument("--reference-period", default="M-2022-4")
        q.add_argument("--target-period", default="M-2022-12")
        q.add_argument("--device", default="cuda")
        q.add_argument("--suite", choices=["primary", "sensitivity"], default="primary")
        q.add_argument("--seeds", default=None)
        q.add_argument("--budget", type=int, default=1000)
        q.add_argument("--steps", type=int, default=1380, help="Fixed optimizer updates; default 30 * ceil(2890/64).")
        q.add_argument("--batch-size", type=int, default=64)
        q.add_argument("--lr", type=float, default=.001)
        q.add_argument("--weight-decay", type=float, default=.0001)
        q.add_argument("--threads", type=int, default=8)
        q.add_argument("--log-every", type=int, default=200)
        q.add_argument("--collapse-classes", default=",".join(map(str, COLLAPSE)))
        q.add_argument("--stable-classes", default=",".join(map(str, STABLE)))
        q.add_argument("--collapse-recall-threshold", type=float, default=.1)
        q.add_argument("--f1-drop-threshold", type=float, default=.05)
    return p


def main():
    args = parser().parse_args()
    if args.seeds is None:
        args.seeds = "0,1,2,3,4" if args.suite == "primary" else "0,1,2"
    seeds = int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Supply distinct non-negative seeds.")
    for name in ["budget", "steps", "batch_size", "threads"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.lr <= 0 or args.weight_decay < 0 or args.log_every < 0:
        raise ValueError("Invalid optimizer/logging settings.")
    if not 0 < args.collapse_recall_threshold <= 1 or not 0 < args.f1_drop_threshold <= 1:
        raise ValueError("Thresholds must be in (0, 1].")
    if args.reference_period == args.target_period:
        raise ValueError("Reference and target periods must differ.")
    # Resolve dataset relative paths consistently with the original scripts.
    os.chdir(CORE)
    if args.command == "plan":
        print(json.dumps({"suite": args.suite, "seeds": seeds,
                          "planned_fits_before_resume": len(make_specs(args.suite)) * len(seeds),
                          "optimizer_steps_per_fit": args.steps, "configs": make_specs(args.suite)},
                         indent=2))
    elif args.command == "preflight":
        preflight(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run_study(args, load_cache(args))
    elif args.command == "summarize":
        with file_lock(Path(args.output_dir) / ".lock"):
            summarize(args.output_dir, args.suite)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
