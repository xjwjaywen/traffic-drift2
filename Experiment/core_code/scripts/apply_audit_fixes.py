"""
Apply all code fixes identified in the audit.
Run from Experiment/core_code/:
    python scripts/apply_audit_fixes.py
"""
import os
import re
import sys

FIXES_APPLIED = []


def fix_file(path, old, new, description):
    """Replace old with new in file. Report result."""
    if not os.path.exists(path):
        print(f"  SKIP (file not found): {path}")
        return False
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        if new in content:
            print(f"  ALREADY APPLIED: {description}")
            return True
        print(f"  SKIP (pattern not found): {description}")
        return False
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"  FIXED: {description}")
    FIXES_APPLIED.append(description)
    return True


def main():
    print("=" * 60)
    print("Applying audit fixes")
    print("=" * 60)

    # ── Fix 1: aggregate_seeds.py — ddof=0 → ddof=1 ──
    print("\n[1] Fix std computation: ddof=0 → ddof=1")
    fix_file(
        "scripts/aggregate_seeds.py",
        'agg[f"{m}_std"] = f"{np.std(vals):.4f}"',
        'agg[f"{m}_std"] = f"{np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}"',
        "aggregate_seeds.py: use sample std (ddof=1)",
    )

    # ── Fix 1b: aggregate_al_sweep.py — ddof=0 → ddof=1 ──
    fix_file(
        "scripts/aggregate_al_sweep.py",
        "out[k] = (float(arr.mean()), float(arr.std()), len(arr))",
        "out[k] = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr))",
        "aggregate_al_sweep.py: use sample std (ddof=1)",
    )

    # ── Fix 2: AURC formula ──
    print("\n[2] Fix AURC formula: /len(arrs) → /max(len(arrs)-1, 1)")
    fix_file(
        "tta_tc/utils/metrics.py",
        "return trapz(arrs) / len(arrs)",
        "return trapz(arrs) / max(len(arrs) - 1, 1)",
        "metrics.py: correct AURC normalization",
    )

    # ── Fix 3: Tent — model.train() only for norm layers ──
    print("\n[3] Fix model.train() in TTA baselines (Dropout issue)")
    tent_old = """    def adapt_batch(self, ppi, flow_stats=None):
        \"\"\"Adapt and classify a batch.\"\"\"
        self.model.train()"""
    tent_new = """    def _set_norm_train(self):
        \"\"\"Set only normalization layers to train mode.\"\"\"
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                m.train()

    def adapt_batch(self, ppi, flow_stats=None):
        \"\"\"Adapt and classify a batch.\"\"\"
        self._set_norm_train()"""
    fix_file("tta_tc/baselines/tent.py", tent_old, tent_new,
             "tent.py: only set norm layers to train mode")

    for fname, method_line in [
        ("tta_tc/baselines/sar.py", "        self.model.train()\n\n        logits = self.model(ppi, flow_stats)\n        probs = F.softmax"),
        ("tta_tc/baselines/eata.py", "        self.model.train()\n\n        logits = self.model(ppi, flow_stats)\n        probs = F.softmax"),
    ]:
        old_line = "        self.model.train()"
        new_lines = (
            "        self.model.eval()\n"
            "        for m in self.model.modules():\n"
            "            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):\n"
            "                m.train()"
        )
        fix_file(fname, method_line,
                 method_line.replace(old_line, new_lines, 1),
                 f"{os.path.basename(fname)}: only set norm layers to train mode")

    for fname in ["tta_tc/baselines/note.py", "tta_tc/baselines/cotta.py", "tta_tc/baselines/mvfc.py"]:
        if os.path.exists(fname):
            with open(fname, "r") as f:
                content = f.read()
            # Replace self.model.train() inside adapt_batch with norm-only train
            # But only the one inside adapt_batch, not __init__
            pattern = r"(    def adapt_batch\(self.*?\n(?:.*?\n)*?)(        self\.model\.train\(\))"
            match = re.search(pattern, content)
            if match and "self.model.eval()\n        for m in self.model.modules():" not in content:
                old_train = "        self.model.train()"
                new_train = (
                    "        self.model.eval()\n"
                    "        for m in self.model.modules():\n"
                    "            if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):\n"
                    "                m.train()"
                )
                # Only replace the first occurrence inside adapt_batch
                idx = content.find("def adapt_batch")
                if idx >= 0:
                    after = content[idx:]
                    train_idx = after.find("self.model.train()")
                    if train_idx >= 0:
                        abs_idx = idx + train_idx
                        content = (content[:abs_idx] +
                                   content[abs_idx:].replace("self.model.train()", new_train.strip(), 1))
                        with open(fname, "w") as f:
                            f.write(content)
                        print(f"  FIXED: {os.path.basename(fname)}: only set norm layers to train mode")
                        FIXES_APPLIED.append(f"{os.path.basename(fname)}: norm-only train")
            else:
                print(f"  ALREADY APPLIED or pattern differs: {os.path.basename(fname)}")

    # ── Fix 4: train.py — add cudnn deterministic ──
    print("\n[4] Add cudnn deterministic settings")
    fix_file(
        "train.py",
        "        torch.cuda.manual_seed_all(args.seed)",
        "        torch.cuda.manual_seed_all(args.seed)\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cudnn.benchmark = False",
        "train.py: enable cudnn deterministic mode",
    )

    # ── Fix 5: evaluate_tta.py — baseline period reset ──
    print("\n[5] Add baseline period reset in sequential eval")
    fix_file(
        "evaluate_tta.py",
        "                for period_name, test_loader in loaders:\n"
        "                    labels, preds, t = evaluate_tta_method(\n"
        "                        method, test_loader, device, f\"{method_key}@{period_name}\"\n"
        "                    )",
        "                for period_name, test_loader in loaders:\n"
        "                    # Reset baseline per period for fair comparison\n"
        "                    del method\n"
        "                    method_model_fresh = copy.deepcopy(model).to(device)\n"
        "                    method = MethodClass(method_model_fresh, adapt_cfg)\n"
        "                    labels, preds, t = evaluate_tta_method(\n"
        "                        method, test_loader, device, f\"{method_key}@{period_name}\"\n"
        "                    )",
        "evaluate_tta.py: reset baseline model per period",
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"Total fixes applied: {len(FIXES_APPLIED)}")
    for i, desc in enumerate(FIXES_APPLIED, 1):
        print(f"  {i}. {desc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
