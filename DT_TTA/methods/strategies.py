"""
DT-TTA adaptation strategies.

All three classes share the two-pass adaptation skeleton from
`tta_tc.baselines.labeled_baselines._PeriodLabeledBase`:
  Pass 1 — collect features/labels/PPI; sample 500 labels
  Adapt — supervised CE on a subset of parameters
  Pass 2 — re-forward all data through adapted model

Differences:
  SelectiveNormAdapt : per-channel-masked GroupNorm γ/β
  FocalStrategy       : SelectiveNormAdapt + bias-only classifier head update
  DiffuseStrategy     : full GroupNorm γ/β + full classifier head
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Allow imports from main project
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))

from tta_tc.baselines.labeled_baselines import _PeriodLabeledBase

from .topology import (
    collect_groupnorm_input_stats,
    compute_channel_drift_scores,
    select_drifted_channels,
    random_channel_mask,
)


# ---------------------------------------------------- helpers --
def _norm_layers_with_names(model):
    """Iterate over (name, GroupNorm/BatchNorm/LayerNorm) in encoder."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.LayerNorm)):
            yield name, m


def _attach_grad_hook_for_mask(module: nn.Module, channel_mask_bool: np.ndarray):
    """
    Register backward hooks on `weight` (γ) and `bias` (β) of a norm layer that
    zero-out gradients on channels where mask is False.
    """
    mask_t = torch.tensor(channel_mask_bool.astype(np.float32),
                          device=next(module.parameters()).device)
    handles = []

    def _hook(grad):
        return grad * mask_t

    if module.weight is not None and module.weight.requires_grad:
        handles.append(module.weight.register_hook(_hook))
    if module.bias is not None and module.bias.requires_grad:
        handles.append(module.bias.register_hook(_hook))
    return handles


# ---------------------------------------------------- DTTABase --
class _DTTABase(_PeriodLabeledBase):
    """Common scaffolding: source-stats holder, channel-mask computation."""

    def __init__(self, model, cfg, source_stats=None):
        super().__init__(model, cfg)
        self.source_stats = source_stats        # dict[layer_name] -> {mean,var,n}
        self.top_k_ratio = cfg.get("dt_top_k_ratio", 0.4)
        self.use_random_mask = cfg.get("dt_random_mask", False)
        self.lr = cfg.get("dt_lr", 1e-3)
        self.epochs = cfg.get("dt_epochs", 30)
        self.batch_size = cfg.get("dt_batch_size", 64)
        self.weight_decay = cfg.get("dt_wd", 1e-4)
        self.last_drift_scores = None
        self.last_channel_masks = None

    @torch.no_grad()
    def _diagnose(self, ppis, flow_stats):
        """Run a forward pass on test PPIs through the source model
        to obtain target GroupNorm input stats; compute drift scores."""
        if self.source_stats is None:
            raise RuntimeError("DT-TTA methods need source_stats. Pre-compute "
                               "with scripts/compute_source_stats.py and load.")

        # Build a tiny single-batch loader over test PPIs (chunked)
        class _OneShotLoader:
            def __init__(self, ppi, fs, batch_size=512):
                self.ppi = ppi
                self.fs = fs
                self.bs = batch_size
            def __iter__(self):
                for s in range(0, self.ppi.size(0), self.bs):
                    e = min(s + self.bs, self.ppi.size(0))
                    yield {"ppi": self.ppi[s:e],
                           "flow_stats": (self.fs[s:e]
                                          if self.fs is not None else None)}
            def __len__(self):
                return (self.ppi.size(0) + self.bs - 1) // self.bs

        loader = _OneShotLoader(ppis, flow_stats)
        target_stats = collect_groupnorm_input_stats(
            self.model, loader, self.device, max_batches=None)
        scores = compute_channel_drift_scores(self.source_stats, target_stats)
        self.last_drift_scores = scores
        if self.use_random_mask:
            seed = self.seed if self.seed is not None else 0
            masks = random_channel_mask(scores, self.top_k_ratio, seed=seed)
        else:
            masks = select_drifted_channels(scores, self.top_k_ratio)
        self.last_channel_masks = masks
        return masks

    def _build_optimizer(self, model, channel_masks=None,
                          adapt_norm: bool = True,
                          adapt_head_full: bool = False,
                          adapt_head_bias_only: bool = False):
        """
        Configure which parameters require gradient and return an Adam opt.
        """
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)

        params_to_opt = []
        hook_handles = []

        if adapt_norm:
            for name, layer in _norm_layers_with_names(model.encoder):
                if layer.weight is not None:
                    layer.weight.requires_grad_(True)
                    params_to_opt.append(layer.weight)
                if layer.bias is not None:
                    layer.bias.requires_grad_(True)
                    params_to_opt.append(layer.bias)
                # If we have a per-layer channel mask, attach grad hook
                if channel_masks is not None and name in channel_masks:
                    hook_handles += _attach_grad_hook_for_mask(
                        layer, channel_masks[name])

        if adapt_head_full:
            for p in model.cls_head.parameters():
                p.requires_grad_(True)
                params_to_opt.append(p)
        elif adapt_head_bias_only:
            for n, p in model.cls_head.named_parameters():
                if "bias" in n:
                    p.requires_grad_(True)
                    params_to_opt.append(p)

        opt = torch.optim.Adam(params_to_opt, lr=self.lr,
                                weight_decay=self.weight_decay)
        return opt, hook_handles

    def _train_loop(self, adapted_model, opt, train_ppi, train_labels,
                    train_fs):
        crit = nn.CrossEntropyLoss()
        adapted_model.train()
        n_train = train_ppi.size(0)
        for _ in range(self.epochs):
            perm = torch.randperm(n_train, device=self.device)
            for s in range(0, n_train, self.batch_size):
                b = perm[s : s + self.batch_size]
                bp = train_ppi[b]
                bl = train_labels[b]
                bf = train_fs[b] if train_fs is not None else None
                logits = adapted_model(bp, bf)
                loss = crit(logits, bl)
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def _predict_all(self, adapted_model, ppis, flow_stats):
        adapted_model.eval()
        all_preds = []
        chunk = 4096
        for s in range(0, ppis.size(0), chunk):
            e = min(s + chunk, ppis.size(0))
            bp = ppis[s:e]
            bf = flow_stats[s:e] if flow_stats is not None else None
            logits = adapted_model(bp, bf)
            all_preds.append(logits.argmax(dim=1))
        return torch.cat(all_preds, dim=0)


# ---------------------------------------------------- SelectiveNormAdapt --
class SelectiveNormAdapt(_DTTABase):
    """Setting (3): only update GroupNorm γ/β on drifted channels."""

    def adapt_period(self, test_loader, period_name: str = ""):
        with torch.no_grad():
            feats, labels, static_logits, ppis, flow_stats = self._collect(
                test_loader, period_name, keep_inputs=True)

        idx = self._sample_idx(feats, static_logits)
        train_ppi = ppis[idx]
        train_labels = labels.to(self.device)[idx]
        train_fs = flow_stats[idx] if flow_stats is not None else None

        # Diagnose drift topology on this period (no labels)
        masks = self._diagnose(ppis, flow_stats)

        adapted_model = copy.deepcopy(self.model).to(self.device)
        opt, hooks = self._build_optimizer(
            adapted_model, channel_masks=masks,
            adapt_norm=True, adapt_head_full=False, adapt_head_bias_only=False)

        self._train_loop(adapted_model, opt, train_ppi, train_labels, train_fs)
        for h in hooks:
            h.remove()

        preds = self._predict_all(adapted_model, ppis, flow_stats)
        return labels.cpu().numpy(), preds.cpu().numpy()


# ---------------------------------------------------- FocalStrategy --
class FocalStrategy(_DTTABase):
    """Setting (4): SelectiveNormAdapt + classifier head BIAS-only update."""

    def adapt_period(self, test_loader, period_name: str = ""):
        with torch.no_grad():
            feats, labels, static_logits, ppis, flow_stats = self._collect(
                test_loader, period_name, keep_inputs=True)

        idx = self._sample_idx(feats, static_logits)
        train_ppi = ppis[idx]
        train_labels = labels.to(self.device)[idx]
        train_fs = flow_stats[idx] if flow_stats is not None else None

        masks = self._diagnose(ppis, flow_stats)

        adapted_model = copy.deepcopy(self.model).to(self.device)
        opt, hooks = self._build_optimizer(
            adapted_model, channel_masks=masks,
            adapt_norm=True, adapt_head_full=False, adapt_head_bias_only=True)

        self._train_loop(adapted_model, opt, train_ppi, train_labels, train_fs)
        for h in hooks:
            h.remove()

        preds = self._predict_all(adapted_model, ppis, flow_stats)
        return labels.cpu().numpy(), preds.cpu().numpy()


# ---------------------------------------------------- DiffuseStrategy --
class DiffuseStrategy(_DTTABase):
    """Setting (5): full GroupNorm γ/β + full classifier head."""

    def adapt_period(self, test_loader, period_name: str = ""):
        with torch.no_grad():
            feats, labels, static_logits, ppis, flow_stats = self._collect(
                test_loader, period_name, keep_inputs=True)

        idx = self._sample_idx(feats, static_logits)
        train_ppi = ppis[idx]
        train_labels = labels.to(self.device)[idx]
        train_fs = flow_stats[idx] if flow_stats is not None else None

        adapted_model = copy.deepcopy(self.model).to(self.device)
        opt, hooks = self._build_optimizer(
            adapted_model, channel_masks=None,           # no masking
            adapt_norm=True, adapt_head_full=True, adapt_head_bias_only=False)

        self._train_loop(adapted_model, opt, train_ppi, train_labels, train_fs)
        for h in hooks:
            h.remove()

        preds = self._predict_all(adapted_model, ppis, flow_stats)
        return labels.cpu().numpy(), preds.cpu().numpy()
