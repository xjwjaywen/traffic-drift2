"""
Tier-2 labeled baselines for active TTA comparison.

Both methods consume a label budget per period (selected by a sampler) and
produce predictions, mirroring TTAEngine.adapt_period(). They share the
same evaluation harness as v9 for apples-to-apples comparison at the
same label budget.

  - KNNLabeled: k-nearest-neighbor classifier in source feature space
  - FineTuneHead: fine-tune only the classification head with queried labels
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..tta.samplers import get_sampler


class _PeriodLabeledBase:
    """Shared two-pass plumbing: collect features, sample labels, predict."""

    def __init__(self, model, cfg: dict):
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.num_classes = cfg["num_classes"]
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.label_budget_per_period = cfg.get("label_budget_per_period", 500)
        self.sampler_name = cfg.get("sampler", "random")
        self.sampler = get_sampler(self.sampler_name)
        self.seed = cfg.get("seed", None)
        self.labels_used = 0

    def reset_period(self):
        self.labels_used = 0

    def reset(self):
        self.labels_used = 0

    @torch.no_grad()
    def _collect(self, test_loader, period_name, keep_inputs=False):
        feats, labels, logits = [], [], []
        ppis, flow_stats_list = [], []
        for batch in tqdm(test_loader, desc=f"{self.__class__.__name__}@{period_name} pass1"):
            ppi = batch["ppi"].to(self.device)
            lbl = batch["label"]
            fs = batch.get("flow_stats")
            if fs is not None:
                fs = fs.to(self.device)
            lg, fr = self.model(ppi, fs, return_repr=True)
            feats.append(fr)
            labels.append(lbl)
            logits.append(lg)
            if keep_inputs:
                ppis.append(ppi)
                flow_stats_list.append(fs if fs is not None else None)
        result = (
            torch.cat(feats, dim=0),
            torch.cat(labels, dim=0),
            torch.cat(logits, dim=0),
        )
        if keep_inputs:
            ppis = torch.cat(ppis, dim=0)
            fs_concat = (
                torch.cat([f for f in flow_stats_list if f is not None], dim=0)
                if any(f is not None for f in flow_stats_list)
                else None
            )
            return result + (ppis, fs_concat)
        return result

    def _sample_idx(self, features, static_logits):
        N = features.size(0)
        budget = min(self.label_budget_per_period, N)
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.seed))
        pred = static_logits.argmax(dim=1)
        idx = self.sampler(
            features=features,
            static_logits=static_logits,
            pred_classes=pred,
            budget=budget,
            num_classes=self.num_classes,
            generator=gen,
        )
        self.labels_used = idx.size(0)
        return idx

    @torch.no_grad()
    def adapt_batch(self, ppi, flow_stats=None, labels=None):
        """Streaming fallback — just static."""
        logits = self.model(ppi, flow_stats)
        return logits, {"adapted": False, "method": "fallback"}


class KNNLabeled(_PeriodLabeledBase):
    """k-NN classifier on the queried 500 labeled samples in feature space."""

    def __init__(self, model, cfg: dict):
        super().__init__(model, cfg)
        self.k = cfg.get("knn_k", 10)
        self.knn_temperature = cfg.get("knn_temperature", 0.1)
        self.tta_blend = cfg.get("tta_blend", 0.5)  # blend with static logits

    @torch.no_grad()
    def adapt_period(self, test_loader, period_name: str = ""):
        feats, labels, static_logits = self._collect(test_loader, period_name)
        N = feats.size(0)

        idx = self._sample_idx(feats, static_logits)
        ref_feats = F.normalize(feats[idx], dim=1)              # (B, D)
        ref_labels = labels.to(self.device)[idx]                # (B,)

        # Chunked cosine similarity then weighted vote
        f_all = F.normalize(feats, dim=1)
        chunk_size = 8192
        all_preds = []
        w = self.tta_blend
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sim = f_all[start:end] @ ref_feats.T               # (chunk, B)
            k = min(self.k, ref_feats.size(0))
            topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)
            topk_labels = ref_labels[topk_idx]                  # (chunk, k)
            weights = F.softmax(topk_sim / self.knn_temperature, dim=1)
            knn_logits = torch.zeros(end - start, self.num_classes, device=self.device)
            knn_logits.scatter_add_(1, topk_labels, weights)
            knn_probs = knn_logits / (knn_logits.sum(dim=1, keepdim=True) + 1e-8)
            static_probs = F.softmax(static_logits[start:end], dim=1)
            final = (1 - w) * static_probs + w * knn_probs
            all_preds.append(final.argmax(dim=1))

        preds = torch.cat(all_preds, dim=0)
        return labels.cpu().numpy(), preds.cpu().numpy()


class FineTuneHead(_PeriodLabeledBase):
    """Fine-tune only classification head on queried labels, then predict."""

    def __init__(self, model, cfg: dict):
        super().__init__(model, cfg)
        self.ft_lr = cfg.get("ft_head_lr", 1e-3)
        self.ft_epochs = cfg.get("ft_head_epochs", 30)
        self.ft_batch_size = cfg.get("ft_head_batch_size", 64)
        self.weight_decay = cfg.get("ft_head_wd", 1e-4)

    def adapt_period(self, test_loader, period_name: str = ""):
        with torch.no_grad():
            feats, labels, static_logits = self._collect(test_loader, period_name)

        N = feats.size(0)
        idx = self._sample_idx(feats, static_logits)
        train_feats = feats[idx].detach()
        train_labels = labels.to(self.device)[idx]

        # Fresh copy of cls head — don't pollute base model across periods
        head = copy.deepcopy(self.model.cls_head).to(self.device)
        for p in head.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(head.parameters(), lr=self.ft_lr,
                                weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()

        head.train()
        n_train = train_feats.size(0)
        for _ in range(self.ft_epochs):
            perm = torch.randperm(n_train, device=self.device)
            for s in range(0, n_train, self.ft_batch_size):
                b = perm[s : s + self.ft_batch_size]
                logits = head(train_feats[b])
                loss = crit(logits, train_labels[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        # Predict in chunks
        all_preds = []
        chunk_size = 8192
        with torch.no_grad():
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                logits = head(feats[start:end])
                all_preds.append(logits.argmax(dim=1))
        preds = torch.cat(all_preds, dim=0)
        return labels.cpu().numpy(), preds.cpu().numpy()


class SupervisedNormAdapt(_PeriodLabeledBase):
    """
    Fine-tune ONLY normalization-layer parameters (GroupNorm γ/β) on the queried
    labels. The rationale: if drift mainly distorts feature distributions
    (diffuse drift), then re-calibrating normalization is more targeted than
    re-fitting the classifier head.

    This fills the missing cell in the experiment matrix:
        Tent (unsupervised norm adapt) failed.
        ft_head (supervised head adapt) succeeded on focal drift.
        SupervisedNormAdapt (supervised norm adapt) — never tested.
    """

    def __init__(self, model, cfg: dict):
        super().__init__(model, cfg)
        self.lr = cfg.get("snorm_lr", 1e-3)
        self.epochs = cfg.get("snorm_epochs", 30)
        self.batch_size = cfg.get("snorm_batch_size", 64)
        self.weight_decay = cfg.get("snorm_wd", 1e-4)

    def adapt_period(self, test_loader, period_name: str = ""):
        # Pass 1 collects features + raw inputs (need raw PPI for re-forward)
        with torch.no_grad():
            feats, labels, static_logits, ppis, flow_stats = self._collect(
                test_loader, period_name, keep_inputs=True
            )

        N = feats.size(0)
        idx = self._sample_idx(feats, static_logits)
        train_ppi = ppis[idx]
        train_labels = labels.to(self.device)[idx]
        train_fs = flow_stats[idx] if flow_stats is not None else None

        # Fresh model copy — don't pollute base model across periods
        adapted_model = copy.deepcopy(self.model).to(self.device)

        # Freeze everything, then unfreeze ONLY normalization params
        for p in adapted_model.parameters():
            p.requires_grad_(False)
        norm_params = adapted_model.encoder.get_norm_params()
        for p in norm_params:
            p.requires_grad_(True)

        opt = torch.optim.Adam(norm_params, lr=self.lr,
                                weight_decay=self.weight_decay)
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

        adapted_model.eval()
        # Re-forward all data through adapted model (chunked)
        all_preds = []
        chunk_size = 4096
        with torch.no_grad():
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                bp = ppis[start:end]
                bf = flow_stats[start:end] if flow_stats is not None else None
                logits = adapted_model(bp, bf)
                all_preds.append(logits.argmax(dim=1))
        preds = torch.cat(all_preds, dim=0)
        return labels.cpu().numpy(), preds.cpu().numpy()
