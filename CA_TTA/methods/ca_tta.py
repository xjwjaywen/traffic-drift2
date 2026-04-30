"""
CA-TTA: Certification-Aware Test-Time Adaptation.

Joint loss during TTA:
    L = L_CE(label, predicted)        # accuracy term
      + lambda * L_cert(soft p_A)      # MACER-style certified margin term

The cert term uses a differentiable surrogate of certified radius:
    L_cert = -inv_norm_cdf(soft_p_A)
where soft_p_A is the average softmax probability on the true class under
N noise samples per input. Maximising soft_p_A increases the certified
radius r = sigma * Phi^{-1}(soft_p_A).

Implements the supervised-label version (uses 500 oracle labels per period,
matching the active-TTA setup).
"""
import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiment", "core_code"))

from tta_tc.baselines.labeled_baselines import _PeriodLabeledBase


# ----------------------------------------------------- helpers --
SQRT_2 = math.sqrt(2.0)


def inv_norm_cdf(p: torch.Tensor) -> torch.Tensor:
    """
    Differentiable inverse standard-normal CDF (Phi^{-1}).

    Uses torch.erfinv: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1).
    Clamp p to (eps, 1-eps) to avoid +/- infinity at boundaries.
    """
    eps = 1e-6
    p = p.clamp(min=eps, max=1.0 - eps)
    return SQRT_2 * torch.erfinv(2.0 * p - 1.0)


def _norm_layers(model):
    """Yield all GroupNorm/BatchNorm/LayerNorm modules in the encoder."""
    for m in model.encoder.modules():
        if isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.LayerNorm)):
            yield m


# ----------------------------------------------------- CA-TTA --
class CertAwareNormAdapt(_PeriodLabeledBase):
    """
    CA-TTA method: supervised norm-only adaptation with a joint
    accuracy + certified-margin objective.

    cfg keys (all optional):
      ca_lambda          : weight on cert term (default 1.0)
      ca_sigma           : Gaussian smoothing sigma (default 0.25)
      ca_n_noise         : noise samples per input per step (default 8)
      ca_lr              : Adam lr (default 1e-3)
      ca_epochs          : epochs (default 30)
      ca_batch_size      : batch size (default 64)
      ca_wd              : weight decay (default 1e-4)
      ca_only_correct    : only apply cert loss to correctly predicted
                           samples (MACER-style hinge); default True
    """

    def __init__(self, model, cfg: dict, source_stats=None):
        super().__init__(model, cfg)
        self.lambda_cert = cfg.get("ca_lambda", 1.0)
        self.sigma = cfg.get("ca_sigma", 0.25)
        self.n_noise = cfg.get("ca_n_noise", 8)
        self.lr = cfg.get("ca_lr", 1e-3)
        self.epochs = cfg.get("ca_epochs", 30)
        self.batch_size = cfg.get("ca_batch_size", 64)
        self.weight_decay = cfg.get("ca_wd", 1e-4)
        self.only_correct = cfg.get("ca_only_correct", True)
        # Cert loss form: "macer" (default), "stability", "none"
        self.loss_type = cfg.get("ca_loss_type", "macer")
        # Channels to perturb during training (matches eval-side smoothing)
        # Default: skip discrete direction channel (channel 1)
        self.smooth_channels = tuple(cfg.get("ca_smooth_channels", (0, 2)))

    def _build_optimizer(self, model):
        # Freeze all, then unfreeze norm parameters
        for p in model.parameters():
            p.requires_grad_(False)
        norm_params = []
        for layer in _norm_layers(model):
            if layer.weight is not None:
                layer.weight.requires_grad_(True)
                norm_params.append(layer.weight)
            if layer.bias is not None:
                layer.bias.requires_grad_(True)
                norm_params.append(layer.bias)
        return torch.optim.Adam(norm_params, lr=self.lr,
                                weight_decay=self.weight_decay)

    def _make_train_noise(self, ppi_rep: torch.Tensor) -> torch.Tensor:
        """Build noise tensor honouring smooth_channels."""
        noise = torch.randn_like(ppi_rep) * self.sigma
        if len(self.smooth_channels) < ppi_rep.size(1):
            mask = torch.zeros_like(ppi_rep)
            for c in self.smooth_channels:
                mask[:, c, :] = 1.0
            noise = noise * mask
        return noise

    def _train_step(self, model, ppi, labels, flow_stats, opt):
        """One training step: joint CE + cert loss."""
        B = ppi.size(0)
        ppi_rep = ppi.unsqueeze(0).repeat(self.n_noise, 1, 1, 1).reshape(
            self.n_noise * B, *ppi.shape[1:])
        noise = self._make_train_noise(ppi_rep)
        ppi_noisy = ppi_rep + noise

        if flow_stats is not None:
            fs_rep = flow_stats.unsqueeze(0).repeat(
                self.n_noise, 1, 1).reshape(self.n_noise * B, -1)
        else:
            fs_rep = None

        logits = model(ppi_noisy, fs_rep)            # (N*B, K)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        probs_reshaped = probs.view(self.n_noise, B, -1)
        soft_probs = probs_reshaped.mean(dim=0)       # (B, K)
        soft_log = (soft_probs + 1e-8).log()
        ce = F.nll_loss(soft_log, labels)

        # Cert margin loss — different formulations
        if self.loss_type == "none" or self.lambda_cert == 0.0:
            cert_term = torch.tensor(0.0, device=ppi.device)
        elif self.loss_type == "stability":
            # Stability training: KL between clean prediction and avg noisy
            with torch.no_grad():
                clean_logits = model(ppi, flow_stats)
                clean_probs = F.softmax(clean_logits, dim=1)
            cert_term = F.kl_div(soft_log, clean_probs, reduction="batchmean")
        else:  # "macer" (default)
            p_A = soft_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            if self.only_correct:
                top_class = soft_probs.argmax(dim=1)
                mask = (top_class == labels) & (p_A > 0.5)
                if mask.any():
                    cert_term = -inv_norm_cdf(p_A[mask]).mean()
                else:
                    cert_term = torch.tensor(0.0, device=ppi.device)
            else:
                # Clamp p_A away from 0/1 to avoid +/- inf
                cert_term = -inv_norm_cdf(p_A.clamp(min=1e-3, max=1.0 - 1e-3)).mean()

        total = ce + self.lambda_cert * cert_term
        opt.zero_grad()
        total.backward()
        opt.step()
        return float(ce), float(cert_term), float(total)

    def adapt_period(self, test_loader, period_name: str = ""):
        with torch.no_grad():
            feats, labels, static_logits, ppis, flow_stats = self._collect(
                test_loader, period_name, keep_inputs=True)

        N = feats.size(0)
        idx = self._sample_idx(feats, static_logits)
        train_ppi = ppis[idx]
        train_labels = labels.to(self.device)[idx]
        train_fs = flow_stats[idx] if flow_stats is not None else None

        adapted_model = copy.deepcopy(self.model).to(self.device)
        opt = self._build_optimizer(adapted_model)

        adapted_model.train()
        n_train = train_ppi.size(0)
        last_log = None
        for ep in range(self.epochs):
            perm = torch.randperm(n_train, device=self.device)
            ep_ce, ep_cert, ep_total = 0.0, 0.0, 0.0
            n_batch = 0
            for s in range(0, n_train, self.batch_size):
                b = perm[s : s + self.batch_size]
                bp = train_ppi[b]
                bl = train_labels[b]
                bf = train_fs[b] if train_fs is not None else None
                ce, cert, total = self._train_step(
                    adapted_model, bp, bl, bf, opt)
                ep_ce += ce
                ep_cert += cert
                ep_total += total
                n_batch += 1
            last_log = (ep_ce / n_batch, ep_cert / n_batch, ep_total / n_batch)
        if last_log:
            print(f"  [CA-TTA] last epoch: ce={last_log[0]:.4f}  "
                  f"cert={last_log[1]:.4f}  total={last_log[2]:.4f}")

        adapted_model.eval()

        # Re-forward all (without noise — cert eval is done separately)
        all_preds = []
        chunk = 4096
        with torch.no_grad():
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                bp = ppis[s:e]
                bf = flow_stats[s:e] if flow_stats is not None else None
                logits = adapted_model(bp, bf)
                all_preds.append(logits.argmax(dim=1))
        preds = torch.cat(all_preds, dim=0)

        # Stash adapted model so the certify caller can use it
        self.last_adapted_model = adapted_model
        return labels.cpu().numpy(), preds.cpu().numpy()
