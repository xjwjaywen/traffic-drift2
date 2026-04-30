"""
Cohen et al. (2019) randomized smoothing for 1D-CNN traffic classifier.

Constraint-aware version: by default, noise is applied ONLY to continuous
PPI channels (packet size, IPT). The discrete direction channel (binary
+/-1 values) is held fixed because adding Gaussian noise to it produces
semantically meaningless network states.

This matches CertTA's separation of additive (size, time) vs structural
(direction / packet-presence) perturbations. The certified radius then
applies to the L2-norm of perturbations in the (size, IPT) subspace.
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, binomtest
from tqdm import tqdm


class SmoothedClassifier:
    """
    Wraps a base classifier f with Gaussian smoothing N(0, sigma^2 I) on
    selected input channels. Direction channel is held fixed by default.

    cfg-style args:
      sigma                : noise std on smoothed channels
      smooth_channels      : list of channel indices to add noise to
                             (default [0, 2] = size + IPT)
    """
    ABSTAIN = -1

    def __init__(self, base_model, num_classes: int, sigma: float,
                 smooth_channels=(0, 2)):
        self.base = base_model
        self.K = num_classes
        self.sigma = sigma
        self.smooth_channels = tuple(smooth_channels)
        self.device = next(base_model.parameters()).device
        # Effective dimensionality for cert radius interpretation
        # — count number of (channel, position) cells we add noise to
        # PPI shape: (3, 30); 30 positions per channel
        self.smooth_dim = 30 * len(self.smooth_channels)

    def _make_noise(self, x_rep: torch.Tensor) -> torch.Tensor:
        """
        Create noise tensor for x_rep (shape (k, 3, 30)). Zero on
        non-smoothed channels.
        """
        noise = torch.randn_like(x_rep) * self.sigma
        if len(self.smooth_channels) < x_rep.size(1):
            mask = torch.zeros_like(x_rep)
            for c in self.smooth_channels:
                mask[:, c, :] = 1.0
            noise = noise * mask
        return noise

    @torch.no_grad()
    def _sample_under_noise(self, x: torch.Tensor, num: int,
                             flow_stats: torch.Tensor = None,
                             chunk: int = 100) -> torch.Tensor:
        """
        Run num noisy forward passes for a SINGLE input x (shape (3, 30)).
        Returns counts: torch.LongTensor (K,).
        """
        counts = torch.zeros(self.K, dtype=torch.long, device=self.device)
        x = x.unsqueeze(0)
        for start in range(0, num, chunk):
            k = min(chunk, num - start)
            x_rep = x.repeat(k, 1, 1)
            noise = self._make_noise(x_rep)
            x_noisy = x_rep + noise
            if flow_stats is not None:
                fs_rep = flow_stats.unsqueeze(0).repeat(k, 1)
                logits = self.base(x_noisy, fs_rep)
            else:
                logits = self.base(x_noisy)
            preds = logits.argmax(dim=1)
            counts.scatter_add_(0, preds, torch.ones_like(preds))
        return counts

    @torch.no_grad()
    def predict_clean(self, x: torch.Tensor,
                      flow_stats: torch.Tensor = None) -> int:
        """No-noise prediction (for clean accuracy)."""
        x_in = x.unsqueeze(0)
        fs_in = flow_stats.unsqueeze(0) if flow_stats is not None else None
        logits = self.base(x_in, fs_in)
        return int(logits.argmax(dim=1).item())

    @torch.no_grad()
    def predict_smoothed(self, x: torch.Tensor, num: int,
                          flow_stats: torch.Tensor = None) -> int:
        """Majority-vote prediction under noise (for smoothed accuracy)."""
        counts = self._sample_under_noise(x, num, flow_stats)
        return int(counts.argmax().item())

    @torch.no_grad()
    def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float,
                flow_stats: torch.Tensor = None):
        """Cohen et al. CERTIFY for a single input."""
        counts0 = self._sample_under_noise(x, n0, flow_stats)
        c_A = counts0.argmax().item()

        counts = self._sample_under_noise(x, n, flow_stats)
        n_A = counts[c_A].item()
        p_A_low = _binom_proportion_low(n_A, n, alpha)

        if p_A_low <= 0.5:
            return self.ABSTAIN, 0.0
        radius = self.sigma * norm.ppf(p_A_low)
        return c_A, float(radius)


def _binom_proportion_low(success: int, total: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound."""
    if total == 0:
        return 0.0
    res = binomtest(success, total)
    ci = res.proportion_ci(confidence_level=1.0 - 2 * alpha, method="exact")
    return float(ci.low)


# -------------------------------------------------------------------- eval --
def certified_accuracy_at_radii(smoother: SmoothedClassifier,
                                  loader,
                                  device,
                                  radii: list,
                                  n0: int = 100,
                                  n: int = 500,
                                  alpha: float = 0.001,
                                  max_samples: int = None,
                                  smoothed_acc_n: int = 100,
                                  desc: str = "certify"):
    """
    Iterate the loader, call CERTIFY on each sample. Also reports
    clean accuracy and smoothed accuracy alongside certified accuracy.

    Returns:
        dict with:
          certified_accuracy : {radius: cert_acc}
          clean_accuracy     : float (no noise)
          smoothed_accuracy  : float (majority vote, no cert constraint)
          abstain_rate       : float
          n_total            : int
          records            : list of (true_label, pred_class, radius, ...)
    """
    correct_at_r = {r: 0 for r in radii}
    correct_clean = 0
    correct_smoothed = 0
    total = 0
    abstain = 0
    records = []

    for batch in tqdm(loader, desc=desc):
        ppi = batch["ppi"]
        labels = batch["label"]
        flow_stats = batch.get("flow_stats")
        for i in range(ppi.size(0)):
            x = ppi[i].to(device)
            y = labels[i].item()
            fs = flow_stats[i].to(device) if flow_stats is not None else None

            # Clean (no noise)
            clean_pred = smoother.predict_clean(x, fs)
            if clean_pred == y:
                correct_clean += 1

            # Smoothed (majority vote)
            sm_pred = smoother.predict_smoothed(x, smoothed_acc_n, fs)
            if sm_pred == y:
                correct_smoothed += 1

            # Certified
            c_A, rad = smoother.certify(x, n0=n0, n=n, alpha=alpha,
                                          flow_stats=fs)
            records.append((y, c_A, rad, clean_pred, sm_pred))
            total += 1
            if c_A == smoother.ABSTAIN:
                abstain += 1
            else:
                for r in radii:
                    if c_A == y and rad >= r:
                        correct_at_r[r] += 1
            if max_samples is not None and total >= max_samples:
                break
        if max_samples is not None and total >= max_samples:
            break

    cert_acc = {r: (correct_at_r[r] / total if total > 0 else 0.0)
                for r in radii}
    return {
        "certified_accuracy": cert_acc,
        "clean_accuracy": correct_clean / total if total > 0 else 0.0,
        "smoothed_accuracy": correct_smoothed / total if total > 0 else 0.0,
        "abstain_rate": abstain / total if total > 0 else 0.0,
        "n_total": total,
        "records": records,
    }
