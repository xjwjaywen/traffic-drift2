"""
Cohen et al. (2019) randomized smoothing for 1D-CNN traffic classifier.

This is the *vanilla* version — uniform Gaussian noise on all PPI features.
Used in CA-TTA Phase 0 to test whether certified accuracy degrades over
time on CESNET-TLS-Year22.
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, binomtest
from tqdm import tqdm


class SmoothedClassifier:
    """
    Wraps a base classifier f with Gaussian smoothing N(0, sigma^2 I).

    Implements the CERTIFY procedure from Cohen, Rosenfeld, Kolter (ICML 2019):
        - n0 noisy samples to predict the top class c_A
        - n  noisy samples to compute a lower confidence bound on p_A
        - certified radius r = sigma * Phi^{-1}(p_A_lower)
    """
    ABSTAIN = -1

    def __init__(self, base_model, num_classes: int, sigma: float):
        self.base = base_model
        self.K = num_classes
        self.sigma = sigma
        self.device = next(base_model.parameters()).device

    @torch.no_grad()
    def _sample_under_noise(self, x: torch.Tensor, num: int,
                             flow_stats: torch.Tensor = None,
                             chunk: int = 100) -> torch.Tensor:
        """
        Run num noisy forward passes for a SINGLE input x (shape (3, 30)).
        Returns counts: torch.LongTensor (K,).
        """
        counts = torch.zeros(self.K, dtype=torch.long, device=self.device)
        x = x.unsqueeze(0)  # (1, 3, 30)
        for start in range(0, num, chunk):
            k = min(chunk, num - start)
            x_rep = x.repeat(k, 1, 1)
            noise = torch.randn_like(x_rep) * self.sigma
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
    def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float,
                flow_stats: torch.Tensor = None):
        """
        Cohen et al. CERTIFY for a single input.

        Returns:
            (predicted_class, certified_radius)
            predicted_class is ABSTAIN (-1) if cannot certify with confidence.
        """
        # Step 1: select top class with n0 samples
        counts0 = self._sample_under_noise(x, n0, flow_stats)
        c_A = counts0.argmax().item()

        # Step 2: estimate lower confidence bound on p_A with n samples
        counts = self._sample_under_noise(x, n, flow_stats)
        n_A = counts[c_A].item()
        p_A_low = _binom_proportion_low(n_A, n, alpha)

        if p_A_low <= 0.5:
            return self.ABSTAIN, 0.0
        radius = self.sigma * norm.ppf(p_A_low)
        return c_A, float(radius)


def _binom_proportion_low(success: int, total: int, alpha: float) -> float:
    """
    One-sided Clopper-Pearson lower confidence bound for the binomial proportion.
    Cohen et al. uses statsmodels.proportion_confint(..., method='beta').
    We use scipy.stats.binomtest.proportion_ci to avoid extra dep.
    """
    if total == 0:
        return 0.0
    res = binomtest(success, total)
    ci = res.proportion_ci(confidence_level=1.0 - 2 * alpha,  # one-sided
                            method="exact")
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
                                  desc: str = "certify"):
    """
    Iterate the loader, call CERTIFY on each sample.
    Returns:
        dict {radius: certified_accuracy}
        plus per-sample (true_label, pred_class, certified_radius) records.
    """
    correct_at_r = {r: 0 for r in radii}
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
            c_A, rad = smoother.certify(x, n0=n0, n=n, alpha=alpha,
                                          flow_stats=fs)
            records.append((y, c_A, rad))
            total += 1
            if c_A == smoother.ABSTAIN:
                abstain += 1
                continue
            for r in radii:
                if c_A == y and rad >= r:
                    correct_at_r[r] += 1
            if max_samples is not None and total >= max_samples:
                break
        if max_samples is not None and total >= max_samples:
            break

    cert_acc = {r: (correct_at_r[r] / total if total > 0 else 0.0)
                for r in radii}
    return {"certified_accuracy": cert_acc,
            "abstain_rate": abstain / total if total > 0 else 0.0,
            "n_total": total,
            "records": records}
