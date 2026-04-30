"""
Drift topology diagnosis for the DT-TTA framework.

Pipeline:
  1. Hook every GroupNorm layer in the encoder; record per-channel mean/var
     of its INPUT activations.
  2. Build (source_stats, target_stats) by running source/target loaders.
  3. Per-channel drift score = symmetric KL between source and target
     Gaussian fits.
  4. Topology classification: focal (concentrated) vs diffuse (spread).
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# -------------------------------------------------------------------- stats --
@torch.no_grad()
def collect_groupnorm_input_stats(model, loader, device, max_batches=None):
    """
    Forward-hook every GroupNorm in the encoder, collect per-channel mean/var
    of its input across the loader.

    Returns:
        dict[layer_name] -> {"mean": (C,) np, "var": (C,) np, "n": int}
    """
    layer_buffers = defaultdict(lambda: {"sum": None, "sumsq": None, "n": 0})
    handles = []

    def _make_hook(name):
        def hook(module, inputs):
            # forward_pre_hook signature: (module, inputs)
            x = inputs[0]              # (B, C, L) for Conv1d-style GN
            if x.dim() == 3:
                # average over spatial dim -> (B, C)
                xv = x.mean(dim=2)
            elif x.dim() == 2:
                xv = x
            else:
                # flatten everything except channel
                xv = x.transpose(1, -1).reshape(-1, x.size(1))
            buf = layer_buffers[name]
            s = xv.sum(dim=0).detach().cpu().numpy()
            ss = (xv * xv).sum(dim=0).detach().cpu().numpy()
            if buf["sum"] is None:
                buf["sum"] = s
                buf["sumsq"] = ss
            else:
                buf["sum"] += s
                buf["sumsq"] += ss
            buf["n"] += xv.size(0)
            return None  # don't modify inputs
        return hook

    # Register hooks on every GroupNorm in the encoder
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            handles.append(module.register_forward_pre_hook(_make_hook(name)))

    model.eval()
    n_batches = 0
    try:
        for batch in tqdm(loader, desc="collect GN stats", leave=False):
            ppi = batch["ppi"].to(device)
            fs = batch.get("flow_stats")
            if fs is not None:
                fs = fs.to(device)
            _ = model(ppi, fs)
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name, buf in layer_buffers.items():
        n = buf["n"]
        if n == 0:
            continue
        mu = buf["sum"] / n
        var = buf["sumsq"] / n - mu * mu
        var = np.clip(var, 1e-8, None)
        out[name] = {"mean": mu, "var": var, "n": n}
    return out


# ------------------------------------------------------------- drift scores --
def compute_channel_drift_scores(source_stats: dict, target_stats: dict):
    """
    Per-channel drift score = symmetric KL between source and target Gaussian.

    Returns:
        dict[layer_name] -> (C,) np array of drift scores
    """
    scores = {}
    for name in source_stats:
        if name not in target_stats:
            continue
        s = source_stats[name]
        t = target_stats[name]
        mu_s, var_s = s["mean"], s["var"]
        mu_t, var_t = t["mean"], t["var"]
        # symmetric KL between two univariate Gaussians per channel
        kl_st = 0.5 * (np.log(var_t / var_s) + (var_s + (mu_s - mu_t) ** 2) / var_t - 1.0)
        kl_ts = 0.5 * (np.log(var_s / var_t) + (var_t + (mu_t - mu_s) ** 2) / var_s - 1.0)
        scores[name] = (kl_st + kl_ts) / 2.0
    return scores


# ------------------------------------------------------ topology classifier --
def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative score distribution."""
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / x[-1] if x[-1] > 0 else 0))  # placeholder


def gini_coefficient(x: np.ndarray) -> float:
    """Standard Gini coefficient: 0 = perfectly uniform, 1 = fully concentrated."""
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0)
    if x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x_sorted) - (n + 1) * x_sorted.sum())
                 / (n * x_sorted.sum()))


def classify_topology(channel_scores: dict,
                      focal_gini_threshold: float = 0.6,
                      diffuse_gini_threshold: float = 0.3):
    """
    Classify drift topology based on score concentration across channels.

    Returns:
        label: "focal" | "diffuse" | "mixed"
        gini_per_layer: dict
        overall_gini: float (weighted by layer size)
    """
    gini_per_layer = {}
    total_n, weighted = 0, 0.0
    for name, scores in channel_scores.items():
        g = gini_coefficient(scores)
        gini_per_layer[name] = g
        weighted += g * scores.size
        total_n += scores.size
    overall = weighted / total_n if total_n > 0 else 0.0

    if overall >= focal_gini_threshold:
        label = "focal"
    elif overall <= diffuse_gini_threshold:
        label = "diffuse"
    else:
        label = "mixed"
    return label, gini_per_layer, overall


# ---------------------------------------------------- channel mask selection --
def select_drifted_channels(channel_scores: dict, top_k_ratio: float = 0.4):
    """
    Per-layer top-K (by ratio) drifted channels.

    Returns:
        dict[layer_name] -> bool mask of shape (C,)
    """
    masks = {}
    for name, scores in channel_scores.items():
        c = scores.size
        k = max(1, int(round(c * top_k_ratio)))
        top_idx = np.argsort(scores)[-k:]
        mask = np.zeros(c, dtype=bool)
        mask[top_idx] = True
        masks[name] = mask
    return masks


def random_channel_mask(channel_scores: dict, top_k_ratio: float = 0.4,
                         seed: int = 0):
    """Random mask matching the size of select_drifted_channels — for control."""
    rng = np.random.default_rng(seed)
    masks = {}
    for name, scores in channel_scores.items():
        c = scores.size
        k = max(1, int(round(c * top_k_ratio)))
        idx = rng.choice(c, size=k, replace=False)
        mask = np.zeros(c, dtype=bool)
        mask[idx] = True
        masks[name] = mask
    return masks
