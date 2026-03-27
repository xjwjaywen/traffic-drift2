"""
TTA-TC: Test-Time Adaptation Engine.

Implements the 7-step class-aware selective adaptation pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from .drift_detector import DriftDetector
from .pbrs_buffer import PBRSBuffer
from .anti_forgetting import FisherRegularizer, StochasticRestorer
from .prototype_loss import PrototypeLoss


class TTAEngine:
    """
    Full TTA-TC adaptation engine.

    7-step pipeline per batch:
        1. OOD filtering (energy score)
        2. Entropy filtering (SAR-style)
        3. PBRS buffering
        4. Source Prototype Anchoring (SPA) loss
        5. Selective update (drift-aware LR)
        6. Anti-forgetting (Fisher + stochastic restore)
        7. EMA teacher update
    """

    def __init__(self, model, cfg: dict, prototypes: torch.Tensor = None):
        """
        Args:
            model: TTATCModel instance (source-trained)
            cfg: configuration dict
            prototypes: (C, hidden_dim) source class prototypes for SPA loss.
                        If None, falls back to a no-op loss.
        """
        self.cfg = cfg
        self.device = next(model.parameters()).device
        num_classes = cfg["num_classes"]

        # Source model (frozen copy for reference)
        self.source_model = copy.deepcopy(model)
        self.source_model.eval()
        for p in self.source_model.parameters():
            p.requires_grad_(False)

        # Adaptation model
        self.model = model
        self.model.eval()

        # Freeze classification head
        for p in self.model.cls_head.parameters():
            p.requires_grad_(False)

        # EMA teacher
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.ema_momentum = cfg.get("ema_momentum", 0.999)

        # Source Prototype Anchoring (SPA) loss
        if prototypes is not None:
            self.proto_loss = PrototypeLoss(
                prototypes=prototypes,
                temperature=cfg.get("spa_temperature", 0.07),
            ).to(self.device)
        else:
            self.proto_loss = None

        # Drift detector
        self.drift_detector = DriftDetector(
            num_classes=num_classes,
            entropy_threshold=cfg.get("entropy_threshold", 0.5),
            abrupt_threshold=cfg.get("abrupt_threshold", 1.0),
        )

        # PBRS buffer
        buffer_size = cfg.get("buffer_size", num_classes * 10)
        self.buffer = PBRSBuffer(buffer_size, num_classes)

        # Anti-forgetting
        self.fisher_reg = FisherRegularizer(model, cfg.get("fisher_alpha", 2000.0))
        self.stochastic_restorer = StochasticRestorer(model, cfg.get("restore_prob", 0.01))

        # Optimizer for adaptation
        adapt_mode = cfg.get("adapt_mode", "encoder")
        adapt_params = self.model.get_adaptation_params(adapt_mode)
        self.optimizer = torch.optim.Adam(
            adapt_params,
            lr=cfg.get("adapt_lr", 1e-4),
            weight_decay=0,
        )

        # Thresholds
        self.energy_threshold = cfg.get("energy_threshold", -5.0)
        self.entropy_threshold_ratio = cfg.get("entropy_filter_ratio", 0.4)
        self.max_entropy = self.entropy_threshold_ratio * math.log(num_classes)

        # Drift-aware LR multiplier
        self.drift_lr_gamma = cfg.get("drift_lr_gamma", 3.0)
        self.adapt_steps = cfg.get("adapt_steps", 1)
        self.abrupt_adapt_steps = cfg.get("abrupt_adapt_steps", 10)

        # Counters
        self.step_count = 0

    def set_fisher(self, dataloader):
        """Pre-compute Fisher information from source data using SPA loss."""
        proto_loss = self.proto_loss  # capture for closure

        def _loss_fn(model, ppi, flow_stats):
            if proto_loss is None:
                return torch.tensor(0.0, device=ppi.device, requires_grad=True), {}
            logits, features = model(ppi, flow_stats, return_repr=True)
            pseudo_labels = logits.detach().argmax(dim=1)
            loss = proto_loss.compute_loss(features, pseudo_labels)
            return loss, {}

        self.fisher_reg.compute_fisher(self.source_model, dataloader, _loss_fn)

    def set_baseline_entropy(self, baseline: "np.ndarray"):
        """Set drift detector baseline from training data."""
        self.drift_detector.set_baseline(baseline)

    @torch.no_grad()
    def _filter_ood(self, logits: torch.Tensor) -> torch.Tensor:
        """Step 1: OOD filtering via energy score."""
        energy = -torch.logsumexp(logits, dim=1)
        return energy < self.energy_threshold

    @torch.no_grad()
    def _filter_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Step 2: Entropy filtering (SAR-style)."""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        return entropy < self.max_entropy

    def adapt_batch(self, ppi: torch.Tensor, flow_stats: torch.Tensor = None):
        """
        Perform one TTA step on a batch.

        Args:
            ppi: (B, 3, 30)
            flow_stats: (B, D) or None
        Returns:
            logits: (B, C) adapted classification logits
            info: dict with adaptation stats
        """
        self.model.eval()
        info = {}

        # Get initial predictions from teacher
        with torch.no_grad():
            teacher_logits = self.teacher(ppi, flow_stats)
            preds = teacher_logits.argmax(dim=1)

        # Step 1: OOD filtering
        ood_mask = self._filter_ood(teacher_logits)
        # Step 2: Entropy filtering
        ent_mask = self._filter_entropy(teacher_logits)
        valid_mask = ood_mask & ent_mask

        info["total_samples"] = ppi.size(0)
        info["valid_samples"] = valid_mask.sum().item()

        if valid_mask.sum() < 2:
            # Not enough valid samples, skip adaptation
            logits = self.model(ppi, flow_stats)
            info["adapted"] = False
            return logits, info

        # Step 3: Add to PBRS buffer
        self.buffer.add(
            ppi[valid_mask],
            flow_stats[valid_mask] if flow_stats is not None else None,
            preds[valid_mask]
        )

        # Sample from buffer
        buf_ppi, buf_fs, buf_weights = self.buffer.sample(
            self.cfg.get("adapt_batch_size", 64), self.device
        )

        if buf_ppi is None:
            logits = self.model(ppi, flow_stats)
            info["adapted"] = False
            return logits, info

        # Step 4+5: Drift detection and selective adaptation
        drifted_classes, abrupt_classes = self.drift_detector.update(teacher_logits)
        info["drifted_classes"] = len(drifted_classes)
        info["abrupt_classes"] = len(abrupt_classes)

        # Drift gate: skip adaptation if no class shows significant drift
        min_drifted = self.cfg.get("min_drifted_classes", 1)
        if len(drifted_classes) < min_drifted and len(abrupt_classes) == 0:
            with torch.no_grad():
                logits = self.model(ppi, flow_stats)
            info["adapted"] = False
            info["skipped_reason"] = "no_drift"
            return logits, info

        # Determine adaptation intensity
        steps = self.adapt_steps
        if abrupt_classes:
            steps = self.abrupt_adapt_steps

        # Perform adaptation steps
        self.model.train()
        for _ in range(steps):
            self.optimizer.zero_grad()

            # Step 4: SPA loss — pull encoder features toward source class prototypes
            buf_logits, buf_features = self.model(buf_ppi, buf_fs, return_repr=True)
            buf_pseudo = buf_logits.detach().argmax(dim=1)
            if self.proto_loss is not None:
                spa_loss = self.proto_loss.compute_loss(buf_features, buf_pseudo)
            else:
                spa_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Step 6: Fisher regularization
            fisher_loss = self.fisher_reg.penalty(self.model)
            total_loss = spa_loss + fisher_loss

            total_loss.backward()
            self.optimizer.step()

        info["spa_loss"] = spa_loss.item()
        info["fisher_loss"] = fisher_loss.item()
        info["adapted"] = True

        # Step 6 (continued): Stochastic restoration
        self.stochastic_restorer.restore(self.model)

        # Step 7: EMA teacher update
        self._update_teacher()

        self.model.eval()
        self.step_count += 1

        # Final prediction with adapted model
        with torch.no_grad():
            logits = self.model(ppi, flow_stats)

        return logits, info

    @torch.no_grad()
    def _update_teacher(self):
        """EMA update of teacher model."""
        m = self.ema_momentum
        for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
            t_param.data.mul_(m).add_(s_param.data, alpha=1 - m)

    def reset(self):
        """Reset model to source weights."""
        self.model.load_state_dict(self.source_model.state_dict())
        self.teacher.load_state_dict(self.source_model.state_dict())
        self.buffer = PBRSBuffer(self.buffer.buffer_size, self.cfg["num_classes"])
        self.drift_detector = DriftDetector(
            self.cfg["num_classes"],
            self.cfg.get("entropy_threshold", 0.5),
        )
        self.step_count = 0
