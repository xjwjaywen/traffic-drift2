"""Anti-forgetting mechanisms for continual TTA."""
import torch
import torch.nn as nn
import copy
from typing import Dict


class FisherRegularizer:
    """
    Fisher Information regularization to prevent catastrophic forgetting.

    Penalizes deviation from source model parameters, weighted by Fisher importance.
    Based on EATA (Niu et al., ICML 2022).
    """

    def __init__(self, model: nn.Module, fisher_alpha: float = 2000.0):
        self.fisher_alpha = fisher_alpha
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.source_params: Dict[str, torch.Tensor] = {}

        # Store source parameters
        for name, param in model.named_parameters():
            self.source_params[name] = param.data.clone()

    def compute_fisher(self, model: nn.Module, dataloader, ssl_loss_fn, num_batches: int = 50):
        """
        Estimate Fisher Information from source data using SSL loss gradients.

        Args:
            model: the source model
            dataloader: source training data loader
            ssl_loss_fn: CombinedSSLLoss instance
            num_batches: number of batches to use
        """
        model.eval()
        fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        count = 0
        for batch in dataloader:
            if count >= num_batches:
                break
            ppi = batch["ppi"].to(next(model.parameters()).device)
            flow_stats = batch.get("flow_stats")
            if flow_stats is not None:
                flow_stats = flow_stats.to(ppi.device)

            model.zero_grad()
            loss, _ = ssl_loss_fn(model, ppi, flow_stats)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
            count += 1

        # Average
        for name in fisher:
            fisher[name] /= max(count, 1)

        self.fisher_dict = fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute Fisher regularization penalty."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                loss += (self.fisher_dict[name] * (param - self.source_params[name]).pow(2)).sum()
        return self.fisher_alpha * loss


class StochasticRestorer:
    """
    Stochastic weight restoration (CoTTA-style).

    With probability p, restore each parameter to its source value.
    Prevents long-term drift from the source model.
    """

    def __init__(self, model: nn.Module, restore_prob: float = 0.01):
        self.restore_prob = restore_prob
        self.source_state = copy.deepcopy(model.state_dict())

    def restore(self, model: nn.Module):
        """Stochastically restore parameters to source values."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.source_state:
                    mask = torch.bernoulli(
                        torch.full_like(param, self.restore_prob)
                    ).bool()
                    param.data[mask] = self.source_state[name].to(param.device)[mask]
