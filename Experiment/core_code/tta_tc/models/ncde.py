"""Neural CDE baseline for packet-level PPI paths."""
import torch
import torch.nn as nn

try:
    import torchcde
except ImportError as exc:  # pragma: no cover - exercised on machines without deps
    torchcde = None
    _TORCHCDE_IMPORT_ERROR = exc
else:
    _TORCHCDE_IMPORT_ERROR = None


class CDEFunc(nn.Module):
    """Vector field f(t, z) mapping hidden states to CDE dynamics."""

    def __init__(
        self,
        hidden_dim: int,
        input_channels: int,
        vector_field_hidden_dim: int = None,
        vector_field_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        width = vector_field_hidden_dim or hidden_dim
        layers = [nn.Linear(hidden_dim, width), nn.Tanh()]
        for _ in range(max(vector_field_layers - 1, 0)):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, hidden_dim * input_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        out = self.net(z)
        return out.view(*z.shape[:-1], self.hidden_dim, self.input_channels)


class NeuralCDEClassifier(nn.Module):
    """
    Minimal NCDE classifier for CESNET PPI tensors.

    Input format follows the existing codebase: (B, 3, 30), where channels are
    packet size, direction, and inter-arrival time. The model treats each flow as
    a 3D path over 30 packet steps and uses torchcde linear interpolation.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 102,
        interpolation: str = "linear",
        solver: str = "rk4",
        solver_step_size: float = 1.0,
        normalize_input: bool = False,
        add_time_channel: bool = False,
        vector_field_hidden_dim: int = None,
        vector_field_layers: int = 2,
    ):
        super().__init__()
        if torchcde is None:
            raise ImportError(
                "torchcde is required for NeuralCDEClassifier. "
                "Install it with `pip install torchcde`."
            ) from _TORCHCDE_IMPORT_ERROR

        if interpolation != "linear":
            raise ValueError(f"H0 NCDE baseline only supports linear interpolation, got {interpolation!r}")

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.interpolation = interpolation
        self.solver = solver
        self.solver_step_size = solver_step_size
        self.normalize_input = normalize_input
        self.add_time_channel = add_time_channel
        self.control_channels = input_channels + int(add_time_channel)

        self.register_buffer("input_mean", torch.zeros(1, 1, input_channels))
        self.register_buffer("input_std", torch.ones(1, 1, input_channels))

        self.initial = nn.Linear(self.control_channels, hidden_dim)
        self.func = CDEFunc(
            hidden_dim=hidden_dim,
            input_channels=self.control_channels,
            vector_field_hidden_dim=vector_field_hidden_dim,
            vector_field_layers=vector_field_layers,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set channel-wise input normalization statistics."""
        mean = mean.detach().view(1, 1, self.input_channels)
        std = std.detach().view(1, 1, self.input_channels).clamp_min(1e-6)
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_std.copy_(std.to(self.input_std.device))

    def forward(self, ppi: torch.Tensor):
        """
        Args:
            ppi: Tensor of shape (B, 3, 30).
        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        if ppi.ndim != 3:
            raise ValueError(f"Expected PPI tensor with shape (B, 3, T), got {tuple(ppi.shape)}")
        if ppi.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} PPI channels, got {ppi.size(1)}"
            )

        path = ppi.transpose(1, 2).contiguous()  # (B, T, 3)
        if self.normalize_input:
            path = (path - self.input_mean) / self.input_std
        if self.add_time_channel:
            time = torch.linspace(0, 1, path.size(1), device=path.device, dtype=path.dtype)
            time = time.view(1, -1, 1).expand(path.size(0), -1, -1)
            path = torch.cat([time, path], dim=-1)

        coeffs = torchcde.linear_interpolation_coeffs(path)
        control = torchcde.LinearInterpolation(coeffs)

        z0 = self.initial(control.evaluate(control.interval[0]))
        solver_options = {}
        if self.solver in {"euler", "midpoint", "rk4"} and self.solver_step_size is not None:
            solver_options["step_size"] = self.solver_step_size

        z_t = torchcde.cdeint(
            X=control,
            z0=z0,
            func=self.func,
            t=control.interval,
            method=self.solver,
            options=solver_options,
        )
        z_final = z_t[:, -1]
        return self.classifier(z_final)
