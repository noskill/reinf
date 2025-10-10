import torch
import torch.nn as nn


def _ensure_sequence(value, fallback):
    if value is None:
        return list(fallback)
    if isinstance(value, int):
        return [value]
    return list(value)


def _build_activation_factory(activation):
    if isinstance(activation, type):
        return activation
    if callable(activation):
        def factory():
            module = activation()
            if not isinstance(module, nn.Module):
                raise TypeError("activation callable must return an nn.Module instance")
            return module
        return factory
    raise TypeError("activation must be an nn.Module subclass or a zero-arg factory")


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation_factory, layer_norm=False, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else None
        self.activation = activation_factory()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else None
        self.use_residual = in_dim == out_dim

    def forward(self, x):
        y = self.linear(x)
        if self.norm is not None:
            y = self.norm(y)
        y = self.activation(y)
        if self.dropout is not None:
            y = self.dropout(y)
        if self.use_residual:
            y = y + x
        return y


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        n_obs,
        n_action,
        hidden_dim=128,
        hidden_dims=None,
        activation=nn.ReLU,
        layer_norm=False,
        dropout=0.0,
    ):
        super().__init__()

        hidden_dims = _ensure_sequence(hidden_dims, [hidden_dim, hidden_dim])
        self.hidden_dims = hidden_dims
        activation_factory = _build_activation_factory(activation)

        blocks = []
        prev_dim = n_obs
        for dim in hidden_dims:
            blocks.append(_ResidualBlock(prev_dim, dim, activation_factory, layer_norm, dropout))
            prev_dim = dim

        self.backbone = nn.ModuleList(blocks)
        self.input_skip = nn.Linear(n_obs, prev_dim) if blocks else None
        self.output_layer = nn.Linear(prev_dim, n_action)
        self.first_layer = blocks[0].linear if blocks else self.output_layer

    def forward(self, x):
        x = x.to(self.output_layer.weight)
        residual = x
        for block in self.backbone:
            x = block(x)
        if self.input_skip is not None:
            x = x + self.input_skip(residual)
        return self.output_layer(x)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        n_obs,
        hidden_dim=128,
        hidden_dims=None,
        activation=nn.ReLU,
        layer_norm=False,
        dropout=0.0,
    ):
        super().__init__()

        hidden_dims = _ensure_sequence(hidden_dims, [hidden_dim, hidden_dim])
        self.hidden_dims = hidden_dims
        activation_factory = _build_activation_factory(activation)

        blocks = []
        prev_dim = n_obs
        for dim in hidden_dims:
            blocks.append(_ResidualBlock(prev_dim, dim, activation_factory, layer_norm, dropout))
            prev_dim = dim

        self.backbone = nn.ModuleList(blocks)
        self.input_skip = nn.Linear(n_obs, prev_dim) if blocks else None
        self.output_layer = nn.Linear(prev_dim, 1)
        self.first_layer = blocks[0].linear if blocks else self.output_layer

    def forward(self, x):
        x = x.to(self.output_layer.weight)
        residual = x
        for block in self.backbone:
            x = block(x)
        if self.input_skip is not None:
            x = x + self.input_skip(residual)
        return self.output_layer(x)


class SkillDiscriminator(nn.Module):
    """Maps observations to skills for DIAYN."""

    def __init__(
        self,
        state_dim,
        skill_dim,
        hidden_dims=None,
        continuous=True,
        activation=nn.SiLU,
        layer_norm=False,
        dropout=0.0,
    ):
        super().__init__()

        hidden_dims = _ensure_sequence(hidden_dims, [512, 512, 256])
        self.continuous = continuous
        self.skill_dim = skill_dim
        activation_factory = _build_activation_factory(activation)
        dropout = float(dropout) if dropout is not None else 0.0

        blocks = []
        prev_dim = state_dim
        for dim in hidden_dims:
            blocks.append(_ResidualBlock(prev_dim, dim, activation_factory, layer_norm, dropout))
            prev_dim = dim

        self.network = nn.ModuleList(blocks)
        self.input_skip = nn.Linear(state_dim, prev_dim) if blocks else None
        self.output_layer = nn.Linear(prev_dim, skill_dim)
        self.first_layer = blocks[0].linear if blocks else self.output_layer

    def forward(self, state, softmax=True):
        features = state
        residual = features
        for block in self.network:
            features = block(features)
        if self.input_skip is not None:
            features = features + self.input_skip(residual)
        output = self.output_layer(features)
        if not self.continuous and softmax:
            output = output.softmax(dim=1)
        return output

    def compute_mi_loss(self, states, skills):
        predictions = self.forward(states, softmax=False)
        if self.continuous:
            return nn.functional.mse_loss(predictions, skills)
        return nn.functional.cross_entropy(predictions, skills)

    def predict_skill(self, state):
        with torch.no_grad():
            output = self.forward(state)
            if not self.continuous:
                output = torch.argmax(output, dim=-1)
        return output


# Backwards compatibility aliases
Value = ValueNetwork

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "Value",
    "SkillDiscriminator",
]
