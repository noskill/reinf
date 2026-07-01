import torch
import torch.nn as nn
from typing import Sequence, Tuple

from utils import make_probe_head


class WMActionHeadPolicy(nn.Module):
    """Policy head that maps precomputed WM state to action logits."""

    def __init__(
        self,
        num_actions,
        device: torch.device,
        input_dim: int,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.layers = make_probe_head(int(input_dim), self.num_actions, 512, 6).to(device)

    def forward(self, state, **kwargs):
        return self.layers(state)


class WMValueHeadPolicy(nn.Module):
    """Policy head that maps precomputed WM state to action logits."""

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.layers = make_probe_head(input_dim, output_dim, 512, 6).to(device)

    def forward(self, state, **kwargs):
        return self.layers(state)


class ActionSwitchPolicy(nn.Module):
    def __init__(self, backbone, hidden_dim, goal_dim):
        super().__init__()
        self.backbone = backbone
        self.goal_dim = goal_dim
        self.projection = make_probe_head(hidden_dim, self.goal_dim * 2 + 1, 512, 2)

    def forward(self, state, reset_mask=None, key_padding_mask=None):
        squeeze_time = state.dim() == 2
        if squeeze_time:
            state = state.unsqueeze(1)
        x = self.backbone(state, key_padding_mask=key_padding_mask, reset_mask=reset_mask)
        params = self.projection(x)
        if squeeze_time:
            params = params.squeeze(1)
        return {
            "goal": params[..., : self.goal_dim * 2],
            "switch": params[..., self.goal_dim * 2 :],
        }
