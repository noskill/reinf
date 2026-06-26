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
        input_dim: int
    ):
        super().__init__()
        self.layers = make_probe_head(int(input_dim), 1, 512, 6).to(device)

    def forward(self, state, **kwargs):
        return self.layers(state)
