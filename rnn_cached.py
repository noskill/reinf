import torch
import torch.nn as nn
from recurrent_cache import CacheModuleMixin
from dataclasses import dataclass


class CachedRNN(torch.nn.Module, CacheModuleMixin):

    def __init__(self, config):
        super().__init__()
        self.backbone = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
        )
        self.backbone.flatten_parameters()
        self.attention_window = config.attention_window
        self._state_h: Optional[torch.Tensor] = None

    def get_cache_state(self):
        if self._state_h is None:
            return None
        return self._state_h.detach().clone()

    def set_cache_state(self, state):
        self._state_h = None if state is None else state.detach().clone()

    def index_cache_state(self, state, batch_indices: torch.Tensor):
        if state is None:
            return None
        return state[:, batch_indices, :].detach().clone()

    def prime_cache(self, x):
        self.backbone.flatten_parameters()
        if self.attention_window is not None and self.attention_window > 0:
            h, h_n = self.window_iterate(x, None)
        else:
            h, h_n = self.backbone(x)
        self._state_h = h_n.detach()
        return h

    def forward(self, x, key_padding_mask, reset_mask):
        self.backbone.flatten_parameters()
        using_cache = reset_mask is not None
        if using_cache:
            assert x.shape[1] == 1, f"CachedRNN internal cache expects online T=1 calls, got sequence length {x.shape[1]}"
            if self._state_h is None or self._state_h.shape[1] != reset_mask.numel():
                self.init_cache(reset_mask.numel())
            else:
                self.reset_cache(reset_mask)
            h0 = self._state_h
        else:
            h0 = None
        # no attention and we are not in episode rollout
        if self.attention_window is not None and self.attention_window > 0 and reset_mask is None:
            h, h_n = self.window_iterate(x, h0)
        else:
            # no attention window
            h, h_n = self.backbone(x, h0)
        if using_cache:
            self._state_h = h_n.detach()
        return h

    def reset_cache(self, reset_mask: torch.Tensor):
        if self._state_h is None:
            return
        if reset_mask is None:
            return
        reset_mask = reset_mask.to(torch.bool).view(-1)
        if reset_mask.numel() != self._state_h.shape[1]:
            raise ValueError("reset_mask size must match num_envs for cache reset")
        self._state_h[:, reset_mask, :] = 0.0

    def clear_cache(self):
        self._state_h = None

    def init_cache(self, size: int):
        self._state_h = torch.zeros(
            (self.backbone.num_layers, size, self.backbone.hidden_size),
            device=self.device,
            dtype=self.dtype,)

    @property
    def dtype(self):
        return next(next(self.backbone.modules()).parameters()).dtype

    @property
    def device(self):
        return next(next(self.backbone.modules()).parameters()).device

    def window_iterate(self, x, h0):
        all_h_chunks = []
        h_current = h0
        window_size = self.attention_window
        for i in range(0, x.size(1), window_size):
            # Slice the current sequence window
            x_chunk = x[:, i : i + window_size, :]

            # Forward pass: current hidden state flows in
            # h_chunk: (batch, window_size, hidden_size)
            # h_current: (num_layers, batch, hidden_size)
            h_chunk, h_current = self.backbone(x_chunk, h_current)

            # Store output chunk for the final 'h' result
            all_h_chunks.append(h_chunk)

            # --- CRITICAL STEP FOR BACKPROP RESTRICTION ---
            # This detaches the hidden state from the graph.
            # Gradients will NOT flow from the next window back into this one.
            h_current = h_current.detach()
            # ----------------------------------------------

        # 3. Reconstruct the full 'h' and 'h_n'
        # h will have shape (batch, 2048, hidden_size)
        h = torch.cat(all_h_chunks, dim=1)
        # h_n is simply the h_current from the last iteration
        h_n = h_current
        return h, h_n


@dataclass
class RNNConfig:
    input_size: int
    hidden_size: int
    num_hidden_layers: int
    attention_window: int = -1
