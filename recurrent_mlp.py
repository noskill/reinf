import torch
import torch.nn as nn
from recurrent_cache import CacheModuleMixin


class RecurrentMLP(nn.Module, CacheModuleMixin):
    def __init__(self, mlp):
        super().__init__()
        
        # The MLP takes both the previous state and current input
        self.mlp = mlp
        self._prev = None

    def get_cache_state(self):
        if self._prev is None:
            return None
        return self._prev.detach().clone()

    def set_cache_state(self, state):
        self._prev = None if state is None else state.detach().clone()

    def index_cache_state(self, state, batch_indices: torch.Tensor):
        if state is None:
            return None
        return state[batch_indices].detach().clone()

    def forward(self, e_seq, reset_mask):
        # e_seq shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = e_seq.size()

        outputs = []
        use_cache = reset_mask is not None

        if use_cache:
            if self._prev is None or self._prev.shape != e_seq[:, 0, :].shape:
                self._prev = torch.zeros_like(e_seq[:, 0, :])
            self.reset_cache(reset_mask)
            f_t = self._prev
        else:
            f_t = torch.zeros_like(e_seq[:, 0, :])

        # Loop through each time step in the sequence
        for t in range(seq_len):
            e_t = e_seq[:, t, :]          # Current external input
            x_t = torch.cat([f_t, e_t], dim=1)  # Combine f_t-1 and e_t
            f_t = self.mlp(x_t)           # Compute next state f_t
            outputs.append(f_t)
            if t % 10 == 0:
                f_t = f_t.detach()
        if use_cache:
            self._prev = f_t.detach()
        return torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, output_dim)

    def reset_cache(self, reset_mask: torch.Tensor):
        if reset_mask is None:
            return
        self._prev[reset_mask] = 0.0

    def clear_cache(self):
        self._prev = None
