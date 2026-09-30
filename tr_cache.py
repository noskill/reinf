"""
Resettable, append-only cache for transformer decoding in vectorized RL.

This module provides a drop-in replacement for Hugging Face style caches that:
- Subclass the HF base types (Cache and CacheLayerMixin) for compatibility.
- Support explicit per-row resets so each batch element can reset or continue
  independently (common in multi-env RL).

It is designed to be used with attention modules that call:
    past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
cache_kwargs is accepted for API compatibility but does not control KV writes.
Absolute positions are handled by the attention module, not by this cache.

Notes
- We keep per-layer, per-row histories internally and materialize a padded
  batch tensor on each update. This avoids sequence leakage across rows when
  some envs reset while others continue.
- Attention masks are not applied here; if your model needs to strictly mask
  padded positions, build an attention_mask in the model and pass it down.
"""

from typing import Dict, List, Optional, Tuple

import torch

try:
    # Prefer importing the HF interfaces for compatibility
    from transformers.cache_utils import Cache, CacheLayerMixin  # type: ignore
except Exception:  # pragma: no cover - fallback typing if transformers is unavailable
    class Cache:  # minimal fallback to allow static analysis
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            raise NotImplementedError

    class CacheLayerMixin:  # minimal fallback mixin
        pass


def _right_pad_time(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a per-row KV tensor [H, T, D] with zeros on the time dim to target_len."""
    H, T, D = t.shape
    if T == target_len:
        return t
    if T > target_len:
        return t[:, :target_len, :]
    pad = t.new_zeros((H, target_len - T, D))
    return torch.cat([t, pad], dim=1)


class LayerCache(CacheLayerMixin):
    """Per-layer append-only cache with selective resets.

    Internally stores a list of row tensors for keys/values with shape [H, T, D].
    Updates append new tokens; reset_rows explicitly clears selected histories.
    A sliding window only removes the oldest tokens.
    """

    def __init__(self, detach_every: int = 0) -> None:
        super().__init__()
        assert detach_every >= 0
        self._k_rows: List[Optional[torch.Tensor]] = []
        self._v_rows: List[Optional[torch.Tensor]] = []
        self.detach_every = detach_every
        self._steps_since_detach: List[int] = []

    def __len__(self) -> int:
        return max(len(self._k_rows), len(self._v_rows))

    @staticmethod
    def _share_row(row: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if row is None:
            return None
        # KV prefixes are immutable: updates allocate a new row with torch.cat
        # and rebind the list entry, so rollout branches can safely share storage.
        return row.detach()

    def clone(self):
        result = LayerCache(detach_every=self.detach_every)
        result._k_rows = [self._share_row(row) for row in self._k_rows]
        result._v_rows = [self._share_row(row) for row in self._v_rows]
        result._steps_since_detach = self._steps_since_detach.copy()
        return result

    def index_batch(self, batch_indices: torch.Tensor):
        if not isinstance(batch_indices, torch.Tensor):
            raise TypeError("batch_indices must be a tensor")
        if batch_indices.dim() != 1:
            raise ValueError(f"batch_indices must be 1D, got {tuple(batch_indices.shape)}")
        if len(self._k_rows) != len(self._v_rows):
            raise RuntimeError("key and value cache row counts must match")

        indices = batch_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        if indices and (min(indices) < 0 or max(indices) >= len(self._k_rows)):
            raise IndexError(f"batch index outside cache size {len(self._k_rows)}")

        result = LayerCache(detach_every=self.detach_every)
        result._k_rows = [self._share_row(self._k_rows[index]) for index in indices]
        result._v_rows = [self._share_row(self._v_rows[index]) for index in indices]
        result._steps_since_detach = [self._steps_since_detach[index] for index in indices]
        return result

    def _ensure_rows(self, batch: int) -> None:
        if len(self._k_rows) < batch:
            self._k_rows.extend([None] * (batch - len(self._k_rows)))
        if len(self._v_rows) < batch:
            self._v_rows.extend([None] * (batch - len(self._v_rows)))
        if len(self._steps_since_detach) < batch:
            self._steps_since_detach.extend([0] * (batch - len(self._steps_since_detach)))

    def reset_rows(self, reset_mask: torch.Tensor) -> None:
        """Reset selected rows (set to None so next write starts fresh).

        Args:
            reset_mask: Bool/byte/long tensor of shape [N] where non-zero means reset.
        """
        if not isinstance(reset_mask, torch.Tensor):
            reset_mask = torch.as_tensor(reset_mask)
        reset_mask = reset_mask.to(torch.bool).view(-1)
        self._ensure_rows(reset_mask.numel())
        for i, flag in enumerate(reset_mask.tolist()):
            if flag:
                self._k_rows[i] = None
                self._v_rows[i] = None
                self._steps_since_detach[i] = 0

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        for i in range(len(self)):
            if self._k_rows[i] is not None:
                self._k_rows[i] = self._k_rows[i].to(device=device, dtype=dtype)
            if self._v_rows[i] is not None:
                self._v_rows[i] = self._v_rows[i].to(device=device, dtype=dtype)
        return self

    def update(
        self,
        key_states: torch.Tensor,   # [B, H, T_new, D]
        value_states: torch.Tensor, # [B, H, T_new, D]
        window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append new K/V to each row, then keep the newest window_size tokens.

        Histories are cleared only by reset_rows. Returned rows are padded to
        a common batch length, with padding identified by the attention mask.
        """
        assert key_states.dim() == 4 and value_states.dim() == 4, "Expected [B, H, T, D] tensors"
        B, H, T_new, D = key_states.shape
        assert self.detach_every == 0 or T_new == 1, "Periodic KV detach requires one-step cache updates"
        self._ensure_rows(B)

        # Update rows independently
        max_len = 0
        for b in range(B):
            k_new = key_states[b]
            v_new = value_states[b]

            k_prev = self._k_rows[b]
            v_prev = self._v_rows[b]

            if k_prev is None:
                k_row = k_new
                v_row = v_new
                self._steps_since_detach[b] = 0
            else:
                k_row = k_prev
                v_row = v_prev
                if self.detach_every > 0 and self._steps_since_detach[b] == self.detach_every:
                    # Cut history before the next block, leaving new K/V differentiable.
                    # This limits BPTT, not total activation memory when all losses
                    # are retained for one backward at the end of the sequence.
                    k_row = k_row.detach()
                    v_row = v_row.detach()
                    self._steps_since_detach[b] = 0
                # Append new tokens
                k_row = torch.cat([k_row, k_new], dim=1)
                v_row = torch.cat([v_row, v_new], dim=1)

            if window_size is not None and window_size > 0 and k_row.shape[1] > window_size:
                k_row = k_row[:, -window_size:, :]
                v_row = v_row[:, -window_size:, :]
            self._k_rows[b] = k_row
            self._v_rows[b] = v_row
            self._steps_since_detach[b] += T_new
            max_len = max(max_len, k_row.shape[1])

        # Build padded batch tensors [B, H, T_max, D]
        k_out = []
        v_out = []
        lengths = []
        for b in range(B):
            k_row = self._k_rows[b]
            v_row = self._v_rows[b]
            if k_row is None:
                k_row = key_states.new_zeros((H, 0, D))
                v_row = value_states.new_zeros((H, 0, D))
            k_padded = _right_pad_time(k_row, max_len)
            v_padded = _right_pad_time(v_row, max_len)
            k_out.append(k_padded)
            v_out.append(v_padded)
            lengths.append(k_row.shape[1])

        K = torch.stack(k_out, dim=0)
        V = torch.stack(v_out, dim=0)
        # Build attention padding mask: 0 for valid, -inf for padded
        L = torch.tensor(lengths, device=K.device, dtype=torch.long)
        # Shape [B, 1, 1, T_max]
        mask = K.new_zeros((B, 1, 1, max_len))
        if max_len > 0:
            arange = torch.arange(max_len, device=K.device).view(1, 1, 1, -1)
            valid = arange < L.view(B, 1, 1, 1)
            neg_inf = torch.finfo(mask.dtype).min
            mask = torch.where(valid, mask, mask.new_full(mask.shape, neg_inf))

        return K, V, mask

    def lazy_initialization(self, key_states: torch.Tensor):
       # No preallocation; rows grow dynamically.
       return None

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        # kv_length = current max row length or fall back to q_length
        max_len = 0
        for k in self._k_rows:
            if k is not None:
                max_len = max(max_len, int(k.shape[1]))
        if max_len == 0:
            max_len = int(cache_position.shape[0])
        return max_len, 0  # (kv_length, kv_offset)

    def get_seq_length(self) -> int:
        max_len = 0
        for k in self._k_rows:
            if k is not None:
                max_len = max(max_len, int(k.shape[1]))
        return max_len

    def get_max_cache_shape(self) -> int:
        # Dynamic cache: no fixed maximum
        return -1


class PositionBasedDynamicCache(Cache):
    """A Cache that appends updates to per-layer LayerCache instances.

    Exposes the HF-compatible `update(key, value, layer_idx, cache_kwargs)` API.
    Also provides `reset(mask)` to clear selected rows across all layers.
    """

    def __init__(self, detach_every: int = 0) -> None:
        super().__init__(layer_class_to_replicate=LayerCache)
        assert detach_every >= 0
        self._layers: Dict[int, LayerCache] = {}
        self.detach_every = detach_every

    def _get_layer(self, layer_idx: int) -> LayerCache:
        if layer_idx not in self._layers:
            self._layers[layer_idx] = LayerCache(detach_every=self.detach_every)
        return self._layers[layer_idx]

    def _new_empty_like(self):
        if hasattr(self, "window_size"):
            return self.__class__(self.window_size, detach_every=self.detach_every)
        return self.__class__(detach_every=self.detach_every)

    def clone(self):
        result = self._new_empty_like()
        result._layers = {
            layer_idx: layer.clone()
            for layer_idx, layer in self._layers.items()
        }
        return result

    def index_batch(self, batch_indices: torch.Tensor):
        result = self._new_empty_like()
        result._layers = {
            layer_idx: layer.index_batch(batch_indices)
            for layer_idx, layer in self._layers.items()
        }
        return result

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        for layer in self._layers.values():
            layer.to(device=device, dtype=dtype)
        return self

    def reset(self, reset_mask: torch.Tensor) -> None:
        for layer in self._layers.values():
            layer.reset_rows(reset_mask)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self._get_layer(layer_idx)
        return layer.update(key_states, value_states)


class WindowedPositionBasedDynamicCache(PositionBasedDynamicCache):
    """Append-only cache with a fixed-size sliding window."""

    def __init__(self, window_size: int, detach_every: int = 0) -> None:
        super().__init__(detach_every=detach_every)
        if window_size is None or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = int(window_size)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self._get_layer(layer_idx)
        return layer.update(
            key_states,
            value_states,
            window_size=self.window_size,
        )


__all__ = [
    "LayerCache",
    "PositionBasedDynamicCache",
    "WindowedPositionBasedDynamicCache",
]
