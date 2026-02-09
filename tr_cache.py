"""
Resettable, position-aware cache for transformer decoding in vectorized RL.

This module provides a drop-in replacement for Hugging Face style caches that:
- Subclass the HF base types (Cache and CacheLayerMixin) for compatibility.
- Support per-row cache_position indexing so each batch element can reset or
  continue independently (common in multi-env RL).

It is designed to be used with attention modules that call:
    past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
where cache_kwargs may contain {"cache_position": LongTensor[B] or [B, 1]}.

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


def _to_scalar_long(x: torch.Tensor) -> int:
    """Convert a 0D/1D tensor to a Python int safely."""
    if x.numel() == 0:
        return 0
    return int(x.reshape(-1)[0].item())


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
    """Per-layer cache that supports per-row positions and selective resets.

    Internally stores a list of row tensors for keys/values with shape [H, T, D].
    On update, the provided cache_position[b] indicates where the new tokens
    should be written; rows can be truncated (reset) or appended independently.
    """

    def __init__(self) -> None:
        super().__init__()
        self._k_rows: List[Optional[torch.Tensor]] = []
        self._v_rows: List[Optional[torch.Tensor]] = []

    def __len__(self) -> int:
        return max(len(self._k_rows), len(self._v_rows))

    def _ensure_rows(self, batch: int) -> None:
        if len(self._k_rows) < batch:
            self._k_rows.extend([None] * (batch - len(self._k_rows)))
        if len(self._v_rows) < batch:
            self._v_rows.extend([None] * (batch - len(self._v_rows)))

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

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        for i in range(len(self)):
            if self._k_rows[i] is not None:
                self._k_rows[i] = self._k_rows[i].to(device=device, dtype=dtype)
            if self._v_rows[i] is not None:
                self._v_rows[i] = self._v_rows[i].to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,   # [B, H, T_new, D]
        value_states: torch.Tensor, # [B, H, T_new, D]
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update per-row cache using provided positions and return padded batch K,V.

        For each batch row b, new tokens are placed starting at cache_position[b]. If
        that position is 0 or the row is empty, the row is replaced. If it is smaller
        than the current row length, the row is truncated first, then the new tokens
        are appended. If it is larger, the row is right-padded with zeros up to that
        position before appending.
        """
        assert key_states.dim() == 4 and value_states.dim() == 4, "Expected [B, H, T, D] tensors"
        B, H, T_new, D = key_states.shape
        self._ensure_rows(B)

        # Normalize cache_position to [B]
        if cache_position is None:
            # Default to append at current length
            positions = torch.zeros(B, dtype=torch.long, device=key_states.device)
            for b in range(B):
                cur = self._k_rows[b]
                positions[b] = 0 if cur is None else cur.shape[1]
        else:
            pos = cache_position
            if pos.dim() > 1:
                pos = pos.view(pos.shape[0])
            positions = pos.to(torch.long)

        # Update rows independently
        max_len = 0
        for b in range(B):
            p = _to_scalar_long(positions[b])
            k_new = key_states[b]
            v_new = value_states[b]

            k_prev = self._k_rows[b]
            v_prev = self._v_rows[b]

            if k_prev is None or p <= 0:
                k_row = k_new
                v_row = v_new
            else:
                prev_len = k_prev.shape[1]
                k_row = k_prev
                v_row = v_prev
                # Truncate or pad to position p
                if p < prev_len:
                    k_row = k_row[:, :p, :]
                    v_row = v_row[:, :p, :]
                elif p > prev_len:
                    pad_k = k_prev.new_zeros((H, p - prev_len, D))
                    pad_v = v_prev.new_zeros((H, p - prev_len, D))
                    k_row = torch.cat([k_row, pad_k], dim=1)
                    v_row = torch.cat([v_row, pad_v], dim=1)
                # Append new tokens
                k_row = torch.cat([k_row, k_new], dim=1)
                v_row = torch.cat([v_row, v_new], dim=1)

            self._k_rows[b] = k_row
            self._v_rows[b] = v_row
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


class PositionBasedDynamicCache(Cache):
    """A Cache that routes updates to per-layer LayerCache using cache_position.

    Exposes the HF-compatible `update(key, value, layer_idx, cache_kwargs)` API.
    Also provides `reset(mask)` to clear selected rows across all layers.
    """

    def __init__(self) -> None:
        super().__init__(layer_class_to_replicate=LayerCache)
        self._layers: Dict[int, LayerCache] = {}

    def _get_layer(self, layer_idx: int) -> LayerCache:
        if layer_idx not in self._layers:
            self._layers[layer_idx] = LayerCache()
        return self._layers[layer_idx]

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        for layer in self._layers.values():
            layer.to(device=device, dtype=dtype)
        return self

    def reset(self, reset_mask: torch.Tensor) -> None:
        for layer in self._layers.values():
            layer.reset_rows(reset_mask)

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position", None)
        layer = self._get_layer(layer_idx)
        return layer.update(key_states, value_states, cache_position=cache_position)


__all__ = [
    "LayerCache",
    "PositionBasedDynamicCache",
]
