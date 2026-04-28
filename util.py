import shutil
from pathlib import Path

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence


def copy_python_sources(script_dir: str, experiment_dir: str) -> None:
    """Copy all Python files from the script directory into the experiment directory."""
    script_path = Path(script_dir)
    destination_path = Path(experiment_dir)

    destination_path.mkdir(parents=True, exist_ok=True)

    for py_file in script_path.glob("*.py"):
        dest_file = destination_path / py_file.name
        try:
            if py_file.resolve() == dest_file.resolve():
                continue
            shutil.copy2(py_file, dest_file)
        except Exception as exc:
            print(f"[WARN] Failed to copy {py_file} to {dest_file}: {exc}")


class StateExtractor:
    def __init__(self, key_slices=None):
        """
        Initialize with a dictionary mapping keys to slices.

        Args:
            key_slices (dict): Dictionary mapping field names to slice objects
        """
        self.key_slices = key_slices or {}
        self.field_shapes = {}  # Store original shapes for reshaping

    def add_mapping(self, key, slice_obj, shape=None):
        """Add a single key-slice mapping."""
        self.key_slices[key] = slice_obj
        if shape is not None:
            self.field_shapes[key] = shape

    def add_field_at_end(self, key, shape):
        """
        Add a new field to the end of the state vector.

        Args:
            key (str): Name of the new field
            shape (tuple): Shape of the new field
        """
        # Calculate size of the new field
        field_size = int(np.prod(shape))

        # Find the current end position
        end_pos = 0
        if self.key_slices:
            end_pos = max(slice_obj.stop for slice_obj in self.key_slices.values())

        # Create the slice and add the mapping
        slice_obj = slice(end_pos, end_pos + field_size)
        self.add_mapping(key, slice_obj, shape)

        return self

    def extract(self, state_batch, keys):
        """
        Extract one or more fields from the state batch.

        Args:
            state_batch: Tensor of shape [batch_size, feature_dim] or [feature_dim]
            keys: The field key(s) to extract - can be a string or list of strings

        Returns:
            If keys is a string: Tensor with the extracted field
            If keys is a list: List of tensors with the extracted fields in the same order
        """
        # Handle single key case
        if isinstance(keys, str):
            if keys not in self.key_slices:
                raise KeyError(f"Key '{keys}' not found in state extractor mappings")

            # Handle both batched and unbatched inputs
            if len(state_batch.shape) > 1:
                # Batched input: extract along the second dimension
                return state_batch[:, self.key_slices[keys]]
            else:
                # Single vector: extract directly
                return state_batch[self.key_slices[keys]]

        # Handle multiple keys case
        elif isinstance(keys, (list, tuple)):
            # Validate all keys
            for key in keys:
                if key not in self.key_slices:
                    raise KeyError(f"Key '{key}' not found in state extractor mappings")

            # Extract each field
            extracted_fields = []
            for key in keys:
                if len(state_batch.shape) > 1:
                    # Batched input
                    extracted_fields.append(state_batch[:, self.key_slices[key]])
                else:
                    # Single vector
                    extracted_fields.append(state_batch[self.key_slices[key]])

            return torch.cat(extracted_fields, dim=1)

        else:
            raise TypeError(f"Expected string or list of strings for keys, got {type(keys)}")

    def extract_and_reshape(self, state_batch, key):
        """
        Extract a field and reshape it to its original dimensions.

        Args:
            state_batch: Tensor of shape [batch_size, feature_dim] or [feature_dim]
            key: The field key to extract

        Returns:
            Reshaped tensor with the extracted field
        """
        extracted = self.extract(state_batch, key)

        if key not in self.field_shapes:
            return extracted

        # Reshape based on whether input is batched or not
        if len(state_batch.shape) > 1:
            batch_size = state_batch.shape[0]
            return extracted.reshape(batch_size, *self.field_shapes[key])
        else:
            return extracted.reshape(self.field_shapes[key])

    def remove(self, state_batch, keys_to_remove):
        """
        Remove specified fields from the state batch.

        Args:
            state_batch: Tensor of shape [batch_size, feature_dim] or [feature_dim]
            keys_to_remove: List of keys to remove from the state

        Returns:
            Tensor with the specified fields removed
        """
        if isinstance(keys_to_remove, str):
            keys_to_remove = [keys_to_remove]

        # Validate keys
        for key in keys_to_remove:
            if key not in self.key_slices:
                raise KeyError(f"Key '{key}' not found in state extractor mappings")

        # Get all indices to keep
        all_indices = []
        for key, slice_obj in self.key_slices.items():
            if key not in keys_to_remove:
                # Add all indices in this slice
                all_indices.extend(range(slice_obj.start, slice_obj.stop))

        # Sort indices to maintain order
        all_indices.sort()

        # Handle both batched and unbatched inputs
        if len(state_batch.shape) > 1:
            # Batched input: index along the second dimension
            return state_batch[:, all_indices]
        else:
            # Single vector: index directly
            return state_batch[all_indices]

    def get_state_without(self, state_batch, keys_to_remove):
        """Alias for remove method."""
        return self.remove(state_batch, keys_to_remove)

    @classmethod
    def from_dict_observation(cls, obs_dict):
        """
        Create a StateExtractor from a dictionary observation.

        Args:
            obs_dict (dict): A sample observation dictionary

        Returns:
            StateExtractor: Initialized with mappings for all fields
        """
        extractor = cls()
        current_idx = 0

        def process_field(key, value):
            nonlocal current_idx

            # Get the size and shape of this field
            if hasattr(value, 'shape'):
                # For tensors/arrays
                if len(value.shape) > 0:
                    # Skip batch dimension if present
                    if len(value.shape) > 1 and value.shape[0] == 1:
                        field_shape = value.shape[1:]
                        field_size = int(np.prod(field_shape))
                    else:
                        field_shape = value.shape
                        field_size = int(np.prod(field_shape))
                else:
                    field_shape = ()
                    field_size = 1
            else:
                # For scalars
                field_shape = ()
                field_size = 1

            # Create the slice and add the mapping
            slice_obj = slice(current_idx, current_idx + field_size)
            extractor.add_mapping(key, slice_obj, field_shape)
            current_idx += field_size

        # Process the dictionary in a deterministic order
        for key in sorted(obs_dict.keys()):
            if isinstance(obs_dict[key], dict):
                # Handle nested dictionaries
                for subkey in sorted(obs_dict[key].keys()):
                    field_name = f"{key}.{subkey}"
                    field_value = obs_dict[key][subkey]
                    process_field(field_name, field_value)
            else:
                # Handle top-level fields
                process_field(key, obs_dict[key])

        return extractor

    def get_fields_size(self, fields):
        """
        Calculate the total size of specified fields.

        Args:
            fields (list or str): Field name(s) to calculate size for

        Returns:
            int: Total size of the specified fields
        """
        if isinstance(fields, str):
            fields = [fields]

        total_size = 0
        for field in fields:
            if field not in self.key_slices:
                raise KeyError(f"Field '{field}' not found in state extractor mappings")

            slice_obj = self.key_slices[field]
            field_size = slice_obj.stop - slice_obj.start
            total_size += field_size

        return total_size


class RunningNorm:
    def __init__(self, epsilon=1e-5, momentum=0.999, min_std=1e-2, device=None):
        self.epsilon = epsilon
        self.momentum = momentum
        self.min_std = min_std
        self.mean = None
        self.std = None

        # For datatype consistency
        self.use_torch = None
        self.device = device
        self.dtype = None

    def _mean(self, x):
        if isinstance(x, torch.Tensor):
            return x.mean(dim=0, keepdim=True)
        else:  # numpy
            return x.mean(axis=0, keepdims=True)

    def _std(self, x):
        if isinstance(x, torch.Tensor):
            # unbiased=False matches numpy's default behavior
            return x.std(dim=0, unbiased=False, keepdim=True)
        else:
            return x.std(axis=0, keepdims=True)

    def _maximum(self, x, val):
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min=val)
        else:
            return np.maximum(x, val)

    def __call__(self, x):
        # Detect and fix the type at first call
        if self.use_torch is None:
            if isinstance(x, torch.Tensor):
                self.use_torch = True
                self.dtype = x.dtype
                self.device = x.device if self.device is None else self.device
            elif isinstance(x, np.ndarray):
                self.use_torch = False
                self.dtype = x.dtype
            else:
                raise TypeError("Input must be NumPy array or PyTorch tensor.")

        # Enforce consistent types on subsequent calls
        if self.use_torch and not isinstance(x, torch.Tensor):
            raise TypeError("Initialized with PyTorch tensor but received NumPy array.")
        if not self.use_torch and not isinstance(x, np.ndarray):
            raise TypeError("Initialized with NumPy array but received PyTorch tensor.")

        # Calculate mean and std using unified API
        batch_mean = self._mean(x)
        batch_std = self._std(x) + self.epsilon

        # Update running mean and std
        if self.mean is None:
            self.mean = batch_mean
            self.std = batch_std
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.std = self.momentum * self.std + (1 - self.momentum) * batch_std
            if not 0.2 < self.std:
                print('std is low ' + str(self.std)) # sanity check

        # Normalize
        normalized = (x - self.mean) / (self.std + self.epsilon)

        return normalized


# =========================
# Episode/Sequence Utilities
# =========================

TensorLike = Union[torch.Tensor, Dict[str, torch.Tensor]]


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def tree_map(x: Any, fn):
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, dict):
        return {k: tree_map(v, fn) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [tree_map(v, fn) for v in x]
        return type(x)(t)
    return x


def to_device(x: Any, device: torch.device):
    return tree_map(x, lambda t: t.to(device) if isinstance(t, torch.Tensor) else t)


def detach(x: Any):
    return tree_map(x, lambda t: t.detach() if isinstance(t, torch.Tensor) else t)


def pad_sequence_list(seq_list: List[torch.Tensor], pad_value: float = 0.0, batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a list of 2D/3D tensors with time in dim=0 into [B, T, ...].
    Returns: (padded, key_padding_mask[B,T], lengths[B]).
    key_padding_mask: True for padded positions.
    """
    if len(seq_list) == 0:
        raise ValueError("pad_sequence_list requires at least one sequence")
    lengths = torch.tensor([s.shape[0] for s in seq_list], dtype=torch.long)
    padded = pad_sequence(seq_list, batch_first=batch_first, padding_value=pad_value)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    arange = torch.arange(max_len, device=lengths.device)
    mask = arange.unsqueeze(0) >= lengths.unsqueeze(1)
    return padded, mask, lengths


def pad_dict_sequence_list(dict_list: List[Dict[str, torch.Tensor]], pad_value: float = 0.0, batch_first: bool = True) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Pad a list of dict sequences where each value is a time-major tensor [T, ...].
    Returns: (padded_dict with [B,T,...] per key, key_padding_mask[B,T], lengths[B]).
    Assumes all dicts have identical keys.
    """
    if len(dict_list) == 0:
        raise ValueError("pad_dict_sequence_list requires at least one sequence")
    keys = list(dict_list[0].keys())
    # Compute lengths from any one key
    lengths = torch.tensor([dict_list[i][keys[0]].shape[0] for i in range(len(dict_list))], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    arange = torch.arange(max_len, device=lengths.device)
    mask = arange.unsqueeze(0) >= lengths.unsqueeze(1)
    padded: Dict[str, torch.Tensor] = {}
    for k in keys:
        seqs = [d[k] for d in dict_list]
        padded[k] = pad_sequence(seqs, batch_first=batch_first, padding_value=pad_value)
    return padded, mask, lengths


def pack_dict_sequence_list(dict_list: List[Dict[str, torch.Tensor]], enforce_sorted: bool = False) -> Dict[str, torch.nn.utils.rnn.PackedSequence]:
    """
    Pack a list of dict sequences into a dict of PackedSequence, one per key.
    Assumes each value tensor is time-major [T, ...].
    """
    if len(dict_list) == 0:
        raise ValueError("pack_dict_sequence_list requires at least one sequence")
    keys = list(dict_list[0].keys())
    packed: Dict[str, torch.nn.utils.rnn.PackedSequence] = {}
    for k in keys:
        seqs = [d[k] for d in dict_list]
        packed[k] = pack_sequence(seqs, enforce_sorted=enforce_sorted)
    return packed


def flatten_padded(x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Flatten [B,T,...] into [N,...] by removing padded steps (mask=True).
    """
    if x.dim() < 2:
        raise ValueError("flatten_padded expects a tensor with at least 2 dims [B,T,...]")
    B, T = x.shape[:2]
    valid = (~key_padding_mask).reshape(B * T)
    return x.reshape(B * T, *x.shape[2:])[valid]


def compute_returns_list(rewards_list: List[torch.Tensor], gamma: float, bootstrap: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """
    Compute discounted returns per episode. Each rewards tensor is [T].
    Optional bootstrap value per-episode (Tensor[B]) to use as V_{T}.
    """
    out: List[torch.Tensor] = []
    for i, r in enumerate(rewards_list):
        T = r.shape[0]
        G = torch.zeros(T, dtype=r.dtype, device=r.device)
        next_ret = bootstrap[i].item() if (bootstrap is not None) else 0.0
        for t in range(T - 1, -1, -1):
            next_ret = r[t] + gamma * next_ret
            G[t] = next_ret
        out.append(G)
    return out


def compute_gae_list(
    rewards_list: List[torch.Tensor],
    values_list: List[torch.Tensor],
    gamma: float,
    lam: float,
    bootstrap: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute per-episode GAE advantages and returns.
    Each rewards/values tensor is [T]; returns = advantages + values.
    Optional per-episode bootstrap value V_T for last step (0 if None).
    """
    adv_list: List[torch.Tensor] = []
    ret_list: List[torch.Tensor] = []
    for i, (r, v) in enumerate(zip(rewards_list, values_list)):
        T = r.shape[0]
        assert v.shape[0] == T, "values and rewards must have same time length per episode"
        adv = torch.zeros(T, dtype=r.dtype, device=r.device)
        gae = 0.0
        v_next = bootstrap[i] if (bootstrap is not None) else torch.tensor(0.0, dtype=v.dtype, device=v.device)
        for t in range(T - 1, -1, -1):
            delta = r[t] + gamma * (v_next if t == T - 1 else v[t + 1]) - v[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = adv + v
        adv_list.append(adv)
        ret_list.append(ret)
    return adv_list, ret_list


class EpisodeBatch:
    """
    Container for per-episode tensors that keeps boundaries by default.

    Expected structure for data lists (per episode):
      - 'states': Tensor [T, ...] or Dict[str, Tensor[T, ...]]
      - 'actions': Tensor [T, A]
      - 'rewards': Tensor [T]
      - Optional: 'log_probs' [T], 'entropy' [T], 'values' [T]
    """

    def __init__(self, data_lists: Dict[str, List[TensorLike]]):
        if not data_lists:
            raise ValueError("EpisodeBatch requires non-empty data_lists")
        # Validate consistent episode counts
        lens = None
        ref_key = None
        for k, lst in data_lists.items():
            if not isinstance(lst, list):
                raise TypeError(f"Field '{k}' must be a list of per-episode tensors")
            if len(lst) == 0:
                continue
            if lens is None:
                ref_key = k
                lens = [self._length_of_episode_item(lst[0])]
                for item in lst[1:]:
                    lens.append(self._length_of_episode_item(item))
            else:
                other_lens = [self._length_of_episode_item(item) for item in lst]
                if len(other_lens) != len(lens):
                    raise ValueError("All fields must have same number of episodes")
        self.data: Dict[str, List[TensorLike]] = data_lists
        self.lengths = torch.tensor(lens or [], dtype=torch.long)

    @staticmethod
    def _length_of_episode_item(x: TensorLike) -> int:
        if isinstance(x, dict):
            any_key = next(iter(x))
            return int(x[any_key].shape[0])
        return int(x.shape[0])

    @property
    def num_episodes(self) -> int:
        return int(self.lengths.numel())

    def to(self, device: torch.device):
        self.data = to_device(self.data, device)
        self.lengths = self.lengths.to(device)
        return self

    def detach(self):
        self.data = detach(self.data)
        return self

    def compute_returns(self, gamma: float, gae_lambda: Optional[float] = None) -> "EpisodeBatch":
        if 'rewards' not in self.data:
            raise KeyError("'rewards' field is required to compute returns")
        rewards_list: List[torch.Tensor] = self.data['rewards']  # type: ignore

        if gae_lambda is None:
            returns = compute_returns_list(rewards_list, gamma)
            self.data['returns'] = returns
        else:
            if 'values' not in self.data:
                raise KeyError("'values' field is required for GAE computation")
            values_list: List[torch.Tensor] = self.data['values']  # type: ignore
            adv, ret = compute_gae_list(rewards_list, values_list, gamma, gae_lambda)
            self.data['advantages'] = adv
            self.data['returns'] = ret
        return self

    def pad(self, pad_value: float = 0.0, fields: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """
        Pad specified fields into [B,T,...]. Returns: (padded_data, key_padding_mask[B,T], lengths[B]).
        If fields is None, pad all present fields.
        For 'states' which can be dicts, returns a dict of padded tensors.
        key_padding_mask is shared across fields based on episode lengths.
        """
        if fields is None:
            fields = list(self.data.keys())

        # Derive mask from lengths once
        lengths = self.lengths
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        device = lengths.device
        arange = torch.arange(max_len, device=device)
        key_padding_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1)

        padded: Dict[str, Any] = {}
        for k in fields:
            if k not in self.data:
                continue
            lst = self.data[k]
            if isinstance(lst[0], dict):
                padded[k], _, _ = pad_dict_sequence_list(lst, pad_value=pad_value, batch_first=True)
            elif isinstance(lst[0], torch.Tensor):
                padded[k], _, _ = pad_sequence_list(lst, pad_value=pad_value, batch_first=True)
            else:
                raise TypeError(f"Unsupported episode item type for field '{k}'")
        return padded, key_padding_mask, lengths

    def flatten(self, fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
          if fields is None:
              fields = list(self.data.keys())
          flat: Dict[str, Any] = {}
          for k in fields:
              if k not in self.data:
                  continue
              lst = self.data[k]
              if isinstance(lst[0], dict):
                  keys = list(lst[0].keys())
                  flat[k] = {kk: torch.cat([ep[kk] for ep in lst], dim=0) for kk in keys} 
              else:
                  flat[k] = torch.cat(lst, dim=0)
          return flat

    def pack(self, fields: Optional[Sequence[str]] = None, enforce_sorted: bool = False) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Create PackedSequence objects per requested field for RNN/GRU/LSTM consumption.
        - For tensor fields: returns PackedSequence
        - For dict fields: returns dict[str -> PackedSequence]
        PackedSequence preserves per-episode boundaries and avoids padding.
        Note: Not suitable for transformer models; use pad() for those.

        Returns: (packed_data, lengths[B])
        """
        if fields is None:
            fields = list(self.data.keys())
        packed: Dict[str, Any] = {}
        for k in fields:
            if k not in self.data:
                continue
            lst = self.data[k]
            if isinstance(lst[0], dict):
                packed[k] = pack_dict_sequence_list(lst, enforce_sorted=enforce_sorted)
            elif isinstance(lst[0], torch.Tensor):
                packed[k] = pack_sequence(lst, enforce_sorted=enforce_sorted)
            else:
                raise TypeError(f"Unsupported episode item type for field '{k}'")
        return packed, self.lengths


def normalize_padded_returns(returns, key_padding_mask):
    valid = ~key_padding_mask
    if valid.sum() == 0:
        import logging
        logging.warning("no valid value to normalize in normalized_padded")
        return
    r_valid = returns[valid]
    r_mean = r_valid.mean()
    r_std = r_valid.std().clamp_min(1e-2)
    normalized_returns = torch.zeros_like(returns)
    normalized_returns[valid] = (r_valid - r_mean) / r_std
    return normalized_returns, r_mean, r_std


def gae(gamma, lambda_discount, rewards, values):
    # B, T
    assert len(rewards.shape) == 2
    assert rewards.shape == values.shape
    # v1, v2, v3
    # v0, v2
    v_next = values[:, 1:]
    v_prev = values[:, :-1]
    dt = torch.zeros_like(values)
    dt[:, :-1] = rewards[:, :-1] + gamma * v_next - v_prev
    dt[:, -1] = rewards[:, -1] - values[:, -1]
    adv = dt.clone()
    for i in range(adv.shape[-1] - 2, -1, -1):
        adv[:, i] = dt[:, i] + gamma * lambda_discount * adv[:, i + 1]
    return adv
