import numpy as np
import torch


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
