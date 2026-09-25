"""Observation encoder and decoder interfaces for world models."""

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from utils import (
    expected_from_logits,
    make_probe_head,
    masked_lr_metrics_logits,
    masked_mse,
    masked_rmse,
    soft_cross_entropy,
)


class ObservationEncoder(nn.Module, ABC):
    """Convert a structured observation into a latent model input."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, observation) -> torch.Tensor:
        raise NotImplementedError


class ObservationDecoder(nn.Module, ABC):
    """Decode model features and evaluate the observation likelihood loss."""

    @abstractmethod
    def decode(self, features: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, prediction, target,
                     key_padding_mask: torch.Tensor, config) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, prediction, target, key_padding_mask: torch.Tensor,
                        config, auxiliary_prediction=None) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_step_error(self, prediction, target, key_padding_mask: torch.Tensor,
                           config) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sequence_length(self, prediction) -> int:
        raise NotImplementedError


class MazeObservationEncoder(ObservationEncoder):
    """Encode the existing left/front/right maze sensor tensor."""

    def __init__(self, encoder: nn.Module, sensor_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.sensor_dim = sensor_dim
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        assert observation.ndim == 3, f"Expected maze observation [B,T,D], got {tuple(observation.shape)}"
        assert observation.shape[-1] == self.sensor_dim, \
            f"Expected maze observation dim {self.sensor_dim}, got {observation.shape[-1]}"
        return self.encoder(observation)


class MazeObservationDecoder(ObservationDecoder):
    """Decode and score the existing three maze distance sensors."""

    def __init__(self, *, sensor_bins: Sequence[int], decoder: Optional[nn.Module] = None,
                 categorical_heads: Optional[Sequence[nn.Module]] = None):
        super().__init__()
        assert len(sensor_bins) == 3
        assert (decoder is None) != (categorical_heads is None)
        if categorical_heads is not None:
            assert len(categorical_heads) == 3

        self.sensor_bins = tuple(int(size) for size in sensor_bins)
        self.decoder = decoder
        self.categorical_heads = None if categorical_heads is None else nn.ModuleList(categorical_heads)

    def decode(self, features: torch.Tensor):
        if self.categorical_heads is not None:
            return tuple(head(features) for head in self.categorical_heads)

        logits = self.decoder(features)
        assert logits.shape[-1] == sum(self.sensor_bins)
        first, second, third = self.sensor_bins
        return (
            logits[..., :first],
            logits[..., first:first + second],
            logits[..., first + second:first + second + third],
        )

    def _target_indices(self, target: torch.Tensor, config) -> torch.Tensor:
        assert target.ndim == 3 and target.shape[-1] == 3, \
            f"Expected maze sensor target [B,T,3], got {tuple(target.shape)}"
        target_idx = target.round().to(torch.long).clamp(min=0)
        if config.sensor_min_idx is not None:
            target_idx = (target_idx - config.sensor_min_idx.view(1, 1, -1)).clamp(min=0)
        return target_idx

    def compute_loss(self, prediction, target: torch.Tensor,
                     key_padding_mask: torch.Tensor, config) -> torch.Tensor:
        assert isinstance(prediction, tuple) and len(prediction) == 3
        if config.sensor_tables is None:
            raise ValueError("categorical sensor loss requires config.sensor_tables")
        target_idx = self._target_indices(target, config)
        return sum(
            soft_cross_entropy(pred, table[target_idx[..., sensor_idx]], key_padding_mask)
            for sensor_idx, (pred, table) in enumerate(zip(prediction, config.sensor_tables))
        )

    def compute_metrics(self, prediction, target: torch.Tensor, key_padding_mask: torch.Tensor,
                        config, auxiliary_prediction=None) -> Mapping[str, torch.Tensor]:
        assert isinstance(prediction, tuple) and len(prediction) == 3
        if auxiliary_prediction is None:
            auxiliary_prediction = prediction
        assert isinstance(auxiliary_prediction, tuple) and len(auxiliary_prediction) == 3

        target_abs = target.round().to(torch.long).clamp(min=0)
        sensor_min = config.sensor_min_idx
        if sensor_min is None:
            sensor_min = torch.zeros(3, device=prediction[0].device, dtype=torch.float32)
        else:
            sensor_min = sensor_min.to(device=prediction[0].device, dtype=torch.float32)

        expected = torch.stack([
            expected_from_logits(logits, float(sensor_min[index].item()))
            for index, logits in enumerate(prediction)
        ], dim=-1)
        target_continuous = target_abs.to(expected.dtype)
        lr_rmse, lr_acc = masked_lr_metrics_logits(
            auxiliary_prediction[0], auxiliary_prediction[2], target_abs,
            key_padding_mask, sensor_min,
        )
        return {
            "mse": masked_mse(expected, target_continuous, key_padding_mask),
            "rmse": masked_rmse(expected, target_continuous, key_padding_mask),
            "lr_rmse": lr_rmse,
            "lr_acc": lr_acc,
        }

    def compute_step_error(self, prediction, target: torch.Tensor,
                           key_padding_mask: torch.Tensor, config) -> torch.Tensor:
        assert isinstance(prediction, tuple) and len(prediction) == 3
        if config.sensor_tables is None:
            raise ValueError("categorical sensor error requires config.sensor_tables")
        target_idx = self._target_indices(target, config)
        step_error = sum(
            -(table[target_idx[..., sensor_idx]] * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            for sensor_idx, (logits, table) in enumerate(zip(prediction, config.sensor_tables))
        )
        assert step_error.shape == key_padding_mask.shape
        return step_error.masked_fill(key_padding_mask, 0.0)

    def sequence_length(self, prediction) -> int:
        assert isinstance(prediction, tuple) and len(prediction) == 3
        sequence_length = prediction[0].shape[1]
        assert all(pred.ndim == 3 and pred.shape[1] == sequence_length for pred in prediction)
        return sequence_length


class AgiMazeObservationEncoder(ObservationEncoder):
    """Encode categorical movement feedback and the multi-hot key inventory."""

    def __init__(self, encoder: nn.Module, movement_result_classes: int,
                 inventory_size: int, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.movement_result_classes = movement_result_classes
        self.inventory_size = inventory_size
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        movement_result = observation["movement_result"]
        inventory = observation["inventory"]
        assert movement_result.ndim == 2, \
            f"Expected movement_result [B,T], got {tuple(movement_result.shape)}"
        assert inventory.ndim == 3 and inventory.shape[:2] == movement_result.shape, \
            f"Expected inventory [B,T,K] aligned with movement_result, got {tuple(inventory.shape)}"
        assert inventory.shape[-1] == self.inventory_size, \
            f"Expected inventory size {self.inventory_size}, got {inventory.shape[-1]}"
        assert ((movement_result >= 0) & (movement_result < self.movement_result_classes)).all()
        assert ((inventory == 0) | (inventory == 1)).all()

        movement_one_hot = F.one_hot(
            movement_result.to(torch.long),
            num_classes=self.movement_result_classes,
        ).to(inventory.dtype)
        encoder_input = torch.cat([movement_one_hot, inventory], dim=-1).to(torch.float32)
        return self.encoder(encoder_input)


class AgiMazeObservationDecoder(ObservationDecoder):
    """Predict categorical movement feedback and independent inventory bits."""

    def __init__(self, movement_result_head: nn.Module, inventory_head: nn.Module,
                 movement_result_classes: int, inventory_size: int):
        super().__init__()
        self.movement_result_head = movement_result_head
        self.inventory_head = inventory_head
        self.movement_result_classes = movement_result_classes
        self.inventory_size = inventory_size

    def decode(self, features: torch.Tensor):
        movement_result = self.movement_result_head(features)
        inventory = self.inventory_head(features)
        assert movement_result.shape[-1] == self.movement_result_classes
        assert inventory.shape[-1] == self.inventory_size
        return {
            "movement_result": movement_result,
            "inventory": inventory,
        }

    def compute_loss(self, prediction, target: Mapping[str, torch.Tensor],
                     key_padding_mask: torch.Tensor, config) -> torch.Tensor:
        del config
        movement_logits = prediction["movement_result"]
        inventory_logits = prediction["inventory"]
        movement_target = target["movement_result"]
        inventory_target = target["inventory"]
        assert movement_logits.ndim == 3
        assert inventory_logits.ndim == 3
        assert movement_target.shape == movement_logits.shape[:2]
        assert inventory_target.shape == inventory_logits.shape
        assert key_padding_mask.shape == movement_target.shape

        valid = ~key_padding_mask
        if not valid.any():
            return movement_logits.sum() * 0.0
        movement_loss = F.cross_entropy(movement_logits[valid], movement_target[valid].to(torch.long))
        inventory_loss = F.binary_cross_entropy_with_logits(
            inventory_logits[valid],
            inventory_target[valid].to(inventory_logits.dtype),
        )
        return movement_loss + inventory_loss

    def compute_metrics(self, prediction, target: Mapping[str, torch.Tensor],
                        key_padding_mask: torch.Tensor, config,
                        auxiliary_prediction=None) -> Mapping[str, torch.Tensor]:
        del config, auxiliary_prediction
        movement_logits = prediction["movement_result"]
        inventory_logits = prediction["inventory"]
        movement_target = target["movement_result"]
        inventory_target = target["inventory"]
        assert movement_logits.shape[:2] == movement_target.shape == key_padding_mask.shape
        assert inventory_logits.shape == inventory_target.shape

        valid = ~key_padding_mask
        if not valid.any():
            zero = movement_logits.sum() * 0.0
            return {"movement_result_acc": zero, "inventory_bit_acc": zero}
        movement_acc = (movement_logits[valid].argmax(dim=-1) == movement_target[valid]).float().mean()
        inventory_pred = inventory_logits[valid] >= 0.0
        inventory_acc = (inventory_pred == inventory_target[valid].to(torch.bool)).float().mean()
        return {
            "movement_result_acc": movement_acc,
            "inventory_bit_acc": inventory_acc,
        }

    def compute_step_error(self, prediction, target: Mapping[str, torch.Tensor],
                           key_padding_mask: torch.Tensor, config) -> torch.Tensor:
        del config
        movement_logits = prediction["movement_result"]
        inventory_logits = prediction["inventory"]
        movement_target = target["movement_result"]
        inventory_target = target["inventory"]
        assert movement_logits.shape[:2] == movement_target.shape == key_padding_mask.shape
        assert inventory_logits.shape == inventory_target.shape

        movement_error = F.cross_entropy(
            movement_logits.transpose(1, 2), movement_target.to(torch.long), reduction="none"
        )
        inventory_error = F.binary_cross_entropy_with_logits(
            inventory_logits, inventory_target.to(inventory_logits.dtype), reduction="none"
        ).mean(dim=-1)
        step_error = movement_error + inventory_error
        return step_error.masked_fill(key_padding_mask, 0.0)

    def sequence_length(self, prediction) -> int:
        movement_result = prediction["movement_result"]
        inventory = prediction["inventory"]
        assert movement_result.ndim == 3 and inventory.ndim == 3
        assert movement_result.shape[:2] == inventory.shape[:2]
        return movement_result.shape[1]


def create_maze_baseline_observation_codec(*, sensor_dim: int, sensor_latent_dim: int,
                                           feature_dim: int,
                                           sensor_bins: Sequence[int],
                                           hidden_dim: int):
    sensor_encoder = make_probe_head(sensor_dim, sensor_latent_dim, hidden_dim, 2)
    sensor_heads = tuple(nn.Linear(feature_dim, int(size)) for size in sensor_bins)
    encoder = MazeObservationEncoder(sensor_encoder, sensor_dim, sensor_latent_dim)
    decoder = MazeObservationDecoder(
        sensor_bins=sensor_bins,
        categorical_heads=sensor_heads,
    )
    probe_decoder = MazeObservationDecoder(
        sensor_bins=sensor_bins,
        categorical_heads=tuple(nn.Linear(sensor_latent_dim, int(size)) for size in sensor_bins),
    )
    return encoder, decoder, probe_decoder


def create_maze_discrete_observation_codec(*, sensor_dim: int, sensor_latent_dim: int,
                                           feature_dim: int, stochastic_dim: int,
                                           hidden_size: int, sensor_bins: Sequence[int]):
    sensor_encoder = nn.Sequential(
        nn.Linear(sensor_dim, sensor_latent_dim),
        nn.ReLU(),
        nn.Linear(sensor_latent_dim, sensor_latent_dim),
        nn.ReLU(),
    )
    sensor_out_dim = sum(int(size) for size in sensor_bins)
    encoder = MazeObservationEncoder(sensor_encoder, sensor_dim, sensor_latent_dim)
    decoder = MazeObservationDecoder(
        sensor_bins=sensor_bins,
        decoder=nn.Linear(feature_dim, sensor_out_dim),
    )
    z_decoder = MazeObservationDecoder(
        sensor_bins=sensor_bins,
        decoder=nn.Linear(stochastic_dim, sensor_out_dim),
    )
    h_decoder = MazeObservationDecoder(
        sensor_bins=sensor_bins,
        decoder=nn.Linear(hidden_size, sensor_out_dim),
    )
    return encoder, decoder, z_decoder, h_decoder


def create_agimaze_baseline_observation_codec(*, movement_result_classes: int,
                                              inventory_size: int, observation_latent_dim: int,
                                              feature_dim: int,
                                              hidden_dim: int):
    observation_dim = movement_result_classes + inventory_size
    encoder = AgiMazeObservationEncoder(
        make_probe_head(observation_dim, observation_latent_dim, hidden_dim, 2),
        movement_result_classes,
        inventory_size,
        observation_latent_dim,
    )
    decoder = AgiMazeObservationDecoder(
        nn.Linear(feature_dim, movement_result_classes),
        nn.Linear(feature_dim, inventory_size),
        movement_result_classes,
        inventory_size,
    )
    probe_decoder = AgiMazeObservationDecoder(
        nn.Linear(observation_latent_dim, movement_result_classes),
        nn.Linear(observation_latent_dim, inventory_size),
        movement_result_classes,
        inventory_size,
    )
    return encoder, decoder, probe_decoder


def create_agimaze_discrete_observation_codec(*, movement_result_classes: int,
                                              inventory_size: int, observation_latent_dim: int,
                                              feature_dim: int, stochastic_dim: int,
                                              hidden_size: int, hidden_dim: int):
    observation_dim = movement_result_classes + inventory_size

    def create_decoder(input_dim: int):
        return AgiMazeObservationDecoder(
            nn.Linear(input_dim, movement_result_classes),
            nn.Linear(input_dim, inventory_size),
            movement_result_classes,
            inventory_size,
        )

    encoder = AgiMazeObservationEncoder(
        make_probe_head(observation_dim, observation_latent_dim, hidden_dim, 2),
        movement_result_classes,
        inventory_size,
        observation_latent_dim,
    )
    return (
        encoder,
        create_decoder(feature_dim),
        create_decoder(stochastic_dim),
        create_decoder(hidden_size),
    )
