#!/usr/bin/env python3
"""Baseline predictor backbones (transformer and recurrent)."""

from typing import Dict, List, Optional

import torch
from torch import nn

from base import PredictionLossMixin
from observation import ObservationDecoder, ObservationEncoder
from transformer import LlamaConfig, LlamaModel, LlamaRMSNorm
from transformer_cached import CachedTransformer
from rnn_cached import CachedRNN
from utils import isotropic_normal_from_params, make_probe_head, scale_upstream_grad
from recurrent_mlp import RecurrentMLP
from recurrent_cache import clear_cache, reset_cache
from util import tree_index


class TransformerBaseline(PredictionLossMixin, nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        observation_encoder: ObservationEncoder,
        observation_decoder: ObservationDecoder,
        sensor_mode: str,
        action_dim: int,
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        action_latent_dim: int,
        probe_hidden_dim: int = 256,
        probe_layers: int = 2,
        contrastive_dim: int = 0,
        contrastive_steps: int = 1,
        cpc_context_dim=128,
        logger=None
    ):
        super().__init__()
        self.backbone = CachedTransformer(config)
        self.sensor_mode = sensor_mode
        self.heading_dim = heading_dim
        self.turn_bins = turn_bins
        self.step_bins = step_bins
        self.input_size = config.input_size
        self.action_latent_dim = action_latent_dim
        self.sensor_latent_dim = observation_encoder.latent_dim
        self.action_dim = action_dim
        if self.action_dim <= 0:
            raise ValueError("action_dim inferred from config.input_size - sensor_dim must be > 0")
        self.probe_hidden_dim = probe_hidden_dim
        self.probe_layers = probe_layers
        self.contrastive_dim = contrastive_dim
        self.contrastive_steps = contrastive_steps
        self._validate_params()
        self.observation_encoder = observation_encoder
        self.observation_decoder = observation_decoder
        self.loc_x_head = make_probe_head(config.hidden_size, loc_x_bins, self.probe_hidden_dim, self.probe_layers)
        self.loc_y_head = make_probe_head(config.hidden_size, loc_y_bins, self.probe_hidden_dim, self.probe_layers)
        self.heading_head = make_probe_head(config.hidden_size, heading_dim, self.probe_hidden_dim, self.probe_layers)
        self.action_encoder = make_probe_head(self.action_dim, self.action_latent_dim, self.probe_hidden_dim, 2)
        self.contrastive_context = nn.Linear(config.hidden_size, cpc_context_dim)
        self.obs_fuse = nn.Linear(config.hidden_size + self.action_latent_dim, config.hidden_size)
        self.turn_head = nn.Linear(config.hidden_size, turn_bins)
        self.step_head = nn.Linear(config.hidden_size, step_bins)
        self.cpc_sensor_latent_head = make_probe_head(
            self.contrastive_dim,
            self.sensor_latent_dim,
            self.probe_hidden_dim,
            3,)
        # inputs prev sfa + new feature
        self.cpc_sfa = RecurrentMLP(make_probe_head(self.contrastive_dim * 2, self.contrastive_dim, 256, 3))
        self.contrastive_target_head = nn.Sequential(
            nn.Linear(cpc_context_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.contrastive_dim),
        )
        self.contrastive_action_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cpc_context_dim + s * self.action_latent_dim, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, self.contrastive_dim + 1),
                )
                for s in range(1, self.contrastive_steps + 1)
            ]
        )
        self._num_envs: Optional[int] = None
        self._cache = None
        self.attention_window = config.attention_window
        self.logger = logger
        self.sfa_cpc_grad_scale = 0.05

    def _validate_params(self):
        if self.probe_hidden_dim < 0:
            raise ValueError("probe_hidden_dim must be >= 0")
        if self.probe_layers < 1:
            raise ValueError("probe_layers must be >= 1")
        if self.contrastive_dim < 0:
            raise ValueError("contrastive_dim must be >= 0")
        if self.contrastive_steps < 1:
            raise ValueError("contrastive_steps must be >= 1")

    def reset_cache(self, reset_mask: torch.Tensor):
        reset_cache(self, reset_mask)

    def clear_cache(self):
        clear_cache(self)

    def get_cache_state(self):
        return {
            "backbone": self.backbone.get_cache_state(),
            "sfa": self.cpc_sfa.get_cache_state(),
        }

    def set_cache_state(self, state):
        self.backbone.set_cache_state(state["backbone"])
        self.cpc_sfa.set_cache_state(state["sfa"])

    def index_cache_state(self, state, batch_indices: torch.Tensor):
        return {
            "backbone": self.backbone.index_cache_state(state["backbone"], batch_indices),
            "sfa": self.cpc_sfa.index_cache_state(state["sfa"], batch_indices),
        }

    def _prepare_prime_cache_inputs(self, obs):
        prime_obs = dict(obs)
        prime_obs.setdefault("actions", None)
        sensor, sensor_latent, actions, prev_actions, key_padding_mask, _ = self._validate_obs_contract(
            prime_obs,
            episode_start=None,
        )
        if key_padding_mask is not None and key_padding_mask.any():
            raise ValueError("prime_cache does not support padded sequence steps")

        batch_size, sequence_length = sensor_latent.shape[:2]
        if sequence_length < 1:
            raise ValueError("prime_cache requires at least one sequence step")
        return sensor, sensor_latent, actions, prev_actions

    def prime_cache(self, obs, episode_start=None):
        sensor, sensor_latent, actions, prev_actions = self._prepare_prime_cache_inputs(obs)
        batch_size, sequence_length = sensor_latent.shape[:2]
        if episode_start is None:
            episode_start = torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=sensor_latent.device,
            )
            episode_start[:, 0] = True
        assert episode_start.shape == (batch_size, sequence_length), \
            f"Expected episode_start [B,T], got {tuple(episode_start.shape)}"
        episode_start = episode_start.to(device=sensor_latent.device, dtype=torch.bool)

        self.clear_cache()
        result = None
        for step_idx in range(sequence_length):
            step_obs = {
                "actions": None if actions is None else actions[:, step_idx:step_idx + 1], # [B,1,A]
                "prev_actions": prev_actions[:, step_idx:step_idx + 1],
            }
            if sensor is not None:
                step_obs["sensor"] = tree_index(sensor, (slice(None), slice(step_idx, step_idx + 1)))
            else:
                step_obs["sensor_latent"] = sensor_latent[:, step_idx:step_idx + 1]
            result = self.forward(
                step_obs,
                episode_start=episode_start[:, step_idx],
            )

        assert result is not None
        return result

    def _project_contrastive_target_h(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.contrastive_target_head is None:
            return None
        return self.contrastive_target_head(h)

    def _project_contrastive_pred_steps(self, h: torch.Tensor, actions: torch.Tensor):
        if self.contrastive_dim <= 0 or self.contrastive_action_heads is None:
            return [], []
        B, T, _ = h.shape
        mean_steps: List[torch.Tensor] = []
        scale_steps: List[torch.Tensor] = []
        for horizon, head in enumerate(self.contrastive_action_heads, start=1):
            if horizon >= T:
                break
            th = T - horizon
            action_chunks = [actions[:, i : i + th, :] for i in range(horizon)]
            action_ctx = torch.cat(action_chunks, dim=-1) if action_chunks else actions.new_zeros((B, th, 0))
            pred_in = torch.cat([h[:, :th, :], action_ctx], dim=-1)
            dist = isotropic_normal_from_params(head(pred_in), self.contrastive_dim)
            mean_steps.append(dist.base_dist.loc)
            scale_steps.append(dist.base_dist.scale[..., :1])
        return mean_steps, scale_steps

    def predict_next_contrastive_dist(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.distributions.Independent:
        assert h.dim() == 2, f"Expected h [B,H], got {tuple(h.shape)}"
        assert action.dim() == 2, f"Expected action [B,A], got {tuple(action.shape)}"
        assert action.shape[0] == h.shape[0], "h and action batch sizes must match"
        assert action.shape[-1] == self.action_dim, f"Expected action dim {self.action_dim}, got {action.shape[-1]}"
        assert self.contrastive_action_heads is not None, "contrastive_action_heads is not initialized"
        assert len(self.contrastive_action_heads) >= 1, "At least one contrastive action head is required"

        context = self.contrastive_context(h)
        action_latent = self.action_encoder(action.to(dtype=h.dtype, device=h.device))
        pred_input = torch.cat([context, action_latent], dim=-1)
        return isotropic_normal_from_params(
            self.contrastive_action_heads[0](pred_input),
            self.contrastive_dim,
        )

    def predict_next_contrastive_emb(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_next_contrastive_dist(h, action).mean

    def _forward_core(
        self,
        obs,
        *,
        episode_start=None,
    ):
        sensor, sensor_latent, actions, prev_actions, key_padding_mask, _ = self._validate_obs_contract(
            obs,
            episode_start=episode_start,
        )
        prev_action_latent = self.action_encoder(prev_actions)
        x = torch.cat([sensor_latent, prev_action_latent], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        h = self.backbone(
            x,
            key_padding_mask=key_padding_mask,
            reset_mask=episode_start)
        preds, aux_inputs = self.forward_preds(h, actions, sensor_latent, episode_start)
        return preds, aux_inputs, h, h[:, -1, :]

    def forward_preds(self, h, actions, sensor_latent, episode_start):
        """
        Compute cpc/sfa and other features based on {st, at}
        """
        if actions is None:
            action_latent = None
            pred_sensor = None
        else:
            action_latent = self.action_encoder(actions)
            obs_feat = torch.tanh(self.obs_fuse(torch.cat([h, action_latent], dim=-1)))
            pred_sensor = self.observation_decoder.decode(obs_feat)
        aux_inputs = self.compute_aux(h, action_latent, episode_start, sensor_latent)
        # State probes (current-step location/heading) read detached state.
        h_probe = h.detach()
        # Action heads use detached or live state based on constructor setting.
        action_feat = h_probe
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        turn = self.turn_head(action_feat)
        step = self.step_head(action_feat)

        preds = (pred_sensor, loc_x, loc_y, heading, turn, step)
        return preds, aux_inputs

    def forward(
        self,
        obs,
        episode_start=None,
    ):
        preds, aux_inputs, state_seq, last_state = self._forward_core(
            obs,
            episode_start=episode_start,
        )
        return {
            "preds": preds,
            "aux": aux_inputs,
            "state": last_state,
            "state_last": last_state,
            "state_seq": state_seq,
        }

    def compute_aux(self, h, action_latent, reset_mask, sensor_latent):
        aux_inputs = {}
        contrastive_input = self.contrastive_context(h)
        aux_inputs["contrastive_tgt_emb"] = self._project_contrastive_target_h(contrastive_input)
        if action_latent is None:
            pred_steps, scale_steps = [], []
        else:
            pred_steps, scale_steps = self._project_contrastive_pred_steps(contrastive_input, action_latent)
        aux_inputs["contrastive_pred_emb_steps"] = pred_steps
        aux_inputs["contrastive_pred_scale_steps"] = scale_steps
        aux_inputs["sensor_latent"] = sensor_latent
        aux_inputs["cpc_sensor_latent_pred"] = self.cpc_sensor_latent_head(aux_inputs["contrastive_tgt_emb"])

        sfa = self.cpc_sfa(scale_upstream_grad(aux_inputs["contrastive_tgt_emb"], scale=self.sfa_cpc_grad_scale),
                            reset_mask)

        if reset_mask is not None:
            assert reset_mask.numel() == sfa.shape[0]

        aux_inputs['sfa'] = sfa
        return aux_inputs


class RNNPredictor(TransformerBaseline):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        observation_encoder: ObservationEncoder,
        observation_decoder: ObservationDecoder,
        sensor_mode: str,
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        action_dim: int,
        action_latent_dim: int,
        probe_hidden_dim: int = 256,
        probe_layers: int = 2,
        state_norm: str = "none",
        contrastive_dim: int = 0,
        contrastive_steps: int = 1,
        logger=None
    ):
        super().__init__(
            config,
            observation_encoder=observation_encoder,
            observation_decoder=observation_decoder,
            sensor_mode=sensor_mode,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            action_dim=action_dim,
            action_latent_dim=action_latent_dim,
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
            logger=logger
        )
        self.backbone = CachedRNN(config)
