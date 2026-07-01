#!/usr/bin/env python3
"""Baseline predictor backbones (transformer and recurrent)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from base import PredictionLossMixin
from transformer import LlamaConfig, LlamaModel, LlamaRMSNorm
from transformer_cached import CachedTransformer
from rnn_cached import CachedRNN
from utils import make_probe_head, scale_upstream_grad
from recurrent_mlp import RecurrentMLP
from recurrent_cache import clear_cache, reset_cache


class TransformerBaseline(PredictionLossMixin, nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        sensor_mode: str,
        sensor_dim: int,
        action_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        action_latent_dim: int,
        sensor_latent_dim: int,
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
        self.sensor_latent_dim = sensor_latent_dim
        self.sensor_dim = sensor_dim
        self.action_dim = action_dim
        if self.action_dim <= 0:
            raise ValueError("action_dim inferred from config.input_size - sensor_dim must be > 0")
        self.probe_hidden_dim = probe_hidden_dim
        self.probe_layers = probe_layers
        self.contrastive_dim = contrastive_dim
        self.contrastive_steps = contrastive_steps
        self._validate_params()
        if sensor_mode == "categorical":
            assert sensor_bins is not None and len(sensor_bins) == 3
            self.sensor_head_l = nn.Linear(config.hidden_size, int(sensor_bins[0]))
            self.sensor_head_f = nn.Linear(config.hidden_size, int(sensor_bins[1]))
            self.sensor_head_r = nn.Linear(config.hidden_size, int(sensor_bins[2]))
        else:
            self.sensor_head = nn.Linear(config.hidden_size, sensor_dim)
        self.loc_x_head = make_probe_head(config.hidden_size, loc_x_bins, self.probe_hidden_dim, self.probe_layers)
        self.loc_y_head = make_probe_head(config.hidden_size, loc_y_bins, self.probe_hidden_dim, self.probe_layers)
        self.heading_head = make_probe_head(config.hidden_size, heading_dim, self.probe_hidden_dim, self.probe_layers)
        self.action_encoder = make_probe_head(self.action_dim, self.action_latent_dim, self.probe_hidden_dim, 2)
        self.sensor_encoder = make_probe_head(self.sensor_dim, self.sensor_latent_dim, self.probe_hidden_dim, 2)
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
                    nn.Linear(config.hidden_size, self.contrastive_dim),
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

    def _project_contrastive_target_h(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.contrastive_target_head is None:
            return None
        return self.contrastive_target_head(h)

    def _project_contrastive_pred_steps(self, h: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        if self.contrastive_dim <= 0 or self.contrastive_action_heads is None:
            return []
        B, T, _ = h.shape
        pred_steps: List[torch.Tensor] = []
        for horizon, head in enumerate(self.contrastive_action_heads, start=1):
            if horizon >= T:
                break
            th = T - horizon
            action_chunks = [actions[:, i : i + th, :] for i in range(horizon)]
            action_ctx = torch.cat(action_chunks, dim=-1) if action_chunks else actions.new_zeros((B, th, 0))
            pred_in = torch.cat([h[:, :th, :], action_ctx], dim=-1)
            pred_steps.append(head(pred_in))
        return pred_steps

    def predict_next_contrastive_emb(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert h.dim() == 2, f"Expected h [B,H], got {tuple(h.shape)}"
        assert action.dim() == 2, f"Expected action [B,A], got {tuple(action.shape)}"
        assert action.shape[0] == h.shape[0], "h and action batch sizes must match"
        assert action.shape[-1] == self.action_dim, f"Expected action dim {self.action_dim}, got {action.shape[-1]}"
        assert self.contrastive_action_heads is not None, "contrastive_action_heads is not initialized"
        assert len(self.contrastive_action_heads) >= 1, "At least one contrastive action head is required"

        context = self.contrastive_context(h)
        action_latent = self.action_encoder(action.to(dtype=h.dtype, device=h.device))
        pred_input = torch.cat([context, action_latent], dim=-1)
        return self.contrastive_action_heads[0](pred_input)

    def _forward_core(
        self,
        obs,
        *,
        episode_start=None,
    ):
        sensor, actions, prev_actions, key_padding_mask, _ = self._validate_obs_contract(
            obs,
            episode_start=episode_start,
        )

        using_internal_cache = episode_start is not None
        prev_action_latent = self.action_encoder(prev_actions)
        sensor_latent = self.sensor_encoder(sensor)
        x = torch.cat([sensor_latent, prev_action_latent], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        h = self.backbone(
            x,
            key_padding_mask=key_padding_mask,
            reset_mask=episode_start)
        action_latent = self.action_encoder(actions)
        obs_feat = torch.tanh(self.obs_fuse(torch.cat([h, action_latent], dim=-1)))
        if self.sensor_mode != "categorical":
            raise ValueError("TransformerBaseline currently supports sensor_mode='categorical' only.")
        pred_sensor = (
            self.sensor_head_l(obs_feat),
            self.sensor_head_f(obs_feat),
            self.sensor_head_r(obs_feat),
        )
        # State probes (current-step location/heading) read detached state.
        h_probe = h.detach()
        # Action heads use detached or live state based on constructor setting.
        action_feat = h_probe
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        turn = self.turn_head(action_feat)
        step = self.step_head(action_feat)
        aux_inputs = self.compute_aux(h, action_latent, episode_start, sensor_latent)
        preds = (pred_sensor, loc_x, loc_y, heading, turn, step)
        return preds, aux_inputs, h, h[:, -1, :]

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
        aux_inputs["contrastive_pred_emb_steps"] = self._project_contrastive_pred_steps(contrastive_input, action_latent)
        aux_inputs["sensor_latent"] = sensor_latent
        aux_inputs["cpc_sensor_latent_pred"] = self.cpc_sensor_latent_head(aux_inputs["contrastive_tgt_emb"])

        sfa = self.cpc_sfa(scale_upstream_grad(aux_inputs["contrastive_tgt_emb"], scale=self.sfa_cpc_grad_scale),
                            reset_mask)

        if reset_mask is not None:
            assert sfa.shape[1] == 1

        aux_inputs['sfa'] = sfa
        return aux_inputs


class RNNPredictor(TransformerBaseline):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        sensor_mode: str,
        sensor_dim: int,
        sensor_bins: Optional[np.ndarray],
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        action_dim: int,
        action_latent_dim: int,
        sensor_latent_dim: int,
        probe_hidden_dim: int = 256,
        probe_layers: int = 2,
        state_norm: str = "none",
        contrastive_dim: int = 0,
        contrastive_steps: int = 1,
        logger=None
    ):
        super().__init__(
            config,
            sensor_mode=sensor_mode,
            sensor_dim=sensor_dim,
            sensor_bins=sensor_bins,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            action_dim=action_dim,
            action_latent_dim=action_latent_dim,
            sensor_latent_dim=sensor_latent_dim,
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
            logger=logger
        )
        self.backbone = CachedRNN(config)
