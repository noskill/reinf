#!/usr/bin/env python3
"""Baseline predictor backbones (transformer and recurrent)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from base import PredictionLossMixin
from tr_cache import PositionBasedDynamicCache, WindowedPositionBasedDynamicCache
from transformer import LlamaConfig, LlamaModel, LlamaRMSNorm
from utils import make_probe_head



class TransformerBaseline(PredictionLossMixin, nn.Module):
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
        obs_dim: int,
        obs_latent_dim: int,
        probe_hidden_dim: int = 256,
        probe_layers: int = 2,
        contrastive_dim: int = 0,
        contrastive_steps: int = 1,
        detach_action_heads: bool = True,
    ):
        super().__init__()
        self.backbone = LlamaModel(config)
        self.sensor_mode = sensor_mode
        self.heading_dim = heading_dim
        self.turn_bins = turn_bins
        self.step_bins = step_bins
        self.input_size = config.input_size
        self.action_dim = int(config.input_size - sensor_dim)
        if self.action_dim <= 0:
            raise ValueError("action_dim inferred from config.input_size - sensor_dim must be > 0")
        self.probe_hidden_dim = int(probe_hidden_dim)
        self.probe_layers = int(probe_layers)
        self.contrastive_dim = int(contrastive_dim)
        self.contrastive_steps = int(contrastive_steps)
        self.detach_action_heads = bool(detach_action_heads)
        if self.probe_hidden_dim < 0:
            raise ValueError("probe_hidden_dim must be >= 0")
        if self.probe_layers < 1:
            raise ValueError("probe_layers must be >= 1")
        if self.contrastive_dim < 0:
            raise ValueError("contrastive_dim must be >= 0")
        if self.contrastive_steps < 1:
            raise ValueError("contrastive_steps must be >= 1")
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
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, obs_latent_dim),
            nn.ReLU(),
            nn.Linear(obs_latent_dim, obs_latent_dim),
        )
        self.obs_fuse = nn.Linear(config.hidden_size + obs_latent_dim, config.hidden_size)
        self.turn_head = nn.Linear(config.hidden_size, turn_bins)
        self.step_head = nn.Linear(config.hidden_size, step_bins)
        self.contrastive_target_head: Optional[nn.Module] = None
        self.contrastive_action_heads: Optional[nn.ModuleList] = None
        if self.contrastive_dim > 0:
            self.contrastive_target_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, self.contrastive_dim),
            )
            self.contrastive_action_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.hidden_size + h * self.action_dim, config.hidden_size),
                        nn.ReLU(),
                        nn.Linear(config.hidden_size, self.contrastive_dim),
                    )
                    for h in range(1, self.contrastive_steps + 1)
                ]
            )
        self._num_envs: Optional[int] = None
        self._cache_position: Optional[torch.Tensor] = None
        self._cache = None
        self.attention_window = config.attention_window

    def init_cache(self, num_envs: int, device: torch.device):
        self._num_envs = int(num_envs)
        if self.attention_window is not None and self.attention_window > 0:
            self._cache = WindowedPositionBasedDynamicCache(self.attention_window).to(device=device)
        else:
            self._cache = PositionBasedDynamicCache().to(device=device)
        self._cache_position = torch.zeros(self._num_envs, dtype=torch.long, device=device)

    def reset_cache(self, reset_mask: torch.Tensor):
        if self._cache_position is None:
            return
        reset_mask = reset_mask.to(torch.bool).view(-1)
        if reset_mask.numel() != self._cache_position.numel():
            raise ValueError("reset_mask size must match num_envs for cache reset")
        self._cache.reset(reset_mask)
        self._cache_position[reset_mask] = 0

    def clear_cache(self):
        self._num_envs = None
        self._cache_position = None
        self._cache = None

    def _project_contrastive_target_h(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.contrastive_target_head is None:
            return None
        return self.contrastive_target_head(h.detach())

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

    def _forward_core(
        self,
        obs,
        *,
        episode_start=None,
        need_aux: bool = False,
    ):
        sensor, actions, prev_actions, key_padding_mask, _ = self._validate_obs_contract(
            obs,
            episode_start=episode_start,
        )

        using_internal_cache = episode_start is not None
        if using_internal_cache:
            episode_start_t = torch.as_tensor(episode_start, dtype=torch.bool, device=sensor.device).view(-1)
            if episode_start_t.numel() != sensor.shape[0]:
                raise ValueError(
                    f"episode_start batch mismatch: expected {sensor.shape[0]}, got {episode_start_t.numel()}"
                )
            if self._cache is None or self._cache_position is None or self._cache_position.numel() != episode_start_t.numel():
                self.init_cache(num_envs=episode_start_t.numel(), device=sensor.device)
            else:
                self.reset_cache(episode_start_t)

        x = torch.cat([sensor, prev_actions], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        h = self.backbone(
            x,
            past_key_values=self._cache if using_internal_cache else None,
            cache_position=self._cache_position.unsqueeze(1) if using_internal_cache else None,
            key_padding_mask=key_padding_mask,
            attention_window=self.attention_window,
        )
        if using_internal_cache:
            self._cache_position.add_(1)
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
        action_feat = h_probe if self.detach_action_heads else h
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        turn = self.turn_head(action_feat)
        step = self.step_head(action_feat)
        aux_inputs = None
        if need_aux:
            aux_inputs = {}
            aux_inputs["contrastive_tgt_emb"] = self._project_contrastive_target_h(h)
            aux_inputs["contrastive_pred_emb_steps"] = self._project_contrastive_pred_steps(h, actions)

        preds = (pred_sensor, loc_x, loc_y, heading, turn, step)
        return preds, aux_inputs, h[:, -1, :]

    def forward(
        self,
        obs,
        return_state=False,
        episode_start=None,
    ):
        need_aux = episode_start is None
        preds, aux_inputs, last_state = self._forward_core(
            obs,
            episode_start=episode_start,
            need_aux=need_aux,
        )
        return {
            "preds": preds,
            "aux": aux_inputs if need_aux else None,
            "state": last_state if return_state else None,
        }

    def compute_all_losses(self, **kwargs) -> Dict[str, torch.Tensor]:
        return super().compute_all_losses(aux_total=None, **kwargs)


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
        obs_dim: int,
        obs_latent_dim: int,
        probe_hidden_dim: int = 256,
        probe_layers: int = 2,
        state_norm: str = "none",
        contrastive_dim: int = 0,
        contrastive_steps: int = 1,
        detach_action_heads: bool = True,
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
            obs_dim=obs_dim,
            obs_latent_dim=obs_latent_dim,
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
            detach_action_heads=detach_action_heads,
        )
        self.state_norm = str(state_norm)
        if self.state_norm not in {"none", "layernorm", "rmsnorm"}:
            raise ValueError("state_norm must be one of {'none', 'layernorm', 'rmsnorm'}")
        if self.state_norm == "layernorm":
            self.state_out_norm = nn.LayerNorm(config.hidden_size)
        elif self.state_norm == "rmsnorm":
            self.state_out_norm = LlamaRMSNorm(config.hidden_size)
        else:
            self.state_out_norm = nn.Identity()
        self.backbone = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
        )
        self._state_h: Optional[torch.Tensor] = None

    def init_cache(self, num_envs: int, device: torch.device):
        self._num_envs = int(num_envs)
        self._state_h = torch.zeros(
            (self.backbone.num_layers, self._num_envs, self.backbone.hidden_size),
            device=device,
            dtype=self.obs_fuse.weight.dtype,
        )

    def reset_cache(self, reset_mask: torch.Tensor):
        if self._state_h is None:
            return
        reset_mask = reset_mask.to(torch.bool).view(-1)
        if reset_mask.numel() != self._state_h.shape[1]:
            raise ValueError("reset_mask size must match num_envs for cache reset")
        self._state_h[:, reset_mask, :] = 0.0

    def clear_cache(self):
        self._num_envs = None
        self._state_h = None

    def _forward_core(
        self,
        obs,
        *,
        episode_start=None,
        need_aux: bool = False,
    ):
        sensor, actions, prev_actions, _key_padding_mask, _ = self._validate_obs_contract(
            obs,
            episode_start=episode_start,
        )
        using_internal_state = episode_start is not None
        if using_internal_state:
            episode_start_t = torch.as_tensor(episode_start, dtype=torch.bool, device=sensor.device).view(-1)
            if episode_start_t.numel() != sensor.shape[0]:
                raise ValueError(
                    f"episode_start batch mismatch: expected {sensor.shape[0]}, got {episode_start_t.numel()}"
                )
            if self._state_h is None or self._state_h.shape[1] != episode_start_t.numel():
                self.init_cache(num_envs=episode_start_t.numel(), device=sensor.device)
            else:
                self.reset_cache(episode_start_t)
            h0 = self._state_h.to(device=sensor.device, dtype=sensor.dtype)
        else:
            h0 = None

        x = torch.cat([sensor, prev_actions], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        if self.attention_window is not None and self.attention_window > 0 and episode_start is None:
            h, h_n = self.window_iterate(x, h0)
        else:
            h, h_n = self.backbone(x, h0)
        if using_internal_state:
            self._state_h = h_n.detach()
        h = self.state_out_norm(h)
        action_latent = self.action_encoder(actions)
        obs_feat = torch.tanh(self.obs_fuse(torch.cat([h, action_latent], dim=-1)))
        if self.sensor_mode != "categorical":
            raise ValueError("RNNPredictor currently supports sensor_mode='categorical' only.")
        pred_sensor = (
            self.sensor_head_l(obs_feat),
            self.sensor_head_f(obs_feat),
            self.sensor_head_r(obs_feat),
        )
        # State probes (current-step location/heading) read detached state.
        h_probe = h.detach()
        # Action heads use detached or live state based on constructor setting.
        action_feat = h_probe if self.detach_action_heads else h
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        turn = self.turn_head(action_feat)
        step = self.step_head(action_feat)
        aux_inputs = None
        if need_aux:
            aux_inputs = {}
            aux_inputs["contrastive_tgt_emb"] = self._project_contrastive_target_h(h)
            aux_inputs["contrastive_pred_emb_steps"] = self._project_contrastive_pred_steps(h, actions)

        preds = (pred_sensor, loc_x, loc_y, heading, turn, step)
        return preds, aux_inputs, h[:, -1, :]

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
