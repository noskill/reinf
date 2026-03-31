#!/usr/bin/env python3
"""Baseline predictor backbones (transformer and recurrent)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from base import PredictionLossMixin
from transformer import LlamaConfig, LlamaMLP, LlamaModel, LlamaRMSNorm
from utils import make_probe_head
class UnifiedPredictor(PredictionLossMixin, nn.Module):
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
    ):
        super().__init__()
        self.backbone = LlamaModel(config)
        self.sensor_mode = sensor_mode
        self.heading_dim = heading_dim
        self.turn_bins = turn_bins
        self.step_bins = step_bins
        self.input_size = config.input_size
        self.probe_hidden_dim = int(probe_hidden_dim)
        self.probe_layers = int(probe_layers)
        self.contrastive_dim = int(contrastive_dim)
        if self.probe_hidden_dim < 0:
            raise ValueError("probe_hidden_dim must be >= 0")
        if self.probe_layers < 1:
            raise ValueError("probe_layers must be >= 1")
        if self.contrastive_dim < 0:
            raise ValueError("contrastive_dim must be >= 0")
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
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_latent_dim),
            nn.ReLU(),
            nn.Linear(obs_latent_dim, obs_latent_dim),
        )
        self.fuse = nn.Linear(config.hidden_size + obs_latent_dim, config.hidden_size)
        self.turn_head = nn.Linear(config.hidden_size, turn_bins)
        self.step_head = nn.Linear(config.hidden_size, step_bins)
        self.contrastive_head: Optional[nn.Module] = None
        if self.contrastive_dim > 0:
            self.contrastive_head = nn.Linear(config.hidden_size, self.contrastive_dim)

    def forward(
        self,
        obs,
        attention_window=None,
        past_key_values=None,
        cache_position=None,
        prev_hidden=None,
        return_state=False,
        return_aux=False,
    ):
        sensor = obs["sensor"]
        actions = obs["actions"]
        key_padding_mask = obs.get("key_padding_mask")

        x = torch.cat([sensor, actions], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        h = self.backbone(
            x,
            past_key_values=past_key_values,
            cache_position=cache_position,
            key_padding_mask=key_padding_mask,
            attention_window=attention_window,
        )
        if self.sensor_mode != "categorical":
            raise ValueError("UnifiedPredictor currently supports sensor_mode='categorical' only.")
        pred_sensor = (
            self.sensor_head_l(h),
            self.sensor_head_f(h),
            self.sensor_head_r(h),
        )
        # Probe heads should not shape backbone dynamics.
        h_probe = h.detach()
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        z = self.obs_encoder(sensor)
        # For chunked training, the first action in a chunk uses previous chunk state.
        if prev_hidden is None:
            prev_hidden = torch.zeros((h.shape[0], 1, h.shape[-1]), device=h.device, dtype=h.dtype)
        elif prev_hidden.dim() == 2:
            prev_hidden = prev_hidden.unsqueeze(1)
        h_prev = torch.cat([prev_hidden, h[:, :-1, :]], dim=1)
        # Action heads are probes: detach inputs so they do not shape backbone dynamics.
        fused = torch.tanh(self.fuse(torch.cat([h_prev.detach(), z.detach()], dim=-1)))
        turn = self.turn_head(fused)
        step = self.step_head(fused)
        aux_inputs = None
        if return_aux:
            aux_inputs = {}
            if self.contrastive_head is not None:
                emb = self.contrastive_head(h)
                aux_inputs["contrastive_pred_emb"] = emb
                aux_inputs["contrastive_tgt_emb"] = emb
        if return_state:
            if return_aux:
                return pred_sensor, loc_x, loc_y, heading, turn, step, h[:, -1, :], aux_inputs
            return pred_sensor, loc_x, loc_y, heading, turn, step, h[:, -1, :]
        if return_aux:
            return pred_sensor, loc_x, loc_y, heading, turn, step, aux_inputs
        return pred_sensor, loc_x, loc_y, heading, turn, step

    def compute_all_losses(self, **kwargs) -> Dict[str, torch.Tensor]:
        return super().compute_all_losses(aux_total=None, **kwargs)


class RNNPredictor(UnifiedPredictor):
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
        transition: str = "gru",
        residual_scale: float = 1.0,
        state_norm: str = "none",
        contrastive_dim: int = 0,
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
        )
        self.transition = str(transition)
        self.residual_scale = float(residual_scale)
        self.state_norm = str(state_norm)
        if self.transition not in {"gru", "residual", "residual_mlp"}:
            raise ValueError("transition must be one of {'gru', 'residual', 'residual_mlp'}")
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
        self.residual_mlp_pre_norm: Optional[nn.Module] = None
        self.residual_mlp_post_norm: Optional[nn.Module] = None
        self.residual_mlp: Optional[nn.Module] = None
        if self.transition == "residual_mlp":
            # LLaMA-style FFN residual block on recurrent state with extra post norm:
            # h <- RMSNorm(h), h <- h + MLP(h), h <- RMSNorm(h).
            self.residual_mlp_pre_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.residual_mlp_post_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.residual_mlp = LlamaMLP(config)

    def forward(
        self,
        obs,
        attention_window=None,
        past_key_values=None,
        cache_position=None,
        prev_hidden=None,
        return_state=False,
        return_aux=False,
    ):
        del attention_window, past_key_values, cache_position
        sensor = obs["sensor"]
        actions = obs["actions"]

        x = torch.cat([sensor, actions], dim=-1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[-1]}")

        if self.transition in {"residual", "residual_mlp"}:
            # True recurrent residual update on top-layer state:
            # g_t = GRU(x_t, h_{t-1}), h_t = h_{t-1} + a * g_t.
            B, T, _ = x.shape
            nl = int(self.backbone.num_layers)
            hs = int(self.backbone.hidden_size)
            h_layers = x.new_zeros((nl, B, hs))
            h_steps: List[torch.Tensor] = []
            for t in range(T):
                g_seq, g_layers = self.backbone(x[:, t : t + 1, :], h_layers)
                g_t = g_seq[:, 0, :]
                h_prev_top = h_layers[-1]
                h_top = h_prev_top + self.residual_scale * g_t
                if self.transition == "residual_mlp":
                    h_mlp_in = self.residual_mlp_pre_norm(h_top)
                    mlp_delta = self.residual_mlp(h_mlp_in)
                    h_top = h_top + mlp_delta
                    h_top = self.residual_mlp_post_norm(h_top)
                h_layers = g_layers
                h_layers[-1] = h_top
                h_steps.append(h_top)
            h = torch.stack(h_steps, dim=1) if h_steps else x.new_zeros((B, 0, hs))
        else:
            h, _ = self.backbone(x)
        h = self.state_out_norm(h)
        if self.sensor_mode != "categorical":
            raise ValueError("RNNPredictor currently supports sensor_mode='categorical' only.")
        pred_sensor = (
            self.sensor_head_l(h),
            self.sensor_head_f(h),
            self.sensor_head_r(h),
        )
        # Probe heads should not shape backbone dynamics.
        h_probe = h.detach()
        loc_x = self.loc_x_head(h_probe)
        loc_y = self.loc_y_head(h_probe)
        heading = self.heading_head(h_probe)
        z = self.obs_encoder(sensor)
        if prev_hidden is None:
            prev_hidden = torch.zeros((h.shape[0], 1, h.shape[-1]), device=h.device, dtype=h.dtype)
        elif prev_hidden.dim() == 2:
            prev_hidden = prev_hidden.unsqueeze(1)
        h_prev = torch.cat([prev_hidden, h[:, :-1, :]], dim=1)
        # Action heads are probes: detach inputs so they do not shape backbone dynamics.
        fused = torch.tanh(self.fuse(torch.cat([h_prev.detach(), z.detach()], dim=-1)))
        turn = self.turn_head(fused)
        step = self.step_head(fused)
        aux_inputs = None
        if return_aux:
            aux_inputs = {}
            if self.contrastive_head is not None:
                emb = self.contrastive_head(h)
                aux_inputs["contrastive_pred_emb"] = emb
                aux_inputs["contrastive_tgt_emb"] = emb
        if return_state:
            if return_aux:
                return pred_sensor, loc_x, loc_y, heading, turn, step, h[:, -1, :], aux_inputs
            return pred_sensor, loc_x, loc_y, heading, turn, step, h[:, -1, :]
        if return_aux:
            return pred_sensor, loc_x, loc_y, heading, turn, step, aux_inputs
        return pred_sensor, loc_x, loc_y, heading, turn, step
