#!/usr/bin/env python3
"""Base mixins and base latent predictor class."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import (
    batch_row_k_step_contrastive_pair_stats,
    expected_from_logits,
    masked_action_acc,
    masked_coord_acc,
    masked_coord_rmse,
    masked_lr_metrics,
    masked_lr_metrics_logits,
    masked_mse,
    masked_rmse,
    make_probe_head,
    sg,
    soft_cross_entropy,
)


@dataclass
class LossConfig:
    sensor_tables: Optional[Sequence[torch.Tensor]] = None
    sensor_min_idx: Optional[torch.Tensor] = None
    loc_x_table: Optional[torch.Tensor] = None
    loc_y_table: Optional[torch.Tensor] = None
    heading_table: Optional[torch.Tensor] = None
    sensor_weight: float = 1.0
    loc_weight: float = 1.0
    head_weight: float = 1.0
    turn_weight: float = 1.0
    step_weight: float = 1.0
    contrastive_weight: float = 0.0
    contrastive_temp: float = 0.1
    contrastive_horizon_discount: float = 1.0
    contrastive_negatives: int = 0
    loc_min: Optional[torch.Tensor] = None


class PredictionLossMixin:
    @staticmethod
    def _shift_prev_actions(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B, T, A], got {tuple(actions.shape)}")
        zeros = actions.new_zeros((actions.shape[0], 1, actions.shape[2]))
        return torch.cat([zeros, actions[:, :-1, :]], dim=1)

    def _validate_obs_contract(self, obs, *, episode_start):
        if not isinstance(obs, dict):
            raise ValueError(f"{self.__class__.__name__} expects dict observation")
        rollout = episode_start is not None

        required = ["sensor", "prev_actions"] if rollout else ["sensor", "actions"]
        missing = [k for k in required if k not in obs]
        if missing:
            mode = "rollout" if rollout else "train"
            raise ValueError(f"{self.__class__.__name__} {mode} obs missing keys: {missing}")

        sensor = obs["sensor"]
        actions = obs.get("actions")
        prev_actions = obs.get("prev_actions")
        key_padding_mask = obs.get("key_padding_mask")
        if rollout and actions is None:
            actions = prev_actions

        if not isinstance(sensor, torch.Tensor):
            sensor = torch.as_tensor(sensor)
        if actions is None:
            raise ValueError("actions must be provided (or inferred from prev_actions in rollout)")
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=sensor.device)
        if sensor.ndim == 2:
            sensor = sensor.unsqueeze(1)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if sensor.ndim != 3:
            raise ValueError(f"Expected sensor [B,T,S] or [B,S], got {tuple(sensor.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,T,A] or [B,A], got {tuple(actions.shape)}")
        if sensor.shape[:2] != actions.shape[:2]:
            raise ValueError(f"sensor/actions shape mismatch: {tuple(sensor.shape)} vs {tuple(actions.shape)}")

        expected_action_dim = getattr(self, "action_dim", None)
        if expected_action_dim is not None and int(actions.shape[-1]) != int(expected_action_dim):
            raise ValueError(
                f"Expected action dim {int(expected_action_dim)}, got {int(actions.shape[-1])}"
            )

        if key_padding_mask is not None:
            if not isinstance(key_padding_mask, torch.Tensor):
                key_padding_mask = torch.as_tensor(key_padding_mask, device=sensor.device)
            if key_padding_mask.ndim == 1:
                key_padding_mask = key_padding_mask.unsqueeze(1)
            if key_padding_mask.ndim != 2 or key_padding_mask.shape != sensor.shape[:2]:
                raise ValueError(
                    f"key_padding_mask must be [B,T]={tuple(sensor.shape[:2])}, got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(torch.bool)

        if prev_actions is not None:
            if not isinstance(prev_actions, torch.Tensor):
                prev_actions = torch.as_tensor(prev_actions, device=sensor.device)
            if prev_actions.ndim == 2:
                prev_actions = prev_actions.unsqueeze(1)
            if prev_actions.ndim != 3:
                raise ValueError(f"Expected prev_actions [B,T,A] or [B,A], got {tuple(prev_actions.shape)}")
            if prev_actions.shape != actions.shape:
                raise ValueError(
                    f"prev_actions shape must match actions: {tuple(prev_actions.shape)} vs {tuple(actions.shape)}"
                )

        if rollout:
            if key_padding_mask is not None:
                raise ValueError("rollout obs must not include key_padding_mask")
            if sensor.shape[1] != 1:
                raise ValueError("rollout obs expects single-step [B,S] or [B,1,S]")
        if prev_actions is None:
            prev_actions = self._shift_prev_actions(actions)

        return (
            sensor.to(torch.float32),
            actions.to(torch.float32),
            prev_actions.to(torch.float32),
            key_padding_mask,
            rollout,
        )

    def compute_soft_target_loss(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Soft-target losses treat discrete targets as smoothed distributions (label smoothing / Gaussian kernels).
        cfg = self.config
        pred_l, pred_f, pred_r = pred_sensor
        idx = y_sensor_idx.clamp(min=0)
        if cfg.sensor_min_idx is not None:
            idx = (idx - cfg.sensor_min_idx.view(1, 1, -1)).clamp(min=0)
        # sensor_tables maps each index to a softened target distribution; we use it for soft CE.
        loss_sensor = (
            soft_cross_entropy(pred_l, cfg.sensor_tables[0][idx[..., 0]], key_padding_mask)
            + soft_cross_entropy(pred_f, cfg.sensor_tables[1][idx[..., 1]], key_padding_mask)
            + soft_cross_entropy(pred_r, cfg.sensor_tables[2][idx[..., 2]], key_padding_mask)
        )

        # Convert target coords to valid bin indices (padding uses -1, so clamp to 0 for table lookup).
        loc_idx = y_loc_xy.clamp(min=0)
        loss_loc_x = soft_cross_entropy(pred_loc_x, cfg.loc_x_table[loc_idx[..., 0]], key_padding_mask)
        loss_loc_y = soft_cross_entropy(pred_loc_y, cfg.loc_y_table[loc_idx[..., 1]], key_padding_mask)

        head_idx = y_head.clamp(min=0)
        loss_head = soft_cross_entropy(pred_head, cfg.heading_table[head_idx], key_padding_mask)
        return {
            "sensor": loss_sensor,
            "loc_x": loss_loc_x,
            "loc_y": loss_loc_y,
            "head": loss_head,
        }

    def compute_observation_losses(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        key_padding_mask: torch.Tensor,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        del aux_inputs
        return self.compute_soft_target_loss(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_head=y_head,
            key_padding_mask=key_padding_mask,
        )

    def compute_action_losses(
        self,
        *,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_turn = F.cross_entropy(
            pred_turn.view(-1, pred_turn.shape[-1]),
            y_turn.view(-1),
            ignore_index=-100,
        )
        loss_step = F.cross_entropy(
            pred_step.view(-1, pred_step.shape[-1]),
            y_step.view(-1),
            ignore_index=-100,
        )
        return {
            "turn": loss_turn,
            "step": loss_step,
        }

    @staticmethod
    def compute_contrastive_loss(
        aux_inputs: Optional[Dict[str, torch.Tensor]],
        key_padding_mask: torch.Tensor,
        temperature: float,
        horizon_discount: float = 1.0,
        max_negatives: int = 0,
    ) -> Dict[str, torch.Tensor]:
        B, T = key_padding_mask.shape
        device = key_padding_mask.device

        def _zero_stats() -> Dict[str, torch.Tensor]:
            return {
                "loss": torch.tensor(0.0, device=device),
                "acc": torch.tensor(0.0, device=device),
                "per_step_loss": torch.zeros((B, T), device=device),
                "per_step_acc": torch.zeros((B, T), device=device),
                "per_step_weight": torch.zeros((B, T), device=device),
                "per_step_valid": torch.zeros((B, T), device=device, dtype=torch.bool),
                "per_horizon_loss": torch.zeros((0,), device=device),
                "per_horizon_acc": torch.zeros((0,), device=device),
                "per_horizon_valid": torch.zeros((0,), device=device, dtype=torch.bool),
            }

        if aux_inputs is None:
            return _zero_stats()
        pred_steps = aux_inputs.get("contrastive_pred_emb_steps")
        tgt_emb = aux_inputs.get("contrastive_tgt_emb")
        if pred_steps is None or tgt_emb is None:
            return _zero_stats()
        return batch_row_k_step_contrastive_pair_stats(
            pred_steps=pred_steps,
            target_emb=tgt_emb,
            key_padding_mask=key_padding_mask,
            temperature=temperature,
            horizon_discount=horizon_discount,
            max_negatives=max_negatives,
        )

    def compute_losses_and_metrics(
        self,
        *,
        preds: Tuple,
        targets: Dict[str, torch.Tensor],
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        losses = self.compute_losses(
            preds=preds,
            targets=targets,
            aux_inputs=aux_inputs,
        )
        metrics = self.compute_metrics(
            preds=preds,
            targets=targets,
            aux_inputs=aux_inputs,
        )
        return losses, metrics

    def compute_losses(
        self,
        *,
        preds: Tuple,
        targets: Dict[str, torch.Tensor],
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        pred_sensor, pred_loc_x, pred_loc_y, pred_head, pred_turn, pred_step = preds
        losses = self.compute_prediction_losses(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_sensor=targets["y_sensor"],
            y_sensor_idx=targets["y_sensor_idx"],
            y_loc_xy=targets["y_loc_xy"],
            y_head=targets["y_head"],
            y_turn=targets["y_turn"],
            y_step=targets["y_step"],
            key_padding_mask=targets["key_padding_mask"],
            aux_inputs=aux_inputs,
        )
        contrastive_results = self.compute_contrastive_loss(
            aux_inputs=aux_inputs,
            key_padding_mask=targets["key_padding_mask"],
            temperature=cfg.contrastive_temp,
            horizon_discount=cfg.contrastive_horizon_discount,
            max_negatives=cfg.contrastive_negatives,
        )
        losses["contrastive"] = contrastive_results["loss"]
        return losses

    def compute_metrics(
        self,
        *,
        preds: Tuple,
        targets: Dict[str, torch.Tensor],
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        pred_sensor, pred_loc_x, pred_loc_y, _pred_head, pred_turn, pred_step = preds
        metrics = self.compute_predictions_metrics(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_sensor=targets["y_sensor"],
            y_sensor_idx=targets["y_sensor_idx"],
            y_loc_xy=targets["y_loc_xy"],
            y_turn=targets["y_turn"],
            y_step=targets["y_step"],
            key_padding_mask=targets["key_padding_mask"],
            sensor_min_idx=cfg.sensor_min_idx,
            loc_min=cfg.loc_min,
            aux_inputs=aux_inputs,
        )
        contrastive_stats = self.compute_contrastive_loss(
            aux_inputs=aux_inputs,
            key_padding_mask=targets["key_padding_mask"],
            temperature=cfg.contrastive_temp,
            horizon_discount=cfg.contrastive_horizon_discount,
            max_negatives=cfg.contrastive_negatives,
        )
        metrics["contrastive_acc"] = contrastive_stats["acc"]
        return metrics

    def compute_predictions_metrics(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_min_idx,
        loc_min: Optional[torch.Tensor],
        loc_max: Optional[torch.Tensor] = None,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        del aux_inputs, loc_max
        pred_l, pred_f, pred_r = pred_sensor
        exp_l = expected_from_logits(pred_l, float(sensor_min_idx[0].item()))
        exp_f = expected_from_logits(pred_f, float(sensor_min_idx[1].item()))
        exp_r = expected_from_logits(pred_r, float(sensor_min_idx[2].item()))
        pred_sensor_exp = torch.stack([exp_l, exp_f, exp_r], dim=-1)
        target_sensor_cont = y_sensor_idx.to(pred_sensor_exp.dtype)
        mse = masked_mse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
        rmse = masked_rmse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
        if sensor_min_idx is not None:
            lr_rmse, lr_acc = masked_lr_metrics_logits(
                pred_l, pred_r, y_sensor_idx, key_padding_mask, sensor_min_idx.to(torch.float32)
            )
        else:
            lr_rmse, lr_acc = masked_lr_metrics_logits(
                pred_l, pred_r, y_sensor_idx, key_padding_mask, torch.zeros(3, device=pred_l.device)
            )

        if loc_min is not None:
            loc_x_rmse = masked_coord_rmse(pred_loc_x, y_loc_xy[..., 0], key_padding_mask, loc_min[0])
            loc_y_rmse = masked_coord_rmse(pred_loc_y, y_loc_xy[..., 1], key_padding_mask, loc_min[1])
            loc_x_acc = masked_coord_acc(pred_loc_x, y_loc_xy[..., 0], key_padding_mask)
            loc_y_acc = masked_coord_acc(pred_loc_y, y_loc_xy[..., 1], key_padding_mask)
        else:
            zero = torch.tensor(0.0, device=pred_loc_x.device)
            loc_x_rmse = zero
            loc_y_rmse = zero
            loc_x_acc = zero
            loc_y_acc = zero

        turn_acc = masked_action_acc(pred_turn, y_turn, key_padding_mask)
        step_acc = masked_action_acc(pred_step, y_step, key_padding_mask)

        return {
            "mse": mse,
            "rmse": rmse,
            "lr_rmse": lr_rmse,
            "lr_acc": lr_acc,
            "loc_x_rmse": loc_x_rmse,
            "loc_y_rmse": loc_y_rmse,
            "loc_x_acc": loc_x_acc,
            "loc_y_acc": loc_y_acc,
            "turn_acc": turn_acc,
            "step_acc": step_acc,
        }

    def compute_prediction_losses(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_total: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_losses = self.compute_observation_losses(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_head=y_head,
            key_padding_mask=key_padding_mask,
            aux_inputs=aux_inputs,
        )
        action_losses = self.compute_action_losses(
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_turn=y_turn,
            y_step=y_step,
        )
        if aux_total is None:
            aux_total = torch.tensor(0.0, device=pred_loc_x.device)
        return {
            "aux_total": aux_total,
            **obs_losses,
            **action_losses,
        }


class DiscreteLatentPredictorBase(PredictionLossMixin, nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
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
        action_dim: int,
        stoch_size: int,
        stoch_classes: int,
        stoch_temp: float,
        kl_dyn_beta: float,
        kl_rep_beta: float,
        kl_free_nats: float,
        recon_beta: float,
        obs_loss_mode: str,
        prior_rollout_weight: float,
        bptt_horizon: int,
        z_only_weight: float,
        h_only_weight: float,
        prior_rollout_steps: int = 0,
        probe_hidden_dim: int = 256,
        probe_layers: int = 2,
        contrastive_dim: int = 0,
        contrastive_steps: int = 1,
        detach_action_heads: bool = True,
    ):
        super().__init__()
        self.sensor_mode = sensor_mode
        self.sensor_dim = int(sensor_dim)
        self.sensor_bins = sensor_bins
        self.heading_dim = heading_dim
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.stoch_size = int(stoch_size)
        self.stoch_classes = int(stoch_classes)
        self.stoch_flat = self.stoch_size * self.stoch_classes
        self.stoch_temp = float(stoch_temp)
        self.kl_dyn_beta = float(kl_dyn_beta)
        self.kl_rep_beta = float(kl_rep_beta)
        self.kl_free_nats = float(kl_free_nats)
        self.recon_beta = float(recon_beta)
        self.prior_rollout_weight = float(prior_rollout_weight)
        self.bptt_horizon = int(bptt_horizon)
        self.z_only_weight = float(z_only_weight)
        self.h_only_weight = float(h_only_weight)
        self.prior_rollout_steps = int(prior_rollout_steps)
        self.probe_hidden_dim = int(probe_hidden_dim)
        self.probe_layers = int(probe_layers)
        self.contrastive_dim = int(contrastive_dim)
        self.contrastive_steps = int(contrastive_steps)
        self.detach_action_heads = bool(detach_action_heads)
        if self.prior_rollout_steps < 0:
            raise ValueError("prior_rollout_steps must be >= 0")
        if self.probe_hidden_dim < 0:
            raise ValueError("probe_hidden_dim must be >= 0")
        if self.probe_layers < 1:
            raise ValueError("probe_layers must be >= 1")
        if self.contrastive_dim < 0:
            raise ValueError("contrastive_dim must be >= 0")
        if self.contrastive_steps < 1:
            raise ValueError("contrastive_steps must be >= 1")
        if obs_loss_mode not in {"soft", "recon"}:
            raise ValueError(f"obs_loss_mode must be 'soft' or 'recon', got {obs_loss_mode}")
        self.obs_loss_mode = obs_loss_mode
        self.loc_x_bins = int(loc_x_bins)
        self.loc_y_bins = int(loc_y_bins)

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_latent_dim),
            nn.ReLU(),
            nn.Linear(obs_latent_dim, obs_latent_dim),
            nn.ReLU(),
        )
        self.prior_head = nn.Linear(self.hidden_size, self.stoch_flat)
        self.post_head = nn.Linear(self.hidden_size + obs_latent_dim, self.stoch_flat)

        feat_dim = self.hidden_size + self.stoch_flat
        self.z_obs_head: Optional[nn.Module] = None
        self.h_obs_head: Optional[nn.Module] = None
        if self.obs_loss_mode == "soft":
            if sensor_mode == "categorical":
                assert sensor_bins is not None and len(sensor_bins) == 3
                sensor_out_dim = int(sensor_bins[0] + sensor_bins[1] + sensor_bins[2])
            else:
                sensor_out_dim = self.sensor_dim
            # Sensor prediction head only; location/heading are probe heads.
            self.obs_head = nn.Linear(feat_dim, sensor_out_dim)
            # Optional z-only sensor head for auxiliary loss.
            self.z_obs_head = nn.Linear(self.stoch_flat, sensor_out_dim)
            # Optional h-only sensor head for auxiliary loss.
            self.h_obs_head = nn.Linear(self.hidden_size, sensor_out_dim)
        else:
            # Recon mode decoder outputs continuous sensor observation.
            self.obs_head = nn.Sequential(
                nn.Linear(feat_dim, obs_latent_dim),
                nn.ReLU(),
                nn.Linear(obs_latent_dim, obs_dim),
            )
        # Probe heads for location/heading: trained from detached state so they don't shape dynamics.
        self.loc_probe_x = make_probe_head(feat_dim, self.loc_x_bins, self.probe_hidden_dim, self.probe_layers)
        self.loc_probe_y = make_probe_head(feat_dim, self.loc_y_bins, self.probe_hidden_dim, self.probe_layers)
        self.head_probe = make_probe_head(feat_dim, self.heading_dim, self.probe_hidden_dim, self.probe_layers)
        self.turn_head = nn.Linear(feat_dim, turn_bins)
        self.step_head = nn.Linear(feat_dim, step_bins)
        self.contrastive_head: Optional[nn.Module] = None
        self.contrastive_action_heads: Optional[nn.ModuleList] = None
        self.z_contrastive_head: Optional[nn.Module] = None
        if self.contrastive_dim > 0:
            # Predictor branches (one-step and action-conditioned) use 2-layer MLPs.
            self.contrastive_head = nn.Sequential(
                nn.Linear(feat_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.contrastive_dim),
            )
            # Twister-style action-conditioned predictors for horizons 1..K.
            self.contrastive_action_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feat_dim + h * self.action_dim, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, self.contrastive_dim),
                    )
                    for h in range(1, self.contrastive_steps + 1)
                ]
            )
            # Target branch for posterior stochastic latent z_t: 2-layer MLP.
            self.z_contrastive_head = nn.Sequential(
                nn.Linear(self.stoch_flat, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.contrastive_dim),
            )

    def _sample_stoch(self, logits: torch.Tensor, training: bool) -> torch.Tensor:
        # logits: [B, T, S, C]
        if training:
            flat = logits.reshape(-1, self.stoch_classes)
            z = F.gumbel_softmax(flat, tau=self.stoch_temp, hard=True, dim=-1)
            return z.reshape(*logits.shape)
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.stoch_classes).to(dtype=probs.dtype)

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean()
        valid = ~mask
        if valid.sum() == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x[valid].mean()

    def _compute_kl_losses(
        self,
        prior_logits: torch.Tensor,  # [B, T, S, C]
        post_logits: torch.Tensor,   # [B, T, S, C]
        key_padding_mask: Optional[torch.Tensor],  # [B, T], True for PAD
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L_dyn - dynamic prediction loss
        eq 4 in twister paper
        """
        # prior - hat z distribution
        log_p = F.log_softmax(prior_logits, dim=-1)
        # posterior
        log_q = F.log_softmax(post_logits, dim=-1)

        q = log_q.exp()

        # make prior match posterior
        kl_dyn = (sg(q) * (sg(log_q) - log_p)).sum(dim=-1).sum(dim=-1)  # [B, T]

        # make obs -> q representation more predictable by matching prior
        kl_rep = (q * (log_q - sg(log_p))).sum(dim=-1).sum(dim=-1)  # [B, T]

        if self.kl_free_nats > 0:
            # same as max(KL, 1)
            kl_dyn = torch.clamp(kl_dyn - self.kl_free_nats, min=0)
            kl_rep = torch.clamp(kl_rep - self.kl_free_nats, min=0)

        dyn_loss = self.kl_dyn_beta * self._masked_mean(kl_dyn, key_padding_mask)
        rep_loss = self.kl_rep_beta * self._masked_mean(kl_rep, key_padding_mask)
        return dyn_loss, rep_loss

    def compute_observation_losses(
        self,
        *,
        pred_sensor=None,
        pred_loc_x: Optional[torch.Tensor] = None,
        pred_loc_y: Optional[torch.Tensor] = None,
        pred_head: Optional[torch.Tensor] = None,
        y_sensor: Optional[torch.Tensor] = None,
        y_sensor_idx: Optional[torch.Tensor] = None,
        y_loc_xy: Optional[torch.Tensor] = None,
        y_head: Optional[torch.Tensor] = None,
        key_padding_mask: torch.Tensor,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        del y_sensor, y_sensor_idx, y_loc_xy, y_head
        cfg = self.config
        if aux_inputs is None or "sensor_target" not in aux_inputs or "loc_target" not in aux_inputs or "head_target" not in aux_inputs:
            raise ValueError("aux_inputs targets are required for observation loss")
        sensor_target = aux_inputs["sensor_target"]
        loc_target = aux_inputs["loc_target"]
        head_target = aux_inputs["head_target"]

        loc_idx = loc_target.round().to(torch.long).clamp(min=0)
        loss_loc_x = soft_cross_entropy(pred_loc_x, cfg.loc_x_table[loc_idx[..., 0]], key_padding_mask)
        loss_loc_y = soft_cross_entropy(pred_loc_y, cfg.loc_y_table[loc_idx[..., 1]], key_padding_mask)
        head_idx = head_target.clamp(min=0)
        loss_head = soft_cross_entropy(pred_head, cfg.heading_table[head_idx], key_padding_mask)

        if self.obs_loss_mode == "recon":
            # Recon mode: single decoder head outputs a continuous obs vector.
            if "obs_hat" not in aux_inputs or "obs_target" not in aux_inputs:
                raise ValueError("aux_inputs['obs_hat'] and aux_inputs['obs_target'] are required for reconstruction loss")
            obs_hat = aux_inputs["obs_hat"]
            obs_target = aux_inputs["obs_target"].to(obs_hat.dtype)
            recon = masked_mse(obs_hat, obs_target, key_padding_mask)
            return {
                "obs_total": self.recon_beta * recon,
                "loc_x": loss_loc_x,
                "loc_y": loss_loc_y,
                "head": loss_head,
            }

        if self.sensor_mode == "categorical":
            pred_l, pred_f, pred_r = pred_sensor
            idx = sensor_target.round().to(torch.long).clamp(min=0)
            if cfg.sensor_min_idx is not None:
                idx = (idx - cfg.sensor_min_idx.view(1, 1, -1)).clamp(min=0)
            loss_sensor = (
                soft_cross_entropy(pred_l, cfg.sensor_tables[0][idx[..., 0]], key_padding_mask)
                + soft_cross_entropy(pred_f, cfg.sensor_tables[1][idx[..., 1]], key_padding_mask)
                + soft_cross_entropy(pred_r, cfg.sensor_tables[2][idx[..., 2]], key_padding_mask)
            )
        else:
            loss_sensor = masked_mse(pred_sensor, sensor_target, key_padding_mask)

        return {
            "sensor": loss_sensor,
            "loc_x": loss_loc_x,
            "loc_y": loss_loc_y,
            "head": loss_head,
        }

    def compute_predictions_metrics(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        sensor_min_idx,
        loc_min: Optional[torch.Tensor],
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        del y_sensor, y_sensor_idx, y_loc_xy
        if aux_inputs is None or "sensor_target" not in aux_inputs or "loc_target" not in aux_inputs or "head_target" not in aux_inputs:
            raise ValueError("aux_inputs targets are required for metrics")

        sensor_target = aux_inputs["sensor_target"]
        loc_target = aux_inputs["loc_target"]

        if self.obs_loss_mode == "recon":
            mse = masked_mse(pred_sensor, sensor_target, key_padding_mask)
            rmse = masked_rmse(pred_sensor, sensor_target, key_padding_mask)
            lr_rmse, lr_acc = masked_lr_metrics(pred_sensor, sensor_target, key_padding_mask)
        else:
            sensor_for_lr = aux_inputs.get("prior_sensor_pred", pred_sensor)
            if self.sensor_mode == "categorical":
                pred_l, pred_f, pred_r = pred_sensor
                lr_l, _lr_f, lr_r = sensor_for_lr
                sensor_target_abs = sensor_target.round().to(torch.long).clamp(min=0)
                idx = sensor_target_abs
                if sensor_min_idx is not None:
                    idx = (idx - sensor_min_idx.view(1, 1, -1)).clamp(min=0)
                    sensor_min_metric = sensor_min_idx.to(torch.float32)
                else:
                    sensor_min_metric = torch.zeros(3, device=pred_l.device, dtype=torch.float32)
                exp_l = expected_from_logits(pred_l, float(sensor_min_metric[0].item()))
                exp_f = expected_from_logits(pred_f, float(sensor_min_metric[1].item()))
                exp_r = expected_from_logits(pred_r, float(sensor_min_metric[2].item()))
                pred_sensor_exp = torch.stack([exp_l, exp_f, exp_r], dim=-1)
                target_sensor_cont = sensor_target_abs.to(pred_sensor_exp.dtype)
                mse = masked_mse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
                rmse = masked_rmse(pred_sensor_exp, target_sensor_cont, key_padding_mask)
                lr_rmse, lr_acc = masked_lr_metrics_logits(
                    lr_l, lr_r, sensor_target_abs, key_padding_mask, sensor_min_metric
                )
            else:
                mse = masked_mse(pred_sensor, sensor_target, key_padding_mask)
                rmse = masked_rmse(pred_sensor, sensor_target, key_padding_mask)
                lr_rmse, lr_acc = masked_lr_metrics(sensor_for_lr, sensor_target, key_padding_mask)

        if loc_min is not None:
            loc_x_idx = loc_target[..., 0].to(torch.long)
            loc_y_idx = loc_target[..., 1].to(torch.long)
            loc_x_rmse = masked_coord_rmse(pred_loc_x, loc_x_idx, key_padding_mask, loc_min[0])
            loc_y_rmse = masked_coord_rmse(pred_loc_y, loc_y_idx, key_padding_mask, loc_min[1])
            loc_x_acc = masked_coord_acc(pred_loc_x, loc_x_idx, key_padding_mask)
            loc_y_acc = masked_coord_acc(pred_loc_y, loc_y_idx, key_padding_mask)
        else:
            zero = torch.tensor(0.0, device=pred_loc_x.device)
            loc_x_rmse = zero
            loc_y_rmse = zero
            loc_x_acc = zero
            loc_y_acc = zero

        turn_acc = masked_action_acc(pred_turn, y_turn, key_padding_mask)
        step_acc = masked_action_acc(pred_step, y_step, key_padding_mask)

        return {
            "mse": mse,
            "rmse": rmse,
            "lr_rmse": lr_rmse,
            "lr_acc": lr_acc,
            "loc_x_rmse": loc_x_rmse,
            "loc_y_rmse": loc_y_rmse,
            "loc_x_acc": loc_x_acc,
            "loc_y_acc": loc_y_acc,
            "turn_acc": turn_acc,
            "step_acc": step_acc,
        }

    def compute_prediction_losses(
        self,
        *,
        pred_sensor,
        pred_loc_x: torch.Tensor,
        pred_loc_y: torch.Tensor,
        pred_head: torch.Tensor,
        pred_turn: torch.Tensor,
        pred_step: torch.Tensor,
        y_sensor: torch.Tensor,
        y_sensor_idx: torch.Tensor,
        y_loc_xy: torch.Tensor,
        y_head: torch.Tensor,
        y_turn: torch.Tensor,
        y_step: torch.Tensor,
        key_padding_mask: torch.Tensor,
        aux_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config

        def compute_sensor_loss(pred_sensor, target_idx, target_raw, mask):
            """Compute sensor loss for either categorical or regression heads."""
            if pred_sensor is None:
                return torch.tensor(0.0, device=pred_loc_x.device)
            if self.sensor_mode == "categorical":
                if cfg.sensor_tables is None:
                    raise ValueError("categorical sensor loss requires cfg.sensor_tables")
                pred_l, pred_f, pred_r = pred_sensor
                idx = target_idx.clamp(min=0)
                if cfg.sensor_min_idx is not None:
                    idx = (idx - cfg.sensor_min_idx.view(1, 1, -1)).clamp(min=0)
                return (
                    soft_cross_entropy(pred_l, cfg.sensor_tables[0][idx[..., 0]], mask)
                    + soft_cross_entropy(pred_f, cfg.sensor_tables[1][idx[..., 1]], mask)
                    + soft_cross_entropy(pred_r, cfg.sensor_tables[2][idx[..., 2]], mask)
                )
            return masked_mse(pred_sensor, target_raw, mask)

        def compute_prior_rollout_sensor_loss(prior_roll_sensor_pred, sensor_target):
            if prior_roll_sensor_pred is None:
                return torch.tensor(0.0, device=pred_loc_x.device)
            if isinstance(prior_roll_sensor_pred, tuple):
                pred_len = int(prior_roll_sensor_pred[0].shape[1])
            else:
                pred_len = int(prior_roll_sensor_pred.shape[1])
            target_raw = sensor_target[:, :pred_len]
            target_idx = target_raw.round().to(torch.long).clamp(min=0)
            mask = key_padding_mask[:, :pred_len]
            return compute_sensor_loss(prior_roll_sensor_pred, target_idx, target_raw, mask)

        def compute_z_only_sensor_loss(z_only_pred, target_raw, target_idx):
            return compute_sensor_loss(z_only_pred, target_idx, target_raw, key_padding_mask)

        def compute_h_only_sensor_loss(h_only_pred, target_raw, target_idx):
            return compute_sensor_loss(h_only_pred, target_idx, target_raw, key_padding_mask)

        aux_losses: Dict[str, torch.Tensor] = {}
        aux_total = torch.tensor(0.0, device=pred_loc_x.device)
        if aux_inputs is not None:
            prior_logits = aux_inputs.get("prior_logits")
            post_logits = aux_inputs.get("post_logits")
            if prior_logits is not None and post_logits is not None:
                kl_dyn, kl_rep = self._compute_kl_losses(prior_logits, post_logits, key_padding_mask)
                aux_losses["kl_dyn"] = kl_dyn
                aux_losses["kl_rep"] = kl_rep
                aux_total = aux_total + kl_dyn + kl_rep
            prior_roll_sensor_pred = aux_inputs.get("prior_roll_sensor_pred")
            sensor_target = aux_inputs.get("sensor_target")
            if self.prior_rollout_weight > 0 and prior_roll_sensor_pred is not None and sensor_target is not None:
                prior_rollout_sensor = compute_prior_rollout_sensor_loss(prior_roll_sensor_pred, sensor_target)
                aux_losses["prior_rollout_sensor"] = prior_rollout_sensor
                aux_total = aux_total + self.prior_rollout_weight * prior_rollout_sensor
            sensor_recon_target = aux_inputs.get("sensor_target", y_sensor)
            sensor_recon_target_idx = sensor_recon_target.round().to(torch.long).clamp(min=0)
            z_only_pred = aux_inputs.get("z_only_pred")
            if self.z_only_weight > 0 and z_only_pred is not None:
                z_only_sensor = compute_z_only_sensor_loss(
                    z_only_pred,
                    sensor_recon_target,
                    sensor_recon_target_idx,
                )
                aux_losses["z_only_sensor"] = z_only_sensor
                aux_total = aux_total + self.z_only_weight * z_only_sensor
            h_only_pred = aux_inputs.get("h_only_pred")
            if self.h_only_weight > 0 and h_only_pred is not None:
                h_only_sensor = compute_h_only_sensor_loss(
                    h_only_pred,
                    sensor_recon_target,
                    sensor_recon_target_idx,
                )
                aux_losses["h_only_sensor"] = h_only_sensor
                aux_total = aux_total + self.h_only_weight * h_only_sensor
        base_losses = super().compute_prediction_losses(
            pred_sensor=pred_sensor,
            pred_loc_x=pred_loc_x,
            pred_loc_y=pred_loc_y,
            pred_head=pred_head,
            pred_turn=pred_turn,
            pred_step=pred_step,
            y_sensor=y_sensor,
            y_sensor_idx=y_sensor_idx,
            y_loc_xy=y_loc_xy,
            y_head=y_head,
            y_turn=y_turn,
            y_step=y_step,
            key_padding_mask=key_padding_mask,
            aux_total=aux_total,
            aux_inputs=aux_inputs,
        )
        if self.obs_loss_mode == "recon":
            recon = base_losses["obs_total"]
            probe_total = cfg.loc_weight * (base_losses["loc_x"] + base_losses["loc_y"]) + cfg.head_weight * base_losses["head"]
            base_losses["obs_total"] = recon + probe_total
            base_losses["recon"] = recon
        return {**base_losses, **aux_losses}

    def _encode_obs(self, obs, *, episode_start=None):
        sensor, actions, prev_actions, key_padding_mask, _ = self._validate_obs_contract(
            obs,
            episode_start=episode_start,
        )
        obs_features = sensor[..., :3]      # use only left/front/right sensors
        B, T, _ = actions.shape
        obs_embed = self.obs_encoder(obs_features)
        return obs_embed, obs_features, actions, prev_actions, key_padding_mask, B, T

    def _split_sensor_logits(self, obs_out: torch.Tensor):
        """Map flattened sensor logits into (left, front, right) heads."""
        if self.sensor_mode == "categorical":
            assert self.sensor_bins is not None
            b0 = int(self.sensor_bins[0])
            b1 = int(self.sensor_bins[1])
            b2 = int(self.sensor_bins[2])
            pred_l = obs_out[..., :b0]
            pred_f = obs_out[..., b0 : b0 + b1]
            pred_r = obs_out[..., b0 + b1 : b0 + b1 + b2]
            return (pred_l, pred_f, pred_r)
        return obs_out

    def _decode_sensor_from_feat(self, feat: torch.Tensor):
        """Decode sensors from full feature [h, z] (main prediction path)."""
        if self.obs_loss_mode == "soft":
            return self._split_sensor_logits(self.obs_head(feat))
        # Recon mode decoder outputs a continuous sensor observation.
        return self.obs_head(feat)

    def _decode_sensor_from_z(self, z_flat: torch.Tensor):
        """Decode sensors from z only (auxiliary path to encourage latent usage)."""
        if self.obs_loss_mode != "soft" or self.z_obs_head is None:
            return None
        return self._split_sensor_logits(self.z_obs_head(z_flat.detach()))

    def _decode_sensor_from_h(self, h_flat: torch.Tensor):
        """Decode sensors from h only (auxiliary probe of deterministic state)."""
        if self.obs_loss_mode != "soft" or self.h_obs_head is None:
            return None
        return self._split_sensor_logits(self.h_obs_head(h_flat.detach()))

    def _decode_feat(self, feat: torch.Tensor):
        """Decode sensors/actions from full feature.

        Probe heads always read detached features. Action heads read detached
        or live features based on model construction setting.
        """
        feat_detached = feat.detach()
        action_feat = feat_detached if self.detach_action_heads else feat
        turn = self.turn_head(action_feat)
        step = self.step_head(action_feat)
        loc_x = self.loc_probe_x(feat_detached)
        loc_y = self.loc_probe_y(feat_detached)
        heading_out = self.head_probe(feat_detached)
        pred_sensor = self._decode_sensor_from_feat(feat)
        if self.obs_loss_mode == "soft":
            return (pred_sensor, loc_x, loc_y, heading_out, turn, step), None
        return (pred_sensor, loc_x, loc_y, heading_out, turn, step), pred_sensor

    def _project_contrastive_pred(self, prior_feat: torch.Tensor) -> Optional[torch.Tensor]:
        """Project prior branch features for contrastive prediction."""
        if self.contrastive_head is None:
            return None
        return self.contrastive_head(prior_feat)

    def _project_contrastive_pred_steps(
        self,
        prior_feat: torch.Tensor,
        actions: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Project action-conditioned prior features for horizons 1..K.

        Horizon h uses input [prior_feat_t, a_t, ..., a_{t+h-1}] and predicts target at t+h.
        """
        if self.contrastive_dim <= 0:
            return []
        if self.contrastive_action_heads is None or len(self.contrastive_action_heads) == 0:
            pred = self._project_contrastive_pred(prior_feat)
            if pred is None or pred.shape[1] <= 1:
                return []
            return [pred[:, :-1, :]]
        B, T, _ = prior_feat.shape
        pred_steps: List[torch.Tensor] = []
        for h, head in enumerate(self.contrastive_action_heads, start=1):
            if h >= T:
                break
            th = T - h
            action_chunks = [actions[:, i : i + th, :] for i in range(h)]
            action_ctx = torch.cat(action_chunks, dim=-1) if action_chunks else actions.new_zeros((B, th, 0))
            pred_in = torch.cat([prior_feat[:, :th, :], action_ctx], dim=-1)
            pred_steps.append(head(pred_in))
        return pred_steps

    def _project_contrastive_target_z(self, z_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Project posterior stochastic latent z_t for contrastive targets."""
        if self.z_contrastive_head is None:
            return None
        return self.z_contrastive_head(z_flat.detach())
