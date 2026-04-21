#!/usr/bin/env python3
"""TSSM discrete predictor implementation."""

from typing import Optional

import numpy as np
import torch

from base import DiscreteLatentPredictorBase
from transformer import LlamaConfig, LlamaModel
from tr_cache import PositionBasedDynamicCache, WindowedPositionBasedDynamicCache


class TSSMDiscretePredictor(DiscreteLatentPredictorBase):
    """Transformer state-space model with discrete stochastic latents.

    Deterministic transition is autoregressive in latent-action space:
      h_t = f_theta(h_{<t}, z_{<t}, a_{<t})
    while stochastic state uses stacked categoricals with two-sided KL.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layers: int,
        heads: int,
        head_dim: int,
        intermediate: int,
        attention_window: Optional[int],
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
        if int(hidden_size) != int(heads) * int(head_dim):
            raise ValueError("For tssm, hidden_size must equal heads * head_dim")
        super().__init__(
            hidden_size=hidden_size,
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
            action_dim=action_dim,
            stoch_size=stoch_size,
            stoch_classes=stoch_classes,
            stoch_temp=stoch_temp,
            kl_dyn_beta=kl_dyn_beta,
            kl_rep_beta=kl_rep_beta,
            kl_free_nats=kl_free_nats,
            recon_beta=recon_beta,
            obs_loss_mode=obs_loss_mode,
            prior_rollout_weight=prior_rollout_weight,
            bptt_horizon=bptt_horizon,
            z_only_weight=z_only_weight,
            h_only_weight=h_only_weight,
            prior_rollout_steps=prior_rollout_steps,
            probe_hidden_dim=probe_hidden_dim,
            probe_layers=probe_layers,
            contrastive_dim=contrastive_dim,
            contrastive_steps=contrastive_steps,
            detach_action_heads=detach_action_heads,
        )
        self.attention_window = attention_window

        trans_cfg = LlamaConfig(
            input_size=self.stoch_flat + self.action_dim,
            hidden_size=self.hidden_size,
            intermediate_size=int(intermediate),
            num_hidden_layers=int(layers),
            num_attention_heads=int(heads),
            num_key_value_heads=int(heads),
            head_dim=int(head_dim),
            attention_window=attention_window,
            use_rope=True,
        )
        self.transition = LlamaModel(trans_cfg)

    def _forward_core(
        self,
        obs,
        *,
        attention_window=None,
        episode_start=None,
        need_aux: bool = False,
    ):
        obs_embed, obs_features, actions, a_prev, key_padding_mask, B, T = self._encode_obs(
            obs,
            episode_start=episode_start,
        )
        device = actions.device

        h_prev = actions.new_zeros((B, self.hidden_size))
        z_prev_flat = actions.new_zeros((B, self.stoch_flat))
        h_init = h_prev
        z_init_flat = z_prev_flat

        if attention_window is None:
            attention_window = self.attention_window
        detach_on_pop = bool(self.training and self.bptt_horizon > 0)
        if attention_window is not None and attention_window > 0:
            cache = WindowedPositionBasedDynamicCache(
                int(attention_window),
                detach_on_pop=detach_on_pop,
            ).to(device=device)
        else:
            cache = PositionBasedDynamicCache(detach_on_pop=detach_on_pop).to(device=device)

        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)

        prior_logits_steps = []
        post_logits_steps = []
        feat_steps = []
        feat_prior_steps = []
        z_only_steps = []
        h_only_steps = []
        for t in range(T):
            if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                z_prev_flat = z_prev_flat.detach()
            trans_in_t = torch.cat([z_prev_flat, a_prev[:, t, :]], dim=-1).unsqueeze(1)
            h_step = self.transition(
                trans_in_t,
                past_key_values=cache,
                cache_position=pos[:, t : t + 1],
                attention_window=attention_window,
            )
            h_t = h_step[:, -1, :]
            prior_logits_t = self.prior_head(h_t).view(B, self.stoch_size, self.stoch_classes)
            z_prior_t = self._sample_stoch(prior_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_prior_flat = z_prior_t.reshape(B, self.stoch_flat)
            post_logits_t = self.post_head(torch.cat([h_t.detach(), obs_embed[:, t, :]], dim=-1)).view(
                B, self.stoch_size, self.stoch_classes
            )
            z_t = self._sample_stoch(post_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_t_flat = z_t.reshape(B, self.stoch_flat)

            feat_steps.append(torch.cat([h_t, z_t_flat], dim=-1))
            feat_prior_steps.append(torch.cat([h_t, z_prior_flat], dim=-1))
            z_only_steps.append(z_t_flat)
            h_only_steps.append(h_t)
            prior_logits_steps.append(prior_logits_t)
            post_logits_steps.append(post_logits_t)

            h_prev = h_t
            z_prev_flat = z_t_flat

        feat = torch.stack(feat_steps, dim=1)                 # [B, T, H + S*C]
        feat_prior = torch.stack(feat_prior_steps, dim=1)     # [B, T, H + S*C]
        prior_logits = torch.stack(prior_logits_steps, dim=1) # [B, T, S, C]
        post_logits = torch.stack(post_logits_steps, dim=1)   # [B, T, S, C]
        z_post = torch.stack(z_only_steps, dim=1)             # [B, T, S*C]
        h_post = torch.stack(h_only_steps, dim=1)             # [B, T, H]

        outputs, obs_hat = self._decode_feat(feat)
        prior_sensor_pred = self._decode_sensor_from_feat(feat_prior)
        z_only_pred = self._decode_sensor_from_z(z_post)
        h_only_pred = self._decode_sensor_from_h(h_post)
        prior_roll_sensor_pred = None
        if self.prior_rollout_weight > 0:
            if attention_window is not None and attention_window > 0:
                cache_roll = WindowedPositionBasedDynamicCache(
                    int(attention_window),
                    detach_on_pop=detach_on_pop,
                ).to(device=device)
            else:
                cache_roll = PositionBasedDynamicCache(detach_on_pop=detach_on_pop).to(device=device)
            h_roll = h_init
            z_roll_flat = z_init_flat
            feat_roll_steps = []
            roll_T = T if self.prior_rollout_steps <= 0 else min(T, self.prior_rollout_steps)
            for t in range(roll_T):
                if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                    z_roll_flat = z_roll_flat.detach()
                trans_in_roll_t = torch.cat([z_roll_flat, a_prev[:, t, :]], dim=-1).unsqueeze(1)
                h_step_roll = self.transition(
                    trans_in_roll_t,
                    past_key_values=cache_roll,
                    cache_position=pos[:, t : t + 1],
                    attention_window=attention_window,
                )
                h_roll = h_step_roll[:, -1, :]
                prior_logits_roll_t = self.prior_head(h_roll).view(B, self.stoch_size, self.stoch_classes)
                z_roll_t = self._sample_stoch(prior_logits_roll_t.unsqueeze(1), self.training).squeeze(1)
                z_roll_flat = z_roll_t.reshape(B, self.stoch_flat)
                feat_roll_steps.append(torch.cat([h_roll, z_roll_flat], dim=-1))
            if feat_roll_steps:
                feat_roll = torch.stack(feat_roll_steps, dim=1)
                prior_roll_sensor_pred = self._decode_sensor_from_feat(feat_roll)

        aux_inputs = None
        if need_aux:
            aux_inputs = {
                "prior_logits": prior_logits,
                "post_logits": post_logits,
                "feat": feat,
                "obs_target": obs_features,
                "sensor_target": obs_features,
                "loc_target": obs["loc"],
                "head_target": obs["heading"],
                "prior_sensor_pred": prior_sensor_pred,
                "prior_roll_sensor_pred": prior_roll_sensor_pred,
                "z_only_pred": z_only_pred,
                "h_only_pred": h_only_pred,
            }
            # Twister-style action-conditioned contrastive predictors for horizons 1..K.
            aux_inputs["contrastive_pred_emb_steps"] = self._project_contrastive_pred_steps(
                feat_prior,
                actions,
            )
            aux_inputs["contrastive_tgt_emb"] = self._project_contrastive_target_z(z_post)
            if obs_hat is not None:
                aux_inputs["obs_hat"] = obs_hat

        last_state = torch.cat([h_prev, z_prev_flat], dim=-1)
        return outputs, aux_inputs, last_state

    def forward(
        self,
        obs,
        attention_window=None,
        return_state=False,
        episode_start=None,
    ):
        need_aux = episode_start is None
        outputs, aux_inputs, last_state = self._forward_core(
            obs,
            attention_window=attention_window,
            episode_start=episode_start,
            need_aux=need_aux,
        )
        return {
            "preds": outputs,
            "aux": aux_inputs if need_aux else None,
            "state": last_state if return_state else None,
        }
