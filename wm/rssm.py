#!/usr/bin/env python3
"""RSSM discrete predictor implementation."""

import torch
from torch import nn

from base import DiscreteLatentPredictorBase
from observation import ObservationDecoder, ObservationEncoder


class RSSMDiscretePredictor(DiscreteLatentPredictorBase):
    def __init__(
        self,
        *,
        observation_encoder: ObservationEncoder,
        observation_decoder: ObservationDecoder,
        z_observation_decoder: ObservationDecoder,
        h_observation_decoder: ObservationDecoder,
        cpc_sensor_probe_decoder: ObservationDecoder,
        hidden_size: int,
        sensor_mode: str,
        loc_x_bins: int,
        loc_y_bins: int,
        heading_dim: int,
        turn_bins: int,
        step_bins: int,
        action_dim: int,
        stoch_size: int,
        stoch_classes: int,
        stoch_temp: float,
        kl_dyn_beta: float,
        kl_rep_beta: float,
        kl_free_nats: float,
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
        logger=None,
    ):
        super().__init__(
            observation_encoder=observation_encoder,
            observation_decoder=observation_decoder,
            z_observation_decoder=z_observation_decoder,
            h_observation_decoder=h_observation_decoder,
            cpc_sensor_probe_decoder=cpc_sensor_probe_decoder,
            hidden_size=hidden_size,
            sensor_mode=sensor_mode,
            loc_x_bins=loc_x_bins,
            loc_y_bins=loc_y_bins,
            heading_dim=heading_dim,
            turn_bins=turn_bins,
            step_bins=step_bins,
            action_dim=action_dim,
            stoch_size=stoch_size,
            stoch_classes=stoch_classes,
            stoch_temp=stoch_temp,
            kl_dyn_beta=kl_dyn_beta,
            kl_rep_beta=kl_rep_beta,
            kl_free_nats=kl_free_nats,
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
            logger=logger,
        )
        self.rnn = nn.GRUCell(self.stoch_flat + self.action_dim, self.hidden_size)
        self._state = None

    def clear_cache(self):
        self._state = None
        if self.cpc_sfa is not None:
            self.cpc_sfa.clear_cache()

    def reset_cache(self, reset_mask):
        assert reset_mask.dtype == torch.bool and reset_mask.ndim == 1
        if self._state is not None:
            assert reset_mask.shape == self._state.shape[:1]
            self._state = self._state.masked_fill(reset_mask[:, None], 0)
        if self.cpc_sfa is not None:
            self.cpc_sfa.reset_cache(reset_mask)

    def get_cache_state(self):
        if self._state is None:
            return None
        state = {"state": self._state.detach().clone()}
        if self.cpc_sfa is not None:
            state["sfa"] = self.cpc_sfa.get_cache_state()
        return state

    def set_cache_state(self, state):
        if state is None:
            self.clear_cache()
            return
        assert state["state"].ndim == 2 and state["state"].shape[1] == self.feat_dim
        self._state = state["state"].detach().clone()
        if self.cpc_sfa is not None:
            self.cpc_sfa.set_cache_state(state["sfa"])

    def index_cache_state(self, state, batch_indices):
        if state is None:
            return None
        indexed = {"state": state["state"][batch_indices].detach().clone()}
        if self.cpc_sfa is not None:
            indexed["sfa"] = self.cpc_sfa.index_cache_state(state["sfa"], batch_indices)
        return indexed

    def _rssm_step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return self.rnn(x_t, h_prev)

    def _forward_core(
        self,
        obs,
        *,
        attention_window=None,
        episode_start=None,
        need_aux: bool = False,
    ):
        del attention_window
        obs_features, obs_embed, actions, a_prev, key_padding_mask, _ = self._validate_obs_contract(
            obs,
            episode_start=episode_start,
        )
        batch_size, sequence_length = obs_embed.shape[:2]
        h_prev = obs_embed.new_zeros((batch_size, self.hidden_size))
        z_prev_flat = obs_embed.new_zeros((batch_size, self.stoch_flat))
        if episode_start is not None:
            self.reset_cache(episode_start)
            if self._state is not None:
                assert self._state.shape == (batch_size, self.feat_dim)
                h_prev, z_prev_flat = self._state.split([self.hidden_size, self.stoch_flat], dim=-1)
        h_init = h_prev
        z_init_flat = z_prev_flat

        prior_logits_steps = []
        post_logits_steps = []
        feat_steps = []
        feat_prior_steps = []
        z_only_steps = []
        h_only_steps = []
        for t in range(sequence_length):
            if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                h_prev = h_prev.detach()
                z_prev_flat = z_prev_flat.detach()
            h_t = self._rssm_step(torch.cat([z_prev_flat, a_prev[:, t, :]], dim=-1), h_prev)
            prior_logits_t = self.prior_head(h_t).view(batch_size, self.stoch_size, self.stoch_classes)
            z_prior_t = self._sample_stoch(prior_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_prior_flat = z_prior_t.reshape(batch_size, self.stoch_flat)
            # Detach h_t so posterior gradients don't flow into the deterministic state.
            post_logits_t = self.post_head(torch.cat([h_t.detach(), obs_embed[:, t, :]], dim=-1)).view(
                batch_size, self.stoch_size, self.stoch_classes
            )
            z_t = self._sample_stoch(post_logits_t.unsqueeze(1), self.training).squeeze(1)
            z_t_flat = z_t.reshape(batch_size, self.stoch_flat)

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

        outputs = self._decode_feat(feat)
        prior_sensor_pred = self._decode_sensor_from_feat(feat_prior)
        z_only_pred = self._decode_sensor_from_z(z_post)
        h_only_pred = self._decode_sensor_from_h(h_post)
        prior_roll_sensor_pred = None
        if need_aux and self.prior_rollout_weight > 0:
            h_roll = h_init
            z_roll_flat = z_init_flat
            feat_roll_steps = []
            roll_T = sequence_length if self.prior_rollout_steps <= 0 else min(sequence_length, self.prior_rollout_steps)
            for t in range(roll_T):
                if self.bptt_horizon > 0 and t > 0 and (t % self.bptt_horizon) == 0:
                    h_roll = h_roll.detach()
                    z_roll_flat = z_roll_flat.detach()
                h_roll = self._rssm_step(torch.cat([z_roll_flat, a_prev[:, t, :]], dim=-1), h_roll)
                prior_logits_roll_t = self.prior_head(h_roll).view(batch_size, self.stoch_size, self.stoch_classes)
                z_roll_t = self._sample_stoch(prior_logits_roll_t.unsqueeze(1), self.training).squeeze(1)
                z_roll_flat = z_roll_t.reshape(batch_size, self.stoch_flat)
                feat_roll_steps.append(torch.cat([h_roll, z_roll_flat], dim=-1))
            if feat_roll_steps:
                feat_roll = torch.stack(feat_roll_steps, dim=1)
                prior_roll_sensor_pred = self._decode_sensor_from_feat(feat_roll)
        aux_inputs = self.compute_cpc_aux(feat_prior, z_post, actions, obs_embed, episode_start)
        aux_inputs["prior_sensor_pred"] = prior_sensor_pred
        if need_aux:
            aux_inputs.update({
                "prior_logits": prior_logits,
                "post_logits": post_logits,
                "feat": feat,
                "sensor_target": obs_features,
                "prior_roll_sensor_pred": prior_roll_sensor_pred,
                "z_only_pred": z_only_pred,
                "h_only_pred": h_only_pred,
            })
        state_seq = feat
        last_state = torch.cat([h_prev, z_prev_flat], dim=-1)
        if episode_start is not None:
            self._state = last_state.detach()
        return outputs, aux_inputs, state_seq, last_state

    def forward(
        self,
        obs,
        attention_window=None,
        episode_start=None,
    ):
        need_aux = episode_start is None
        outputs, aux_inputs, state_seq, last_state = self._forward_core(
            obs,
            attention_window=attention_window,
            episode_start=episode_start,
            need_aux=need_aux,
        )
        return {
            "preds": outputs,
            "aux": aux_inputs,
            "state": last_state,
            "state_last": last_state,
            "state_seq": state_seq,
        }
