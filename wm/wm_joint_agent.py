#!/usr/bin/env python3
"""Joint policy + world-model agent wrapper for maze experiments."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_utils_wm import create_model
from base import LossConfig
from baselines import TransformerBaseline
from pool import EpisodesOldPoolMixin
from reinforce import Reinforce
from utils import make_label_smoothing_table, make_soft_table


def create_maze_world_model(
    *,
    model_args,
    device: torch.device,
    maze_dim: int,
    turn_bins: int,
    step_bins: int,
    contrastive_temp: float = 0.1,
    contrastive_horizon_discount: float = 0.75,
    contrastive_negatives: int = 0,
    sensor_weight: float = 1.0,
    loc_weight: float = 0.0,
    head_weight: float = 0.0,
    turn_weight: float = 0.0,
    step_weight: float = 0.0,
    sensor_sigma: float = 1.0,
    pos_sigma: float = 1.0,
    heading_smoothing: float = 0.0,
    sensor_max_bin: int = 64,
) -> TransformerBaseline:
    if int(model_args.contrastive_dim) <= 0:
        raise ValueError("contrastive_dim must be > 0 for AC-CPC intrinsic reward")
    if sensor_max_bin < 1:
        raise ValueError("sensor_max_bin must be >= 1")
    if maze_dim < 2:
        raise ValueError("maze_dim must be >= 2")

    sensor_bin_count = int(sensor_max_bin) + 1
    sensor_bins = np.array([sensor_bin_count, sensor_bin_count, sensor_bin_count], dtype=np.int64)
    model_args.device = device
    active_attention_window = (
        model_args.attention_window
        if (model_args.attention_window is not None and model_args.attention_window > 0)
        else None
    )
    model_config_extra: Dict = {}
    model = create_model(
        model_args,
        input_dim=5,  # 3 sensor + 2 action
        sensor_dim=3,
        sensor_bins=sensor_bins,
        loc_x_bins=int(maze_dim),
        loc_y_bins=int(maze_dim),
        heading_dim=4,
        turn_bins=int(turn_bins),
        step_bins=int(step_bins),
        obs_dim=3,
        action_dim=2,
        active_attention_window=active_attention_window,
        model_config_extra=model_config_extra,
    )
    model.detach_action_heads = False
    loc_min = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
    sensor_min_idx = torch.zeros(3, dtype=torch.long, device=device)
    model.config = LossConfig(
        sensor_tables=[
            make_soft_table(sensor_bin_count, sensor_sigma, device),
            make_soft_table(sensor_bin_count, sensor_sigma, device),
            make_soft_table(sensor_bin_count, sensor_sigma, device),
        ],
        sensor_min_idx=sensor_min_idx,
        loc_x_table=make_soft_table(int(maze_dim), pos_sigma, device),
        loc_y_table=make_soft_table(int(maze_dim), pos_sigma, device),
        heading_table=make_label_smoothing_table(4, heading_smoothing, device),
        sensor_weight=float(sensor_weight),
        loc_weight=float(loc_weight),
        head_weight=float(head_weight),
        turn_weight=float(turn_weight),
        step_weight=float(step_weight),
        contrastive_weight=1.0,  # CPC term kept explicit in trainer loss.
        contrastive_temp=float(contrastive_temp),
        contrastive_horizon_discount=float(contrastive_horizon_discount),
        contrastive_negatives=int(contrastive_negatives),
        loc_min=loc_min,
    )
    return model


class WMActionHeadPolicy(torch.nn.Module):
    """Discrete policy logits computed from WM action heads."""

    def __init__(
        self,
        wm_model: TransformerBaseline,
        action_table: Sequence[Tuple[int, int]],
        device: torch.device,
        backprop_through_wm: bool = True,
    ):
        super().__init__()
        self.wm_model = wm_model
        self.action_dim = int(self.wm_model.action_dim)
        self.backprop_through_wm = bool(backprop_through_wm)

        turns = sorted({int(t) for t, _ in action_table})
        steps = sorted({int(s) for _, s in action_table})
        turn_to_idx = {v: i for i, v in enumerate(turns)}
        step_to_idx = {v: i for i, v in enumerate(steps)}
        turn_ids = [turn_to_idx[int(t)] for t, _ in action_table]
        step_ids = [step_to_idx[int(s)] for _, s in action_table]
        self.register_buffer("action_turn_ids", torch.tensor(turn_ids, dtype=torch.long, device=device))
        self.register_buffer("action_step_ids", torch.tensor(step_ids, dtype=torch.long, device=device))
        turn_vals = torch.tensor([int(t) for t, _ in action_table], dtype=torch.float32, device=device)
        step_vals = torch.tensor([int(s) for _, s in action_table], dtype=torch.float32, device=device)
        self.register_buffer(
            "action_cont_table",
            torch.stack([turn_vals, step_vals], dim=-1),
        )
        self._prev_actions: Optional[torch.Tensor] = None

    def reset_policy_state(self):
        self._prev_actions = None
        if hasattr(self.wm_model, "clear_cache"):
            self.wm_model.clear_cache()

    def record_sampled_actions(self, actions: torch.Tensor):
        if actions is None:
            return
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.action_cont_table.device)
        actions = actions.to(self.action_cont_table.device).long()
        if actions.dim() > 1:
            actions = actions.view(actions.shape[0], -1)[:, 0]
        if self._prev_actions is None or self._prev_actions.shape[0] != actions.shape[0]:
            self._prev_actions = torch.zeros(
                (actions.shape[0], self.action_dim),
                dtype=self.action_cont_table.dtype,
                device=self.action_cont_table.device,
            )
        self._prev_actions = self.action_cont_table[actions].to(self._prev_actions.dtype).detach()

    def forward(self, state, **kwargs):
        episode_start = kwargs.get("episode_start", None)
        if not isinstance(state, dict):
            raise ValueError("WMActionHeadPolicy expects dict observation with 'sensor'")
        sensor = state["sensor"].to(torch.float32)
        return_sequence = sensor.dim() == 3
        if sensor.dim() == 2:
            sensor = sensor.unsqueeze(1)
        if sensor.dim() != 3:
            raise ValueError(f"Expected sensor shape [B,3] or [B,T,3], got {tuple(sensor.shape)}")
        batch_size, seq_len = int(sensor.shape[0]), int(sensor.shape[1])
        if episode_start is None:
            prev_actions = sensor.new_zeros((batch_size, seq_len, self.action_dim))
            actions = prev_actions
            model_obs = {
                "sensor": sensor,
                "actions": actions,
                "prev_actions": prev_actions,
            }
            batch_out = self.wm_model(
                model_obs,
            )
            preds = batch_out.get("preds")
            if preds is None or len(preds) != 6:
                raise ValueError("forward(...)[\"preds\"] must be a 6-tuple")
            _pred_sensor, _loc_x, _loc_y, _heading, turn_logits, step_logits = preds
        else:
            if not isinstance(episode_start, torch.Tensor):
                episode_start_t = torch.as_tensor(episode_start, dtype=torch.bool, device=sensor.device)
            else:
                episode_start_t = episode_start.to(sensor.device).bool()
            if self._prev_actions is None:
                self._prev_actions = torch.zeros((batch_size, self.action_dim), dtype=sensor.dtype, device=sensor.device)
            if episode_start_t.any():
                self._prev_actions[episode_start_t, :] = 0.0
            prev_actions = self._prev_actions.unsqueeze(1).expand(batch_size, seq_len, self.action_dim)
            model_obs = {
                "sensor": sensor,
                "actions": None,
                "prev_actions": prev_actions,
            }
            roll_out = self.wm_model(
                model_obs,
                episode_start=episode_start_t,
            )
            preds = roll_out.get("preds")
            if preds is None or len(preds) != 6:
                raise ValueError("forward(...)[\"preds\"] must be a 6-tuple")
            _pred_sensor, _loc_x, _loc_y, _heading, turn_logits, step_logits = preds
        # Build logits over discrete action_table entries by factorizing
        # p(action_i)=p(turn_i)*p(step_i), i.e. log p(action_i)=log p(turn_i)+log p(step_i).
        turn_logp = F.log_softmax(turn_logits, dim=-1)
        step_logp = F.log_softmax(step_logits, dim=-1)
        action_logits = turn_logp[..., self.action_turn_ids] + step_logp[..., self.action_step_ids]
        if not return_sequence:
            action_logits = action_logits[:, 0, :]
        return action_logits


class JointWMReinforce(Reinforce):
    """Reinforce agent with joint AC-CPC world-model updates."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        sampler,
        policy_lr: float,
        num_envs: int,
        discount: float,
        logger,
        wm_model: TransformerBaseline,
        wm_optimizer: torch.optim.Optimizer,
        action_table: Sequence[Tuple[int, int]],
        device: torch.device,
        entropy_coef: float = 0.01,
        intrinsic_reward_scale: float = 1.0,
        env_reward_scale: float = 1.0,
        wm_updates_per_policy: int = 1,
        wm_replay_capacity: int = 2048,
        wm_train_episodes: int = 64,
        sensor_max_bin: int = 64,
        maze_dim: int = 10,
    ):
        super().__init__(
            policy=policy,
            sampler=sampler,
            policy_lr=policy_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger,
            entropy_coef=entropy_coef,
        )
        self.wm_model = wm_model
        self.wm_model_copy = copy.deepcopy(wm_model)
        self.wm_optimizer = wm_optimizer
        self.optimizer_policy = self.wm_optimizer
        self.action_table = [(int(turn), int(step)) for turn, step in action_table]
        self.device = torch.device(device)

        self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        self.env_reward_scale = float(env_reward_scale)
        self.wm_updates_per_policy = int(wm_updates_per_policy)
        self.wm_replay_capacity = int(wm_replay_capacity)
        self.wm_train_episodes = int(wm_train_episodes)
        self.sensor_max_bin = int(sensor_max_bin)
        self.maze_dim = int(maze_dim)

        turns = sorted({int(t) for t, _ in self.action_table})
        steps = sorted({int(s) for _, s in self.action_table})
        self.turn_to_idx = {v: i for i, v in enumerate(turns)}
        self.step_to_idx = {v: i for i, v in enumerate(steps)}

        self._turn_vals = torch.tensor([t for t, _ in self.action_table], dtype=torch.float32, device=self.device)
        self._step_vals = torch.tensor([s for _, s in self.action_table], dtype=torch.float32, device=self.device)
        self._turn_cls = torch.tensor(
            [self.turn_to_idx[t] for t, _ in self.action_table], dtype=torch.long, device=self.device
        )
        self._step_cls = torch.tensor(
            [self.step_to_idx[s] for _, s in self.action_table], dtype=torch.long, device=self.device
        )

        if self.wm_replay_capacity <= 0:
            raise ValueError("wm_replay_capacity must be > 0")
        self._wm_pool = _WMEpisodePool(num_envs=self.num_envs, pool_size=self.wm_replay_capacity)

    def episode_start(self):
        super().episode_start()
        policy = getattr(self, "policy", None)
        if policy is not None and hasattr(policy, "reset_policy_state"):
            policy.reset_policy_state()
        self._wm_pool.reset_episodes()

    def get_action(self, state, episode_start):
        action = super().get_action(state, episode_start)
        policy = getattr(self, "policy", None)
        policy.record_sampled_actions(action)
        return action

    def update(self, rewards, dones, info=None, **kwargs):
        del info, kwargs
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        env_rewards = self.env_reward_scale * rewards_t
        for env_idx in range(self.num_envs):
            self.add_reward(env_idx, env_rewards[env_idx])
        self.process_dones(dones_t)

        if not self.should_learn():
            self.logger.log_scalar("reward/env_mean", env_rewards.mean().item())
            return False

        episodes = self.get_train_episodes()
        policy_wm_episodes = []
        for ep in episodes:
            wm_episode = self._build_wm_episode_from_policy_episode(ep)
            if wm_episode:
                policy_wm_episodes.append(wm_episode)

        replay_episodes = self._sample_replay_episodes()
        obs_mix, targets_mix, _policy_batch_mix = self._build_wm_batch(policy_wm_episodes + replay_episodes)

        obs_new, targets_new, policy_meta = self._build_wm_batch(policy_wm_episodes)
        n_policy_episodes = int(len(policy_wm_episodes))
        updates = max(1, self.wm_updates_per_policy)
        wm_loss_sums: Dict[str, float] = {}
        wm_metric_sums: Dict[str, float] = {}
        total_loss_sum = 0.0
        wm_loss_sum = 0.0
        policy_loss_sum = 0.0
        entropy_mean_sum = 0.0
        policy_valid_steps_sum = 0.0
        reward_env_mean_sum = 0.0
        reward_intr_mean_sum = 0.0
        reward_total_mean_sum = 0.0

        self.wm_model.train()
        lp_rewards = self.lp_reward(obs_new, obs_mix)

        for _ in range(updates):
            # Policy update on new-policy dataset with LP reward.
            self.wm_optimizer.zero_grad(set_to_none=True)
            policy_forward = self.wm_model(obs_new)
            policy_preds = policy_forward["preds"]
            policy_loss, policy_stats = self._compute_policy_loss_from_joint_batch(
                preds=policy_preds,
                meta=policy_meta,
                intrinsic_steps=intrinsic_steps,
                n_policy_episodes=n_policy_episodes,
            )
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm_model.parameters(), 1.0)
            self.wm_optimizer.step()

            total_loss = wm_loss + policy_loss
            total_loss_sum += self._as_scalar(total_loss)
            wm_loss_sum += self._as_scalar(wm_loss)
            policy_loss_sum += self._as_scalar(policy_loss)
            entropy_mean_sum += float(policy_stats["entropy_mean"])
            policy_valid_steps_sum += float(policy_stats["valid_steps"])
            reward_env_mean_sum += float(policy_stats["reward_env_mean"])
            reward_intr_mean_sum += float(policy_stats["reward_intrinsic_mean"])
            reward_total_mean_sum += float(policy_stats["reward_total_mean"])
            for key, value in losses.items():
                wm_loss_sums[key] = wm_loss_sums.get(key, 0.0) + self._as_scalar(value)
            for key, value in metrics.items():
                wm_metric_sums[key] = wm_metric_sums.get(key, 0.0) + self._as_scalar(value)

        self._remember_wm_episodes(policy_wm_episodes, prebuilt=True)
        self.print_episode_stats(self.get_completed_episodes())
        self.version += 1
        self.clear_completed()
        self.logger.log_scalar("joint/loss_total", total_loss_sum / updates)
        self.logger.log_scalar("policy/loss", policy_loss_sum / updates)
        self.logger.log_scalar("policy/entropy_mean", entropy_mean_sum / updates)
        self.logger.log_scalar("policy/valid_steps", policy_valid_steps_sum / updates)
        self.logger.log_scalar("reward/env_mean", reward_env_mean_sum / updates)
        self.logger.log_scalar("reward/intrinsic_mean", reward_intr_mean_sum / updates)
        self.logger.log_scalar("reward/total_mean", reward_total_mean_sum / updates)
        for key in sorted(wm_loss_sums.keys()):
            self.logger.log_scalar(f"wm/loss/{key}", wm_loss_sums[key] / updates)
        self.logger.log_scalar("wm/loss/total", wm_loss_sum / updates)
        for key in sorted(wm_metric_sums.keys()):
            self.logger.log_scalar(f"wm/metric/{key}", wm_metric_sums[key] / updates)
        self.logger.log_scalar("wm/replay_size", float(len(self._wm_pool.episode_pool)))
        self.logger.log_scalar("wm/replay_sampled_episodes", float(len(replay_episodes)))
        self.logger.log_scalar("wm/policy_sampled_episodes", float(n_policy_episodes))
        joint_sampled_episodes = float(len(policy_wm_episodes) + len(replay_episodes))
        self.logger.log_scalar("wm/joint_sampled_episodes", joint_sampled_episodes)
        sampled_transitions = float(
            sum(len(ep) for ep in policy_wm_episodes) + sum(len(ep) for ep in replay_episodes)
        )
        self.logger.log_scalar("wm/joint_sampled_transitions", sampled_transitions)

        return True

    def lp_reward(self, obs_new, obs_mix):
        with torch.no_grad():
            # sync weights
            for param, target_param in zip(self.wm_model.parameters(), self.wm_model_copy.parameters()):
                target_param.copy_(param.data)

        cpc_error_before = self._evaluate_step_cpc_error_no_grad(self.wm_model_copy, obs_new, targets_new)

        # World-model update on mixed dataset (new + replay).
        self.wm_optimizer_copy.zero_grad()
        forward_out = self.wm_model_copy(
            obs_mix,
        )
        preds = forward_out["preds"]
        aux_inputs = forward_out["aux"]
        losses = self.wm_model_copy.compute_prediction_losses(
            preds=preds,
            targets=targets_mix,
            aux_inputs=aux_inputs,
        )
        metrics = self.wm_model_copy.compute_prediction_metrics(
            preds=preds,
            targets=targets_mix,
            aux_inputs=aux_inputs,
        )
        wm_loss = self._compute_wm_total_loss(losses)
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm_model_copy.parameters(), 1.0)
        self.wm_optimizer_copy.step()

        cpc_error_after = self._evaluate_step_cpc_error_no_grad(self.wm_optimizer_copy, obs_new, targets_new)
        intrinsic_steps = cpc_error_before - cpc_error_after
        return intrinsic_steps

    def action_idx_to_val(self, actions_idx: torch.Tensor) -> torch.Tensor:
        idx = actions_idx.to(self.device).long().view(-1)
        turn = self._turn_vals[idx]
        step = self._step_vals[idx]
        return torch.stack([turn, step], dim=-1)

    @staticmethod
    def _as_scalar(value) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            if value.numel() == 1:
                return float(value.detach().cpu())
            return float(value.detach().float().mean().cpu())
        return float(value)

    def _build_action_logits_from_preds(self, preds: Tuple) -> torch.Tensor:
        _pred_sensor, _loc_x, _loc_y, _heading, turn_logits, step_logits = preds
        turn_logp = F.log_softmax(turn_logits, dim=-1)
        step_logp = F.log_softmax(step_logits, dim=-1)
        return turn_logp[..., self.policy.action_turn_ids] + step_logp[..., self.policy.action_step_ids]

    def _compute_step_cpc_error_from_aux(
        self,
        *,
        aux_inputs: Optional[Dict[str, torch.Tensor]],
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = key_padding_mask.shape
        out = torch.zeros((B, T), dtype=torch.float32, device=key_padding_mask.device)
        if aux_inputs is None:
            return out
        contrastive_stats = self.wm_model._compute_contrastive_stats(
            aux_inputs=aux_inputs,
            key_padding_mask=key_padding_mask,
            temperature=self.wm_model.config.contrastive_temp,
            horizon_discount=self.wm_model.config.contrastive_horizon_discount,
            max_negatives=self.wm_model.config.contrastive_negatives,
        )
        per_step_loss = contrastive_stats.get("per_step_loss")
        per_step_valid = contrastive_stats.get("per_step_valid")
        if per_step_loss is None or per_step_valid is None:
            return out
        valid = per_step_valid
        out[valid] = per_step_loss[valid].to(out.dtype)
        return out

    def _evaluate_step_cpc_error_no_grad(self, wm_model, obs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        was_training = wm_model.training
        wm_model.eval()
        with torch.no_grad():
            forward_out = wm_model(obs)
            aux_inputs = forward_out["aux"]
            cpc_error = self._compute_step_cpc_error_from_aux(
                aux_inputs=aux_inputs,
                key_padding_mask=targets["key_padding_mask"],
            ).to(torch.float32)
        if was_training:
            wm_model.train()
        return cpc_error

    def _discount_rewards_with_mask(self, rewards: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(rewards)
        B, T = rewards.shape
        for b in range(B):
            ret = rewards.new_tensor(0.0)
            for t in range(T - 1, -1, -1):
                if valid_mask[b, t]:
                    ret = rewards[b, t] + self.discount * ret
                    out[b, t] = ret
                else:
                    ret = rewards.new_tensor(0.0)
        return out

    def _compute_policy_loss_from_joint_batch(
        self,
        *,
        preds: Tuple,
        meta: Dict[str, torch.Tensor],
        intrinsic_steps: torch.Tensor,
        n_policy_episodes: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        action_logits = self._build_action_logits_from_preds(preds)
        assert n_policy_episodes > 0
        policy_actions = meta["policy_action_idx"][:n_policy_episodes]
        key_padding_mask = meta["key_padding_mask"][:n_policy_episodes]
        policy_valid = (~key_padding_mask) & (policy_actions >= 0)
        assert policy_valid.any()

        env_rewards = meta["policy_env_reward"][:n_policy_episodes]
        intrinsic_rewards = self.intrinsic_reward_scale * intrinsic_steps[:n_policy_episodes]
        total_rewards = env_rewards + intrinsic_rewards
        discounted_returns = self._discount_rewards_with_mask(total_rewards, policy_valid)
        returns_flat = discounted_returns[policy_valid]
        ret_mean = returns_flat.detach().mean()
        ret_std = returns_flat.detach().std(unbiased=False).clamp_min(1e-7)
        normalized_returns = (returns_flat - ret_mean) / ret_std

        logits_flat = action_logits[:n_policy_episodes][policy_valid]
        actions_flat = policy_actions[policy_valid].to(torch.long)
        dist = Categorical(logits=logits_flat)
        log_probs = dist.log_prob(actions_flat)
        entropy = dist.entropy()
        e_loss = -entropy.mean()
        policy_loss = -(log_probs.clamp(-10, 10) * normalized_returns).mean() + self.entropy_coef * e_loss
        return policy_loss, {
            "entropy_mean": float(entropy.detach().mean().cpu()),
            "valid_steps": float(policy_valid.sum().detach().cpu()),
            "reward_env_mean": float(env_rewards[policy_valid].detach().mean().cpu()),
            "reward_intrinsic_mean": float(intrinsic_rewards[policy_valid].detach().mean().cpu()),
            "reward_total_mean": float(total_rewards[policy_valid].detach().mean().cpu()),
        }

    def _compute_wm_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.wm_model.config
        obs_total = losses.get("obs_total")
        if obs_total is None:
            obs_total = (
                cfg.sensor_weight * losses["sensor"]
                + cfg.loc_weight * (losses["loc_x"] + losses["loc_y"])
                + cfg.head_weight * losses["head"]
            )
        total_loss = (
            obs_total
            + cfg.turn_weight * losses["turn"]
            + cfg.step_weight * losses["step"]
            + losses.get("aux_total", torch.tensor(0.0, device=self.device))
        )
        total_loss = total_loss + cfg.contrastive_weight * losses.get(
            "contrastive", torch.tensor(0.0, device=self.device)
        )
        return total_loss

    def _build_wm_episode_from_policy_episode(self, episode):
        wm_episode = []
        L = len(episode)
        if L < 2:
            return wm_episode
        for t in range(L - 1):
            tr_t = episode[t]
            tr_tp1 = episode[t + 1]
            if len(tr_t) < 5 or len(tr_tp1) < 1:
                continue
            s_t = tr_t[0]
            a_t = tr_t[1]
            s_tp1 = tr_tp1[0]
            if not isinstance(s_t, dict) or not isinstance(s_tp1, dict):
                continue
            action_idx = torch.as_tensor(a_t, dtype=torch.long, device=self.device).view(-1)[0]
            action_cont = self.action_idx_to_val(action_idx.view(1))[0]
            env_reward = torch.as_tensor(tr_t[4], dtype=torch.float32, device=self.device)
            transition = {
                "obs_sensor": torch.as_tensor(s_t["sensor"], dtype=torch.float32, device=self.device),
                "action_cont": action_cont.to(torch.float32),
                "action_idx": action_idx.to(torch.long),
                "env_reward": env_reward.to(torch.float32),
                "next_sensor": torch.as_tensor(s_tp1["sensor"], dtype=torch.float32, device=self.device),
                "next_heading": torch.as_tensor(s_tp1["heading_idx"], dtype=torch.long, device=self.device),
                "next_location": torch.as_tensor(s_tp1["location"], dtype=torch.long, device=self.device),
                "turn_idx": self._turn_cls[action_idx].detach().to(torch.long),
                "step_idx": self._step_cls[action_idx].detach().to(torch.long),
            }
            wm_episode.append(transition)
        return wm_episode

    def _remember_wm_episodes(self, episodes, *, prebuilt: bool = False):
        for ep in episodes:
            wm_episode = ep if prebuilt else self._build_wm_episode_from_policy_episode(ep)
            if wm_episode:
                self._wm_pool.add_completed_episode(wm_episode)

    def _sample_replay_episodes(self) -> List[List[Dict[str, torch.Tensor]]]:
        replay = self._wm_pool.episode_pool
        if not replay:
            return []
        if self.wm_train_episodes <= 0:
            return list(replay)
        k = min(len(replay), self.wm_train_episodes)
        return random.sample(replay, k)

    def _build_wm_batch(self, episodes: List[List[Dict[str, torch.Tensor]]]):
        if not episodes:
            raise ValueError("episodes must be non-empty")
        lengths = [len(ep) for ep in episodes]
        if min(lengths) <= 0:
            raise ValueError("episodes must contain at least one transition each")
        batch_size = len(episodes)
        max_len = max(lengths)
        obs_sensor = torch.zeros((batch_size, max_len, 3), dtype=torch.float32, device=self.device)
        obs_actions = torch.zeros((batch_size, max_len, 2), dtype=torch.float32, device=self.device)
        y_sensor = torch.zeros((batch_size, max_len, 3), dtype=torch.float32, device=self.device)
        y_sensor_idx = torch.full((batch_size, max_len, 3), -1, dtype=torch.long, device=self.device)
        y_loc_xy = torch.full((batch_size, max_len, 2), -1, dtype=torch.long, device=self.device)
        y_head = torch.full((batch_size, max_len), -100, dtype=torch.long, device=self.device)
        y_turn = torch.full((batch_size, max_len), -100, dtype=torch.long, device=self.device)
        y_step = torch.full((batch_size, max_len), -100, dtype=torch.long, device=self.device)
        key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=self.device)
        policy_action_idx = torch.full((batch_size, max_len), -100, dtype=torch.long, device=self.device)
        policy_env_reward = torch.zeros((batch_size, max_len), dtype=torch.float32, device=self.device)

        for i, ep in enumerate(episodes):
            L = len(ep)
            key_padding_mask[i, :L] = False
            for t, tr in enumerate(ep):
                if isinstance(tr, tuple):
                    tr = tr[0]
                obs_sensor[i, t] = tr["obs_sensor"]
                obs_actions[i, t] = tr["action_cont"]
                if "action_idx" in tr:
                    policy_action_idx[i, t] = tr["action_idx"].to(torch.long)
                if "env_reward" in tr:
                    policy_env_reward[i, t] = tr["env_reward"].to(torch.float32)
                y_sensor[i, t] = tr["next_sensor"]
                y_sensor_idx[i, t] = tr["next_sensor"].round().to(torch.long).clamp_(0, self.sensor_max_bin)
                y_loc_xy[i, t] = tr["next_location"].to(torch.long).clamp_(0, self.maze_dim - 1)
                y_head[i, t] = tr["next_heading"].to(torch.long)
                y_turn[i, t] = tr["turn_idx"].to(torch.long)
                y_step[i, t] = tr["step_idx"].to(torch.long)
                # Ignore first-step action CE: t=0 is conditioned on synthetic prev_action (BOS), not a real action history.
                y_turn[i, 0] = -100
                y_step[i, 0] = -100

        obs = {
            "sensor": obs_sensor,
            "actions": obs_actions,
            "key_padding_mask": key_padding_mask,
        }
        targets = {
            "y_sensor": y_sensor,
            "y_sensor_idx": y_sensor_idx,
            "y_loc_xy": y_loc_xy,
            "y_head": y_head,
            "y_turn": y_turn,
            "y_step": y_step,
            "key_padding_mask": key_padding_mask,
        }
        meta = {
            "policy_action_idx": policy_action_idx,
            "policy_env_reward": policy_env_reward,
            "key_padding_mask": key_padding_mask,
        }
        return obs, targets, meta

    def get_state_dict(self):
        base_state = super().get_state_dict()
        return {
            "policy_state": base_state,
            "wm_model": self.wm_model.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "wm_replay_size": len(self._wm_pool.episode_pool),
        }

    def load_state_dict(self, state_dict):
        policy_state = state_dict.get("policy_state", state_dict)
        super().load_state_dict(policy_state)
        wm_state = state_dict.get("wm_model")
        if wm_state is not None:
            self.wm_model.load_state_dict(wm_state)
        wm_opt_state = state_dict.get("wm_optimizer")
        if wm_opt_state is not None:
            self.wm_optimizer.load_state_dict(wm_opt_state)


class _WMEpisodePool(EpisodesOldPoolMixin):
    """WM replay store following EpisodesOldPoolMixin lifecycle."""

    def __init__(self, num_envs: int, pool_size: int):
        self.num_envs = int(num_envs)
        self.version = 0
        super().__init__(pool_size=int(pool_size))

    def add_completed_episode(self, episode) -> None:
        if not episode:
            return
        if len(self.episode_pool) < self.pool_size:
            self.episode_pool.append(episode)
        else:
            idx = random.randint(0, self.pool_size - 1)
            self.episode_pool[idx] = episode
