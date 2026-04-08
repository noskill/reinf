#!/usr/bin/env python3
"""Joint policy + world-model agent wrapper for maze experiments."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from base import LossConfig
from baselines import RNNPredictor
from pool import EpisodesOldPoolMixin
from transformer import LlamaConfig
from utils import make_label_smoothing_table, make_soft_table


def create_maze_rnn_world_model(
    *,
    device: torch.device,
    maze_dim: int,
    turn_bins: int,
    step_bins: int,
    hidden_size: int = 176,
    layers: int = 3,
    intermediate: Optional[int] = None,
    obs_latent_dim: int = 64,
    probe_hidden_dim: int = 128,
    probe_layers: int = 2,
    contrastive_dim: int = 64,
    contrastive_steps: int = 1,
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
) -> RNNPredictor:
    if intermediate is None:
        intermediate = hidden_size * 4
    if contrastive_dim <= 0:
        raise ValueError("contrastive_dim must be > 0 for AC-CPC intrinsic reward")
    if sensor_max_bin < 1:
        raise ValueError("sensor_max_bin must be >= 1")
    if maze_dim < 2:
        raise ValueError("maze_dim must be >= 2")

    sensor_bin_count = int(sensor_max_bin) + 1
    sensor_bins = np.array([sensor_bin_count, sensor_bin_count, sensor_bin_count], dtype=np.int64)

    cfg = LlamaConfig(
        input_size=5,  # 3 sensor + 2 action
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate),
        num_hidden_layers=int(layers),
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=max(1, int(hidden_size) // 4),
        attention_dropout=0.0,
        attention_window=None,
    )
    model = RNNPredictor(
        cfg,
        sensor_mode="categorical",
        sensor_dim=3,
        sensor_bins=sensor_bins,
        loc_x_bins=int(maze_dim),
        loc_y_bins=int(maze_dim),
        heading_dim=4,
        turn_bins=int(turn_bins),
        step_bins=int(step_bins),
        obs_dim=3,
        obs_latent_dim=int(obs_latent_dim),
        probe_hidden_dim=int(probe_hidden_dim),
        probe_layers=int(probe_layers),
        transition="gru",
        residual_scale=1.0,
        state_norm="none",
        contrastive_dim=int(contrastive_dim),
        contrastive_steps=int(contrastive_steps),
    ).to(device)

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

    def __init__(self, wm_model: RNNPredictor, action_table: Sequence[Tuple[int, int]], device: torch.device):
        super().__init__()
        self.obs_encoder = wm_model.obs_encoder
        self.fuse = wm_model.fuse
        self.turn_head = wm_model.turn_head
        self.step_head = wm_model.step_head
        self.hidden_size = int(wm_model.fuse.out_features)

        turns = sorted({int(t) for t, _ in action_table})
        steps = sorted({int(s) for _, s in action_table})
        turn_to_idx = {v: i for i, v in enumerate(turns)}
        step_to_idx = {v: i for i, v in enumerate(steps)}
        turn_ids = [turn_to_idx[int(t)] for t, _ in action_table]
        step_ids = [step_to_idx[int(s)] for _, s in action_table]
        self.register_buffer("action_turn_ids", torch.tensor(turn_ids, dtype=torch.long, device=device))
        self.register_buffer("action_step_ids", torch.tensor(step_ids, dtype=torch.long, device=device))

    def forward(self, state, **kwargs):
        del kwargs
        if not isinstance(state, dict):
            raise ValueError("WMActionHeadPolicy expects dict observation with 'sensor'")
        sensor = state["sensor"].to(torch.float32)
        if sensor.dim() == 2:
            sensor = sensor.unsqueeze(1)
        if sensor.dim() != 3:
            raise ValueError(f"Expected sensor shape [B,3] or [B,T,3], got {tuple(sensor.shape)}")
        batch_size = sensor.shape[0]
        z = self.obs_encoder(sensor)
        h_prev = torch.zeros((batch_size, sensor.shape[1], self.hidden_size), dtype=z.dtype, device=z.device)
        fused = torch.tanh(self.fuse(torch.cat([h_prev.detach(), z.detach()], dim=-1)))
        turn_logits = self.turn_head(fused)
        step_logits = self.step_head(fused)
        turn_last = turn_logits[:, -1, :]
        step_last = step_logits[:, -1, :]
        turn_logp = F.log_softmax(turn_last, dim=-1)
        step_logp = F.log_softmax(step_last, dim=-1)
        action_logits = turn_logp[:, self.action_turn_ids] + step_logp[:, self.action_step_ids]
        return action_logits


class JointWMAgent:
    """Wraps an on-policy agent and augments rewards via AC-CPC intrinsic signal."""

    def __init__(
        self,
        *,
        policy_agent,
        wm_model: RNNPredictor,
        wm_optimizer: torch.optim.Optimizer,
        action_table: Sequence[Tuple[int, int]],
        device: torch.device,
        intrinsic_reward_scale: float = 1.0,
        env_reward_scale: float = 1.0,
        wm_updates_per_policy: int = 1,
        wm_replay_capacity: int = 2048,
        wm_train_episodes: int = 64,
        sensor_max_bin: int = 64,
        maze_dim: int = 10,
    ):
        self.policy_agent = policy_agent
        self.wm_model = wm_model
        self.wm_optimizer = wm_optimizer
        self.action_table = [(int(turn), int(step)) for turn, step in action_table]
        self.device = torch.device(device)
        self.logger = policy_agent.logger

        self.num_envs = int(policy_agent.num_envs)
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
        self._turn_norm = max(1.0, float(torch.max(torch.abs(self._turn_vals)).item()))
        self._step_norm = max(1.0, float(torch.max(torch.abs(self._step_vals)).item()))

        if self.wm_replay_capacity <= 0:
            raise ValueError("wm_replay_capacity must be > 0")
        self._wm_pool = _WMEpisodePool(num_envs=self.num_envs, pool_size=self.wm_replay_capacity)

    def episode_start(self):
        self.policy_agent.episode_start()
        self._wm_pool.reset_episodes()

    def get_action(self, state, episode_start):
        return self.policy_agent.get_action(state, episode_start)

    def update(self, rewards, dones, info=None, **kwargs):
        del info, kwargs
        rewards_t = rewards
        if not isinstance(rewards_t, torch.Tensor):
            rewards_t = torch.tensor(rewards_t, dtype=torch.float32, device=self.device)
        rewards_t = rewards_t.to(self.device).float()

        dones_t = dones
        if not isinstance(dones_t, torch.Tensor):
            dones_t = torch.tensor(dones_t, dtype=torch.bool, device=self.device)
        dones_t = dones_t.to(self.device).bool()

        env_rewards = self.env_reward_scale * rewards_t
        for env_idx in range(self.num_envs):
            self.policy_agent.add_reward(env_idx, env_rewards[env_idx])
        self.policy_agent.process_dones(dones_t)

        if not self.policy_agent.should_learn():
            if self.logger is not None:
                self.logger.log_scalar("reward/env_mean", env_rewards.mean().item())
                self.logger.log_scalar("reward/intrinsic_mean", 0.0)
                self.logger.log_scalar("reward/total_mean", env_rewards.mean().item())
            return False

        episodes = self.policy_agent.get_train_episodes()
        episodes_aug, reward_stats = self._augment_policy_episodes_with_intrinsic(episodes)
        self.policy_agent.learn_from_episodes(episodes_aug)
        self.policy_agent.print_episode_stats(self.policy_agent.get_completed_episodes())
        self.policy_agent.version += 1
        self.policy_agent.clear_completed()

        self._remember_wm_episodes(episodes_aug)
        self._train_world_model_from_replay()

        if self.logger is not None:
            self.logger.log_scalar("reward/env_mean", reward_stats["env_mean"])
            self.logger.log_scalar("reward/intrinsic_mean", reward_stats["intrinsic_mean"])
            self.logger.log_scalar("reward/total_mean", reward_stats["total_mean"])
        return True

    def _actions_to_cont(self, actions_idx: torch.Tensor) -> torch.Tensor:
        idx = actions_idx.to(self.device).long().view(-1)
        turn = self._turn_vals[idx] / self._turn_norm
        step = self._step_vals[idx] / self._step_norm
        return torch.stack([turn, step], dim=-1)

    def _compute_step_cpc_reward(
        self, obs_t: Dict[str, torch.Tensor], action_t: torch.Tensor, obs_tp1: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.wm_model.contrastive_dim <= 0:
            batch = int(obs_t["sensor"].shape[0])
            return torch.zeros((batch,), dtype=torch.float32, device=self.device)
        batch = int(obs_t["sensor"].shape[0])
        was_training = self.wm_model.training
        self.wm_model.eval()
        with torch.no_grad():
            sensor_pair = torch.stack([obs_t["sensor"], obs_tp1["sensor"]], dim=1)
            action_pair = torch.zeros((batch, 2, 2), dtype=torch.float32, device=self.device)
            action_pair[:, 0, :] = action_t
            key_padding_mask = torch.zeros((batch, 2), dtype=torch.bool, device=self.device)
            obs_batch = {
                "sensor": sensor_pair,
                "actions": action_pair,
                "key_padding_mask": key_padding_mask,
            }
            model_out = self.wm_model(obs_batch, return_aux=True)
            aux_inputs = model_out[-1]
            pred_steps = aux_inputs.get("contrastive_pred_emb_steps", [])
            tgt_emb = aux_inputs.get("contrastive_tgt_emb")
            if tgt_emb is None or not pred_steps:
                intrinsic = torch.zeros((batch,), dtype=torch.float32, device=self.device)
            else:
                pred = pred_steps[0][:, 0, :]
                key = tgt_emb[:, 1, :]
                pred = F.normalize(pred, dim=-1)
                key = F.normalize(key, dim=-1)
                logits = torch.matmul(pred, key.transpose(0, 1)) / float(self.wm_model.config.contrastive_temp)
                logp = F.log_softmax(logits, dim=1)
                intrinsic = torch.diag(logp).to(torch.float32)
        if was_training:
            self.wm_model.train()
        return intrinsic

    def compute_additional_reward(self, episodes):
        rewards = []
        for ep in episodes:
            L = len(ep)
            intrinsic_per_step = torch.zeros((L,), dtype=torch.float32, device=self.device)
            if L > 1:
                intrinsic_per_step = self._compute_episode_intrinsic(ep)
            rewards.append(intrinsic_per_step)
        return rewards 

    def _augment_policy_episodes_with_intrinsic(self, episodes):
        intrinsic_per_episode = self.compute_additional_reward(episodes)
        augmented = []
        env_sum = 0.0
        intr_sum = 0.0
        total_sum = 0.0
        count = 0

        for ep, intr in zip(episodes, intrinsic_per_episode):
            ep_aug = []
            for t, tr in enumerate(ep):
                if len(tr) < 5:
                    ep_aug.append(tr)
                    continue
                reward_env = torch.as_tensor(tr[4], dtype=torch.float32, device=self.device)
                reward_intr = self.intrinsic_reward_scale * intr[t]
                reward_total = reward_env + reward_intr
                ep_aug.append(tr[:-1] + (reward_total,))
                env_sum += float(reward_env.detach().cpu())
                intr_sum += float(reward_intr.detach().cpu())
                total_sum += float(reward_total.detach().cpu())
                count += 1
            augmented.append(ep_aug)

        denom = max(1, count)
        return augmented, {
            "env_mean": env_sum / denom,
            "intrinsic_mean": intr_sum / denom,
            "total_mean": total_sum / denom,
        }

    def _compute_episode_intrinsic(self, episode) -> torch.Tensor:
        L = len(episode)
        out = torch.zeros((L,), dtype=torch.float32, device=self.device)
        pos = []
        sensors_t = []
        sensors_tp1 = []
        actions_idx = []
        for t in range(L - 1):
            tr_t = episode[t]
            tr_tp1 = episode[t + 1]
            if len(tr_t) < 2 or len(tr_tp1) < 1:
                continue
            s_t = tr_t[0]
            a_t = tr_t[1]
            s_tp1 = tr_tp1[0]
            if not isinstance(s_t, dict) or not isinstance(s_tp1, dict):
                continue
            sensors_t.append(torch.as_tensor(s_t["sensor"], dtype=torch.float32, device=self.device))
            sensors_tp1.append(torch.as_tensor(s_tp1["sensor"], dtype=torch.float32, device=self.device))
            actions_idx.append(torch.as_tensor(a_t, dtype=torch.long, device=self.device).view(-1)[0])
            pos.append(t)

        if not pos:
            return out

        obs_t = {"sensor": torch.stack(sensors_t, dim=0)}
        obs_tp1 = {"sensor": torch.stack(sensors_tp1, dim=0)}
        actions_idx_t = torch.stack(actions_idx, dim=0)
        action_cont = self._actions_to_cont(actions_idx_t)
        intrinsic = self._compute_step_cpc_reward(obs_t, action_cont, obs_tp1)
        out[torch.as_tensor(pos, dtype=torch.long, device=self.device)] = intrinsic
        return out

    def _build_wm_episode_from_policy_episode(self, episode):
        wm_episode = []
        L = len(episode)
        if L < 2:
            return wm_episode
        for t in range(L - 1):
            tr_t = episode[t]
            tr_tp1 = episode[t + 1]
            if len(tr_t) < 2 or len(tr_tp1) < 1:
                continue
            s_t = tr_t[0]
            a_t = tr_t[1]
            s_tp1 = tr_tp1[0]
            if not isinstance(s_t, dict) or not isinstance(s_tp1, dict):
                continue
            action_idx = torch.as_tensor(a_t, dtype=torch.long, device=self.device).view(-1)[0]
            action_cont = self._actions_to_cont(action_idx.view(1))[0]
            transition = {
                "obs_sensor": torch.as_tensor(s_t["sensor"], dtype=torch.float32, device=self.device),
                "action_cont": action_cont.to(torch.float32),
                "next_sensor": torch.as_tensor(s_tp1["sensor"], dtype=torch.float32, device=self.device),
                "next_heading": torch.as_tensor(s_tp1["heading_idx"], dtype=torch.long, device=self.device),
                "next_location": torch.as_tensor(s_tp1["location"], dtype=torch.long, device=self.device),
                "turn_idx": self._turn_cls[action_idx].detach().to(torch.long),
                "step_idx": self._step_cls[action_idx].detach().to(torch.long),
            }
            wm_episode.append(transition)
        return wm_episode

    def _remember_wm_episodes(self, episodes):
        for ep in episodes:
            wm_episode = self._build_wm_episode_from_policy_episode(ep)
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
            return None, None
        lengths = [len(ep) for ep in episodes if len(ep) > 0]
        if not lengths:
            return None, None
        episodes = [ep for ep in episodes if len(ep) > 0]
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

        for i, ep in enumerate(episodes):
            L = len(ep)
            key_padding_mask[i, :L] = False
            for t, tr in enumerate(ep):
                if isinstance(tr, tuple):
                    tr = tr[0]
                obs_sensor[i, t] = tr["obs_sensor"]
                obs_actions[i, t] = tr["action_cont"]
                y_sensor[i, t] = tr["next_sensor"]
                y_sensor_idx[i, t] = tr["next_sensor"].round().to(torch.long).clamp_(0, self.sensor_max_bin)
                y_loc_xy[i, t] = tr["next_location"].to(torch.long).clamp_(0, self.maze_dim - 1)
                y_head[i, t] = tr["next_heading"].to(torch.long)
                y_turn[i, t] = tr["turn_idx"].to(torch.long)
                y_step[i, t] = tr["step_idx"].to(torch.long)
            if L > 0:
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
        return obs, targets

    def _train_world_model_from_replay(self) -> None:
        episodes = self._sample_replay_episodes()
        obs, targets = self._build_wm_batch(episodes)
        if obs is None:
            return

        self.wm_model.train()
        loss_value = 0.0
        cpc_value = 0.0
        cpc_acc = 0.0
        for _ in range(max(1, self.wm_updates_per_policy)):
            self.wm_optimizer.zero_grad(set_to_none=True)
            out = self.wm_model(
                obs,
                return_losses=True,
                return_metrics=True,
                targets=targets,
            )
            losses = out["losses"]
            metrics = out["metrics"]
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
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm_model.parameters(), 1.0)
            self.wm_optimizer.step()

            loss_value = float(total_loss.detach().cpu())
            cpc_value = float(losses.get("contrastive", torch.tensor(0.0, device=self.device)).detach().cpu())
            cpc_acc = float(metrics.get("contrastive_acc", torch.tensor(0.0, device=self.device)).detach().cpu())

        if self.logger is not None:
            self.logger.log_scalar("wm/loss", loss_value)
            self.logger.log_scalar("wm/cpc", cpc_value)
            self.logger.log_scalar("wm/cpc_acc", cpc_acc)
            self.logger.log_scalar("wm/replay_size", float(len(self._wm_pool.episode_pool)))

    def get_state_dict(self):
        return {
            "policy_state": self.policy_agent.get_state_dict(),
            "wm_model": self.wm_model.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "wm_replay_size": len(self._wm_pool.episode_pool),
        }

    def load_state_dict(self, state_dict):
        policy_state = state_dict.get("policy_state", state_dict)
        self.policy_agent.load_state_dict(policy_state)
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
