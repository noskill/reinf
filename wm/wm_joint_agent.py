#!/usr/bin/env python3
"""Joint policy + world-model agent wrapper for maze experiments."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from base import LossConfig
from baselines import TransformerBaseline
from pool import EpisodesOldPoolMixin
from reinforce import Reinforce
from ppo import PPO
from util import RunningNorm
from utils import make_label_smoothing_table, make_soft_table
from util import RunningNorm, EpisodeBatch, to_device, gae, normalize_padded_returns, flatten_padded, unflatten_padded
from clustering import SmartClusteringNovelty


class BaseWMOnPolicy:
    def __init__(
        self,
        wm_model: TransformerBaseline,
        action_table: Sequence[Tuple[int, int]],
        intrinsic_reward_scale: float = 1.0,
        env_reward_scale: float = 1.0,
        wm_updates_per_policy: int = 1,
        wm_fixed: bool = False,
        wm_replay_capacity: int = 2048,
        wm_train_episodes: int = 64,
        wm_stability_coef: float = 1e-4,
        wm_divergence_novelty_coef: float = 0.0,
        sensor_max_bin: int = 64,
        maze_dim: int = 10,
        device=None,
        logger=None,
        **kwargs
    ):
        self.logger = logger
        self.wm_model = wm_model
        self.action_table = [(int(turn), int(step)) for turn, step in action_table]
        self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        self.env_reward_scale = float(env_reward_scale)
        self.wm_updates_per_policy = int(wm_updates_per_policy)
        self.wm_fixed = bool(wm_fixed)
        self.wm_replay_capacity = int(wm_replay_capacity)
        self.wm_train_episodes = int(wm_train_episodes)
        self.wm_stability_coef = float(wm_stability_coef)
        self.wm_divergence_novelty_coef = float(wm_divergence_novelty_coef)
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
        self._prev_actions: Optional[torch.Tensor] = None
        turn_ids = [self.turn_to_idx[int(t)] for t, _ in self.action_table]
        step_ids = [self.step_to_idx[int(s)] for _, s in self.action_table]
        self.action_turn_ids = torch.tensor(turn_ids, dtype=torch.long, device=self.device)
        self.action_step_ids = torch.tensor(step_ids, dtype=torch.long, device=self.device)
        self.action_cont_table = torch.stack([self._turn_vals, self._step_vals], dim=-1)
        self.action_dim = int(self.action_cont_table.shape[-1])
        self._divergence_running_norm = RunningNorm(device=self.device)
        self._episode_novelty_running_norm = RunningNorm(device=self.device)
        self.hparams = {
            "wm_updates_per_policy": self.wm_updates_per_policy,
            "wm_fixed": self.wm_fixed,
            "wm_replay_capacity": self.wm_replay_capacity,
            "wm_train_episodes": self.wm_train_episodes,
            "wm_stability_coef": self.wm_stability_coef,
            "wm_divergence_novelty_coef": self.wm_divergence_novelty_coef,
            "intrinsic_reward_scale": self.intrinsic_reward_scale,
            "env_reward_scale": self.env_reward_scale,
        }
        self.log_hparams()
        self.create_wm_optimizers(**kwargs)
        self.clustering = SmartClusteringNovelty(adaptation_frequency=100_000)

    def log_hparams(self):
        pass # todo

    def create_wm_optimizers(self, **kwargs):
        self.wm_optimizer = torch.optim.AdamW(
            [p for p in self.wm_model.parameters() if p.requires_grad],
            lr=kwargs.get('wm_lr', 0.0001),
            weight_decay=kwargs.get('wm_weight_decay', 0.0001),
        )

    def rl_sampler(self):
        return self.sampler

    def rl_add_transition_batch(self, *args, **kwargs):
        return self.add_transition_batch(*args, **kwargs)

    def rl_add_reward(self, *args, **kwargs):
        return self.add_reward(*args, **kwargs)

    def rl_get_state_dict(self):
        return super().get_state_dict()

    def rl_load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def episode_start(self):
        super().episode_start()
        self._prev_actions = None
        if hasattr(self.wm_model, "clear_cache"):
            self.wm_model.clear_cache()
        self._wm_pool.reset_episodes()

    def record_sampled_actions(self, actions: torch.Tensor):
        """
        Prev action is needed for world model
        """
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

    def call_wm(self, state, episode_start):
        if not isinstance(state, dict):
            raise ValueError("JointWMReinforce expects dict observation with 'sensor'")
        sensor = state["sensor"]
        if sensor.dim() == 2:
            sensor = sensor.unsqueeze(1)
        if sensor.dim() != 3:
            raise ValueError(f"Expected sensor shape [B,3] or [B,T,3], got {tuple(sensor.shape)}")
        batch_size, seq_len = int(sensor.shape[0]), int(sensor.shape[1])
        # we are in training mode
        if episode_start is None:
            prev_actions = sensor.new_zeros((batch_size, seq_len, self.action_dim))
            actions = prev_actions
            model_obs = {
                "sensor": sensor,
                "actions": actions,
                "prev_actions": prev_actions,
            }
            wm_out = self.wm_model(
                model_obs,
            )
        # we are in the episode collection
        else:
            if self._prev_actions is None:
                self._prev_actions = torch.zeros((batch_size, self.action_dim), dtype=sensor.dtype, device=sensor.device)
            if episode_start.any():
                self._prev_actions[episode_start, :] = 0.0
            prev_actions = self._prev_actions.unsqueeze(1).expand(batch_size, seq_len, self.action_dim)
            model_obs = {
                "sensor": sensor,
                "actions": None,
                "prev_actions": prev_actions,
            }
            wm_out = self.wm_model(
                model_obs,
                episode_start=episode_start,
            )
        return wm_out

    def process_states(self, state, episode_start):
        wm_out = self.call_wm(state, episode_start)
        wm_state = wm_out["state_last"]
        return wm_state.detach()

    def get_action(self, state, episode_start):
        # adapted from reinforce.py
        policy = self.get_policy_for_action()
        policy_states = self.process_states(state, episode_start)
        policy_kwargs = dict(episode_start=episode_start)
        sampler = self.rl_sampler()
        policy_params = sampler.policy_params(policy, policy_states, policy_kwargs)
        actions, log_probs, dist = sampler(policy_params)

        # ------------------------------------------------------------------
        # Obtain entropy safely. Some `TransformedDistribution` instances do
        # not implement an analytic entropy; fall back to the base
        # distribution in that case.
        # ------------------------------------------------------------------
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            if hasattr(dist, 'base_dist'):
                entropy = dist.base_dist.entropy()
            else:
                raise

        # Validate that each action produces *exactly* one log-prob and one
        # entropy scalar.
        assert log_probs.dim() == 1 or (log_probs.dim() == 2 and log_probs.shape[1] == 1), \
            f"log_probs shape {log_probs.shape} invalid; expected (B,) or (B,1)."
        assert entropy.dim() == 1 or (entropy.dim() == 2 and entropy.shape[1] == 1), \
            f"entropy shape {entropy.shape} invalid; expected (B,) or (B,1)."

        # Ensure entropy is a column vector (B,1) for consistent downstream
        # handling, **without** accidentally reducing across the batch.
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(-1)

        self.rl_add_transition_batch(policy_states, actions, log_probs, entropy)
        wm_states = {
            "sensor": state["sensor"].detach().to("cpu"),
            "heading_idx": state["heading_idx"].detach().to("cpu"),
            "location": state["location"].detach().to("cpu"),
            "policy_state": policy_states.detach().to("cpu"),
        }
        actions_cpu = actions.detach().to("cpu")
        log_probs_cpu = torch.zeros_like(log_probs).detach().to("cpu")
        entropy_cpu = torch.zeros_like(entropy).detach().to("cpu")
        self._wm_pool.add_transition_batch(wm_states, actions_cpu, log_probs_cpu, entropy_cpu)
        self.record_sampled_actions(actions)
        return actions

    def update(self, rewards, dones, info=None, **kwargs):
        env_rewards = self.env_reward_scale * rewards
        for env_idx in range(self.num_envs):
            self.rl_add_reward(env_idx, env_rewards[env_idx])
            self._wm_pool.add_reward(env_idx, env_rewards[env_idx].detach().to("cpu"))
        self.process_dones(dones)
        self._wm_pool.process_dones(dones)

        if not self.should_learn():
            return False
        policy_episodes = [ep for ep in self.get_train_episodes() if len(ep) >= 2]
        wm_new_episodes = [ep for ep in self._wm_pool.get_completed_episodes() if len(ep) >= 2]

        replay_episodes = self._sample_replay_episodes()
        obs_mix, targets_mix, _ = self._build_wm_batch(wm_new_episodes + replay_episodes)
        obs_new, targets_new, _ = self._build_wm_batch(wm_new_episodes)
        wm_updates = 0 if self.wm_fixed else max(1, self.wm_updates_per_policy)
        wm_updates_for_logging = max(1, wm_updates)
        wm_loss_sums: Dict[str, float] = {}
        wm_metric_sums: Dict[str, float] = {}
        total_loss_sum = 0.0
        wm_loss_sum = 0.0

        if self.wm_fixed:
            self.wm_model.eval()
        else:
            self.wm_model.train()
        self.train()
        cpc_error_before = self._evaluate_step_cpc_error_no_grad(self.wm_model, obs_new, targets_new)
        sensor_error_before = self._evaluate_step_sensor_error_no_grad(self.wm_model, obs_new, targets_new)
        with torch.no_grad():
            state_out_fixed = self.wm_model(obs_new)
            old_state_seq = state_out_fixed["state_seq"].detach()
            embs = state_out_fixed['aux']['contrastive_tgt_emb']
        divergence_novelty = self._compute_state_divergence_novelty(
            state_seq=embs,
            key_padding_mask=obs_new["key_padding_mask"],
        )
        episode_novelty = self._compute_episode_novelty(
            state_seq=embs,
            key_padding_mask=obs_new["key_padding_mask"],
        )
        novelty_signal = 0.5 * (divergence_novelty + episode_novelty)
        self._log_novelty_vs_distance_from_start(
            novelty_signal=novelty_signal,
            wm_episodes=wm_new_episodes,
            key_padding_mask=obs_new["key_padding_mask"],
        )

        for _ in range(wm_updates):
            # World-model update on mixed dataset.
            self.wm_optimizer.zero_grad()
            wm_forward = self.wm_model(obs_mix)
            preds = wm_forward["preds"]
            aux = wm_forward['aux']
            wm_loss_results, wm_loss_metrics = self.wm_model.compute_losses_and_metrics(
                preds=preds,
                targets=targets_mix,
                aux_inputs=aux,
            )
            wm_loss_base = self._compute_wm_total_loss(wm_loss_results)
            wm_stability_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            wm_state_drift = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if self.wm_stability_coef > 0.0:
                state_out = self.wm_model(obs_new)
                new_state = state_out["state_seq"]
                wm_stability_loss, wm_state_drift = self._compute_state_stability_loss(
                    new_state=new_state,
                    old_state=old_state_seq,
                    key_padding_mask=obs_new["key_padding_mask"],
                )
            wm_loss = wm_loss_base + self.wm_stability_coef * wm_stability_loss
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm_model.parameters(), 1.0)
            self.wm_optimizer.step()

            total_loss = wm_loss.detach()
            total_loss_sum += self._as_scalar(total_loss)
            wm_loss_sum += self._as_scalar(wm_loss)
            for key, value in wm_loss_results.items():
                wm_loss_sums[key] = wm_loss_sums.get(key, 0.0) + self._as_scalar(value)
            wm_loss_sums["stability"] = wm_loss_sums.get("stability", 0.0) + self._as_scalar(wm_stability_loss)
            for key, value in wm_loss_metrics.items():
                wm_metric_sums[key] = wm_metric_sums.get(key, 0.0) + self._as_scalar(value)
            wm_metric_sums["state_drift"] = wm_metric_sums.get("state_drift", 0.0) + self._as_scalar(wm_state_drift)

        if wm_updates > 0:
            cpc_error_after = self._evaluate_step_cpc_error_no_grad(self.wm_model, obs_new, targets_new)
            sensor_error_after = self._evaluate_step_sensor_error_no_grad(self.wm_model, obs_new, targets_new)
        else:
            cpc_error_after = cpc_error_before
            sensor_error_after = sensor_error_before

        flat_emb = flatten_padded(embs, obs_new["key_padding_mask"]).detach()
        cluster_dist_novelty = unflatten_padded(self.clustering(flat_emb),
                                                obs_new["key_padding_mask"], dtype=embs.dtype).to(embs)
        self.clustering.update(flat_emb)
        for key, value in getattr(self.clustering, "last_stats", {}).items():
            self.logger.log_scalar(f"cluster/{key}", value)

        emb_sample = flat_emb
        if flat_emb.shape[0] > 4096:
            sample_idx = torch.randperm(flat_emb.shape[0], device=flat_emb.device)[:4096]
            emb_sample = flat_emb[sample_idx]
        emb_pairwise_dist = torch.pdist(emb_sample.to(torch.float32))
        self.logger.log_scalar("emb/std_dim_mean", flat_emb.std(dim=0, unbiased=False).mean().item())
        self.logger.log_scalar("emb/std_dim_max", flat_emb.std(dim=0, unbiased=False).max().item())
        self.logger.log_scalar("emb/norm_mean", flat_emb.norm(dim=1).mean().item())
        self.logger.log_scalar("emb/norm_std", flat_emb.norm(dim=1).std(unbiased=False).item())
        self.logger.log_scalar("euclidean/emb_dist_mean", emb_pairwise_dist.mean().item())
        self.logger.log_scalar("euclidean/emb_dist_std", emb_pairwise_dist.std(unbiased=False).item())

        emb_pred_error = self.embedding_prediction_error(state_out_fixed['aux'],
                                                         key_padding_mask=obs_new["key_padding_mask"],
                                                         metric="cosine",
                                                         horizon_discount=0.75)
        intrinsic_rewards = self._compute_intrinsic_rewards(
            cpc_error_before=cpc_error_before,
            cpc_error_after=cpc_error_after,
            sensor_error_before=sensor_error_before,
            sensor_error_after=sensor_error_after,
            key_padding_mask=obs_new["key_padding_mask"],
            divergence_novelty=divergence_novelty,
            episode_novelty=episode_novelty,
            cluster_dist_novelty=cluster_dist_novelty,
            embedding_prediction_error=emb_pred_error,
        )
        self._apply_intrinsic_rewards_to_episodes(
            policy_episodes=policy_episodes,
            intrinsic_rewards=intrinsic_rewards,
        )
        self._log_intrinsic_vs_episode_coverage(
            intrinsic_rewards=intrinsic_rewards,
            wm_episodes=wm_new_episodes,
        )

        episode_batch = self._prepare_episode_batch(policy_episodes)
        self.train_policy_batch_joint(episode_batch)

        advantages, _, _ = episode_batch.pad(fields=['advantages_gae', 'advantages_mc'])
        advantages_gae = advantages['advantages_gae']
        self._log_intrinsic_vs_episode_coverage(
            intrinsic_rewards=advantages_gae,
            wm_episodes=wm_new_episodes,
            log_name='advantages_gae'
        )
        advantages_mc = advantages['advantages_mc']
        self._log_intrinsic_vs_episode_coverage(
            intrinsic_rewards=advantages_mc,
            wm_episodes=wm_new_episodes,
            log_name='advantages_mc'
        )

        self.print_episode_stats(self.get_completed_episodes())
        self.version += 1
        self.clear_completed()
        self._wm_pool.clear_completed()
        scalar_sums = {
            "joint/loss_total": total_loss_sum,
            "reward/intrinsic_mean": float(intrinsic_rewards.mean().detach().cpu()) * wm_updates_for_logging,
            "wm/loss/total": wm_loss_sum,
        }
        extra_scalars = {
            "wm/fixed": float(self.wm_fixed),
            "wm/actual_updates": float(wm_updates),
            "wm/replay_size": float(len(self._wm_pool.episode_pool)),
            "wm/replay_sampled_episodes": float(len(replay_episodes)),
            "wm/policy_sampled_episodes": float(len(policy_episodes)),
            "wm/joint_sampled_episodes": float(len(wm_new_episodes) + len(replay_episodes)),
            "wm/joint_sampled_transitions": float(
                sum(max(0, len(ep) - 1) for ep in wm_new_episodes)
                + sum(max(0, len(ep) - 1) for ep in replay_episodes)
            ),
        }
        self._log_update_stats(
            updates=wm_updates_for_logging,
            scalar_sums=scalar_sums,
            extra_scalars=extra_scalars,
            wm_loss_sums=wm_loss_sums,
            wm_metric_sums=wm_metric_sums,
            info=info
        )

        return True

    def train(self):
        self.policy.train()

    def _compute_intrinsic_rewards(
        self,
        *,
        cpc_error_before: torch.Tensor,
        cpc_error_after: torch.Tensor,
        sensor_error_before: torch.Tensor,
        sensor_error_after: torch.Tensor,
        key_padding_mask: torch.Tensor,
        divergence_novelty: torch.Tensor,
        episode_novelty: torch.Tensor,
        cluster_dist_novelty,
        embedding_prediction_error,
    ) -> torch.Tensor:
        lp_rewards_new = cpc_error_before - cpc_error_after
        sensor_lp_rewards_new = sensor_error_before - sensor_error_after
        sensor_lp_rewards_pos = sensor_lp_rewards_new.clamp_min(0.0)
        valid = ~key_padding_mask

        lp_surprise = lp_rewards_new + cpc_error_before * 0.001
        novelty_signal = 0.5 * (divergence_novelty + episode_novelty)

        # normal - sum of all rewards
        #reward = lp_surprise + self.wm_divergence_novelty_coef * novelty_signal
        # just lp
        #reward = lp_rewards_new

        # just surprise
        #reward = cpc_error_before

        # per-batch divergence
        #reward = divergence_novelty

        # per-episode divergence
        # reward = episode_novelty

        # clustering-based
        #reward = cluster_dist_novelty

        # sensor prediction learning progress
        # reward = sensor_lp_rewards_pos

        # sensor prediction surprise
        # reward = sensor_error_before

        #embedding_prediction_error
        # reward = embedding_prediction_error


        # lp + divergence
        reward = sensor_lp_rewards_pos + self.wm_divergence_novelty_coef * divergence_novelty + 0.02 * sensor_error_before

        reward_next = reward[:, 1:]
        reward_curr = reward[:, :-1]
        reward_progress = torch.zeros_like(reward)
        reward_progress[:, 1:] = reward_next - reward_curr
        # per-step improvment
        # reward = reward_progress
        intrinsic_rewards = self.intrinsic_reward_scale * reward

        self.logger.log_scalar("reward/lp_mean", lp_rewards_new.detach().mean().cpu().item())
        self.logger.log_scalar("reward/surprise_mean", cpc_error_before.detach().mean().cpu().item() * 0.001)
        self.logger.log_scalar("reward/sensor_lp_mean", self._masked_mean(sensor_lp_rewards_new, valid))
        self.logger.log_scalar("reward/sensor_lp_pos_mean", self._masked_mean(sensor_lp_rewards_pos, valid))
        self.logger.log_scalar("reward/sensor_surprise_mean", self._masked_mean(sensor_error_before, valid))
        self.logger.log_scalar("reward/novelty_signal_mean", self._masked_mean(novelty_signal, valid))
        self._log_first_last_window_means(
            values=intrinsic_rewards,
            valid_mask=valid,
            metric_prefix="reward/intrinsic")
        return intrinsic_rewards

    def _apply_intrinsic_rewards_to_episodes(
        self,
        *,
        policy_episodes,
        intrinsic_rewards: torch.Tensor,
    ) -> None:
        for ep_idx, episode in enumerate(policy_episodes[: intrinsic_rewards.shape[0]]):
            max_t = min(len(episode) - 1, intrinsic_rewards.shape[1])
            for t in range(max_t):
                transition = episode[t]
                updated_reward = torch.as_tensor(transition[4], dtype=torch.float32, device=self.device) + intrinsic_rewards[ep_idx, t]
                episode[t] = (transition[0], transition[1], transition[2], transition[3], updated_reward)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if mask.any():
            return values[mask].detach().mean().cpu().item()
        return 0.0

    @staticmethod
    def _first_last_window_masks(valid_mask: torch.Tensor, window_size: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        time_idx = torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0)
        first_window = min(window_size, valid_mask.shape[1])
        last_window_start = max(0, valid_mask.shape[1] - window_size)
        first_mask = valid_mask & (time_idx < first_window)
        last_mask = valid_mask & (time_idx >= last_window_start)
        return first_mask, last_mask

    def _log_first_last_window_means(
        self,
        *,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        metric_prefix: str,
        window_size: int = 20,
    ) -> Tuple[float, float]:
        first_mask, last_mask = self._first_last_window_masks(valid_mask, window_size=window_size)
        first_mean = self._masked_mean(values, first_mask)
        last_mean = self._masked_mean(values, last_mask)
        if self.logger is not None:
            self.logger.log_scalar(f"{metric_prefix}_first{window_size}_mean", first_mean)
            self.logger.log_scalar(f"{metric_prefix}_last{window_size}_mean", last_mean)
        return first_mean, last_mean

    def _log_intrinsic_vs_episode_coverage(
        self,
        *,
        intrinsic_rewards: torch.Tensor,
        wm_episodes,
        log_name='intrinsic'
    ) -> None:
        if self.logger is None:
            return
        max_rows = min(len(wm_episodes), int(intrinsic_rewards.shape[0]))
        if max_rows <= 0:
            return
        reward_sums = []
        coverage_vals = []
        max_cells = float(self.maze_dim * self.maze_dim)
        for ep_idx in range(max_rows):
            episode = wm_episodes[ep_idx]
            ep_len = min(len(episode) - 1, int(intrinsic_rewards.shape[1]))
            if ep_len <= 0:
                continue
            reward_sum = intrinsic_rewards[ep_idx, :ep_len].detach().sum().cpu().item()
            visited = set()
            for t in range(min(len(episode), ep_len + 1)):
                state_t = episode[t][0]
                loc_t = state_t["location"]
                if isinstance(loc_t, torch.Tensor):
                    x = int(loc_t[0].item())
                    y = int(loc_t[1].item())
                else:
                    x = int(loc_t[0])
                    y = int(loc_t[1])
                visited.add((x, y))
            coverage = len(visited) / max_cells
            reward_sums.append(reward_sum)
            coverage_vals.append(coverage)
        if not reward_sums:
            return
        rewards_t = torch.tensor(reward_sums, dtype=torch.float32, device=self.device)
        coverage_t = torch.tensor(coverage_vals, dtype=torch.float32, device=self.device)
        if rewards_t.numel() >= 2:
            rewards_c = rewards_t - rewards_t.mean()
            coverage_c = coverage_t - coverage_t.mean()
            denom = rewards_c.norm() * coverage_c.norm()
            if denom > 0:
                corr = (rewards_c * coverage_c).sum() / denom
            else:
                corr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            corr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.logger.log_scalar(f"reward/{log_name}_episode_sum_mean", rewards_t.mean().item())
        self.logger.log_scalar("episode/location_coverage_mean", coverage_t.mean().item())
        self.logger.log_scalar(f"reward/{log_name}_vs_episode_coverage_corr", corr.item())

    def _log_novelty_vs_distance_from_start(
        self,
        *,
        novelty_signal: torch.Tensor,
        wm_episodes,
        key_padding_mask: torch.Tensor,
    ) -> None:
        max_rows = min(len(wm_episodes), int(novelty_signal.shape[0]))
        if max_rows <= 0:
            return
        max_len = int(novelty_signal.shape[1])
        distance = torch.zeros((max_rows, max_len), dtype=torch.float32, device=self.device)
        valid = torch.zeros((max_rows, max_len), dtype=torch.bool, device=self.device)
        for ep_idx in range(max_rows):
            episode = wm_episodes[ep_idx]
            ep_len = min(len(episode) - 1, max_len)
            if ep_len <= 0:
                continue
            loc_start = episode[0][0]["location"]
            if isinstance(loc_start, torch.Tensor):
                x0 = float(loc_start[0].item())
                y0 = float(loc_start[1].item())
            else:
                x0 = float(loc_start[0])
                y0 = float(loc_start[1])
            for t in range(ep_len):
                loc_t = episode[t][0]["location"]
                if isinstance(loc_t, torch.Tensor):
                    xt = float(loc_t[0].item())
                    yt = float(loc_t[1].item())
                else:
                    xt = float(loc_t[0])
                    yt = float(loc_t[1])
                dx = xt - x0
                dy = yt - y0
                distance[ep_idx, t] = (dx * dx + dy * dy) ** 0.5
            valid[ep_idx, :ep_len] = True

        valid = valid & (~key_padding_mask[:max_rows, :max_len])
        if not valid.any():
            return

        novelty_valid = novelty_signal[:max_rows, :max_len][valid].to(torch.float32)
        distance_valid = distance[valid].to(torch.float32)
        if novelty_valid.numel() >= 2:
            novelty_centered = novelty_valid - novelty_valid.mean()
            distance_centered = distance_valid - distance_valid.mean()
            denom_step = novelty_centered.norm() * distance_centered.norm()
            if denom_step > 0:
                step_corr = (novelty_centered * distance_centered).sum() / denom_step
            else:
                step_corr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            step_corr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.logger.log_scalar("reward/novelty_vs_distance_from_start_step_corr", step_corr.item())
        self.logger.log_scalar("episode/distance_from_start_mean", distance_valid.mean().item())
        self._log_first_last_window_means(
            values=distance,
            valid_mask=valid,
            metric_prefix="episode/distance_from_start",
        )

        novelty_episode_means = []
        distance_episode_means = []
        for ep_idx in range(max_rows):
            episode_valid = valid[ep_idx]
            if not episode_valid.any():
                continue
            novelty_episode_means.append(novelty_signal[ep_idx][episode_valid].mean())
            distance_episode_means.append(distance[ep_idx][episode_valid].mean())
        if len(novelty_episode_means) >= 2:
            novelty_episode = torch.stack(novelty_episode_means).to(torch.float32)
            distance_episode = torch.stack(distance_episode_means).to(torch.float32)
            novelty_episode = novelty_episode - novelty_episode.mean()
            distance_episode = distance_episode - distance_episode.mean()
            denom_episode = novelty_episode.norm() * distance_episode.norm()
            if denom_episode > 0:
                episode_corr = (novelty_episode * distance_episode).sum() / denom_episode
            else:
                episode_corr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            episode_corr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.logger.log_scalar("reward/novelty_vs_distance_from_start_episode_corr", episode_corr.item())

    def _log_update_stats(
        self,
        *,
        updates: int,
        scalar_sums: Dict[str, float],
        extra_scalars: Dict[str, float],
        wm_loss_sums: Dict[str, float],
        wm_metric_sums: Dict[str, float],
        info
    ) -> None:
        for key, value in scalar_sums.items():
            self.logger.log_scalar(key, value / updates)
        for key in sorted(wm_loss_sums.keys()):
            self.logger.log_scalar(f"wm/loss/{key}", wm_loss_sums[key] / updates)
        for key in sorted(wm_metric_sums.keys()):
            self.logger.log_scalar(f"wm/metric/{key}", wm_metric_sums[key] / updates)
        for key, value in extra_scalars.items():
            self.logger.log_scalar(key, value)
        if info is not None:
            per_env = info.get("per_env", None)
            if per_env:
                coverage_vals = [float(item["coverage"]) for item in per_env if "coverage" in item]
                effective_vals = [float(item["effective_coverage"]) for item in per_env if "effective_coverage" in item]
                wall_vals = [float(item["wall_coverage"]) for item in per_env if "wall_coverage" in item]
                walls_explored_vals = [float(item["walls_explored"]) for item in per_env if "walls_explored" in item]
                walls_total_vals = [float(item["walls_total"]) for item in per_env if "walls_total" in item]
                if coverage_vals:
                    self.logger.log_scalar("coverage", sum(coverage_vals) / len(coverage_vals))
                if effective_vals:
                    self.logger.log_scalar("effective_coverage", sum(effective_vals) / len(effective_vals))
                if wall_vals:
                    self.logger.log_scalar("wall_coverage", sum(wall_vals) / len(wall_vals))
                if walls_explored_vals:
                    self.logger.log_scalar("walls_explored", sum(walls_explored_vals) / len(walls_explored_vals))
                if walls_total_vals:
                    self.logger.log_scalar("walls_total", sum(walls_total_vals) / len(walls_total_vals))

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
        return turn_logp[..., self.action_turn_ids] + step_logp[..., self.action_step_ids]

    def _compute_step_cpc_error_from_aux(
        self,
        wm_model,
        *,
        aux_inputs: Optional[Dict[str, torch.Tensor]],
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = key_padding_mask.shape
        out = torch.zeros((B, T), dtype=torch.float32, device=key_padding_mask.device)
        if aux_inputs is None:
            return out
        contrastive_results = wm_model.compute_contrastive_loss(
            aux_inputs=aux_inputs,
            key_padding_mask=key_padding_mask,
            temperature=wm_model.config.contrastive_temp,
            horizon_discount=wm_model.config.contrastive_horizon_discount,
        )
        per_step_loss = contrastive_results.get("per_step_loss")
        per_step_valid = contrastive_results.get("per_step_valid")
        if per_step_loss is None or per_step_valid is None:
            return out
        valid = per_step_valid
        out[valid] = per_step_loss[valid].to(out.dtype)
        return out

    def _compute_state_divergence_novelty(
        self,
        *,
        state_seq: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = ~key_padding_mask
        batch_size, step_size, _ = state_seq.shape
        ids = torch.arange(batch_size, device=state_seq.device).unsqueeze(1).expand(batch_size, step_size)
        flat_seq = flatten_padded(state_seq, key_padding_mask)
        flat_ids = flatten_padded(ids.unsqueeze(2), key_padding_mask)
        dist = torch.cdist(flat_seq, flat_seq)
        # id_dist = torch.abs(flat_ids.unsqueeze(1) - flat_ids.unsqueeze(0))
        same_batch_mask = torch.cdist(flat_ids.float(), flat_ids.float(), p=1) == 0
        dist[same_batch_mask] = float('inf')
        min_cross_batch_dist, min_indices = torch.min(dist, dim=1)
        novelty = unflatten_padded(min_cross_batch_dist, key_padding_mask)
        novelty_raw = novelty
        if valid.any():
            # novelty_valid = self._divergence_running_norm(novelty_raw[valid].reshape(-1, 1)).reshape(-1)
            novelty_valid = novelty_raw[valid]
            novelty[valid] = novelty_valid.to(novelty.dtype)
            divergence_raw_valid = novelty_raw[valid].to(torch.float32)
            self.logger.log_scalar("reward/divergence_raw_batch_std", divergence_raw_valid.std(unbiased=False).item())
            self.logger.log_scalar("reward/divergence_raw_batch_p50", torch.quantile(divergence_raw_valid, 0.50).item())
            self.logger.log_scalar("reward/divergence_raw_batch_p90", torch.quantile(divergence_raw_valid, 0.90).item())
            self.logger.log_scalar("reward/divergence_raw_batch_p99", torch.quantile(divergence_raw_valid, 0.99).item())

        if self.logger is not None:
            divergence_raw_mean = self._masked_mean(novelty_raw, valid)
            self._log_first_last_window_means(
                values=novelty_raw,
                valid_mask=valid,
                metric_prefix="reward/divergence_raw",
            )
            self.logger.log_scalar("reward/divergence_raw_mean", divergence_raw_mean)
            divergence_mean = self._masked_mean(novelty, valid)
            self._log_first_last_window_means(
                values=novelty,
                valid_mask=valid,
                metric_prefix="reward/divergence",
            )
            self.logger.log_scalar("reward/divergence_mean", divergence_mean)

        return novelty

    def _compute_episode_novelty(
        self,
        *,
        state_seq: torch.Tensor,
        key_padding_mask: torch.Tensor,
        delta: int = 10,
        num_lags: int = 6,
    ) -> torch.Tensor:
        valid = ~key_padding_mask
        batch_size, seq_len, _ = state_seq.shape
        novelty_raw = torch.zeros((batch_size, seq_len), dtype=state_seq.dtype, device=state_seq.device)
        novelty_count = torch.zeros_like(novelty_raw)

        for lag_idx in range(1, num_lags + 1):
            lag = lag_idx * delta
            if lag >= seq_len:
                break
            curr = state_seq[:, lag:, :]
            prev = state_seq[:, :-lag, :]
            dist = (curr - prev).pow(2).sum(dim=-1).sqrt()
            valid_pair = valid[:, lag:] & valid[:, :-lag]
            valid_pair_f = valid_pair.to(dist.dtype)
            novelty_raw[:, lag:] += dist * valid_pair_f
            novelty_count[:, lag:] += valid_pair_f

        has_novelty = novelty_count > 0
        novelty_raw = torch.where(
            has_novelty,
            novelty_raw / novelty_count.clamp_min(1.0),
            torch.zeros_like(novelty_raw),
        )

        novelty = torch.zeros_like(novelty_raw)
        valid_novelty = has_novelty & valid
        if valid_novelty.any():
            # novelty_valid = self._episode_novelty_running_norm(
            #     novelty_raw[valid_novelty].reshape(-1, 1)
            # ).reshape(-1)
            novelty_valid = novelty_raw[valid_novelty]
            novelty[valid_novelty] = novelty_valid.to(novelty.dtype)

        episode_novelty_raw_mean = self._masked_mean(novelty_raw, valid_novelty)
        self._log_first_last_window_means(
            values=novelty_raw,
            valid_mask=valid_novelty,
            metric_prefix="reward/episode_novelty_raw",
        )
        self.logger.log_scalar("reward/episode_novelty_raw_mean", episode_novelty_raw_mean)
        episode_novelty_mean = self._masked_mean(novelty, valid_novelty)
        self._log_first_last_window_means(
            values=novelty,
            valid_mask=valid_novelty,
            metric_prefix="reward/episode_novelty",
        )
        self.logger.log_scalar("reward/episode_novelty_mean", episode_novelty_mean)

        return novelty

    def _compute_state_stability_loss(
        self,
        *,
        new_state: torch.Tensor,
        old_state: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = new_state.to(torch.float32) - old_state.to(torch.float32)
        if diff.dim() >= 3:
            valid = (~key_padding_mask).to(diff.dtype)
            valid_exp = valid.unsqueeze(-1)
            denom = (valid_exp.sum() * diff.shape[-1]).clamp_min(1.0)
            stability_loss = (diff.pow(2) * valid_exp).sum() / denom
            drift = diff.norm(dim=-1)
            drift_mean = (drift * valid).sum() / valid.sum().clamp_min(1.0)
            return stability_loss, drift_mean
        stability_loss = diff.pow(2).mean()
        drift_mean = diff.norm(dim=-1).mean()
        return stability_loss, drift_mean

    def _evaluate_step_sensor_error_no_grad(
        self,
        wm_model,
        obs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        was_training = wm_model.training
        wm_model.eval()
        with torch.no_grad():
            forward_out = wm_model(obs)
            pred_sensor = forward_out["preds"][0]
            sensor_error = self._compute_step_sensor_error_from_preds(
                wm_model,
                pred_sensor=pred_sensor,
                targets=targets,
            ).to(torch.float32)
        if was_training:
            wm_model.train()
        return sensor_error

    @staticmethod
    def _compute_step_sensor_error_from_preds(
        wm_model,
        *,
        pred_sensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        key_padding_mask = targets["key_padding_mask"]
        if getattr(wm_model, "sensor_mode", "categorical") == "categorical":
            cfg = wm_model.config
            pred_l, pred_f, pred_r = pred_sensor
            idx = targets["y_sensor_idx"].clamp(min=0)
            if cfg.sensor_min_idx is not None:
                idx = (idx - cfg.sensor_min_idx.view(1, 1, -1)).clamp(min=0)
            idx_l = idx[..., 0].clamp(max=cfg.sensor_tables[0].shape[0] - 1)
            idx_f = idx[..., 1].clamp(max=cfg.sensor_tables[1].shape[0] - 1)
            idx_r = idx[..., 2].clamp(max=cfg.sensor_tables[2].shape[0] - 1)
            error = (
                -(cfg.sensor_tables[0][idx_l] * F.log_softmax(pred_l, dim=-1)).sum(dim=-1)
                -(cfg.sensor_tables[1][idx_f] * F.log_softmax(pred_f, dim=-1)).sum(dim=-1)
                -(cfg.sensor_tables[2][idx_r] * F.log_softmax(pred_r, dim=-1)).sum(dim=-1)
            )
        else:
            error = (pred_sensor - targets["y_sensor"]).pow(2).mean(dim=-1)
        return error.masked_fill(key_padding_mask, 0.0)

    def _evaluate_step_cpc_error_no_grad(self, wm_model, obs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        was_training = wm_model.training
        wm_model.eval()
        with torch.no_grad():
            forward_out = wm_model(obs)
            aux_inputs = forward_out["aux"]
            cpc_error = self._compute_step_cpc_error_from_aux(
                wm_model,
                aux_inputs=aux_inputs,
                key_padding_mask=targets["key_padding_mask"],
            ).to(torch.float32)
        if was_training:
            wm_model.train()
        return cpc_error

    def _compute_wm_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.wm_model.config
        obs_total = losses.get("obs_total")
        if obs_total is None:
            obs_total = (
                cfg.sensor_weight * losses["sensor"]
                + cfg.loc_weight * (losses["loc_x"] + losses["loc_y"])
                + cfg.head_weight * losses["head"]
            )
        default = torch.tensor(0.0, device=self.device)
        total_loss = (
            obs_total
            + cfg.turn_weight * losses["turn"]
            + cfg.step_weight * losses["step"]
            + losses.get("aux_total", default)
        )
        total_loss = total_loss + cfg.contrastive_weight * losses.get(
            "contrastive", default)
        total_loss = total_loss + losses.get('sfa', default) + losses['sensor_cpc'] * 0.1
        assert 'sfa' in losses
        return total_loss

    def _sample_replay_episodes(self) -> List[List[Dict[str, torch.Tensor]]]:
        replay = self._wm_pool.episode_pool
        if not replay:
            return []
        if self.wm_train_episodes <= 0:
            return list(replay)
        k = min(len(replay), self.wm_train_episodes)
        return random.sample(replay, k)

    def _build_wm_batch(self, episodes):
        if not episodes:
            raise ValueError("episodes must be non-empty")
        valid_episodes = [ep for ep in episodes if len(ep) >= 2]
        if not valid_episodes:
            raise ValueError("episodes must contain at least two steps each")
        lengths = [len(ep) - 1 for ep in valid_episodes]
        batch_size = len(valid_episodes)
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

        for i, ep in enumerate(valid_episodes):
            L = len(ep) - 1
            key_padding_mask[i, :L] = False
            for t in range(L):
                tr_t = ep[t]
                tr_tp1 = ep[t + 1]
                state_t = tr_t[0]
                state_tp1 = tr_tp1[0]
                action_idx = torch.as_tensor(tr_t[1], dtype=torch.long, device=self.device).view(-1)[0]
                obs_sensor[i, t] = torch.as_tensor(state_t["sensor"], dtype=torch.float32, device=self.device)
                obs_actions[i, t] = self.action_idx_to_val(action_idx.view(1))[0].to(torch.float32)
                y_sensor[i, t] = torch.as_tensor(state_tp1["sensor"], dtype=torch.float32, device=self.device)
                y_sensor_idx[i, t] = y_sensor[i, t].round().to(torch.long).clamp_(0, self.sensor_max_bin)
                y_loc_xy[i, t] = torch.as_tensor(state_tp1["location"], dtype=torch.long, device=self.device).clamp_(
                    0, self.maze_dim - 1
                )
                y_head[i, t] = torch.as_tensor(state_tp1["heading_idx"], dtype=torch.long, device=self.device)
                y_turn[i, t] = self._turn_cls[action_idx].to(torch.long)
                y_step[i, t] = self._step_cls[action_idx].to(torch.long)
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
        meta = {}
        return obs, targets, meta

    def get_state_dict(self):
        base_state = self.rl_get_state_dict()
        return {
            "policy_state": base_state,
            "wm_model": self.wm_model.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "wm_replay_size": len(self._wm_pool.episode_pool),
        }

    def load_state_dict(self, state_dict):
        policy_state = state_dict.get("policy_state", state_dict)
        self.rl_load_state_dict(policy_state)
        wm_state = state_dict.get("wm_model")
        if wm_state is not None:
            self.wm_model.load_state_dict(wm_state)
        wm_opt_state = state_dict.get("wm_optimizer")
        if wm_opt_state is not None:
            self.wm_optimizer.load_state_dict(wm_opt_state)

    # def compute_advantage(self, states_batch, returns_batch, rewards_batch, padding_mask):
    #     #return self.compute_advantage_gae(states_batch, rewards_batch, padding_mask)
    #     # return self.compute_advantage_monte_carlo(states_batch, returns_batch, padding_mask)
    #     # return self.return_advantage(returns_batch, padding_mask)
    #
    # def return_advantage(self, returns_batch, padding_mask):
    #     advantages = returns_batch
    #     adv_sel = torch.masked_select(advantages, torch.logical_not(padding_mask))
    #     advantage_std = adv_sel.std()
    #     advantage_std_clamped = advantage_std.clamp_min(1e-2)
    #     advantage_mean = adv_sel.mean()
    #     self.logger.log_scalar("Raw advantage_ret mean:", advantage_mean.item())
    #     self.logger.log_scalar("Raw advantage_ret std:", advantage_std.item())
    #     return returns_batch / advantage_std_clamped
    #
    # def compute_advantage_gae(self, states_batch, rewards_batch, padding_mask, **kwargs):
    #     """
    #     normal gae but no normalisation
    #     """
    #     # Policy Update
    #     with torch.no_grad():
    #         updated_values = self.value(states_batch).squeeze(-1)
    #         updated_values = updated_values * (1 - padding_mask.float())
    #     advantages = gae(self.discount, self.lambda_discount, rewards_batch, updated_values)
    #
    #     adv_sel = torch.masked_select(advantages, torch.logical_not(padding_mask))
    #     advantage_std = adv_sel.std()
    #     advantage_std_clamped = advantage_std.clamp_min(1e-2)
    #     advantage_mean = adv_sel.mean()
    #     self.logger.log_scalar("Raw advantagegae mean:", advantage_mean.item())
    #     self.logger.log_scalar("Raw advantagegae std:", advantage_std.item())
    #     return advantages / advantage_std_clamped
    #
    # def compute_advantage_monte_carlo(self, states_batch, returns_batch, padding_mask, **kwargs):
    #     """
    #     normal mc-advantage, but no normalisation
    #     """
    #     # Policy Update
    #     with torch.no_grad():
    #         updated_values = self.value(states_batch).squeeze(-1)
    #
    #     advantages = returns_batch - updated_values
    #     adv_sel = torch.masked_select(advantages, torch.logical_not(padding_mask))
    #     advantage_std = adv_sel.std()
    #     advantage_std_clamped = advantage_std.clamp_min(1e-2)
    #     advantage_mean = adv_sel.mean()
    #     self.logger.log_scalar("Raw advantagemc mean:", advantage_mean.item())
    #     self.logger.log_scalar("Raw advantagemc std:", advantage_std.item())
    #     return advantages / advantage_std_clamped

    @staticmethod
    def embedding_prediction_error(aux, key_padding_mask, metric="cosine", horizon_discount=0.75):
        pred_steps = aux["contrastive_pred_emb_steps"]
        tgt_emb = aux["contrastive_tgt_emb"]

        B, T = key_padding_mask.shape
        out = torch.zeros((B, T), device=tgt_emb.device, dtype=tgt_emb.dtype)
        weight = torch.zeros((B, T), device=tgt_emb.device, dtype=tgt_emb.dtype)

        norm_denom = sum(float(horizon_discount) ** k for k in range(len(pred_steps)))

        for k, pred in enumerate(pred_steps):
            offset = k + 1
            target = tgt_emb[:, offset:, :]
            valid = (~key_padding_mask[:, :-offset]) & (~key_padding_mask[:, offset:])

            if metric == "cosine":
                pred_n = F.normalize(pred, dim=-1)
                target_n = F.normalize(target, dim=-1)
                err = 1.0 - (pred_n * target_n).sum(dim=-1)
            else:
                err = (pred - target).pow(2).mean(dim=-1)

            w = (float(horizon_discount) ** k) / norm_denom
            out[:, :-offset][valid] += w * err[valid]
            weight[:, :-offset][valid] += w

        valid_out = weight > 0
        out[valid_out] = out[valid_out] / weight[valid_out]
        return out


class JointWMReinforce(BaseWMOnPolicy, Reinforce):
    """Reinforce agent with joint AC-CPC world-model updates."""
    def __init__(
        self,
        policy,
        sampler,
        policy_lr=0.0001,
        num_envs=8,
        discount=0.999,
        device=torch.device("cpu"),
        logger=None,
        entropy_coef=0.005,
        target_entropy=2,
        exp_adv=False,
        **kwargs
    ):
        Reinforce.__init__(self,
            policy=policy,
            sampler=sampler,
            policy_lr=policy_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger,
            entropy_coef=entropy_coef,
            target_entropy=target_entropy,
            exp_adv=exp_adv,
        )
        BaseWMOnPolicy.__init__(self, **kwargs)


class JointWMPPO(BaseWMOnPolicy):
    """PPO agent with joint AC-CPC world-model updates."""
    def __init__(self, policy,
                 value,
                 sampler,
                 policy_lr=0.0001,
                 value_lr=0.001, num_envs=8,
                 discount=0.999,
                 device=torch.device('cpu'),
                 logger=None,
                 num_learning_epochs=4,
                 entropy_coef=0.005,
                 clip_param=None,
                 exp_adv=None,
                 target_entropy=2,
                 **kwargs):

        BaseWMOnPolicy.__init__(self, device=device, logger=logger, **kwargs)
        self.agents = [self.agent]

    @property
    def device(self):
        return self.agent.device

    @property
    def num_envs(self):
        return self.agent.num_envs

    def episode_start(self):
        for agent in self.agents:
            agent.episode_start()
        self._prev_actions = None
        if hasattr(self.wm_model, "clear_cache"):
            self.wm_model.clear_cache()
        self._wm_pool.reset_episodes()

    def get_policy_for_action(self):
        return self.agent.get_policy_for_action()

    def rl_sampler(self):
        return self.agent.sampler

    def rl_add_transition_batch(self, *args, **kwargs):
        return self.agent.add_transition_batch(*args, **kwargs)

    def rl_add_reward(self, *args, **kwargs):
        return self.agent.add_reward(*args, **kwargs)

    def rl_get_state_dict(self):
        return self.agent.get_state_dict()

    def rl_load_state_dict(self, state_dict):
        return self.agent.load_state_dict(state_dict)

    def process_dones(self, dones):
        return self.agent.process_dones(dones)

    def should_learn(self):
        return self.agent.should_learn()

    def get_train_episodes(self):
        return self.agent.get_train_episodes()

    def train(self):
        self.agent.policy.train()

    def _prepare_episode_batch(self, batch):
        return self.agent._prepare_episode_batch(batch)

    def train_policy_batch_joint(self, batch):
        return self.agent.train_policy_batch_joint(batch)

    def print_episode_stats(self, episodes):
        return self.agent.print_episode_stats(episodes)

    def get_completed_episodes(self):
        return self.agent.get_completed_episodes()

    @property
    def version(self):
        return self.agent.version

    @version.setter
    def version(self, value):
        self.agent.version = value

    def clear_completed(self):
        return self.agent.clear_completed()


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
