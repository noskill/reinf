import torch
import math
from typing import Dict

from ppo import PPOBase
from wm_joint_agent import BaseWMOnPolicy


class DoubleAgent(BaseWMOnPolicy):
    """PPO two-policy agent with joint AC-CPC world-model updates."""
    def __init__(self, agent_high: PPOBase,
                 agent_low: PPOBase,
                 device,
                 logger,
                 **kwargs):
        self.agent_high = agent_high
        self.agent_low = agent_low
        self.agents = [self.agent_high, self.agent_low]
        self.low_warmup_updates = int(kwargs.pop("low_warmup_updates", 0))
        self.low_warmup_goal_noise = float(kwargs.pop("low_warmup_goal_noise", 0.1))
        self.high_warmup_horizon = int(kwargs.pop("high_warmup_horizon", 5))
        self.high_warmup_goal_coef = float(kwargs.pop("high_warmup_goal_coef", 0.005))
        self.high_warmup_goal_epochs = int(kwargs.pop("high_warmup_goal_epochs", 1))
        self.reset_high_agent_on_load = bool(kwargs.pop("reset_high_agent_on_load", False))
        BaseWMOnPolicy.__init__(self, device=device, logger=logger, **kwargs)

    @property
    def device(self):
        return self.agent_low.device

    @property
    def num_envs(self):
        return self.agent_low.num_envs

    def episode_start(self):
        for agent in self.agents:
            agent.episode_start()
        self._prev_actions = None
        if hasattr(self.wm_model, "clear_cache"):
            self.wm_model.clear_cache()
        self._wm_pool.reset_episodes()

    @property
    def version(self):
        return self.agent_low.version

    @version.setter
    def version(self, value):
        self.agent_low.version = value
        self.agent_high.version = value

    def rl_add_reward(self, env_idx, reward):
        self.agent_low.add_reward(env_idx, reward)
        self.agent_high.add_reward(env_idx, reward)

    def process_dones(self, dones):
        self.agent_low.process_dones(dones)
        self.agent_high.process_dones(dones)

    def train(self):
        for agent in self.agents:
            agent.policy.train()
            agent.value.train()

    def clear_completed(self):
        self.agent_low.clear_completed()
        self.agent_high.clear_completed()

    def rl_get_state_dict(self):
        return {
            "high": self.agent_high.get_state_dict(),
            "low": self.agent_low.get_state_dict(),
        }

    def rl_load_state_dict(self, state_dict):
        self.agent_high.load_state_dict(state_dict["high"])
        self.agent_low.load_state_dict(state_dict["low"])
        if self.reset_high_agent_on_load:
            self.reset_high_agent()

    @staticmethod
    def _reset_module_parameters(module):
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if reset is not None:
                reset()

    def reset_high_agent(self):
        self._reset_module_parameters(self.agent_high.policy)
        self._reset_module_parameters(self.agent_high.value)
        self.agent_high.policy_old.load_state_dict(self.agent_high.policy.state_dict())
        self.agent_high.create_optimizers()
        self.agent_high.episode_start()

    def should_learn(self):
        return ( self.agent_low.should_learn()
            and self.agent_high.should_learn()
            and len(self._wm_pool.get_completed_episodes()) >= self.num_envs
        )

    def _in_low_warmup(self):
        return self.version < self.low_warmup_updates

    def _squeeze_time(self, x):
        if x.dim() == 3:
            assert x.shape[1] == 1, f"Expected online one-step feature, got time dim {x.shape[1]}"
            return x[:, -1, :]
        return x

    def _cpc_surprise(self, cpc, episode_start):
        prev_h, prev_action = self.agent_high.prediction_context()
        if prev_h is None:
            return cpc.new_zeros((cpc.shape[0], 1))

        assert prev_action.dim() == 1, f"Expected previous discrete action [B], got {tuple(prev_action.shape)}"
        assert prev_h.shape[0] == cpc.shape[0], "previous h and current cpc batch sizes must match"
        assert prev_action.shape[0] == cpc.shape[0], "previous action and current cpc batch sizes must match"

        prev_action_cont = self.action_cont_table[prev_action.to(self.action_cont_table.device).long()]
        with torch.no_grad():
            pred_cpc = self.wm_model.predict_next_contrastive_emb(prev_h, prev_action_cont)
            surprise = torch.linalg.vector_norm(pred_cpc - cpc, dim=-1, keepdim=True)

        reset_mask = torch.as_tensor(episode_start, dtype=torch.bool, device=surprise.device).view(-1)
        assert reset_mask.numel() == cpc.shape[0], "episode_start must match batch size"
        if reset_mask.any():
            surprise = surprise.clone()
            surprise[reset_mask] = 0.0
        return surprise.detach()

    def _low_goal_distances(self, episode):
        states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
        goal_dim = self.wm_model.contrastive_dim
        goal_delta = states[:, -(goal_dim + 1):-1]
        return torch.linalg.vector_norm(goal_delta, dim=-1)

    def _extract_low_base_sfa(self, states):
        assert states.dim() == 2, f"Expected low states [T,D], got {tuple(states.shape)}"
        goal_dim = self.wm_model.contrastive_dim
        base = states[:, :-(2 * goal_dim + 1)]
        sfa = base[:, -goal_dim:]
        assert sfa.shape[-1] == goal_dim, "failed to extract SFA from low-level state"
        return base, sfa

    def _build_low_state(self, base, sfa, goal, goal_valid):
        assert goal_valid.shape == (*goal.shape[:-1], 1), "goal_valid must be [...,1]"
        return torch.cat([base, goal, goal - sfa, goal_valid.to(goal)], dim=-1)

    def _compute_low_actual_rewards(self, episode, intrinsic_rewards=None, goal_coef=1.0, intrinsic_coef=0.1):
        distances = self._low_goal_distances(episode)
        goal_rewards = torch.zeros_like(distances)
        if distances.shape[0] > 1:
            goal_rewards[:-1] = distances[:-1] - distances[1:]
        rewards = float(goal_coef) * goal_rewards
        if intrinsic_rewards is not None:
            intrinsic_bonus = torch.zeros_like(rewards)
            valid_steps = min(rewards.shape[0], intrinsic_rewards.shape[0])
            intrinsic_bonus[:valid_steps] = intrinsic_rewards[:valid_steps].to(rewards).clamp_min(0.0)
            rewards = rewards + float(intrinsic_coef) * intrinsic_bonus
        return rewards.detach().to("cpu")

    def _overwrite_low_rewards(self, episodes, rewards_per_episode):
        assert len(episodes) == len(rewards_per_episode), "episode/reward count mismatch"
        for episode, rewards in zip(episodes, rewards_per_episode):
            assert len(episode) == rewards.shape[0], "episode/reward length mismatch"
            for step_idx, reward in enumerate(rewards):
                episode[step_idx] = (*episode[step_idx][:4], reward)
        self.logger.log_scalar(
            "low/actual_reward_mean",
            torch.cat(rewards_per_episode).mean() if rewards_per_episode else torch.tensor(0.0),
        )
        self.logger.log_scalar(
            "low/goal_distance_mean",
            torch.cat([self._low_goal_distances(ep).detach().cpu() for ep in episodes]).mean()
            if episodes else torch.tensor(0.0),
        )

    def _log_low_reward_components(self, low_episodes, intrinsic_rewards, goal_coef=1.0, intrinsic_coef=0.1):
        goal_rewards = [self._compute_low_actual_rewards(ep, intrinsic_rewards=None) for ep in low_episodes]
        intrinsic_bonus = []
        for ep_idx, episode in enumerate(low_episodes):
            bonus = torch.zeros(len(episode), dtype=torch.float32, device=self.device)
            valid_steps = min(len(episode), intrinsic_rewards.shape[1])
            bonus[:valid_steps] = intrinsic_rewards[ep_idx, :valid_steps].to(bonus).clamp_min(0.0)
            intrinsic_bonus.append(bonus.detach().cpu())
        self.logger.log_scalar(
            "low/goal_progress_reward_mean",
            float(goal_coef) * torch.cat(goal_rewards).mean() if goal_rewards else torch.tensor(0.0),
        )
        self.logger.log_scalar(
            "low/intrinsic_bonus_mean",
            float(intrinsic_coef) * torch.cat(intrinsic_bonus).mean() if intrinsic_bonus else torch.tensor(0.0),
        )
        self.logger.log_scalar("low/reward_goal_coef", float(goal_coef))
        self.logger.log_scalar("low/reward_intrinsic_coef", float(intrinsic_coef))
        if low_episodes:
            goal_valid = [torch.stack([step[0][-1] for step in episode], dim=0).to(torch.float32) for episode in low_episodes]
            self.logger.log_scalar("low/goal_valid_mean", torch.cat(goal_valid).mean())

    def _compute_low_hindsight(self, low_episodes, intrinsic_rewards, horizon=5):
        assert len(low_episodes) <= intrinsic_rewards.shape[0], "low episodes and intrinsic rewards are misaligned"
        states_per_episode = []
        negative_states_per_episode = []
        actions_per_episode = []
        weights_per_episode = []
        target_distances = []
        progress_terms = []
        intrinsic_terms = []
        batch_sfa = [
            self._extract_low_base_sfa(torch.stack([step[0] for step in episode], dim=0).to(self.device))[1]
            for episode in low_episodes
        ]
        flat_batch_sfa = torch.cat(batch_sfa, dim=0)

        for ep_idx, episode in enumerate(low_episodes):
            states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
            actions = torch.stack([step[1] for step in episode], dim=0).to(self.device).long()
            base, sfa = self._extract_low_base_sfa(states)
            hindsight_states = torch.zeros_like(states)
            negative_states = torch.zeros_like(states)
            weights = torch.zeros(sfa.shape[0], dtype=sfa.dtype, device=sfa.device)

            if sfa.shape[0] > 1:
                step_horizon = min(int(horizon), sfa.shape[0] - 1)
                valid_steps = sfa.shape[0] - step_horizon
                goals = sfa[step_horizon:]
                negative_indices = torch.randint(flat_batch_sfa.shape[0], (valid_steps,), device=self.device)
                negative_goals = flat_batch_sfa[negative_indices].to(sfa)
                goal_valid = torch.ones((valid_steps, 1), dtype=sfa.dtype, device=sfa.device)
                hindsight_states[:valid_steps] = self._build_low_state(base[:valid_steps], sfa[:valid_steps], goals, goal_valid)
                negative_states[:valid_steps] = self._build_low_state(base[:valid_steps], sfa[:valid_steps], negative_goals, goal_valid)
                distances = torch.linalg.vector_norm(goals - sfa[:valid_steps], dim=-1)
                next_distances = torch.linalg.vector_norm(goals - sfa[1 : valid_steps + 1], dim=-1)
                progress = (distances - next_distances).clamp_min(0.0)
                _intrinsic = intrinsic_rewards[ep_idx, :sfa.shape[0]].to(sfa).clamp_min(0.0)
                for t in range(valid_steps):
                    intrinsic_term = 0.1 * _intrinsic[t : t + step_horizon].mean()
                    weights[t] = progress[t] + intrinsic_term
                    intrinsic_terms.append(intrinsic_term.detach().view(1))
                target_distances.append(distances.detach())
                progress_terms.append(progress.detach())

            states_per_episode.append(hindsight_states.detach().cpu())
            negative_states_per_episode.append(negative_states.detach().cpu())
            actions_per_episode.append(actions.detach().cpu())
            weights_per_episode.append(weights.detach().cpu())

        if target_distances:
            self.logger.log_scalar("low/hindsight_target_dist", torch.cat(target_distances).mean())
        if progress_terms:
            self.logger.log_scalar("low/hindsight_progress_weight_mean", torch.cat(progress_terms).mean())
        if intrinsic_terms:
            self.logger.log_scalar("low/hindsight_intrinsic_weight_mean", torch.cat(intrinsic_terms).mean())
        return states_per_episode, actions_per_episode, weights_per_episode, negative_states_per_episode

    def _compute_high_switch_rewards(self, episode):
        states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
        switches = torch.stack([step[1]["switch"] for step in episode], dim=0).to(self.device).to(torch.float32).view(-1)
        progress = states[:, -2]
        steps_since_switch = states[:, -1]

        # progress[t] is measured before switch[t] is applied. Rewarding switch[t]
        # with progress[t] would credit the gate for the old goal. Instead, use
        # next-step progress: if switch[t] starts a new goal, progress[t+1] is
        # measured against that new goal; if switch[t] keeps the goal, progress[t+1]
        # measures whether keeping it kept working.
        next_progress = torch.zeros_like(progress)
        if progress.shape[0] > 1:
            next_progress[:-1] = progress[1:]

        rewards = next_progress - 0.01 * switches
        stalled = (switches < 0.5) & (steps_since_switch > 2.0) & (next_progress < 0.005)
        rewards[stalled] -= 1.0
        return rewards.detach()

    def _compute_high_goal_rewards(self, episode, intrinsic_rewards):
        states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
        progress = states[:, -2]

        # The sampled goal at t only affects behavior after t, so use next-step
        # progress as the achievability gate for crediting intrinsic reward.
        next_progress = torch.zeros_like(progress)
        if progress.shape[0] > 1:
            next_progress[:-1] = progress[1:]

        rewards = torch.zeros_like(progress)
        valid_steps = min(progress.shape[0], intrinsic_rewards.shape[0])
        intrinsic = intrinsic_rewards[:valid_steps].to(progress).clamp_min(0.0)
        rewards[:valid_steps] = intrinsic * next_progress[:valid_steps].clamp(0.0, 1.0)
        return rewards.detach()

    def _extract_high_sfa(self, states):
        assert states.dim() == 2, f"Expected high states [T,D], got {tuple(states.shape)}"
        goal_dim = self.wm_model.contrastive_dim
        sfa_end = -(goal_dim + 3)
        sfa_start = sfa_end - goal_dim
        sfa = states[:, sfa_start:sfa_end]
        assert sfa.shape[-1] == goal_dim, "failed to extract SFA from high-level state"
        return sfa

    def _compute_high_goal_hindsight(self, high_episodes, intrinsic_rewards, horizon=5):
        assert len(high_episodes) <= intrinsic_rewards.shape[0], "high episodes and intrinsic rewards are misaligned"
        targets_per_episode = []
        weights_per_episode = []
        target_distances = []
        goal_current_distances = []
        goal_future_distances = []
        goal_next_progress = []

        for ep_idx, episode in enumerate(high_episodes):
            states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
            sfa = self._extract_high_sfa(states)
            goals = torch.stack([step[1]["goal"] for step in episode], dim=0).to(self.device)
            assert goals.shape == sfa.shape, "high goal and SFA shapes must match"
            targets = torch.zeros_like(sfa)
            weights = torch.zeros(sfa.shape[0], dtype=sfa.dtype, device=sfa.device)

            goal_current_distances.append(torch.linalg.vector_norm(goals - sfa, dim=-1).detach())
            if sfa.shape[0] > 1:
                d_curr = torch.linalg.vector_norm(goals[:-1] - sfa[:-1], dim=-1)
                d_next = torch.linalg.vector_norm(goals[:-1] - sfa[1:], dim=-1)
                goal_next_progress.append((d_curr - d_next).detach())

            if sfa.shape[0] > 1:
                step_horizon = min(int(horizon), sfa.shape[0] - 1)
                valid_steps = sfa.shape[0] - step_horizon

                # Hindsight target: train the goal proposal at time t to predict an
                # SFA state the low-level policy actually reached shortly after t.
                # The loss later masks to switch_t == 1, so only new-goal proposal
                # decisions are supervised by these future achieved states.
                targets[:valid_steps] = sfa[step_horizon:]

                # Use intrinsic reward along the achieved segment as the learning
                # weight: future states reached through more novel/surprising/LP-rich
                # segments should be more likely high-level goals.
                # that's similiar to   w(s, z_future) = exp(Q(s, z_future) - V(s))
                # in advantage-weighted regression / AWR / AWAC than vanilla PPO.
                # todo: try w_t = segment_intrinsic_return - value_goal(s_t)
                _intrinsic = intrinsic_rewards[ep_idx, :sfa.shape[0]].to(sfa).clamp_min(0.0)
                for t in range(valid_steps):
                    weights[t] = _intrinsic[t : t + step_horizon].mean()
                target_distances.append(torch.linalg.vector_norm(targets[:valid_steps] - sfa[:valid_steps], dim=-1).detach())
                goal_future_distances.append(torch.linalg.vector_norm(goals[:valid_steps] - targets[:valid_steps], dim=-1).detach())

            targets_per_episode.append(targets.detach())
            weights_per_episode.append(weights.detach())


        self.logger.log_scalar("high/goal_hindsight_target_dist", torch.cat(target_distances).mean())
        self.logger.log_scalar("high/goal_dist_to_current_sfa", torch.cat(goal_current_distances).mean())
        self.logger.log_scalar("high/goal_dist_to_future_sfa", torch.cat(goal_future_distances).mean())
        self.logger.log_scalar("high/goal_next_progress", torch.cat(goal_next_progress).mean())
        return targets_per_episode, weights_per_episode

    def _compute_high_warmup_targets(self, high_episodes, horizon=None):
        if horizon is None:
            horizon = self.high_warmup_horizon
        targets_per_episode = []
        weights_per_episode = []
        target_distances = []
        target_deltas = []

        for episode in high_episodes:
            states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
            sfa = self._extract_high_sfa(states)
            targets = torch.zeros_like(sfa)
            weights = torch.zeros(sfa.shape[0], dtype=sfa.dtype, device=sfa.device)

            if sfa.shape[0] > 1:
                step_horizon = min(int(horizon), sfa.shape[0] - 1)
                valid_steps = sfa.shape[0] - step_horizon
                targets[:valid_steps] = sfa[step_horizon:]
                weights[:valid_steps] = 1.0
                delta = targets[:valid_steps] - sfa[:valid_steps]
                target_distances.append(torch.linalg.vector_norm(delta, dim=-1).detach())
                target_deltas.append(delta.detach())

            targets_per_episode.append(targets.detach())
            weights_per_episode.append(weights.detach())

        if target_distances:
            self.logger.log_scalar("high/warmup_goal_target_dist", torch.cat(target_distances).mean())
        if target_deltas:
            deltas = torch.cat(target_deltas, dim=0).to(torch.float32)
            std = deltas.std(dim=0, unbiased=False)
            var = deltas.var(dim=0, unbiased=False).clamp_min(1e-8)
            diag_entropy = 0.5 * torch.log(2.0 * math.pi * math.e * var).sum()
            self.logger.log_scalar("high/warmup_delta/std_dim_mean", std.mean().item())
            self.logger.log_scalar("high/warmup_delta/std_dim_min", std.min().item())
            self.logger.log_scalar("high/warmup_delta/std_dim_max", std.max().item())
            self.logger.log_scalar("high/warmup_delta/diag_entropy", diag_entropy.item())
        return targets_per_episode, weights_per_episode

    def _overwrite_high_rewards(self, episodes, goal_rewards_per_episode, switch_rewards_per_episode):
        assert len(episodes) == len(switch_rewards_per_episode), "episode/reward count mismatch"
        assert len(episodes) == len(goal_rewards_per_episode), "episode/goal reward count mismatch"
        for episode, switch_rewards in zip(episodes, switch_rewards_per_episode):
            assert len(episode) == switch_rewards.shape[0], "episode/reward length mismatch"
        for episode, goal_rewards, switch_rewards in zip(episodes, goal_rewards_per_episode, switch_rewards_per_episode):
            assert len(episode) == goal_rewards.shape[0] == switch_rewards.shape[0], "episode/reward length mismatch"
            for step_idx, (goal_reward, switch_reward) in enumerate(zip(goal_rewards, switch_rewards)):
                episode[step_idx] = (*episode[step_idx][:4], goal_reward, switch_reward)
        self.logger.log_scalar(
            "high/switch_reward_mean",
            torch.cat(switch_rewards_per_episode).mean() if switch_rewards_per_episode else torch.tensor(0.0),
        )
        self.logger.log_scalar(
            "high/goal_reward_mean",
            torch.cat(goal_rewards_per_episode).mean() if goal_rewards_per_episode else torch.tensor(0.0),
        )

    def get_action(self, state, episode_start):
        wm_out = self.call_wm(state, episode_start)

        aux = wm_out["aux"]
        cpc = self._squeeze_time(aux["contrastive_tgt_emb"])
        sfa = self._squeeze_time(aux["sfa"])
        base_features = torch.cat([wm_out["state_last"], cpc, sfa], dim=-1)
        surprise = self._cpc_surprise(cpc.detach(), episode_start)
        base_features = base_features.detach()
        sfa = sfa.detach()

        prev_goal, steps_since_switch, progress = self.agent_high.goal_context(sfa, episode_start)
        batch_size = sfa.shape[0]
        zeros = sfa.new_zeros((batch_size, 1))
        high_state = torch.cat([base_features, prev_goal,
                                surprise,
                                progress, steps_since_switch, ], dim=-1,)
        sampled_goal = self.agent_high.get_action(high_state, episode_start).detach()
        if self._in_low_warmup():
            goal = torch.zeros_like(sfa)
            goal_valid = zeros
        else:
            goal = sampled_goal
            goal_valid = torch.ones_like(zeros)

        low_state = self._build_low_state(base_features, sfa, goal, goal_valid)
        actions = self.agent_low.get_action(low_state, episode_start)
        self.agent_high.store_prediction_context(wm_out["state_last"], actions)

        wm_states = {
            "sensor": state["sensor"].detach().to("cpu"),
            "heading_idx": state["heading_idx"].detach().to("cpu"),
            "location": state["location"].detach().to("cpu"),
            "policy_state": low_state.detach().to("cpu"),
        }
        actions_cpu = actions.detach().to("cpu")
        log_probs_cpu = torch.zeros((actions_cpu.shape[0],))
        entropy_cpu = torch.zeros((actions_cpu.shape[0], 1))
        self._wm_pool.add_transition_batch(wm_states, actions_cpu, log_probs_cpu, entropy_cpu)
        self.record_sampled_actions(actions)
        return actions
    
    def update(self, rewards, dones, info=None, **kwargs):
        """
        Process environment response, trigger learning once episode accumulation is completed.
    
        Overall process:
        1. Store low/high policy episodes and a separate WM rollout batch.
        2. Train the WM on recent on-policy episodes mixed with replay.
        3. Compute WM intrinsic rewards on the recent batch.
        4. Train low level with one combined policy step:
           - PPO reward is intrinsic-only during warmup.
           - PPO reward is online-goal progress + small intrinsic bonus after warmup.
           - Hindsight ranking loss compares achieved future-SFA goals against random
             negative SFA goals from the current batch.
             future achieved SFA is used as the hindsight goal input to the low-level policy. The objective says: “the action actually taken at t should be more likely under the future-
  achieved goal than under a random unrelated goal.
        5. Train high level:
           - during warmup, skip high PPO and supervise the goal head to future SFA
             targets on all valid timesteps.
           - after warmup, train switch/goal PPO rewards and switch-gated goal
             hindsight weighted by intrinsic segment reward.
        """
        env_rewards = self.env_reward_scale * rewards
        for env_idx in range(self.num_envs):
            self.rl_add_reward(env_idx, env_rewards[env_idx])
            self._wm_pool.add_reward(env_idx, env_rewards[env_idx].detach().to("cpu"))
        self.process_dones(dones)
        self._wm_pool.process_dones(dones)

        if not self.should_learn():
            return False

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
            key_padding_mask=obs_new["key_padding_mask"])
        
        episode_novelty = self._compute_episode_novelty(
            state_seq=embs,
            key_padding_mask=obs_new["key_padding_mask"])
        
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

        cpc_error_after = self._evaluate_step_cpc_error_no_grad(self.wm_model, obs_new, targets_new)
        sensor_error_after = self._evaluate_step_sensor_error_no_grad(self.wm_model, obs_new, targets_new)
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
            cluster_dist_novelty=None,
            embedding_prediction_error=emb_pred_error)

        in_warmup = self._in_low_warmup()
        low_goal_coef = 0.0 if in_warmup else 1.0
        low_intrinsic_coef = 1.0 if in_warmup else 0.1
        low_episodes = [ep for ep in self.agent_low.get_train_episodes() if len(ep) >= 2]
        low_rewards = [
            self._compute_low_actual_rewards(
                ep,
                intrinsic_rewards[ep_idx],
                goal_coef=low_goal_coef,
                intrinsic_coef=low_intrinsic_coef,
            )
            for ep_idx, ep in enumerate(low_episodes)
        ]
        self._log_low_reward_components(
            low_episodes,
            intrinsic_rewards,
            goal_coef=low_goal_coef,
            intrinsic_coef=low_intrinsic_coef,
        )
        self._overwrite_low_rewards(low_episodes, low_rewards)
        if low_episodes:
            low_h_states, low_h_actions, low_h_weights, low_h_negative_states = self._compute_low_hindsight(low_episodes, intrinsic_rewards)
            self.agent_low.learn_from_episodes(
                low_episodes,
                hindsight=(low_h_states, low_h_actions, low_h_weights, low_h_negative_states),
            )

        high_episodes = [ep for ep in self.agent_high.get_train_episodes() if len(ep) >= 2]
        high_goal_rewards = [
            self._compute_high_goal_rewards(ep, intrinsic_rewards[ep_idx])
            for ep_idx, ep in enumerate(high_episodes)
        ]
        high_switch_rewards = [self._compute_high_switch_rewards(ep) for ep in high_episodes]
        high_goal_targets, high_goal_weights = self._compute_high_goal_hindsight(high_episodes, intrinsic_rewards)
        self._overwrite_high_rewards(high_episodes, high_goal_rewards, high_switch_rewards)

        if high_episodes:
            if in_warmup:
                warmup_targets, warmup_weights = self._compute_high_warmup_targets(high_episodes)
                old_coef = self.agent_high.goal_hindsight_coef
                self.agent_high.goal_hindsight_coef = self.high_warmup_goal_coef
                self.agent_high.train_goal_hindsight(
                    high_episodes,
                    warmup_targets,
                    warmup_weights,
                    num_epochs=self.high_warmup_goal_epochs,
                    switch_only=False,
                )
                self.agent_high.goal_hindsight_coef = old_coef
            else:
                self.agent_high.learn_from_episodes(high_episodes)
                self.agent_high.train_goal_hindsight(high_episodes, high_goal_targets, high_goal_weights)

        scalar_sums = {
            "joint/loss_total": total_loss_sum,
            "reward/intrinsic_mean": float(intrinsic_rewards.mean().detach().cpu()) * wm_updates_for_logging,
            "wm/loss/total": wm_loss_sum,
        }
        extra_scalars = {
            "wm/fixed": float(self.wm_fixed),
            "double/warmup_active": float(in_warmup),
            "double/low_warmup_updates": float(self.low_warmup_updates),
            "double/high_warmup_goal_coef": float(self.high_warmup_goal_coef),
            "double/high_warmup_goal_epochs": float(self.high_warmup_goal_epochs),
            "wm/actual_updates": float(wm_updates),
            "wm/replay_size": float(len(self._wm_pool.episode_pool)),
            "wm/replay_sampled_episodes": float(len(replay_episodes)),
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
            info=info,
        )

        self.version += 1
        self.clear_completed()
        self._wm_pool.clear_completed()

        return True
