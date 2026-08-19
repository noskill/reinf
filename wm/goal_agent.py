import numpy as np
import math
import torch
from torch.distributions import Independent, Normal, kl_divergence

from ppo import PPO
from util import EpisodeBatch, compute_returns_list, flatten_padded, gae, to_device



class GoalAgent(PPO):
    """PPO policy for option-like SFA goals.

    The policy outputs two action heads through GoalSwitchSampler:
    - goal: Normal sample in absolute SFA space or as an SFA delta
    - switch: Bernoulli switch flag, where 1 samples/uses a new goal

    The executed high-level goal is:
        active_goal_t = prev_goal * (1 - switch) + sampled_goal * switch

    With goal_output_delta enabled, sampled_goal is first converted to
    current_sfa + sampled_delta while replay keeps the sampled delta action.

    Goal and switch heads can receive separate rewards and therefore use
    separate log-probs, returns and advantages.
    """

    def __init__(self, *args, **kwargs):
        self.goal_hindsight_coef = kwargs.pop("goal_hindsight_coef", 0.05)
        self.goal_hindsight_epochs = kwargs.pop("goal_hindsight_epochs", 4)
        self.goal_output_delta = kwargs.pop("goal_output_delta", False)
        self.goal_target_entropy = kwargs.pop("goal_target_entropy", None)
        self.switch_target_entropy = float(kwargs.pop("switch_target_entropy", 0.65))
        if not 0.0 <= self.switch_target_entropy <= math.log(2.0):
            raise ValueError("switch_target_entropy must be in [0, log(2)]")
        super().__init__(*args, **kwargs)
        self.data_types = [
            "states",
            "actions",
            "log_probs",
            "entropy",
            "goal_rewards",
            "switch_rewards",
        ]
        self.pad_fields = [
            "states",
            "actions",
            "log_probs",
            "goal_rewards",
            "switch_rewards",
            "goal_returns",
            "switch_returns",
        ]
        self._active_goal = None
        self._last_switch = None
        self.steps_since_switch = None
        self._goal_start = None
        self._goal_reference = None
        self._prev_h = None
        self._prev_action = None

    def _goal_target_entropy(self, goal_dim, device, dtype):
        target = goal_dim if self.goal_target_entropy is None else self.goal_target_entropy
        return torch.tensor(target, device=device, dtype=dtype)

    def compute_entropy_loss(self, entropy, device, dtype, target_entropy=None, return_parts=False):
        if target_entropy is None:
            return super().compute_entropy_loss(entropy, device, dtype, return_parts=return_parts)
        if not torch.is_tensor(target_entropy):
            target_entropy = torch.tensor(target_entropy, device=device, dtype=dtype)
        entropy = entropy.to(device=device, dtype=dtype)
        target_entropy = target_entropy.to(device=device, dtype=dtype)
        entropy_error = entropy - target_entropy
        normalizer = target_entropy.abs().clamp_min(1.0)
        entropy_loss_items = (entropy_error / normalizer).square()
        entropy_loss = entropy_loss_items.mean()
        if return_parts:
            return entropy_loss, entropy_loss_items
        return entropy_loss

    def episode_start(self):
        super().episode_start()
        self._active_goal = None
        self.steps_since_switch = None
        self._goal_start = None
        self._goal_reference = None
        self._prev_h = None
        self._prev_action = None
        
    def reset_cache(self, reset_mask):
        if self._active_goal is None:
            return
        self._active_goal[reset_mask] = 0.0
        self.steps_since_switch[reset_mask] = 0.0
        self._goal_start[reset_mask] = 0.0
        if self._prev_h is not None:
            self._prev_h[reset_mask] = 0.0
            self._prev_action[reset_mask] = 0.0

    def store_prediction_context(self, h, action):
        assert h.dim() == 2, f"Expected h [B,H], got {tuple(h.shape)}"
        assert action.dim() in (1, 2), f"Expected action [B] or [B,A], got {tuple(action.shape)}"
        assert h.shape[0] == action.shape[0], "h and action batch sizes must match"
        self._prev_h = h.detach()
        self._prev_action = action.detach()

    def prediction_context(self):
        return self._prev_h, self._prev_action

    def init_goal_cache(self, reference_goal):
        if self._active_goal is None or self._active_goal.shape != reference_goal.shape:
            self._active_goal = torch.zeros_like(reference_goal)
            self._goal_start = torch.zeros_like(reference_goal)
            self.steps_since_switch = torch.zeros(
                (reference_goal.shape[0], 1),
                dtype=reference_goal.dtype,
                device=reference_goal.device,
            )

    def goal_progress(self, current):
        if self._active_goal is None or self._goal_start is None:
            return current.new_zeros((current.shape[0], 1))
        dist_start = torch.linalg.vector_norm(self._goal_start - self._active_goal, dim=-1, keepdim=True)
        dist_current = torch.linalg.vector_norm(current - self._active_goal, dim=-1, keepdim=True)
        return ((dist_start - dist_current) / dist_start.clamp_min(1e-6)).clamp(-1.0, 1.0)

    def goal_context(self, current, episode_start):
        reset_mask = torch.as_tensor(episode_start, dtype=torch.bool, device=current.device).view(-1)
        assert reset_mask.numel() == current.shape[0], "episode_start must match batch size"
        self.init_goal_cache(current)
        if reset_mask.any():
            self.reset_cache(reset_mask)
        self._goal_reference = current.detach()
        progress = self.goal_progress(current)
        if reset_mask.any():
            progress[reset_mask] = 0.0
        return (
            self._active_goal.detach(),
            self.steps_since_switch.detach(),
            progress.detach(),
        )

    def get_action(self, state, episode_start):
        policy = self.get_policy_for_action()
        active_states = self.process_states(state, episode_start)
        policy_kwargs = dict(reset_mask=episode_start)
        policy_params = policy(active_states, **policy_kwargs)
        actions, log_probs, dist = self.sampler(policy_params)

        goal_action = actions["goal"]
        switch = actions["switch"].to(goal_action)
        reset_mask = torch.as_tensor(
            episode_start,
            dtype=torch.bool,
            device=goal_action.device,
        ).view(-1)
        assert reset_mask.numel() == goal_action.shape[0], "episode_start must match batch size"

        self.init_goal_cache(goal_action)

        force_switch_mask = reset_mask | (self.steps_since_switch.view(-1) >= 10.0)
        if force_switch_mask.any():
            # At episode boundaries there is no previous goal to keep. Also cap
            # stale goals so the untrained switch policy cannot keep one target
            # forever during early runs.
            switch = switch.clone()
            switch[force_switch_mask] = 1.0

            # Store the actually executed switch action, not the raw sampled one,
            # so replayed PPO log-probs match behaviour collection.
            actions = dict(actions)
            actions["switch"] = switch

            # Since we overrode sampled switch actions, recompute
            # its Bernoulli log-prob under the original switch distribution.
            log_probs = dict(log_probs)
            switch_logp = dist["switch"].log_prob(switch)
            if switch_logp.shape[-1:] == (1,):
                switch_logp = switch_logp.squeeze(-1)
            log_probs["switch"] = switch_logp

        assert self._goal_reference is not None, "goal_context must be called before get_action"
        sampled_goal = self._goal_action_to_absolute(self._goal_reference, goal_action)
        active_goal = self._active_goal * (1.0 - switch) + sampled_goal * switch
        self._active_goal = active_goal.detach()
        self._last_switch = switch.detach()
        self._goal_start = self._goal_start * (1.0 - switch.detach()) + self._goal_reference * switch.detach()

        entropy = self._dist_entropy(dist)
        self.steps_since_switch = (self.steps_since_switch + 1.0) * (1.0 - switch.detach())
        self.add_transition_batch(active_states, actions, log_probs, entropy)
        return active_goal

    def add_transition(self, env_idx, state, action, log_prob, entropy):
        self.episodes[env_idx].append((state, action, log_prob, entropy))

    def add_transition_batch(self, states, actions, log_probs, entropies, env_ids=None):
        assert isinstance(actions, dict), "GoalEmbeddingPolicy expects dict actions"
        assert isinstance(log_probs, dict), "GoalEmbeddingPolicy expects dict log_probs"
        assert isinstance(entropies, dict), "GoalEmbeddingPolicy expects dict entropies"
        batch_size = actions["goal"].shape[0]

        for key, value in actions.items():
            assert value.shape[0] == batch_size, f"actions['{key}'] batch mismatch"
        for key, value in log_probs.items():
            assert value.shape[0] == batch_size, f"log_probs['{key}'] batch mismatch"
        for key, value in entropies.items():
            assert value.shape[0] == batch_size, f"entropies['{key}'] batch mismatch"
        if env_ids is not None:
            assert env_ids.shape[0] == batch_size, "env_ids length must match batch size"

        for idx in range(batch_size):
            env_idx = idx if env_ids is None else int(env_ids[idx].item())
            assert 0 <= env_idx < self.num_envs, f"env_idx {env_idx} out of range [0, {self.num_envs})"
            if isinstance(states, dict):
                state = {key: value[idx] for key, value in states.items()}
            else:
                state = states[idx]
            action = {key: value[idx] for key, value in actions.items()}
            log_prob = {key: value[idx] for key, value in log_probs.items()}
            entropy = {key: value[idx] for key, value in entropies.items()}
            self.add_transition(env_idx, state, action, log_prob, entropy)

    def add_reward(self, env_idx, reward):
        if not self.episodes[env_idx]:
            return
        if isinstance(reward, dict):
            goal_reward = reward["goal"]
            switch_reward = reward["switch"]
        else:
            goal_reward = reward
            switch_reward = reward
        self.episodes[env_idx][-1] += (
            torch.as_tensor(goal_reward, dtype=torch.float32),
            torch.as_tensor(switch_reward, dtype=torch.float32),
        )

    def update(self, rewards, dones, info=None, **kwargs):
        for env_idx in range(self.num_envs):
            if isinstance(rewards, dict):
                reward = {key: value[env_idx] for key, value in rewards.items()}
            else:
                reward = rewards[env_idx]
            self.add_reward(env_idx, reward)

        self.process_dones(dones)
        return False

    def learn_from_episodes(self, episodes, num_minibatches=4, virtual_imitation=None):
        batch = self._prepare_episode_batch(episodes)
        if batch is None:
            return
        self.train_policy_batch(
            batch,
            num_minibatches=num_minibatches,
            virtual_imitation=virtual_imitation,
        )

    def train_goal_hindsight(
        self,
        episodes,
        targets_per_episode,
        weights_per_episode,
        num_epochs=None,
        switch_only=True,
        train_sigma=True):
        if num_epochs is None:
            num_epochs = self.goal_hindsight_epochs
        if num_epochs < 1:
            raise ValueError(f"goal hindsight epochs must be >= 1, got {num_epochs}")
        assert len(episodes) == len(targets_per_episode) == len(weights_per_episode), "hindsight batch count mismatch"
        if not episodes:
            return

        states_per_episode = []
        switches_per_episode = []
        for episode, targets, weights in zip(episodes, targets_per_episode, weights_per_episode):
            states = torch.stack([step[0] for step in episode], dim=0)
            switches = torch.stack([step[1]["switch"] for step in episode], dim=0).view(-1)
            assert states.shape[0] == targets.shape[0] == weights.shape[0], "hindsight sequence length mismatch"
            states_per_episode.append(states)
            switches_per_episode.append(switches)

        batch = EpisodeBatch({
            "states": states_per_episode,
            "switches": switches_per_episode,
            "goal_targets": targets_per_episode,
            "goal_weights": weights_per_episode,
        }).to(self.device)
        padded, padding_mask, _ = batch.pad(fields=["states", "switches", "goal_targets", "goal_weights"])
        padding_mask = padding_mask.to(self.device)
        states_padded = to_device(padded["states"], self.device)
        targets_padded = padded["goal_targets"].to(self.device)
        switches_flat = flatten_padded(padded["switches"].unsqueeze(-1).to(self.device), padding_mask).squeeze(-1)
        weights_flat = flatten_padded(padded["goal_weights"].unsqueeze(-1).to(self.device), padding_mask).squeeze(-1)
        selected = weights_flat > 0.0
        if switch_only:
            selected = selected & (switches_flat > 0.5)
        if not selected.any():
            return

        weights_selected = weights_flat[selected].detach().clamp(0.0, 5.0)
        assert hasattr(self.sampler, "goal_sampler"), "hindsight goal training expects GoalSwitchSampler"

        goal_dim = targets_padded.shape[-1]
        state_sfa = self._current_sfa(states_padded, goal_dim)
        target_actions_padded = self._absolute_goal_to_action(states_padded, targets_padded)
        target_flat = flatten_padded(targets_padded, padding_mask).detach()
        target_action_flat = flatten_padded(target_actions_padded, padding_mask).detach()
        state_sfa_flat = flatten_padded(state_sfa, padding_mask).detach()
        target_selected = target_flat[selected]
        target_action_selected = target_action_flat[selected]
        delta_selected = target_selected - state_sfa_flat[selected]
        sigma_target = delta_selected.std(dim=0, unbiased=False).clamp_min(1e-4).detach()
        
        for epoch_idx in range(num_epochs):
            self.optimizer_policy.zero_grad()
            params = self.policy(states_padded, reset_mask=None, key_padding_mask=padding_mask)
            _, _, dist = self.sampler.goal_sampler(
                params["goal"],
                actions=target_actions_padded,
                return_distribution=True,
            )
            mu_flat = self._goal_mu(dist, padding_mask)
            sigma_flat = self._goal_sigma(dist, padding_mask)
            assert mu_flat is not None and sigma_flat is not None, "goal hindsight regression expects Normal goal dist"
            mu_selected = mu_flat[selected]
            sigma_selected = sigma_flat[selected]

            mu_loss_items = (
                (mu_selected - target_action_selected).pow(2).mean(dim=-1)
                * weights_selected
            )
            sigma_loss_items = (sigma_selected - sigma_target).pow(2).mean(dim=-1) * weights_selected
            loss = self.goal_hindsight_coef * mu_loss_items.mean()
            if train_sigma:
                loss = loss + self.goal_hindsight_coef * sigma_loss_items.mean()

            if epoch_idx == 0:
                grad_losses = {
                    "goal_hindsight_mu": self._grad_probe_loss(self.goal_hindsight_coef * mu_loss_items),
                }
                if train_sigma:
                    grad_losses["goal_hindsight_sigma"] = self._grad_probe_loss(
                        self.goal_hindsight_coef * sigma_loss_items
                    )
                self._log_loss_grad_stats(grad_losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.optimizer_policy.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        last_mu_target_dist = torch.linalg.vector_norm(
                mu_selected - target_action_selected,
                dim=-1,
            ).mean().detach()
        self.logger.log_scalar("high/hindsight/loss", loss.item())
        self.logger.log_scalar(
            "high/hindsight/loss_sigma",
            sigma_loss_items.detach().mean().item() if train_sigma else 0.0,
        )
        self.logger.log_scalar("high/hindsight/mu_target_dist", last_mu_target_dist.item())
        self.logger.log_scalar("high/hindsight/std", sigma_selected.mean().detach().item())
        self.logger.log_scalar("high/hindsight/target_std", sigma_target.mean().item())
        self.logger.log_scalar("high/hindsight/selected_frac", selected.float().mean().item())

    def _select_goal_policy_params(self, episodes, step_indices, policy):
        assert len(episodes) == len(step_indices), \
            "Goal policy episode and step counts must match"
        assert episodes, "Goal policy selection requires at least one episode"

        states_per_episode = [
            torch.stack([step[0] for step in episode], dim=0)
            for episode in episodes
        ]
        batch = EpisodeBatch({"states": states_per_episode}).to(self.device)
        padded, padding_mask, lengths = batch.pad(fields=["states"])
        states_padded = padded["states"].to(self.device)
        padding_mask = padding_mask.to(self.device)
        lengths = lengths.to(self.device)
        step_indices = torch.as_tensor(step_indices, dtype=torch.long, device=self.device)
        assert step_indices.shape == lengths.shape, \
            f"Expected one goal step per episode, got {step_indices.shape}"
        assert ((step_indices >= 0) & (step_indices < lengths)).all(), \
            "Goal step index is outside its episode"

        params = policy(
            states_padded,
            reset_mask=None,
            key_padding_mask=padding_mask,
        )
        assert isinstance(params, dict) and "goal" in params, \
            "Goal policy selection expects goal parameters"
        goal_params = params["goal"]
        assert goal_params.dim() == 3, \
            f"Expected padded goal parameters [B,T,2D], got {goal_params.shape}"
        row_indices = torch.arange(lengths.shape[0], device=self.device)
        selected_states = states_padded[row_indices, step_indices]
        selected_goal_params = goal_params[row_indices, step_indices]
        return selected_states, selected_goal_params

    def _current_sfa(self, states, goal_dim):
        assert states.dim() in (2, 3), \
            f"Expected high-level states [B,D] or [B,T,D], got {states.shape}"
        current_sfa = states[..., -(2 * goal_dim + 3):-(goal_dim + 3)]
        assert current_sfa.shape[-1] == goal_dim, \
            f"Expected current SFA dim {goal_dim}, got {current_sfa.shape}"
        return current_sfa

    def _goal_action_to_absolute(self, states_or_sfa, goal_action):
        """
        if policy outputs delta apply it, otherwise it's just goal_action
        """
        if not self.goal_output_delta:
            return goal_action
        current_sfa = states_or_sfa
        if current_sfa.shape[-1] != goal_action.shape[-1]:
            current_sfa = self._current_sfa(states_or_sfa, goal_action.shape[-1])
        while current_sfa.dim() < goal_action.dim():
            current_sfa = current_sfa.unsqueeze(-2)
        assert current_sfa.shape[:-2] == goal_action.shape[:-2], \
            f"Current SFA batch {current_sfa.shape} does not match goal action {goal_action.shape}"
        assert current_sfa.shape[-2] in (1, goal_action.shape[-2]), \
            f"Current SFA candidates {current_sfa.shape} do not match goal action {goal_action.shape}"
        absolute_goal = current_sfa + goal_action
        assert absolute_goal.shape == goal_action.shape
        return absolute_goal

    def _absolute_goal_to_action(self, states, absolute_goal):
        if not self.goal_output_delta:
            return absolute_goal
        current_sfa = self._current_sfa(states, absolute_goal.shape[-1])
        while current_sfa.dim() < absolute_goal.dim():
            current_sfa = current_sfa.unsqueeze(-2)
        assert current_sfa.shape[:-2] == absolute_goal.shape[:-2], \
            f"Current SFA batch {current_sfa.shape} does not match absolute goal {absolute_goal.shape}"
        assert current_sfa.shape[-2] in (1, absolute_goal.shape[-2]), \
            f"Current SFA candidates {current_sfa.shape} do not match goal {absolute_goal.shape}"
        goal_action = absolute_goal - current_sfa
        assert goal_action.shape == absolute_goal.shape
        return goal_action

    def sample_goal_candidates(self, episodes, step_indices, num_samples=1):
        assert num_samples >= 1, f"num_samples must be >= 1, got {num_samples}"
        with torch.no_grad():
            selected_states, selected_goal_params = self._select_goal_policy_params(
                episodes,
                step_indices,
                self.get_policy_for_action(),
            )
            samples = []
            log_probs = []
            for _ in range(num_samples):
                goal_action, log_prob, _ = self.sampler.goal_sampler(selected_goal_params)
                samples.append(self._goal_action_to_absolute(selected_states, goal_action))
                log_probs.append(log_prob)
        goals = torch.stack(samples, dim=1).detach()
        old_log_probs = torch.stack(log_probs, dim=1).detach()
        assert goals.dim() == 3, f"Expected sampled goals [B,N,G], got {goals.shape}"
        assert old_log_probs.shape == goals.shape[:2], \
            f"Expected sampled goal log-probs [B,N], got {old_log_probs.shape}"
        return goals, old_log_probs

    def train_virtual_goal_ppo(self, episodes, step_indices, goals, old_log_probs, rewards):
        assert goals.dim() == 3, f"Expected virtual goals [B,N,G], got {goals.shape}"
        assert old_log_probs.shape == goals.shape[:2], \
            "Virtual goal old log-probs must match [B,N]"
        assert rewards.shape == goals.shape[:2], \
            "Virtual goal rewards must match [B,N]"
        assert goals.shape[1] > 1, \
            "Virtual goal PPO requires multiple candidates per source state"
        assert torch.isfinite(goals).all(), "Non-finite virtual goals"
        assert torch.isfinite(old_log_probs).all(), "Non-finite virtual goal old log-probs"
        assert torch.isfinite(rewards).all(), "Non-finite virtual goal rewards"

        self.optimizer_policy.zero_grad()
        selected_states, selected_goal_params = self._select_goal_policy_params(
            episodes,
            step_indices,
            self.policy,
        )
        candidate_count = goals.shape[1]
        goal_params = selected_goal_params.unsqueeze(1).expand(-1, candidate_count, -1)
        goal_actions = self._absolute_goal_to_action(
            selected_states,
            goals.to(self.device),
        )
        _, new_log_probs, goal_dist = self.sampler.goal_sampler(
            goal_params,
            actions=goal_actions,
            return_distribution=True,
        )
        old_log_probs = old_log_probs.to(new_log_probs)
        rewards = rewards.to(new_log_probs)
        entropy = self._one_dist_entropy(goal_dist)
        assert entropy.shape == new_log_probs.shape, \
            f"Expected virtual goal entropy {new_log_probs.shape}, got {entropy.shape}"

        # Compare candidate goals sampled from the same source state. This removes
        # source-state difficulty while preserving which sampled goal made more
        # virtual progress than the alternatives.
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)).detach()
        target_entropy = self._goal_target_entropy(
            goals.shape[-1],
            entropy.device,
            entropy.dtype,
        )
        loss = self.policy_loss(
            old_log_probs,
            advantages,
            entropy,
            new_log_probs,
            target_entropy=target_entropy,
            loss_log_prefix="virtual_goal",
        )
        assert loss.shape == (), f"Expected scalar virtual goal PPO loss, got {loss.shape}"
        assert torch.isfinite(loss), "Non-finite virtual goal PPO loss"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.log_grads()
        self.optimizer_policy.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.logger.log_scalar("high/virtual_goal/loss", loss.detach())
        self.logger.log_scalar("high/virtual_goal/reward_mean", rewards.mean().detach())
        self.logger.log_scalar("high/virtual_goal/entropy", entropy.mean().detach())
        self.logger.log_scalar(
            "high/virtual_goal/advantage_abs_mean",
            advantages.abs().mean(),
        )

    def train_virtual_goal_imitation(self, episodes, step_indices, target_goals, coef=1.0, num_epochs=1):
        assert target_goals.dim() in (2, 3), \
            f"Expected virtual imitation targets [B,G] or [B,N,G], got {target_goals.shape}"
        assert target_goals.shape[0] == len(episodes), \
            "Expected one virtual imitation target per source episode"
        assert torch.isfinite(target_goals).all(), "Non-finite virtual imitation targets"
        assert coef > 0.0, f"Expected positive virtual imitation coefficient, got {coef}"
        assert num_epochs >= 1, f"Expected at least one virtual imitation epoch, got {num_epochs}"

        last_loss = None
        for _ in range(num_epochs):
            self.optimizer_policy.zero_grad()
            loss, target_goals = self.compute_virtual_goal_imitation_loss(
                episodes,
                step_indices,
                target_goals,
                coef=coef,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.log_grads()
            self.optimizer_policy.step()

            last_loss = loss.detach()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self._log_virtual_goal_imitation_metrics(
            episodes,
            step_indices,
            target_goals,
            last_loss,
        )

    def compute_virtual_goal_imitation_loss(self, episodes, step_indices, target_goals,coef=1.0):
        assert target_goals.dim() in (2, 3), f"Expected virtual imitation targets [B,G] or [B,N,G], got {target_goals.shape}"
        assert target_goals.shape[0] == len(episodes), "Expected one virtual imitation target per source episode"
        assert torch.isfinite(target_goals).all(), "Non-finite virtual imitation targets"
        assert coef > 0.0, f"Expected positive virtual imitation coefficient, got {coef}"

        target_goals = target_goals.to(self.device)
        if target_goals.dim() == 2:
            target_goals = target_goals.unsqueeze(1)
        selected_states, goal_params = self._select_goal_policy_params(
            episodes,
            step_indices,
            self.policy,
        )
        goal_dist = self._goal_distribution(goal_params)
        mu = goal_dist.base_dist.loc
        assert mu.shape == target_goals.shape[:1] + target_goals.shape[2:], \
            f"Expected goal mean [B,G] for targets {target_goals.shape}, got {mu.shape}"
        target_actions = self._absolute_goal_to_action(selected_states, target_goals)
        mu_expanded = mu.unsqueeze(1).expand_as(target_actions)
        loss = coef * (mu_expanded - target_actions).pow(2).mean()
        assert loss.shape == (), f"Expected scalar virtual imitation loss, got {loss.shape}"
        assert torch.isfinite(loss), "Non-finite virtual imitation loss"
        return loss, target_goals

    def _log_virtual_goal_imitation_metrics(self, episodes, step_indices, target_goals, loss):
        with torch.no_grad():
            updated_states, updated_goal_params = self._select_goal_policy_params(
                episodes,
                step_indices,
                self.policy,
            )
            updated_mu = self._goal_distribution(updated_goal_params).base_dist.loc
            target_actions = self._absolute_goal_to_action(updated_states, target_goals)
            updated_mu_expanded = updated_mu.unsqueeze(1).expand_as(target_actions)
            target_mean = target_actions.mean(dim=1)
            oracle_mse = (target_actions - target_mean.unsqueeze(1)).pow(2).mean()
            model_mse = (updated_mu_expanded - target_actions).pow(2).mean()
            mu_target_dist = torch.linalg.vector_norm(
                updated_mu_expanded - target_actions,
                dim=-1,
            ).mean()
            mu_target_mean_dist = torch.linalg.vector_norm(
                updated_mu - target_mean,
                dim=-1,
            ).mean()
            mu_nearest_target_dist = torch.linalg.vector_norm(
                updated_mu_expanded - target_actions,
                dim=-1,
            ).min(dim=1).values.mean()
        self.logger.log_scalar("high/virtual_imitation/loss", loss)
        self.logger.log_scalar(
            "high/virtual_imitation/mu_target_dist",
            mu_target_dist,
        )
        self.logger.log_scalar("high/virtual_imitation/model_mse", model_mse)
        self.logger.log_scalar("high/virtual_imitation/oracle_mse", oracle_mse)
        self.logger.log_scalar(
            "high/virtual_imitation/excess_mse",
            model_mse - oracle_mse,
        )
        self.logger.log_scalar(
            "high/virtual_imitation/mu_target_mean_dist",
            mu_target_mean_dist,
        )
        self.logger.log_scalar(
            "high/virtual_imitation/mu_nearest_target_dist",
            mu_nearest_target_dist,
        )

    def train_goal_constraint(self, episodes, step_indices, constraint_fn, max_kl, lr_scale, num_goal_samples):
        assert max_kl > 0.0, f"Expected positive goal constraint max KL, got {max_kl}"
        assert lr_scale > 0.0, f"Expected positive goal constraint LR scale, got {lr_scale}"
        assert num_goal_samples >= 1, \
            f"Expected at least one goal constraint sample, got {num_goal_samples}"
        selected_states, initial_goal_params = self._select_goal_policy_params(
            episodes,
            step_indices,
            self.policy,
        )
        old_goal_dist = self._goal_distribution(initial_goal_params.detach())
        expanded_goal_params = initial_goal_params.unsqueeze(1).expand(
            -1,
            num_goal_samples,
            -1,
        )
        _, sample_noise = self._sample_goal_with_noise(expanded_goal_params)
        sampled_goal, _ = self._sample_goal_with_noise(
            expanded_goal_params,
            noise=sample_noise,
        )
        expanded_states = selected_states.unsqueeze(1).expand(
            -1,
            num_goal_samples,
            -1,
        )
        flat_states = expanded_states.reshape(
            selected_states.shape[0] * num_goal_samples,
            selected_states.shape[1],
        )
        sampled_absolute_goal = self._goal_action_to_absolute(
            expanded_states,
            sampled_goal,
        )
        flat_goals = sampled_absolute_goal.reshape(
            selected_states.shape[0] * num_goal_samples,
            sampled_absolute_goal.shape[-1],
        )
        loss, expected_distance_before = constraint_fn(flat_states, flat_goals)
        assert loss.shape == (), f"Expected scalar goal constraint loss, got {loss.shape}"
        assert torch.isfinite(loss), "Non-finite goal constraint loss"
        assert loss.requires_grad, "Goal constraint loss must backpropagate to the high policy"

        policy_parameters = list(self.policy.parameters())
        old_parameters = [parameter.detach().clone() for parameter in policy_parameters]
        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_parameters, 1.0)
        self.log_grads()
        policy_lrs = [group["lr"] for group in self.optimizer_policy.param_groups]
        try:
            for group, policy_lr in zip(self.optimizer_policy.param_groups, policy_lrs):
                group["lr"] = policy_lr * lr_scale
            self.optimizer_policy.step()
        finally:
            for group, policy_lr in zip(self.optimizer_policy.param_groups, policy_lrs):
                group["lr"] = policy_lr

        new_parameters = [parameter.detach().clone() for parameter in policy_parameters]
        projection_scale = 1.0
        with torch.no_grad():
            _, updated_goal_params = self._select_goal_policy_params(
                episodes,
                step_indices,
                self.policy,
            )
            updated_goal_dist = self._goal_distribution(updated_goal_params)
            kl_before_projection = kl_divergence(old_goal_dist, updated_goal_dist).mean()
            kl_after_projection = kl_before_projection
            for _ in range(20):
                if kl_after_projection <= max_kl:
                    break
                projection_scale *= 0.5
                for parameter, old_parameter, new_parameter in zip(
                    policy_parameters,
                    old_parameters,
                    new_parameters,
                ):
                    parameter.copy_(
                        old_parameter + projection_scale * (new_parameter - old_parameter)
                    )
                _, updated_goal_params = self._select_goal_policy_params(
                    episodes,
                    step_indices,
                    self.policy,
                )
                updated_goal_dist = self._goal_distribution(updated_goal_params)
                kl_after_projection = kl_divergence(old_goal_dist, updated_goal_dist).mean()
            assert kl_after_projection <= max_kl, \
                f"Failed to project goal policy update below KL {max_kl}: {kl_after_projection}"

            expanded_updated_goal_params = updated_goal_params.unsqueeze(1).expand(
                -1,
                num_goal_samples,
                -1,
            )
            updated_goal, _ = self._sample_goal_with_noise(
                expanded_updated_goal_params,
                noise=sample_noise,
            )
            updated_absolute_goal = self._goal_action_to_absolute(
                expanded_states,
                updated_goal,
            )
            flat_updated_goals = updated_absolute_goal.reshape(
                selected_states.shape[0] * num_goal_samples,
                updated_absolute_goal.shape[-1],
            )
            _, expected_distance_after = constraint_fn(flat_states, flat_updated_goals)
        self.logger.log_scalar("high/achievability_constraint/loss", loss.detach())
        self.logger.log_scalar(
            "high/achievability_constraint/expected_distance_before",
            expected_distance_before,
        )
        self.logger.log_scalar(
            "high/achievability_constraint/expected_distance_after",
            expected_distance_after.detach(),
        )
        self.logger.log_scalar(
            "high/achievability_constraint/kl_before_projection",
            kl_before_projection,
        )
        self.logger.log_scalar(
            "high/achievability_constraint/kl_after_projection",
            kl_after_projection,
        )
        self.logger.log_scalar(
            "high/achievability_constraint/projection_scale",
            projection_scale,
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _goal_distribution(self, goal_params):
        goal_sampler = self.sampler.goal_sampler
        assert not goal_sampler.transform, \
            "Goal distribution diagnostics require an untransformed Normal distribution"
        reference_goal = goal_params.new_zeros(
            (*goal_params.shape[:-1], goal_sampler.action_dim)
        )
        _, _, goal_dist = goal_sampler(
            goal_params,
            actions=reference_goal,
            return_distribution=True,
        )
        assert isinstance(goal_dist, Independent), \
            f"Expected Independent goal distribution, got {type(goal_dist).__name__}"
        assert isinstance(goal_dist.base_dist, Normal), \
            f"Expected Normal goal base distribution, got {type(goal_dist.base_dist).__name__}"
        return goal_dist

    def _sample_goal_with_noise(self, goal_params, noise=None):
        goal_sampler = self.sampler.goal_sampler
        goal_dist = self._goal_distribution(goal_params)
        loc = goal_dist.base_dist.loc
        scale = goal_dist.base_dist.scale
        if noise is None:
            noise = torch.randn_like(loc)
        else:
            assert noise.shape == loc.shape, \
                f"Expected shared goal noise {loc.shape}, got {noise.shape}"
        goal = loc + scale * noise
        goal = goal.clamp(goal_sampler.a_min + 1e-4, goal_sampler.a_max - 1e-4)
        return goal, noise.detach()

    def _prepare_episode_batch(self, episodes):
        if isinstance(episodes, EpisodeBatch):
            episode_batch = episodes
        else:
            episode_batch = self._extract_episode_data(episodes)
        if episode_batch.num_episodes == 0:
            return None

        episode_batch = episode_batch.to(self.device)
        episode_batch.data["goal_returns"] = self._compute_goal_option_returns(
            episode_batch.data["goal_rewards"],
            episode_batch.data["actions"],
        )
        episode_batch.data["switch_returns"] = compute_returns_list(
            episode_batch.data["switch_rewards"],
            self.discount,
        )
        return episode_batch

    def _compute_goal_option_returns(self, rewards_per_episode, actions_per_episode):
        assert len(rewards_per_episode) == len(actions_per_episode), \
            "Goal reward/action episode counts must match"
        returns_per_episode = []
        for rewards, actions in zip(rewards_per_episode, actions_per_episode):
            switches = actions["switch"].view(-1).to(rewards)
            assert rewards.shape == switches.shape, \
                f"Expected aligned goal rewards/switches, got {rewards.shape} and {switches.shape}"
            returns = torch.zeros_like(rewards)
            next_return = rewards.new_zeros(())
            for step_idx in range(rewards.shape[0] - 1, -1, -1):
                if step_idx + 1 < rewards.shape[0] and switches[step_idx + 1] > 0.5:
                    next_return = rewards.new_zeros(())
                next_return = rewards[step_idx] + self.discount * next_return
                returns[step_idx] = next_return
            returns_per_episode.append(returns)
        return returns_per_episode

    def _extract_episode_data(self, episodes):
        data_lists = {data_type: [] for data_type in self.data_types}

        for episode in episodes:
            if not episode:
                continue
            episode_data = {data_type: [] for data_type in self.data_types}
            for step_data in episode:
                for idx, data_type in enumerate(self.data_types):
                    episode_data[data_type].append(step_data[idx])

            for data_type in self.data_types:
                first = episode_data[data_type][0]
                if data_type.endswith("rewards"):
                    data_lists[data_type].append(torch.stack(episode_data[data_type], dim=0))
                elif isinstance(first, dict):
                    keys = set(first.keys())
                    for item in episode_data[data_type]:
                        assert isinstance(item, dict) and set(item.keys()) == keys, \
                            f"All {data_type} dicts in an episode must have identical keys"
                    data_lists[data_type].append({
                        key: torch.stack([item[key] for item in episode_data[data_type]], dim=0)
                        for key in keys
                    })
                else:
                    data_lists[data_type].append(torch.stack(episode_data[data_type], dim=0))

        return EpisodeBatch(data_lists)

    def train_policy_batch(self, episode_batch, num_minibatches=4, virtual_imitation=None):
        if virtual_imitation is not None:
            assert num_minibatches == 1, \
                "Combined high PPO and virtual imitation requires one full-batch policy step"
            assert len(virtual_imitation) == 4, \
                "virtual_imitation must be (episodes, step_indices, target_goals, coef)"
        padded, padding_mask, _ = episode_batch.pad(fields=self.pad_fields)
        padding_mask = padding_mask.to(self.device)
        states_padded = to_device(padded["states"], self.device)
        actions_padded = to_device(padded["actions"], self.device)
        old_logp_padded = to_device(padded["log_probs"], self.device)
        switch_rewards_padded = padded["switch_rewards"].to(self.device)
        goal_returns_padded = padded["goal_returns"].to(self.device)
        switch_returns_padded = padded["switch_returns"].to(self.device)

        states_flat = self._flatten_states(states_padded, padding_mask)
        goal_returns_flat = flatten_padded(goal_returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        switch_returns_flat = flatten_padded(switch_returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        self.train_value_heads(states_flat, goal_returns_flat, switch_returns_flat, num_minibatches=num_minibatches)

        goal_advantages = self.compute_return_advantage_head(
            states_padded,
            goal_returns_padded,
            padding_mask,
            head="goal",
        )
        switch_advantages = self.compute_advantage_head(
            states_padded,
            switch_rewards_padded,
            padding_mask,
            head="switch",
        )
        self._log_training_stats(actions_padded)

        self.policy.load_state_dict(self.policy_old.state_dict())
        obs_for_policy = self._build_policy_obs(states_padded, padding_mask)
        update_idx = 0
        imitation_metrics = None
        for minibatch in self.iterate_minibatches(
            num_minibatches,
            padding_mask.shape[0],
            obs_for_policy,
            actions_padded,
            padding_mask,
            old_logp_padded,
            goal_advantages,
            switch_advantages,
        ):
            mb_obs, mb_actions, mb_padding, mb_old_logp, mb_goal_adv, mb_switch_adv = minibatch
            logp_new, entropy, mu = self.compute_distribution_params(mb_obs, mb_actions, mb_padding)

            goal_old = flatten_padded(mb_old_logp["goal"], mb_padding).detach()
            switch_old = flatten_padded(mb_old_logp["switch"], mb_padding).detach()
            goal_adv = flatten_padded(mb_goal_adv, mb_padding)
            switch_adv = flatten_padded(mb_switch_adv, mb_padding)
            switch_padded = mb_actions["switch"]
            assert switch_padded.dim() == 3 and switch_padded.shape[-1] == 1, \
                f'Expected switch actions [B,T,1], got {tuple(switch_padded.shape)}'
            switch = flatten_padded(switch_padded.squeeze(-1), mb_padding).to(goal_adv)
            new_goal_mask = switch

            if goal_old.numel() == 0:
                continue

            self.optimizer_policy.zero_grad()
            goal_loss, goal_loss_parts = self._masked_policy_loss(
                goal_old,
                goal_adv,
                entropy["goal"],
                logp_new["goal"],
                new_goal_mask,
                mu=mu,
                return_parts=True,
            )

            switch_loss, switch_loss_parts = self.policy_loss(
                switch_old,
                switch_adv,
                entropy["switch"],
                logp_new["switch"],
                target_entropy=self.switch_target_entropy,
                return_parts=True,
                loss_log_prefix="switch",
            )
            policy_loss = goal_loss + switch_loss

            imitation_loss = None
            if virtual_imitation is not None:
                imitation_episodes, imitation_steps, imitation_targets, imitation_coef = \
                    virtual_imitation
                imitation_loss, normalized_targets = self.compute_virtual_goal_imitation_loss(
                    imitation_episodes,
                    imitation_steps,
                    imitation_targets,
                    coef=imitation_coef,
                )
                policy_loss = policy_loss + imitation_loss
                imitation_metrics = (
                    imitation_episodes,
                    imitation_steps,
                    normalized_targets,
                    imitation_loss.detach(),
                )
                self.logger.log_scalar("combined/policy_loss", policy_loss.detach())
            if update_idx == 0:
                grad_losses = {}
                if "ppo" in goal_loss_parts:
                    grad_losses["goal_ppo"] = self._grad_probe_loss(goal_loss_parts["ppo"])
                if "entropy" in goal_loss_parts:
                    grad_losses["goal_entropy"] = self._grad_probe_loss(goal_loss_parts["entropy"])
                grad_losses["switch_ppo"] = self._grad_probe_loss(switch_loss_parts["ppo"])
                grad_losses["switch_entropy"] = self._grad_probe_loss(switch_loss_parts["entropy"])
                if imitation_loss is not None:
                    grad_losses["virtual_imitation"] = imitation_loss
                self._log_loss_grad_stats(grad_losses)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.log_grads()
            self.optimizer_policy.step()
            update_idx += 1

        if virtual_imitation is not None:
            assert update_idx == 1, \
                f"Combined high policy update must take one optimizer step, got {update_idx}"
        self.policy_old.load_state_dict(self.policy.state_dict())
        if imitation_metrics is not None:
            self._log_virtual_goal_imitation_metrics(*imitation_metrics)

    def train_value_heads(self, states_flat, goal_returns, switch_returns, value_epochs=2, num_minibatches=2):
        if goal_returns.numel() == 0:
            return
        old_num_learning_epochs = self.num_learning_epochs
        self.num_learning_epochs = value_epochs
        try:
            for minibatch in self.iterate_minibatches(
                num_minibatches,
                goal_returns.shape[0],
                states_flat,
                goal_returns,
                switch_returns,
            ):
                states_mb, goal_ret_mb, switch_ret_mb = minibatch
                self.optimizer_value.zero_grad()
                goal_loss = self.value_loss_head(states_mb, goal_ret_mb, head="goal")
                switch_loss = self.value_loss_head(states_mb, switch_ret_mb, head="switch")
                value_loss = goal_loss + switch_loss
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
                self.optimizer_value.step()
        finally:
            self.num_learning_epochs = old_num_learning_epochs

    def value_loss_head(self, states_batch, returns, head):
        pred_values = self.value_head(states_batch, head)
        value_loss = torch.nn.functional.smooth_l1_loss(pred_values, returns)
        self.logger.log_scalar(f"{head}/value_loss", value_loss)
        return value_loss

    def value_head(self, states_batch, head):
        values = self.value(states_batch)
        if isinstance(values, dict):
            values = values[head]
        elif isinstance(values, (tuple, list)):
            values = values[0 if head == "goal" else 1]
        elif values.shape[-1] == 2:
            values = values[..., 0 if head == "goal" else 1]
        if values.shape[-1:] == (1,):
            values = values.squeeze(-1)
        return values

    def compute_advantage_head(self, states_batch, rewards_batch, padding_mask, head):
        with torch.no_grad():
            values = self.value_head(states_batch, head)
            values = values * (1 - padding_mask.float())
        advantages = gae(self.discount, self.lambda_discount, rewards_batch, values)
        advantages, advantage_mean, advantage_std = self._center_advantages(
            advantages,
            padding_mask,
        )
        self.logger.log_scalar(f"{head}/advantage_mean", advantage_mean.item())
        self.logger.log_scalar(f"{head}/advantage_std", advantage_std.item())
        return advantages

    def compute_return_advantage_head(self, states_batch, returns_batch, padding_mask, head):
        with torch.no_grad():
            values = self.value_head(states_batch, head)
        advantages = returns_batch - values
        advantages, advantage_mean, advantage_std = self._center_advantages(
            advantages,
            padding_mask,
        )
        self.logger.log_scalar(f"{head}/advantage_mean", advantage_mean.item())
        self.logger.log_scalar(f"{head}/advantage_std", advantage_std.item())
        return advantages

    def _center_advantages(self, advantages, padding_mask):
        valid = ~padding_mask
        assert valid.any(), "High-level advantage batch must contain valid steps"
        valid_advantages = advantages[valid]
        advantage_mean = valid_advantages.mean()
        advantage_std = valid_advantages.std()
        centered = torch.zeros_like(advantages)
        centered[valid] = valid_advantages - advantage_mean
        return centered, advantage_mean, advantage_std

    def compute_distribution_params(self, observations, actions, key_padding_mask):
        policy_params = self.policy(observations, reset_mask=None)
        _, logp_new, dist = self.sampler(
            policy_params,
            actions=actions,
            return_distribution=True,
        )

        entropy = self._dist_entropy(dist)
        self._log_goal_dist_stats(dist["goal"], key_padding_mask, prefix="goal")
        out_logp = {
            "goal": flatten_padded(logp_new["goal"], key_padding_mask),
            "switch": flatten_padded(logp_new["switch"], key_padding_mask),
        }
        out_entropy = {
            "goal": flatten_padded(entropy["goal"], key_padding_mask),
            "switch": flatten_padded(entropy["switch"], key_padding_mask),
        }
        mu = self._goal_mu(dist["goal"], key_padding_mask)
        return out_logp, out_entropy, mu

    def _masked_policy_loss(self, log_probs_old, advantages, entropy, log_probs_new, mask, mu=None, return_parts=False):
        selected = mask > 0.5
        if not selected.any():
            loss = log_probs_new.sum() * 0.0
            if return_parts:
                return loss, {}
            return loss
        mu_selected = mu[selected] if mu is not None else None
        target_entropy = None
        if mu_selected is not None:
            target_entropy = self._goal_target_entropy(mu_selected.shape[-1], entropy.device, entropy.dtype)
        return self.policy_loss(
            log_probs_old[selected],
            advantages[selected],
            entropy[selected],
            log_probs_new[selected],
            mu=mu_selected,
            target_entropy=target_entropy,
            return_parts=return_parts,
            loss_log_prefix="goal",
        )

    def _dist_entropy(self, dist):
        return {
            "goal": self._one_dist_entropy(dist["goal"]),
            "switch": self._one_dist_entropy(dist["switch"]),
        }

    def _one_dist_entropy(self, dist):
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            if hasattr(dist, "base_dist"):
                entropy = dist.base_dist.entropy()
            else:
                raise
        if entropy.shape[-1:] == (1,):
            entropy = entropy.squeeze(-1)
        return entropy

    def _goal_mu(self, goal_dist, key_padding_mask):
        base = getattr(goal_dist, "base_dist", goal_dist)
        base_normal = getattr(base, "base_dist", base)
        if not hasattr(base_normal, "loc"):
            return None
        return flatten_padded(base_normal.loc, key_padding_mask)

    def _goal_sigma(self, goal_dist, key_padding_mask):
        base = getattr(goal_dist, "base_dist", goal_dist)
        base_normal = getattr(base, "base_dist", base)
        if not hasattr(base_normal, "scale"):
            return None
        return flatten_padded(base_normal.scale, key_padding_mask)

    def _log_goal_dist_stats(self, goal_dist, key_padding_mask, prefix="goal"):
        sigma = self._goal_sigma(goal_dist, key_padding_mask)
        if sigma is None or sigma.numel() == 0:
            return
        self.logger.log_scalar(f"{prefix}/dist_std_mean", sigma.mean().item())

    def _log_training_stats(self, actions_batch):
        goal = flatten_padded(actions_batch["goal"], torch.zeros(
            actions_batch["goal"].shape[:2],
            dtype=torch.bool,
            device=actions_batch["goal"].device,
        )) if actions_batch["goal"].dim() >= 3 else actions_batch["goal"]
        switch = actions_batch["switch"]
        self.logger.log_scalar("goal/action_mean", goal.mean())
        self.logger.log_scalar("goal/action_std", goal.std())
        self.logger.log_scalar("goal/switch_rate", switch.mean())

    def print_episode_stats(self, completed_episodes):
        if not completed_episodes:
            return
        lengths = [len(episode) for episode in completed_episodes]
        goal_returns = [sum(step[-2] for step in episode) for episode in completed_episodes]
        switch_returns = [sum(step[-1] for step in episode) for episode in completed_episodes]
        self.logger.log_scalar("Average episode length", np.mean(lengths))
        self.logger.log_scalar("goal/Average return", torch.mean(torch.stack(goal_returns)))
        self.logger.log_scalar("switch/Average return", torch.mean(torch.stack(switch_returns)))
