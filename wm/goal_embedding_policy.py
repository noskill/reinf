import numpy as np
import torch

from ppo import PPO
from util import EpisodeBatch, compute_returns_list, flatten_padded, gae, normalize_padded_returns, to_device


class GoalEmbeddingPolicy(PPO):
    """PPO policy for option-like SFA goals.

    The policy outputs two action heads through GoalSwitchSampler:
    - goal: Normal sample in embedding/SFA space
    - switch: Bernoulli switch flag, where 1 samples/uses a new goal

    The executed high-level goal is:
        active_goal_t = prev_goal * (1 - switch) + sampled_goal * switch

    Goal and switch heads can receive separate rewards and therefore use
    separate log-probs, returns and advantages.
    """

    def __init__(self, *args, **kwargs):
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

    def episode_start(self):
        super().episode_start()
        self._active_goal = None

    def get_action(self, state, episode_start):
        policy = self.get_policy_for_action()
        active_states = self.process_states(state, episode_start)
        policy_kwargs = dict(reset_mask=episode_start)
        sampler = self.sampler
        policy_params = sampler.policy_params(policy, active_states, policy_kwargs)
        actions, log_probs, dist = sampler(policy_params)

        goal = actions["goal"]
        switch = actions["switch"].to(goal)
        reset_mask = torch.as_tensor(episode_start, dtype=torch.bool, device=goal.device).view(-1)
        assert reset_mask.numel() == goal.shape[0], "episode_start must match batch size"

        if self._active_goal is None or self._active_goal.shape != goal.shape:
            self._active_goal = torch.zeros_like(goal)

        if reset_mask.any():
            # At episode boundaries there is no previous goal to keep, so force
            # the switch action to "sample/use a new goal" for reset envs.
            switch = switch.clone()
            switch[reset_mask] = 1.0

            # Store the actually executed switch action, not the raw sampled one,
            # so replayed PPO log-probs match behaviour collection.
            actions = dict(actions)
            actions["switch"] = switch

            # Since we overrode the sampled switch action on reset envs, recompute
            # its Bernoulli log-prob under the original switch distribution.
            log_probs = dict(log_probs)
            switch_logp = dist["switch"].log_prob(switch)
            if switch_logp.shape[-1:] == (1,):
                switch_logp = switch_logp.squeeze(-1)
            log_probs["switch"] = switch_logp

        active_goal = self._active_goal * (1.0 - switch) + goal * switch
        self._active_goal = active_goal.detach()
        self._last_switch = switch.detach()

        entropy = self._dist_entropy(dist)
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
        if self.should_learn():
            episodes = self.get_train_episodes()
            self.learn_from_episodes(episodes)
            self.print_episode_stats(self.get_completed_episodes())
            self.version += 1
            self.clear_completed()
            return True
        return False

    def learn_from_episodes(self, episodes, num_minibatches=4):
        batch = self._prepare_episode_batch(episodes)
        if batch is None:
            return
        self.train_policy_batch(batch, num_minibatches=num_minibatches)

    def _prepare_episode_batch(self, episodes):
        if isinstance(episodes, EpisodeBatch):
            episode_batch = episodes
        else:
            episode_batch = self._extract_episode_data(episodes)
        if episode_batch.num_episodes == 0:
            return None

        episode_batch = episode_batch.to(self.device)
        episode_batch.data["goal_returns"] = compute_returns_list(
            episode_batch.data["goal_rewards"],
            self.discount,
        )
        episode_batch.data["switch_returns"] = compute_returns_list(
            episode_batch.data["switch_rewards"],
            self.discount,
        )
        return episode_batch

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
                    data_lists[data_type].append(torch.stack(episode_data[data_type], dim=0).to(torch.float32))
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

    def train_policy_batch(self, episode_batch, num_minibatches=4):
        padded, padding_mask, _ = episode_batch.pad(fields=self.pad_fields)
        padding_mask = padding_mask.to(self.device)
        states_padded = to_device(padded["states"], self.device)
        actions_padded = to_device(padded["actions"], self.device)
        old_logp_padded = to_device(padded["log_probs"], self.device)
        goal_rewards_padded = padded["goal_rewards"].to(self.device)
        switch_rewards_padded = padded["switch_rewards"].to(self.device)
        goal_returns_padded = padded["goal_returns"].to(self.device)
        switch_returns_padded = padded["switch_returns"].to(self.device)

        states_flat = self._flatten_states(states_padded, padding_mask)
        goal_returns_flat = flatten_padded(goal_returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        switch_returns_flat = flatten_padded(switch_returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        self.train_value_heads(states_flat, goal_returns_flat, switch_returns_flat, num_minibatches=num_minibatches)

        goal_advantages = self.compute_advantage_head(
            states_padded,
            goal_rewards_padded,
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

            goal_old = self._flatten_time(mb_old_logp["goal"], mb_padding).detach()
            switch_old = self._flatten_time(mb_old_logp["switch"], mb_padding).detach()
            goal_adv = self._flatten_time(mb_goal_adv, mb_padding)
            switch_adv = self._flatten_time(mb_switch_adv, mb_padding)
            switch = self._flatten_time(mb_actions["switch"], mb_padding).to(goal_adv)
            new_goal_mask = switch

            if goal_old.numel() == 0:
                continue

            self.optimizer_policy.zero_grad()
            goal_loss = self._masked_policy_loss(
                goal_old,
                goal_adv,
                entropy["goal"],
                logp_new["goal"],
                new_goal_mask,
                mu=mu,
            )
            switch_loss = self.policy_loss(
                switch_old,
                switch_adv,
                entropy["switch"],
                logp_new["switch"],
            )
            policy_loss = goal_loss + switch_loss
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.log_grads()
            self.optimizer_policy.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

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
        return values.squeeze(-1)

    def compute_advantage_head(self, states_batch, rewards_batch, padding_mask, head):
        with torch.no_grad():
            values = self.value_head(states_batch, head)
            values = values * (1 - padding_mask.float())
        advantages = gae(self.discount, self.lambda_discount, rewards_batch, values)
        advantages, advantage_mean, advantage_std = normalize_padded_returns(advantages, padding_mask)
        self.logger.log_scalar(f"{head}/advantage_mean", advantage_mean.item())
        self.logger.log_scalar(f"{head}/advantage_std", advantage_std.item())
        return advantages

    def compute_distribution_params(self, observations, actions, key_padding_mask):
        policy_params = self.sampler.policy_params(
            self.policy,
            observations,
            dict(reset_mask=None),
        )
        _, logp_new, dist = self.sampler(
            policy_params,
            actions=actions,
            return_distribution=True,
        )
        entropy = self._dist_entropy(dist)
        out_logp = {
            "goal": self._flatten_time(logp_new["goal"], key_padding_mask),
            "switch": self._flatten_time(logp_new["switch"], key_padding_mask),
        }
        out_entropy = {
            "goal": self._flatten_time(entropy["goal"], key_padding_mask),
            "switch": self._flatten_time(entropy["switch"], key_padding_mask),
        }
        mu = self._goal_mu(dist["goal"], key_padding_mask)
        return out_logp, out_entropy, mu

    def _masked_policy_loss(self, log_probs_old, advantages, entropy, log_probs_new, mask, mu=None):
        selected = mask > 0.5
        if not selected.any():
            return log_probs_new.sum() * 0.0
        mu_selected = mu[selected] if mu is not None else None
        return self.policy_loss(
            log_probs_old[selected],
            advantages[selected],
            entropy[selected],
            log_probs_new[selected],
            mu=mu_selected,
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

    def _flatten_time(self, tensor, key_padding_mask):
        if tensor.dim() == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.dim() == 2:
            return flatten_padded(tensor.unsqueeze(-1), key_padding_mask).squeeze(-1)
        return flatten_padded(tensor, key_padding_mask).squeeze(-1)

    def _log_training_stats(self, actions_batch):
        goal = flatten_padded(actions_batch["goal"], torch.zeros(
            actions_batch["goal"].shape[:2],
            dtype=torch.bool,
            device=actions_batch["goal"].device,
        )) if actions_batch["goal"].dim() >= 3 else actions_batch["goal"]
        switch = actions_batch["switch"].to(torch.float32)
        self.logger.log_scalar("goal/action_mean", goal.to(torch.float32).mean())
        self.logger.log_scalar("goal/action_std", goal.to(torch.float32).std())
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
