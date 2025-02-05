import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import TransformedDistribution

np = numpy

from agent_reinf import Agent
from pool import *


class ReinforceBase(Agent):
    def __init__(
        self,
        policy,
        sampler,
        policy_lr=0.001,
        value_lr=0.001,
        num_envs=8,
        discount=0.99,
        device=torch.device("cpu"),
        logger=None,
    ):
        self.num_envs = num_envs
        self.policy = policy
        self.sampler = sampler
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.discount = discount
        self.mean_reward = -10000
        self.mean_std = 0
        self.device = device
        self.version = 0
        self.entropy_coef = 0.05
        self.entropy_thresh = 0.5
        logger.add_hparams(dict(policy_lr=policy_lr, discount=discount, entropy_coef=self.entropy_coef,
                                entropy_thresh=self.entropy_thresh),
                           dict())
        super().__init__(logger=logger)

    def episode_start(self):
        self.reset_episodes()
        self.active_envs = np.ones(self.num_envs, dtype=bool)

    def get_action(self, state, done):
        active_mask = ~done
        if not any(active_mask):
            return np.zeros((self.num_envs,) + self.policy.action_shape)

        active_states = torch.FloatTensor(state[active_mask])
        actions, log_probs, dist = self.sampler(self.policy, active_states)
        if isinstance(dist, TransformedDistribution):
            entropy = dist.base_dist.entropy()
        else:
            entropy = dist.entropy()
        full_actions = torch.zeros(
            (active_mask.shape[0], actions.shape[1] if len(actions.shape) > 1 else 1)
        ).to(actions)
        full_actions[active_mask] = actions.reshape(full_actions[active_mask].shape)

        active_env_indices = np.where(active_mask)[0]
        for idx, env_idx in enumerate(active_env_indices):
            self.add_transition(env_idx, active_states[idx], actions[idx], log_probs[idx], entropy[idx])

        return full_actions.flatten()

    def update(self, obs, actions, rewards, dones, next_obs):
        for env_idx in range(self.num_envs):
            self.add_reward(env_idx, rewards[env_idx])

        self.process_dones(dones)

        if self.should_learn():
            episodes = self.get_train_episodes()
            self.learn_from_episodes(episodes)
            self.print_episode_stats(self.get_completed_episodes())
            self.version += 1  # Increment version before clearing
            self.clear_completed()
            return True
        return False

    def print_episode_stats(self, completed_episodes):
        if not completed_episodes:
            return

        # Calculate episode lengths and returns
        episode_lengths = []
        episode_returns = []

        for episode in completed_episodes:
            episode_lengths.append(len(episode))
            # Sum up rewards for the episode
            episode_return = sum(reward for _, _, _,_, reward in episode)
            episode_returns.append(episode_return)

        # Calculate statistics
        mean_length = np.mean(episode_lengths)
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        self.logger.log_scalar(f"Average episode length",  mean_length)
        self.logger.log_scalar(f"Average return",  mean_return)
        self.logger.log_scalar(f"return std",  std_return)
        self.logger.log_scalar(f"min return",  min(episode_returns))
        self.logger.log_scalar(f"max return",  max(episode_returns))

    def learn_from_episodes(self, episodes):
        # Extract per-episode tensors/lists
        states_list, log_probs_list, rewards_list, actions_list, entropy_list = self._extract_episode_data(episodes)
        if not states_list:
            return

        # Prepare batches: compute discounted returns, and then aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = self._prepare_batches(
            states_list, log_probs_list, rewards_list, actions_list, entropy_list
        )

        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)

        # Log statistics
        self._log_training_stats(actions_batch)

        # Finally, train the policy (using log probs computed from current policy)
        self.train_policy(log_probs_batch, normalized_returns, entropy_batch)

    def _extract_episode_data(self, episodes):
        """
        From a list of episodes, each consisting of a sequence of (state, action, log_prob, entropy, reward),
        return four lists: states_list, log_probs_list, rewards_list, and actions_list, entropy_list.
        Each element in the returned lists corresponds to one episode.
        """
        states_list = []
        log_probs_list = []
        rewards_list = []
        actions_list = []
        entropy_list = []
        for episode in episodes:
            if not episode:
                continue

            states, log_probs, rewards, actions, entropy = [], [], [], [], []
            for (s, a, log_p, e, r) in episode:
                states.append(s)
                log_probs.append(log_p)
                rewards.append(r)
                actions.append(a)
                entropy.append(e)

            # Stack (or convert) collected data so that each episode becomes a tensor
            states_list.append(torch.stack(states))
            log_probs_list.append(torch.stack(log_probs))
            rewards_list.append(torch.tensor(rewards, dtype=torch.float32))
            actions_list.append(torch.stack(actions))
            entropy_list.append(torch.stack(entropy))
        return states_list, log_probs_list, rewards_list, actions_list, entropy_list

    def _compute_discounted_returns(self, rewards):
        """
        Given a 1D tensor of rewards for one episode,
        compute the discounted return using self.discount.
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.discount * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def _prepare_batches(self, states_list, log_probs_list, rewards_list, actions_list, entropy_list):
        """
        From lists of per-episode data, compute discounted returns for each episode
        then concat all episodes to form large batches.
        Also, recompute log probabilities from current policy for consistency.
        """
        returns_list = [self._compute_discounted_returns(rewards) for rewards in rewards_list]

        # Concatenate all episodes along dimension 0 (the time dimension)
        states_batch = torch.cat(states_list, dim=0).to(self.device)
        returns_batch = torch.cat(returns_list, dim=0).to(self.device)
        actions_batch = torch.cat(actions_list, dim=0).to(self.device)
        log_probs_batch = torch.cat(log_probs_list, dim=0).to(self.device).reshape(returns_batch.shape)
        entropy_batch = torch.cat(entropy_list, dim=0).to(self.device)

        return states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch

    def _normalize_returns(self, returns_batch):
        """
        Normalize returns using a running mean and std.
        """
        # Update running mean if it's the first update
        if self.mean_reward == -10000:
            self.mean_reward = returns_batch.mean()

        n = 5.0
        self.mean_reward = (
            self.mean_reward * (n - 1) / n + (returns_batch.mean() / n)
        )
        self.mean_std = (
            self.mean_std * (n - 1) / n + returns_batch.std() / n
        )

        returns_std = returns_batch.std() + 1e-8  # prevent division by zero
        return (returns_batch - self.mean_reward) / returns_std

    def _log_training_stats(self, actions_batch):
        """
        Log various statistics about the actions and returns.
        """
        self.logger.log_scalar(
            "actions mean",
            actions_batch.to(torch.float32).mean()
        )
        self.logger.log_scalar(
            "action std",
            actions_batch.to(torch.float32).std()
        )

    def train_policy(self, log_probs, returns, entropy=torch.zeros(1)):
        assert log_probs.shape == returns.shape, "Expected same shape for log_probs and returns!"
        assert log_probs.shape == entropy.shape, "Expected same shape for log_probs and entropy!"
        self.optimizer_policy.zero_grad()
        m = (entropy < self.entropy_thresh)
        e_loss =  -(self.entropy_coef * entropy * m).to(log_probs).mean()
        policy_loss = -(log_probs * returns).mean() + e_loss
        if policy_loss > 100:
            import pdb;pdb.set_trace()
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.4)
        grads = []
        # Print policy gradients
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
        self.logger.log_scalar(f"policy grad std :", np.std(grads))
        self.logger.log_scalar(f"entropy mean :", entropy.mean())
        self.logger.log_scalar(f"entropy loss :", e_loss)
        self.optimizer_policy.step()
        self.logger.log_scalar("policy loss:", policy_loss.item())



class Reinforce(ReinforceBase, EpisodesPoolMixin):

    def __init__(
        self,
        policy,
        sampler,
        policy_lr=0.001,
        num_envs=8,
        discount=0.99,
        device=torch.device("cpu"),
        logger=None,
    ):
        super().__init__(policy, sampler, policy_lr=policy_lr,
                         num_envs=num_envs, discount=discount,
                         device=device, logger=logger)


class ReinforceWithOldEpisodes(ReinforceBase, EpisodesOldPoolMixin):

    def learn_from_episodes(self, episodes):
        # Extract per-episode tensors/lists
        states_list, log_probs_list, rewards_list, actions_list, dist = self._extract_episode_data(episodes)
        if not states_list:
            return

        # Prepare batches: compute discounted returns, and then aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = self._prepare_batches(
            states_list, log_probs_list, rewards_list, actions_list
        )

        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)

        # Log statistics
        self._log_training_stats(actions_batch)
        _, _, dist = self.sampler(self.policy, states_batch)
        log_p1 = dist.log_prob(actions_batch).reshape(returns_batch.shape)
        # Finally, train the policy (using log probs computed from current policy)
        self.train_policy(log_p1, normalized_returns)


class ReinforcePred(Reinforce):
    def __init__(
        self,
        policy,
        sampler,
        predictor,
        predictor_lr=0.001,
        num_envs=8,
        discount=0.99,
        device=torch.device("cpu"),
        logger=None,
        policy_lr=0.001
    ):
        # Initialize using the parent class which setups episodes pooling etc.
        super().__init__(
            policy,
            sampler,
            policy_lr=policy_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger,
        )
        # Store the predictor network and its optimizer
        self.predictor = predictor.to(self.device)
        self.optimizer_predictor = optim.Adam(self.predictor.parameters(), lr=predictor_lr)
        # Running average of prediction error (for bonus computation)
        self.avg_pred_error = 0.0

    def update(self, obs, actions, rewards, dones, next_obs):
        """
        For each environment, use the last transition (state,action) to predict next state,
        compute the prediction error (MSE), update the predictor network, then add
        an exploration bonus equal to (prediction error - running average error) to each reward.
        Finally, call the parent's update() method with modified rewards.
        """
        bonus_rewards = np.zeros_like(rewards)
        available_indices = []
        pred_input_states = []
        pred_input_actions = []
        target_next_states = []

        # For each environment with at least one stored transition:
        for env_idx in range(self.num_envs):
            if len(self.episodes[env_idx]) > 0:
                # The stored transition is a tuple: (state, action, log_prob, reward)
                # We use the state and action from the last transition.
                transition = self.episodes[env_idx][-1]
                state_saved, action_saved, *_ = transition
                available_indices.append(env_idx)
                pred_input_states.append(state_saved)
                pred_input_actions.append(action_saved)
                # Convert next_obs for the environment to a tensor and send to device.
                target_next_states.append(torch.FloatTensor(next_obs[env_idx]).to(self.device))

        if available_indices:
            states_tensor = torch.stack(pred_input_states)  # shape: [batch, state_dim]
            actions_tensor = torch.stack(pred_input_actions)  # shape: [batch, action_dim]
            # Concatenate state and action along feature dimension.
            predictor_input = torch.cat([states_tensor.to(actions_tensor.device), actions_tensor], dim=-1)
            predicted_next = self.predictor(predictor_input)
            targets = torch.stack(target_next_states)
            # Compute per-sample mean squared error (MSE)
            loss_tensor = F.mse_loss(predicted_next, targets, reduction='none')
            # Average error per sample (mean over state-dimensions)
            pred_errors = loss_tensor.mean(dim=1)
            pred_loss = pred_errors.mean()

            # Update the predictor network
            self.optimizer_predictor.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 10.0)
            self.optimizer_predictor.step()

            # Update the running average (simple exponential moving average)
            mean_error = pred_errors.mean().item()
            if self.avg_pred_error == 0:
                self.avg_pred_error = mean_error
            else:
                self.avg_pred_error = 0.99 * self.avg_pred_error + 0.01 * mean_error

            # Compute bonus reward for each sample
            bonus = pred_errors.detach().cpu().numpy() - self.avg_pred_error
            for i, env_idx in enumerate(available_indices):
                bonus_rewards[env_idx] = bonus[i]

        # Log the running average prediction error.
        self.logger.log_scalar("avg_pred_error", self.avg_pred_error)

        # Combine the environment reward with the exploration bonus.
        mod_rewards = rewards + bonus_rewards

        # Pass the modified rewards to the parent update() method.
        return super().update(obs, actions, mod_rewards, dones, next_obs)
