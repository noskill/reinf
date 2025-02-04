import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

        full_actions = torch.zeros(
            (active_mask.shape[0], actions.shape[1] if len(actions.shape) > 1 else 1)
        ).to(actions)
        full_actions[active_mask] = actions.reshape(full_actions[active_mask].shape)

        active_env_indices = np.where(active_mask)[0]
        for idx, env_idx in enumerate(active_env_indices):
            self.add_transition(env_idx, active_states[idx], actions[idx], log_probs[idx])

        return full_actions.flatten()

    def update(self, obs, actions, rewards, dones, next_obs):
        for env_idx in range(self.num_envs):
            self.add_reward(env_idx, rewards[env_idx])

        self.process_dones(dones)

        if self.should_learn():
            episodes = self.get_completed_episodes()
            self.learn_from_episodes(episodes)
            self.print_episode_stats(episodes)
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
            episode_return = sum(reward for _, _, _, reward in episode)
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

    def train_value(self, returns, states_batch):
        # Value Network Update
        value_epochs = 2
        mini_batch_size = 256
        
        for epoch in range(value_epochs):
            indices = np.random.permutation(len(states_batch))
            for start in range(0, len(states_batch), mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                pred_values = self.value(states_batch[batch_idx].detach()).squeeze(-1)
                value_loss = F.mse_loss(pred_values, returns[batch_idx])
                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
                self.optimizer_value.step()
                
        self.logger.log_scalar(f"value loss",  value_loss)

    def learn_from_episodes(self, episodes):
        # Extract per-episode tensors/lists
        states_list, log_probs_list, rewards_list, actions_list = self._extract_episode_data(episodes)
        if not states_list:
            return

        # Prepare batches: compute discounted returns, and then aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch = self._prepare_batches(
            states_list, log_probs_list, rewards_list, actions_list
        )
        
        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)
        
        # Log statistics
        self._log_training_stats(actions_batch)
        
        # Finally, train the policy (using log probs computed from current policy)
        self.train_policy(log_probs_batch, normalized_returns)

    def _extract_episode_data(self, episodes):
        """
        From a list of episodes, each consisting of a sequence of (state, action, log_prob, reward),
        return four lists: states_list, log_probs_list, rewards_list, and actions_list.
        Each element in the returned lists corresponds to one episode.
        """
        states_list = []
        log_probs_list = []
        rewards_list = []
        actions_list = []
        
        for episode in episodes:
            if not episode:
                continue

            states, log_probs, rewards, actions = [], [], [], []
            for (s, a, log_p, r) in episode:
                states.append(s)
                log_probs.append(log_p)
                rewards.append(r)
                actions.append(a)
            
            # Stack (or convert) collected data so that each episode becomes a tensor
            states_list.append(torch.stack(states))
            log_probs_list.append(torch.stack(log_probs))
            rewards_list.append(torch.tensor(rewards, dtype=torch.float32))
            actions_list.append(torch.stack(actions))
        
        return states_list, log_probs_list, rewards_list, actions_list

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

    def _prepare_batches(self, states_list, log_probs_list, rewards_list, actions_list):
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
        
        # Recompute log probabilities according to the current policy
        _, _, dist = self.sampler(self.policy, states_batch)
        log_probs_batch = dist.log_prob(actions_batch)
        
        # Ensure log_probs_batch has shape matching returns_batch
        if log_probs_batch.shape != returns_batch.shape:
            log_probs_batch = log_probs_batch.reshape(returns_batch.shape)
        
        return states_batch, returns_batch, log_probs_batch, actions_batch

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

    def train_policy(self, log_probs, returns):
        assert log_probs.shape == returns.shape, "Expected same shape for log_probs and returns!"
        self.optimizer_policy.zero_grad()
        policy_loss = -(log_probs * returns).mean()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        grads = []
        # Print policy gradients
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
        self.logger.log_scalar(f"policy grad std :", np.std(grads))
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
   def learn_from_episodes(self, completed_episodes):
        # Prepare lists to store all data from all completed episodes
        all_states = []
        all_returns = []
        all_log_probs = []
        all_actions = []
        
        # Step 1: Compute returns per episode, store them
        for episode in completed_episodes:
            if len(episode) == 0:  # Skip empty episodes
                continue
                
            states = []
            log_probs = []
            rewards = []
            actions = []
            
            # Unpack transitions: (state, action, log_prob, reward)
            for (s, a, log_p, r) in episode:
                states.append(s)
                log_probs.append(log_p)
                rewards.append(r)
                actions.append(a)
                
            # Convert to tensors
            states = torch.stack(states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            log_probs = torch.stack(log_probs).reshape(rewards.shape)
            actions = torch.stack(actions)
            
            # Compute discounted returns Gt (backward)
            returns = []
            G = 0
            for r_t in reversed(rewards):
                G = r_t + self.discount * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            
            # Append tensors directly to lists
            all_states.append(states)
            all_returns.append(returns)
            all_log_probs.append(log_probs)
            all_actions.append(actions)
        
        # Skip update if no valid episodes
        if not all_states:
            return
        
        # Concatenate all episodes
        states_batch = torch.cat(all_states, dim=0).to(self.device)
        all_returns_cat = torch.cat(all_returns, dim=0).to(self.device)
        
        all_log_probs_cat = torch.cat(all_log_probs, dim=0).to(self.device)
        all_actions_cat = torch.cat(all_actions, dim=0).to(self.device)

        _, _, dist = self.sampler(self.policy, states_batch)
        log_p1 = dist.log_prob(all_actions_cat).reshape(all_returns_cat.shape)

        # Get distribution parameters (only needed for printing stats)
        # with torch.no_grad():
        #     _, _, transformed_dist = self.sampler(self.policy, states_batch)
        
        self.logger.log_scalar(f"actions mean", all_actions_cat.to(torch.float32).mean())
        self.logger.log_scalar("action std",  all_actions_cat.to(torch.float32).std())
        
        # Normalize returns
        if self.mean_reward == -10000:
            self.mean_reward = all_returns_cat.mean()
        
        n = 5
        self.mean_reward = (
            self.mean_reward * (n - 1) / n + (all_returns_cat.mean() / n)
        )
        self.mean_std = (
            self.mean_std * (n - 1) / n + all_returns_cat.std() / n
        )
        returns_std = all_returns_cat.std() + 1e-8

        all_returns_cat = (all_returns_cat - self.mean_reward) / returns_std
        self.train_policy(log_p1, all_returns_cat)
        return
        
