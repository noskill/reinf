import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
np = numpy
import random

from agent_reinf import Agent


class Reinforce(Agent):
    def __init__(self, policy, sampler, policy_lr=0.001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None):
        super().__init__(logger=logger)
        self.policy = policy
        self.sampler = sampler
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=policy_lr)#, weight_decay=0.001)
        self.num_envs = num_envs
        self.episodes = [[] for _ in range(num_envs)]  # Separate buffer for each env
        self.discount = discount
        self.completed = []
        self.mean_reward = -10000
        self.mean_std = 0
        self.device = device
        self.version = 0
        self.versions = [x for x in range(self.num_envs)]
        # New attributes for episode pool
        self.pool_size = 50
        self.episode_pool = []

    def episode_start(self):
        self.episodes = [[] for _ in range(self.num_envs)]
        self.active_envs = np.ones(self.num_envs, dtype=bool)
        self.completed = []

    def get_action(self, state, done):
        # Only get actions for active environments
        active_mask = ~done
        if not any(active_mask):
            return np.zeros((self.num_envs,) + self.policy.action_shape)

        # Process only active environments
        active_states = torch.FloatTensor(state[active_mask])
        actions, log_probs, dist = self.sampler(self.policy, active_states)

        # Create full action array
        full_actions = torch.zeros((active_mask.shape[0], actions.shape[1] if len(actions.shape) > 1 else 1)).to(actions)
        full_actions[active_mask] = actions.reshape(full_actions[active_mask].shape)
        # Store transitions only for active environments
        active_env_indices = np.where(active_mask)[0]
        for idx, env_idx in enumerate(active_env_indices):
            self.episodes[env_idx].append((
                active_states[idx],
                actions[idx],
                log_probs[idx]
            ))

        return full_actions.flatten()

    def update(self, obs, actions, rewards, dones, next_obs):
        for env_idx in range(self.num_envs):
            if self.episodes[env_idx] and len(self.episodes[env_idx]) > 0:
                self.episodes[env_idx][-1] += (rewards[env_idx],)
        
        completed_episodes = []
        for env_idx, done in enumerate(dones):
            if done and len(self.episodes[env_idx]) > 0:
                completed_episodes.append(self.episodes[env_idx])
                self.episodes[env_idx] = []
    
        self.completed.extend(completed_episodes)
        if len(self.completed) >= self.num_envs:
            # Add completed episodes to the pool
            for episode in self.completed:
                if len(self.episode_pool) < self.pool_size:
                    self.episode_pool.append(episode)
                else:
                    # Replace random episode in the pool
                    idx = random.randint(0, self.pool_size - 1)
                    self.episode_pool[idx] = episode

            # Select random subset from pool for learning
            learning_subset = random.sample(
                self.episode_pool, 
                min(self.num_envs, len(self.episode_pool))
            )
            
            self.learn_from_episodes(learning_subset)
            self.print_episode_stats(learning_subset)
            self.completed = []
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
