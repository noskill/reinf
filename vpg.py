import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
np = numpy

from agent_reinf import Agent


class Value(nn.Module):
    def __init__ (self, state_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        value = self.value_head(x)
        return value


class VPG(Agent):
    def __init__(self, policy, value, sampler, policy_lr=0.001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu')):
        self.policy = policy
        self.value = value
        self.sampler = sampler
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=policy_lr)#, weight_decay=0.001)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=value_lr, weight_decay=0.001)
        self.num_envs = num_envs
        self.episodes = [[] for _ in range(num_envs)]  # Separate buffer for each env
        self.discount = discount
        self.completed = []
        self.mean_reward = -10000
        self.mean_std = 0
        self.device = device
        self.version = 0
        self.versions = [x for x in range(self.num_envs)]

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
        actions, log_probs, params = self.sampler(self.policy, active_states)

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
        
        # Process completed episodes
        completed_episodes = []
        for env_idx, done in enumerate(dones):
            if done and len(self.episodes[env_idx]) > 0:
                completed_episodes.append(self.episodes[env_idx])
                self.episodes[env_idx] = []
                self.versions[env_idx] = self.version
    
        self.completed.extend(completed_episodes)
        if len(self.completed) >= self.num_envs:
            self.learn_from_episodes(self.completed)
            self.print_episode_stats(self.completed)
            self.completed = []
            self.version += 1
            for env_idx in range(self.num_envs):
                if self.versions[env_idx] != self.version:
                    self.episodes[env_idx] = []
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

        print(f"Episodes completed: {len(completed_episodes)}")
        print(f"Average episode length: {mean_length:.1f}")
        print(f"Average return: {mean_return:.2f} ± {std_return:.2f}")

        # Optionally, you could also track min/max returns
        print(f"Min/Max returns: {min(episode_returns):.2f}/{max(episode_returns):.2f}")

    def train_value(self, returns, states):
        # Value Network Update
        value_epochs = 5
        mini_batch_size = 512
        
        for epoch in range(value_epochs):
            indices = np.random.permutation(len(states_batch))
            for start in range(0, len(states_batch), mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                pred_values = self.value(states_batch[batch_idx].detach()).squeeze(-1)
                value_loss = F.mse_loss(pred_values, all_returns_cat[batch_idx])
                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
                self.optimizer_value.step()

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
            log_probs = torch.stack(log_probs)
            rewards = torch.tensor(rewards, dtype=torch.float32)
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


        # Get distribution parameters (only needed for printing stats)
        with torch.no_grad():
            _, _, transformed_dist = self.sampler(self.policy, states_batch)
        
        print(f"actions mean {all_actions_cat.to(torch.float32).mean()}, actions std \
              {all_actions_cat.to(torch.float32).std()}")
        
        # Normalize returns
        if self.mean_reward == -10000:
            self.mean_reward = all_returns_cat.mean()
        
        n = 50
        self.mean_reward = (
            self.mean_reward * (n - 1) / n + (all_returns_cat.mean() / n)
        )
        self.mean_std = (
            self.mean_std * (n - 1) / n + all_returns_cat.std() / n
        )
        returns_std = all_returns_cat.std() + 1e-8
        returns = (all_returns_cat - all_returns_cat.mean()) / (all_returns_cat.std() + 1e-8)
        

        all_returns_cat = (all_returns_cat - self.mean_reward) / returns_std



        self.train_policy(all_log_probs_cat, all_returns_cat)
        return
        # Print value gradients
        # for name, param in self.value.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.norm().item())
        
        # Policy Update
        with torch.no_grad():
            updated_values = self.value(states_batch).squeeze(-1)
        
        advantages = all_returns_cat - updated_values
        print("Raw advantage mean:", advantages.mean().item(), 
            "Raw advantage std:", advantages.std().item())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        self.optimizer_policy.zero_grad()
        policy_loss = -(log_probs * returns).mean()

        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        print("policy gradients")
        # Print policy gradients
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                print(name, param.grad.norm().item())
        self.optimizer_policy.step()
        
        # Print losses
        print("policy loss:", policy_loss.item())
        print("value loss:", value_loss.item())

    def learn_from_episodes1(self, completed_episodes):
        for ep in completed_episodes:
            self.train(ep)
            return

    def train(self, episode):
        """Updates the policy network's weights."""
        
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
                
        returns = []
        G = 0
        # Calculate returns for each timestep
        for r in reversed(rewards):
            G = r + self.discount * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        print("returns " + str(returns.sum()))
        self.train_policy(returns, log_probs)
        
    def train_policy(self, returns, log_probs):
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = 0
        # Calculate loss
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
            
        # Update network weights
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()
