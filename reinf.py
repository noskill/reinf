from __future__ import annotations
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gymnasium as gym

plt.rcParams["figure.figsize"] = (10, 5)

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates action probabilities.
        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()
        hidden_space1 = 16
        hidden_space2 = 32
        
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities.
        Args:
            x: Observation from the environment
        Returns:
            action_probs: Probability distribution over actions
        """
        return self.net(x.float())

class REINFORCE:
    """REINFORCE algorithm."""
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm
        to solve the CartPole task.
        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.probs = []
        self.rewards = []
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> int:
        """Returns an action, conditioned on the policy and observation.
        Args:
            state: Observation from the environment
        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_probs = self.net(state)
        
        # Create a categorical distribution and sample an action
        distribution = Categorical(action_probs)
        action = distribution.sample()
        
        # Store log probability of the sampled action
        self.probs.append(distribution.log_prob(action))
        
        return action.item()

    def update(self):
        """Updates the policy network's weights."""
        returns = []
        G = 0
        # Calculate returns for each timestep
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = 0
        # Calculate loss
        for log_prob, G in zip(self.probs, returns):
            loss += -log_prob * G
            
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.probs = []
        self.rewards = []

# Create and wrap the environment
env = gym.make("CartPole-v1")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)

total_num_episodes = int(1000)

# Get observation and action space dimensions
obs_space_dims = env.observation_space.shape[0]  # 4 for CartPole
action_space_dims = env.action_space.n  # 2 for CartPole

rewards_over_seeds = []

for seed in [1, 2, 3, 5, 8]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []
    
    for episode in range(total_num_episodes):
        obs, info = wrapped_env.reset(seed=seed)
        done = False
        
        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            done = terminated or truncated
            
        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()
        
        if episode % 100 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print(f"Episode: {episode}, Average Reward: {avg_reward}")
    
    rewards_over_seeds.append(reward_over_episodes)

env.close()
