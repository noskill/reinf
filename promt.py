Please fix my entropy calculation: it doesn't work with transformed distribution. use H(X) = H(Y) + E[log |det(dy/dx)|]

vpg_main.py  
  
```  
import numpy
import torch
import sys
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from torch import nn
from vpg import *
from sample import *


class NA1(nn.Module):
    def __init__(self, n_obs, n_action, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_action)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)


def make_env():
    return gym.make('Pendulum-v1')


test_env = make_env()
# Create vectorized environment
num_envs=16
env = SyncVectorEnv([make_env for _ in range(num_envs)])  # 8 environments
num_batch = 10
n_episodes = 10000
dist_params = 2
obs_dim = test_env.observation_space.shape[0]
action_dim = test_env.action_space.shape[0]

discount = 0.99
device = torch.device('cuda')
na = NA1(obs_dim, action_dim * dist_params, hidden_dim=256).to(device)
value = Value(obs_dim).to(device)

sampler = lambda policy, state: sample_action_normal(policy, state, a_min=-2, a_max=2)
agent = VPG(na, value, 
            sampler, 
            num_envs=num_envs, discount=discount, device=device, policy_lr=0.001, value_lr=0.001)
agent.policy.apply(init_normal)
value.apply(init_normal)
for i in range(n_episodes):
    obs, info = env.reset()
    done = numpy.zeros(num_envs, dtype=bool)
    agent.episode_start()
    while not done.all():
        action = agent.get_action(obs, done)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        agent.update(obs, action, reward, done, next_obs)
        obs = next_obs
    if i % 10 == 0:
        print('iteration ' + str(i))
    sys.stdout.flush()


  
```  
  
  
  
sample.py

```
import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from agent import Agent


np = numpy


def sample_action_normal(policy, state, a_min=-2.0, a_max=2.0):
    """
    Samples an action from a Normal distribution parametrized by policy(state),
    then applies tanh-squashing to keep outputs in [-1,1], and finally
    scales/affines them to [a_min,a_max]. Returns (action, log_prob, raw_params).
    """
    # Ensure state is a properly formatted tensor
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    if state.dim() == 1:
        state = state.unsqueeze(0) # batch dimension if necessary

    # Forward pass: e.g. network outputs [mu, log_sigma]
    out = policy(state)
    mu = out[..., :1]
    log_sigma = out[..., 1:]

    # Convert log_sigma to a positive std; softplus is a common choice
    sigma = F.softplus(log_sigma) + 1e-6

    # Base Normal distribution
    base_dist = Normal(mu, sigma)

    # A sequence of transforms: first Tanh to go from (-∞, ∞) to (-1, 1), then
    # an affine transform to go from (-1, 1) to (a_min, a_max)
    transforms = [
        TanhTransform(cache_size=1),
        AffineTransform(loc=(a_min + a_max) / 2.0, scale=(a_max - a_min) / 2.0)
        ]
    transformed_dist = TransformedDistribution(base_dist, transforms)

    # Sample an action (no gradient through sample method; use rsample for reparameterization)
    action = transformed_dist.sample()

    # Optionally clamp to ensure numerical stability
    action = torch.clamp(action, a_min + 1e-6, a_max - 1e-6)

    # Compute log probability. Note .log_prob(action) is shaped [batch_size],
    # so add dimension if you want shape [batch_size, 1].
    log_prob = transformed_dist.log_prob(action).unsqueeze(-1)

    return action, log_prob, transformed_dist


def sample_action_beta(policy, state, a_min=-2, a_max=2):
    # Ensure state is a properly formatted tensor
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    if state.dim() == 1:
        state = state.unsqueeze(0)  # Add batch dimension if necessary
    params = policy(state)
    alpha = params[..., :1]  # First half of outputs
    beta = params[..., 1:]   # Second half of outputs
    # Clip parameters to prevent extreme values
    # torch.abs is an option
    alpha = F.softplus(alpha) + 1e-6
    beta = F.softplus(beta) + 1e-6
    base_dist = Beta(alpha, beta)
    transforms = [AffineTransform(loc=a_min, scale=a_max - a_min)]
    transformed_dist = TransformedDistribution(base_dist, transforms)

    # Sample action without gradient flow through the sampling process
    action = transformed_dist.sample()
    action = torch.clamp(action, -a_min + 1e-6, a_max - 1e-6)
    # Compute log-probability of the sampled action
    # The log_prob() method depends on the distribution parameters (alpha, beta)
    log_prob = transformed_dist.log_prob(action)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob, transformed_dist


```
  
vpg.py  
  
```
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
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=value_lr)
        self.num_envs = num_envs
        self.episodes = [[] for _ in range(num_envs)]  # Separate buffer for each env
        self.discount = discount
        self.completed = []
        self.mean_reward = -10000
        self.mean_std = 0
        self.device = device

    def episode_start(self):
        self.episodes = [[] for _ in range(self.num_envs)]

    def get_action(self, state, done):
        # state is now a batch of states
        state = torch.FloatTensor(state)
        actions, log_probs, params = self.sampler(self.policy, state)
        # Store transitions for each environment
        for env_idx in range(self.num_envs):
            assert not done[env_idx]
            self.episodes[env_idx].append((
                state[env_idx],
                actions[env_idx],
                log_probs[env_idx]
            ))

        return actions.cpu().detach().numpy()

    def update(self, obs, actions, rewards, dones, next_obs):
        # Add rewards to the stored transitions
        for env_idx in range(self.num_envs):
            self.episodes[env_idx][-1] += (rewards[env_idx],)

        # Process completed episodes
        completed_episodes = []
        for env_idx, done in enumerate(dones):
            if done and len(self.episodes[env_idx]) > 0:
                completed_episodes.append(self.episodes[env_idx])
                self.episodes[env_idx] = []
        self.completed.extend(completed_episodes)
        if len(self.completed) >= self.num_envs:
            self.learn_from_episodes(self.completed)
            self.print_episode_stats(self.completed)
            self.completed = []

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

    def learn_from_episodes(self, completed_episodes):
        # Prepare lists to store all data from all completed episodes
        all_states = []
        all_returns = []
        all_log_probs = []

        # Step 1: Compute returns per episode, store them
        for episode in completed_episodes:
            states = []
            log_probs = []
            rewards = []

            # Unpack transitions: (state, action, log_prob, reward)
            for (s, _, log_p, r) in episode:
                states.append(s)
                log_probs.append(log_p)
                rewards.append(r)

            states = torch.stack(states, dim=0)
            log_probs = torch.stack(log_probs, dim=0)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            # Compute discounted returns Gt (backward)
            returns = []
            G = 0
            for r_t in reversed(rewards):
                G = r_t + self.discount * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            # Collect into bigger lists
            all_states.append(states)
            all_returns.append(returns)
            all_log_probs.append(log_probs)

        # Concatenate across all episodes for easier batch training
        states_batch = torch.cat(all_states, dim=0).to(self.device) # shape: [sum_of_T, state_dim]
        
        _, _, transformed_dist = self.sampler(self.policy, states_batch)
        all_returns_cat = torch.cat(all_returns, dim=0).to(self.device) # shape: [sum_of_T]
        all_returns_cat = all_returns_cat / 100
        all_log_probs_cat = torch.cat(all_log_probs, dim=0).to(self.device)
        

        # ----------------------------
        # (Optional) Normalize returns
        # ----------------------------
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

        # ------------------------------------------
        # 1) Value Network Update (fit to returns)
        # ------------------------------------------
        for s in range(10):
            self.optimizer_value.zero_grad()
            pred_values = self.value(states_batch).squeeze(-1)
            value_loss = F.mse_loss(pred_values, all_returns_cat)
            value_loss.backward()
            self.optimizer_value.step()


        # ------------------------------------------------
        # 2) Policy Update (now that value is re-trained)
        # ------------------------------------------------
        # Re-run the (updated) value network to compute advantage

        with torch.no_grad():
            updated_values = self.value(states_batch).squeeze(-1)

        advantages = all_returns_cat - updated_values
        
        # Get the base distribution (Normal distribution in your case)
        base_dist = transformed_dist.base_dist

        # Calculate base entropy
        base_entropy = base_dist.entropy()

        # Calculate the log det jacobian for the transforms
        log_det_jacobian = sum(transform.log_abs_det_jacobian(
            transform.prev_transform.forward(base_dist.rsample()) if idx > 0 
            else base_dist.rsample(),
            transform.forward(transform.prev_transform.forward(base_dist.rsample()) if idx > 0 
            else base_dist.rsample())
        ).mean() for idx, transform in enumerate(transformed_dist.transforms))

        # Total entropy is base entropy minus log det jacobian
        entropy = (base_entropy - log_det_jacobian).mean()

        # Final policy loss
        self.optimizer_policy.zero_grad()
        policy_loss = - (all_log_probs_cat * advantages).mean() - entropy * 0.0001
        policy_loss.backward()

        # Optional gradient clipping for safety
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)

        self.optimizer_policy.step()

        # -----------------
        # Diagnostic print
        # -----------------
        print("policy loss:", policy_loss.item())
        print("value loss:", value_loss.item())




```
