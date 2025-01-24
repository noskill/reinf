import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        AffineTransform(loc=(a_min + a_max) / 2.0,
        scale=(a_max - a_min) / 2.0)
        ]
    transformed_dist = TransformedDistribution(base_dist, transforms)

    # Sample an action (no gradient through sample method; use rsample for reparameterization)
    action = transformed_dist.sample()

    # Optionally clamp to ensure numerical stability
    action = torch.clamp(action, a_min + 1e-6, a_max - 1e-6)

    # Compute log probability. Note .log_prob(action) is shaped [batch_size],
    # so add dimension if you want shape [batch_size, 1].
    log_prob = transformed_dist.log_prob(action).unsqueeze(-1)

    return action, log_prob, out


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
    alpha = F.softplus(alpha) + 1e-6
    beta = F.softplus(beta) + 1e-6
    base_dist = Beta(alpha, beta)
    transforms = [AffineTransform(loc=a_min, scale=a_max - a_min)]
    transformed_dist = TransformedDistribution(base_dist, transforms)

    # Sample action without gradient flow through the sampling process
    action = transformed_dist.sample()
    action = torch.clamp(action, -2 + 1e-6, 2 - 1e-6)
    # Compute log-probability of the sampled action
    # The log_prob() method depends on the distribution parameters (alpha, beta)
    log_prob = transformed_dist.log_prob(action)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob, params


class Agent():
    pass


class Reinforce(Agent):
    def __init__(self, policy, sampler, learning_rate=0.0002, num_envs=8, discount=0.99):
        self.policy = policy
        self.sampler = sampler
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)#, weight_decay=0.001)
        self.num_envs = num_envs
        self.episodes = [[] for _ in range(num_envs)]  # Separate buffer for each env
        self.discount = discount
        self.completed = []

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
        if len(self.completed) >= 32:
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
        policy_losses = []

        for episode in completed_episodes:
            episode_losses = []
            # Calculate returns for this episode
            returns = []
            R = 0
            for _, _, _, reward in reversed(episode):
                R = reward + self.discount * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Use stored log probabilities
            for (_, _, log_prob, _), R in zip(episode, returns):
                episode_losses.append(-log_prob * R)
            policy_losses.append(torch.cat(episode_losses).mean())

        if policy_losses:
            self.optimizer.zero_grad()
            losses = torch.stack(policy_losses)
            if torch.isnan(losses).any():
                import pdb;pdb.set_trace()
            total_loss = losses.mean()
            total_loss.backward()  # Single backward pass
            for i, p in enumerate(self.policy.parameters()):
                if torch.isnan(p).any():
                    import pdb;pdb.set_trace()
                if torch.isnan(p.grad).any():
                    import pdb;pdb.set_trace()

            torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=1.0)
            params = [x for x in self.policy.parameters()]
            self.optimizer.step()
            # print(max([x.max() for x in params]))
            for p in params:
                if torch.isnan(p).any():
                    import pdb;pdb.set_trace()
            print("total loss " + str(total_loss))
