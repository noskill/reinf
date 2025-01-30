import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from agent import Agent


np = numpy


class VPG(Agent):
    def __init__(self, policy, value, sampler, learning_rate=0.001, num_envs=8, discount=0.99):
        self.policy = policy
        self.value = value
        self.sampler = sampler
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)#, weight_decay=0.001)
        self.num_envs = num_envs
        self.episodes = [[] for _ in range(num_envs)]  # Separate buffer for each env
        self.discount = discount
        self.completed = []
        self.mean_reward = 0

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
        policy_losses = []

        returns_batch = []
        for episode in completed_episodes:
            episode_losses = []
            # Calculate returns for this episode
            returns = []
            R = 0
            for _, _, _, reward in reversed(episode):
                R = reward + self.discount * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns_batch.append(returns)

        returns_batch = torch.stack(returns_batch)
        n = 20
        self.mean_reward = self.mean_reward * (n - 1)/n + returns_batch.mean() /n
        returns_batch = (returns_batch - self.mean_reward) / (returns_batch.std() + 1e-8)
        
        for i, episode in enumerate(completed_episodes):
            # Use stored log probabilities
            for (_, _, log_prob, _), R in zip(episode, returns_batch[i]):
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
