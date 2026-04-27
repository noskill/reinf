import numpy
import torch
from torch.distributions import *
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import TransformedDistribution
from util import RunningNorm, EpisodeBatch, to_device, normalize_padded_returns, flatten_padded
from sample import *
np = numpy

from agent_reinf import Agent
from pool import EpisodesPoolMixin


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
        entropy_coef=0.001,
        sequence_model=False,
        **kwargs
    ):
        self.num_envs = num_envs
        self.policy = policy
        self.sampler = sampler
        self.policy_lr = policy_lr
        self.discount = discount
        self.mean_reward = -10000
        self.mean_std = 0
        self.device = device
        self.version = 0
        self.create_optimizers(**kwargs)
        self.entropy_coef = entropy_coef
        self.entropy_thresh = 0.2
        super().__init__(logger=logger, **kwargs)
        self.target_entropy = torch.tensor(2.0, dtype=torch.float16)
        self.hparams.update( {
            'policy_lr': policy_lr,
            'discount': discount,
            'entropy_coef': self.entropy_coef,
            'entropy_thresh': self.target_entropy
        })
        self.state_normalizer = None
        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'rewards']
        self.is_sequence_model = sequence_model

    def create_optimizers(self, **kwargs):
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

    def get_policy_for_action(self):
        return self.policy

    def episode_start(self):
        self.reset_episodes()
        self.active_envs = np.ones(self.num_envs, dtype=bool)

    def process_states(self, state, episode_start):
        return state

    def get_action(self, state, episode_start):
        policy = self.get_policy_for_action()
        active_states = self.process_states(state, episode_start)
        policy_kwargs = dict(episode_start=episode_start)
        actions, log_probs, dist = self.sampler(policy, active_states, policy_kwargs)

        # ------------------------------------------------------------------
        # Obtain entropy safely. Some `TransformedDistribution` instances do
        # not implement an analytic entropy; fall back to the base
        # distribution in that case.
        # ------------------------------------------------------------------
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            if hasattr(dist, 'base_dist'):
                entropy = dist.base_dist.entropy()
            else:
                raise

        # Validate that each action produces *exactly* one log-prob and one
        # entropy scalar.
        assert log_probs.dim() == 1 or (log_probs.dim() == 2 and log_probs.shape[1] == 1), \
            f"log_probs shape {log_probs.shape} invalid; expected (B,) or (B,1)."
        assert entropy.dim() == 1 or (entropy.dim() == 2 and entropy.shape[1] == 1), \
            f"entropy shape {entropy.shape} invalid; expected (B,) or (B,1)."

        # Ensure entropy is a column vector (B,1) for consistent downstream
        # handling, **without** accidentally reducing across the batch.
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(-1)

        self.add_transition_batch(active_states, actions, log_probs, entropy)
        return actions

    def update(self, rewards, dones, info=None, **kwargs):
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
            episode_return = sum(item[-1] for item in episode)
            episode_returns.append(episode_return)

        # Calculate statistics
        mean_length = np.mean(episode_lengths)
        mean_return = torch.mean(torch.stack(episode_returns))
        std_return = torch.std(torch.stack(episode_returns))
        self.logger.log_scalar(f"Average episode length",  mean_length)
        self.logger.log_scalar(f"Average return",  mean_return)
        self.logger.log_scalar(f"return std",  std_return)
        self.logger.log_scalar(f"min return",  min(episode_returns))
        self.logger.log_scalar(f"max return",  max(episode_returns))

    def _flatten_padded_states(self, states_padded, key_padding_mask):
        if isinstance(states_padded, dict):
            return {k: flatten_padded(v, key_padding_mask) for k, v in states_padded.items()}
        return flatten_padded(states_padded, key_padding_mask)

    def train_sequence_policy(self, episode_batch: EpisodeBatch):
        # Pad only fields needed for policy re-evaluation.
        padded, key_padding_mask, _ = episode_batch.pad(fields=['states', 'actions'])
        states_padded = to_device(padded['states'], self.device)
        actions_padded = padded['actions'].to(self.device)
        key_padding_mask = key_padding_mask.to(self.device)

        # Compute advantages/effective rewards from episode batch.
        effective_reward = self.compute_advantage_ebatch(episode_batch)
        if effective_reward.dim() == 2:
            returns_flat = flatten_padded(effective_reward.unsqueeze(-1), key_padding_mask).squeeze(-1)
        else:
            returns_flat = effective_reward
        returns_flat = returns_flat.to(self.device)
        self.logger.log_scalar("adv normalized max", returns_flat.max())

        actions_flat = flatten_padded(actions_padded, key_padding_mask)
        self._log_training_stats(actions_flat)

        # Policy forward on padded trajectories and evaluate recorded actions.
        obs_for_policy = states_padded
        if isinstance(obs_for_policy, dict):
            obs_for_policy = dict(obs_for_policy)
            obs_for_policy['key_padding_mask'] = key_padding_mask

        _, _, dist = self.sampler(
            self.policy,
            obs_for_policy,
            policy_kwargs=dict(episode_start=None),
            actions=actions_padded,
            return_distribution=True,
        )

        log_probs_seq = dist.log_prob(actions_padded)
        try:
            entropy_seq = dist.entropy()
        except NotImplementedError:
            entropy_seq = dist.base_dist.entropy()

        log_probs_flat = flatten_padded(log_probs_seq.unsqueeze(-1), key_padding_mask).squeeze(-1)
        entropy_flat = flatten_padded(entropy_seq.unsqueeze(-1), key_padding_mask).squeeze(-1)
        states_flat = self._flatten_padded_states(states_padded, key_padding_mask)

        self.train_policy(
            log_probs_flat,
            returns_flat,
            entropy_flat,
            states_batch=states_flat,
            actions_batch=actions_flat,
        )

    def learn_from_episodes(self, episodes):
        """
        Accept either a raw list of episodes (legacy) or an EpisodeBatch.
        For transformer policies, keep episode boundaries via padding + mask and
        compute per-step log-probs/entropy from current policy before flattening
        valid steps for the REINFORCE loss.
        """
        # Convert input into EpisodeBatch with returns
        episode_batch = self._prepare_episode_batch(episodes)
        if episode_batch is None:
            return
        self.train_sequence_policy(episode_batch)

    def _prepare_episode_batch(self, episodes):
        if isinstance(episodes, EpisodeBatch):
            episode_batch = episodes
        else:
            episode_batch = self._extract_episode_data(episodes)
        if episode_batch.num_episodes == 0:
            return
        episode_batch = episode_batch.to(self.device)
        episode_batch.compute_returns(self.discount)
        return episode_batch

    def _normalize_returns(self, returns_batch):
        return self._normalize_returns_running(returns_batch)

    def compute_advantage_ebatch(self, episode_batch):
        padded, key_padding_mask, _ = episode_batch.pad(fields=['returns'])
        key_padding_mask = key_padding_mask.to(self.device)
        returns_padded = padded['returns'].to(self.device)
        normalized_returns, _, _ = normalize_padded_returns(returns_padded, key_padding_mask)
        return normalized_returns

    def _extract_episode_data(self, episodes) -> EpisodeBatch:
        """
        From a list of episodes, each consisting of a sequence of tuples with various data elements,
        extract and organize the data by type across all episodes.
        
        converts list of episodes to a dictionary where keys are data types and values are lists of tensors (one tensor per episode)

        Args:
            episodes: List of episodes, where each episode is a list of data tuples
            
        Returns:
            EpisodeBatch
        """
        # Define the structure of our data tuple and corresponding empty lists
        data_types = self.data_types
        data_lists = {data_type: [] for data_type in data_types}

        # Process each episode
        for episode in episodes:
            if not episode:
                continue

            # Initialize temporary lists for this episode
            episode_data = {data_type: [] for data_type in data_types}

            # Extract data from each step in the episode
            for step_data in episode:
                # Unpack the step data into the corresponding lists
                for i, data_type in enumerate(data_types):
                    episode_data[data_type].append(step_data[i])

            # Convert lists to tensors and add to the main data lists
            for data_type in data_types:
                if data_type == 'rewards':
                    # Rewards are typically scalar values
                    data_lists[data_type].append(torch.tensor(episode_data[data_type], dtype=torch.float32))
                elif data_type == 'states':
                    # Support dict observations by stacking per key
                    first_state = episode_data[data_type][0]
                    if isinstance(first_state, dict):
                        # Validate consistent keys across the episode
                        keys = set(first_state.keys())
                        for s in episode_data[data_type]:
                            assert isinstance(s, dict) and set(s.keys()) == keys, \
                                "All state dicts in an episode must have identical keys"
                        stacked = {k: torch.stack([s[k] for s in episode_data[data_type]], dim=0) for k in keys}
                        data_lists[data_type].append(stacked)
                    else:
                        data_lists[data_type].append(torch.stack(episode_data[data_type], dim=0))
                else:
                    # Other data are tensors; stack along time
                    data_lists[data_type].append(torch.stack(episode_data[data_type], dim=0))

        # Return the organized data
        batch = EpisodeBatch(data_lists)
        return batch

    def _normalize_returns_running(self, returns_batch):
        """
        Normalize returns using a running mean and std.
        """
        # Update running mean if it's the first update
        if self.mean_reward == -10000:
            self.mean_reward = returns_batch.mean()

        n = 10.0
        self.mean_reward = (
            self.mean_reward * (n - 1) / n + (returns_batch.mean() / n)
        )
        self.mean_std = (
            self.mean_std * (n - 1) / n + returns_batch.std() / n
        )

        returns_std = returns_batch.std() + 1e-7  # prevent division by zero
        return (returns_batch - returns_batch.mean()) / returns_std

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

    def train_policy(self, log_probs, returns, entropy=torch.zeros(1), states_batch=None, **kwargs):
        self.optimizer_policy.zero_grad()
        policy_loss = self.compute_loss(log_probs, returns=returns, entropy=entropy, states_batch=states_batch, **kwargs)
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        grads = []
        # Log gradients
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
        self.logger.log_scalar(f"policy grad std :", np.std(grads))
        self.optimizer_policy.step()

    def compute_entropy_loss(self, entropy, device, dtype):
        self.logger.log_scalar("entropy mean:", entropy.mean())
        e_loss = F.relu(self.target_entropy.to(device, dtype) - entropy.to(device, dtype)).mean()
        return e_loss

    def compute_loss(self, log_probs, returns, entropy=torch.zeros(1), states_batch=None, **kwargs):
        assert log_probs.shape == returns.shape, "Expected same shape for log_probs and returns!"
        assert log_probs.shape == entropy.shape, "Expected same shape for log_probs and entropy!"
        # Entropy loss to increase entropy
        e_loss = self.compute_entropy_loss(entropy, log_probs.device, log_probs.dtype)

        # mu loss for normal distribution (continuous action space) - keep values not too extreme
        mu_loss = torch.tensor(0.0, device=self.device)
        if states_batch is not None and isinstance(self.sampler, NormalActionSampler):
            out = self.policy(states_batch, episode_start=None)
            mu, _ = self.sampler.split_out(out)
            #mu = torch.clamp(mu, -1e6, 1e6)
            mu_loss = torch.mean(mu**2)
            self.logger.log_scalar("mu loss:", mu_loss.item())

        mu_coef = 0.001
        log_clamped = log_probs.clamp(-10, 10)
        policy_loss = -(log_clamped * returns).mean() + self.entropy_coef * e_loss + mu_loss * mu_coef
        self.logger.log_scalar("policy loss:", policy_loss.item())
        self.logger.log_scalar(f"entropy mean:", entropy.mean())
        self.logger.log_scalar(f"entropy loss:", e_loss)
        if not torch.isfinite(policy_loss):
            raise ValueError(f"Non-finite loss: {policy_loss.item()}")
        if policy_loss.abs() > 100:
            self.logger.log_scalar("policy loss overflow", policy_loss.item())
            self.logger.warn("policy loss overflow " + str(policy_loss.item()))
        return policy_loss

    def get_state_dict(self):
        return {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict() if hasattr(self, 'value') else None,
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict() if hasattr(self, 'optimizer_value') else None,
            'mean_reward': self.mean_reward,
            'mean_std': self.mean_std,
            'state_normalizer': self.state_normalizer.__dict__ if self.state_normalizer else None
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.mean_reward = state_dict['mean_reward']
        self.mean_std = state_dict['mean_std']
        if state_dict['state_normalizer'] and self.state_normalizer:
            self.state_normalizer.__dict__.update(state_dict['state_normalizer'])


class Reinforce(ReinforceBase, EpisodesPoolMixin):

    def __init__(
        self,
        policy,
        sampler,
        policy_lr=0.001,
        num_envs=8,
        discount=0.99,
        device=torch.device("cpu"),
        logger=None, **kwargs
    ):
        super().__init__(policy, sampler, policy_lr=policy_lr,
                         num_envs=num_envs, discount=discount,
                         device=device, logger=logger, **kwargs)
