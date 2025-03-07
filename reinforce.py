import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import TransformedDistribution
from sample import *
np = numpy

from agent_reinf import Agent
from pool import *


class RunningNorm:
    def __init__(self, epsilon=1e-4):
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def __call__(self, x):
        if self.mean is None:
            self.mean = x.mean(0, keepdim=True)
            self.std = x.std(0, keepdim=True) + self.epsilon
        else:
            self.mean = 0.99 * self.mean + 0.01 * x.mean(0, keepdim=True)
            self.std = 0.99 * self.std + 0.01 * x.std(0, keepdim=True)
        return (x - self.mean) / self.std


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
        entropy_coef=0.01,
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
        self.create_optimizers()
        self.entropy_coef = entropy_coef
        self.entropy_thresh = 0.2
        super().__init__(logger=logger)
        self.hparams.update( {
            'policy_lr': policy_lr,
            'discount': discount,
            'entropy_coef': self.entropy_coef,
            'entropy_thresh': self.entropy_thresh
        })
        self.state_normalizer = None
        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'rewards']

    def create_optimizers(self):
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

    def get_policy_for_action(self):
        return self.policy

    def episode_start(self):
        self.reset_episodes()
        self.active_envs = np.ones(self.num_envs, dtype=bool)

    def process_states(self, state, done):
        active_mask = ~done
        if not any(active_mask):
            return np.zeros((self.num_envs,) + self.policy.action_shape)

        active_states = state[active_mask]
        if not isinstance(active_states, torch.Tensor):
            active_states = torch.FloatTensor(active_states)
        return active_states

    def get_action(self, state, done):
        policy = self.get_policy_for_action()
        active_states = self.process_states(state, done)
        actions, log_probs, dist = self.sampler(policy, active_states)
        if isinstance(dist, TransformedDistribution):
            entropy = dist.base_dist.entropy()
        else:
            entropy = dist.entropy()
        entropy = entropy.mean(dim=-1, keepdims=True)
        active_mask = ~done
        full_actions = torch.zeros(
            (active_mask.shape[0], actions.shape[1] if len(actions.shape) > 1 else 1)
        ).to(actions)
        full_actions[active_mask] = actions.reshape(full_actions[active_mask].shape)

        active_env_indices = torch.where(active_mask)[0]
        for idx, env_idx in enumerate(active_env_indices):
            self.add_transition(env_idx, active_states[idx], actions[idx], log_probs[idx], entropy[idx])

        return full_actions

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
            episode_return = sum(item[-1] for item in episode)
            episode_returns.append(episode_return)

        # Calculate statistics
        mean_length = np.mean(episode_lengths)
        mean_return = torch.mean(torch.stack(episode_returns))
        std_return = torch.mean(torch.stack(episode_returns))
        self.logger.log_scalar(f"Average episode length",  mean_length)
        self.logger.log_scalar(f"Average return",  mean_return)
        self.logger.log_scalar(f"return std",  std_return)
        self.logger.log_scalar(f"min return",  min(episode_returns))
        self.logger.log_scalar(f"max return",  max(episode_returns))

    def learn_from_episodes(self, episodes):
        # Extract per-episode tensors/lists
        data_dict = self._extract_episode_data(episodes)
        states_list = data_dict['states']
        log_probs_list = data_dict['log_probs']
        actions_list = data_dict['actions']
        rewards_list = data_dict['rewards']
        entropy_list = data_dict['entropy']
        if not states_list:
            return

        # Prepare batches: compute discounted returns, and then aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = self._prepare_batches(
            states_list, log_probs_list, rewards_list, actions_list, entropy_list
        )

        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)
        self.logger.log_scalar("adv normalized max", normalized_returns.max())

        # Log statistics
        self._log_training_stats(actions_batch)
        if len(entropy_batch.shape) == 2 and entropy_batch.shape[1] == 1:
            entropy_batch = entropy_batch.flatten()
        # Finally, train the policy (using log probs computed from current policy)
        self.train_policy(log_probs_batch, normalized_returns, entropy_batch, states_batch, actions_batch)

    def _extract_episode_data2(self, episodes):
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

    def _extract_episode_data(self, episodes):
        """
        From a list of episodes, each consisting of a sequence of tuples with various data elements,
        extract and organize the data by type across all episodes.

        Args:
            episodes: List of episodes, where each episode is a list of data tuples

        Returns:
            Dictionary where keys are data types and values are lists of tensors (one tensor per episode)
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
                else:
                    # Other data are typically already tensors
                    data_lists[data_type].append(torch.stack(episode_data[data_type]))

        # Return the organized data
        return data_lists

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

        n = 10.0
        self.mean_reward = (
            self.mean_reward * (n - 1) / n + (returns_batch.mean() / n)
        )
        self.mean_std = (
            self.mean_std * (n - 1) / n + returns_batch.std() / n
        )

        returns_std = returns_batch.std() + 1e-8  # prevent division by zero
        return (returns_batch - self.mean_reward) / (self.mean_std + 1e-8)

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

    def train_policy(self, log_probs, returns, entropy=torch.zeros(1), states_batch=None, actions=None):
        assert log_probs.shape == returns.shape, "Expected same shape for log_probs and returns!"
        assert log_probs.shape == entropy.shape, "Expected same shape for log_probs and entropy!"
        self.optimizer_policy.zero_grad()

        # Entropy loss
        e_loss = -( entropy).to(log_probs).mean()

        # mu loss for normal distribution (continuous action space)
        mu_loss = torch.tensor(0.0, device=self.device)
        if states_batch is not None and isinstance(self.sampler, NormalActionSampler):
            out = self.policy(states_batch)
            mu = out[..., :1]
            mu_loss = torch.mean(mu**2)
            self.logger.log_scalar("mu loss:", mu_loss.item())

        mu_coef = 0.001
        policy_loss = -(log_probs * returns).mean() + self.entropy_coef * e_loss + mu_loss * mu_coef
        if policy_loss.abs() > 100:
            import pdb;pdb.set_trace()
            return
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.4)
        grads = []
        # Log gradients
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
        self.logger.log_scalar(f"policy grad std :", np.std(grads))
        self.logger.log_scalar(f"entropy mean :", entropy.mean())
        self.logger.log_scalar(f"entropy loss :", e_loss)
        self.optimizer_policy.step()
        self.logger.log_scalar("policy loss:", policy_loss.item())

    def get_state_dict(self):
        return {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict() if hasattr(self, 'value') else None,
            'policy_old': self.policy_old.state_dict() if hasattr(self, 'policy_old') else None,
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict() if hasattr(self, 'optimizer_value') else None,
            'mean_reward': self.mean_reward,
            'mean_std': self.mean_std,
            'state_normalizer': self.state_normalizer.__dict__ if self.state_normalizer else None
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
        if state_dict['value'] and hasattr(self, 'value'):
            self.value.load_state_dict(state_dict['value'])
        if state_dict['policy_old'] and hasattr(self, 'policy_old'):
            self.policy_old.load_state_dict(state_dict['policy_old'])
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        if state_dict['optimizer_value'] and hasattr(self, 'optimizer_value'):
            self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
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
                         device=device, logger=logger)


class ReinforceWithOldEpisodes(ReinforceBase, EpisodesOldPoolMixin):
    def __init__(
        self,
        policy,
        sampler,
        policy_lr=0.001,
        num_envs=8,
        discount=0.99,
        device=torch.device("cpu"),
        logger=None,
        pool_size=50,
    ):
        super().__init__(
            policy,
            sampler,
            policy_lr=policy_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger
        )
        self.pool_size = pool_size

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

        # Recompute log probabilities with current policy
        _, log_probs, dist = self.sampler(self.policy, states_batch)
        if isinstance(dist, TransformedDistribution):
            entropy = dist.base_dist.entropy()
        else:
            entropy = dist.entropy()

        # Reshape log probabilities to match returns shape if necessary
        if log_probs.shape != returns_batch.shape:
            log_probs = log_probs.reshape(returns_batch.shape)

        # Finally, train the policy using current policy's log probs
        self.train_policy(log_probs, normalized_returns, entropy, states_batch)


class ReinforceWithPrediction(ReinforceBase, EpisodesPoolMixin):
    def __init__(
        self,
        policy,
        predictor,
        sampler,
        policy_lr=0.001,
        predictor_lr=0.001,
        num_envs=8,
        discount=0.99,
        device=torch.device("cpu"),
        logger=None,
        prediction_bonus_scale=0.1
    ):
        super().__init__(
            policy,
            sampler,
            policy_lr=policy_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger,
        )
        self.predictor = predictor
        self.prediction_bonus_scale = prediction_bonus_scale
        weight_decay_predictor = 0.001
        self.optimizer_predictor = optim.Adam(
            self.predictor.parameters(),
            lr=predictor_lr,
            weight_decay=weight_decay_predictor
        )
        self.prediction_error_mean = 0
        self.prediction_error_std = 1
        self.hparams.update(
            dict(
                predictor_lr=predictor_lr,
                weight_decay_pred=weight_decay_predictor,
                prediction_bonus_scale=prediction_bonus_scale
            ))

    def _extract_episode_data_with_next_states(self, episodes):
        states_list, log_probs_list, rewards_list, actions_list, entropy_list = [], [], [], [], []
        next_states_list = []

        for episode in episodes:
            if not episode:
                continue
            states, actions, log_probs, entropy, rewards = [], [], [], [], []
            next_states = []

            for i in range(len(episode)-1):
                s, a, log_p, e, r = episode[i]
                next_s, _, _, _, _ = episode[i+1]

                states.append(s)
                actions.append(a)
                log_probs.append(log_p)
                entropy.append(e)
                rewards.append(r)
                next_states.append(next_s)

            # Handle last transition
            if episode:
                s, a, log_p, e, r = episode[-1]
                states.append(s)
                actions.append(a)
                log_probs.append(log_p)
                entropy.append(e)
                rewards.append(r)
                next_states.append(s)  # Use current state as next state for last transition

            states_list.append(torch.stack(states))
            actions_list.append(torch.stack(actions))
            log_probs_list.append(torch.stack(log_probs))
            entropy_list.append(torch.stack(entropy))
            rewards_list.append(torch.tensor(rewards, dtype=torch.float32))
            next_states_list.append(torch.stack(next_states))

        return states_list, log_probs_list, rewards_list, actions_list, entropy_list, next_states_list

    def train_predictor(self, states_batch, actions_batch, next_states_batch):
        predictor_epochs = 2
        mini_batch_size = 256

        for epoch in range(predictor_epochs):
            indices = np.random.permutation(len(states_batch))
            for start in range(0, len(states_batch), mini_batch_size):
                self.optimizer_predictor.zero_grad()
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                pred_next_states = self.predictor(
                    states_batch[batch_idx].detach(),
                    actions_batch[batch_idx].detach()
                )
                prediction_loss = F.mse_loss(pred_next_states, next_states_batch[batch_idx])

                prediction_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 0.4)
                self.optimizer_predictor.step()

        return prediction_loss.item()

    def compute_prediction_bonus(self, states, actions, next_states):
        with torch.no_grad():
            predicted_next_states = self.predictor(states, actions)
            prediction_errors = F.mse_loss(predicted_next_states, next_states, reduction='none').mean(dim=-1)

            # Update running statistics
            n = 100.0  # smoothing factor
            self.prediction_error_mean = (
                self.prediction_error_mean * (n - 1) / n + prediction_errors.mean().item() / n
            )
            self.prediction_error_std = (
                self.prediction_error_std * (n - 1) / n + prediction_errors.std().item() / n
            )

            return prediction_errors * self.prediction_bonus_scale

    def learn_from_episodes(self, episodes):
        # Extract data including next states
        states_list, log_probs_list, rewards_list, actions_list, entropy_list, next_states_list = (
            self._extract_episode_data_with_next_states(episodes)
        )

        if not states_list:
            return

        # Prepare batches
        states_batch = torch.cat(states_list, dim=0).to(self.device)
        actions_batch = torch.cat(actions_list, dim=0).to(self.device)
        next_states_batch = torch.cat(next_states_list, dim=0).to(self.device)
        returns_batch = torch.cat([self._compute_discounted_returns(r) for r in rewards_list], dim=0).to(self.device)
        log_probs_batch = torch.cat(log_probs_list, dim=0).to(self.device)
        entropy_batch = torch.cat(entropy_list, dim=0).to(self.device)

        # Compute prediction bonus
        prediction_bonus = self.compute_prediction_bonus(states_batch, actions_batch, next_states_batch)
        augmented_returns = returns_batch + prediction_bonus

        # Normalize returns
        normalized_returns = self._normalize_returns(augmented_returns)

        # Train predictor
        prediction_loss = self.train_predictor(states_batch, actions_batch, next_states_batch)
        self.logger.log_scalar("prediction_loss", prediction_loss)
        self.logger.log_scalar("prediction_error_mean", self.prediction_error_mean)
        self.logger.log_scalar("prediction_error_std", self.prediction_error_std)

        # Log statistics
        self._log_training_stats(actions_batch)

        # Train policy
        if len(entropy_batch.shape) > 2 and entropy_batch.shape[1] > 1:
            entropy_batch = entropy_batch.flatten()

        # Reshape log probabilities to match returns shape if necessary
        if log_probs_batch.shape != returns_batch.shape:
            log_probs_batch = log_probs_batch.reshape(returns_batch.shape)
        if entropy_batch.shape != returns_batch.shape:
            entropy_batch = entropy_batch.reshape(returns_batch.shape)
        self.train_policy(log_probs_batch, normalized_returns, entropy_batch, states_batch)
