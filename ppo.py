from vpg import VPGBase
import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sample import NormalActionSampler
np = numpy
from pool import EpisodesPoolMixin
from reinforce import RunningNorm
import copy


class PPOBase(VPGBase):
    def __init__(self, policy, value, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, num_learning_epochs=4, clip_param=None, **kwargs):
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, **kwargs)
        self.policy_old = copy.deepcopy(self.policy)
        self.eps = 0.2
        self.hparams.update({'eps': self.eps})
        self.state_normalizer = RunningNorm()
        self.num_learning_epochs = num_learning_epochs

    def learn_from_episodes(self, episodes, num_minibatches=4):
        # Extract per-episode tensors/lists
        data_dict = self._extract_episode_data(episodes)
        states_list = data_dict['states']
        log_probs_list = data_dict['log_probs']
        actions_list = data_dict['actions']
        rewards_list = data_dict['rewards']
        entropy_list = data_dict['entropy']
        if not states_list:
            return

        # Prepare batches: compute discounted returns, aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = \
            self._prepare_batches(states_list, log_probs_list, rewards_list, actions_list, entropy_list)

        # Normalize returns
        normalized_returns = self._normalize_returns(returns_batch)

        # Log statistics
        self._log_training_stats(actions_batch)

        # Train value network before policy update begins
        self.train_value(normalized_returns, states_batch)

        # Sync policies
        self.policy.load_state_dict(self.policy_old.state_dict())

        # Loop through epochs and call separated method
        for i in range(self.num_learning_epochs):
            self._learn_epoch(
                states_batch, log_probs_batch, normalized_returns,
                actions_batch, entropy_batch, num_minibatches
            )

        # Update old policy to match newly optimized policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _learn_epoch(self, states_batch, log_probs_batch, normalized_returns,
                    actions_batch, entropy_batch, num_minibatches):
        dataset = torch.utils.data.TensorDataset(
            states_batch, log_probs_batch, normalized_returns, actions_batch, entropy_batch
        )

        batch_size = len(states_batch) // num_minibatches
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for (states_minibatch, log_probs_minibatch, returns_minibatch,
            actions_minibatch, entropy_minibatch) in data_loader:

            if len(states_minibatch) < max(batch_size // 2, 2):  # safer minimum size check
                continue

            # Policy Update: compute advantages from current value estimates
            with torch.no_grad():
                updated_values = self.value(states_minibatch).squeeze(-1)

            advantages = returns_minibatch - updated_values
            advantage_std = advantages.std()

            if torch.isnan(advantage_std).any():
                import pdb; pdb.set_trace()

            self.logger.log_scalar("Raw advantage mean:", advantages.mean().item())
            self.logger.log_scalar("Raw advantage std:", advantage_std.item())

            # Ensure proper entropy shape
            if len(entropy_minibatch.shape) == 2 and entropy_minibatch.shape[1] == 1:
                entropy_minibatch = entropy_minibatch.flatten()

            # Call policy training step
            self.train_policy(
                log_probs_minibatch, advantages, entropy_minibatch, states_minibatch, actions_minibatch
            )

    def train_policy(self, log_probs, advantages, entropy=torch.zeros(1), states_batch=None, actions_batch=None):
        self.optimizer_policy.zero_grad()

        # First, ensure log_probs, advantages, and entropy have the same shape (1D or 2D).
        # This is already in your code:
        assert log_probs.shape == advantages.shape, f"Expected same shape for log_probs ({log_probs.shape}) and advantages ({advantages.shape})!"
        assert log_probs.shape == entropy.shape, f"Expected same shape for log_probs ({log_probs.shape}) and entropy ({entropy.shape})!"

        # Optionally squeeze them to 1D if they are shaped [B,1].
        # E.g. if you prefer 1D vectors for the ratio calculation, do something like:
        if log_probs.dim() == 2 and log_probs.shape[1] == 1:
            log_probs = log_probs.squeeze(-1)
            advantages = advantages.squeeze(-1)
            entropy = entropy.squeeze(-1)

        # mu loss for normal distribution (continuous action space)
        mu_loss = torch.tensor(0.0, device=self.device)
        if states_batch is not None and isinstance(self.sampler, NormalActionSampler):
            out = self.policy(states_batch)
            mu = out[..., :1]
            # shape check for mu here if you want
            mu_loss = 0.01 * torch.mean(mu**2 * (mu.abs() > 2))
            self.logger.log_scalar("mu loss:", mu_loss.item())

        # Sample from new and old policy
        _, _, dist = self.sampler(self.policy, states_batch)
        # with torch.no_grad():
        #     _, _, dist_old = self.sampler(self.policy_old, states_batch)

        log_probs_new = dist.log_prob(actions_batch).mean(-1, keepdim=True)
        log_probs_old = log_probs.detach()
        # log_probs_old = dist_old.log_prob(actions_batch)

        # If these are shape [B,1], you might likewise squeeze them:
        if log_probs_new.dim() == 2 and log_probs_new.shape[1] == 1:
            log_probs_new = log_probs_new.squeeze(-1)
        if log_probs_old.dim() == 2 and log_probs_old.shape[1] == 1:
            log_probs_old = log_probs_old.squeeze(-1)

        # Now assert that they are the same shape as log_probs (which is advantage-size).
        assert log_probs_new.shape == log_probs.shape, (
            f"log_probs_new ({log_probs_new.shape}) does not match log_probs ({log_probs.shape})."
        )
        assert log_probs_old.shape == log_probs.shape, (
            f"log_probs_old ({log_probs_old.shape}) does not match log_probs ({log_probs.shape})."
        )

        # If dist is a TransformedDistribution, use base_dist for entropy
        if isinstance(dist, TransformedDistribution):
            entropy_dist = dist.base_dist.entropy()
        else:
            entropy_dist = dist.entropy()

        # Possibly squeeze it as well:
        if entropy_dist.dim() == 2 and entropy_dist.shape[-1] == 1:
            entropy_dist = entropy_dist.squeeze(-1)

        # Entropy loss
        m = (entropy_dist < self.entropy_thresh)
        e_loss = -(self.entropy_coef * entropy_dist * m).to(log_probs).mean()

        # they should be the same shape
        assert log_probs_new.shape == log_probs_old.shape
        log_ratio = log_probs_new - log_probs_old
        # Clamp log ratio BEFORE exponentiating
        log_ratio_clamped = log_ratio.clamp(-10, 10)

        # Finally, compute ratio
        ratio = torch.exp(log_ratio_clamped).clamp(-18, 18)

        # Double‐check ratio shape
        assert ratio.shape == log_probs.shape, (
            f"ratio ({ratio.shape}) does not match log_probs ({log_probs.shape}); possible broadcasting!"
        )

        # Clip ratio
        ratio_clip = torch.clip(ratio, 1 - self.eps, 1 + self.eps)

        # Policy loss
        policy_loss_ppo = -torch.min(ratio * advantages, ratio_clip * advantages)

        self.logger.log_scalar("ratio max", ratio.max())
        self.logger.log_scalar("abs(ratio) mean", ratio.abs().mean())

        if torch.isnan(policy_loss_ppo).any():
            import pdb; pdb.set_trace()

        policy_loss = policy_loss_ppo.mean() + e_loss + mu_loss
        if policy_loss > 100:
            import pdb; pdb.set_trace()

        policy_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)

        # Log gradients
        grads = []
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
                if torch.isnan(param.grad).any():
                    import pdb;pdb.set_trace()

        self.logger.log_scalar("policy grad std:", np.std(grads))
        self.logger.log_scalar("entropy mean:", entropy_dist.mean())
        self.logger.log_scalar("entropy loss:", e_loss)

        self.optimizer_policy.step()
        self.logger.log_scalar("policy loss:", policy_loss.item())

    def get_policy_for_action(self):
        return self.policy_old

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.policy_old.load_state_dict(self.policy.state_dict())


class PPO(PPOBase, EpisodesPoolMixin):
    pass


class PPOMI(PPO):
    def __init__(self, policy, value, T, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, **kwargs):
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, **kwargs)
        self.T = T
        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'rewards']

    def learn_from_episodes(self, episodes, num_minibatches=4):
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
        self.train_value(normalized_returns, states_batch)

        self.policy.load_state_dict(self.policy_old.state_dict())
        for i in range(4):
            # Create minibatches
            dataset = torch.utils.data.TensorDataset(states_batch, log_probs_batch, normalized_returns, actions_batch, entropy_batch)
            batch_size = len(states_batch) // num_minibatches
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for states_minibatch, log_probs_minibatch, returns_minibatch, actions_minibatch, entropy_minibatch in data_loader:
                if len(states_minibatch) < batch_size // 2 or len(states_minibatch) < 2:
                    continue
                # Policy Update
                with torch.no_grad():
                    updated_values = self.value(states_minibatch).squeeze(-1)

                advantages = returns_minibatch - updated_values
                std = advantages.std()
                if torch.isnan(std).any():
                    import pdb;pdb.set_trace()
                self.logger.log_scalar("Raw advantage mean:", advantages.mean().item())
                self.logger.log_scalar("Raw advantage std:", std.item())

                # Finally, train the policy (using log probs computed from current policy)
                if len(entropy_minibatch.shape) == 2 and entropy_minibatch.shape[1] == 1:
                    entropy_minibatch = entropy_minibatch.flatten()

                self.train_policy(log_probs_minibatch, advantages, entropy_minibatch, states_minibatch, actions_minibatch)
        self.policy_old.load_state_dict(self.policy.state_dict())

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
