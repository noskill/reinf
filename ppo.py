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
from util import RunningNorm, EpisodeBatch, normalize_padded_returns, flatten_padded, to_device
import copy


class PPOBase(VPGBase):
    """PPO-Clip baseline with sequence support.

    - Supports transformer policies and dict observations using EpisodeBatch
      padding + key_padding_mask; keeps episode boundaries for log-prob eval.
    - Follows valid PPO semantics: fixes old_log_probs and advantages for the
      whole update; recomputes new log_probs/entropy each minibatch/epoch.
    - Uses a single train_policy interface that takes precomputed
      log_probs_new (and optional mu for Normal policies) to avoid re-forward
      duplication and ease maintenance.
    """
    def __init__(self, policy, value, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, num_learning_epochs=4, clip_param=None, sequence_pad_fields=None, **kwargs):
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, **kwargs)
        self.policy_old = copy.deepcopy(self.policy)
        self.eps = 0.2
        self.hparams.update({'eps': self.eps})
        self.state_normalizer = RunningNorm()
        self.num_learning_epochs = num_learning_epochs
        self.sequence_pad_fields = sequence_pad_fields or ['states', 'actions', 'returns', 'log_probs']

    def learn_from_episodes(self, episodes, num_minibatches=4):
        """
        Split orchestration similar to ReinforceBase: decide sequence vs flat
        and delegate to dedicated methods.
        """
        if isinstance(episodes, EpisodeBatch):
            batch = episodes
        else:
            data = self._extract_episode_data(episodes)
            batch = data if isinstance(data, EpisodeBatch) else EpisodeBatch(data)

        if batch.num_episodes == 0:
            return
        batch = batch.to(self.device)
        batch.compute_returns(self.discount)

        is_sequence_model = hasattr(self.policy, 'temporal_decoder')
        if is_sequence_model:
            self.train_sequence_policy(batch, num_minibatches)
        else:
            self.train_flat_policy(batch, num_minibatches)

    def _sequence_pad_batch(self, episode_batch: EpisodeBatch):
        padded, key_padding_mask, _ = episode_batch.pad(fields=self.sequence_pad_fields)
        return padded, key_padding_mask

    def _sequence_flatten_states(self, states_padded, key_padding_mask):
        if isinstance(states_padded, dict):
            return {k: flatten_padded(v, key_padding_mask) for k, v in states_padded.items()}
        return flatten_padded(states_padded, key_padding_mask)

    def _sequence_build_policy_obs(self, states_padded, key_padding_mask, padded):
        if isinstance(states_padded, dict):
            obs_for_policy = dict(states_padded)
            obs_for_policy['key_padding_mask'] = key_padding_mask
            return obs_for_policy
        return states_padded

    def _sequence_compute_advantages(self, states_flat, returns_flat):
        with torch.no_grad():
            values_flat = self.value(states_flat).squeeze(-1)
        advantages_flat = returns_flat - values_flat
        adv_mean, adv_std = advantages_flat.mean(), advantages_flat.std() + 1e-8
        advantages_flat = (advantages_flat - adv_mean) / adv_std
        return advantages_flat

    def _sequence_update_value(self, states_flat, returns_flat, num_minibatches):
        value_epochs = 2
        N = returns_flat.shape[0]
        if N == 0:
            return None
        mini = max(1, N // max(1, num_minibatches))
        for _ in range(value_epochs):
            perm = torch.randperm(N, device=self.device)
            for mb in torch.split(perm, mini):
                self.optimizer_value.zero_grad()
                if isinstance(states_flat, dict):
                    v_pred = self.value({k: v[mb] for k, v in states_flat.items()}).squeeze(-1)
                else:
                    v_pred = self.value(states_flat[mb]).squeeze(-1)
                v_loss = F.mse_loss(v_pred, returns_flat[mb])
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
                self.optimizer_value.step()
        return v_loss

    def _sequence_update_policy(self, obs_for_policy, actions_padded, key_padding_mask,
                                old_logp_flat, advantages_flat, num_minibatches):
        N = old_logp_flat.shape[0]
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(N, device=self.device)
            for mb in torch.split(perm, max(1, N // max(1, num_minibatches))):
                _, logp_new_padded, dist = self.sampler(
                    self.policy,
                    obs_for_policy,
                    policy_kwargs=dict(episode_start=None),
                    actions=actions_padded,
                    return_distribution=True,
                )
                try:
                    entropy_padded = dist.entropy()
                except NotImplementedError:
                    entropy_padded = dist.base_dist.entropy()

                logp_new_flat = flatten_padded(logp_new_padded.unsqueeze(-1), key_padding_mask).squeeze(-1)
                entropy_flat = flatten_padded(entropy_padded.unsqueeze(-1), key_padding_mask).squeeze(-1)

                mu_mb = None
                if isinstance(self.sampler, NormalActionSampler):
                    base = dist.base_dist if isinstance(dist, TransformedDistribution) else dist
                    base_normal = base.base_dist if isinstance(base, Independent) else base
                    if hasattr(base_normal, 'loc'):
                        mu_padded = base_normal.loc  # [B,T,A]
                        mu_flat = flatten_padded(mu_padded, key_padding_mask)
                        mu_mb = mu_flat[mb]

                self.train_policy(
                    log_probs_old=old_logp_flat[mb],
                    advantages=advantages_flat[mb],
                    entropy=entropy_flat[mb],
                    log_probs_new=logp_new_flat[mb],
                    mu=mu_mb,
                )

    def train_sequence_policy(self, episode_batch: EpisodeBatch, num_minibatches: int = 4):
        padded, key_padding_mask = self._sequence_pad_batch(episode_batch)
        states_padded = padded['states']
        actions_padded = padded['actions'].to(self.device)
        returns_padded = padded['returns'].to(self.device)
        old_logp_padded = padded['log_probs'].to(self.device)

        normalized_returns = normalize_padded_returns(returns_padded, key_padding_mask)
        states_flat = self._sequence_flatten_states(states_padded, key_padding_mask)
        returns_flat = flatten_padded(normalized_returns.unsqueeze(-1), key_padding_mask).squeeze(-1)
        old_logp_flat = flatten_padded(old_logp_padded.unsqueeze(-1), key_padding_mask).squeeze(-1)

        advantages_flat = self._sequence_compute_advantages(states_flat, returns_flat)
        self._log_training_stats(flatten_padded(actions_padded, key_padding_mask))

        obs_for_policy = self._sequence_build_policy_obs(states_padded, key_padding_mask, padded)

        v_loss = self._sequence_update_value(states_flat, returns_flat, num_minibatches)
        if v_loss is None:
            return
        self.logger.log_scalar("value loss", v_loss)

        self.policy.load_state_dict(self.policy_old.state_dict())
        self._sequence_update_policy(
            obs_for_policy,
            actions_padded,
            key_padding_mask,
            old_logp_flat,
            advantages_flat,
            num_minibatches,
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train_flat_policy(self, episode_batch: EpisodeBatch, num_minibatches: int = 4):
        flat = episode_batch.flatten(fields=['states', 'actions', 'returns', 'log_probs', 'entropy'])
        states_batch = to_device(flat['states'], self.device)
        actions_batch = flat['actions'].to(self.device)
        returns_batch = flat['returns'].to(self.device)
        old_logp_batch = flat['log_probs'].to(self.device)

        normalized_returns = self._normalize_returns(returns_batch)
        self._log_training_stats(actions_batch)

        if isinstance(states_batch, dict):
            value_epochs = 2
            N = returns_batch.shape[0]
            mini = max(1, N // max(1, num_minibatches))
            for _ in range(value_epochs):
                perm = torch.randperm(N, device=self.device)
                for mb in torch.split(perm, mini):
                    self.optimizer_value.zero_grad()
                    v_pred = self.value({k: v[mb] for k, v in states_batch.items()}).squeeze(-1)
                    v_loss = F.mse_loss(v_pred, normalized_returns[mb])
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
                    self.optimizer_value.step()
            self.logger.log_scalar("value loss", v_loss)
        else:
            self.train_value(normalized_returns, states_batch)

        # Compute fixed advantages once (freeze value for policy updates)
        with torch.no_grad():
            v_all = self.value(states_batch).squeeze(-1) if not isinstance(states_batch, dict) else self.value(states_batch).squeeze(-1)
        adv_all = normalized_returns - v_all
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        # Initialize current policy from old before PPO updates
        self.policy.load_state_dict(self.policy_old.state_dict())

        N = old_logp_batch.shape[0]
        batch_size = max(1, N // max(1, num_minibatches))
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(N, device=self.device)
            for mb_idx in torch.split(perm, batch_size):
                if isinstance(states_batch, dict):
                    states_mb = {k: v[mb_idx] for k, v in states_batch.items()}
                else:
                    states_mb = states_batch[mb_idx]
                actions_mb = actions_batch[mb_idx]
                old_logp_mb = old_logp_batch[mb_idx]
                adv_mb = adv_all[mb_idx]

                _, logp_new_mb, dist = self.sampler(self.policy, states_mb, actions=actions_mb, return_distribution=True)
                try:
                    entropy_new_mb = dist.entropy()
                except NotImplementedError:
                    # Handle TransformedDistribution or base distributions
                    entropy_new_mb = dist.base_dist.entropy() if hasattr(dist, 'base_dist') else entropy_new_mb

                mu_mb = None
                if isinstance(self.sampler, NormalActionSampler):
                    base = dist.base_dist if isinstance(dist, TransformedDistribution) else dist
                    base_normal = base.base_dist if isinstance(base, Independent) else base
                    if hasattr(base_normal, 'loc'):
                        mu_mb = base_normal.loc  # [B,A]

                self.train_policy(
                    log_probs_old=old_logp_mb,
                    advantages=adv_mb,
                    entropy=entropy_new_mb,
                    log_probs_new=logp_new_mb,
                    mu=mu_mb,
                )

        self.policy_old.load_state_dict(self.policy.state_dict())

    def _learn_epoch(self, states_batch, log_probs_batch, normalized_returns,
                    actions_batch, entropy_batch, num_minibatches):
        """
        Flat-path epoch learner retained for compatibility.
        Computes log_probs_new per minibatch and passes into train_policy.
        """
        dataset = torch.utils.data.TensorDataset(
            states_batch, log_probs_batch, normalized_returns, actions_batch, entropy_batch
        )

        batch_size = max(1, len(states_batch) // max(1, num_minibatches))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for (states_minibatch, log_probs_minibatch, returns_minibatch,
            actions_minibatch, entropy_minibatch) in data_loader:

            if len(states_minibatch) < max(batch_size // 2, 2):
                continue

            # Advantages from current value network
            with torch.no_grad():
                updated_values = self.value(states_minibatch).squeeze(-1)
            advantages = returns_minibatch - updated_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # New log-probs and entropy for this minibatch
            _, logp_new_mb, dist = self.sampler(self.policy, states_minibatch, actions=actions_minibatch, return_distribution=True)
            try:
                entropy_new_mb = dist.entropy()
            except NotImplementedError:
                entropy_new_mb = dist.base_dist.entropy()

            # Optional mu regularization
            mu_mb = None
            if isinstance(self.sampler, NormalActionSampler):
                base = dist.base_dist if isinstance(dist, TransformedDistribution) else dist
                base_normal = base.base_dist if isinstance(base, Independent) else base
                if hasattr(base_normal, 'loc'):
                    mu_mb = base_normal.loc

            # Train policy
            self.train_policy(
                log_probs_old=log_probs_minibatch,
                advantages=advantages,
                entropy=entropy_new_mb,
                log_probs_new=logp_new_mb,
                mu=mu_mb,
            )

    def train_policy(self, log_probs_old, advantages, entropy, log_probs_new, mu=None):
        """
        Compute PPO loss given precomputed log_probs_new (current policy)
        and log_probs_old (behavior policy) along with advantages and entropy.
        Optionally apply a small mu regularization term for Normal policies.
        """
        self.optimizer_policy.zero_grad()

        # Shape alignment and checks
        for name, t in (('log_probs_old', log_probs_old), ('advantages', advantages), ('entropy', entropy), ('log_probs_new', log_probs_new)):
            assert t.dim() in (1, 2), f"{name} must be 1D or 2D, got {t.shape}"
        if log_probs_old.dim() == 2 and log_probs_old.shape[1] == 1:
            log_probs_old = log_probs_old.squeeze(-1)
        if advantages.dim() == 2 and advantages.shape[1] == 1:
            advantages = advantages.squeeze(-1)
        if entropy.dim() == 2 and entropy.shape[1] == 1:
            entropy = entropy.squeeze(-1)
        if log_probs_new.dim() == 2 and log_probs_new.shape[1] == 1:
            log_probs_new = log_probs_new.squeeze(-1)

        assert log_probs_old.shape == advantages.shape == log_probs_new.shape, (
            f"Shape mismatch: old {log_probs_old.shape}, new {log_probs_new.shape}, adv {advantages.shape}")

        # Entropy loss (thresholded)
        m = (entropy < self.entropy_thresh)
        e_loss = -(self.entropy_coef * entropy * m).to(log_probs_old).mean()

        # Optional mu regularizer
        mu_loss = torch.tensor(0.0, device=self.device)
        if mu is not None:
            mu = torch.clamp(mu, -1e6, 1e6)
            mu_loss = 0.01 * torch.mean(mu**2 * (mu.abs() > 2))
            self.logger.log_scalar("mu loss:", mu_loss.item())

        # PPO clipped objective
        log_ratio = log_probs_new - log_probs_old
        # Clamp log ratio BEFORE exponentiating
        log_ratio_clamped = log_ratio.clamp(-10, 10)

        # Finally, compute ratio
        ratio = torch.exp(log_ratio_clamped).clamp(-18, 18)

        # Double‐check ratio shape
        assert ratio.shape == advantages.shape, (
            f"ratio ({ratio.shape}) does not match advantages ({advantages.shape}); possible broadcasting!"
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
        first_layer = getattr(self.policy, "first_layer", None)
        if first_layer is not None and hasattr(first_layer, "weight") and first_layer.weight.grad is not None:
            self.logger.log_scalar(
                "policy_first_layer_grad_norm",
                first_layer.weight.grad.norm().item()
            )
        self.logger.log_scalar("entropy mean:", entropy.mean())
        self.logger.log_scalar("entropy loss:", e_loss)

        self.optimizer_policy.step()
        self.logger.log_scalar("policy loss:", policy_loss.item())

    def get_policy_for_action(self):
        return self.policy_old

    def load_state_dict(self, sd, ignore_missing=False):
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
            batch_size = max(1, len(states_batch) // num_minibatches)
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

                # Recompute new log-probs and entropy under current policy
                _, logp_new_mb, dist = self.sampler(self.policy, states_minibatch, actions=actions_minibatch, return_distribution=True)
                try:
                    entropy_new_mb = dist.entropy()
                except NotImplementedError:
                    entropy_new_mb = dist.base_dist.entropy()

                # Optional mu
                mu_mb = None
                if isinstance(self.sampler, NormalActionSampler):
                    out = self.policy(states_minibatch)
                    mu_mb, _ = self.sampler.split_out(out)

                # Train the policy using precomputed log_probs_new
                self.train_policy(
                    log_probs_old=log_probs_minibatch,
                    advantages=advantages,
                    entropy=entropy_new_mb,
                    log_probs_new=logp_new_mb,
                    mu=mu_mb,
                )
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
