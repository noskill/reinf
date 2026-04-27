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
from util import RunningNorm, EpisodeBatch, flatten_padded, to_device
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
    def __init__(self, policy, value, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, num_learning_epochs=4, clip_param=None, **kwargs):
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, **kwargs)
        self.policy_old = copy.deepcopy(self.policy)
        self.eps = float(clip_param) if clip_param is not None else 0.2
        self.hparams.update({'eps': self.eps})
        self.state_normalizer = RunningNorm()
        self.num_learning_epochs = num_learning_epochs
        self.pad_fields = ['states', 'actions', 'returns', 'rewards', 'log_probs']

    def learn_from_episodes(self, episodes, num_minibatches=4):
        batch = self._prepare_episode_batch(episodes)
        if batch is None:
            return
        self.train_policy_batch(batch, num_minibatches=num_minibatches)

    def _flatten_states(self, states_padded, key_padding_mask):
        if isinstance(states_padded, dict):
            return {k: flatten_padded(v, key_padding_mask) for k, v in states_padded.items()}
        return flatten_padded(states_padded, key_padding_mask)

    def _build_policy_obs(self, states_padded, key_padding_mask):
        if isinstance(states_padded, dict):
            obs_for_policy = dict(states_padded)
            obs_for_policy['key_padding_mask'] = key_padding_mask
            return obs_for_policy
        return states_padded

    def _update_policy(self, obs_for_policy, actions_padded, key_padding_mask, old_logp_flat, advantages_flat, num_minibatches):
        N = old_logp_flat.shape[0]
        if N == 0:
            return
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

    def train_policy_batch(self, episode_batch: EpisodeBatch, num_minibatches: int = 4):
        padded, padding_mask, _ = episode_batch.pad(fields=self.pad_fields)
        padding_mask = padding_mask.to(self.device)
        states_padded = to_device(padded['states'], self.device)
        actions_padded = padded['actions'].to(self.device)
        returns_padded = padded['returns'].to(self.device)
        old_logp_padded = padded['log_probs'].to(self.device)
        rewards_padded = padded['rewards'].to(self.device)

        if old_logp_padded.dim() == 3 and old_logp_padded.shape[-1] == 1:
            old_logp_padded = old_logp_padded.squeeze(-1)
        old_logp_flat = flatten_padded(old_logp_padded.unsqueeze(-1), padding_mask).squeeze(-1).detach()

        advantages = self.compute_advantage_gae(
            states_batch=states_padded,
            rewards_batch=rewards_padded,
            padding_mask=padding_mask,
        )
        if advantages.dim() == 2:
            advantages = flatten_padded(advantages.unsqueeze(-1), padding_mask).squeeze(-1)

        self._log_training_stats(flatten_padded(actions_padded, padding_mask))

        states_flat = self._flatten_states(states_padded, padding_mask)
        returns_flat = flatten_padded(returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        mini_batch_size = max(1, int(returns_flat.shape[0]) // max(1, num_minibatches))
        self.train_value(
            returns_flat,
            states_flat,
            value_epochs=2,
            mini_batch_size=mini_batch_size,
        )

        self.policy.load_state_dict(self.policy_old.state_dict())
        obs_for_policy = self._build_policy_obs(states_padded, padding_mask)
        self._update_policy(
            obs_for_policy,
            actions_padded,
            padding_mask,
            old_logp_flat,
            advantages,
            num_minibatches,
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train_sequence_policy(self, episode_batch: EpisodeBatch, num_minibatches: int = 4):
        self.train_policy_batch(episode_batch, num_minibatches=num_minibatches)

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
        e_loss = self.compute_entropy_loss(entropy, log_probs_old.device, log_probs_old.dtype)

        # Optional mu regularizer
        mu_coef = 0.001
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

        policy_loss = policy_loss_ppo.mean() + self.entropy_coef * e_loss + mu_loss * mu_coef
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
