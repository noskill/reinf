from vpg import VPGBase
import numpy
import torch
from torch import optim
from torch.distributions import TransformedDistribution, Independent
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
    def __init__(self, policy, value, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, log_prefix=None, num_learning_epochs=4, clip_param=None, **kwargs):
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, log_prefix=log_prefix, num_learning_epochs=num_learning_epochs, **kwargs)
        self.policy_old = copy.deepcopy(self.policy)
        self.eps = float(clip_param) if clip_param is not None else 0.2
        self.hparams.update({'eps': self.eps})
        self.state_normalizer = RunningNorm()
        self.pad_fields = ['states', 'actions', 'returns', 'rewards', 'log_probs']

    def create_optimizers(self, **kwargs):
        if kwargs.get('joint', False):
            # Create a single optimizer over policy+value params.
            # Deduplicate parameters in case modules are shared.
            seen = set()
            joint_params = []
            for parameter in list(self.policy.parameters()) + list(self.value.parameters()):
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                joint_params.append(parameter)
            self.optimizer_policy = optim.Adam(joint_params, lr=self.policy_lr)
            self.optimizer_value = self.optimizer_policy
            return
        super().create_optimizers()
        self.optimizer_value = optim.Adam(self.value.parameters(),
                                          lr=self.value_lr, weight_decay=self.weight_decay_value)

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

    def compute_distribution_params(self, observations, actions, key_padding_mask):
        policy_params = self.sampler.policy_params(
            self.policy,
            observations,
            dict(episode_start=None),
        )
        _, logp_new_padded, dist = self.sampler(
            policy_params,
            actions=actions,
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
                mu_mb = mu_flat
        return logp_new_flat, entropy_flat, mu_mb


    def train_policy_batch_joint(self, episode_batch: EpisodeBatch, num_minibatches: int = 4):
        """joint updates of value and policy"""
        padded, padding_mask, _ = episode_batch.pad(fields=self.pad_fields)
        padding_mask = padding_mask.to(self.device)
        states_padded = to_device(padded['states'], self.device)
        actions_padded = padded['actions'].to(self.device)
        returns_padded = padded['returns'].to(self.device)
        old_logp_padded = padded['log_probs'].to(self.device)
        rewards_padded = padded['rewards'].to(self.device)

        if old_logp_padded.dim() == 3 and old_logp_padded.shape[-1] == 1:
            old_logp_padded = old_logp_padded.squeeze(-1)

        advantages = self.compute_advantage(states_batch=states_padded,
            rewards_batch=rewards_padded,
            returns_batch=returns_padded,
            padding_mask=padding_mask
        )
        # for logging/debug
        advantages_gae = self.compute_advantage_gae(
            states_batch=states_padded,
            rewards_batch=rewards_padded,
            padding_mask=padding_mask,
        )

        advantages_mc = self.compute_advantage_monte_carlo(
            states_batch=states_padded,
            returns_batch=returns_padded,
            padding_mask=padding_mask,
        )


        lengths = episode_batch.lengths.to(advantages_gae.device)
        episode_batch.data["advantages_gae"] = [
            advantages_gae[i, : lengths[i]].detach()
            for i in range(advantages_gae.shape[0])
        ]

        episode_batch.data["advantages_mc"] = [
            advantages_mc[i, : lengths[i]].detach()
            for i in range(advantages_mc.shape[0])
        ]


        self._log_training_stats(flatten_padded(actions_padded, padding_mask))

        self.policy.load_state_dict(self.policy_old.state_dict())
        obs_for_policy = self._build_policy_obs(states_padded, padding_mask)
        for minibatch in self.iterate_minibatches(
                num_minibatches,
                actions_padded.shape[0],
                obs_for_policy,
                actions_padded,
                padding_mask,
                old_logp_padded,
                advantages,
                returns_padded):

            mb_obs, mb_actions, mb_padding, mb_old_logp, mb_advantages, ret_mb = minibatch
            logp_new_flat, entropy_flat, mu_mb = self.compute_distribution_params(
                    mb_obs,
                    mb_actions,
                    mb_padding,
                )
            mb_old_logp_flat = flatten_padded(mb_old_logp.unsqueeze(-1), mb_padding).squeeze(-1).detach()
            advantages_flat = flatten_padded(mb_advantages.unsqueeze(-1), mb_padding).squeeze(-1)
            if mb_old_logp_flat.numel() == 0:
                continue
            self.optimizer_policy.zero_grad()

            mb_returns_flat = flatten_padded(ret_mb.unsqueeze(-1), mb_padding).squeeze(-1)
            if isinstance(mb_obs, dict):
                mb_value_obs = {k: v for k, v in mb_obs.items() if k != 'key_padding_mask'}
            else:
                mb_value_obs = mb_obs
            mb_value_obs_flat = self._flatten_states(mb_value_obs, mb_padding)
            value_loss = self.value_loss(mb_value_obs_flat, mb_returns_flat)

            policy_loss = self.policy_loss(
                mb_old_logp_flat,
                advantages_flat,
                entropy_flat,
                logp_new_flat,
                mu=mu_mb,
            )

            joint_loss = policy_loss + value_loss
            joint_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
            self.log_grads()
            self.optimizer_policy.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

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

        states_flat = self._flatten_states(states_padded, padding_mask)
        returns_flat = flatten_padded(returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        self._log_training_stats(flatten_padded(actions_padded, padding_mask))

        # if int(os.getenv('TRAIN_VALUE_BEFORE', 0)) == 0:
        #     advantages = self.compute_advantage_gae(
        #         states_batch=states_padded,
        #         rewards_batch=rewards_padded,
        #         padding_mask=padding_mask,
        #     )
        #     self.train_value(
        #         returns_flat,
        #         states_flat,
        #         value_epochs=2,
        #         num_minibatches=num_minibatches
        #     )
        # else:
        self.train_value(
            returns_flat,
            states_flat,
            value_epochs=2,
            num_minibatches=num_minibatches
        )

        advantages = self.compute_advantage(states_batch=states_padded,
            rewards_batch=rewards_padded,
            returns_batch=returns_padded,
            padding_mask=padding_mask
        )
        # for logging/debug
        advantages_gae = self.compute_advantage_gae(
            states_batch=states_padded,
            rewards_batch=rewards_padded,
            padding_mask=padding_mask,
        )

        advantages_mc = self.compute_advantage_monte_carlo(
            states_batch=states_padded,
            returns_batch=returns_padded,
            padding_mask=padding_mask,
        )


        lengths = episode_batch.lengths.to(advantages_gae.device)
        episode_batch.data["advantages_gae"] = [
            advantages_gae[i, : lengths[i]].detach()
            for i in range(advantages_gae.shape[0])
        ]

        episode_batch.data["advantages_mc"] = [
            advantages_mc[i, : lengths[i]].detach()
            for i in range(advantages_mc.shape[0])
        ]

        self.policy.load_state_dict(self.policy_old.state_dict())
        obs_for_policy = self._build_policy_obs(states_padded, padding_mask)
        for minibatch in self.iterate_minibatches(
                num_minibatches,
                actions_padded.shape[0],
                obs_for_policy,
                actions_padded,
                padding_mask,
                old_logp_padded,
                advantages):
            mb_obs, mb_actions, mb_padding, mb_old_logp, mb_advantages = minibatch
            logp_new_flat, entropy_flat, mu_mb = self.compute_distribution_params(
                    mb_obs,
                    mb_actions,
                    mb_padding,
                )
            mb_old_logp_flat = flatten_padded(mb_old_logp.unsqueeze(-1), mb_padding).squeeze(-1).detach()
            advantages_flat = flatten_padded(mb_advantages.unsqueeze(-1), mb_padding).squeeze(-1)
            if mb_old_logp_flat.numel() == 0:
                continue
            self.optimizer_policy.zero_grad()
            policy_loss = self.policy_loss(mb_old_logp_flat, advantages_flat, entropy_flat, logp_new_flat, mu=mu_mb)
            policy_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.log_grads()

            self.optimizer_policy.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def policy_loss(self, log_probs_old, advantages, entropy, log_probs_new, mu=None, target_entropy=None, return_parts=False):
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
        e_loss, e_loss_items = self.compute_entropy_loss(
            entropy,
            log_probs_old.device,
            log_probs_old.dtype,
            target_entropy=target_entropy,
            return_parts=True,
        )

        # Optional mu regularizer
        mu_loss = torch.tensor(0.0, device=self.device)
        if mu is not None:
            mu_loss = self.mu_loss(mu)
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

        ppo_loss = policy_loss_ppo.mean()
        entropy_term = self.entropy_coef * e_loss
        mu_term = self.mu_coef * mu_loss
        policy_loss = ppo_loss + entropy_term + mu_term
        if policy_loss > 100:
            import pdb; pdb.set_trace()
        self.logger.log_scalar("entropy mean:", entropy.mean())
        self.logger.log_scalar("entropy loss:", e_loss)
        self.logger.log_scalar("policy loss:", policy_loss.item())
        if return_parts:
            return policy_loss, {
                "ppo": policy_loss_ppo,
                "entropy": self.entropy_coef * e_loss_items,
                "mu": mu_term,
            }
        return policy_loss

    def log_grads(self):
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

    def _flatten_optional_grads(self, loss, parameters):
        if not loss.requires_grad:
            return None
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        flat = [grad.detach().reshape(-1) for grad in grads if grad is not None]
        if not flat:
            return None
        return torch.cat(flat)

    def _log_loss_grad_stats(self, losses, prefix="grad_probe"):
        parameters = [param for param in self.policy.parameters() if param.requires_grad]
        grads = {
            name: self._flatten_optional_grads(loss, parameters)
            for name, loss in losses.items()
        }
        grads = {name: grad for name, grad in grads.items() if grad is not None}
        for name, grad in grads.items():
            self.logger.log_scalar(f"{prefix}/grad_norm_{name}", grad.norm().item())
        names = sorted(grads)
        for left_idx, left_name in enumerate(names):
            for right_name in names[left_idx + 1:]:
                left = grads[left_name]
                right = grads[right_name]
                denom = (left.norm() * right.norm()).clamp_min(1e-12)
                cosine = torch.dot(left, right).div(denom)
                self.logger.log_scalar(f"{prefix}/grad_cos_{left_name}_{right_name}", cosine.item())

    def _grad_probe_loss(self, loss, count=None):
        if loss.dim() == 0:
            return loss
        if count is None:
            count = getattr(self, "grad_probe_episodes", loss.shape[0])
        count = min(int(count), int(loss.shape[0]))
        assert count > 0, f"Cannot probe empty loss part with shape {loss.shape}"
        return loss[:count].mean()

    def get_policy_for_action(self):
        return self.policy_old

    def load_state_dict(self, sd, ignore_missing=False):
        super().load_state_dict(sd)
        self.policy_old.load_state_dict(self.policy.state_dict())


class PPO(PPOBase, EpisodesPoolMixin):
    pass
