from vpg import VPG
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


class PPO(VPG, EpisodesPoolMixin):
    def __init__(self, policy, value, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, **kwargs):
        super().__init__(policy, value, sampler, policy_lr=policy_lr, 
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger)
        self.policy_old = copy.deepcopy(self.policy)
        self.eps = 0.2
        self.hparams.update({'eps': self.eps})
        self.state_normalizer = RunningNorm()

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

        log_probs_new = dist.log_prob(actions_batch)
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

        # Finally, compute ratio
        # Avoid doing .reshape(log_probs.shape); they should be the same shape anyway.
        ratio = torch.exp(log_probs_new - log_probs_old).clamp(-20, 20)

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

    def get_action(self, state, done):
        active_mask = ~done
        if not any(active_mask):
            return np.zeros((self.num_envs,) + self.policy_old.action_shape)

        active_states = state[active_mask]
        if not isinstance(active_states, torch.Tensor):
            active_states = torch.FloatTensor(active_states)
        actions, log_probs, dist = self.sampler(self.policy_old, active_states)
        if isinstance(dist, TransformedDistribution):
            entropy = dist.base_dist.entropy()
        else:
            entropy = dist.entropy()
        full_actions = torch.zeros(
            (active_mask.shape[0], actions.shape[1] if len(actions.shape) > 1 else 1)
        ).to(actions)
        full_actions[active_mask] = actions.reshape(full_actions[active_mask].shape)

        active_env_indices = torch.where(active_mask)[0]
        for idx, env_idx in enumerate(active_env_indices):
            self.add_transition(env_idx, active_states[idx], actions[idx], log_probs[idx], entropy[idx])

        return full_actions.flatten()
