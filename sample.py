# sample.py

import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.nn.functional as F

class ActionSampler:
    def __call__(self, policy, state):
        raise NotImplementedError

    def _prepare_state(self, state):
        if isinstance(state, dict):
            return state
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if necessary
        return state


class NormalActionSampler(ActionSampler):
    def __init__(self, action_dim, a_min=-2.0, a_max=2.0, reparameterize=False, transform=True):
        """
        Parameters:
            action_dim (int): Number of action dimensions.
            a_min (float): Minimum action value.
            a_max (float): Maximum action value.
            reparameterize (bool): If True, uses differentiable rsample(), otherwise non-differentiable sample().
        """
        self.action_dim = action_dim
        self.a_min = a_min
        self.a_max = a_max
        self.reparameterize = reparameterize
        self.transform = transform

    def __call__(self, policy, state, policy_kwargs=dict(), actions=None, return_distribution=False):
        state = self._prepare_state(state)

        # Forward pass: network outputs [.., 2 * action_dim]
        out = policy(state, **policy_kwargs)
        assert out.shape[-1] == 2 * self.action_dim, \
            f"Network output dimension {out.shape[-1]} does not match 2 * action_dim ({2 * self.action_dim})"

        # Split outputs into mu and log_sigma
        mu, log_sigma = torch.chunk(out, chunks=2, dim=-1)

        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, -20, 2)
        sigma = F.softplus(log_sigma) + 1e-6

        # Base Normal distribution (diagonal covariance), aggregate over event dims
        base_dist = Normal(mu, sigma)
        independent_dist = Independent(base_dist, 1)

        # Optional tanh + affine transform to match action bounds
        if self.transform:
            transforms = [
                TanhTransform(cache_size=1),
                AffineTransform(
                    loc=(self.a_min + self.a_max) / 2.0,
                    scale=(self.a_max - self.a_min) / 2.0,
                ),
            ]
            dist = TransformedDistribution(independent_dist, transforms)
        else:
            dist = independent_dist

        # Evaluate provided actions or sample
        if actions is not None:
            # Use given actions to compute log-probs; do not sample
            log_prob = dist.log_prob(actions)
            action = actions
        else:
            if self.reparameterize:
                action = dist.rsample()
            else:
                action = dist.sample()
            action = torch.clamp(action, self.a_min + 1e-4, self.a_max - 1e-4)
            log_prob = dist.log_prob(action)

        # Entropy: prefer direct computation; fall back to independent (pre-transform)
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            entropy = independent_dist.entropy()

        # Shape safety: allow [B] or [B,T]
        if log_prob.dim() not in (1, 2):
            raise AssertionError(f"log_prob has invalid shape {log_prob.shape}; expected (B,) or (B,T)")
        if entropy.dim() not in (1, 2):
            raise AssertionError(f"entropy has invalid shape {entropy.shape}; expected (B,) or (B,T)")

        if return_distribution:
            return action, log_prob, dist
        return action, log_prob, dist

    def split_out(self, out):
        return  torch.chunk(out, chunks=2, dim=-1)


class BetaActionSampler(ActionSampler):
    def __init__(self, a_min=-2.0, a_max=2.0):
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, policy, state, policy_kwargs=dict(), actions=None, return_distribution=False):
        state = self._prepare_state(state)

        params = policy(state, **policy_kwargs)
        # Assume 1D action by default; if more dims, split equally
        half = params.shape[-1] // 2
        alpha = params[..., :half]
        beta = params[..., half:]

        # Clip parameters
        alpha = F.softplus(alpha) + 1e-6
        beta = F.softplus(beta) + 1e-6

        base_dist = Beta(alpha, beta)
        # Apply affine scaling to match action bounds
        transforms = [AffineTransform(loc=self.a_min, scale=self.a_max - self.a_min)]
        dist = TransformedDistribution(base_dist, transforms)

        if actions is not None:
            action = actions
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            action = torch.clamp(action, self.a_min + 1e-6, self.a_max - 1e-6)
            log_prob = dist.log_prob(action)

        # Sum over event dims if multi-dimensional action
        if log_prob.dim() >= 2 and log_prob.shape[-1] != 0:
            # If Beta returned elementwise, reduce last dim
            if action.dim() == log_prob.dim() and log_prob.shape[-1] == action.shape[-1]:
                log_prob = log_prob.sum(dim=-1)

        try:
            entropy = dist.entropy()
        except NotImplementedError:
            entropy = base_dist.entropy()

        if return_distribution:
            return action, log_prob, dist
        return action, log_prob, dist


class DiscreteActionSampler(ActionSampler):
    def __call__(self, policy, state, policy_kwargs=dict(), actions=None, return_distribution=False):
        state = self._prepare_state(state)
        logits = policy(state, **policy_kwargs)
        dist = Categorical(logits=logits)

        if actions is not None:
            action = actions.to(torch.long)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Keep shape as [B] or [B,T]; add last dim only if needed by callers
        if log_prob.dim() == 1:
            log_prob_out = log_prob
        else:
            log_prob_out = log_prob  # [B,T]

        if return_distribution:
            return action.to(torch.int32), log_prob_out, dist
        return action.to(torch.int32), log_prob_out, dist
