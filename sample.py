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

    def __call__(self, policy, state):
        state = self._prepare_state(state)

        # Forward pass: network outputs [mu, log_sigma] per action dimension
        out = policy(state)
        assert out.shape[-1] == 2 * self.action_dim, \
            f"Network output dimension {out.shape[-1]} does not match 2 * action_dim ({2 * self.action_dim})"

        # Split outputs into mu and log_sigma
        mu, log_sigma = torch.chunk(out, chunks=2, dim=-1)

        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, -20, 2)

        # Get sigma from log_sigma
        sigma = F.softplus(log_sigma) + 1e-6

        # Base Normal distribution (diagonal covariance)
        base_dist = Normal(mu, sigma)

        # Construct the distribution such that `log_prob` already aggregates over
        # the *event* dimensions, yielding exactly **one** scalar per action
        # vector in the batch.

        # Prepare an `Independent` wrapper up-front so we can refer to it
        # regardless of the branch taken below.
        independent_dist = Independent(base_dist, 1)

        if self.transform:
            # 1. First re-interpret the Normal as an `Independent` distribution so
            #    the event dim (=`action_dim`) gets treated jointly.
            # 2. Apply non-linear squashing (tanh) followed by affine scaling to
            #    match the environment action bounds.
            transforms = [
                TanhTransform(cache_size=1),
                AffineTransform(
                    loc=(self.a_min + self.a_max) / 2.0,
                    scale=(self.a_max - self.a_min) / 2.0,
                ),
            ]
            transformed_dist = TransformedDistribution(independent_dist, transforms)
        else:
            transformed_dist = independent_dist

        # Sample action (with reparameterization if requested)
        if self.reparameterize:
            action = transformed_dist.rsample()
        else:
            action = transformed_dist.sample()

        # Clamp action to valid boundaries
        action = torch.clamp(action, self.a_min + 1e-4, self.a_max - 1e-4)

        # Log probability of selected action. `TransformedDistribution.log_prob` already
        # aggregates across the event dimensions, so additional averaging is unnecessary
        # and incorrectly shrinks the tensor. We keep the original shape returned by
        # `log_prob` to maintain consistency with the action/entropy tensors.
        log_prob = transformed_dist.log_prob(action)

        # ------------------------------------------------------------------
        # Shape safety checks
        # ------------------------------------------------------------------
        # Expect *exactly* one log-prob scalar per sample (B,) or (B,1)
        assert log_prob.dim() == 1 or (log_prob.dim() == 2 and log_prob.shape[1] == 1), \
            f"log_prob has invalid shape {log_prob.shape}; expected (B,) or (B,1)."

        # Likewise, each entropy call on `transformed_dist` must obey this rule
        # Compute entropy if available; fall back to base distribution otherwise
        try:
            entropy = transformed_dist.entropy()
        except NotImplementedError:
            entropy = transformed_dist.base_dist.entropy()

        assert entropy.dim() == 1 or (entropy.dim() == 2 and entropy.shape[1] == 1), \
            f"entropy has invalid shape {entropy.shape}; expected (B,) or (B,1)."

        return action, log_prob, transformed_dist

    def split_out(self, out):
        return  torch.chunk(out, chunks=2, dim=-1)


class BetaActionSampler(ActionSampler):
    def __init__(self, a_min=-2.0, a_max=2.0):
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, policy, state):
        state = self._prepare_state(state)

        params = policy(state)
        alpha = params[..., :1]
        beta = params[..., 1:]

        # Clip parameters
        alpha = F.softplus(alpha) + 1e-6
        beta = F.softplus(beta) + 1e-6

        base_dist = Beta(alpha, beta)
        transforms = [AffineTransform(loc=self.a_min, scale=self.a_max - self.a_min)]
        transformed_dist = TransformedDistribution(base_dist, transforms)

        action = transformed_dist.sample()
        action = torch.clamp(action, -self.a_min + 1e-6, self.a_max - 1e-6)

        log_prob = transformed_dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, transformed_dist


class DiscreteActionSampler(ActionSampler):
    def __call__(self, policy, state):
        state = self._prepare_state(state)
        logits = policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.to(torch.int32), log_prob.unsqueeze(-1), dist
