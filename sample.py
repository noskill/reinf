import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from agent import Agent


np = numpy


def sample_action_normal(policy, state, a_min=-2.0, a_max=2.0):
    """
    Samples an action from a Normal distribution parametrized by policy(state),
    then applies tanh-squashing to keep outputs in [-1,1], and finally
    scales/affines them to [a_min,a_max]. Returns (action, log_prob, raw_params).
    """
    # Ensure state is a properly formatted tensor
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    if state.dim() == 1:
        state = state.unsqueeze(0) # batch dimension if necessary

    # Forward pass: e.g. network outputs [mu, log_sigma]
    out = policy(state)
    mu = out[..., :1]
    log_sigma = out[..., 1:]

    # Convert log_sigma to a positive std; softplus is a common choice
    sigma = F.softplus(log_sigma) + 1e-6

    # Base Normal distribution
    base_dist = Normal(mu, sigma)

    # A sequence of transforms: first Tanh to go from (-∞, ∞) to (-1, 1), then
    # an affine transform to go from (-1, 1) to (a_min, a_max)
    transforms = [
        TanhTransform(cache_size=1),
        AffineTransform(loc=(a_min + a_max) / 2.0, scale=(a_max - a_min) / 2.0)
        ]
    transformed_dist = TransformedDistribution(base_dist, transforms)

    # Sample an action (no gradient through sample method; use rsample for reparameterization)
    action = transformed_dist.sample()

    # Optionally clamp to ensure numerical stability
    action = torch.clamp(action, a_min + 1e-6, a_max - 1e-6)

    # Compute log probability. Note .log_prob(action) is shaped [batch_size],
    # so add dimension if you want shape [batch_size, 1].
    log_prob = transformed_dist.log_prob(action).unsqueeze(-1)

    return action, log_prob, transformed_dist


def sample_action_beta(policy, state, a_min=-2, a_max=2):
    # Ensure state is a properly formatted tensor
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    if state.dim() == 1:
        state = state.unsqueeze(0)  # Add batch dimension if necessary
    params = policy(state)
    alpha = params[..., :1]  # First half of outputs
    beta = params[..., 1:]   # Second half of outputs
    # Clip parameters to prevent extreme values
    # torch.abs is an option
    alpha = F.softplus(alpha) + 1e-6
    beta = F.softplus(beta) + 1e-6
    base_dist = Beta(alpha, beta)
    transforms = [AffineTransform(loc=a_min, scale=a_max - a_min)]
    transformed_dist = TransformedDistribution(base_dist, transforms)

    # Sample action without gradient flow through the sampling process
    action = transformed_dist.sample()
    action = torch.clamp(action, -a_min + 1e-6, a_max - 1e-6)
    # Compute log-probability of the sampled action
    # The log_prob() method depends on the distribution parameters (alpha, beta)
    log_prob = transformed_dist.log_prob(action)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob, transformed_dist


def discrete_sampler(policy, state):
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    logits = policy(state)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.to(torch.int32), log_prob.unsqueeze(-1), dist
