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
    def __init__(self, a_min=-2.0, a_max=2.0):
        self.a_min = a_min
        self.a_max = a_max
        
    def __call__(self, policy, state):
        state = self._prepare_state(state)
        
        # Forward pass: network outputs [mu, log_sigma]
        out = policy(state)
        mu = out[..., :1]
        log_sigma = out[..., 1:]
        assert out.shape[1] == 2
        log_sigma = torch.clamp(log_sigma, -20, 2)
        
        # Convert log_sigma to a positive std
        sigma = F.softplus(log_sigma) + 1e-6
        
        # Base Normal distribution
        base_dist = Normal(mu, sigma)
        
        # Transforms sequence
        transforms = [
            TanhTransform(cache_size=1),
            AffineTransform(
                loc=(self.a_min + self.a_max) / 2.0,
                scale=(self.a_max - self.a_min) / 2.0
            )
        ]
        
        transformed_dist = TransformedDistribution(base_dist, transforms)
        
        # Sample action
        action = transformed_dist.sample()
        action = torch.clamp(action, self.a_min + 1e-4, self.a_max - 1e-4)
        
        # Compute log probability
        log_prob = transformed_dist.log_prob(action).unsqueeze(-1)
        
        return action, log_prob, transformed_dist


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
