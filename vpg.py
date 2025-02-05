import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
np = numpy

from reinforce import ReinforceBase


class Value(nn.Module):
    def __init__ (self, state_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        value = self.value_head(x)
        return value


class VPGBase(ReinforceBase):
    def __init__(self, policy, value, sampler, policy_lr=0.001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None):
        super().__init__(policy, sampler, policy_lr=policy_lr, 
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger)
        self.value = value
        self.sampler = sampler
        weight_decay_value = 0.001
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=value_lr, weight_decay=weight_decay_value)
        logger.add_hparams(dict(value_lr=value_lr, weight_decay_vl=weight_decay_value), dict())

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
        
        self.train_value(normalized_returns, states_batch)
        
        # Policy Update
        with torch.no_grad():
            updated_values = self.value(states_batch).squeeze(-1)

        advantages = normalized_returns - updated_values
        self.logger.log_scalar("Raw advantage mean:", advantages.mean().item())
        self.logger.log_scalar("Raw advantage std:", advantages.std().item())
        # Finally, train the policy (using log probs computed from current policy)
        if len(entropy_batch.shape) == 2 and entropy_batch.shape[1] == 1:
            entropy_batch = entropy_batch.flatten()
        self.train_policy(log_probs_batch, advantages, entropy_batch, states_batch)

    def train_value(self, returns, states_batch):
        # Value Network Update
        value_epochs = 2
        mini_batch_size = 256
        
        for epoch in range(value_epochs):

            indices = np.random.permutation(len(states_batch))
            for start in range(0, len(states_batch), mini_batch_size):
                self.optimizer_value.zero_grad()
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                pred_values = self.value(states_batch[batch_idx].detach()).squeeze(-1)
                value_loss = F.mse_loss(pred_values, returns[batch_idx])
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
                self.optimizer_value.step()
                
        self.logger.log_scalar(f"value loss",  value_loss)


from pool import *


class VPG(VPGBase, EpisodesPoolMixin):
    pass


class VPG(VPGBase, EpisodesPoolMixin):
    pass



