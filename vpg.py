import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
np = numpy
from pool import EpisodesPoolMixin
from reinforce import ReinforceBase
from util import RunningNorm


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
    def __init__(self, policy, value, sampler, policy_lr=0.0001, value_lr=0.001, num_envs=8,
                 discount=0.99, device=torch.device('cpu'), logger=None, entropy_coef=0.01, **kwargs):
        self.value = value
        self.value_lr = value_lr
        self.weight_decay_value = 0
        super().__init__(policy, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, entropy_coef=entropy_coef, **kwargs)
        self.sampler = sampler
        self.create_optimizers()
        self.hparams.update({
            'value_lr': value_lr,
            'weight_decay_vl': self.weight_decay_value
        })
        self.state_normalizer = RunningNorm()

    def create_optimizers(self):
        super().create_optimizers()
        self.optimizer_value = optim.Adam(self.value.parameters(),
                                          lr=self.value_lr, weight_decay=self.weight_decay_value)

    def learn_from_episodes(self, episodes):
        # Extract per-episode tensors/lists
        data_dict = self._extract_episode_data(episodes)
        states_list = data_dict['states']
        log_probs_list = data_dict['log_probs']
        actions_list = data_dict['actions']
        rewards_list = data_dict['rewards']
        entropy_list = data_dict['entropy']
        if not states_list:
            return

        # Prepare batches: compute discounted returns, and then aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = self._prepare_batches(
            states_list, log_probs_list, rewards_list, actions_list, entropy_list
        )

        states_batch = self.state_normalizer(states_batch)
        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)

        # Log statistics
        self._log_training_stats(actions_batch)
        self.train_value(normalized_returns, states_batch, value_epochs=3)

        # Policy Update
        with torch.no_grad():
            updated_values = self.value(states_batch).squeeze(-1)

        advantages = normalized_returns - updated_values
        advantage_std = advantages.std()
        advantage_mean = advantages.mean()
        self.logger.log_scalar("Raw advantage mean:", advantage_mean.item())
        self.logger.log_scalar("Raw advantage std:", advantage_std.item())
        advantages = (advantages - advantage_mean) / advantage_std
        # Finally, train the policy (using log probs computed from current policy)
        if len(entropy_batch.shape) == 2 and entropy_batch.shape[1] == 1:
            entropy_batch = entropy_batch.flatten()
        self.train_policy(log_probs_batch, advantages,
                          entropy_batch, states_batch, actions_batch)

    def train_value(self, returns, states_batch, value_epochs = 2,  mini_batch_size = 128):
        # Value Network Update
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

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.value.load_state_dict(sd['value'])
        self.optimizer_value.load_state_dict(sd['optimizer_value'])


class VPG(VPGBase, EpisodesPoolMixin):
    pass



