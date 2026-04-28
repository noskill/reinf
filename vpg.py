import numpy
import torch
from torch.distributions import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
np = numpy
from pool import EpisodesPoolMixin
from reinforce import ReinforceBase
from util import RunningNorm, EpisodeBatch, to_device, gae, flatten_padded, normalize_padded_returns


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
        self.lambda_discount = self.discount
        self.sampler = sampler
        self.create_optimizers(**kwargs)
        self.hparams.update({
            'value_lr': value_lr,
            'weight_decay_vl': self.weight_decay_value,
            'lambda_discount': self.lambda_discount
        })
        self.state_normalizer = RunningNorm()

    def create_optimizers(self, **kwargs):
        super().create_optimizers()
        self.optimizer_value = optim.Adam(self.value.parameters(),
                                          lr=self.value_lr, weight_decay=self.weight_decay_value)

    def learn_from_episodes(self, episodes):
        # Convert input into EpisodeBatch with returns
        episode_batch = self._prepare_episode_batch(episodes)
        if episode_batch is None:
            return
        # Non-sequence fallback: flatten all steps across episodes
        flat = episode_batch.flatten(fields=['states', 'returns'])
        states_batch = to_device(flat['states'], self.device)
        returns_batch = to_device(flat['returns'], self.device)

        # keep returns unnormalised - in advantage code we are substracting from
        # unnormalised rewards
        self.train_value(returns_batch, states_batch, value_epochs=3)
        self.train_policy_batch(episode_batch)

    def compute_advantage_ebatch(self, episode_batch):
        return self.compute_advantage_gae_ebatch(episode_batch)
    
    def compute_advantage(self, states_batch, returns_batch, rewards_batch, padding_mask):
        return self.compute_advantage_gae(states_batch=states_batch, returns_batch=returns_batch,
                                          rewards_batch=rewards_batch, padding_mask=padding_mask)

    def compute_advantage_monte_carlo_ebatch(self, episode_batch):
        # Non-sequence fallback: flatten all steps across episodes
        pad, padding_mask, length = episode_batch.pad(fields=['states', 'actions', 'returns', 'rewards'])
        # B,T
        states_batch = to_device(pad['states'], self.device)
        actions_batch = to_device(pad['actions'], self.device)
        returns_batch = to_device(pad['returns'], self.device)
        advantages = self.compute_advantage_monte_carlo(states_batch, returns_batch, padding_mask)
        return advantages

    def compute_advantage_monte_carlo(self, states_batch, returns_batch, padding_mask, **kwargs):
        # Policy Update
        with torch.no_grad():
            updated_values = self.value(states_batch).squeeze(-1)

        advantages = returns_batch - updated_values
        advantages, advantage_mean, advantage_std = normalize_padded_returns(advantages, padding_mask)
        self.logger.log_scalar("Raw advantage mean:", advantage_mean.item())
        self.logger.log_scalar("Raw advantage std:", advantage_std.item())
        return advantages

    def compute_advantage_gae_ebatch(self, episode_batch):
        pad, padding_mask, length = episode_batch.pad(fields=['states', 'actions', 'returns', 'rewards'])
        # B,T
        rewards_batch = to_device(pad['rewards'], self.device)
        states_batch = to_device(pad['states'], self.device)
        return self.compute_advantage_gae(states_batch, rewards_batch, padding_mask)
    
    def compute_advantage_gae(self, states_batch, rewards_batch, padding_mask, **kwargs):
        # Policy Update
        with torch.no_grad():
            updated_values = self.value(states_batch).squeeze(-1)
            updated_values = updated_values * (1 - padding_mask.float())
        advantages = gae(self.discount, self.lambda_discount, rewards_batch, updated_values)

        adv_sel = torch.masked_select(advantages, torch.logical_not(padding_mask))
        advantage_std = adv_sel.std()
        advantage_std_clamped = advantage_std.clamp_min(1e-2)
        advantage_mean = adv_sel.mean()
        self.logger.log_scalar("Raw advantage mean:", advantage_mean.item())
        self.logger.log_scalar("Raw advantage std:", advantage_std.item())
        advantages = (advantages - advantage_mean) * (1 - padding_mask.to(advantages)) / advantage_std_clamped
        return advantages

    def train_value(self, returns, states_batch, value_epochs = 2,  mini_batch_size = 128):
        # Value Network Update
        n_samples = int(returns.shape[0])
        if n_samples == 0:
            return None

        mini_batch_size = max(1, int(mini_batch_size))
        for _ in range(value_epochs):
            indices = torch.randperm(n_samples, device=returns.device)
            for start in range(0, n_samples, mini_batch_size):
                self.optimizer_value.zero_grad()
                batch_idx = indices[start:start + mini_batch_size]
                if isinstance(states_batch, dict):
                    batch_states = {k: v[batch_idx].detach() for k, v in states_batch.items()}
                else:
                    batch_states = states_batch[batch_idx].detach()
                pred_values = self.value(batch_states).squeeze(-1)
                value_loss = F.smooth_l1_loss(pred_values, returns[batch_idx])
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
                self.optimizer_value.step()
        self.logger.log_scalar("value loss", value_loss)
        
        # explained variance
        with torch.no_grad():
            pred_values = self.value(states_batch).squeeze(-1)
            residual = returns - pred_values
            returns_var = returns.var(unbiased=False)
            if returns_var > 1e-12:
                explained_variance = 1.0 - residual.var(unbiased=False) / returns_var
            else:
                explained_variance = torch.tensor(0.0, device=returns.device)
            self.logger.log_scalar("value explained variance", explained_variance)


    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.value.load_state_dict(sd['value'])
        self.optimizer_value.load_state_dict(sd['optimizer_value'])


class VPG(VPGBase, EpisodesPoolMixin):
    pass
