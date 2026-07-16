import numpy as np
import torch

from ppo import PPO
from util import EpisodeBatch, compute_returns_list, flatten_padded, gae, normalize_padded_returns, to_device


class LowLevelAgent(PPO):
    grad_probe_episodes = 10

    def update(self, rewards, dones, info=None, **kwargs):
        for env_idx in range(self.num_envs):
            self.add_reward(env_idx, rewards[env_idx])
        self.process_dones(dones)
        return False

    def _prepare_hindsight_batch(self, states_per_episode, actions_per_episode, weights_per_episode, negative_states_per_episode):
        assert len(states_per_episode) == len(actions_per_episode) == len(weights_per_episode) == len(negative_states_per_episode), "hindsight batch count mismatch"
        if not states_per_episode:
            return None
        return EpisodeBatch({
            "states": states_per_episode,
            "negative_states": negative_states_per_episode,
            "actions": actions_per_episode,
            "weights": weights_per_episode,
        }).to(self.device)

    def compute_hindsight_loss(self, states_padded, negative_states_padded, actions_padded, 
                               weights_padded,padding_mask, coef=0.05, margin=0.2, return_parts=False):
        weights_flat = flatten_padded(weights_padded.unsqueeze(-1).to(self.device), padding_mask).squeeze(-1)
        selected = weights_flat > 0.0
        if not selected.any():
            self.logger.log_scalar("hindsight_loss", 0.0)
            self.logger.log_scalar("hindsight_selected_frac", 0.0)
            loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if return_parts:
                return loss, {}, None
            return loss

        weights_selected = weights_flat[selected].detach().clamp(0.0, 5.0)
        logits_pos = self.policy(states_padded)
        _, logp_pos_padded, dist = self.sampler(logits_pos, actions=actions_padded, return_distribution=True)
        logits_neg = self.policy(negative_states_padded)
        _, logp_neg_padded, _ = self.sampler(logits_neg, actions=actions_padded, return_distribution=True)
        logp_pos_flat = flatten_padded(logp_pos_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        logp_neg_flat = flatten_padded(logp_neg_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        rank_error = torch.relu(margin + logp_neg_flat[selected] - logp_pos_flat[selected])
        rank_loss_items = coef * rank_error * weights_selected
        rank_loss = rank_loss_items.mean()
        loss = rank_loss
        self.logger.log_scalar("hindsight_loss", loss.detach().item())
        self.logger.log_scalar("hindsight_rank_loss", rank_loss.detach().item())
        self.logger.log_scalar("hindsight_logp_pos", logp_pos_flat[selected].mean().detach().item())
        self.logger.log_scalar("hindsight_logp_neg", logp_neg_flat[selected].mean().detach().item())
        self.logger.log_scalar("hindsight_rank_acc", (logp_pos_flat[selected] > logp_neg_flat[selected]).float().mean().detach().item())
        self.logger.log_scalar("hindsight_weight_mean", weights_flat[selected].mean().item())
        self.logger.log_scalar("hindsight_selected_frac", selected.float().mean().item())
        if return_parts:
            return loss, {"rank": rank_loss_items}
        return loss

    def learn_from_episodes(self, episodes, num_minibatches=4, hindsight=None):
        batch = self._prepare_episode_batch(episodes)
        if batch is None:
            return
        hindsight_batch = None
        if hindsight is not None:
            hindsight_batch = self._prepare_hindsight_batch(*hindsight)
        self.train_policy_batch(batch, num_minibatches=num_minibatches, hindsight_batch=hindsight_batch)

    def train_policy_batch(self, episode_batch: EpisodeBatch, num_minibatches: int = 4, hindsight_batch=None):
        padded, padding_mask, _ = episode_batch.pad(fields=self.pad_fields)
        padding_mask = padding_mask.to(self.device)
        states_padded = to_device(padded['states'], self.device)
        actions_padded = padded['actions'].to(self.device)
        returns_padded = padded['returns'].to(self.device)
        old_logp_padded = padded['log_probs'].to(self.device)
        rewards_padded = padded['rewards'].to(self.device)
        assert torch.isfinite(rewards_padded).all()
        assert torch.isfinite(returns_padded).all()


        if old_logp_padded.dim() == 3 and old_logp_padded.shape[-1] == 1:
            old_logp_padded = old_logp_padded.squeeze(-1)

        states_flat = self._flatten_states(states_padded, padding_mask)
        returns_flat = flatten_padded(returns_padded.unsqueeze(-1), padding_mask).squeeze(-1)
        self._log_training_stats(flatten_padded(actions_padded, padding_mask))

        self.train_value(
            returns_flat,
            states_flat,
            value_epochs=2,
            num_minibatches=num_minibatches
        )

        for n, p in self.value.named_parameters():
            assert torch.isfinite(p).all(), f"NaN in low value param {n}"


        advantages = self.compute_advantage(states_batch=states_padded,
            rewards_batch=rewards_padded,
            returns_batch=returns_padded,
            padding_mask=padding_mask
        )
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

        hindsight_minibatches = None
        if hindsight_batch is not None:
            h_padded, h_padding_mask, _ = hindsight_batch.pad(fields=["states", "negative_states", "actions", "weights"])
            h_padding_mask = h_padding_mask.to(self.device)
            h_states_padded = to_device(h_padded["states"], self.device)
            h_negative_states_padded = to_device(h_padded["negative_states"], self.device)
            h_actions_padded = h_padded["actions"].to(self.device)
            h_weights_padded = h_padded["weights"].to(self.device)
            hindsight_minibatches = list(self.iterate_minibatches(
                num_minibatches,
                h_actions_padded.shape[0],
                h_states_padded,
                h_negative_states_padded,
                h_actions_padded,
                h_padding_mask,
                h_weights_padded,
            ))

        self.policy.load_state_dict(self.policy_old.state_dict())
        obs_for_policy = self._build_policy_obs(states_padded, padding_mask)
        update_idx = 0
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
            policy_loss, policy_loss_parts = self.policy_loss(
                mb_old_logp_flat,
                advantages_flat,
                entropy_flat,
                logp_new_flat,
                mu=mu_mb,
                return_parts=True,
            )
            hindsight_loss = torch.tensor(0.0, dtype=policy_loss.dtype, device=policy_loss.device)
            hindsight_loss_parts = {}
            h_minibatch = None
            if hindsight_minibatches:
                h_minibatch = hindsight_minibatches[update_idx % len(hindsight_minibatches)]
                h_mb_states, h_mb_negative_states, h_mb_actions, h_mb_padding, h_mb_weights = h_minibatch
                hindsight_loss, hindsight_loss_parts = self.compute_hindsight_loss(
                    h_mb_states,
                    h_mb_negative_states,
                    h_mb_actions,
                    h_mb_weights,
                    h_mb_padding,
                    return_parts=True,
                )
                self.logger.log_scalar("combined_hindsight_loss", hindsight_loss.detach().item())
            if update_idx == 0:
                grad_losses = {
                    "ppo": self._grad_probe_loss(policy_loss_parts["ppo"]),
                    "entropy": self._grad_probe_loss(policy_loss_parts["entropy"]),
                }
                if "rank" in hindsight_loss_parts:
                    grad_losses["rank"] = self._grad_probe_loss(hindsight_loss_parts["rank"])
                self._log_loss_grad_stats(grad_losses)
             
            total_loss = policy_loss + hindsight_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.log_grads()
            self.optimizer_policy.step()
            update_idx += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
