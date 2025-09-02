"""
Empowerment-based PPO agent for the two-agent communication toy task.

This module provides a thin wrapper around the existing ``PPOBase``
implementation that adds

1.  an inverse model `q_psi(a | s, s')` used to approximate the
    channel-capacity style empowerment objective, **and**
2.  helper utilities to compute the intrinsic reward ``r_e =
    log q_psi(a | s, s')`` *online* during data collection.

The core reinforcement-learning logic (advantage estimation, PPO loss
etc.) remains untouched and is inherited from ``PPOBase``.  The new
components only deal with collecting ``(s, a, s')`` triples, training
the inverse model, and returning intrinsic rewards.

The design deliberately keeps changes local so that other algorithms in
the code base remain unaffected.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inverse model  q_psi(a | s , s')  – simple MLP classifier
# ---------------------------------------------------------------------------


class InverseModel(nn.Module):
    """Predict the action `a` that caused the transition `(s → s')`."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),  # output = logits over alphabet
        )


    def forward(self, s: torch.Tensor, s_next: torch.Tensor):
        x = torch.cat([s, s_next], dim=-1)
        return self.net(x)  # logits


# ---------------------------------------------------------------------------
# Empowerment-augmented PPO agent
# ---------------------------------------------------------------------------


from ppo import PPOBase  # noqa – local import after torch
from pool import EpisodesPoolMixin


class EmpowermentPPO(PPOBase, EpisodesPoolMixin):
    """PPO variant that maximises empowerment via an inverse model."""

    def __init__(
        self,
        policy: nn.Module,
        value: nn.Module,
        sampler,
        obs_dim: int,
        num_actions: int,
        *,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        inverse_lr: float = 3e-4,
        entropy_coef: float = 0.0,
        num_envs: int = 8,
        discount: float = 0.99,
        device: torch.device | str = "cpu",
        logger=None,
        num_learning_epochs: int = 4,
        clip_r_e: Tuple[float, float] = (-20.0, 20.0),
        **kwargs,
    ):
        super().__init__(
            policy,
            value,
            sampler,
            policy_lr=policy_lr,
            value_lr=value_lr,
            num_envs=num_envs,
            discount=discount,
            device=torch.device(device),
            logger=logger,
            num_learning_epochs=num_learning_epochs,
            entropy_coef=entropy_coef,
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Inverse model + optimiser
        # ------------------------------------------------------------------
        self.inverse_model = InverseModel(obs_dim, num_actions).to(self.device)
        self.optimizer_inverse = torch.optim.Adam(self.inverse_model.parameters(), lr=inverse_lr)

        # Store training hyper-parameters for checkpointing/logging
        self.hparams.update(
            {
                "inverse_lr": inverse_lr,
                "clip_r_e": clip_r_e,
                "entropy_coef": entropy_coef,
            }
        )

        self.clip_r_e = clip_r_e

        # Counters for inverse-model logging / printing
        self._inverse_step = 0
        # Print every N inverse-model updates (default 10)
        self.inverse_print_interval = kwargs.get("inverse_log_interval", 10)



    # ------------------------------------------------------------------
    # Interface used by the *trainer* to compute intrinsic rewards online.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def intrinsic_reward(self, s: torch.Tensor, s_next: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return r_e = log q_psi(a | s, s') for each env in the batch.

        Args:
            s:       (B, obs_dim)     tensor of *current* observations
            s_next:  (B, obs_dim)     tensor of next observations
            a:       (B,) or (B,1)    tensor of *integer* actions taken
        """
        logits = self.inverse_model(s.to(self.device), s_next.to(self.device))
        log_probs = F.log_softmax(logits, dim=-1)

        if a.dim() == 2 and a.shape[1] == 1:
            a = a.squeeze(-1)
        assert a.dim() == 1, "Action tensor must be shape (batch,) after squeeze"

        # Gather the log-probabilities of the actually executed actions
        idx = a.long().unsqueeze(-1)
        r_e = log_probs.gather(-1, idx).squeeze(-1)

        # Clip for numerical stability
        low, high = self.clip_r_e
        r_e = torch.clip(r_e, low, high)
        return r_e.detach()  # detach → treat as constant w.r.t. policy gradients



    # ------------------------------------------------------------------
    # Override learning step to include inverse-model update
    # ------------------------------------------------------------------

    def learn_from_episodes(self, episodes, num_minibatches: int = 4):
        """Override to 1) augment rewards with empowerment and 2) train inverse model."""

        # ---------------------------------------------------------------
        # 1. Unpack episode data (states, actions, etc.)
        # ---------------------------------------------------------------
        data = self._extract_episode_data(episodes)
        states_list = data["states"]
        actions_list = data["actions"]
        log_probs_list = data["log_probs"]
        rewards_list = data["rewards"]  # extrinsic (zero) – will be augmented
        entropy_list = data["entropy"]

        if not states_list:
            return

        episode_lengths = [len(s) for s in states_list]

        # ---------------------------------------------------------------
        # 2.  Compute empowerment reward per episode
        # ---------------------------------------------------------------
        #   additional_reward_per_episode is a list with the same length as
        #   rewards_list; each element is a tensor of per-step intrinsic
        #   rewards.
        additional_reward_per_episode = self.compute_additional_reward(
            states_list, actions_list, entropy_list, episode_lengths=episode_lengths
        )

        # Combine extrinsic + intrinsic rewards
        combined_rewards = []
        for idx in range(len(rewards_list)):
            r_ext = rewards_list[idx].to(additional_reward_per_episode[idx])
            r_total = r_ext + additional_reward_per_episode[idx]
            combined_rewards.append(r_total)

        # ---------------------------------------------------------------
        # 3.  Prepare batches (states, returns, etc.)
        # ---------------------------------------------------------------
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = (
            self._prepare_batches(
                states_list, log_probs_list, combined_rewards, actions_list, entropy_list
            )
        )

        # ---------------------------------------------------------------
        # 4.  Train value & policy  (standard PPOBase functionality)
        # ---------------------------------------------------------------

        # Normalise returns before value update
        normalized_returns = self._normalize_returns(returns_batch)

        # Train value network first
        self.train_value(normalized_returns, states_batch)

        # Sync policies before policy optimisation
        self.policy.load_state_dict(self.policy_old.state_dict())

        # Policy optimisation epochs
        for _ in range(self.num_learning_epochs):
            self._learn_epoch(
                states_batch,
                log_probs_batch,
                normalized_returns,
                actions_batch,
                entropy_batch,
                num_minibatches,
            )

        # Update old policy to match optimised policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # ---------------------------------------------------------------
        # 5.  Train inverse model on (s,a,s') pairs across all episodes
        # ---------------------------------------------------------------

        self._train_inverse_model(states_list, actions_list)

    # ------------------------------------------------------------------
    # Auxiliary routines
    # ------------------------------------------------------------------

    def _train_inverse_model(self, states_list, actions_list):
        """One gradient step of inverse model using collected transitions."""
        if not states_list:
            return

        # Build dataset of (s, s_next, a)  -- skip last step of each episode
        s_batch = []
        s_next_batch = []
        a_batch = []
        for s_ep, a_ep in zip(states_list, actions_list):
            if len(s_ep) < 2:
                continue  # cannot form pairs
            s_batch.append(s_ep[:-1])
            s_next_batch.append(s_ep[1:])
            a_batch.append(a_ep[:-1])

        if not s_batch:
            return

        s_batch = torch.cat(s_batch, dim=0).to(self.device)
        s_next_batch = torch.cat(s_next_batch, dim=0).to(self.device)
        a_batch = torch.cat(a_batch, dim=0).to(self.device)
        if a_batch.dim() == 2 and a_batch.shape[-1] == 1:
            a_batch = a_batch.squeeze(-1)
        a_batch = a_batch.long()

        logits = self.inverse_model(s_batch, s_next_batch)
        loss = F.cross_entropy(logits, a_batch)

        # Accuracy -----------------------------------------------------
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == a_batch).float().mean()

        self.optimizer_inverse.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(), 1.0)
        self.optimizer_inverse.step()

        if self.logger is not None:
            self.logger.log_scalar("inverse_loss", loss.item())
            self.logger.log_scalar("inverse_acc", acc.item())

        # Periodic console print --------------------------------------
        self._inverse_step += 1
        if self._inverse_step % self.inverse_print_interval == 0:
            print(
                f"[InverseModel] step {self._inverse_step}  loss={loss.item():.4f}  acc={acc.item():.3f}"
            )

    # ------------------------------------------------------------------
    # Empowerment reward helper
    # ------------------------------------------------------------------

    def compute_additional_reward(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        entropies: list[torch.Tensor],
        *,
        episode_lengths: list[int],
        **kwargs,
    ) -> list[torch.Tensor]:
        """Return *intrinsic* reward tensors, one per episode.

        This variant uses the *saved* per-step entropies that were already
        computed in ``get_action`` and stored via ``add_transition`` thereby
        avoiding an extra log-probability calculation from the policy.

        Args:
            states:         list of (L_i × obs_dim) tensors per episode
            actions:        list of (L_i) or (L_i × 1) tensors per episode
            entropies:      list of (L_i) tensors with distribution entropy
            episode_lengths: list of lengths L_i

        Returns:
            List ``R_i`` of shape (L_i,) containing the intrinsic reward for
            **each** timestep of each episode.
        """

        res: list[torch.Tensor] = []
        for s_ep, a_ep, ent_ep, L in zip(states, actions, entropies, episode_lengths):
            if L < 2:
                res.append(torch.zeros(L, device=self.device))
                continue

            # Current and next states (skip last step for next-state reference)
            s_curr = s_ep[:-1]
            s_next = s_ep[1:]

            # Corresponding actions and entropies (ensure 1-D shape)
            a_curr = a_ep[:-1]
            if a_curr.dim() == 2 and a_curr.shape[-1] == 1:
                a_curr = a_curr.squeeze(-1)

            ent_curr = ent_ep[:-1]
            if ent_curr.dim() == 2 and ent_curr.shape[-1] == 1:
                ent_curr = ent_curr.squeeze(-1)

            # ----------------------------------------------------------------
            # Inverse-model term  r_e = log q_ψ(a | s , s′)
            # ----------------------------------------------------------------
            with torch.no_grad():
                logp = F.log_softmax(
                    self.inverse_model(s_curr.to(self.device), s_next.to(self.device)),
                    dim=-1,
                )
                idx = a_curr.long().unsqueeze(-1)
                r_e = logp.gather(-1, idx).squeeze(-1)

            # Clip for numerical stability
            low, high = self.clip_r_e
            r_e = torch.clamp(r_e, low, high)

            # Total intrinsic reward: inverse-model + entropy bonus
            r_intr = r_e + ent_curr.detach()

            # Pad with zero for the final step (no successor state)
            r_episode = torch.zeros(L, device=self.device)
            r_episode[:-1] = r_intr

            res.append(r_episode)

        return res

    # ------------------------------------------------------------------
    # State-dict helpers so that checkpoints include inverse model.
    # ------------------------------------------------------------------

    def get_state_dict(self):
        sd = super().get_state_dict()
        sd.update(
            {
                "inverse_model": self.inverse_model.state_dict(),
                "optimizer_inverse": self.optimizer_inverse.state_dict(),
            }
        )
        return sd

    def load_state_dict(self, sd, ignore_missing: bool = False):
        super().load_state_dict(sd, ignore_missing=ignore_missing)
        if "inverse_model" in sd:
            self.inverse_model.load_state_dict(sd["inverse_model"], strict=not ignore_missing)
        if "optimizer_inverse" in sd and not ignore_missing:
            self.optimizer_inverse.load_state_dict(sd["optimizer_inverse"])
