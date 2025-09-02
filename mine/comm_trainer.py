"""
Minimal on-policy trainer that does **not** depend on IsaacLab.  It is a
slimmed-down version of the generic ``OnPolicyTrainer`` in the repository
and is tailored to the communication-empowerment experiment.
"""

from __future__ import annotations

import os
import sys
import time
import random
from typing import Optional

import numpy as np
import torch


class CommTrainer:
    """Loop around the environment and EmpowermentPPO agent."""

    def __init__(
        self,
        env,
        agent,
        *,
        n_episodes: int = 20_000,
        save_interval: int = 1000,
        checkpoint_dir: str = "./logs/runs/comm_experiment",
        seed: int = 0,
        checkpoint: Optional[str] = None,
    ):
        self.env = env
        self.agent = agent
        self.n_episodes = int(n_episodes)
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = checkpoint

        self.seed = int(seed)

        # RNG ----------------------------------------------------------------
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Episode counter is kept inside the agent's logger
        # If a checkpoint path is supplied, resume training.
        if checkpoint is not None and os.path.isfile(checkpoint):
            self._load_checkpoint(checkpoint)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr
        return torch.as_tensor(arr, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run training loop for ``self.n_episodes`` additional episodes.

        If a checkpoint was loaded earlier, ``self.agent.logger.episode_count``
        already contains the number of episodes seen so far.  Training will
        therefore continue until *episode_count = start_count + n_episodes*.
        """

        target_episode = self.agent.logger.episode_count + self.n_episodes

        while self.agent.logger.episode_count < target_episode:
            obs_np, _ = self.env.reset()
            obs = self._to_tensor(obs_np)
            done = torch.zeros(self.env.num_envs, dtype=torch.bool)

            self.agent.episode_start()

            while not done.all():
                # Debug shapes
                # print('obs', obs.shape, 'done', done)
                action = self.agent.get_action(obs, done)  # Tensor (N,1)

                # Convert to numpy ints for env
                act_np = action.squeeze(-1).detach().cpu().numpy().astype(np.int64)

                next_obs_np, rew_np, terminated, truncated, _ = self.env.step(act_np)
                next_obs = self._to_tensor(next_obs_np)
                reward = self._to_tensor(rew_np)

                done = torch.as_tensor(terminated | truncated)

                # External reward is zero; intrinsic handled inside agent.
                changed = self.agent.update(obs, action, reward, done, next_obs)

                obs = next_obs

                if changed:
                    break  # Update happened (num_env episodes collected)

            # Logging / checkpointing ---------------------------------
            # Advance episode counter *once per completed episode*.
            self.agent.logger.increment_episode()
            ep_count = self.agent.logger.episode_count
            if ep_count % 10 == 0:
                print(
                    f"Episode {ep_count}/{self.n_episodes}  –  time {time.strftime('%H:%M:%S')}"
                )

            if ep_count > 0 and ep_count % self.save_interval == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_episode_{ep_count}.pt",
                )
                self._save_checkpoint(ckpt_path)

        # Final checkpoint
        self._save_checkpoint(os.path.join(self.checkpoint_dir, "final.pt"))

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path):
        torch.save({
            "agent_state": self.agent.get_state_dict(),
            "episode": self.agent.logger.episode_count,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }, path)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.agent.load_state_dict(ckpt["agent_state"])
        self.agent.logger.episode_count = ckpt["episode"]
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        np.random.set_state(ckpt["rng_state"]["numpy"])
        random.setstate(ckpt["rng_state"]["random"])

        print(f"Resumed from checkpoint {path}")
