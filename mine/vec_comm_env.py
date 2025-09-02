"""
Vectorised variant of the two–agent communication environment defined in
``mine/comm_env.py``.

Instead of representing a *single* conversation, ``VecCommEnv`` holds
``num_envs`` completely independent copies internally and executes them in
lock-step.  This makes it compatible with RL agents that expect *batched*
observations/actions (i.e. parallel environments).

API – only the essentials used by our PPO implementation are provided:

    reset() -> obs(ndarray [N x (K+1)]), info(dict)
    step(action_ndarray [N]) -> obs, reward, terminated, truncated, info

where ``terminated`` and ``truncated`` are boolean arrays of length ``N``.

Note:  This is *not* a Gymnasium ``VectorEnv`` subclass – it is a plain
``gym.Env`` that returns batched data.  This design choice keeps the code
base simple because the existing agent code already assumes that shape.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VecCommEnv(gym.Env):
    """Batch of independent two-agent communication games."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        num_envs: int = 8,
        *,
        K: int = 4,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__()

        assert K >= 2, "Alphabet size K must be at least 2."
        assert num_envs >= 1, "Need at least one sub-environment"

        self.K = K
        self._n = num_envs
        self.max_steps = max_steps

        # Observation for *one* env is length K+1 – batch it along 1st dim.
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(K + 1,),
            dtype=np.float32,
        )

        # Actions: each env expects a single discrete symbol.
        self.action_space: spaces.MultiDiscrete = spaces.MultiDiscrete([
            K for _ in range(self._n)
        ])

        # RNG -------------------------------------------------------------
        self.np_random = None
        self.seed(seed)

        # Internal state arrays -----------------------------------------
        self._turn = np.zeros(self._n, dtype=np.int8)      # whose turn (0/1)
        self._message = np.zeros(self._n, dtype=np.int8)   # last heard symbol
        self._step_id = np.zeros(self._n, dtype=np.int32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)

        # Init each sub-env independently
        self._turn.fill(0)
        self._message = self.np_random.integers(self.K, size=self._n, dtype=np.int8)
        self._step_id.fill(0)

        obs = self._make_obs()
        info = {}
        return obs, info

    def step(self, action):
        """Advance *all* sub-envs by one physical symbol."""
        action = np.asarray(action, dtype=np.int64).reshape(self._n)

        # Sanity check (ignore actions for already-done envs)
        assert (action >= 0).all() and (action < self.K).all(), "Action out of range"

        # Apply actions only to environments that are still running
        alive = self._step_id < self.max_steps
        self._message[alive] = action[alive]

        # Toggle agent turn and advance step counter for alive envs
        self._turn[alive] = 1 - self._turn[alive]
        self._step_id[alive] += 1

        # Prepare outputs ------------------------------------------------
        obs = self._make_obs()

        reward = np.zeros(self._n, dtype=np.float32)  # extrinsic reward zero

        terminated = (self._step_id >= self.max_steps)
        truncated = np.zeros(self._n, dtype=bool)  # no separate truncation

        info = {}
        return obs, reward, terminated, truncated, info

    # Convenience attributes -------------------------------------------
    @property
    def num_envs(self):
        """Number of parallel sub-environments."""
        return self._n

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_obs(self):
        """Batch observation for the *currently active* agent in each env."""
        obs = np.zeros((self._n, self.K + 1), dtype=np.float32)

        # One-hot symbols ------------------------------------------------
        rows = np.arange(self._n)
        obs[rows, self._message] = 1.0

        # Agent-id bit ---------------------------------------------------
        obs[:, self.K] = self._turn.astype(np.float32)

        return obs


if __name__ == "__main__":
    env = VecCommEnv(num_envs=3)
    obs, _ = env.reset()
    print("obs shape", obs.shape)
    for _ in range(4):
        act = env.action_space.sample()
        print("act", act)
        obs, r, term, trunc, _ = env.step(act)
        print("obs", obs, "term", term)
