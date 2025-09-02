"""
Two-Agent Empowerment Toy Environment (alternating-turn version)

This minimal Gymnasium environment exactly follows the timing and interface
described in *mine/readme_comm.txt*.

State  : one-hot(last symbol received) concatenated with agent-id bit.
Action : discrete symbol a ∈ {0 … K-1} to be sent to the partner.

The environment itself provides **no external reward** – the intrinsic
empowerment reward is computed in the training script by an inverse model.

The episode terminates after `max_steps` physical steps (i.e. individual
symbols).  Because the agents act in strict alternation a `max_steps` of 200
corresponds to 100 full «dialogue turns» per agent.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CommEnv(gym.Env):
    """Alternating-turn, symbol-passing toy environment for two agents."""

    metadata = {"render.modes": []}

    def __init__(self, K: int = 4, max_steps: int = 200, seed: int | None = None):
        super().__init__()

        assert K >= 2, "Alphabet size K must be at least 2."

        self.K = K
        self.max_steps = max_steps

        # Observation: one-hot(K) + agent_id scalar → length K + 1 vector in {0,1}
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(K + 1,),
            dtype=np.float32,
        )

        # Action: discrete symbol to send
        self.action_space: spaces.Discrete = spaces.Discrete(K)

        self.np_random = None  # will be set in reset/seed

        # Internal state
        self._turn: int = 0  # 0 or 1 – which agent is acting *this* step
        self._message: int = 0  # last message received ( symbol ∈ {0..K-1} )
        self._step_id: int = 0

        if seed is not None:
            self.seed(seed)

    # ---------------------------------------------------------------------
    # gymnasium API
    # ---------------------------------------------------------------------

    def seed(self, seed: int | None = None):
        """Set RNG seed (gymnasium compatibility)."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)

        self._turn = 0
        # Start with a random symbol so that the first observation already
        # contains information.
        self._message = int(self.np_random.integers(self.K))  # 0 … K-1
        self._step_id = 0

        obs = self._make_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Sanity-check
        if isinstance(action, np.ndarray):
            action = int(action.squeeze())  # convert from vectorised envs
        action = int(action)
        assert 0 <= action < self.K, "Action outside alphabet!"

        # Environment simply writes the symbol to the medium so that the partner
        # will receive it next step.
        self._message = action

        # Switch active agent (strict alternation)
        self._turn = 1 - self._turn

        self._step_id += 1

        done = self._step_id >= self.max_steps
        truncated = False  # No time-limit truncation separate from termination

        obs = self._make_obs()
        reward = 0.0  # no extrinsic reward
        info = {}
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_obs(self):
        """Return observation for *current* active agent."""
        one_hot = np.zeros(self.K, dtype=np.float32)
        one_hot[self._message] = 1.0
        agent_bit = np.array([float(self._turn)], dtype=np.float32)
        return np.concatenate([one_hot, agent_bit])


if __name__ == "__main__":
    # Quick sanity check
    env = CommEnv()
    obs, _ = env.reset()
    print("Initial obs", obs)
    for _ in range(5):
        a = env.action_space.sample()
        print("action", a)
        obs, r, d, _, _ = env.step(a)
        print("obs", obs, "reward", r)
        if d:
            break
