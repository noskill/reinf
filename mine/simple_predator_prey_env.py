"""
Simple predator-prey environment: one predator chases one prey in 2D continuous space,
with energy depletion and capture termination.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimplePredatorPreyEnv(gym.Env):
    """Simple continuous 2D predator-prey environment with energy constraint."""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.size = 1.0
        self.predator_speed = 0.05
        self.prey_speed = 0.05
        self.capture_radius = 0.05
        low_obs = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([1.0, 1.0, np.sqrt(2.0), 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.predator_pos = np.random.rand(2).astype(np.float32)
        self.prey_pos = np.random.rand(2).astype(np.float32)
        self.energy = 1.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        norm = np.linalg.norm(a)
        if norm > 1e-6:
            move = a / norm * self.predator_speed
        else:
            move = np.zeros(2, dtype=np.float32)
        self.predator_pos = np.clip(self.predator_pos + move, 0.0, self.size)
        self.energy = max(0.0, self.energy - 0.02)
        rand_dir = np.random.rand(2) * 2.0 - 1.0
        norm = np.linalg.norm(rand_dir)
        move_prey = (rand_dir / norm * self.prey_speed) if norm > 1e-6 else np.zeros(2, dtype=np.float32)
        self.prey_pos = np.clip(self.prey_pos + move_prey, 0.0, self.size)
        obs = self._get_obs()
        dist = np.linalg.norm(self.prey_pos - self.predator_pos)
        done = bool(dist < self.capture_radius or self.energy <= 0.0)
        reward = 0.0
        return obs, reward, done, False, {}

    def _get_obs(self):
        vec = self.prey_pos - self.predator_pos
        dist = np.linalg.norm(vec)
        return np.array([vec[0], vec[1], dist, self.energy], dtype=np.float32)


def make_vec_predator_prey_env(num_envs: int):
    """Create a vectorized set of SimplePredatorPreyEnv instances."""
    from gymnasium.vector import SyncVectorEnv

    def make_env():
        return SimplePredatorPreyEnv()

    return SyncVectorEnv([make_env for _ in range(num_envs)])