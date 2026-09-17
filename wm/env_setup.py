"""
Helper to create vectorized environments
"""

import gymnasium as gym
import numpy as np

from maze.rl_env import MazeVecEnv
from maze.agimazeenv import AgiMazeVecEnv


class MazeTrainerEnvAdapter:
    """Adapter for MazeVecEnv to match OnPolicyTrainer expectations."""

    def __init__(self, env: MazeVecEnv):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.max_episode_length = int(env.max_steps)
        self.obs_dim = 7
        self.action_space = gym.spaces.Discrete(env.action_space_n)
        self.observation_space = gym.spaces.Dict(
            {
                "sensor": gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32),
                "heading_idx": gym.spaces.Box(low=0, high=3, shape=(), dtype=np.int64),
                "location": gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.int64),
                "step_count": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
            })

    @property
    def unwrapped(self):
        return self

    @property
    def max_dim(self):
        return self.env.max_dim

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()


def setup_environment(args):
    env_type = args.env_type
    if env_type == 'maze':
        base_env = MazeVecEnv(
            num_envs=args.num_envs,
            maze_path=args.maze_path,
            random_dim=args.random_dim,
            random_extra_openings=args.random_extra_openings,
            randomize_each_reset=args.randomize_each_reset,
            max_steps=args.max_steps,
            render=args.render,
            seed=args.seed,
            auto_reset=True,
            return_torch=True,
            device=args.device)
        env = MazeTrainerEnvAdapter(base_env)
    elif env_type == 'agimaze':
        base = AgiMazeVecEnv(
                num_envs=args.num_envs,
                extra_openings = 6,
                num_pit_pairs = 1,
                start_tile=(0, 0),
                max_steps=args.max_steps,
                seed=args.seed,
                auto_reset=True,
                return_torch=True,
                device=args.device)
        env = base
    
    return env
