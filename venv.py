import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv, VecEnvObs
import gymnasium as gym


class VectorEnvWrapper:
    def __init__(self, env, return_dict_observations=False):
        self.env = env
        self.device = getattr(env.unwrapped, "device", "cpu")
        self.return_dict_observations = return_dict_observations

    # Forward the observation_space property.
    @property
    def observation_space(self) -> gym.spaces.Space:
        policy_space = self.unwrapped.single_observation_space["policy"]
        if self.return_dict_observations:
            return policy_space
        return gym.spaces.flatten_space(policy_space)

    # Forward the action_space property.
    # bound for 100 like stable-baselines3
    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)
        return action_space

    def reset(self):
        obs_dict, info = self.env.reset()
        return self.extract_obs(obs_dict)

    def extract_obs(self, obs_dict):
        obs = obs_dict["policy"]
        if self.return_dict_observations:
            return obs
        if hasattr(obs, 'keys'):
            keys = sorted(obs.keys())
            return torch.cat([obs[key] for key in keys], dim=1)
        return obs

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        obs = self.extract_obs(obs)
        done = terminated | truncated
        return obs, rew, done, info

    def close(self):
        self.env.close()

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.
        """
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs
