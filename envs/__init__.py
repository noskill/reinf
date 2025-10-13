import gymnasium as gym
from gymnasium.envs.registration import register
from .franka_stack import CustomFrankaStackEnv
from .frank_env_cfg import FrankaCubeStackWithVelocityEnvCfg, DeterministicFrankaCubeStackEnvCfg


register(
    id="Isaac-Franka-CubeStack-Custom-v0",
    entry_point="envs.franka_stack:CustomFrankaStackEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeStackWithVelocityEnvCfg,
    },
)


register(
    id="Isaac-Franka-CubeStack-Custom-v0-det",
    entry_point="envs.franka_stack:CustomFrankaStackEnv",
    kwargs={
        "env_cfg_entry_point": DeterministicFrankaCubeStackEnvCfg,
    },
)

register(
    id="Isaac-Franka-CubeStack-Custom-v0-det-no-rew",
    entry_point="envs.franka_stack:CustomFrankaStackEnv",
    kwargs={
        "env_cfg_entry_point": DeterministicFrankaCubeStackEnvCfg,
        "reward_on": False,
    },
)
