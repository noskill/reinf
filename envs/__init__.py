import gymnasium as gym
from gymnasium.envs.registration import register
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg
from .franka_stack import CustomFrankaStackEnv
from .frank_env_cfg import DeterministicFrankaCubeStackEnvCfg


register(
    id="Isaac-Franka-CubeStack-Custom-v0",
    entry_point="envs.franka_stack:CustomFrankaStackEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeStackEnvCfg,
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
