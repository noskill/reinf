#!/usr/bin/env python3
"""
Example conversion of a gym-based REINFORCE training script to the IsaacLab style.
Before running, please launch IsaacSim (or your IsaacLab-supported simulator).
"""

import os
import sys
import argparse

# IsaacLab imports
from isaaclab.app import AppLauncher

#
# 1. Parse CLI arguments and launch IsaacLab app.
#
parser = argparse.ArgumentParser(description="Train a REINFORCE agent with IsaacLab.")
parser.add_argument("--agent_cfg_path", type=str, default=None, help="Path to the REINFORCE agent configuration file (e.g. config/agents/reinforce_cfg.yaml).",)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (e.g. Isaac-Cartpole-v0, Pendulum-v1, etc.).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for environment randomization.")

# Let AppLauncher add its own CLI args.
AppLauncher.add_app_launcher_args(parser)

# Parse known args so that later Hydra reads its own args correctly.
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
# Clear sys.argv for Hydra (only pass hydra-specific arguments)
sys.argv = [sys.argv[0]] + hydra_args

# Launch IsaacLab (and thereby IsaacSim/Omniverse)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# Import your agent components (assumed to be defined in your modules)
from reinforce import Reinforce  # Your REINFORCE implementation
from sample import DiscreteActionSampler, NormalActionSampler  # Action samplers
from log import Logger  # Your logging utility
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
from omegaconf import OmegaConf

#
# 2. Define your Policy network.
#
class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_action, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_action)

    def forward(self, x):
        # Here we expect x to be a torch tensor.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

#
# 3. Create a simple vectorized environment wrapper.
#
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv, VecEnvObs
class ReinforceVecEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.device = getattr(env.unwrapped, "device", "cpu")

    # Forward the observation_space property.
    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = self.unwrapped.single_observation_space["policy"]
        return observation_space

    # Forward the action_space property.
    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)
        return action_space

    def reset(self):
        obs_dict, info = self.env.reset()
        obs = obs_dict["policy"]
        return obs

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        assert len(obs.keys()) == 1
        obs = obs["policy"]
        done = terminated | truncated
        return obs, rew, done, info

    def close(self):
        self.env.close()

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs
#
# 4. Define a trainer class that uses REINFORCE.
#
class ReinforceTrainer:
    def __init__(self, env, num_envs=8, n_episodes=2000, hidden_dim=256,
                 discount=0.99, policy_lr=0.0001, device=torch.device("cuda"),
                 seed=148234324, experiment_dir="runs/reinforce-entropy"):
        self.env = env
        self.num_envs = num_envs
        self.n_episodes = n_episodes
        self.device = device
        self.seed = seed
        self.discount = discount
        self.policy_lr = policy_lr

        # Use the provided environment as "test_env" to inspect its spaces.
        self.test_env = env
        obs_space = self.test_env.observation_space
        act_space = self.test_env.action_space
        if hasattr(obs_space, "shape"):
            self.obs_dim = obs_space.shape[0]
        else:
            self.obs_dim = gym.spaces.flatdim(obs_space)
        # Determine action dimension.
        if isinstance(act_space, gym.spaces.Discrete):
            self.action_dim = act_space.n
        elif isinstance(act_space, gym.spaces.Box):
            self.action_dim = act_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(act_space)}")

        # Sampler parameters.
        self.dist_params = 2
        # self.sampler = DiscreteActionSampler()
        # if args_cli.task == "Pendulum-v1":
        #     self.dist_params = 2
            # self.sampler = NormalActionSampler(a_min=-2, a_max=2)
        self.sampler = NormalActionSampler(a_min=-2, a_max=2)
        
        # Create the policy network.
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim * self.dist_params, hidden_dim=hidden_dim).to(device)
        self.policy.apply(self._init_normal)

        self.logger = Logger(experiment_dir)
        self.agent = Reinforce(self.policy, self.sampler, num_envs=num_envs,
                               discount=discount, device=device, policy_lr=policy_lr,
                               logger=self.logger)

    @staticmethod
    def _init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def train(self):
        # Set seeds for reproducibility.
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        for i in range(self.n_episodes):
            obs = self.env.reset()
            done = torch.zeros(len(obs), dtype=bool)
            self.agent.episode_start()

            while not done.all():
                action = self.agent.get_action(obs, done)
                if action.ndim == 1:
                    action = action.unsqueeze(1)
                next_obs, reward, terminated, info = self.env.step(action)
                done = terminated
                changed = self.agent.update(obs, action, reward, done, next_obs)
                if changed:
                    print('changed')
                    break
                obs = next_obs

            self.logger.increment_episode()
            if i % 10 == 0:
                print(f"Iteration {i}")
            sys.stdout.flush()

#
# 5. Main entry point using Hydra configuration.
#
@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg=dict()):
    """
    Main training function.
    The hydra_task_config decorator will load your environment configuration (env_cfg)
    and the agent configuration (agent_cfg) from the specified entry points.
    """
    
    if agent_cfg is None or not bool(agent_cfg):
        if hasattr(args_cli, "agent_cfg_path") and args_cli.agent_cfg_path is not None:
            print(f"[INFO] Loading agent config from command line argument: {args_cli.agent_cfg_path}")
            agent_cfg = OmegaConf.load(args_cli.agent_cfg_path)
            agent_cfg = OmegaConf.to_container(agent_cfg, resolve=True)
        else:
            raise ValueError(
            "No agent configuration was provided. "
            "Either register an agent config via the gym registry or pass --agent_cfg_path."
            )
    # Override some environment settings from CLI if provided.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Choose a seed if not provided via CLI.
    if args_cli.seed is None:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed

    # Define logging directories.
    log_root_path = os.path.abspath(os.path.join("logs", "reinforce", agent_cfg.get("experiment_name", "default")))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "run_name" in agent_cfg:
        log_dir += f"_{agent_cfg['run_name']}"
    experiment_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {experiment_dir}")

    # Create the IsaacLab environment (using the provided task configuration).
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    # (Optional) If using a multi-agent environment and your agent expects a single-agent instance,
    # you can convert it:
    # if isinstance(env.unwrapped, SomeMultiAgentEnv):
    #     env = multi_agent_to_single_agent(env)

    # Wrap video recording if requested.
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap the environment for REINFORCE.
    env = ReinforceVecEnvWrapper(env)

    # Instantiate our trainer.
    trainer = ReinforceTrainer(env,
                               num_envs=env.num_envs,
                               n_episodes=agent_cfg.get("n_episodes", 5000),
                               hidden_dim=agent_cfg.get("hidden_dim", 256),
                               discount=agent_cfg.get("discount", 0.99),
                               policy_lr=agent_cfg.get("policy_lr", 0.0001),
                               device=env_cfg.sim.device,
                               seed=args_cli.seed,
                               experiment_dir=experiment_dir)
    trainer.agent.log_hparams()
    trainer.train()

    # Close the environment.
    env.close()

#
# 6. Run main and close the simulator.
#
if __name__ == "__main__":
    main()
    simulation_app.close()
