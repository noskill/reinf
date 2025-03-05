#!/usr/bin/env python3
"""
Simplified training script for REINFORCE, VPG, and PPO agents using IsaacLab.
"""

import os
import sys
import argparse

# IsaacLab imports
from isaaclab.app import AppLauncher

#
# 1. Parse CLI arguments and launch IsaacLab app.
#
parser = argparse.ArgumentParser(description="Train an agent with IsaacLab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (e.g. Isaac-Cartpole-v0, Pendulum-v1, etc.).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for environment randomization.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint from")
parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N episodes")
parser.add_argument("--algorithm", type=str, choices=["ppo", "reinforce", "vpg"], default="reinforce", help="Algorithm to use for training.")
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

# Import your agent components
from reinforce import Reinforce  # Your REINFORCE implementation
from vpg import VPG  # Your VPG implementation
from ppo import PPO  # Your PPO implementation
from sample import DiscreteActionSampler, NormalActionSampler  # Action samplers
from log import Logger  # Your logging utility
from agent import Agent  # Your BaseAgent class
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg

# Import the PolicyNetwork from reinforce_main.py
from reinforce_main import PolicyNetwork

# Import the Value network (named Value) from vpg_main.py
from vpg_main import Value

# Import setup_env from on_policy_train.py
from on_policy_train import setup_env, OnPolicyTrainer


# Utility functions for creating networks and agents

def create_networks(obs_dim: int, action_dim: int, hidden_dim=256, device='cpu'):
    # Use the imported PolicyNetwork
    policy = PolicyNetwork(
        n_obs=obs_dim,
        n_action=action_dim,
        hidden_dim=hidden_dim
    ).to(device)
        
    # Use the imported Value network (named Value)
    value = Value(
        n_obs=obs_dim,
        hidden_dim=hidden_dim
    ).to(device)
        
    return policy, value

def create_sampler(action_space):
    shape = action_space.shape
    if len(shape) != 1:
        raise RuntimeError("unexpected action space size " + str(shape))
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscreteActionSampler()
    elif isinstance(action_space, gym.spaces.Box):
        return NormalActionSampler(shape[0], a_min=-2, a_max=2)
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg=None):
    # Set random seed if not provided
    if args_cli.seed is None:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed

    # Override environment settings from CLI if provided
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Define logging directories
    experiment_name = f"{args_cli.algorithm}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_dir = os.path.abspath(os.path.join("logs", "runs", experiment_name))
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    print(f"[INFO] Logging experiment in directory: {experiment_dir}")

    env = setup_env(env_cfg, args_cli, experiment_dir)
    num_envs = env.num_envs

    # Determine observation and action dimensions
    obs_space = env.observation_space
    action_space = env.action_space
    obs_dim = gym.spaces.flatdim(obs_space)

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    else:
        action_dim = action_space.shape[0] * 2  # For mean and std in continuous actions

    # Create logger
    logger = Logger(experiment_dir)

    # Create networks using imported classes
    device = env_cfg.sim.device
    hidden_dim = 64  # You can adjust this value
    policy, value = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device,
    )

    # Create sampler
    sampler = create_sampler(action_space)

    # Define hyperparameters
    policy_lr = 0.0001
    value_lr = 0.001
    discount = 0.99
    entropy_coef = 0.01
    n_episodes = 5000

    # Create agent based on selected algorithm, using your existing Agent class
    if args_cli.algorithm == "ppo":
        num_learning_epochs = 4
        agent = PPO(
            policy=policy,
            value=value,
            num_envs=num_envs,
            policy_lr=policy_lr,
            value_lr=value_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            num_learning_epochs=num_learning_epochs,
            sampler=sampler,
            logger=logger
        )
    elif args_cli.algorithm == "vpg":
        value_loss_coef = 0.5
        value_clip = 0.2
        agent = VPG(
            policy=policy,
            value=value,
            num_envs=num_envs,
            policy_lr=policy_lr,
            value_lr=value_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            value_clip=value_clip,
            sampler=sampler,
            logger=logger
        )
    else:
        # Default to REINFORCE
        # For REINFORCE, the value network is not used
        # Adjust the action_dim accordingly
        if not isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.shape[0] * 2
        else:
            action_dim = action_space.n
        policy = PolicyNetwork(
            n_obs=obs_dim,
            n_action=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        agent = Reinforce(
            policy=policy,
            sampler=sampler,
            num_envs=num_envs,
            policy_lr=policy_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            logger=logger
        )

    # Create trainer
    trainer = OnPolicyTrainer(
        env=env, 
        agent=agent, 
        n_episodes=n_episodes,
        checkpoint_dir=checkpoint_dir,
        save_interval=args_cli.save_interval,
        checkpoint=args_cli.checkpoint,
        seed=args_cli.seed
    )
    trainer.train()

    # Close the environment
    env.close()

# 5. Run main() and close the simulation app
if __name__ == "__main__":
    main()
    simulation_app.close()
