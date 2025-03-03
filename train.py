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
parser = argparse.ArgumentParser(description="Train an agent with IsaacLab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (e.g. Isaac-Cartpole-v0, Pendulum-v1, etc.).")
parser.add_argument("--seed", type=int, default=54234, help="Seed used for environment randomization.")
parser.add_argument(
    "--algorithm", 
    type=str, 
    choices=["ppo", "reinforce", "vpg", "ppomi"],
    default="ppo",
    help="Algorithm to use for training"
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint from")
parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N episodes")

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
from typing import Union
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
from config.agents import TrainingCfg, PPOAlgorithmCfg, ReinforceAlgorithmCfg, CartpolePPOCfg, CartpoleReinforceCfg, SamplerCfg, VPGAlgorithmCfg, CartpoleVGPCfg
from omegaconf import OmegaConf
from ppo import PPO, PPOMI
from vpg import VPG
from on_policy_train import setup_env, OnPolicyTrainer


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, activation=None):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=None):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Here we expect x to be a torch tensor.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



def create_networks(obs_dim: int, action_dim: int, config: TrainingCfg, device):
    policy = PolicyNetwork(
        input_dim=obs_dim,
        output_dim=action_dim,
        hidden_dim=config.policy.hidden_dims[0],
        activation=config.policy.activation,
    ).to(device)
    
    value = None
    if config.value is not None:
        value = ValueNetwork(
            input_dim=obs_dim,
            hidden_dim=config.value.hidden_dims[0],
            activation=config.value.activation,
        ).to(device)
    
    return policy, value


def create_sampler(sampler_cfg: SamplerCfg):
    if sampler_cfg.type == "normal":
        return NormalActionSampler(
            a_min=-2,
            a_max=2
            # a_min=sampler_cfg.action_min,
            # a_max=sampler_cfg.action_max
        )
    elif sampler_cfg.type == "discrete":
        return DiscreteActionSampler()
    else:
        raise ValueError(f"Unknown sampler type: {sampler_cfg.type}")


def create_agent(
    algorithm_cfg: Union[PPOAlgorithmCfg, ReinforceAlgorithmCfg],
    policy,
    value,
    env,
    device,
    sampler_cfg: SamplerCfg,
    logger
):
    sampler = create_sampler(sampler_cfg)
    common_kwargs = {
        "policy": policy,
        "num_envs": env.num_envs,
        "policy_lr": algorithm_cfg.policy_lr,
        "discount": algorithm_cfg.discount,
        "device": device,
        "sampler": sampler,
        "logger": logger
    }
    
    if type(algorithm_cfg) == PPOAlgorithmCfg:
        return PPO(
            value=value,
            clip_param=algorithm_cfg.clip_param,
            entropy_coef=algorithm_cfg.entropy_coef,
            num_learning_epochs=algorithm_cfg.num_learning_epochs,
            **common_kwargs
        )
    elif type(algorithm_cfg) == ReinforceAlgorithmCfg:
        return Reinforce(
            entropy_coef=algorithm_cfg.entropy_coef,
            **common_kwargs
        )
    elif type(algorithm_cfg) == VPGAlgorithmCfg:
        return VPG(
            value=value,
            entropy_coef=algorithm_cfg.entropy_coef,
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm type: {type(algorithm_cfg)}")



from omegaconf import OmegaConf
from dataclasses import asdict


def merge_config_overrides(training_cfg, hydra_args):
    """
    Merges overrides from hydra_cfg into the given training_cfg object.
    """
    override_dict = {}
    for override in hydra_args:
        # Look for lines like +algorithm.policy_lr=0.0001 or algorithm.discount=0.95
        # We strip leading '+' in case user types +algorithm.policy_lr=0.0001
        override_clean = override.lstrip('+')
        if '=' in override_clean:
            key, value_str = override_clean.split('=', 1)
            override_dict[key] = value_str

    if "algorithm.policy_lr" in override_dict:
        training_cfg.algorithm.policy_lr = float(override_dict["algorithm.policy_lr"])
    return training_cfg

@hydra_task_config(args_cli.task, "")  # Use empty string for agent_cfg_entry_point
def main(
    env_cfg: Union[ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg],
    agent_cfg=None  # This will be None initially
):

    # Load config based on algorithm choice
    if args_cli.algorithm == "ppo":
        training_cfg = CartpolePPOCfg()
    elif args_cli.algorithm == 'vpg':
        training_cfg = CartpoleVGPCfg()
    elif args_cli.algorithm == "ppomi":
        training_cfg = CartpolePPOMICfg()
    else:
        training_cfg = CartpoleReinforceCfg()

    training_cfg = merge_config_overrides(training_cfg, hydra_args)
    # Create experiment directory
    experiment_dir = training_cfg.logging.get_experiment_dir(
        algorithm_type=training_cfg.algorithm.type,
        experiment_name=training_cfg.experiment_name
    )
    
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    # Update training config with checkpoint settings
    training_cfg.checkpoint = args_cli.checkpoint
    training_cfg.save_interval = args_cli.save_interval
    training_cfg.checkpoint_dir = checkpoint_dir  # Always use checkpoints subdirectory
    
    print(f"[INFO] Logging experiment in directory: {experiment_dir}")
    
    # Setup environment with experiment directory
    env = setup_env(env_cfg, args_cli, experiment_dir)
    
    # Create logger
    logger = Logger(experiment_dir)
    
    # Create networks and agent
    policy, value = create_networks(
        obs_dim=gym.spaces.flatdim(env.observation_space),
        action_dim=env.action_space.shape[0] * 2, # mu, sigma
        config=training_cfg,
        device=env_cfg.sim.device,
    )
    agent = create_agent(
        algorithm_cfg=training_cfg.algorithm,
        policy=policy,
        value=value,
        env=env,
        device=env_cfg.sim.device,
        sampler_cfg=training_cfg.sampler,
        logger=logger
    )
    
    trainer = OnPolicyTrainer(
        env=env, 
        agent=agent, 
        n_episodes=training_cfg.algorithm.n_episodes,
        checkpoint_dir=checkpoint_dir,
        save_interval=args_cli.save_interval,
        checkpoint=args_cli.checkpoint,
        seed=args_cli.seed
    )
    trainer.train()
    
    env.close()
# 6. Run main and close the simulator.
#
if __name__ == "__main__":
    main()
    simulation_app.close()
