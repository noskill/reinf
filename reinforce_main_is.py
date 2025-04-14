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
parser.add_argument("--video-length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video-interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (e.g. Isaac-Cartpole-v0, Pendulum-v1, etc.).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for environment randomization.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint from")
parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N episodes")
parser.add_argument("--max-episode-s", type=int, default=None, help="set environment max episode length in seconds")
parser.add_argument("--algorithm", type=str, choices=["ppo", "reinforce", "vpg", "ppod", "ppodr"], default="reinforce", help="Algorithm to use for training.")
parser.add_argument("--experiment-dir", type=str, default=None, help="Directory to save files.")
parser.add_argument("--experiment-name", type=str, default=None, help="experiment name subdir.")
parser.add_argument("--embedding-dim", type=int, default=8, help="embedding dimentions for diayn implementation")
parser.add_argument("--continious-skills", action="store_true", default=False, help="whether should use continious distribution for skills")
parser.add_argument("--skill-dim", type=int, default=8, help="size of sample space of categorical skill distribution or size of skill vector for continious skills")
parser.set_defaults(flatten_obs=True)
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

from log import Logger
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg

from on_policy_train import setup_env, OnPolicyTrainer
from agent_util import create_agent


def save_args_to_log(args, log_dir):
    """Save command-line arguments to a log directory"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'args.json')
    
    # Convert args to dictionary and save to JSON
    args_dict = vars(args)
    with open(log_file, 'w') as f:
        import json
        json.dump(args_dict, f, indent=4)
    
    return log_file


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg=None):
    # Set random seed if not provided
    if args_cli.seed is None:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed
    if args_cli.max_episode_s is not None:
        env_cfg.episode_length_s = args_cli.max_episode_s

    # Override environment settings from CLI if provided
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Define logging directories
    if args_cli.experiment_name is None:
        experiment_name = f"{args_cli.algorithm}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        experiment_name = args_cli.experiment_name
    if args_cli.experiment_dir is None:
        experiment_dir = os.path.abspath(os.path.join("logs", "runs", experiment_name))
    else:
        experiment_dir = args_cli.experiment_dir
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    print(f"[INFO] Logging experiment in directory: {experiment_dir}")

    env = setup_env(env_cfg, args_cli, experiment_dir)

    # Create logger
    logger = Logger(experiment_dir)

    agent = create_agent(args_cli, env_cfg, env, logger)
    n_episodes = 1500
    
    save_args_to_log(args_cli, experiment_dir)
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
