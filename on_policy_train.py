import os
import time
import sys
from venv import VectorEnvWrapper
from typing import Union
import gymnasium as gym
import torch
import random
import numpy as np


def setup_env(env_cfg: Union['ManagerBasedRLEnvCfg', 'DirectRLEnvCfg', 'DirectMARLEnvCfg'], args_cli, experiment_dir):
    """
    Creates and configures the training environment.

    Args:
        env_cfg: Environment configuration from IsaacLab
        args_cli: Command line arguments

    Returns:
        Wrapped environment ready for training
    """
    # Override some environment settings from CLI if provided.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create the base environment using gym.make
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
        disable_env_checker=True  # This disables the PassiveEnvChecker
    )

    seed = int(args_cli.seed)
    base_env = env.unwrapped
    base_env.seed(seed)

    print('set seed to ' + str(seed))
    # Add video recording wrapper if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "train"),
            "step_trigger": lambda step: step == 0 or step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # Convert multi-agent to single-agent if needed
    # if isinstance(env.unwrapped, MultiAgentEnv):
    #     env = multi_agent_to_single_agent(env)

    # Add the vectorized environment wrapper
    env = VectorEnvWrapper(env)

    return env


class OnPolicyTrainer:
    def __init__(self, env, agent, n_episodes, checkpoint_dir, save_interval, checkpoint=None, seed=42):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.checkpoint = checkpoint
        self.seed = seed

    def train(self):
        # Set random seeds
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load checkpoint if specified
        start_episode = 0
        if self.checkpoint:
            print(f"Loading checkpoint from {self.checkpoint}")
            self.load_checkpoint(self.checkpoint)
            start_episode = self.agent.logger.episode_count

        for i in range(start_episode, start_episode + self.n_episodes):
            obs = self.env.reset()
            done = torch.zeros(self.env.num_envs, dtype=bool)
            self.agent.episode_start()
            step = 0
            info = None
            while not done.all():
                action = self.agent.get_action(obs, done)
                if action.ndim == 1:
                    action = action.unsqueeze(1)
                next_obs, reward, terminated, info = self.env.step(action)
                if torch.isnan(reward).any():
                    import pdb;pdb.set_trace()
                done = terminated
                # terminal_reward = -2 _np.exp(-(step/300)_*2) * done
                terminal_reward = 0
                changed = self.agent.update(obs, action, reward + terminal_reward, done, next_obs, info=info)
                if changed:
                    break
                obs = next_obs
            if info is not None:
                if "grasp_success_rate" in info:
                    self.agent.logger.log_scalar("grasp_success_rate", info["grasp_success_rate"])
                if "stack_success_rate" in info:
                    self.agent.logger.log_scalar("stack_success_rate", info["stack_success_rate"])
            self.agent.logger.increment_episode()

            # Save checkpoint periodically
            if i > 0 and i % self.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_episode_{i}.pt"
                )
                print(f"Saving checkpoint to {checkpoint_path}")
                self.save_checkpoint(checkpoint_path)

            if i % 10 == 0:
                print(f"Iteration {i}")

            sys.stdout.flush()

    def save_checkpoint(self, path):
        agent_state = self.agent.get_state_dict()
        checkpoint = {
            'agent_state': agent_state,
            'episode': self.agent.logger.episode_count,
            'seed': self.seed,
            'n_episodes': self.n_episodes,
            'checkpoint_dir': self.checkpoint_dir,
            'save_interval': self.save_interval,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state'])
        self.agent.logger.episode_count = checkpoint['episode']
        self.seed = checkpoint['seed']
        # Optionally restore other parameters if needed
        if 'n_episodes' in checkpoint:
            self.n_episodes = checkpoint['n_episodes']
        if 'checkpoint_dir' in checkpoint:
            self.checkpoint_dir = checkpoint['checkpoint_dir']
        if 'save_interval' in checkpoint:
            self.save_interval = checkpoint['save_interval']
