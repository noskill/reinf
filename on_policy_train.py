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
    env = VectorEnvWrapper(env, return_dict_observations=True)

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
        self.max_episode_steps = getattr(self.env.unwrapped, "max_episode_length", None)
        self.early_termination_scale = float(getattr(agent, "early_termination_penalty_scale", 0.0))

    def compute_terminal_reward(self, terminated, info):
        if self.early_termination_scale <= 0:
            device = getattr(self.env, "device", torch.device("cpu"))
            if isinstance(terminated, torch.Tensor):
                return torch.zeros_like(terminated, dtype=torch.float32, device=terminated.device)
            return torch.zeros(terminated.shape, dtype=torch.float32, device=device)

        if not isinstance(terminated, torch.Tensor):
            terminated_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.env.device)
        else:
            terminated_tensor = terminated.to(self.env.device).bool()

        if self.max_episode_steps is None or info is None or "episode_length_buf" not in info:
            return torch.zeros_like(terminated_tensor, dtype=torch.float32)

        steps_value = info["episode_length_buf"]
        if not isinstance(steps_value, torch.Tensor):
            steps_tensor = torch.tensor(steps_value, dtype=torch.float32, device=self.env.device)
        else:
            steps_tensor = steps_value.to(self.env.device).float()

        remained_steps = (self.max_episode_steps - (steps_tensor + 1.0)).clamp_min(0.0)
        terminated_float = terminated_tensor.float()
        penalty = -self.early_termination_scale * (remained_steps / float(self.max_episode_steps))
        penalty = penalty * terminated_float

        if self.agent.logger is not None and terminated_tensor.any():
            early_mask = (remained_steps > 0.0) & terminated_tensor
            timeout_mask = (~early_mask) & terminated_tensor

            early_count = early_mask.float().sum().item()
            timeout_count = timeout_mask.float().sum().item()
            total_terminated = early_count + timeout_count

            if early_count > 0:
                self.agent.logger.log_scalar("early_termination_penalty_sum", penalty[early_mask].sum().item())
            self.agent.logger.log_scalar("early_termination_events", early_count)
            self.agent.logger.log_scalar("timeout_events", timeout_count)

            if timeout_count > 0:
                ratio = early_count / timeout_count
            else:
                ratio = 0.0
            self.agent.logger.log_scalar("early_to_timeout_ratio", ratio)

        return penalty

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
            episode_start = torch.ones(self.env.num_envs, dtype=bool)
            self.agent.episode_start()
            info = None
            while True:
                action = self.agent.get_action(obs, episode_start)
                if action.ndim == 1:
                    action = action.unsqueeze(1)
                next_obs, reward, terminated, info = self.env.step(action)
                if torch.isnan(reward).any():
                    import pdb;pdb.set_trace()
                episode_start = terminated
                terminal_reward = self.compute_terminal_reward(terminated, info)
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32, device=self.env.device)
                reward = reward + terminal_reward.to(reward.device)
                changed = self.agent.update(rewards=reward, dones=episode_start, info=info)
                if changed and info is not None and "cube_displacement" in info and self.agent.logger is not None:
                    displacement = info["cube_displacement"]
                    if isinstance(displacement, torch.Tensor):
                        values = displacement.detach().float().view(-1)
                        if values.numel() > 0:
                            mean = values.mean().item()
                            std = values.std(unbiased=False).item() if values.numel() > 1 else 0.0
                            self.agent.logger.log_scalar("cube_displacement_mean", mean)
                            self.agent.logger.log_scalar("cube_displacement_std", std)
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
