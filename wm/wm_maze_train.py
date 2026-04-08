#!/usr/bin/env python3
"""Main entrypoint for joint WM+policy maze training."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from log import Logger
from reinforce import Reinforce
from sample import DiscreteActionSampler
from wm_joint_agent import JointWMAgent, WMActionHeadPolicy, create_maze_rnn_world_model


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
MAZE_DIR = THIS_DIR / "maze"

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(MAZE_DIR) not in sys.path:
    sys.path.insert(0, str(MAZE_DIR))

from on_policy_train import OnPolicyTrainer  # noqa: E402
from rl_env import MazeVecEnv  # noqa: E402


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
            }
        )

    @property
    def unwrapped(self):
        return self

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()


def build_agent(
    *,
    action_dim: int,
    num_envs: int,
    policy_lr: float,
    discount: float,
    entropy_coef: float,
    device: torch.device,
    logger: Logger,
    policy_module: torch.nn.Module,
):
    policy = policy_module.to(device)
    sampler = DiscreteActionSampler()
    return Reinforce(
        policy=policy,
        sampler=sampler,
        num_envs=num_envs,
        policy_lr=policy_lr,
        discount=discount,
        device=device,
        entropy_coef=entropy_coef,
        logger=logger,
    )


def default_experiment_dir(experiment_name: str | None) -> str:
    if experiment_name is None:
        experiment_name = f"maze_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    return os.path.abspath(os.path.join("logs", "runs", experiment_name))


def save_run_args(args, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as file_obj:
        json.dump(vars(args), file_obj, indent=2)

    cmd_line = shlex.join([sys.executable] + sys.argv)
    with open(os.path.join(run_dir, "command.sh"), "w", encoding="utf-8") as file_obj:
        file_obj.write("#!/usr/bin/env bash\n")
        file_obj.write("set -euo pipefail\n")
        file_obj.write(f"{cmd_line}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train joint WM+policy in MazeVecEnv.")
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=100)

    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--experiment-dir", type=str, default=None)

    parser.add_argument("--maze-path", type=str, default=None)
    parser.add_argument("--random-dim", type=int, default=10)
    parser.add_argument("--random-extra-openings", type=int, default=0)
    parser.add_argument("--randomize-each-reset", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--intrinsic-reward-scale", type=float, default=1.0)
    parser.add_argument("--env-reward-scale", type=float, default=1.0)
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--wm-weight-decay", type=float, default=0.01)
    parser.add_argument("--wm-updates-per-policy", type=int, default=1)
    parser.add_argument("--wm-replay-capacity", type=int, default=2048)
    parser.add_argument("--wm-train-episodes", type=int, default=64)
    parser.add_argument("--wm-hidden-size", type=int, default=176)
    parser.add_argument("--wm-layers", type=int, default=3)
    parser.add_argument("--wm-intermediate", type=int, default=704)
    parser.add_argument("--wm-obs-latent-dim", type=int, default=64)
    parser.add_argument("--wm-probe-hidden-dim", type=int, default=128)
    parser.add_argument("--wm-probe-layers", type=int, default=2)
    parser.add_argument("--wm-contrastive-dim", type=int, default=64)
    parser.add_argument("--wm-contrastive-steps", type=int, default=1)
    parser.add_argument("--wm-contrastive-temp", type=float, default=0.1)
    parser.add_argument("--wm-contrastive-discount", type=float, default=0.75)
    parser.add_argument("--wm-contrastive-negatives", type=int, default=0)
    parser.add_argument("--wm-sensor-max-bin", type=int, default=64)
    parser.add_argument("--wm-sensor-weight", type=float, default=1.0)
    parser.add_argument("--wm-loc-weight", type=float, default=0.0)
    parser.add_argument("--wm-head-weight", type=float, default=0.0)
    parser.add_argument("--wm-turn-weight", type=float, default=0.0)
    parser.add_argument("--wm-step-weight", type=float, default=0.0)
    parser.add_argument("--wm-sensor-sigma", type=float, default=1.0)
    parser.add_argument("--wm-pos-sigma", type=float, default=1.0)
    parser.add_argument("--wm-heading-smoothing", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_dir = args.experiment_dir or default_experiment_dir(args.experiment_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_run_args(args, run_dir)

    if args.render and args.num_envs != 1:
        raise ValueError("--render requires --num-envs 1")

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
        device=args.device,
    )
    env = MazeTrainerEnvAdapter(base_env)

    logger = Logger(run_dir)
    if args.wm_contrastive_dim <= 0:
        raise ValueError("--wm-contrastive-dim must be > 0")
    if args.wm_contrastive_steps < 1:
        raise ValueError("--wm-contrastive-steps must be >= 1")
    if args.wm_sensor_max_bin < 1:
        raise ValueError("--wm-sensor-max-bin must be >= 1")

    maze_dim = int(base_env._mazes[0].dim)
    turn_bins = len({int(turn) for turn, _ in base_env.action_table})
    step_bins = len({int(step) for _, step in base_env.action_table})
    wm_model = create_maze_rnn_world_model(
        device=torch.device(args.device),
        maze_dim=maze_dim,
        turn_bins=turn_bins,
        step_bins=step_bins,
        hidden_size=args.wm_hidden_size,
        layers=args.wm_layers,
        intermediate=args.wm_intermediate,
        obs_latent_dim=args.wm_obs_latent_dim,
        probe_hidden_dim=args.wm_probe_hidden_dim,
        probe_layers=args.wm_probe_layers,
        contrastive_dim=args.wm_contrastive_dim,
        contrastive_steps=args.wm_contrastive_steps,
        contrastive_temp=args.wm_contrastive_temp,
        contrastive_horizon_discount=args.wm_contrastive_discount,
        contrastive_negatives=args.wm_contrastive_negatives,
        sensor_weight=args.wm_sensor_weight,
        loc_weight=args.wm_loc_weight,
        head_weight=args.wm_head_weight,
        turn_weight=args.wm_turn_weight,
        step_weight=args.wm_step_weight,
        sensor_sigma=args.wm_sensor_sigma,
        pos_sigma=args.wm_pos_sigma,
        heading_smoothing=args.wm_heading_smoothing,
        sensor_max_bin=args.wm_sensor_max_bin,
    )
    policy_module = WMActionHeadPolicy(
        wm_model=wm_model,
        action_table=base_env.action_table,
        device=torch.device(args.device),
    )

    policy_agent = build_agent(
        action_dim=env.action_space.n,
        num_envs=env.num_envs,
        policy_lr=args.policy_lr,
        discount=args.discount,
        entropy_coef=args.entropy_coef,
        device=torch.device(args.device),
        logger=logger,
        policy_module=policy_module,
    )
    wm_optimizer = torch.optim.AdamW(
        [p for p in wm_model.parameters() if p.requires_grad],
        lr=args.wm_lr,
        weight_decay=args.wm_weight_decay,
    )
    agent = JointWMAgent(
        policy_agent=policy_agent,
        wm_model=wm_model,
        wm_optimizer=wm_optimizer,
        action_table=base_env.action_table,
        device=torch.device(args.device),
        intrinsic_reward_scale=args.intrinsic_reward_scale,
        env_reward_scale=args.env_reward_scale,
        wm_updates_per_policy=args.wm_updates_per_policy,
        wm_replay_capacity=args.wm_replay_capacity,
        wm_train_episodes=args.wm_train_episodes,
        sensor_max_bin=args.wm_sensor_max_bin,
        maze_dim=maze_dim,
    )

    trainer = OnPolicyTrainer(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
