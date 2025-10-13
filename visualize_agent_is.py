#!/usr/bin/env python3
"""
Run / visualize a trained on-policy agent (REINFORCE / VPG / PPO / PPOD variants)
inside IsaacLab.

This script mirrors the CLI of ``reinforce_main_is.py`` but **does not train** —
it simply loads a saved checkpoint and lets the policy act in the environment so
that you can watch the behaviour in Isaac Sim (or record video).

Usage example
-------------
```bash
python visualize_agent_is.py \
    --task Isaac-Franka-Stack-v0 \
    --checkpoint logs/runs/reinforce_2025-07-16_07-24-29/checkpoints/checkpoint_episode_1000.pt
```
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime

import torch

# IsaacLab imports – AppLauncher must be created **before** any other Isaac
# modules are imported.
from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI – keep it very similar to the training script so that checkpoints created
# with ``reinforce_main_is.py`` can be re-used directly.
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Visualise a trained agent in IsaacLab.")

# visualisation / env options --------------------------------------------------
parser.add_argument("--video", action="store_true", default=False, help="Record video during the rollout.")
parser.add_argument("--video-length", type=int, default=400, help="Video length in simulation steps.")
parser.add_argument("--video-interval", type=int, default=2000, help="Interval between videos when --video is set.")
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run the sim in real-time.")

# task / checkpoint ------------------------------------------------------------
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Isaac-Franka-Stack-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint produced by reinforce_main_is.py")

# misc ------------------------------------------------------------------------
parser.add_argument("--algorithm", type=str, choices=["ppo", "reinforce", "vpg", "ppod", "ppodr", "ppod_novel"], default="reinforce", help="Algorithm that was used during training (needed to rebuild the agent architecture).")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (for speed – viewer shows the first one).")
parser.add_argument("--seed", type=int, default=None, help="RNG seed (defaults to random).")
parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to play. 0 = infinite loop.")

# hyper-parameters (only required so that ``create_agent`` finds the attributes)
parser.add_argument("--policy-lr", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--value-lr", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--disc-lr", type=float, default=None, help=argparse.SUPPRESS)
# diayn specific (placeholder so attribute exists)
parser.add_argument("--embedding-dim", type=int, default=8, help=argparse.SUPPRESS)
parser.add_argument("--continious-skills", action="store_true", default=False, help=argparse.SUPPRESS)
parser.add_argument("--skill-dim", type=int, default=8, help=argparse.SUPPRESS)

# let AppLauncher extend with its own flags (device, headless, etc.) ------------
AppLauncher.add_app_launcher_args(parser)

# we need to split args because hydra will parse its own later ------------------
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras when we intend to record video -------------------------
if args_cli.video:
    args_cli.enable_cameras = True

# clean up argv so that hydra only sees its own flags --------------------------
sys.argv = [sys.argv[0]] + hydra_args

# -----------------------------------------------------------------------------
# Launch Omniverse / IsaacSim *before* importing the rest of IsaacLab
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # noqa: F841 – keep handle alive


import grab
import envs

# -----------------------------------------------------------------------------
# Regular python imports that rely on IsaacSim being initialised
# -----------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from on_policy_train import setup_env  # noqa: E402
from agent_util import create_agent  # noqa: E402
from log import Logger  # noqa: E402


def load_agent(args, env_cfg, env) -> "Agent":  # noqa: F821 – forward reference
    """Re-create the agent architecture and load weights from checkpoint."""

    # create dummy logger so that Agent class requirements are satisfied
    logger = Logger(os.path.join("logs", "visualize"))

    # reconstruct agent (networks, sampler, etc.)
    agent = create_agent(args, env_cfg, env, logger)

    # load checkpoint
    chk = torch.load(args.checkpoint, map_location=env_cfg.sim.device)
    agent.load_state_dict(chk["agent_state"])

    # switch networks to eval mode (where applicable)
    if hasattr(agent, "policy") and hasattr(agent.policy, "eval"):
        agent.policy.eval()
    if hasattr(agent, "value") and hasattr(agent.value, "eval"):
        agent.value.eval()
    return agent


# -----------------------------------------------------------------------------
# Main (wrapped into hydra_task_config so that we get proper env cfg)
# -----------------------------------------------------------------------------


@hydra_task_config(args_cli.task, "")
def main(env_cfg):
    # RNG seed -----------------------------------------------------------------
    if args_cli.seed is None:
        args_cli.seed = random.randint(0, 10_000)
    env_cfg.seed = args_cli.seed

    # overwrite num envs / device from CLI ------------------------------------
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # directory where we will store optional videos ---------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "play", f"{args_cli.algorithm}_{timestamp}"))

    # build gym environment ----------------------------------------------------
    env = setup_env(env_cfg, args_cli, log_dir)

    # attempt to let physics keep up with wall-clock --------------------------
    if args_cli.real_time and hasattr(env.unwrapped, "set_real_time"):  # type: ignore[attr-defined]
        try:
            env.unwrapped.set_real_time(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    # build & load agent -------------------------------------------------------
    agent = load_agent(args_cli, env_cfg, env)

    # ---------------------------------------------------------------------
    # Rollout loop – very small and tidy
    # ---------------------------------------------------------------------
    num_episodes_target = args_cli.num_episodes
    episode_idx = 0

    obs = env.reset()
    device = torch.device(env_cfg.sim.device)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    while num_episodes_target == 0 or episode_idx < num_episodes_target:
        # query policy for next action
        action = agent.get_action(obs, done)
        if action.ndim == 1:
            action = action.unsqueeze(1)

        obs, _, done, info = env.step(action)

        # when all envs are done, start new episode -------------------------
        if done.all():
            episode_idx += 1
            obs = env.reset()
            done[:] = False

    env.close()


# -----------------------------------------------------------------------------
# Entry-point – run main() and shut down the simulator afterwards
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
    # close the omniverse app explicitly so we return to prompt
    simulation_app.close()
