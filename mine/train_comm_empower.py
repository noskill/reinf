"""
Train two agents in the communication empowerment task using the
EmpowermentPPO algorithm implemented in ``mine/empowerment_ppo.py``.

The script is intentionally compact – it merely wires together

* the **vectorised** communication environment (``VecCommEnv``),
* a minimal MLP policy and value network,
* the EmpowermentPPO agent, and
* the existing ``OnPolicyTrainer`` infrastructure.

Run from the project root via

    python -m mine.train_comm_empower  --episodes 20000

You may adjust hyper-parameters through CLI flags.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Make the repository root importable when this file is executed *as a script*
# (e.g. ``python mine/train_comm_empower.py``) instead of as a module
# (``python -m mine.train_comm_empower``).  This prevents the well-known
# "'mine' is not a package" error due to Python treating the *containing*
# directory as a plain script directory rather than a package.
# ---------------------------------------------------------------------------

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mine.vec_comm_env import VecCommEnv
from mine.empowerment_ppo import EmpowermentPPO
# Local lightweight trainer (no IsaacLab dependency)
from mine.comm_trainer import CommTrainer
from sample import DiscreteActionSampler
from log import Logger  # simple TensorBoard wrapper used elsewhere in repo


# ---------------------------------------------------------------------------
# Simple MLPs for policy and value
# ---------------------------------------------------------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),  # logits – no softmax
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Communication empowerment experiment")
    parser.add_argument("--K", type=int, default=4, help="Alphabet size")
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=str, default="logs/runs/comm_experiment")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume training")
    parser.add_argument("--entropy-coef", type=float, default=1.0,
                        help="Entropy coefficient to add to policy loss (encourages exploration, default 0)")
    parser.add_argument("--inverse-lr", type=float, default=3e-4,
                        help="Learning rate for the inverse model q_ψ (default 3e-4). Use smaller values (e.g. 1e-5) to stabilise empowerment reward if action entropy collapses.")
    args = parser.parse_args()

    # For this lightweight experiment we stick to CPU.  This avoids device
    # mismatch issues with tensors implicitly created on the CPU inside the
    # existing agent code.
    device = torch.device("cpu")

    # Environment -------------------------------------------------------
    env = VecCommEnv(num_envs=args.num_envs, K=args.K)
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.nvec[0]

    # Networks ----------------------------------------------------------
    policy_net = PolicyNet(obs_dim, num_actions).to(device)
    value_net = ValueNet(obs_dim).to(device)

    # Sampler -----------------------------------------------------------
    sampler = DiscreteActionSampler()

    # Logger ------------------------------------------------------------
    tb_dir = os.path.join(args.run_dir)
    logger = Logger(tb_dir)

    # Agent -------------------------------------------------------------
    agent = EmpowermentPPO(
        policy_net,
        value_net,
        sampler,
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_envs=args.num_envs,
        device=device,
        logger=logger,
        inverse_lr=args.inverse_lr,
        entropy_coef=args.entropy_coef,
    )

    # Trainer -----------------------------------------------------------
    trainer = CommTrainer(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        checkpoint_dir=args.run_dir,
        save_interval=1000,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )

    trainer.train()


if __name__ == "__main__":
    main()
