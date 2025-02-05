import numpy as np
import torch
import random
import sys
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from torch import nn
from vpg import *
from sample import *
from gymnasium.spaces import Discrete, Box
from log import Logger
from reinforce_main import RLTrainer, PolicyNetwork


class Value(nn.Module):
    def __init__(self, n_obs, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class VPGTrainer(RLTrainer):
    def __init__(
        self,
        env_name,
        num_envs=8,
        n_episodes=2000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.001,
        value_lr=0.001,
        device=torch.device('cuda'),
        seed=148234324,
        experiment_dir="runs/vpg"
    ):
        super().__init__(
            env_name=env_name,
            num_envs=num_envs,
            n_episodes=n_episodes,
            hidden_dim=hidden_dim,
            discount=discount,
            policy_lr=policy_lr,
            device=device,
            seed=seed, 
            experiment_dir=experiment_dir
        )
        
        # Initialize value network
        self.value_net = Value(self.obs_dim, hidden_dim=hidden_dim).to(device)
        
        # Initialize VPG agent instead of Reinforce
        self.agent = VPG(
            self.policy,
            self.value_net,
            self.sampler,
            num_envs=num_envs,
            discount=discount,
            device=device,
            policy_lr=policy_lr,
            value_lr=value_lr,
            logger=self.logger
        )
        
        # Initialize value network weights
        self.value_net.apply(self._init_normal)


def main():
    name = 'Pendulum-v1'  # or 'CartPole-v1'
    trainer = VPGTrainer(
        env_name=name,
        num_envs=8,
        n_episodes=5000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.001,
        value_lr=0.001,
        experiment_dir='runs/vpg-entropy'
    )
    trainer.train()


if __name__ == '__main__':
    main()
