import numpy
import torch
import sys
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from torch import nn
from agent_reinf import *
from sample import *


class NA1(nn.Module):
    def __init__(self, n_obs, n_action):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.4)


def make_env():
    return gym.make('Pendulum-v1')

# Create vectorized environment
num_envs=16
env = SyncVectorEnv([make_env for _ in range(num_envs)])  # 8 environments
num_batch = 10
n_episodes = 10000
dist_params = 2

discount = 0.95
na = NA1(3, 1 * dist_params)
na.to(torch.device('cuda'))
agent = Reinforce(na, sample_action_beta, num_envs=num_envs, discount=discount)
agent.policy.apply(init_normal)

for i in range(n_episodes):
    obs, info = env.reset()
    done = numpy.zeros(num_envs, dtype=bool)
    agent.episode_start()
    while not done.all():
        action = agent.get_action(obs, done)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        agent.update(obs, action, reward, done, next_obs)
        obs = next_obs
    if i % 10 == 0:
        print('iteration ' + str(i))
    sys.stdout.flush()
