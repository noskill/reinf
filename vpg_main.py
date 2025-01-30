import numpy
import torch
import random
import sys
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from torch import nn
from vpg import *
from sample import *
from gymnasium.spaces import Discrete, Box


class NA1(nn.Module):
    def __init__(self, n_obs, n_action, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_action)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)


def get_action_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

def make_env():
#    return gym.make('Pendulum-v1')
    return gym.make('CartPole-v1')

test_env = make_env()
# Create vectorized environment
num_envs=8
env = SyncVectorEnv([make_env for _ in range(num_envs)])  # 8 environments
num_batch = 10
n_episodes = 300
dist_params = 1
obs_dim = test_env.observation_space.shape[0]
action_dim = get_action_dim(test_env.action_space)


discount = 0.99
device = torch.device('cuda')
na = NA1(obs_dim, action_dim * dist_params, hidden_dim=256).to(device)
value = Value(obs_dim).to(device)

sampler = lambda policy, state: sample_action_normal(policy, state, a_min=-2, a_max=2)
agent = VPG(na, value,
            discrete_sampler,
            num_envs=num_envs, 
            discount=discount, 
            device=device, 
            policy_lr=0.001, 
            value_lr=0.001)
agent.policy.apply(init_normal)
value.apply(init_normal)

seed = 148234324
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
for i in range(n_episodes):
    obs, info = env.reset()
    done = numpy.zeros(num_envs, dtype=bool)
    agent.episode_start()
    while not done.all():
        action = agent.get_action(obs, done)
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated | truncated
        changed = agent.update(obs, action, reward, done, next_obs)
        if changed:
            break
        obs = next_obs
    if i % 10 == 0:
        print('iteration ' + str(i))
    sys.stdout.flush()
