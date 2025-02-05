import numpy as np
import torch
import random
import sys
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from torch import nn
from reinforce import *
from sample import *
from gymnasium.spaces import Discrete, Box
from log import Logger

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_action, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_action)

    def forward(self, x):
        x = F.relu(self.layer1(x.to(self.layer1.weight)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class RLTrainer:
    def __init__(
        self,
        env_name,
        num_envs=8,
        n_episodes=2000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.0001,
        device=torch.device('cuda'),
        seed=148234324,
        experiment_dir="runs/reinforce-entropy"
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.n_episodes = n_episodes
        self.device = device
        self.seed = seed
        
        # Initialize environment
        self.test_env = self._make_env()
        self.env = SyncVectorEnv([self._make_env for _ in range(num_envs)])
        
        # Setup environment parameters
        self.obs_dim = self.test_env.observation_space.shape[0]
        self.action_dim = self._get_action_dim(self.test_env.action_space)
        self.dist_params = 1
        self.sampler = discrete_sampler
        
        if env_name == 'Pendulum-v1':
            self.dist_params = 2
            self.sampler = lambda policy, state: sample_action_normal(
                policy, state, a_min=-2, a_max=2
            )

        # Initialize policy network and agent
        self.policy = PolicyNetwork(
            self.obs_dim, 
            self.action_dim * self.dist_params, 
            hidden_dim=hidden_dim
        ).to(device)
        
        self.logger = Logger(experiment_dir)
        
        self.agent = Reinforce(
            self.policy,
            self.sampler,
            num_envs=num_envs,
            discount=discount,
            device=device,
            policy_lr=policy_lr,
            logger=self.logger
        )

    def _make_env(self):
        return gym.make(self.env_name)

    @staticmethod
    def _get_action_dim(action_space):
        if isinstance(action_space, Discrete):
            return action_space.n
        elif isinstance(action_space, Box):
            return action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    @staticmethod
    def _init_normal(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def train(self):
        # Set seeds
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize policy weights
        self.policy.apply(self._init_normal)

        for i in range(self.n_episodes):
            obs, info = self.env.reset(seed=random.randint(0, 1000))
            done = np.zeros(len(obs), dtype=bool)
            self.agent.episode_start()

            while not done.all():
                action = self.agent.get_action(obs, done)
                if self.env_name == "Pendulum-v1":
                    action = action.reshape(-1, 1)

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action.cpu().numpy()
                )
                done = terminated | truncated
                
                changed = self.agent.update(obs, action, reward, done, next_obs)
                if changed:
                    break
                    
                obs = next_obs

            self.logger.increment_episode()
            if i % 10 == 0:
                print('iteration ' + str(i))
            sys.stdout.flush()


def main():
    name = 'CartPole-v1'
    name = 'Pendulum-v1'

    trainer = RLTrainer(
        env_name=name,
        num_envs=8,
        n_episodes=5000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.0001
    )
    trainer.train()


if __name__ == '__main__':
    main()
