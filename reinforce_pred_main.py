import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from reinforce import ReinforcePred
from sample import discrete_sampler, sample_action_normal
from log import Logger # assume a simple Logger that defines log_scalar() and increment_episode()


class PredictorNetwork(nn.Module):
    def __init__ (self, input_dim, output_dim, hidden_dim=64):
        """
        input_dim = state_dim + action_dim,
        output_dim = state_dim (the predicted next state)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_action, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(n_obs, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_action)
    
    def forward(self, x):
        # Ensure that x is on the correct device by using layer weights’ device
        x = F.relu(self.layer1(x.to(self.layer1.weight.device)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class RLTrainerPred:
    def __init__(
        self,
        env_name,
        num_envs=8,
        n_episodes=2000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.0001,
        predictor_lr=0.001,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=148234324,
        experiment_dir="runs/reinforce_pred"
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.n_episodes = n_episodes
        self.device = device
        self.seed = seed
        
        # Initialize environment; here we use SyncVectorEnv
        self.test_env = self._make_env()
        self.env = SyncVectorEnv([self._make_env for _ in range(num_envs)])
        
        # Get observation and action dimensions.
        self.obs_dim = self.test_env.observation_space.shape[0]
        self.action_dim = self._get_action_dim(self.test_env.action_space)
        
        # Decide on sampler and distribution parameters.
        self.dist_params = 1
        self.sampler = discrete_sampler
        if env_name == 'Pendulum-v1':
            self.dist_params = 2
            self.sampler = lambda policy, state: sample_action_normal(
                policy, state, a_min=-2.0, a_max=2.0
            )
        
        # Create the policy network.
        self.policy = PolicyNetwork(
            self.obs_dim,
            self.action_dim * self.dist_params,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.logger = Logger(experiment_dir)
        
        # Build the predictor network.
        # Its input is the concatenation of state and action.
        self.predictor = PredictorNetwork(
            input_dim=self.obs_dim + self.action_dim,  # action_dim (not *dist_params) for continuous output.
            output_dim=self.obs_dim,
            hidden_dim=64
        ).to(device)
        
        # Instantiate the new agent that uses both policy and predictor.
        self.agent = ReinforcePred(
            self.policy,
            self.sampler,
            self.predictor,
            predictor_lr=predictor_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            policy_lr=policy_lr,
            logger=self.logger,
        )
    
    def _make_env(self):
        return gym.make(self.env_name)
    
    @staticmethod
    def _get_action_dim(action_space):
        from gymnasium.spaces import Discrete, Box
        if isinstance(action_space, Discrete):
            return action_space.n
        elif isinstance(action_space, Box):
            return action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
    
    @staticmethod
    def _init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def train(self):
        # Set seeds for reproducibility.
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize the policy network weights.
        self.policy.apply(self._init_normal)
        
        for i in range(self.n_episodes):
            # Reset the vectorized environment. The reset also produces an info dict.
            obs, info = self.env.reset(seed=random.randint(0, 1000))
            done = np.zeros(len(obs), dtype=bool)
            self.agent.episode_start()  # Reset agent’s episodes.
            
            while not done.all():
                action = self.agent.get_action(obs, done)
                # For Pendulum, reshape action as needed.
                if self.env_name == "Pendulum-v1":
                    action = action.reshape(-1, 1)
                # Step the environment.
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
                print("Iteration", i)
            sys.stdout.flush()


###############################################################################
# Main
###############################################################################
def main():
    # For the predictor experiment a continuous-action environment is preferable.
    # Uncomment one of the following:
    # env_name = 'CartPole-v1'  # (Discrete action; be cautious with predictor inputs.)
    env_name = 'Pendulum-v1'    # Continuous; recommended for predictor.
    
    trainer = RLTrainerPred(
        env_name=env_name,
        num_envs=8,
        n_episodes=2000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.0001,
        predictor_lr=0.001,
    )
    trainer.train()

if __name__ == '__main__':
    main()
