# reinforce_pred_main.py
from reinforce_main import RLTrainer
from reinforce import ReinforceWithPrediction
from torch import nn
import torch
import torch.nn.functional as F


class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class PredictionRLTrainer(RLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize predictor network
        self.predictor = StatePredictor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=self.policy.layer1.out_features
        ).to(self.device)
        
        # Initialize agent with prediction
        self.agent = ReinforceWithPrediction(
            self.policy,
            self.predictor,
            self.sampler,
            num_envs=self.num_envs,
            discount=self.discount,
            device=self.device,
            policy_lr=self.policy_lr,
            predictor_lr=self.policy_lr,
            logger=self.logger
        )

def main():
    name = 'Pendulum-v1'
    trainer = PredictionRLTrainer(
        env_name=name,
        num_envs=8,
        n_episodes=5000,
        hidden_dim=256,
        discount=0.99,
        policy_lr=0.0001,
        experiment_dir="runs/reinforce-prediction"
    )
    trainer.agent.log_hparams()
    trainer.train()

if __name__ == '__main__':
    main()
