import numpy
import copy
import torch
import torch.optim as optim
from torch.distributions import *
import torch.nn as nn
import torch.nn.functional as F
from ppo import PPOBase
from pool import EpisodesPoolMixin


class SkillDiscriminator(nn.Module):
    """
    Discriminator network that maps states to skill vectors.
    
    In DIAYN with continuous skills, this network tries to predict the continuous
    skill vector z that generated a particular state s.
    """
    def __init__(self, state_dim, skill_dim, hidden_dims=[256, 256], continuous=True):
        """
        Args:
            state_dim (int): Dimension of the state space
            skill_dim (int): Dimension of the skill space
            hidden_dims (list): Dimensions of hidden layers
            continuous (bool): Whether skills are continuous or discrete
        """
        super(SkillDiscriminator, self).__init__()
        
        self.continuous = continuous
        self.skill_dim = skill_dim
        
        # Build the network
        layers = []
        prev_dim = state_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Network backbone
        self.network = nn.Sequential(*layers)
        
        # Output layer depends on whether skills are continuous or discrete
        if continuous:
            # For continuous skills: predict the skill vector directly
            self.output_layer = nn.Linear(prev_dim, skill_dim)
        else:
            # For discrete skills: output logits for each skill category
            self.output_layer = nn.Linear(prev_dim, skill_dim)
    
    def forward(self, state):
        """
        Forward pass through the discriminator.
        
        Args:
            state (torch.Tensor): State observation [batch_size x state_dim]
            
        Returns:
            torch.Tensor: Predicted skill vectors or logits
        """
        features = self.network(state)
        output = self.output_layer(features)
        if not self.continuous:
            output = output.softmax(dim=1)
        return output
    
    def compute_mi_loss(self, states, skills):
        """
        Compute the loss for maximizing mutual information.
        
        For continuous skills, we use MSE loss to predict the exact skill vector.
        For discrete skills, we use cross-entropy loss.
        
        Args:
            states (torch.Tensor): Batch of states [batch_size x state_dim]
            skills (torch.Tensor): Batch of skills 
                - For continuous: [batch_size x skill_dim]
                - For discrete: [batch_size] (indices)
            
        Returns:
            torch.Tensor: Loss (to be minimized)
        """
        predictions = self.forward(states)
        
        if self.continuous:
            # For continuous skills: use MSE loss
            loss = F.mse_loss(predictions, skills)
        else:
            # For discrete skills: use cross-entropy loss
            loss = F.cross_entropy(predictions, skills)
            
        return loss
    
    def predict_skill(self, state):
        """
        Predict the skill that generated this state.
        
        Args:
            state (torch.Tensor): State observation
            
        Returns:
            torch.Tensor: Predicted skill vector or index
        """
        with torch.no_grad():
            output = self.forward(state)
            
            if not self.continuous:
                # For discrete skills: return the most likely skill index
                output = torch.argmax(output, dim=-1)
                
        return output


class PPODPool(EpisodesPoolMixin):

    def add_transition(self, env_idx, state, action, log_prob, entropy):
        self.episodes[env_idx].append((state, action, log_prob, entropy, self._skills[env_idx]))


class PPOD(PPOBase, PPODPool):
    """ Diversity is all you need implementation"""

    def __init__(self, policy, value, sampler, obs_dim, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, num_learning_epochs=4, clip_param=None, n_skills=8, embedding_dim=8, **kwargs):
        self.n_skills = n_skills
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.n_skills, embedding_dim).to(device)
        self.embedding_old = copy.deepcopy(self.embedding).to(device)
        self.discriminator = SkillDiscriminator(obs_dim, n_skills, continuous=False).to(device)
        self.discriminator_lr = value_lr
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, **kwargs)

        logits = torch.tensor([1 / self.n_skills for i in range(self.n_skills)])
        logits = torch.stack([logits for _ in range(self.num_envs)])
        self.skill_logits = logits.to(device)
        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'skill', 'rewards']

    def create_optimizers(self):
        super().create_optimizers()
        self.optimizer_policy = optim.Adam(list(self.policy.parameters()) + 
                                           list(self.embedding.parameters()), lr=self.policy_lr)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), self.discriminator_lr)

    def episode_start(self):
        self.reset_episodes()
        self.active_envs = numpy.ones(self.num_envs, dtype=bool)
        self._skills = Categorical(self.skill_logits).sample()

    def process_states(self, state, done):
        active_mask = ~done
        if not any(active_mask):
            return np.zeros((self.num_envs,) + self.policy.action_shape)

        active_states = state[active_mask]
        active_skills = self._skills[active_mask]
        if not isinstance(active_states, torch.Tensor):
            active_states = torch.FloatTensor(active_states)
        emb = self.embedding_old(active_skills)
        active_states = torch.cat([active_states, emb], dim=1)
        return active_states

    def learn_from_episodes(self, episodes, num_minibatches=4):
        # Extract per-episode tensors/lists
        data_dict = self._extract_episode_data(episodes)
        states_list = data_dict['states']
        log_probs_list = data_dict['log_probs']
        actions_list = data_dict['actions']
        rewards_list = data_dict['rewards']
        entropy_list = data_dict['entropy']
        skill_list = data_dict['skill']
        if not states_list:
            return

        # Prepare batches: compute discounted returns, and then aggregate states, returns, etc.
        states_batch, returns_batch, log_probs_batch, actions_batch, entropy_batch = self._prepare_batches(
            states_list, log_probs_list, rewards_list, actions_list, entropy_list
        )
        skill_batch = torch.cat(skill_list, dim=0).to(self.device)
        
        states_without_skill = states_batch[:, :-self.embedding_dim]
        self.train_discriminator(states_without_skill.detach(), skill_batch.detach())
        with torch.no_grad():
            skill_estimate = self.discriminator(states_without_skill)
            row_indices = torch.arange(skill_estimate.size(0)).to(skill_batch.device)
            skill_pred = skill_estimate[row_indices, skill_batch]

        
        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch + skill_pred)
        # Log statistics
        self._log_training_stats(actions_batch)
        self.train_value(normalized_returns, states_batch)

        # Sync policies
        self.policy.load_state_dict(self.policy_old.state_dict())
        self.embedding.load_state_dict(self.embedding_old.state_dict())
        # Loop through epochs and call separated method
        for i in range(self.num_learning_epochs):
            self._learn_epoch(
                states_without_skill, log_probs_batch, normalized_returns,
                actions_batch, entropy_batch, num_minibatches, skill_batch
            )
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.embedding_old.load_state_dict(self.embedding.state_dict())

    def _learn_epoch(self, states_without_skill, log_probs_batch, normalized_returns,
                    actions_batch, entropy_batch, num_minibatches, skills):
        dataset = torch.utils.data.TensorDataset(
            states_without_skill, log_probs_batch, normalized_returns, actions_batch, entropy_batch, skills
        )

        batch_size = len(states_without_skill) // num_minibatches
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for (states_minibatch, log_probs_minibatch, returns_minibatch,
            actions_minibatch, entropy_minibatch, skills_minibatch) in data_loader:

            if len(states_minibatch) < max(batch_size // 2, 2):  # safer minimum size check
                continue

            # Policy Update: compute advantages from current value estimates
            with torch.no_grad():
                emb = self.embedding(skills_minibatch)
                state_emb = torch.cat([states_minibatch, emb], dim=1)
                updated_values = self.value(state_emb).squeeze(-1)

            advantages = returns_minibatch - updated_values
            advantage_std = advantages.std()

            if torch.isnan(advantage_std).any():
                import pdb; pdb.set_trace()

            self.logger.log_scalar("Raw advantage mean:", advantages.mean().item())
            self.logger.log_scalar("Raw advantage std:", advantage_std.item())

            # Ensure proper entropy shape
            if len(entropy_minibatch.shape) == 2 and entropy_minibatch.shape[1] == 1:
                entropy_minibatch = entropy_minibatch.flatten()

            # Call policy training step
            self.train_policy(
                log_probs_minibatch, advantages, entropy_minibatch, state_emb, actions_minibatch
            )

    def train_discriminator(self, states_batch, skills):
        train_epochs = 2
        mini_batch_size = 128

        for epoch in range(train_epochs):
            indices = numpy.random.permutation(len(states_batch))
            for start in range(0, len(states_batch), mini_batch_size):
                self.optimizer_discriminator.zero_grad()
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                loss = self.discriminator.compute_mi_loss(states_batch[batch_idx], skills[batch_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
                self.optimizer_discriminator.step()

        self.logger.log_scalar(f"desc loss",  loss)
