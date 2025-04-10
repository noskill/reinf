import numpy
import copy
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal, Uniform
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

    def forward(self, state, softmax=True):
        """
        Forward pass through the discriminator.

        Args:
            state (torch.Tensor): State observation [batch_size x state_dim]

        Returns:
            torch.Tensor: Predicted skill vectors or logits
        """
        features = self.network(state)
        output = self.output_layer(features)
        if not self.continuous and softmax:
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
        predictions = self.forward(states, softmax=False)
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


class SkillEmbedding(nn.Module):
    """
    Unified skill embedding module that handles both categorical and continuous skills.
    """
    def __init__(self, skill_dim, embedding_dim, continuous, device=torch.device('cpu')):
        super().__init__()
        self.skill_dim = skill_dim
        self.embedding_dim = embedding_dim
        self.continuous = continuous
        self.device = device

        if continuous:
            # For continuous skills: use a linear projection
            self.model = nn.Linear(skill_dim, embedding_dim).to(device)
            # Skill sampling distribution (Uniform between -1 and 1)
            #self.skill_dist = Uniform(low=-1.0, high=1.0)
            self.skill_dist = Normal(loc=torch.zeros(self.skill_dim).to(device), scale=torch.ones(self.skill_dim).to(device))
        else:
            # For categorical skills: use an embedding lookup
            self.model = nn.Embedding(skill_dim, embedding_dim).to(device)
            logits = torch.tensor([1 / skill_dim for i in range(skill_dim)])
            self.logits = logits.to(device)
    
    def forward(self, skills):
        """Project skills to embedding space"""
        if not self.continuous:
            skills = skills.to(int).squeeze()
        return self.model(skills)

    def sample_skills(self, batch_size=1):
        """Sample skills based on the embedding type"""
        if self.continuous:
            # Sample continuous skill vectors
            return self.skill_dist.sample((batch_size,))
        else:
            # Sample categorical skill indices
            logits = torch.stack([self.logits for _ in range(batch_size)])
            return Categorical(logits).sample()


class SkillAugmentedPolicy(nn.Module):
    """
    A wrapper class that combines a policy network with skill embedding.
    This simplifies the interface and avoids computational graph issues.
    """
    def __init__(self, policy, skill_embedding, skill_dim):
        super().__init__()
        self.policy = policy
        self.skill_embedding = skill_embedding
        self.skill_dim = skill_dim
        
    def forward(self, states):
        """
        Forward pass that embeds skills and concatenates with states
        before passing to the policy.
        
        Args:
            states: State observations [batch_size x state_dim]
            skills: Skill vectors or indices [batch_size x skill_dim] or [batch_size]
            
        Returns:
            Policy output for the augmented states
        """
        augmented_states = self.augment_states(states)
        # Pass through the policy
        return self.policy(augmented_states)
    
    def augment_states(self, states):
        skills = states[:, -self.skill_dim:]
        st = states[:, :-self.skill_dim]

        # Embed the skills
        skill_emb = self.skill_embedding(skills)

        # Concatenate states with skill embeddings
        augmented_states = torch.cat([st, skill_emb], dim=1)
        return augmented_states

    def get_action_shape(self):
        """Return the action shape from the underlying policy"""
        return self.policy.action_shape


class PPOD(PPOBase, PPODPool):
    """Unified DIAYN PPO supporting both Continuous and Categorical skills."""

    def __init__(self, policy, value, sampler, obs_dim,
                 skill_dim=8, embedding_dim=8, continious=False,
                 policy_lr=0.0001, num_envs=8, discount=0.99,
                 device=torch.device('cpu'), logger=None, num_learning_epochs=4, discriminator=None, 
                 discriminator_fields=None, **kwargs):
        self.skill_dim = skill_dim
        self.embedding_dim = embedding_dim
        self.continious = continious
        self.device = device
        self.skill_state_len = skill_dim
        if not self.continious:
            self.skill_state_len = 1 # discrete distribution - just 1 number
        self.discriminator = discriminator
        # Create skill embedding modules
        self.skill_embedding = SkillEmbedding(
            skill_dim, embedding_dim, continuous=continious, device=device)

        # Create the augmented policy
        policy = SkillAugmentedPolicy(policy, self.skill_embedding, self.skill_state_len)

        self.discriminator_lr = policy_lr / 10
        self.desc_fields = discriminator_fields
        # Initialize superclasses (PPOBase, Pool)
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                         num_envs=num_envs, discount=discount, device=device, logger=logger,
                         num_learning_epochs=num_learning_epochs, **kwargs)

        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'skill', 'rewards']

    def create_optimizers(self):
        super().create_optimizers()
        self.optimizer_policy = optim.Adam(
            list(self.policy.parameters()),
            lr=self.policy_lr
        )
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=0.001
        )

    def episode_start(self):
        self.reset_episodes()
        self._skills = self.skill_embedding.sample_skills(self.num_envs)

    def process_states(self, state, done):
        active_mask = ~done
        if not any(active_mask):
            return torch.zeros((self.num_envs,) + self.policy.action_shape, device=self.device)

        active_states = state[active_mask]
        active_skills = self._skills[active_mask]

        if not isinstance(active_states, torch.Tensor):
            active_states = torch.FloatTensor(active_states).to(self.device)
        if active_skills.dim() == 1:
            active_skills = active_skills.unsqueeze(dim=1)
        active_states = torch.cat([active_states, active_skills], dim=1)
        return active_states
    
    def compute_reward_from_discriminator(self, states, skill, discriminator=None):
        """Compute rewards using the specified discriminator (or default to self.discriminator)"""

        discriminator = discriminator or self.discriminator
        desc_input = self.get_descriminator_input(states)

        skill_estimate = discriminator(desc_input)
        if self.continious:
            dist = Normal(skill_estimate, torch.ones_like(skill_estimate))
            prior_dist = Normal(torch.zeros_like(skill), torch.ones_like(skill))

            # Compute full log-likelihood differences (summed over dimensions)
            discriminator_log_prob = dist.log_prob(skill).sum(dim=-1)
            prior_log_prob = prior_dist.log_prob(skill).sum(dim=-1)

            # DIAYN reward: log q(z|s) - log p(z)
            reward = discriminator_log_prob - prior_log_prob
        else:
            assert skill_estimate.dim() == 2
            batch_size, num_skills = skill_estimate.shape
            
            # Apply log_softmax to get log probabilities
            log_probs = F.log_softmax(skill_estimate, dim=-1)
 
            # Skill should be a 1D tensor of indices [batch_size]
            assert skill.dim() == 1
            assert skill.shape[0] == batch_size
            
            # Convert to long and add dimension for gather
            skill_indices = skill.long().unsqueeze(-1)

            # Gather the log probabilities for the specific skills
            discriminator_log_prob = log_probs.gather(dim=-1, index=skill_indices).squeeze(-1)
            assert discriminator_log_prob.dim() == 1
            assert discriminator_log_prob.shape[0] == batch_size

            # Prior log probability (uniform categorical prior)
            prior_log_prob = -torch.log(torch.tensor(num_skills, device=skill_estimate.device, dtype=torch.float))
            prior_log_prob = prior_log_prob.expand(batch_size)
            # This should be a vector of length batch_size
            assert prior_log_prob.shape == (batch_size,)
            
            # DIAYN reward: log q(z|s) - log p(z)
            reward = discriminator_log_prob - prior_log_prob
            assert reward.shape == (batch_size,)
        return reward.detach()

    def evaluate_discriminator(self, discriminator, states, skills):
        desc_input = self.get_descriminator_input(states)
        """Evaluate discriminator performance (MSE for continuous, accuracy for categorical)"""
        with torch.no_grad():
            if self.continious:
                # For continuous skills: compute MSE (lower is better)
                skill_estimate = discriminator(desc_input)
                performance = F.mse_loss(skill_estimate, skills).item()
            else:
                # For categorical skills: compute accuracy (higher is better)
                skill_estimate = discriminator(desc_input)
                pred = torch.argmax(skill_estimate, dim=1)
                performance = (pred == skills).float().mean().item()
            return performance

    def learn_from_episodes(self, episodes, num_minibatches=4):
        data_dict = self._extract_episode_data(episodes)
        states, log_probs, actions, rewards, entropy, skills = \
            data_dict['states'], data_dict['log_probs'], data_dict['actions'], \
            data_dict['rewards'], data_dict['entropy'], data_dict['skill']

        if not states:
            return

        states_batch, returns, log_probs, actions, entropy = self._prepare_batches(
            states, log_probs, rewards, actions, entropy)
        skills_batch = torch.cat(skills, dim=0).to(self.device)
        states_no_skill = states_batch[:, :-self.skill_state_len]

        # Evaluate discriminator performance
        performance = self.evaluate_discriminator(self.discriminator, states_batch, skills_batch)
        if self.continious:
            self.logger.log_scalar("discriminator_mse", performance)
        else:
            self.logger.log_scalar("discriminator_accuracy", performance)

        # Compute discriminator-based rewards
        disc_rewards = []
        idx = 0
        for episode_states, episode_skill in zip(states, skills):
            reward_desc = self.compute_reward_from_discriminator(episode_states, episode_skill)
            total_reward = reward_desc + rewards[idx].to(reward_desc.device)
            disc_rewards.append(total_reward)
            idx += 1

        new_returns = torch.cat([self._compute_discounted_returns(r) for r in disc_rewards], dim=0).to(self.device)
        new_returns = self._normalize_returns(new_returns)
        augmented_states = self.policy_old.augment_states(states_batch)
        # Training
        self.train_value(new_returns, augmented_states, value_epochs=4)

        # Sync networks before training
        self.policy.load_state_dict(self.policy_old.state_dict())

        # Train policy
        for _ in range(self.num_learning_epochs):
            self._learn_epoch(states_batch, log_probs, new_returns,
                              actions, entropy, num_minibatches, skills_batch)

        # Train discriminator
        self.train_discriminator(states_batch, skills_batch)

        # Sync networks after training
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _learn_epoch(self, states_batch, log_probs_batch, normalized_returns,
                     actions_batch, entropy_batch, num_minibatches, skills):
        dataset = torch.utils.data.TensorDataset(
            states_batch, log_probs_batch, normalized_returns,
            actions_batch, entropy_batch, skills)

        batch_size = len(states_batch) // num_minibatches
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        for states_mb, log_probs_mb, returns_mb, actions_mb, entropy_mb, skills_mb in data_loader:
            if len(states_mb) < max(batch_size // 2, 2):  # Skip too small batches
                continue
            state_emb = self.policy.augment_states(states_mb)

            # Compute advantages
            with torch.no_grad():
                updated_values = self.value(state_emb).squeeze(-1)

            advantages = returns_mb - updated_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            # Train policy
            self.train_policy(log_probs_mb, advantages, entropy_mb, states_mb, actions_mb)

    def get_descriminator_input(self, states):
        if self.desc_fields is None:
            result = self.state_extractor.remove(states, ["actions", "skill"])
        else:
            result = self.state_extractor.extract(states, self.desc_fields)
        return result

    def train_discriminator(self, states, skills):
        """Train the discriminator to predict skills from states"""
        train_epochs = 1
        mini_batch_size = 128
        desc_input = self.get_descriminator_input(states)
        for epoch in range(train_epochs):
            indices = torch.randperm(len(desc_input))
            for start in range(0, len(desc_input), mini_batch_size):
                self.optimizer_discriminator.zero_grad()
                end = min(start + mini_batch_size, len(desc_input))
                batch_idx = indices[start:end]
                loss = self.discriminator.compute_mi_loss(desc_input[batch_idx], skills[batch_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
                self.optimizer_discriminator.step()

        self.logger.log_scalar("discriminator_loss", loss.item())

    # checkpointing support
    def get_state_dict(self):
        sd = super().get_state_dict()
        sd['skill_embedding'] = self.skill_embedding.state_dict()
        sd['discriminator'] = self.discriminator.state_dict()
        sd['optimizer_discriminator'] = self.optimizer_discriminator.state_dict()
        return sd

    def load_state_dict(self, sd, ignore_missing=False):
        super().load_state_dict(sd, ignore_missing)
        try:
            self.skill_embedding.load_state_dict(sd['skill_embedding'])
            self.discriminator.load_state_dict(sd['discriminator'])
            self.optimizer_discriminator.load_state_dict(sd['optimizer_discriminator'])
        except KeyError as e:
            if not ignore_missing:
                raise e


class PPODRunning(PPOD):
    """ DIAYN implementation with running average discriminator stabilization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create target discriminator (for stable reward computation)
        self.target_discriminator = copy.deepcopy(self.discriminator)

        # Polyak averaging coefficient (τ)
        self.tau = 0.1  # Slow update rate
        
        # Synchronization settings
        self.sync_interval = 100
        self.sync_threshold = 0.05
        self.reset_threshold = 0.2
        self.previous_performance = None
        self.iteration_count = 0

    def learn_from_episodes(self, episodes, num_minibatches=4):
        data_dict = self._extract_episode_data(episodes)
        states, log_probs, actions, rewards, entropy, skills = \
            data_dict['states'], data_dict['log_probs'], data_dict['actions'], \
            data_dict['rewards'], data_dict['entropy'], data_dict['skill']

        if not states:
            return
            
        states_batch, returns, log_probs, actions, entropy = self._prepare_batches(
            states, log_probs, rewards, actions, entropy)
        skills_batch = torch.cat(skills, dim=0).to(self.device)

        # Evaluate main and target discriminator performance
        target_perf = self.evaluate_discriminator(self.target_discriminator, states_batch, skills_batch)
        main_perf = self.evaluate_discriminator(self.discriminator, states_batch, skills_batch)
        performance_drop = None
        # Log performance metrics
        if self.continious:
            self.logger.log_scalar("discriminator_mse", main_perf)
            self.logger.log_scalar("target_discriminator_mse", target_perf)
            # For MSE, lower is better
            is_target_better = target_perf < main_perf
            performance_drop = main_perf - (self.previous_performance or main_perf)
        else:
            self.logger.log_scalar("discriminator_accuracy", main_perf)
            self.logger.log_scalar("target_discriminator_accuracy", target_perf)
            # For accuracy, higher is better
            is_target_better = target_perf > main_perf
            performance_drop = (self.previous_performance or main_perf) - main_perf
        
        if performance_drop is not None:
            self.logger.log_scalar("performance_drop", performance_drop)
            self.logger.log_scalar("is_target_better", is_target_better * 1.0)
        # Reset logic - if performance dropped significantly and target is better
        if self.previous_performance is not None and performance_drop > self.reset_threshold and is_target_better:
            self.logger.log_scalar("discriminator_reset", 1.0)
            print(f"Discriminator performance dropped by {performance_drop:.4f}, resetting to target")
            self.sync_discriminator_to_target()
            
            # Re-evaluate after reset
            main_perf = self.evaluate_discriminator(self.discriminator, states_batch, skills_batch)
        
        self.previous_performance = main_perf

        # Compute discriminator-based rewards using TARGET discriminator
        disc_rewards = []
        idx = 0
        for episode_states, episode_skill in zip(states, skills):
            # Use the parent method but specify the target discriminator
            reward_desc = self.compute_reward_from_discriminator(
                episode_states, episode_skill, discriminator=self.target_discriminator
            )
            total_reward = reward_desc + rewards[idx].to(reward_desc.device)
            disc_rewards.append(total_reward)
            idx += 1

        new_returns = torch.cat([self._compute_discounted_returns(r) for r in disc_rewards], dim=0).to(self.device)
        new_returns = self._normalize_returns(new_returns)
        augmented_states = self.policy_old.augment_states(states_batch)
        # Training
        self.train_value(new_returns, augmented_states, value_epochs=4)

        # Sync networks before training
        self.policy.load_state_dict(self.policy_old.state_dict())
        
        if entropy.dim() == 2:
            entropy = entropy.mean(dim=1)
        # Train policy
        for _ in range(self.num_learning_epochs):
            self._learn_epoch(states_batch, log_probs, new_returns,
                              actions, entropy, num_minibatches, skills_batch)

        # Train discriminator
        self.train_discriminator(states_batch, skills_batch)
        
        # Update target discriminator via Polyak averaging
        self.update_target_discriminator()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update_target_discriminator(self):
        """Soft update target discriminator weights using Polyak averaging"""
        with torch.no_grad():
            for target_param, param in zip(self.target_discriminator.parameters(),
                                          self.discriminator.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )

    def sync_discriminator_to_target(self):
        """Fully synchronize main discriminator to target"""
        self.discriminator.load_state_dict(self.target_discriminator.state_dict())
        
    def sync_target_to_discriminator(self):
        """Fully synchronize target discriminator to main"""
        self.target_discriminator.load_state_dict(self.discriminator.state_dict())
        
    def get_state_dict(self):
        sd = super().get_state_dict()
        sd['target_discriminator'] = self.target_discriminator.state_dict()
        return sd

    def load_state_dict(self, sd, ignore_missing=False):
        super().load_state_dict(sd, ignore_missing)
        try:
            self.target_discriminator.load_state_dict(sd['target_discriminator'])
        except KeyError as e:
            if not ignore_missing:
                raise e

