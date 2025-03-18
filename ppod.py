import numpy
import copy
import torch
import torch.optim as optim
from torch.distributions import *
import torch.nn as nn
import torch.nn.functional as F
from ppo import PPOBase
from pool import EpisodesPoolMixin


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=1000):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Create positional encoding
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                            (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: indices of shape (batch_size, seq_len)
        # categorical_embedding: (batch_size, seq_len, embedding_dim)
        import pdb;pdb.set_trace()
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1))
        pos_encoding = self.pe[:, :positions.size(1)]

        return pos_encoding


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


class PPOD(PPOBase, PPODPool):
    """ Diversity is all you need implementation"""

    def __init__(self, policy, value, sampler, obs_dim, policy_lr=0.0001, value_lr=0.001, num_envs=8, discount=0.99, device=torch.device('cpu'), logger=None, num_learning_epochs=4, clip_param=None, n_skills=8, embedding_dim=8, continious=False, **kwargs):
        self.n_skills = n_skills
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.n_skills, embedding_dim).to(device)
        self.embedding_old = copy.deepcopy(self.embedding).to(device)
        self.discriminator = SkillDiscriminator(obs_dim, n_skills, continuous=continious).to(device)
        self.discriminator_lr = policy_lr / 10
        self.continious = continious
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                    num_envs=num_envs, discount=discount,
                    device=device, logger=logger, **kwargs)
        logits = torch.tensor([1 / self.n_skills for i in range(self.n_skills)])
        logits = torch.stack([logits for _ in range(self.num_envs)])
        self.skill_logits = logits.to(device)
        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'skill', 'rewards']

    def create_optimizers(self):
        super().create_optimizers()
        # self.optimizer_policy = optim.Adam(list(self.policy.parameters()) +
        #                                    list(self.embedding.parameters()), lr=self.policy_lr)
        self.optimizer_policy = optim.Adam(list(self.policy.parameters()), lr=self.policy_lr)

        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, weight_decay=0.001)

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

    def desc_accuracy(self, discriminator, states_without_skill, skill_batch):
        with torch.no_grad():
            skill_estimate = discriminator(states_without_skill)
            row_indices = torch.arange(skill_estimate.size(0)).to(skill_batch.device)
            skill_pred = skill_estimate[row_indices, skill_batch]
            # Compute predicted class indices (argmax along columns)
            pred = torch.argmax(skill_estimate, dim=1)

            # Compare predictions to ground truth "selected"
            correct = skill_batch == pred  # Boolean tensor: True where correct

            accuracy = correct.float().mean()  # Convert to float and compute mean
            return accuracy, skill_pred

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

        accuracy, skill_pred = self.desc_accuracy(self.discriminator, states_without_skill, skill_batch)
        self.logger.log_scalar("desc accuracy before", accuracy)


        rew_desc = []
        for episode_states, episode_skill in zip(states_list, skill_list):
            s_without_skill = episode_states[:, :-self.embedding_dim]
            with torch.no_grad():
                skill_estimate = self.discriminator(s_without_skill)
                row_indices = torch.arange(skill_estimate.size(0)).to(episode_skill.device)
                skill_pred = skill_estimate[row_indices, episode_skill]
                rew_desc.append(skill_pred)

        # returns list is a list of 1d torch tensors of variying length. Each list corresponds to one episode
        # skill_pred is 1d tensor, length is sum of length of each episode
        rewards_list_new = []
        for rew_desc, rew in zip(rew_desc, rewards_list):
            rew_new = rew_desc.detach() + rew.to(rew_desc)
            rewards_list_new.append(rew_new)
        returns_list_new = [self._compute_discounted_returns(rewards) for rewards in rewards_list_new]
        returns_batch = torch.cat(returns_list_new, dim=0).to(self.device)

        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)
        # Log statistics
        self._log_training_stats(actions_batch)
        self.train_value(normalized_returns, states_batch, value_epochs=4)
        # Sync policies
        self.policy.load_state_dict(self.policy_old.state_dict())
        self.embedding.load_state_dict(self.embedding_old.state_dict())
        # Loop through epochs and call separated method
        for i in range(self.num_learning_epochs):
            self._learn_epoch(
                states_without_skill, log_probs_batch, normalized_returns,
                actions_batch, entropy_batch, num_minibatches, skill_batch
            )
        self.train_discriminator(states_without_skill.detach(), skill_batch.detach())
        accuracy, _ = self.desc_accuracy(self.discriminator, states_without_skill, skill_batch)
        self.logger.log_scalar("desc accuracy after", accuracy)  # e.g., 0.8532
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
            advantage_std = advantages.std() + 1e-7
            self.logger.log_scalar("Raw advantage mean:", advantages.mean().item())
            self.logger.log_scalar("Raw advantage std:", advantage_std.item())
            advantage_mean = advantages.mean()
            advantages = (advantages - advantage_mean) / advantage_std

            if torch.isnan(advantage_std).any():
                import pdb; pdb.set_trace()


            advantages = torch.clamp(advantages, -10.0, 10.0)
            # Ensure proper entropy shape
            if len(entropy_minibatch.shape) == 2 and entropy_minibatch.shape[1] == 1:
                entropy_minibatch = entropy_minibatch.flatten()

            # Call policy training step
            self.train_policy(
                log_probs_minibatch, advantages, entropy_minibatch, state_emb, actions_minibatch
            )

    def train_discriminator(self, states_batch, skills):
        train_epochs = 1
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
        # Log gradients
        grads = []
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
                if torch.isnan(param.grad).any():
                    import pdb;pdb.set_trace()

        self.logger.log_scalar("desc grad std:", numpy.std(grads))
        self.logger.log_scalar(f"desc loss",  loss)

    def get_state_dict(self):
        sd = super().get_state_dict()
        sd['embedding'] = self.embedding.state_dict()
        sd['discriminator'] = self.discriminator.state_dict()
        sd['optimizer_discriminator'] = self.optimizer_discriminator.state_dict()
        return sd

    def load_state_dict(self, sd, ignore_missing=False):
        super().load_state_dict(sd)
        try:
            self.embedding.load_state_dict(sd['embedding'])
        except  KeyError as e:
            if not ignore_missing:
                raise e
        self.discriminator.load_state_dict(sd['discriminator'])
        self.optimizer_discriminator.load_state_dict(sd['optimizer_discriminator'])
        self.embedding_old.load_state_dict(self.embedding.state_dict())


class PPODRunning(PPOD):
    """ Diversity is all you need implementation with running average discriminator stabilisation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create target discriminator (for stable reward computation)
        self.target_discriminator = copy.deepcopy(self.discriminator)

        # Initialize target with same weights as main discriminator
        self.target_discriminator.load_state_dict(self.discriminator.state_dict())

        # Polyak averaging coefficient (τ)
        self.tau = 0.1  # Slow update rate (0.005-0.01 is typical)
        # Synchronization settings
        self.sync_interval = 100  # Check for sync every 100 iterations
        self.sync_threshold = 0.05  # Sync if target is better by this margin
        self.reset_threshold = 0.2  # Reset to target if accuracy drops by this much
        self.previous_accuracy = None
        self.iteration_count = 0

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

        target_accuracy_before, _ = self.desc_accuracy(self.target_discriminator, states_without_skill, skill_batch)
        accuracy_before, skill_pred = self.desc_accuracy(self.discriminator, states_without_skill, skill_batch)
        self.logger.log_scalar("desc accuracy before", accuracy_before)
        self.logger.log_scalar("target desc accuracy before", target_accuracy_before)

        # Check for significant accuracy drop
        if self.previous_accuracy is not None:
            accuracy_drop = self.previous_accuracy - accuracy_before

            # If accuracy dropped significantly, reset to target
            if accuracy_drop > self.reset_threshold and target_accuracy_before > accuracy_before:
                print(f"Discriminator accuracy dropped by {accuracy_drop:.4f}, resetting to target")
                self.sync_discriminator_to_target()
                # Recompute accuracy after reset
                accuracy_before, _ = self.desc_accuracy(self.discriminator, states_without_skill, skill_batch)
        self.previous_accuracy = accuracy_before
        # Compute rewards using TARGET discriminator for stability
        rew_desc = []
        for episode_states, episode_skill in zip(states_list, skill_list):
            s_without_skill = episode_states[:, :-self.embedding_dim]
            with torch.no_grad():
                skill_estimate = self.target_discriminator(s_without_skill)
                row_indices = torch.arange(skill_estimate.size(0)).to(episode_skill.device)
                skill_pred = skill_estimate[row_indices, episode_skill]
                rew_desc.append(skill_pred)

        # returns list is a list of 1d torch tensors of variying length. Each list corresponds to one episode
        # skill_pred is 1d tensor, length is sum of length of each episode
        rewards_list_new = []
        for rew_desc, rew in zip(rew_desc, rewards_list):
            rew_new = rew_desc.detach() + rew.to(rew_desc)
            rewards_list_new.append(rew_new)
        returns_list_new = [self._compute_discounted_returns(rewards) for rewards in rewards_list_new]
        returns_batch = torch.cat(returns_list_new, dim=0).to(self.device)

        # Normalize the returns
        normalized_returns = self._normalize_returns(returns_batch)
        # Log statistics
        self._log_training_stats(actions_batch)
        self.train_value(normalized_returns, states_batch, value_epochs=4)
        # Sync policies
        self.policy.load_state_dict(self.policy_old.state_dict())
        self.embedding.load_state_dict(self.embedding_old.state_dict())
        # Loop through epochs and call separated method
        for i in range(self.num_learning_epochs):
            self._learn_epoch(
                states_without_skill, log_probs_batch, normalized_returns,
                actions_batch, entropy_batch, num_minibatches, skill_batch
            )
        self.train_discriminator(states_without_skill.detach(), skill_batch.detach())
        self.update_target_discriminator()
        accuracy, _ = self.desc_accuracy(self.target_discriminator, states_without_skill, skill_batch)
        self.logger.log_scalar("desc accuracy after", accuracy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.embedding_old.load_state_dict(self.embedding.state_dict())

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
        self.logger.log_scalar("discriminator_sync_to_target", 1.0)

    def sync_target_to_discriminator(self):
        """Fully synchronize target discriminator to main"""
        self.target_discriminator.load_state_dict(self.discriminator.state_dict())
        self.logger.log_scalar("target_sync_to_discriminator", 1.0)

