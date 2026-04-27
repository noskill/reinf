import numpy
import copy
import torch
import random
import torch.optim as optim
from torch.distributions import Categorical, Normal, Uniform, TransformedDistribution, Independent
from sample import NormalActionSampler
import torch.nn as nn
import torch.nn.functional as F
from ppo import PPOBase
from pool import EpisodesPoolMixin
from networks import SkillDiscriminator
from util import EpisodeBatch, normalize_padded_returns, flatten_padded, to_device


class PPODPool(EpisodesPoolMixin):

    def add_transition(self, env_idx, state, action, log_prob, entropy):
        self.episodes[env_idx].append((state, action, log_prob, entropy, self._skills[env_idx]))


"""DIAYN PPO (PPOD) components and training.

This module now separates skill sampling (SkillSampler) from policy embedding:
- Policy should be a skill-aware model that accepts obs dict with 'skills'.
- Discriminator inputs are built from obs dict fields; no StateExtractor.
"""


class SkillSampler:
    """Samples skills for DIAYN.

    - Discrete: samples indices in [0, num_skills)
    - Continuous: samples vectors from N(0, I) in R^{skill_dim}
    """

    def __init__(self, *, continuous: bool, num_skills: int, skill_dim: int, device: torch.device):
        self.continuous = continuous
        self.num_skills = num_skills
        self.skill_dim = skill_dim
        self.device = device

        if not continuous:
            self._logits = torch.ones(num_skills, device=device) / float(num_skills)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.continuous:
            return torch.randn(batch_size, self.skill_dim, device=self.device)
        else:
            cat = Categorical(probs=self._logits)
            return cat.sample((batch_size,))


class PPOD(PPOBase, PPODPool):
    """Unified DIAYN PPO supporting both Continuous and Categorical skills."""

    def __init__(self, policy, value, sampler, obs_dim,
                 skill_dim=8, embedding_dim=8, continious=False,
                 policy_lr=0.0001, num_envs=8, discount=0.99,
                 device=torch.device('cpu'), logger=None, num_learning_epochs=4, discriminator=None,
                 discriminator_fields=None, desc_discard_steps=0, disc_lr=-1, **kwargs):
        self.skill_dim = skill_dim
        self.embedding_dim = embedding_dim
        self.continious = continious
        self.device = device
        self.skill_state_len = skill_dim
        if not self.continious:
            self.skill_state_len = 1 # discrete distribution - just 1 number
        self.discriminator = discriminator
        # Create sampler for skills (separate from embedding)
        self.skill_sampler = SkillSampler(
            continuous=continious,
            num_skills=skill_dim if not continious else 0,
            skill_dim=skill_dim,
            device=device,
        )

        # Expect 'policy' to be skill-aware (e.g., IsaacLabSkillPolicy)
        assert disc_lr > 0
        self.discriminator_lr = disc_lr
        self.desc_fields = discriminator_fields
        if self.desc_fields is None:
            self.desc_fields = ['cube_positions', 'cube_orientations']
        self.desc_discard_steps = desc_discard_steps
        # Initialize superclasses (PPOBase, Pool)
        super().__init__(policy, value, sampler, policy_lr=policy_lr,
                         num_envs=num_envs, discount=discount, device=device, logger=logger,
                         num_learning_epochs=num_learning_epochs, **kwargs)
        if 'skill' not in self.pad_fields:
            self.pad_fields = list(self.pad_fields) + ['skill']
        self.sequence_pad_fields = self.pad_fields

        self.data_types = ['states', 'actions', 'log_probs', 'entropy', 'skill', 'rewards']
        self.p_drop = kwargs.get('p_drop', 0)
        print('feature dropout ', str(self.p_drop))

    def prepare_discriminator_batches(self, states_list, skills_list):
        """
        Filter states and skills for discriminator training by removing warmup steps from each episode.

        Args:
            states_list: List of state tensors for each episode
            skills_list: List of skill tensors for each episode

        Returns:
            Tuple of (filtered_states_batch, filtered_skills_batch) as concatenated tensors
        """
        # Build flattened obs dict across episodes after discarding warmup
        obs_accum = None  # dict of lists per key
        filtered_skills = []
        episode_lengths = []

        for episode_states, episode_skills in zip(states_list, skills_list):
            # Support both dict-obs and flat tensors from older pipelines
            if isinstance(episode_states, dict):
                T = next(iter(episode_states.values())).shape[0]
                if T <= self.desc_discard_steps:
                    continue
                sliced_obs = {k: v[self.desc_discard_steps:] for k, v in episode_states.items()}
                episode_len = next(iter(sliced_obs.values())).shape[0]
                if episode_len == 0:
                    continue
                if obs_accum is None:
                    obs_accum = {k: [] for k in sliced_obs.keys()}
                for k, v in sliced_obs.items():
                    obs_accum[k].append(v)
            else:
                # Legacy flat tensor path: wrap under key 'flat'
                T = episode_states.shape[0]
                if T <= self.desc_discard_steps:
                    continue
                sliced_flat = episode_states[self.desc_discard_steps:]
                episode_len = sliced_flat.shape[0]
                if episode_len == 0:
                    continue
                if obs_accum is None:
                    obs_accum = {'flat': []}
                obs_accum['flat'].append(sliced_flat)
            episode_lengths.append(episode_len)

            skill_slice = episode_skills[self.desc_discard_steps:]
            if skill_slice.dim() == 0:
                skill_slice = skill_slice.repeat(episode_len)
            filtered_skills.append(skill_slice)

        # Concatenate obs per key across episodes
        obs_batch = None
        if obs_accum is not None:
            obs_batch = {k: torch.cat(v_list, dim=0).to(self.device) for k, v_list in obs_accum.items()}
        filtered_skills_batch = torch.cat(filtered_skills, dim=0).to(self.device) if filtered_skills else torch.empty(0, device=self.device)

        return obs_batch, filtered_skills_batch, episode_lengths

    def create_optimizers(self):
        super().create_optimizers()
        self.optimizer_policy = optim.Adam(
            list(self.policy.parameters()),
            lr=self.policy_lr
        )
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=0.0002
        )

    def episode_start(self):
        self.reset_episodes()
        # Sample fresh skills per environment
        self._skills = self.skill_sampler.sample(self.num_envs)
        import pdb;pdb.set_trace()

    def process_states(self, state, episode_start):
        """Attach current skills to obs dict for all envs.

        Expects 'state' as dict of tensors [N,...]. Resamples skills where
        episode_start is True.
        """
        if self._skills is None or self._skills.shape[0] != self.num_envs:
            self._skills = self.skill_sampler.sample(self.num_envs)

        if episode_start is not None:
            reset_mask = episode_start
            if not isinstance(reset_mask, torch.Tensor):
                reset_mask = torch.as_tensor(reset_mask, device=self.device)
            reset_mask = reset_mask.to(torch.bool).view(-1)
            if reset_mask.any():
                new_skills = self.skill_sampler.sample(int(reset_mask.sum().item()))
                self._skills[reset_mask] = new_skills

        obs_active = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                obs_active[k] = v.to(self.device)
            else:
                obs_active[k] = torch.as_tensor(v, device=self.device)
        obs_active['skills'] = self._skills.to(self.device)
        return obs_active

    def compute_additional_reward(self, states_list, skills_list, discriminator=None, **kwargs):
        result = []
        for states, skills in zip(states_list, skills_list):
            result.append(self.compute_reward_from_discriminator(states, skills, discriminator=discriminator))
        return result

    def _sequence_build_policy_obs(self, states_padded, key_padding_mask, padded):
        obs_for_policy = super()._sequence_build_policy_obs(states_padded, key_padding_mask, padded)
        skills = padded.get('skill')
        if isinstance(obs_for_policy, dict) and skills is not None:
            obs_for_policy['skills'] = skills
        return obs_for_policy

    def _build_flat_policy_obs(self, states_mb, skills_mb):
        if isinstance(states_mb, dict):
            obs = dict(states_mb)
            if skills_mb is not None:
                obs['skills'] = skills_mb
            return obs
        return states_mb

    def train_flat_policy(self, episode_batch: EpisodeBatch, num_minibatches: int = 4):
        flat = episode_batch.flatten(fields=['states', 'actions', 'returns', 'log_probs', 'entropy', 'skill'])
        states_batch = to_device(flat['states'], self.device)
        actions_batch = flat['actions'].to(self.device)
        returns_batch = flat['returns'].to(self.device)
        old_logp_batch = flat['log_probs'].to(self.device)
        skills_batch = flat.get('skill')

        normalized_returns = self._normalize_returns(returns_batch)
        self._log_training_stats(actions_batch)

        if isinstance(states_batch, dict):
            value_epochs = 2
            N = returns_batch.shape[0]
            mini = max(1, N // max(1, num_minibatches))
            for _ in range(value_epochs):
                perm = torch.randperm(N, device=self.device)
                for mb in torch.split(perm, mini):
                    self.optimizer_value.zero_grad()
                    v_pred = self.value({k: v[mb] for k, v in states_batch.items()}).squeeze(-1)
                    v_loss = F.mse_loss(v_pred, normalized_returns[mb])
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.4)
                    self.optimizer_value.step()
            self.logger.log_scalar("value loss", v_loss)
        else:
            self.train_value(normalized_returns, states_batch)

        with torch.no_grad():
            v_all = self.value(states_batch).squeeze(-1)
        adv_all = normalized_returns - v_all
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        self.policy.load_state_dict(self.policy_old.state_dict())

        N = old_logp_batch.shape[0]
        batch_size = max(1, N // max(1, num_minibatches))
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(N, device=self.device)
            for mb_idx in torch.split(perm, batch_size):
                if isinstance(states_batch, dict):
                    states_mb = {k: v[mb_idx] for k, v in states_batch.items()}
                else:
                    states_mb = states_batch[mb_idx]
                skills_mb = skills_batch[mb_idx] if skills_batch is not None else None
                obs_mb = self._build_flat_policy_obs(states_mb, skills_mb)
                actions_mb = actions_batch[mb_idx]
                old_logp_mb = old_logp_batch[mb_idx]
                adv_mb = adv_all[mb_idx]

                _, logp_new_mb, dist = self.sampler(self.policy, obs_mb, actions=actions_mb, return_distribution=True)
                try:
                    entropy_new_mb = dist.entropy()
                except NotImplementedError:
                    entropy_new_mb = dist.base_dist.entropy() if hasattr(dist, 'base_dist') else entropy_new_mb

                mu_mb = None
                if isinstance(self.sampler, NormalActionSampler):
                    base = dist.base_dist if isinstance(dist, TransformedDistribution) else dist
                    base_normal = base.base_dist if isinstance(base, Independent) else base
                    if hasattr(base_normal, 'loc'):
                        mu_mb = base_normal.loc

                self.train_policy(
                    log_probs_old=old_logp_mb,
                    advantages=adv_mb,
                    entropy=entropy_new_mb,
                    log_probs_new=logp_new_mb,
                    mu=mu_mb,
                )

        self.policy_old.load_state_dict(self.policy.state_dict())

    def compute_reward_from_discriminator(self, states, skill, discriminator=None):
        """Compute rewards using the specified discriminator (or default to self.discriminator)

        states Tensor episode_length x N
        skill Tensor episode_length  for discrete (episode_length x self.skill_dim) for continious
        """
        discriminator = discriminator or self.discriminator
        # Slice warmup portion and pass full obs to discriminator
        if isinstance(states, dict):
            T = next(iter(states.values())).shape[0]
            result = torch.zeros(T, device=self.device)
            obs_slice = {k: v[self.desc_discard_steps:] for k, v in states.items()}
            skill = skill[self.desc_discard_steps:]
            skill_estimate = discriminator(obs_slice)
        else:
            result = torch.zeros(states.shape[0]).to(states)
            states = states[self.desc_discard_steps:]
            skill = skill[self.desc_discard_steps:]
            skill_estimate = discriminator(states)
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
        result[self.desc_discard_steps:] = reward.detach()
        return result

    def evaluate_discriminator(self, discriminator, obs, skills):
        """Evaluate discriminator performance (MSE for continuous, accuracy for categorical).

        Pass full observations to the discriminator; the model is responsible
        for selecting relevant fields internally.
        """
        with torch.no_grad():
            pred = discriminator(obs)
            if self.continious:
                return F.mse_loss(pred, skills).item()
            else:
                acc = (pred.argmax(dim=-1) == skills.view(-1).long()).float().mean()
                return acc.item()

    def filter_short_episodes(self, data_dict, min_length):
        """
        Filter out episodes that are shorter than the specified minimum length.

        Args:
            data_dict: Dictionary containing episode data
            min_length: Minimum episode length to keep

        Returns:
            Filtered data dictionary or None if no valid episodes remain
        """
        states = data_dict['states']
        valid_indices = []

        for i, episode_states in enumerate(states):
            if isinstance(episode_states, dict):
                any_key = next(iter(episode_states))
                episode_len = int(episode_states[any_key].shape[0])
            else:
                episode_len = len(episode_states)
            if episode_len > min_length:
                valid_indices.append(i)

        if not valid_indices:
            return None  # No valid episodes

        # Create a new filtered dictionary
        filtered_dict = {}
        for key, value in data_dict.items():
            filtered_dict[key] = [value[i] for i in valid_indices]

        return filtered_dict

    def _log_descriptor_stats(self, obs, episode_lengths=None):
        if self.logger is None:
            return
        with torch.no_grad():
            desc_input = self.get_descriminator_input(obs, p_drop=0)
            if isinstance(desc_input, torch.Tensor) and desc_input.numel() > 0:
                norms = desc_input.norm(dim=1)
                self.logger.log_scalar("descriptor_norm_mean", norms.mean().item())
                if norms.shape[0] > 1:
                    self.logger.log_scalar("descriptor_norm_std", norms.std(unbiased=False).item())
                feature_var = desc_input.var(dim=0, unbiased=False).mean().item()
                self.logger.log_scalar("descriptor_feature_var_mean", feature_var)

    def learn_from_episodes(self, episodes, num_minibatches=4):
        if isinstance(episodes, EpisodeBatch):
            batch = episodes
        else:
            batch = self._extract_episode_data(episodes)
        if batch.num_episodes == 0:
            return

        batch = batch.to(self.device)
        filtered_dict = self.filter_short_episodes(batch.data, self.desc_discard_steps)
        if filtered_dict is None:
            return

        batch = EpisodeBatch(filtered_dict).to(self.device)
        states_list = batch.data['states']
        log_probs_list = batch.data['log_probs']
        actions_list = batch.data['actions']
        rewards_list = batch.data['rewards']
        entropy_list = batch.data['entropy']
        skills_list = batch.data['skill']
        if not states_list:
            return

        desc_obs_batch, desc_skills_batch, desc_episode_lengths = self.prepare_discriminator_batches(
            states_list, skills_list)
        if desc_obs_batch is not None:
            self._log_descriptor_stats(desc_obs_batch, episode_lengths=desc_episode_lengths)

        performance = self.evaluate_discriminator(self.discriminator, desc_obs_batch, desc_skills_batch)
        if self.continious:
            self.logger.log_scalar("discriminator_mse", performance)
        else:
            self.logger.log_scalar("discriminator_accuracy", performance)

        additional_reward_per_episode = self.compute_additional_reward(states_list, skills_list)
        new_rewards = []
        for rewards, reward_desc in zip(rewards_list, additional_reward_per_episode):
            new_rewards.append(rewards.to(reward_desc.device) + reward_desc)

        train_data = dict(batch.data)
        train_data['rewards'] = new_rewards
        train_batch = EpisodeBatch(train_data).to(self.device)
        train_batch.compute_returns(self.discount)

        is_sequence_model = hasattr(self.policy, 'temporal_decoder')
        if is_sequence_model:
            self.train_sequence_policy(train_batch, num_minibatches=num_minibatches)
        else:
            self.train_flat_policy(train_batch, num_minibatches=num_minibatches)

        self.train_discriminator(desc_obs_batch, desc_skills_batch, episode_lengths=desc_episode_lengths)

    def get_descriminator_input(self, obs, p_drop=0):
        """Build discriminator features from obs dict.

        - If desc_fields is provided, use that subset; otherwise use all fields
          except 'actions' and 'skills'.
        - Concatenate fields along the last dim after flattening per-field dims.
        """
        if isinstance(obs, dict):
            keys = self.desc_fields if self.desc_fields is not None else [k for k in obs.keys() if k not in ('actions', 'skills')]
            feats = []
            B = None
            for k in keys:
                x = obs[k]
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x, device=self.device)
                if B is None:
                    B = x.shape[0]
                x = x.view(x.shape[0], -1)
                feats.append(x)
            result = torch.cat(feats, dim=-1) if feats else torch.zeros((B or 0, 0), device=self.device)
        else:
            # Assume already a flat tensor batch
            result = obs

        if p_drop > 0:
            p_drop = float(p_drop)
            if p_drop >= 1.0:
                result = torch.zeros_like(result)
            else:
                mask = (torch.rand_like(result) >= p_drop).to(result.dtype)
                result = result * mask
        return result

    def train_discriminator(self, obs, skills, episode_lengths):
        """Train the discriminator to predict skills from observations.

        obs: dict of tensors [N,...] or flat tensor [N, F]
        skills: [N] for discrete or [N, D] for continuous
        """
        train_epochs = 4 
        mini_batch_size = 128
        # Determine dataset size
        if isinstance(obs, dict):
            num_samples = next(iter(obs.values())).shape[0]
        else:
            num_samples = obs.shape[0]
        num_episodes = len(episode_lengths)
        if num_samples == 0:
            return
        holdout_size = int(max(1, round(num_episodes * 0.05))) if num_episodes > 2 else 0

        if holdout_size >= num_episodes:
            holdout_size = max(1, num_episodes - 1)

        assert holdout_size < num_episodes
        sum_holdout = sum(episode_lengths[:holdout_size])
        if isinstance(obs, dict):
            train_obs = {k: v[sum_holdout:] for k, v in obs.items()}
            holdout_obs = {k: v[:sum_holdout] for k, v in obs.items()}
        else:
            train_obs = obs[sum_holdout:]
            holdout_obs = obs[:sum_holdout]
        train_skills = skills[sum_holdout:]
        holdout_skills = skills[:sum_holdout]


        total_loss = 0.0
        loss_steps = 0
        first_layer_grad_sum = 0.0
        first_layer_grad_steps = 0

        for epoch in range(train_epochs):
            if len(train_inputs) == 0:
                break
            N = train_skills.shape[0]
            indices = torch.randperm(N)
            for start in range(0, len(train_inputs), mini_batch_size):
                self.optimizer_discriminator.zero_grad()
                end = min(start + mini_batch_size, len(train_inputs))
                batch_idx = indices[start:end]
                if isinstance(train_obs, dict):
                    obs_mb = {k: v[batch_idx] for k, v in train_obs.items()}
                else:
                    obs_mb = train_obs[batch_idx]
                preds = self.discriminator(obs_mb)
                if self.continious:
                    loss = F.mse_loss(preds, train_skills[batch_idx])
                else:
                    loss = F.cross_entropy(preds, train_skills[batch_idx].long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)

                first_layer = getattr(self.discriminator, "first_layer", None)
                if first_layer is not None and hasattr(first_layer, "weight") and first_layer.weight.grad is not None:
                    first_layer_grad_sum += first_layer.weight.grad.norm().item()
                    first_layer_grad_steps += 1

                self.optimizer_discriminator.step()

                total_loss += loss.item()
                loss_steps += 1

        if loss_steps > 0:
            self.logger.log_scalar("discriminator_loss", total_loss / loss_steps)
        if first_layer_grad_steps > 0:
            self.logger.log_scalar(
                "discriminator_first_layer_grad_norm",
                first_layer_grad_sum / first_layer_grad_steps
            )

        with torch.no_grad():
            if isinstance(holdout_obs, dict):
                N_hold = next(iter(holdout_obs.values())).shape[0]
            else:
                N_hold = holdout_obs.shape[0]
            if N_hold > 0:
                preds = self.discriminator(holdout_obs)
                if self.continious:
                    holdout_mse = F.mse_loss(preds, holdout_skills)
                    self.logger.log_scalar("discriminator_holdout_mse", holdout_mse.item())
                else:
                    acc = (preds.argmax(dim=-1) == holdout_skills.view(-1).long()).float().mean()
                    self.logger.log_scalar("discriminator_holdout_accuracy", acc.item())

    # checkpointing support
    def get_state_dict(self):
        sd = super().get_state_dict()
        sd['skill_embedding'] = self.skill_embedding.state_dict()
        sd['discriminator'] = self.discriminator.state_dict()
        sd['optimizer_discriminator'] = self.optimizer_discriminator.state_dict()
        sd['desc_fields'] = self.desc_fields
        sd['desc_discard_steps'] = self.desc_discard_steps
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
            else:
                logger.warning("error in loading state dictionary", e)


class PPODRunning(PPOD):
    """ DIAYN implementation with running average discriminator stabilization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create target discriminator (for stable reward computation)
        self.target_discriminator = copy.deepcopy(self.discriminator)

        # Polyak averaging coefficient (τ)
        self.tau = 0.05  # Slow update rate

        # Synchronization settings
        self.sync_interval = 50
        self.sync_threshold = 0.05
        self.reset_threshold = 0.2
        self.previous_performance = None
        self.iteration_count = 0

    def learn_from_episodes(self, episodes, num_minibatches=4):
        if isinstance(episodes, EpisodeBatch):
            batch = episodes
        else:
            batch = self._extract_episode_data(episodes)
        if batch.num_episodes == 0:
            return

        batch = batch.to(self.device)
        filtered_dict = self.filter_short_episodes(batch.data, self.desc_discard_steps)
        if filtered_dict is None:
            return
        batch = EpisodeBatch(filtered_dict).to(self.device)

        states = batch.data['states']
        rewards = batch.data['rewards']
        skills = batch.data['skill']
        if not states:
            return

        desc_obs_batch, desc_skills_batch, episode_lengths = self.prepare_discriminator_batches(states, skills)
        # Evaluate main and target discriminator performance
        target_perf = self.evaluate_discriminator(self.target_discriminator, desc_obs_batch, desc_skills_batch)
        main_perf = self.evaluate_discriminator(self.discriminator, desc_obs_batch, desc_skills_batch)
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
            main_perf = self.evaluate_discriminator(self.discriminator, desc_obs_batch, desc_skills_batch)

        self.previous_performance = main_perf

        # Book-keeping for periodic hard synchronisation of the target network
        self.iteration_count += 1
        if self.sync_interval > 0 and (self.iteration_count % self.sync_interval) == 0:
            # Periodically reset the fast discriminator back to the slow target to curb drift
            self.sync_discriminator_to_target()

        # Compute discriminator-based rewards using TARGET discriminator
        additional_reward_per_episode = self.compute_additional_reward(
            states, skills, discriminator=self.target_discriminator)
        new_rewards = []
        for rewards_ep, reward_desc in zip(rewards, additional_reward_per_episode):
            new_rewards.append(rewards_ep.to(reward_desc.device) + reward_desc)

        train_data = dict(batch.data)
        train_data['rewards'] = new_rewards
        train_batch = EpisodeBatch(train_data).to(self.device)
        train_batch.compute_returns(self.discount)

        is_sequence_model = hasattr(self.policy, 'temporal_decoder')
        if is_sequence_model:
            self.train_sequence_policy(train_batch, num_minibatches=num_minibatches)
        else:
            self.train_flat_policy(train_batch, num_minibatches=num_minibatches)

        # Train discriminator
        self.train_discriminator(desc_obs_batch, desc_skills_batch, episode_lengths=episode_lengths)

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
        print('syncing discriminator to target')
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
            # Ensure the fast discriminator starts exactly from the slow copy when resuming
            self.sync_discriminator_to_target()
            self.iteration_count = 0
        except KeyError as e:
            if not ignore_missing:
                raise e

    def update(self, obs, actions, rewards, dones, next_obs, info=None):
        if 'grasp_success_rate' in info:
            self.logger.log_scalar('grasp_success_rate', info['grasp_success_rate'])

        if 'stack_success_rate' in info:
            self.logger.log_scalar('stack_success_rate', info['stack_success_rate'])
        return super().update(obs, actions, rewards, dones, next_obs, info=info)
