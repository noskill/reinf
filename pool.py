import random


class EpisodesPoolMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes = [[] for _ in range(self.num_envs)]
        self.completed = []
        self.versions = [None for _ in range(self.num_envs)]

    def reset_episodes(self):
        self.episodes = [[] for _ in range(self.num_envs)]
        self.completed = []

    def add_transition(self, env_idx, state, action, log_prob):
        self.episodes[env_idx].append((state, action, log_prob))

    def add_reward(self, env_idx, reward):
        if self.episodes[env_idx] and len(self.episodes[env_idx]) > 0:
            self.episodes[env_idx][-1] += (reward,)

    def process_dones(self, dones):
        completed_episodes = []
        for env_idx, done in enumerate(dones):
            if done and len(self.episodes[env_idx]) > 0:
                completed_episodes.append(self.episodes[env_idx])
                self.episodes[env_idx] = []
                self.versions[env_idx] = self.version
        self.completed.extend(completed_episodes)
        return completed_episodes

    def should_learn(self):
        return len(self.completed) >= self.num_envs

    def get_completed_episodes(self):
        return self.completed

    def clear_completed(self):
        self.completed = []
        for env_idx in range(self.num_envs):
            if self.versions[env_idx] != self.version:
                self.episodes[env_idx] = []


class EpisodesOldPoolMixin(EpisodesPoolMixin):
    def __init__(self, *args, pool_size=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        self.episode_pool = []

    def process_dones(self, dones):
        completed_episodes = super().process_dones(dones)
        
        # Add completed episodes to the pool
        for episode in completed_episodes:
            if len(self.episode_pool) < self.pool_size:
                self.episode_pool.append(episode)
            else:
                # Replace random episode in the pool
                idx = random.randint(0, self.pool_size - 1)
                self.episode_pool[idx] = episode
        
        return completed_episodes

    def get_completed_episodes(self):
        if len(self.episode_pool) == 0:
            return []
        # Select random subset from pool for learning
        return random.sample(
            self.episode_pool, 
            min(self.num_envs, len(self.episode_pool))
        )

    def clear_completed(self):
        # Clear completed but don't touch the pool
        self.completed = []
