import gymnasium as gym
import numpy as np
import torch

from env_setup import MazeTrainerEnvAdapter


class SharedBodyMazeEnv(MazeTrainerEnvAdapter):
    """One body, averaged probability votes and one-step detached messages."""

    def __init__(self, env, num_agents, message_dim):
        super().__init__(env)
        if num_agents not in (2, 3):
            raise ValueError("Shared-body maze requires two or three agents")
        if message_dim < 0:
            raise ValueError("message_dim must be non-negative")
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.num_actions = env.action_space_n
        self.sensor_indices = torch.tensor([(0, 1), (1, 2), (0, 2)][:num_agents], device=self.device)
        self.device = self.sensor_indices.device
        peer_indices = [[sender for sender in range(num_agents) if sender != receiver]
                        for receiver in range(num_agents)]
        if message_dim == 0:
            peer_indices = [[] for _ in range(num_agents)]
        self.peer_indices = torch.tensor(peer_indices, dtype=torch.long, device=self.device)
        self.peer_count = self.peer_indices.shape[1]
        message_shape = (self.peer_count, self.message_dim)
        self.local_observation_space = gym.spaces.Dict({
            "sensor": gym.spaces.Dict({
                "local": gym.spaces.Box(0, np.inf, shape=(2,), dtype=np.float32),
                "message": gym.spaces.Box(-np.inf, np.inf, shape=message_shape, dtype=np.float32),
                "message_valid": gym.spaces.Box(0, 1, shape=(self.peer_count,), dtype=np.bool_)})})
        self.observation_space = gym.spaces.Dict({
            "sensor": gym.spaces.Dict({
                "local": gym.spaces.Box(0, np.inf, shape=(num_agents, 2), dtype=np.float32),
                "message": gym.spaces.Box(-np.inf, np.inf, shape=(num_agents, *message_shape), dtype=np.float32),
                "message_valid": gym.spaces.Box(0, 1, shape=(num_agents, self.peer_count), dtype=np.bool_)}),
            "location": gym.spaces.Box(0, np.iinfo(np.int64).max, shape=(num_agents, 2), dtype=np.int64),
            "heading_idx": gym.spaces.Box(0, 3, shape=(num_agents,), dtype=np.int64)})
        self.action_space = gym.spaces.Dict({
            "votes": gym.spaces.Box(0, 1, shape=(num_agents, self.num_actions), dtype=np.float32),
            "messages": gym.spaces.Box(-np.inf, np.inf, shape=(num_agents, message_dim), dtype=np.float32)})
        self._messages = torch.zeros(self.num_envs, num_agents, message_dim, device=self.device)
        self._message_valid = torch.zeros(self.num_envs, num_agents, dtype=torch.bool, device=self.device)

    def _local_observation(self, obs, messages, valid):
        return {
            "sensor": {"local": obs["sensor"][:, self.sensor_indices],
                       "message": messages[:, self.peer_indices].detach(),
                       "message_valid": valid[:, self.peer_indices]},
            "location": obs["location"][:, None].expand(-1, self.num_agents, -1),
            "heading_idx": obs["heading_idx"][:, None].expand(-1, self.num_agents)}

    def reset(self):
        self._messages.zero_()
        self._message_valid.zero_()
        return self._local_observation(self.env.reset(), self._messages, self._message_valid)

    @torch.no_grad()
    def step(self, action):
        if not isinstance(action, dict) or set(action) != {"votes", "messages"}:
            raise ValueError("Shared-body action must contain votes and messages")
        for name, space in self.action_space.spaces.items():
            value = action[name]
            expected_shape = (self.num_envs, *space.shape)
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"Shared-body {name} must be a floating-point tensor")
            if value.device != self.device:
                raise ValueError(f"Shared-body {name} must be on {self.device}, got {value.device}")
            if value.shape != expected_shape or not torch.isfinite(value).all():
                raise ValueError(f"Expected finite {name} {expected_shape}, got {tuple(value.shape)}")
        votes = action["votes"].detach()
        if (votes < 0).any() or not torch.allclose(votes.sum(-1), torch.ones_like(votes[..., 0]), atol=1e-6, rtol=1e-6):
            raise ValueError("Each vote must be a nonnegative probability vector summing to one")
        messages = action["messages"].detach().clone()
        body_probs = votes.mean(dim=1)
        distribution = torch.distributions.Categorical(probs=body_probs)
        body_action = distribution.sample()
        obs, reward, done, info = self.env.step(body_action)
        physical_next = {
            "sensor": torch.as_tensor(np.stack([item["sensor"] for item in info["per_env"]]), device=self.device),
            "location": torch.as_tensor(np.stack([item["location"] for item in info["per_env"]]), device=self.device),
            "heading_idx": torch.tensor([item["heading_idx"] for item in info["per_env"]], device=self.device)}
        valid = torch.ones_like(self._message_valid)
        info["shared_body"] = {
            "action": body_action, "body_probs": body_probs, "votes": votes.clone(),
            "next_observation": self._local_observation(physical_next, messages, valid)}
        messages[done] = 0
        valid[done] = False
        self._messages, self._message_valid = messages, valid
        return self._local_observation(obs, messages, valid), reward, done, info
