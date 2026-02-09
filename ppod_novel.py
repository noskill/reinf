import torch
from typing import Dict, List, Optional, Sequence, Tuple

from ppod import PPOD, PPODRunning
from util import EpisodeBatch


class NoveltyReward:
    """Compute novelty rewards from dict observations.

    Expects:
      - cube_positions: [N, 3*K] or [N, K, 3]
      - cube_velocities: [N, 3*K] or [N, K, 3]
    """

    def __init__(
        self,
        novelty,
        *,
        position_field: str = "cube_positions",
        velocity_field: str = "cube_velocities",
        velocity_threshold: float = 0.02,
        xy_limit: float = 0.7,
    ):
        self.novelty = novelty
        self.position_field = position_field
        self.velocity_field = velocity_field
        self.velocity_threshold = float(velocity_threshold)
        self.xy_limit = float(xy_limit)

    def _concat_obs(self, states_list: Sequence[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        if not states_list:
            return {}, []
        keys = list(states_list[0].keys())
        for s in states_list:
            if set(s.keys()) != set(keys):
                raise ValueError("All state dicts must share identical keys for novelty computation")
        episode_lengths = [int(s[keys[0]].shape[0]) for s in states_list]
        obs_concat = {k: torch.cat([s[k] for s in states_list], dim=0) for k in keys}
        return obs_concat, episode_lengths

    def _reshape_cube_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if x.dim() == 3 and x.shape[-1] == 3:
            return x
        if x.dim() == 2 and x.shape[-1] % 3 == 0:
            return x.view(x.shape[0], -1, 3)
        return None

    def _novelty_mask(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.position_field not in obs or self.velocity_field not in obs:
            return torch.ones(obs[next(iter(obs))].shape[0], dtype=torch.bool, device=next(iter(obs.values())).device)
        positions = self._reshape_cube_features(obs[self.position_field])
        velocities = self._reshape_cube_features(obs[self.velocity_field])
        if positions is None or velocities is None:
            return torch.ones(obs[self.position_field].shape[0], dtype=torch.bool, device=obs[self.position_field].device)
        if positions.shape[0] == 0:
            return torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
        xy = positions[..., :2].abs()
        within_xy_mask = xy.view(positions.shape[0], -1).le(self.xy_limit).all(dim=1)
        vel_norm = torch.norm(velocities, dim=-1)
        stable_mask = vel_norm.le(self.velocity_threshold).all(dim=1)
        return within_xy_mask & stable_mask

    def compute(
        self,
        states_list: Sequence[Dict[str, torch.Tensor]],
        *,
        get_descriptor_input,
        desc_discard_steps: int = 0,
        logger=None,
    ) -> List[torch.Tensor]:
        obs_concat, episode_lengths = self._concat_obs(states_list)
        if not obs_concat:
            return []
        desc_input = get_descriptor_input(obs_concat, p_drop=0)
        mask = self._novelty_mask(obs_concat)
        novelty_values = torch.zeros(desc_input.shape[0], device=desc_input.device, dtype=desc_input.dtype)
        masked_count = int(mask.sum().item())
        if masked_count > 0:
            masked_input = desc_input[mask].detach()
            novelty_subset = torch.from_numpy(self.novelty(masked_input)).to(novelty_values)
            novelty_values[mask] = novelty_subset
        if logger is not None:
            ratio = masked_count / max(1, desc_input.shape[0])
            logger.log_scalar("novelty_mask_ratio", ratio)
            if novelty_values.numel() > 0:
                logger.log_scalar("novelty_mean", novelty_values.mean().item())
                logger.log_scalar("novelty_max", novelty_values.max().item())
                logger.log_scalar("novelty_std", novelty_values.std(unbiased=False).item())
                logger.log_scalar("novelty_sum", novelty_values.sum().item())
        result = []
        offset = 0
        for ep_len in episode_lengths:
            ep = novelty_values[offset:offset + ep_len].detach().clone()
            if desc_discard_steps > 0 and ep.numel() > 0:
                ep[:desc_discard_steps] = 0
            result.append(ep)
            offset += ep_len
        return result

    def update(
        self,
        states_list: Sequence[Dict[str, torch.Tensor]],
        *,
        get_descriptor_input,
        desc_discard_steps: int = 0,
        logger=None,
    ) -> None:
        obs_concat, _ = self._concat_obs(states_list)
        if not obs_concat:
            return
        desc_input = get_descriptor_input(obs_concat, p_drop=0)
        mask = self._novelty_mask(obs_concat)
        if mask.any():
            self.novelty.update(desc_input[mask].detach())
        if logger is not None and desc_input.numel() > 0:
            norms = desc_input.norm(dim=1)
            logger.log_scalar("descriptor_norm_mean", norms.mean().item())
            if norms.shape[0] > 1:
                logger.log_scalar("descriptor_norm_std", norms.std(unbiased=False).item())
            feature_var = desc_input.var(dim=0, unbiased=False).mean().item()
            logger.log_scalar("descriptor_feature_var_mean", feature_var)


class AdditionalRewardMixin:
    """Mixin that augments PPOD rewards with an external provider."""

    def __init__(self, *args, additional_reward=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_reward = additional_reward

    def _log_diayn_stats(self, rewards_list: Sequence[torch.Tensor]) -> None:
        if self.logger is None:
            return
        segments = [r[self.desc_discard_steps:] for r in rewards_list if r.numel() > self.desc_discard_steps]
        if not segments:
            return
        concat = torch.cat(segments)
        self.logger.log_scalar("reward_diayn_mean", concat.mean().item())
        self.logger.log_scalar("reward_diayn_std", concat.std(unbiased=False).item())
        self.logger.log_scalar("reward_diayn_sum", concat.sum().item())

    def compute_additional_reward(self, states_list, skills_list, **kwargs):
        base_rewards = super().compute_additional_reward(states_list, skills_list, **kwargs)
        self._log_diayn_stats(base_rewards)
        if self.additional_reward is None:
            return base_rewards
        extra = self.additional_reward.compute(
            states_list,
            get_descriptor_input=self.get_descriminator_input,
            desc_discard_steps=self.desc_discard_steps,
            logger=self.logger,
        )
        if len(extra) != len(base_rewards):
            raise ValueError("additional_reward.compute must return per-episode rewards")
        combined = []
        for base, add in zip(base_rewards, extra):
            combined.append(base + add.to(base))
        return combined

    def learn_from_episodes(self, episodes, num_minibatches=4):
        super().learn_from_episodes(episodes, num_minibatches=num_minibatches)
        if self.additional_reward is None:
            return
        if isinstance(episodes, EpisodeBatch):
            batch = episodes
        else:
            batch = self._extract_episode_data(episodes)
        if batch.num_episodes == 0:
            return
        filtered = self.filter_short_episodes(batch.data, self.desc_discard_steps)
        if filtered is None or not filtered['states']:
            return
        states_list = filtered['states']
        self.additional_reward.update(
            states_list,
            get_descriptor_input=self.get_descriminator_input,
            desc_discard_steps=self.desc_discard_steps,
            logger=self.logger,
        )


class PPODNovel(AdditionalRewardMixin, PPOD):
    def __init__(self, *args, novelty=None, **kwargs):
        novelty_kwargs = dict(
            position_field=kwargs.pop("novelty_position_field", "cube_positions"),
            velocity_field=kwargs.pop("novelty_velocity_field", "cube_velocities"),
            velocity_threshold=kwargs.pop("novelty_velocity_threshold", 0.02),
            xy_limit=kwargs.pop("novelty_xy_limit", 0.7),
        )
        additional_reward = NoveltyReward(novelty, **novelty_kwargs) if novelty is not None else None
        super().__init__(*args, additional_reward=additional_reward, **kwargs)


class PPODNovelRunning(AdditionalRewardMixin, PPODRunning):
    def __init__(self, *args, novelty=None, **kwargs):
        novelty_kwargs = dict(
            position_field=kwargs.pop("novelty_position_field", "cube_positions"),
            velocity_field=kwargs.pop("novelty_velocity_field", "cube_velocities"),
            velocity_threshold=kwargs.pop("novelty_velocity_threshold", 0.02),
            xy_limit=kwargs.pop("novelty_xy_limit", 0.7),
        )
        additional_reward = NoveltyReward(novelty, **novelty_kwargs) if novelty is not None else None
        super().__init__(*args, additional_reward=additional_reward, **kwargs)
