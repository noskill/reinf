import torch
from ppod import PPOD, PPODRunning


class _PPODNovelMixin:
    def __init__(self, *args, novelty=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.novelty = novelty
        self.use_novelty = True
        self.novelty_velocity_threshold = kwargs.get('novelty_velocity_threshold', 0.03)
        self.novelty_xy_limit = kwargs.get('novelty_xy_limit', 1.7)

    def compute_additional_reward(self, states, skill, discriminator=None, episode_lengths=None, **kwargs):
        rew_desc_per_episode = []
        s = 0
        for el in episode_lengths:
            st = states[s: s + el]
            sk = skill[s: s + el]
            rew_desc = self.compute_reward_from_discriminator(st, sk, discriminator=discriminator)
            rew_desc_per_episode.append(rew_desc)
            s += el
        result = rew_desc_per_episode
        desc_segments = [tensor[self.desc_discard_steps:] for tensor in result if tensor.numel() > self.desc_discard_steps]
        if desc_segments:
            desc_concat = torch.cat(desc_segments)
            self.logger.log_scalar("reward_diayn_mean", desc_concat.mean().item())
            self.logger.log_scalar("reward_diayn_std", desc_concat.std(unbiased=False).item())
            self.logger.log_scalar("reward_diayn_sum", desc_concat.sum().item())
        if self.use_novelty:
            # 1d array
            rew_novel = self.compute_novelty_reward(states, episode_lengths=episode_lengths)
            rew_acc = []
            # assign rewards from concatentated list to per/episode
            s = 0
            for idx, el in enumerate(episode_lengths):
                item = torch.from_numpy(rew_novel[s: s + el][self.desc_discard_steps:]).to(result[idx])
                result[idx][self.desc_discard_steps:] += item
                rew_acc.append(item)
                s += el
            if rew_acc:
                rew_novel_left = torch.cat(rew_acc)
                self.logger.log_scalar("rew_novel_left_mean", rew_novel_left.mean().item())
                self.logger.log_scalar("rew_novel_left_max", rew_novel_left.max().item())
                self.logger.log_scalar("rew_novel_left_std", rew_novel_left.std(unbiased=False).item())
                self.logger.log_scalar("rew_novel_left_sum", rew_novel_left.sum().item())
                if desc_segments:
                    novelty_abs = rew_novel_left.abs().mean()
                    desc_abs = desc_concat.abs().mean().clamp_min(1e-8)
                    self.logger.log_scalar("reward_novelty_to_diayn_abs_ratio", (novelty_abs / desc_abs).item())
            self.logger.log_scalar("novelty_mean", rew_novel.mean())
            self.logger.log_scalar("novelty_max", rew_novel.max())
            self.logger.log_scalar("novelty_std", rew_novel.std())
            self.logger.log_scalar("novelty_sum", rew_novel.sum())

        return result

    def _novelty_mask(self, states, episode_lengths=None):
        if episode_lengths is not None and not isinstance(episode_lengths, (list, tuple)):
            episode_lengths = list(episode_lengths)
        if 'cube_positions' not in self.state_extractor.key_slices or 'cube_velocities' not in self.state_extractor.key_slices:
            return torch.ones(states.shape[0], dtype=torch.bool, device=states.device)

        cube_positions = self.state_extractor.extract(states, 'cubes_positions_centered')
        num_entries = cube_positions.shape[1] // 3 if cube_positions.dim() > 1 else 0
        if states.shape[0] == 0 or num_entries == 0:
            return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)
        cube_positions = cube_positions.view(states.shape[0], num_entries, 3)

        # Boundaries for XY plane
        xy = cube_positions[..., :2].abs()
        within_xy_mask = xy.view(states.shape[0], -1).le(self.novelty_xy_limit).all(dim=1)

        cube_velocities = self.state_extractor.extract(states, 'cube_velocities')
        velocities = cube_velocities.view(states.shape[0], num_entries, 3)
        vel_norm = torch.norm(velocities, dim=-1)
        stable_mask = vel_norm.le(self.novelty_velocity_threshold).all(dim=1)

        if self.logger is not None and states.shape[0] > 0:
            if within_xy_mask.float().mean().item() < 0.1:
                import pdb;pdb.set_trace()
            self.logger.log_scalar(
                "novelty_position_pass_ratio",
                within_xy_mask.float().mean().item()
            )
            self.logger.log_scalar(
                "novelty_velocity_pass_ratio",
                stable_mask.float().mean().item()
            )

        return within_xy_mask & stable_mask

    def compute_novelty_reward(self, states, episode_lengths=None):
        desc_input = self.get_descriminator_input(states, p_drop=0)
        mask = self._novelty_mask(states, episode_lengths)

        novelty_values = torch.zeros(states.shape[0], device=states.device, dtype=desc_input.dtype)
        masked_count = mask.sum().item()
        if masked_count > 0:
            masked_input = desc_input[mask].detach()
            novelty_subset = torch.from_numpy(self.novelty(masked_input)).to(novelty_values)
            novelty_values[mask] = novelty_subset

        if self.logger is not None:
            ratio = masked_count / max(1, states.shape[0])
            self.logger.log_scalar("novelty_mask_ratio", ratio)
            if states.shape[0] > 0:
                self._log_descriptor_stats(states, episode_lengths)

        return novelty_values.detach().cpu().numpy()

    def learn_from_episodes(self, episodes, num_minibatches=4):
        super().learn_from_episodes(episodes, num_minibatches=num_minibatches)
        if self.use_novelty:
            data_dict = self._extract_episode_data(episodes)
            filtered_dict = self.filter_short_episodes(data_dict, self.desc_discard_steps)
            if filtered_dict is not None and filtered_dict['states']:
                states = torch.cat(filtered_dict['states'], dim=0).to(self.device)
                if states.numel() > 0:
                    episode_lengths = [len(s) for s in filtered_dict['states']]
                    desc_input = self.get_descriminator_input(states, p_drop=0)
                    mask = self._novelty_mask(states, episode_lengths)
                    if mask.any():
                        self.novelty.update(desc_input[mask].detach())
                if states.shape[0] > 0:
                    self._log_descriptor_stats(states, episode_lengths)


class PPODNovel(_PPODNovelMixin, PPOD):
    def __init__(self, *args, novelty=None, **kwargs):
        super().__init__(*args, novelty=novelty, **kwargs)


class PPODNovelRunning(_PPODNovelMixin, PPODRunning):
    def __init__(self, *args, novelty=None, **kwargs):
        super().__init__(*args, novelty=novelty, **kwargs)
