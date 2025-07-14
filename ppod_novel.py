import torch
from ppod import PPOD
from util import RunningNorm


class PPODNovel(PPOD):
    def __init__(self, policy, value, sampler, obs_dim,
                 skill_dim=8, embedding_dim=8, continious=False,
                 policy_lr=0.0001, num_envs=8, discount=0.99,
                 device=torch.device('cpu'), logger=None, num_learning_epochs=4, discriminator=None,
                 discriminator_fields=None, desc_discard_steps=0, novelty=None, **kwargs):
        super().__init__(policy, value, sampler, obs_dim,
                 skill_dim=skill_dim, embedding_dim=embedding_dim, continious=continious,
                 policy_lr=policy_lr, num_envs=num_envs, discount=discount,
                 device=device, logger=logger, num_learning_epochs=num_learning_epochs, discriminator=discriminator,
                 discriminator_fields=discriminator_fields, desc_discard_steps=desc_discard_steps, **kwargs)
        self.novelty=novelty
        self.use_novelty = True

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
        if self.use_novelty:
            # 1d array
            rew_novel = self.compute_novelty_reward(states)
            rew_acc = []
            # assign rewards from concatentated list to per/episode
            s = 0
            for idx, el in enumerate(episode_lengths):
                item = torch.from_numpy(rew_novel[s: s + el][self.desc_discard_steps:]).to(result[idx])
                result[idx][self.desc_discard_steps:] += item
                rew_acc.append(item)
                s += el
            rew_novel_left = torch.cat(rew_acc)
            self.logger.log_scalar("novelty_mean", rew_novel.mean())
            self.logger.log_scalar("novelty_max", rew_novel.max())
            self.logger.log_scalar("novelty_std", rew_novel.std())
            self.logger.log_scalar("rew_novel_left_mean", rew_novel_left.mean())
            self.logger.log_scalar("rew_novel_left_max", rew_novel_left.max())
            self.logger.log_scalar("rew_novel_left_std", rew_novel_left.std())
            
        return result

    def compute_novelty_reward(self, states):
        desc_input = self.get_descriminator_input(states, p_drop=0)
        rew_novel = self.novelty(desc_input)
        return rew_novel

    def learn_from_episodes(self, episodes, num_minibatches=4):
        super().learn_from_episodes(episodes, num_minibatches=num_minibatches)
        if self.use_novelty:
            data_dict = self._extract_episode_data(episodes)
            filtered_dict = self.filter_short_episodes(data_dict, self.desc_discard_steps)
            states = filtered_dict['states']
            states = torch.cat(states, dim=0).to(self.device)
            self.novelty.update(self.get_descriminator_input(states, p_drop=0))
