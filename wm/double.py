import torch
import torch.nn.functional as F
from typing import Dict

from ppo import PPOBase
from wm_joint_agent import BaseWMOnPolicy, _WMEpisodePool
from util import EpisodeBatch


class DoubleAgent(BaseWMOnPolicy):
    """PPO two-policy agent with joint AC-CPC world-model updates."""
    def __init__(self, agent_high: PPOBase,
                 agent_low: PPOBase,
                 device,
                 logger,
                 **kwargs):
        self.agent_high = agent_high
        self.agent_low = agent_low
        self.agents = [self.agent_high, self.agent_low]
        self.low_fixed = kwargs.pop("low_fixed", False)
        self.high_warmup_goal_coef = float(kwargs.pop("high_warmup_goal_coef", 0.005))
        self.high_warmup_goal_epochs = int(kwargs.pop("high_warmup_goal_epochs", 1))
        self.achievability_warmup_updates = int(kwargs.pop("achievability_warmup_updates", 300))
        self.high_goal_max_fraction = float(kwargs.pop("high_goal_max_fraction", 0.5))
        self.high_achievability_coef = float(kwargs.pop("high_achievability_coef", 0.1))
        self.high_achievability_max_kl = float(kwargs.pop("high_achievability_max_kl", 0.05))
        self.high_achievability_lr_scale = float(kwargs.pop("high_achievability_lr_scale", 4.0))
        self.high_achievability_policy_updates = int(
            kwargs.pop("high_achievability_policy_updates", 1)
        )
        self.high_achievability_overfit = bool(kwargs.pop("high_achievability_overfit", False))
        self.virtual_horizon = int(kwargs.pop("virtual_horizon", 3))
        self.high_virtual_goal_ppo = kwargs.pop("high_virtual_goal_ppo", False)
        self.high_virtual_goal_imitation = kwargs.pop("high_virtual_goal_imitation", False)
        self.high_virtual_goal_imitation_coef = kwargs.pop(
            "high_virtual_goal_imitation_coef",
            1.0,
        )
        self.high_virtual_goal_imitation_epochs = kwargs.pop(
            "high_virtual_goal_imitation_epochs",
            1,
        )
        self.high_virtual_goal_imitation_all_targets = kwargs.pop(
            "high_virtual_goal_imitation_all_targets",
            False,
        )
        self.high_virtual_goal_imitation_fixed_dataset = kwargs.pop(
            "high_virtual_goal_imitation_fixed_dataset",
            False,
        )
        self.high_virtual_goal_imitation_replay_capacity = kwargs.pop(
            "high_virtual_goal_imitation_replay_capacity",
            0,
        )
        self.high_virtual_goal_imitation_replay_samples = kwargs.pop(
            "high_virtual_goal_imitation_replay_samples",
            self.num_envs,
        )
        self.high_virtual_goal_candidates = kwargs.pop("high_virtual_goal_candidates", 4)
        self.high_virtual_goal_attempts = kwargs.pop("high_virtual_goal_attempts", 2)
        self.high_virtual_goal_distance_penalty = kwargs.pop(
            "high_virtual_goal_distance_penalty",
            0.1,
        )
        self.low_virtual_cpc_error_threshold = float(
            kwargs.pop("low_virtual_cpc_error_threshold", 2.0)
        )
        if self.achievability_warmup_updates < 0:
            raise ValueError("achievability_warmup_updates must be >= 0")
        if not 0.0 <= self.high_goal_max_fraction <= 1.0:
            raise ValueError("high_goal_max_fraction must be in [0, 1]")
        if self.high_achievability_coef < 0.0:
            raise ValueError("high_achievability_coef must be >= 0")
        if self.high_achievability_max_kl <= 0.0:
            raise ValueError("high_achievability_max_kl must be positive")
        if self.high_achievability_lr_scale <= 0.0:
            raise ValueError("high_achievability_lr_scale must be positive")
        if self.high_achievability_policy_updates < 1:
            raise ValueError("high_achievability_policy_updates must be >= 1")
        if self.virtual_horizon < 1:
            raise ValueError("virtual_horizon must be >= 1")
        if self.high_virtual_goal_candidates < 2:
            raise ValueError("high_virtual_goal_candidates must be >= 2")
        if self.high_virtual_goal_attempts < 1:
            raise ValueError("high_virtual_goal_attempts must be >= 1")
        if self.high_virtual_goal_distance_penalty < 0.0:
            raise ValueError("high_virtual_goal_distance_penalty must be >= 0")
        if self.high_virtual_goal_ppo and self.high_virtual_goal_imitation:
            raise ValueError(
                "--high-virtual-goal-ppo and --high-virtual-goal-imitation are mutually exclusive"
            )
        if self.high_virtual_goal_imitation_coef <= 0.0:
            raise ValueError("high_virtual_goal_imitation_coef must be positive")
        if self.high_virtual_goal_imitation_epochs < 1:
            raise ValueError("high_virtual_goal_imitation_epochs must be >= 1")
        if self.high_virtual_goal_imitation_replay_capacity < 0:
            raise ValueError("high_virtual_goal_imitation_replay_capacity must be >= 0")
        if self.high_virtual_goal_imitation_replay_samples < 1:
            raise ValueError("high_virtual_goal_imitation_replay_samples must be >= 1")
        if self.high_virtual_goal_imitation_fixed_dataset \
                and self.high_virtual_goal_imitation_replay_capacity > 0:
            raise ValueError(
                "Fixed high imitation dataset and high imitation replay are mutually exclusive"
            )
        self._low_virtual_cpc_error_ema = None
        self._virtual_sfa_delta_mean = None
        self._virtual_sfa_delta_std_ema = None
        self._low_virtual_started = False
        self._achievability_updates = 0
        self._high_goal_enabled = None
        self._high_virtual_imitation_dataset = None
        self._high_virtual_imitation_pool = None
        if self.high_virtual_goal_imitation_replay_capacity > 0:
            self._high_virtual_imitation_pool = _WMEpisodePool(
                num_envs=self.high_virtual_goal_imitation_replay_samples,
                pool_size=self.high_virtual_goal_imitation_replay_capacity,
            )
        self.reset_high_agent_on_load = bool(kwargs.pop("reset_high_agent_on_load", False))
        BaseWMOnPolicy.__init__(self, device=device, logger=logger, **kwargs)
        if self.low_fixed:
            for module in (
                self.agent_low.policy,
                self.agent_low.value,
                self.agent_low.achievability_head,
            ):
                module.requires_grad_(False)
                module.eval()
        if self.high_achievability_overfit:
            if not self.wm_fixed:
                raise ValueError("high_achievability_overfit requires --wm-fixed")
            if self.high_goal_max_fraction != 0.0:
                raise ValueError("high_achievability_overfit requires --high-goal-max-fraction 0")
            if self.high_virtual_goal_ppo:
                raise ValueError("high_achievability_overfit is incompatible with --high-virtual-goal-ppo")
            if self.high_virtual_goal_imitation:
                raise ValueError(
                    "high_achievability_overfit is incompatible with "
                    "--high-virtual-goal-imitation"
                )
        self.br = False

    @property
    def device(self):
        return self.agent_low.device

    @property
    def num_envs(self):
        return self.agent_low.num_envs

    def episode_start(self):
        for agent in self.agents:
            agent.episode_start()
        self._prev_actions = None
        self._high_goal_enabled = None
        if hasattr(self.wm_model, "clear_cache"):
            self.wm_model.clear_cache()
        self._wm_pool.reset_episodes()

    @property
    def version(self):
        return self.agent_low.version

    @version.setter
    def version(self, value):
        self.agent_low.version = value
        self.agent_high.version = value

    def rl_add_reward(self, env_idx, reward):
        self.agent_low.add_reward(env_idx, reward)
        self.agent_high.add_reward(env_idx, reward)

    def process_dones(self, dones):
        self.agent_low.process_dones(dones)
        self.agent_high.process_dones(dones)

    def train(self):
        self.agent_high.policy.train()
        self.agent_high.value.train()
        if self.low_fixed:
            self.agent_low.policy.eval()
            self.agent_low.value.eval()
            self.agent_low.achievability_head.eval()
        else:
            self.agent_low.policy.train()
            self.agent_low.value.train()
            self.agent_low.achievability_head.train()

    def clear_completed(self):
        self.agent_low.clear_completed()
        self.agent_high.clear_completed()

    def rl_get_state_dict(self):
        return {
            "high": self.agent_high.get_state_dict(),
            "low": self.agent_low.get_state_dict(),
            "double": {
                "low_virtual_cpc_error_ema": self._low_virtual_cpc_error_ema,
                "virtual_sfa_delta_std_ema": self._virtual_sfa_delta_std_ema,
                "low_virtual_started": self._low_virtual_started,
                "achievability_updates": self._achievability_updates,
            },
        }

    def rl_load_state_dict(self, state_dict):
        self.agent_high.load_state_dict(state_dict["high"])
        self.agent_low.load_state_dict(state_dict["low"])
        double_state = state_dict.get("double")
        if double_state is None:
            self.logger.warn("Checkpoint has no double-agent phase state; restarting phase counters")
        else:
            self._low_virtual_cpc_error_ema = double_state["low_virtual_cpc_error_ema"]
            self._virtual_sfa_delta_std_ema = double_state["virtual_sfa_delta_std_ema"]
            self._low_virtual_started = bool(double_state["low_virtual_started"])
            self._achievability_updates = int(double_state["achievability_updates"])
        if self.reset_high_agent_on_load:
            self.reset_high_agent()

    @staticmethod
    def _reset_module_parameters(module):
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if reset is not None:
                reset()

    def reset_high_agent(self):
        self._reset_module_parameters(self.agent_high.policy)
        self._reset_module_parameters(self.agent_high.value)
        self.agent_high.policy_old.load_state_dict(self.agent_high.policy.state_dict())
        self.agent_high.create_optimizers()
        self.agent_high.episode_start()

    def should_learn(self):
        return ( self.agent_low.should_learn()
            and self.agent_high.should_learn()
            and len(self._wm_pool.get_completed_episodes()) >= self.num_envs
        )

    def _high_goal_mix_probability(self):
        if not self._high_goal_ready():
            return 0.0
        return self.high_goal_max_fraction

    def _high_goal_ready(self):
        return self._low_virtual_started and self._achievability_updates >= self.achievability_warmup_updates

    def _update_high_goal_enabled(self, episode_start, batch_size):
        reset_mask = torch.as_tensor(episode_start, dtype=torch.bool, device=self.device).view(-1)
        assert reset_mask.shape == (batch_size,), "episode_start must match environment batch"
        if self._high_goal_enabled is None:
            assert reset_mask.all(), "First high-goal mode update must start every environment"
            self._high_goal_enabled = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        assert self._high_goal_enabled.shape == (batch_size,)
        if reset_mask.any():
            probability = self._high_goal_mix_probability()
            sampled = torch.rand(int(reset_mask.sum()), device=self.device) < probability
            self._high_goal_enabled[reset_mask] = sampled
        return self._high_goal_enabled

    def _low_virtual_ready(self, cpc_error_after, key_padding_mask):
        cpc_error_mean = float(self._masked_mean(cpc_error_after, ~key_padding_mask))
        if self._low_virtual_cpc_error_ema is None:
            self._low_virtual_cpc_error_ema = cpc_error_mean
        else:
            self._low_virtual_cpc_error_ema = (
                0.95 * self._low_virtual_cpc_error_ema + 0.05 * cpc_error_mean
            )
        if not self._low_virtual_started:
            self._low_virtual_started = (
                self._low_virtual_cpc_error_ema <= self.low_virtual_cpc_error_threshold
            )
        return self._low_virtual_started, cpc_error_mean, self._low_virtual_cpc_error_ema

    def _squeeze_time(self, x):
        if x.dim() == 3:
            assert x.shape[1] == 1, f"Expected online one-step feature, got time dim {x.shape[1]}"
            return x[:, -1, :]
        return x

    def _achievability_success_threshold(self):
        assert self._virtual_sfa_delta_mean is not None, \
            "Achievability success threshold requires initialized virtual SFA delta statistics"
        return float(self._virtual_sfa_delta_mean)

    def _cpc_surprise(self, cpc, episode_start):
        prev_h, prev_action = self.agent_high.prediction_context()
        if prev_h is None:
            return cpc.new_zeros((cpc.shape[0], 1))

        assert prev_action.dim() == 1, f"Expected previous discrete action [B], got {tuple(prev_action.shape)}"
        assert prev_h.shape[0] == cpc.shape[0], "previous h and current cpc batch sizes must match"
        assert prev_action.shape[0] == cpc.shape[0], "previous action and current cpc batch sizes must match"

        prev_action_cont = self.action_cont_table[prev_action.to(self.action_cont_table.device).long()]
        with torch.no_grad():
            pred_cpc = self.wm_model.predict_next_contrastive_emb(prev_h, prev_action_cont)
            surprise = torch.linalg.vector_norm(pred_cpc - cpc, dim=-1, keepdim=True)

        reset_mask = torch.as_tensor(episode_start, dtype=torch.bool, device=surprise.device).view(-1)
        assert reset_mask.numel() == cpc.shape[0], "episode_start must match batch size"
        if reset_mask.any():
            surprise = surprise.clone()
            surprise[reset_mask] = 0.0
        return surprise.detach()

    def _replay_wm_to_step(self, episodes, step_idx):
        if (step_idx < 0).any():
            raise ValueError(f"step_idx must be >= 0, got {step_idx}")

        replay_episodes = episodes
        batch_size = len(replay_episodes)
        live_cache = self.wm_model.get_cache_state()
        was_training = self.wm_model.training
        replay_out = None
        replay_cache = None

        try:
            self.wm_model.eval()
            with torch.no_grad():
                sensor = [torch.stack([
                        torch.as_tensor(episode[current_step][0]["sensor"], dtype=torch.float32)
                        for current_step in range(step_idx[i] + 1)]).to(self.device)
                    for i, episode in enumerate(replay_episodes)]

                action_indices = [torch.stack([
                        torch.as_tensor(episode[current_step][1], dtype=torch.long).view(-1)[0]
                        for current_step in range(step_idx[i] + 1)]).to(self.device)
                    for i, episode in enumerate(replay_episodes)]

                actions = [self.action_idx_to_val(ai).to(sensor[0]) for ai in action_indices]
                prefix_batch = EpisodeBatch({
                    "actions": actions,
                    "sensor": sensor,
                }).to(self.device)
                padded, key_padding_mask, lengths = prefix_batch.left_pad()
                row_indices = torch.arange(key_padding_mask.size(0), device=lengths.device)
                cache_reset = key_padding_mask.clone()
                cache_reset[row_indices, -lengths] = True

                replay_out = self.wm_model.prime_cache({
                    "sensor": padded["sensor"],
                    "actions": padded["actions"],
                }, episode_start=cache_reset)

                replay_cache = self.wm_model.get_cache_state()
        finally:
            self.wm_model.set_cache_state(live_cache)
            self.wm_model.train(was_training)

        assert replay_out is not None and replay_cache is not None
        episode_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)
        return replay_out, replay_cache, episode_indices

    def _collect_low_virtual_hindsight(
        self,
        episodes,
        step_idx,
        num_rollouts=4,
        horizon=3,
        max_prediction_scale=None,
        step_effect_threshold=0.1,
    ):
        if num_rollouts < 1:
            raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not 0.0 <= step_effect_threshold <= 1.0:
            raise ValueError(f"step_effect_threshold must be in [0, 1], got {step_effect_threshold}")

        replay_out, replay_cache, _ = self._replay_wm_to_step(episodes, step_idx)
        replay_h = replay_out["state_last"]
        replay_cpc = self._squeeze_time(replay_out["aux"]["contrastive_tgt_emb"])
        replay_sfa = self._squeeze_time(replay_out["aux"]["sfa"])
        replay_sensor_latent = self._squeeze_time(replay_out["aux"]["sensor_latent"])
        batch_size = replay_h.shape[0]
        branch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(num_rollouts)
        num_virtual_episodes = branch_indices.numel()
        if num_virtual_episodes < 2:
            raise ValueError("virtual hindsight requires at least two rollout branches for negative goals")

        h = replay_h[branch_indices]
        cpc = replay_cpc[branch_indices]
        sfa = replay_sfa[branch_indices]
        sensor_latent = replay_sensor_latent[branch_indices]
        base = torch.cat([h, cpc, sfa], dim=-1)
        live_cache = self.wm_model.get_cache_state()
        was_training = self.wm_model.training
        base_steps = []
        sfa_steps = []
        cpc_steps = []
        sensor_latent_steps = []
        action_steps = []
        prediction_scales = []

        try:
            self.wm_model.eval()
            branch_cache = self.wm_model.index_cache_state(replay_cache, branch_indices)
            self.wm_model.set_cache_state(branch_cache)
            with torch.no_grad():
                for _ in range(horizon):
                    goal = torch.zeros_like(sfa)
                    goal_valid = sfa.new_zeros((num_virtual_episodes, 1))
                    low_state = self._build_low_state(base, sfa, goal, goal_valid)
                    policy = self.agent_low.get_policy_for_action()
                    policy_params = self.agent_low.sampler.policy_params(policy, low_state)
                    actions, _, _ = self.agent_low.sampler(policy_params)
                    actions = actions.to(torch.long)

                    base_steps.append(base)
                    sfa_steps.append(sfa)
                    cpc_steps.append(cpc)
                    sensor_latent_steps.append(sensor_latent)
                    action_steps.append(actions)

                    action_values = self.action_idx_to_val(actions).to(h)
                    cpc_dist = self.wm_model.predict_next_contrastive_dist(h, action_values)
                    pred_cpc = cpc_dist.mean
                    scale = cpc_dist.base_dist.scale[..., 0]
                    prediction_scales.append(scale)
                    pred_sensor_latent = self.wm_model.cpc_sensor_latent_head(pred_cpc)
                    next_out = self.wm_model(
                        {
                            "sensor_latent": pred_sensor_latent,
                            "actions": None,
                            "prev_actions": action_values,
                        },
                        episode_start=torch.zeros(
                            num_virtual_episodes,
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    )
                    h = next_out["state_last"]
                    cpc = self._squeeze_time(next_out["aux"]["contrastive_tgt_emb"])
                    sfa = self._squeeze_time(next_out["aux"]["sfa"])
                    sensor_latent = self._squeeze_time(next_out["aux"]["sensor_latent"])
                    base = torch.cat([h, cpc, sfa], dim=-1)
        finally:
            self.wm_model.set_cache_state(live_cache)
            self.wm_model.train(was_training)

        base_seq = torch.stack(base_steps, dim=1)
        sfa_seq = torch.stack(sfa_steps, dim=1)
        cpc_seq = torch.stack(cpc_steps, dim=1)
        sensor_latent_seq = torch.stack(sensor_latent_steps, dim=1)
        actions_seq = torch.stack(action_steps, dim=1)
        scale_seq = torch.stack(prediction_scales, dim=1)
        goal = sfa.unsqueeze(1).expand(-1, horizon, -1)

        goal_distances = torch.cdist(sfa, sfa)
        negative_goal = torch.empty_like(goal)
        has_valid_negative = torch.zeros(
            (num_virtual_episodes, horizon),
            dtype=torch.bool,
            device=self.device,
        )
        for rollout_step in range(horizon):
            action_mismatch = actions_seq[:, rollout_step, None] != actions_seq[None, :, rollout_step]
            candidates = action_mismatch & (goal_distances > 1e-8)
            has_valid_negative[:, rollout_step] = candidates.any(dim=1)
            scores = goal_distances.masked_fill(~candidates, float("-inf"))
            negative_indices = scores.argmax(dim=1)
            negative_goal[:, rollout_step] = sfa[negative_indices]

        goal_valid = sfa.new_ones((num_virtual_episodes, horizon, 1))
        states = self._build_low_state(base_seq, sfa_seq, goal, goal_valid)
        negative_states = self._build_low_state(base_seq, sfa_seq, negative_goal, goal_valid)

        next_sfa_seq = torch.cat([sfa_seq[:, 1:], sfa.unsqueeze(1)], dim=1)
        next_cpc_seq = torch.cat([cpc_seq[:, 1:], cpc.unsqueeze(1)], dim=1)
        next_sensor_latent_seq = torch.cat(
            [sensor_latent_seq[:, 1:], sensor_latent.unsqueeze(1)],
            dim=1,
        )
        sfa_delta = torch.linalg.vector_norm(next_sfa_seq - sfa_seq, dim=-1)
        cpc_delta = torch.linalg.vector_norm(next_cpc_seq - cpc_seq, dim=-1)
        sensor_delta = torch.linalg.vector_norm(next_sensor_latent_seq - sensor_latent_seq, dim=-1)
        sfa_delta_mean = sfa_delta.mean()
        sfa_delta_std = sfa_delta.std(unbiased=False)
        cpc_delta_std = cpc_delta.std(unbiased=False)
        sensor_delta_std = sensor_delta.std(unbiased=False)
        if not torch.isfinite(sfa_delta_mean) or sfa_delta_mean <= 0.0:
            raise ValueError(f"Expected positive finite virtual SFA delta mean, got {sfa_delta_mean}")
        if not torch.isfinite(sfa_delta_std) or sfa_delta_std <= 0.0:
            raise ValueError(f"Expected positive finite virtual SFA delta std, got {sfa_delta_std}")
        self._virtual_sfa_delta_mean = float(sfa_delta_mean.detach())
        sfa_delta_std_value = float(sfa_delta_std.detach())
        if self._virtual_sfa_delta_std_ema is None:
            self._virtual_sfa_delta_std_ema = sfa_delta_std_value
        else:
            self._virtual_sfa_delta_std_ema = (
                0.95 * self._virtual_sfa_delta_std_ema + 0.05 * sfa_delta_std_value
            )
        normalized_sfa_delta = (sfa_delta / (sfa_delta_std + 1e-8)).clamp(0.0, 1.0)
        normalized_cpc_delta = (cpc_delta / (cpc_delta_std + 1e-8)).clamp(0.0, 1.0)
        normalized_sensor_delta = (sensor_delta / (sensor_delta_std + 1e-8)).clamp(0.0, 1.0)
        step_effect = torch.stack(
            [normalized_sfa_delta, normalized_cpc_delta, normalized_sensor_delta],
            dim=-1,
        ).max(dim=-1).values
        step_valid = step_effect > float(step_effect_threshold)

        episode_goal_distance = torch.linalg.vector_norm(sfa - sfa_seq[:, 0], dim=-1)
        episode_effect = (episode_goal_distance / (sfa_delta_std + 1e-8)).clamp(0.0, 1.0)
        valid = torch.isfinite(scale_seq).all(dim=1)
        if max_prediction_scale is not None:
            valid = valid & (scale_seq.max(dim=1).values <= float(max_prediction_scale))
        weights = (
            valid.to(sfa).unsqueeze(1)
            * has_valid_negative.to(sfa)
            * episode_effect.unsqueeze(1)
            * step_valid.to(sfa)
        )

        self.logger.log_scalar("low/virtual_episode_count", float(num_virtual_episodes))
        self.logger.log_scalar("low/virtual_prediction_scale_mean", scale_seq.mean())
        self.logger.log_scalar("low/virtual_valid_frac", valid.to(torch.float32).mean())
        self.logger.log_scalar("low/virtual_negative_valid_frac", has_valid_negative.to(torch.float32).mean())
        self.logger.log_scalar("low/virtual_goal_distance", episode_goal_distance.mean())
        self.logger.log_scalar("low/virtual_episode_effect_mean", episode_effect.mean())
        self.logger.log_scalar("low/virtual_step_effect_mean", step_effect.mean())
        self.logger.log_scalar("low/virtual_step_valid_frac", step_valid.to(torch.float32).mean())
        self.logger.log_scalar("low/virtual_weight_mean", weights.mean())
        self.logger.log_scalar("low/virtual_sfa_delta_mean", sfa_delta_mean)
        self.logger.log_scalar("low/virtual_sfa_delta_std", sfa_delta_std)
        self.logger.log_scalar("low/virtual_sfa_delta_std_ema", self._virtual_sfa_delta_std_ema)
        self.logger.log_scalar("low/virtual_cpc_delta_mean", cpc_delta.mean())
        self.logger.log_scalar("low/virtual_cpc_delta_std", cpc_delta_std)
        self.logger.log_scalar("low/virtual_sensor_delta_mean", sensor_delta.mean())
        self.logger.log_scalar("low/virtual_sensor_delta_std", sensor_delta_std)

        terminal_sfa = sfa.reshape(batch_size, num_rollouts, sfa.shape[-1])
        rollout_goal_distances = episode_goal_distance.reshape(batch_size, num_rollouts)
        rollout_valid = valid.reshape(batch_size, num_rollouts)
        assert rollout_valid.any(dim=1).all(), \
            "Every source state must have at least one valid virtual rollout"
        best_scores = rollout_goal_distances.masked_fill(~rollout_valid, float("-inf"))
        best_indices = best_scores.argmax(dim=1)
        row_indices = torch.arange(batch_size, device=self.device)
        best_goals = terminal_sfa[row_indices, best_indices]
        best_goal_distances = rollout_goal_distances[row_indices, best_indices]
        candidate_indices = torch.randint(num_rollouts, (batch_size,), device=self.device)
        candidate_goals = terminal_sfa[row_indices, candidate_indices]
        source_sfa = replay_sfa
        candidate_goal_distance = torch.linalg.vector_norm(candidate_goals - source_sfa, dim=-1)
        self.logger.log_scalar("low/virtual_candidate_goal_distance", candidate_goal_distance.mean())
        self.logger.log_scalar(
            "high/virtual_imitation/target_distance",
            best_goal_distances.mean(),
        )

        hindsight = (
            [s.detach().cpu() for s in states],
            [a_seq.detach().cpu() for a_seq in actions_seq],
            [w.detach().cpu() for w in weights],
            [neg.detach().cpu() for neg in negative_states],
        )
        return (
            hindsight,
            candidate_goals.detach(),
            best_goals.detach(),
            terminal_sfa.detach(),
        )

    def _get_high_virtual_imitation_dataset(
        self,
        high_episodes,
        step_indices,
        target_goals,
    ):
        """Return a fixed dataset for the imitation overfitting test."""
        if not self.high_virtual_goal_imitation_fixed_dataset:
            return high_episodes, step_indices, target_goals
        if self._high_virtual_imitation_dataset is None:
            frozen_episodes = [
                [(step[0].detach().cpu(),) for step in episode]
                for episode in high_episodes
            ]
            self._high_virtual_imitation_dataset = (
                frozen_episodes,
                step_indices.detach().cpu(),
                target_goals.detach().cpu(),
            )
        return self._high_virtual_imitation_dataset

    def _get_high_virtual_imitation_replay(
        self,
        high_episodes,
        step_indices,
        target_goals,
    ):
        if self._high_virtual_imitation_pool is None:
            return high_episodes, step_indices, target_goals
        assert len(high_episodes) == len(step_indices) == target_goals.shape[0], \
            "High imitation replay inputs must have one record per source episode"

        for episode, step_idx, targets in zip(high_episodes, step_indices, target_goals):
            source_step = int(step_idx.item())
            assert 0 <= source_step < len(episode), \
                f"High imitation source step {source_step} outside episode length {len(episode)}"
            prefix = [
                (step[0].detach().cpu(),)
                for step in episode[:source_step + 1]
            ]
            record = (prefix, targets.detach().cpu())
            self._high_virtual_imitation_pool.add_completed_episode(record)

        records = self._high_virtual_imitation_pool.get_train_episodes()
        assert records, "High imitation replay must return records after insertion"
        replay_episodes = [record[0] for record in records]
        replay_steps = torch.tensor(
            [len(episode) - 1 for episode in replay_episodes],
            dtype=torch.long,
        )
        replay_targets = torch.stack([record[1] for record in records], dim=0)
        return replay_episodes, replay_steps, replay_targets

    def _sample_temporally_distant_steps(self, episodes, source_steps, min_delta=30):
        source_steps = torch.as_tensor(source_steps, dtype=torch.long)
        assert source_steps.shape == (len(episodes),), \
            "Expected one source step per episode"
        distant_steps = []
        for episode_idx, (episode, source_step) in enumerate(zip(episodes, source_steps)):
            source_step = int(source_step.item())
            episode_steps = torch.arange(len(episode), dtype=torch.long)
            valid_steps = episode_steps[(episode_steps - source_step).abs() >= int(min_delta)]
            assert valid_steps.numel() > 0, \
                f"Episode {episode_idx} has no step with |delta_t| >= {min_delta} from {source_step}"
            sampled_idx = torch.randint(valid_steps.numel(), (1,)).item()
            distant_steps.append(valid_steps[sampled_idx])
        return torch.stack(distant_steps)

    def _rollout_low_virtual_goals(
        self,
        wm_episodes,
        step_idx,
        goals,
        num_attempts=4,
        horizon=3,
    ):
        if num_attempts < 1:
            raise ValueError(f"num_attempts must be >= 1, got {num_attempts}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if self._virtual_sfa_delta_std_ema is None:
            raise RuntimeError("Virtual hindsight must initialize the SFA delta threshold first")
        if not hasattr(self.wm_model, "cpc_sensor_latent_head"):
            raise TypeError("virtual achievability rollouts require cpc_sensor_latent_head")

        replay_out, replay_cache, _ = self._replay_wm_to_step(wm_episodes, step_idx)

        replay_h = replay_out["state_last"]
        replay_cpc = self._squeeze_time(replay_out["aux"]["contrastive_tgt_emb"])
        replay_sfa = self._squeeze_time(replay_out["aux"]["sfa"])
        batch_size = replay_h.shape[0]
        goals = goals.to(self.device)
        assert goals.dim() == 3 and goals.shape[0] == batch_size \
            and goals.shape[2] == replay_sfa.shape[-1], \
            f"Expected achievability goals [B,C,G], got {goals.shape}"
        goal_count = goals.shape[1]
        assert torch.isfinite(goals).all(), "Non-finite virtual achievability goals"

        goal_source_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, goal_count).reshape(-1)
        flat_goals = goals.reshape(batch_size * goal_count, goals.shape[-1])
        attempt_goal_indices = torch.arange(flat_goals.shape[0], device=self.device).repeat_interleave(num_attempts)
        branch_indices = goal_source_indices[attempt_goal_indices]
        goals = flat_goals[attempt_goal_indices]
        h = replay_h[branch_indices]
        cpc = replay_cpc[branch_indices]
        sfa = replay_sfa[branch_indices]
        base = torch.cat([h, cpc, sfa], dim=-1)
        goal_valid = sfa.new_ones((branch_indices.numel(), 1))
        source_states = self._build_low_state(base, sfa, goals, goal_valid)

        live_cache = self.wm_model.get_cache_state()
        was_training = self.wm_model.training
        future_sfa_steps = []
        try:
            self.wm_model.eval()
            branch_cache = self.wm_model.index_cache_state(replay_cache, branch_indices)
            self.wm_model.set_cache_state(branch_cache)
            with torch.no_grad():
                for _ in range(horizon):
                    low_state = self._build_low_state(base, sfa, goals, goal_valid)
                    policy = self.agent_low.get_policy_for_action()
                    policy_params = self.agent_low.sampler.policy_params(policy, low_state)
                    actions, _, _ = self.agent_low.sampler(policy_params)
                    actions = actions.to(torch.long)

                    action_values = self.action_idx_to_val(actions).to(h)
                    pred_cpc = self.wm_model.predict_next_contrastive_dist(h, action_values).mean
                    pred_sensor_latent = self.wm_model.cpc_sensor_latent_head(pred_cpc)
                    next_out = self.wm_model(
                        {
                            "sensor_latent": pred_sensor_latent,
                            "actions": None,
                            "prev_actions": action_values,
                        },
                        episode_start=torch.zeros(
                            branch_indices.numel(),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    )
                    h = next_out["state_last"]
                    cpc = self._squeeze_time(next_out["aux"]["contrastive_tgt_emb"])
                    sfa = self._squeeze_time(next_out["aux"]["sfa"])
                    future_sfa_steps.append(sfa)
                    base = torch.cat([h, cpc, sfa], dim=-1)
        finally:
            self.wm_model.set_cache_state(live_cache)
            self.wm_model.train(was_training)

        future_sfa = torch.stack(future_sfa_steps, dim=1)
        assert future_sfa.shape == (branch_indices.numel(), horizon, goals.shape[-1])
        distances = torch.linalg.vector_norm(future_sfa - goals.unsqueeze(1), dim=-1)
        min_distances = distances.min(dim=1).values
        success_threshold = self._achievability_success_threshold()
        achieved = min_distances <= success_threshold

        goal_distances = torch.linalg.vector_norm(
            flat_goals.reshape(batch_size, goal_count, -1) - replay_sfa.unsqueeze(1),
            dim=-1,
        )
        min_distances_by_type = min_distances.reshape(batch_size, goal_count, num_attempts)
        achieved_by_type = achieved.reshape(batch_size, goal_count, num_attempts)
        source_states_by_type = source_states.reshape(
            batch_size,
            goal_count,
            num_attempts,
            source_states.shape[-1],
        )[:, :, 0, :]
        return (
            source_states_by_type,
            goal_distances,
            min_distances_by_type,
            achieved_by_type,
            success_threshold,
        )

    def _collect_low_virtual_achievability(
        self,
        wm_episodes,
        step_idx,
        goals,
        goal_names,
        num_attempts=4,
        horizon=3,
    ):
        goal_count = goals.shape[1]
        assert len(goal_names) == goal_count, \
            "Achievability goal names must match the goal-type dimension"
        assert len(set(goal_names)) == goal_count, "Achievability goal names must be unique"
        (
            source_states_by_type,
            goal_distances,
            min_distances_by_type,
            achieved_by_type,
            success_threshold,
        ) = self._rollout_low_virtual_goals(
            wm_episodes,
            step_idx,
            goals,
            num_attempts=num_attempts,
            horizon=horizon,
        )
        batch_size = goals.shape[0]
        best_distances_by_type = min_distances_by_type.min(dim=-1).values
        min_distances = best_distances_by_type.reshape(-1)
        achieved = achieved_by_type.reshape(-1)

        self.logger.log_scalar("low/achievability_virtual/success_threshold", success_threshold)
        self.logger.log_scalar("low/achievability_virtual/min_distance_mean", min_distances.mean())
        self.logger.log_scalar("low/achievability_virtual/min_distance_std", min_distances.std(unbiased=False))
        self.logger.log_scalar("low/achievability_virtual/success_rate", achieved.float().mean())
        with torch.no_grad():
            achievability_prediction = self.agent_low.achievability_head(
                source_states_by_type.reshape(batch_size * goal_count, -1)
            ).reshape(batch_size, goal_count, 3)
            predicted_distance = achievability_prediction[..., 0].exp()
            predicted_success = achievability_prediction[..., 2].sigmoid()
        for goal_idx, goal_name in enumerate(goal_names):
            target_distance = best_distances_by_type[:, goal_idx]
            target_success = achieved_by_type[:, goal_idx].float().mean(dim=-1)
            self.logger.log_scalar(
                f"low/achievability_virtual/{goal_name}_goal_distance",
                goal_distances[:, goal_idx].mean(),
            )
            self.logger.log_scalar(
                f"low/achievability_virtual/{goal_name}_min_distance",
                target_distance.mean(),
            )
            self.logger.log_scalar(
                f"low/achievability_virtual/{goal_name}_success_rate",
                target_success.mean(),
            )
            self.logger.log_scalar(
                f"low/achievability_virtual/{goal_name}_predicted_success_rate",
                predicted_success[:, goal_idx].mean(),
            )
            self.logger.log_scalar(
                f"low/achievability_virtual/{goal_name}_success_rate_mae",
                (predicted_success[:, goal_idx] - target_success).abs().mean(),
            )
            self.logger.log_scalar(
                f"low/achievability_virtual/{goal_name}_distance_mae",
                (predicted_distance[:, goal_idx] - target_distance).abs().mean(),
            )

        source_states_by_goal = source_states_by_type.reshape(batch_size * goal_count, -1)
        attempt_distances_by_goal = min_distances_by_type.reshape(
            batch_size * goal_count,
            num_attempts,
        )
        success_rates_by_goal = achieved_by_type.float().mean(dim=-1).reshape(
            batch_size * goal_count,
        )
        return (
            source_states_by_goal.detach().cpu(),
            attempt_distances_by_goal.detach().cpu(),
            success_rates_by_goal.detach().cpu(),
        )

    def _collect_high_virtual_goal_rewards(self, wm_episodes, step_idx, goals):
        _, initial_distances, min_distances, _, _ = self._rollout_low_virtual_goals(
            wm_episodes,
            step_idx,
            goals,
            num_attempts=self.high_virtual_goal_attempts,
            horizon=self.virtual_horizon,
        )
        final_distances = min_distances.mean(dim=-1)
        progress = initial_distances - final_distances
        rewards = progress - self.high_virtual_goal_distance_penalty * final_distances
        assert rewards.shape == goals.shape[:2], \
            f"Expected virtual high rewards [B,N], got {rewards.shape}"
        assert torch.isfinite(rewards).all(), "Non-finite virtual high goal rewards"
        self.logger.log_scalar(
            "high/virtual_goal/initial_distance",
            initial_distances.mean(),
        )
        self.logger.log_scalar(
            "high/virtual_goal/final_distance",
            final_distances.mean(),
        )
        return rewards.detach()

    def _low_goal_distances(self, episode):
        states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
        goal_dim = self.wm_model.contrastive_dim
        goal_delta = states[:, -(goal_dim + 1):-1]
        return torch.linalg.vector_norm(goal_delta, dim=-1)

    def _low_goal_transition_distances(self, episode):
        states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
        goal_dim = self.wm_model.contrastive_dim
        base, sfa = self._extract_low_base_sfa(states)
        del base
        goals = states[:, -(2 * goal_dim + 1):-(goal_dim + 1)]
        assert goals.shape == sfa.shape, "Failed to align low-level goals with SFA"

        current_distances = torch.linalg.vector_norm(goals - sfa, dim=-1)
        next_distances = current_distances.clone()
        if states.shape[0] > 1:
            # Keep goal[t] fixed when measuring the result of action[t]. Using
            # the stored distance at t+1 would compare against a newly switched
            # goal and corrupt progress exactly at option boundaries.
            next_distances[:-1] = torch.linalg.vector_norm(
                goals[:-1] - sfa[1:],
                dim=-1,
            )
        return current_distances, next_distances

    def _log_real_achievability_ranking(
        self,
        episodes,
        positive_offset=3,
        negative_offset=20,
    ):
        assert 0 < positive_offset < negative_offset
        positive_states = []
        negative_states = []
        success_threshold = self._achievability_success_threshold()

        for episode in episodes:
            states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
            if states.shape[0] <= negative_offset:
                continue
            base, sfa = self._extract_low_base_sfa(states)
            source_count = states.shape[0] - negative_offset
            source_base = base[:source_count]
            source_sfa = sfa[:source_count]
            positive_goal = sfa[positive_offset:positive_offset + source_count]
            negative_goal = sfa[negative_offset:negative_offset + source_count]

            near_future = torch.stack(
                [sfa[offset:offset + source_count] for offset in range(1, positive_offset + 1)],
                dim=1,
            )
            negative_near_distance = torch.linalg.vector_norm(
                near_future - negative_goal.unsqueeze(1),
                dim=-1,
            ).min(dim=1).values
            valid = negative_near_distance > success_threshold
            if not valid.any():
                continue

            goal_valid = source_sfa.new_ones((int(valid.sum()), 1))
            positive_states.append(self._build_low_state(
                source_base[valid],
                source_sfa[valid],
                positive_goal[valid],
                goal_valid,
            ))
            negative_states.append(self._build_low_state(
                source_base[valid],
                source_sfa[valid],
                negative_goal[valid],
                goal_valid,
            ))

        if not positive_states:
            return

        positive_states = torch.cat(positive_states, dim=0)
        negative_states = torch.cat(negative_states, dim=0)
        with torch.no_grad():
            positive_prediction = self.agent_low.achievability_head(positive_states)
            negative_prediction = self.agent_low.achievability_head(negative_states)
            assert positive_prediction.shape == negative_prediction.shape == (positive_states.shape[0], 3), \
                "Expected paired achievability predictions [N,3]"
            positive_distance = positive_prediction[:, 0].exp()
            negative_distance = negative_prediction[:, 0].exp()
            positive_success = positive_prediction[:, 2].sigmoid()
            negative_success = negative_prediction[:, 2].sigmoid()
            distance_margin = negative_distance - positive_distance
            success_margin = positive_success - negative_success
            distance_rank_accuracy = (
                (distance_margin > 0.0).to(torch.float32)
                + 0.5 * (distance_margin == 0.0).to(torch.float32)
            ).mean()
            success_rank_accuracy = (
                (success_margin > 0.0).to(torch.float32)
                + 0.5 * (success_margin == 0.0).to(torch.float32)
            ).mean()

        self.logger.log_scalar(
            "achievability/real_distance_rank_accuracy",
            distance_rank_accuracy,
        )
        self.logger.log_scalar(
            "achievability/real_success_rank_accuracy",
            success_rank_accuracy,
        )
        self.logger.log_scalar(
            "achievability/real_distance_rank_margin",
            distance_margin.mean(),
        )
        self.logger.log_scalar(
            "achievability/real_success_rank_margin",
            success_margin.mean(),
        )

    def _extract_low_base_sfa(self, states):
        assert states.dim() == 2, f"Expected low states [T,D], got {tuple(states.shape)}"
        goal_dim = self.wm_model.contrastive_dim
        base = states[:, :-(2 * goal_dim + 1)]
        sfa = base[:, -goal_dim:]
        assert sfa.shape[-1] == goal_dim, "failed to extract SFA from low-level state"
        return base, sfa

    def _build_low_state(self, base, sfa, goal, goal_valid):
        assert goal_valid.shape == (*goal.shape[:-1], 1), "goal_valid must be [...,1]"
        return torch.cat([base, goal, goal - sfa, goal_valid.to(goal)], dim=-1)

    def _compute_low_actual_rewards(self, episode, intrinsic_rewards=None, goal_coef=1.0, intrinsic_coef=0.1):
        current_distances, next_distances = self._low_goal_transition_distances(episode)
        goal_rewards = torch.zeros_like(current_distances)
        if current_distances.shape[0] > 1:
            goal_rewards[:-1] = current_distances[:-1] - next_distances[:-1]
        rewards = float(goal_coef) * goal_rewards
        if intrinsic_rewards is not None:
            intrinsic_bonus = torch.zeros_like(rewards)
            valid_steps = min(rewards.shape[0], intrinsic_rewards.shape[0])
            intrinsic_bonus[:valid_steps] = intrinsic_rewards[:valid_steps].to(rewards).clamp_min(0.0)
            rewards = rewards + float(intrinsic_coef) * intrinsic_bonus
        return rewards.detach().to("cpu")

    def _episode_goal_enabled(self, episode):
        goal_valid = torch.stack([step[0][-1] for step in episode], dim=0).to(torch.float32).view(-1)
        assert ((goal_valid == 0.0) | (goal_valid == 1.0)).all(), \
            "Real low-level goal_valid must be binary"
        assert (goal_valid == goal_valid[0]).all(), \
            "Real low-level goal mode must remain constant within an episode"
        return bool(goal_valid[0].item())

    def _overwrite_low_rewards(self, episodes, rewards_per_episode):
        assert len(episodes) == len(rewards_per_episode), "episode/reward count mismatch"
        for episode, rewards in zip(episodes, rewards_per_episode):
            assert len(episode) == rewards.shape[0], "episode/reward length mismatch"
            for step_idx, reward in enumerate(rewards):
                episode[step_idx] = (*episode[step_idx][:4], reward)
        self.logger.log_scalar(
            "low/actual_reward_mean",
            torch.cat(rewards_per_episode).mean() if rewards_per_episode else torch.tensor(0.0),
        )
        goal_episodes = [episode for episode in episodes if self._episode_goal_enabled(episode)]
        if goal_episodes:
            self.logger.log_scalar(
                "low/goal_distance_mean",
                torch.cat([
                    self._low_goal_distances(episode).detach().cpu()
                    for episode in goal_episodes
                ]).mean(),
            )

    def _log_low_reward_components(self, low_episodes, intrinsic_rewards, goal_enabled):
        assert len(low_episodes) == len(goal_enabled), "Low episode/mode count mismatch"
        goal_rewards = []
        intrinsic_bonus = []
        for ep_idx, (episode, enabled) in enumerate(zip(low_episodes, goal_enabled)):
            raw_goal_reward = self._compute_low_actual_rewards(episode, intrinsic_rewards=None)
            goal_rewards.append(raw_goal_reward if enabled else torch.zeros_like(raw_goal_reward))
            bonus = torch.zeros(len(episode), dtype=torch.float32, device=self.device)
            valid_steps = min(len(episode), intrinsic_rewards.shape[1])
            bonus[:valid_steps] = intrinsic_rewards[ep_idx, :valid_steps].to(bonus).clamp_min(0.0)
            intrinsic_coef = 0.1 if enabled else 1.0
            intrinsic_bonus.append((intrinsic_coef * bonus).detach().cpu())
        self.logger.log_scalar(
            "low/goal_progress_reward_mean",
            torch.cat(goal_rewards).mean() if goal_rewards else torch.tensor(0.0),
        )
        self.logger.log_scalar(
            "low/intrinsic_bonus_mean",
            torch.cat(intrinsic_bonus).mean() if intrinsic_bonus else torch.tensor(0.0),
        )
        self.logger.log_scalar(
            "low/goal_valid_mean",
            torch.tensor(goal_enabled, dtype=torch.float32).mean() if goal_enabled else torch.tensor(0.0),
        )

    def _compute_low_hindsight(self, low_episodes, intrinsic_rewards, horizon=5):
        assert len(low_episodes) <= intrinsic_rewards.shape[0], "low episodes and intrinsic rewards are misaligned"
        states_per_episode = []
        negative_states_per_episode = []
        actions_per_episode = []
        weights_per_episode = []
        target_distances = []
        progress_terms = []
        intrinsic_terms = []
        batch_sfa = [
            self._extract_low_base_sfa(torch.stack([step[0] for step in episode], dim=0).to(self.device))[1]
            for episode in low_episodes
        ]
        flat_batch_sfa = torch.cat(batch_sfa, dim=0)

        for ep_idx, episode in enumerate(low_episodes):
            states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
            actions = torch.stack([step[1] for step in episode], dim=0).to(self.device).long()
            base, sfa = self._extract_low_base_sfa(states)
            hindsight_states = torch.zeros_like(states)
            negative_states = torch.zeros_like(states)
            weights = torch.zeros(sfa.shape[0], dtype=sfa.dtype, device=sfa.device)

            if sfa.shape[0] > 1:
                step_horizon = min(int(horizon), sfa.shape[0] - 1)
                valid_steps = sfa.shape[0] - step_horizon
                goals = sfa[step_horizon:]
                negative_indices = torch.randint(flat_batch_sfa.shape[0], (valid_steps,), device=self.device)
                negative_goals = flat_batch_sfa[negative_indices].to(sfa)
                goal_valid = torch.ones((valid_steps, 1), dtype=sfa.dtype, device=sfa.device)
                hindsight_states[:valid_steps] = self._build_low_state(base[:valid_steps], sfa[:valid_steps], goals, goal_valid)
                negative_states[:valid_steps] = self._build_low_state(base[:valid_steps], sfa[:valid_steps], negative_goals, goal_valid)
                distances = torch.linalg.vector_norm(goals - sfa[:valid_steps], dim=-1)
                next_distances = torch.linalg.vector_norm(goals - sfa[1 : valid_steps + 1], dim=-1)
                progress = (distances - next_distances).clamp_min(0.0)
                _intrinsic = intrinsic_rewards[ep_idx, :sfa.shape[0]].to(sfa).clamp_min(0.0)
                for t in range(valid_steps):
                    intrinsic_term = 0.1 * _intrinsic[t : t + step_horizon].mean()
                    weights[t] = progress[t] + intrinsic_term
                    intrinsic_terms.append(intrinsic_term.detach().view(1))
                target_distances.append(distances.detach())
                progress_terms.append(progress.detach())

            states_per_episode.append(hindsight_states.detach().cpu())
            negative_states_per_episode.append(negative_states.detach().cpu())
            actions_per_episode.append(actions.detach().cpu())
            weights_per_episode.append(weights.detach().cpu())

        if target_distances:
            self.logger.log_scalar("low/hindsight_target_dist", torch.cat(target_distances).mean())
        if progress_terms:
            self.logger.log_scalar("low/hindsight_progress_weight_mean", torch.cat(progress_terms).mean())
        if intrinsic_terms:
            self.logger.log_scalar("low/hindsight_intrinsic_weight_mean", torch.cat(intrinsic_terms).mean())
        return states_per_episode, actions_per_episode, weights_per_episode, negative_states_per_episode

    def _compute_high_switch_rewards(self, episode):
        states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
        switches = torch.stack([step[1]["switch"] for step in episode], dim=0).to(self.device).to(torch.float32).view(-1)
        progress = states[:, -2]
        steps_since_switch = states[:, -1]

        # progress[t] is measured before switch[t] is applied. Rewarding switch[t]
        # with progress[t] would credit the gate for the old goal. Instead, use
        # next-step progress: if switch[t] starts a new goal, progress[t+1] is
        # measured against that new goal; if switch[t] keeps the goal, progress[t+1]
        # measures whether keeping it kept working.
        next_progress = torch.zeros_like(progress)
        if progress.shape[0] > 1:
            next_progress[:-1] = progress[1:]

        rewards = next_progress - 0.01 * switches
        stalled = (switches < 0.5) & (steps_since_switch > 2.0) & (next_progress < 0.005)
        rewards[stalled] -= 1.0
        return rewards.detach()

    def _compute_high_goal_rewards(self, high_episode, low_episode, low_rewards):
        assert len(high_episode) == len(low_episode), "High and low episodes must align"
        assert low_rewards.shape == (len(high_episode),), "Low rewards must align with high episode"

        switches = torch.stack(
            [step[1]["switch"] for step in high_episode],
            dim=0,
        ).to(self.device).to(torch.float32).view(-1)
        _, next_distances = self._low_goal_transition_distances(low_episode)
        assert switches.shape == next_distances.shape, "Switches and goal distances must align"

        option_ends = torch.zeros_like(switches, dtype=torch.bool)
        option_ends[-1] = True
        if option_ends.shape[0] > 1:
            option_ends[:-1] = switches[1:] > 0.5

        distance_penalties = torch.zeros_like(next_distances)
        distance_penalties[option_ends] = (
            self.high_virtual_goal_distance_penalty * next_distances[option_ends]
        )
        rewards = low_rewards.to(self.device) - distance_penalties
        return rewards.detach(), distance_penalties.detach()

    def _compute_high_achievability_constraint(self, high_states, goal):
        assert high_states.dim() == 2, \
            f"Expected selected high states [N,D], got {high_states.shape}"
        assert goal.dim() == 2, f"Expected sampled goals [N,G], got {goal.shape}"
        assert high_states.shape[0] == goal.shape[0], \
            "High states and sampled goals must have matching batches"

        goal_dim = self.wm_model.contrastive_dim
        assert goal.shape[1] == goal_dim, \
            f"Expected goal dimension {goal_dim}, got {goal.shape[1]}"
        base = high_states[:, :-(goal_dim + 3)].detach()
        sfa = base[:, -goal_dim:]
        assert sfa.shape == goal.shape, "Failed to align current SFA with sampled goal"
        goal_valid = goal.new_ones((goal.shape[0], 1))
        achievability_states = self._build_low_state(base, sfa, goal, goal_valid)

        head_parameters = list(self.agent_low.achievability_head.parameters())
        requires_grad = [parameter.requires_grad for parameter in head_parameters]
        for parameter in head_parameters:
            parameter.requires_grad_(False)
        try:
            prediction = self.agent_low.achievability_head(achievability_states)
        finally:
            for parameter, enabled in zip(head_parameters, requires_grad):
                parameter.requires_grad_(enabled)
        assert prediction.shape == (goal.shape[0], 3), \
            f"Expected achievability prediction [N,3], got {prediction.shape}"

        log_distance_mean = prediction[:, 0]
        log_distance_scale = F.softplus(prediction[:, 1]) + 1e-4
        expected_distance = torch.exp(
            log_distance_mean + 0.5 * log_distance_scale.square()
        )
        assert torch.isfinite(expected_distance).all(), \
            "Non-finite expected achievability distance"
        loss = self.high_achievability_coef * expected_distance.mean()
        return loss, expected_distance.mean()

    def _extract_high_sfa(self, states):
        assert states.dim() == 2, f"Expected high states [T,D], got {tuple(states.shape)}"
        goal_dim = self.wm_model.contrastive_dim
        sfa_end = -(goal_dim + 3)
        sfa_start = sfa_end - goal_dim
        sfa = states[:, sfa_start:sfa_end]
        assert sfa.shape[-1] == goal_dim, "failed to extract SFA from high-level state"
        return sfa

    def _compute_high_current_sfa_targets(self, high_episodes):
        targets_per_episode = []
        weights_per_episode = []

        for episode in high_episodes:
            states = torch.stack([step[0] for step in episode], dim=0).to(self.device)
            sfa = self._extract_high_sfa(states)
            targets_per_episode.append(sfa.detach())
            weights_per_episode.append(torch.ones(sfa.shape[0], dtype=sfa.dtype, device=sfa.device))

        return targets_per_episode, weights_per_episode

    def _overwrite_high_rewards(self, episodes, goal_rewards_per_episode, switch_rewards_per_episode):
        assert len(episodes) == len(switch_rewards_per_episode), "episode/reward count mismatch"
        assert len(episodes) == len(goal_rewards_per_episode), "episode/goal reward count mismatch"
        for episode, switch_rewards in zip(episodes, switch_rewards_per_episode):
            assert len(episode) == switch_rewards.shape[0], "episode/reward length mismatch"
        for episode, goal_rewards, switch_rewards in zip(episodes, goal_rewards_per_episode, switch_rewards_per_episode):
            assert len(episode) == goal_rewards.shape[0] == switch_rewards.shape[0], "episode/reward length mismatch"
            for step_idx, (goal_reward, switch_reward) in enumerate(zip(goal_rewards, switch_rewards)):
                episode[step_idx] = (*episode[step_idx][:4], goal_reward, switch_reward)
        self.logger.log_scalar(
            "high/switch_reward_mean",
            torch.cat(switch_rewards_per_episode).mean() if switch_rewards_per_episode else torch.tensor(0.0),
        )
        self.logger.log_scalar(
            "high/goal_reward_mean",
            torch.cat(goal_rewards_per_episode).mean() if goal_rewards_per_episode else torch.tensor(0.0),
        )

    def get_action(self, state, episode_start):
        wm_out = self.call_wm(state, episode_start)

        aux = wm_out["aux"]
        cpc = self._squeeze_time(aux["contrastive_tgt_emb"])
        sfa = self._squeeze_time(aux["sfa"])
        base_features = torch.cat([wm_out["state_last"], cpc, sfa], dim=-1)
        surprise = self._cpc_surprise(cpc.detach(), episode_start)
        base_features = base_features.detach()
        sfa = sfa.detach()

        prev_goal, steps_since_switch, progress = self.agent_high.goal_context(sfa, episode_start)
        batch_size = sfa.shape[0]
        zeros = sfa.new_zeros((batch_size, 1))
        high_state = torch.cat([base_features, prev_goal,
                                surprise,
                                progress, steps_since_switch, ], dim=-1,)
        sampled_goal = self.agent_high.get_action(high_state, episode_start).detach()

        goal_enabled = self._update_high_goal_enabled(episode_start, batch_size)

        if goal_enabled.any() and not self.br:
            self.br = True
            print('goal enabled')
        goal_valid = goal_enabled.to(sfa).unsqueeze(-1)
        goal = sampled_goal * goal_valid

        low_state = self._build_low_state(base_features, sfa, goal, goal_valid)
        actions = self.agent_low.get_action(low_state, episode_start)
        self.agent_high.store_prediction_context(wm_out["state_last"], actions)

        wm_states = {
            "sensor": state["sensor"].detach().to("cpu"),
            "heading_idx": state["heading_idx"].detach().to("cpu"),
            "location": state["location"].detach().to("cpu"),
            "policy_state": low_state.detach().to("cpu"),
        }
        actions_cpu = actions.detach().to("cpu")
        log_probs_cpu = torch.zeros((actions_cpu.shape[0],))
        entropy_cpu = torch.zeros((actions_cpu.shape[0], 1))
        self._wm_pool.add_transition_batch(wm_states, actions_cpu, log_probs_cpu, entropy_cpu)
        self.record_sampled_actions(actions)
        return actions

    def update(self, rewards, dones, info=None, **kwargs):
        """
        Train the WM and both policy levels after a complete environment batch.

        1. Train the WM on recent real episodes mixed with replay, then compute the
           intrinsic rewards used by the policies.
        2. Train low-level PPO on real episodes. Goal-free episodes use intrinsic
           reward; goal-conditioned episodes use goal progress plus a small intrinsic
           bonus.
        3. Before virtual rollout training is available, train low-level with
           intrinsic-reward PPO and real-trajectory hindsight ranking. Real episodes
           remain goal-free during collection (zero goal SFA with goal_valid=0).
        4. Once virtual rollout training is available, extend low-level goal
           conditioning with virtual hindsight ranking.
        5. During that phase, supervise the high-level goal mean toward current SFA.
           The high-level scale is not trained by this pretraining loss.
        6. Once the WM CPC error passes the virtual-readiness threshold, generate short
           goal-free virtual trajectories
               s_t -> a_t -> virtual s_t+1 -> a_t+1 -> virtual s_t+2 -> ...
           For each trajectory, use its achieved terminal SFA as the hindsight goal
           for every recorded transition. Train the low-level policy to increase
           pi(a_t | s_t, achieved_terminal_sfa) and rank that action above the same
           action conditioned on a negative goal from another virtual trajectory.
        7. From the same source state, run four virtual attempts for each of four
           goals: a reachable virtual goal, a goal from another episode, a goal
           sampled at least 30 real timesteps away in the same episode, and a goal
           sampled from the current high-level policy. Train the achievability head
           to predict the attempt-distance distribution and empirical success rate.
        8. After the achievability-head warmup, mix the configured fraction of real
           episodes whose goals are sampled by the high-level agent. Continue training
           the low-level agent on both goal-free real episodes and virtual hindsight data.
        9. Train the high-level goal head either by imitating achieved terminal SFA
           from the goal-free virtual rollouts, through the frozen achievability
           critic, or with optional goal-only PPO on virtual progress. The imitation
           diagnostic can use all rollout endpoints, freeze its first virtual
           dataset, or replay source prefixes sampled from a bounded episode pool.
           When real high-level PPO and imitation are both active, combine their
           losses into one full-batch policy optimizer step.
        10. Train high-level PPO only on real goal-conditioned episodes. Its goal
            reward is the low-level reward accumulated until the next switch.
        """
        env_rewards = self.env_reward_scale * rewards
        for env_idx in range(self.num_envs):
            self.rl_add_reward(env_idx, env_rewards[env_idx])
            self._wm_pool.add_reward(env_idx, env_rewards[env_idx].detach().to("cpu"))
        self.process_dones(dones)
        self._wm_pool.process_dones(dones)

        if not self.should_learn():
            return False

        wm_new_episodes = [ep for ep in self._wm_pool.get_completed_episodes() if len(ep) >= 2]

        replay_episodes = self._sample_replay_episodes()
        obs_mix, targets_mix, _ = self._build_wm_batch(wm_new_episodes + replay_episodes)
        obs_new, targets_new, _ = self._build_wm_batch(wm_new_episodes)
        wm_updates = 0 if self.wm_fixed else max(1, self.wm_updates_per_policy)
        wm_updates_for_logging = max(1, wm_updates)
        wm_loss_sums: Dict[str, float] = {}
        wm_metric_sums: Dict[str, float] = {}
        total_loss_sum = 0.0
        wm_loss_sum = 0.0

        if self.wm_fixed:
            self.wm_model.eval()
        else:
            self.wm_model.train()
        self.train()
        cpc_error_before = self._evaluate_step_cpc_error_no_grad(self.wm_model, obs_new, targets_new)
        sensor_error_before = self._evaluate_step_sensor_error_no_grad(self.wm_model, obs_new, targets_new)
        with torch.no_grad():
            state_out_fixed = self.wm_model(obs_new)
            old_state_seq = state_out_fixed["state_seq"].detach()
            embs = state_out_fixed['aux']['contrastive_tgt_emb']
        divergence_novelty = self._compute_state_divergence_novelty(
            state_seq=embs,
            key_padding_mask=obs_new["key_padding_mask"])

        episode_novelty = self._compute_episode_novelty(
            state_seq=embs,
            key_padding_mask=obs_new["key_padding_mask"])

        for _ in range(wm_updates):
            # World-model update on mixed dataset.
            self.wm_optimizer.zero_grad()
            wm_forward = self.wm_model(obs_mix)
            preds = wm_forward["preds"]
            aux = wm_forward['aux']
            wm_loss_results, wm_loss_metrics = self.wm_model.compute_losses_and_metrics(
                preds=preds,
                targets=targets_mix,
                aux_inputs=aux,
            )
            wm_loss_base = self._compute_wm_total_loss(wm_loss_results)
            wm_stability_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            wm_state_drift = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if self.wm_stability_coef > 0.0:
                state_out = self.wm_model(obs_new)
                new_state = state_out["state_seq"]
                wm_stability_loss, wm_state_drift = self._compute_state_stability_loss(
                    new_state=new_state,
                    old_state=old_state_seq,
                    key_padding_mask=obs_new["key_padding_mask"],
                )
            wm_loss = wm_loss_base + self.wm_stability_coef * wm_stability_loss
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm_model.parameters(), 1.0)
            self.wm_optimizer.step()

            total_loss = wm_loss.detach()
            total_loss_sum += self._as_scalar(total_loss)
            wm_loss_sum += self._as_scalar(wm_loss)
            for key, value in wm_loss_results.items():
                wm_loss_sums[key] = wm_loss_sums.get(key, 0.0) + self._as_scalar(value)
            wm_loss_sums["stability"] = wm_loss_sums.get("stability", 0.0) + self._as_scalar(wm_stability_loss)
            for key, value in wm_loss_metrics.items():
                wm_metric_sums[key] = wm_metric_sums.get(key, 0.0) + self._as_scalar(value)
            wm_metric_sums["state_drift"] = wm_metric_sums.get("state_drift", 0.0) + self._as_scalar(wm_state_drift)

        cpc_error_after = self._evaluate_step_cpc_error_no_grad(self.wm_model, obs_new, targets_new)
        sensor_error_after = self._evaluate_step_sensor_error_no_grad(self.wm_model, obs_new, targets_new)
        emb_pred_error = self.embedding_prediction_error(state_out_fixed['aux'],
                                                         key_padding_mask=obs_new["key_padding_mask"],
                                                         metric="cosine",
                                                         horizon_discount=0.75)
        intrinsic_rewards = self._compute_intrinsic_rewards(
            cpc_error_before=cpc_error_before,
            cpc_error_after=cpc_error_after,
            sensor_error_before=sensor_error_before,
            sensor_error_after=sensor_error_after,
            key_padding_mask=obs_new["key_padding_mask"],
            divergence_novelty=divergence_novelty,
            episode_novelty=episode_novelty,
            cluster_dist_novelty=None,
            embedding_prediction_error=emb_pred_error)
        low_virtual_ready, low_virtual_cpc_error, low_virtual_cpc_error_ema = self._low_virtual_ready(
            cpc_error_after,
            obs_new["key_padding_mask"],
        )

        low_episodes = [ep for ep in self.agent_low.get_train_episodes() if len(ep) >= 2]
        high_episodes = [ep for ep in self.agent_high.get_train_episodes() if len(ep) >= 2]
        if len(low_episodes) != len(wm_new_episodes) or len(high_episodes) != len(wm_new_episodes):
            raise ValueError("Low, high, and WM completed episode counts must match")
        goal_enabled = [self._episode_goal_enabled(episode) for episode in low_episodes]
        low_rewards = [
            self._compute_low_actual_rewards(
                ep,
                intrinsic_rewards[ep_idx],
                goal_coef=1.0 if enabled else 0.0,
                intrinsic_coef=0.1 if enabled else 1.0,
            )
            for ep_idx, (ep, enabled) in enumerate(zip(low_episodes, goal_enabled))
        ]
        self._log_low_reward_components(
            low_episodes,
            intrinsic_rewards,
            goal_enabled,
        )
        self._overwrite_low_rewards(low_episodes, low_rewards)
        high_virtual_imitation_batch = None
        if low_episodes:
            low_h_states, low_h_actions, low_h_weights, low_h_negative_states = \
                self._compute_low_hindsight(low_episodes, intrinsic_rewards)
            virtual_achievability = None
            if low_virtual_ready:
                max_prefix_step = torch.tensor(list(len(episode) - 1 for episode in wm_new_episodes), dtype=torch.long)
                # randomize ep start
                virtual_step_idx = (torch.rand(len(max_prefix_step)) * (max_prefix_step + 1)).long()

                virtual_hindsight, candidate_goals, best_virtual_goals, \
                    all_virtual_goals = \
                    self._collect_low_virtual_hindsight(
                        wm_new_episodes,
                        step_idx=virtual_step_idx,
                        horizon=self.virtual_horizon,
                    )
                assert candidate_goals.shape[0] > 1, \
                    "Other-episode achievability goals require at least two episodes"
                other_episode_offset = int(torch.randint(1, candidate_goals.shape[0], (1,)).item())
                other_episode_goals = candidate_goals.roll(other_episode_offset, dims=0)
                distant_step_idx = self._sample_temporally_distant_steps(
                    low_episodes,
                    virtual_step_idx,
                    min_delta=30,
                )
                distant_out, _, _ = self._replay_wm_to_step(wm_new_episodes, distant_step_idx)
                distant_goals = self._squeeze_time(distant_out["aux"]["sfa"]).detach()
                if self.high_virtual_goal_ppo:
                    high_virtual_goals, high_virtual_old_log_probs = \
                        self.agent_high.sample_goal_candidates(
                            high_episodes,
                            virtual_step_idx,
                            num_samples=self.high_virtual_goal_candidates,
                        )
                    high_policy_goals = high_virtual_goals[:, 0]
                else:
                    sampled_high_goals, _ = self.agent_high.sample_goal_candidates(
                        high_episodes,
                        virtual_step_idx,
                        num_samples=1,
                    )
                    high_policy_goals = sampled_high_goals.squeeze(1)
                assert high_policy_goals.shape == candidate_goals.shape, \
                    "Sampled high-policy goals must match virtual candidate goals"
                achievability_goals = torch.stack(
                    [candidate_goals, other_episode_goals, distant_goals, high_policy_goals],
                    dim=1,
                )
                virtual_achievability = self._collect_low_virtual_achievability(
                    wm_new_episodes,
                    step_idx=virtual_step_idx,
                    goals=achievability_goals,
                    goal_names=("reachable", "other_episode", "distant_time", "high_policy"),
                    horizon=self.virtual_horizon,
                )
                low_h_states.extend(virtual_hindsight[0])
                low_h_actions.extend(virtual_hindsight[1])
                low_h_weights.extend(virtual_hindsight[2])
                low_h_negative_states.extend(virtual_hindsight[3])
            hindsight = (
                low_h_states,
                low_h_actions,
                low_h_weights,
                low_h_negative_states,
            )
            if not self.high_achievability_overfit and not self.low_fixed:
                self.agent_low.learn_from_episodes(
                    low_episodes,
                    hindsight=hindsight,
                )
            if low_virtual_ready and self.high_virtual_goal_imitation:
                virtual_imitation_targets = (
                    all_virtual_goals
                    if self.high_virtual_goal_imitation_all_targets
                    else best_virtual_goals
                )
                imitation_episodes, imitation_steps, virtual_imitation_targets = \
                    self._get_high_virtual_imitation_dataset(
                        high_episodes,
                        virtual_step_idx,
                        virtual_imitation_targets,
                    )
                imitation_episodes, imitation_steps, virtual_imitation_targets = \
                    self._get_high_virtual_imitation_replay(
                        imitation_episodes,
                        imitation_steps,
                        virtual_imitation_targets,
                    )
                high_virtual_imitation_batch = (
                    imitation_episodes,
                    imitation_steps,
                    virtual_imitation_targets,
                    self.high_virtual_goal_imitation_coef,
                )
            if virtual_achievability is not None:
                if not self.high_achievability_overfit and not self.low_fixed:
                    self.agent_low.train_achievability(*virtual_achievability)
                    self._achievability_updates += 1
                self._log_real_achievability_ranking(low_episodes)
                if self.high_virtual_goal_ppo:
                    high_virtual_rewards = self._collect_high_virtual_goal_rewards(
                        wm_new_episodes,
                        virtual_step_idx,
                        high_virtual_goals,
                    )
                    self.agent_high.train_virtual_goal_ppo(
                        high_episodes,
                        virtual_step_idx,
                        high_virtual_goals,
                        high_virtual_old_log_probs,
                        high_virtual_rewards,
                    )
                elif not self.high_virtual_goal_imitation \
                        and self._high_goal_ready() \
                        and self.high_achievability_coef > 0.0:
                    for _ in range(self.high_achievability_policy_updates):
                        self.agent_high.train_goal_constraint(
                            high_episodes,
                            virtual_step_idx,
                            self._compute_high_achievability_constraint,
                            max_kl=self.high_achievability_max_kl,
                            lr_scale=self.high_achievability_lr_scale,
                            num_goal_samples=self.high_virtual_goal_candidates,
                        )

        high_goal_rewards = []
        high_goal_distance_penalties = []
        high_switch_rewards = []
        for high_episode, low_episode, episode_low_rewards, enabled in zip(
            high_episodes,
            low_episodes,
            low_rewards,
            goal_enabled,
        ):
            if enabled:
                goal_rewards, distance_penalties = self._compute_high_goal_rewards(
                    high_episode,
                    low_episode,
                    episode_low_rewards,
                )
                switch_rewards = self._compute_high_switch_rewards(high_episode)
            else:
                goal_rewards = torch.zeros(len(high_episode), dtype=torch.float32, device=self.device)
                distance_penalties = torch.zeros_like(goal_rewards)
                switch_rewards = torch.zeros_like(goal_rewards)
            high_goal_rewards.append(goal_rewards)
            high_goal_distance_penalties.append(distance_penalties)
            high_switch_rewards.append(switch_rewards)
        self._overwrite_high_rewards(high_episodes, high_goal_rewards, high_switch_rewards)
        self.logger.log_scalar(
            "high/goal_terminal_distance_penalty_mean",
            torch.cat(high_goal_distance_penalties).mean()
            if high_goal_distance_penalties else torch.tensor(0.0),
        )
        # if self.br:
        #     import pdb;pdb.set_trace()
        high_ppo_episode_count = 0
        if high_episodes:
            high_virtual_imitation_trained = False
            if self._high_goal_ready():
                enabled_high_episodes = [
                    episode
                    for episode, enabled in zip(high_episodes, goal_enabled)
                    if enabled
                ]
                high_ppo_episode_count = len(enabled_high_episodes)
                if enabled_high_episodes:
                    if high_virtual_imitation_batch is not None:
                        assert self.high_virtual_goal_imitation_epochs == 1, \
                            "Combined high PPO and virtual imitation supports one optimizer step"
                    self.agent_high.learn_from_episodes(
                        enabled_high_episodes,
                        num_minibatches=1,
                        virtual_imitation=high_virtual_imitation_batch,
                    )
                    high_virtual_imitation_trained = high_virtual_imitation_batch is not None
            if high_virtual_imitation_batch is not None and not high_virtual_imitation_trained:
                imitation_episodes, imitation_steps, imitation_targets, imitation_coef = \
                    high_virtual_imitation_batch
                self.agent_high.train_virtual_goal_imitation(
                    imitation_episodes,
                    imitation_steps,
                    imitation_targets,
                    coef=imitation_coef,
                    num_epochs=self.high_virtual_goal_imitation_epochs,
                )
                high_virtual_imitation_trained = True
            if not self._high_goal_ready() and not high_virtual_imitation_trained:
                goal_targets, goal_weights = self._compute_high_current_sfa_targets(high_episodes)
                old_coef = self.agent_high.goal_hindsight_coef
                self.agent_high.goal_hindsight_coef = self.high_warmup_goal_coef
                self.agent_high.train_goal_hindsight(
                    high_episodes,
                    goal_targets,
                    goal_weights,
                    num_epochs=self.high_warmup_goal_epochs,
                    switch_only=False,
                    train_sigma=False,
                )
                self.agent_high.goal_hindsight_coef = old_coef
        scalar_sums = {
            "joint/loss_total": total_loss_sum,
            "reward/intrinsic_mean": float(intrinsic_rewards.mean().detach().cpu()) * wm_updates_for_logging,
            "wm/loss/total": wm_loss_sum,
        }
        extra_scalars = {
            "wm/fixed": float(self.wm_fixed),
            "low/fixed": float(self.low_fixed),
            "double/high_warmup_goal_coef": float(self.high_warmup_goal_coef),
            "double/high_warmup_goal_epochs": float(self.high_warmup_goal_epochs),
            "double/achievability_updates": float(self._achievability_updates),
            "double/achievability_warmup_updates": float(self.achievability_warmup_updates),
            "double/high_goal_mix_probability": self._high_goal_mix_probability(),
            "double/high_goal_max_fraction": self.high_goal_max_fraction,
            "double/high_achievability_max_kl": self.high_achievability_max_kl,
            "double/virtual_horizon": float(self.virtual_horizon),
            "double/high_virtual_goal_ppo": float(self.high_virtual_goal_ppo),
            "double/high_virtual_goal_imitation": float(self.high_virtual_goal_imitation),
            "double/high_virtual_goal_imitation_coef": self.high_virtual_goal_imitation_coef,
            "double/high_virtual_goal_imitation_epochs": float(
                self.high_virtual_goal_imitation_epochs
            ),
            "double/high_virtual_goal_imitation_all_targets": float(
                self.high_virtual_goal_imitation_all_targets
            ),
            "double/high_virtual_goal_imitation_fixed_dataset": float(
                self.high_virtual_goal_imitation_fixed_dataset
            ),
            "double/high_virtual_goal_imitation_replay_capacity": float(
                self.high_virtual_goal_imitation_replay_capacity
            ),
            "double/high_virtual_goal_imitation_replay_samples": float(
                self.high_virtual_goal_imitation_replay_samples
            ),
            "high/virtual_imitation/replay_size": float(
                len(self._high_virtual_imitation_pool.episode_pool)
                if self._high_virtual_imitation_pool is not None
                else 0
            ),
            "double/high_goal_output_delta": float(
                self.agent_high.goal_output_delta
            ),
            "double/high_achievability_overfit": float(self.high_achievability_overfit),
            "double/high_achievability_policy_updates": float(
                self.high_achievability_policy_updates
            ),
            "double/high_virtual_goal_candidates": float(self.high_virtual_goal_candidates),
            "double/high_virtual_goal_attempts": float(self.high_virtual_goal_attempts),
            "double/high_virtual_goal_distance_penalty": self.high_virtual_goal_distance_penalty,
            "high/current_sfa_pretrain_active": float(not self._high_goal_ready()),
            "high/ready": float(self._high_goal_ready()),
            "high/ppo_episode_fraction": (
                sum(goal_enabled) / len(goal_enabled) if goal_enabled else 0.0
            ),
            "high/ppo_update_episode_count": float(high_ppo_episode_count),
            "high/ppo_update_active": float(high_ppo_episode_count > 0),
            "low/virtual_ready": float(low_virtual_ready),
            "low/hindsight_active": 1.0,
            "low/virtual_cpc_error_after": low_virtual_cpc_error,
            "low/virtual_cpc_error_ema": low_virtual_cpc_error_ema,
            "low/virtual_cpc_error_threshold": self.low_virtual_cpc_error_threshold,
            "wm/actual_updates": float(wm_updates),
            "wm/replay_size": float(len(self._wm_pool.episode_pool)),
            "wm/replay_sampled_episodes": float(len(replay_episodes)),
            "wm/joint_sampled_episodes": float(len(wm_new_episodes) + len(replay_episodes)),
            "wm/joint_sampled_transitions": float(
                sum(max(0, len(ep) - 1) for ep in wm_new_episodes)
                + sum(max(0, len(ep) - 1) for ep in replay_episodes)
            ),
        }
        self._log_update_stats(
            updates=wm_updates_for_logging,
            scalar_sums=scalar_sums,
            extra_scalars=extra_scalars,
            wm_loss_sums=wm_loss_sums,
            wm_metric_sums=wm_metric_sums,
            info=info,
        )

        self.version += 1
        self.clear_completed()
        self._wm_pool.clear_completed()

        return True
