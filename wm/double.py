import torch

from ppo import PPOBase
from wm_joint_agent import BaseWMOnPolicy


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
        BaseWMOnPolicy.__init__(self, device=device, logger=logger, **kwargs)

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
        if hasattr(self.wm_model, "clear_cache"):
            self.wm_model.clear_cache()
        self._wm_pool.reset_episodes()

    @property
    def version(self):
        return self.agent_low.version

    @version.setter
    def version(self, value):
        self.agent_low.version = value

    def _squeeze_time(self, x):
        if x.dim() == 3:
            assert x.shape[1] == 1, f"Expected online one-step feature, got time dim {x.shape[1]}"
            return x[:, -1, :]
        return x

    def _base_features(self, wm_out):
        h = wm_out["state_last"]
        aux = wm_out["aux"]
        cpc = self._squeeze_time(aux["contrastive_tgt_emb"])
        sfa = self._squeeze_time(aux["sfa"])
        return torch.cat([h, cpc, sfa], dim=-1), sfa

    def get_action(self, state, episode_start):
        wm_out = self.call_wm(state, episode_start)

        base_features, sfa = self._base_features(wm_out)
        base_features = base_features.detach()
        sfa = sfa.detach()

        prev_goal, steps_since_switch, progress = self.agent_high.goal_context(sfa, episode_start)
        batch_size = sfa.shape[0]
        zeros = sfa.new_zeros((batch_size, 1))
        high_state = torch.cat([base_features, prev_goal,
                                zeros,  # surprise placeholder
                                progress, steps_since_switch, ], dim=-1,)
        goal = self.agent_high.get_action(high_state, episode_start).detach()

        low_state = torch.cat([base_features, goal, goal - sfa], dim=-1)
        actions = self.agent_low.get_action(low_state, episode_start)

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
