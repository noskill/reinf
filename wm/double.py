from wm_join_agent import BaseWMOnPolicy


class DoubleAgent(BaseWMOnPolicy):
    """PPO two-policy agent with joint AC-CPC world-model updates."""
    def __init__(self, policy_high, policy_low,
                 value_high, value_low,
                 sampler,
                 policy_lr=0.0001,
                 value_lr=0.001, num_envs=8,
                 discount=0.999,
                 device=torch.device('cpu'),
                 logger=None,
                 num_learning_epochs=4,
                 entropy_coef=0.005,
                 clip_param=None,
                 exp_adv=None,
                 target_entropy=2,
                 **kwargs):
        
        self.agent_high = PPO(
            policy=policy_high,
            value=value_high,
            sampler=sampler,
            policy_lr=policy_lr,
            value_lr=value_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger,
            entropy_coef=entropy_coef,
            target_entropy=target_entropy,
            num_learning_epochs=num_learning_epochs,
            clip_param=clip_param,
            exp_adv=exp_adv,
        )
        self.agent_low = PPO(
            policy=policy_low,
            value=value_low,
            sampler=sampler,
            policy_lr=policy_lr,
            value_lr=value_lr,
            num_envs=num_envs,
            discount=discount,
            device=device,
            logger=logger,
            entropy_coef=entropy_coef,
            target_entropy=target_entropy,
            num_learning_epochs=num_learning_epochs,
            clip_param=clip_param,
            exp_adv=exp_adv,
        )
        
        BaseWMOnPolicy.__init__(self, device=device, logger=logger, **kwargs)
        self.agents = [self.agent_high, self.agent_low]

    def get_action(self, state, episode_start):
        """
        Sample high-level(gated) goal state embedding.
        Use it to condition low-level policy
        """
        wm_out = self.call_wm(state, episode_start)
        sfa = wm_out['aux']['sfa']
        h = wm_out["state_last"]
        # high-level policy recieves current state h + sfa features
        policy_kwargs = dict(episode_start=episode_start)
        
        actions = self.agent_high.get_action(
        actions, log_probs, dist = self.rl_sampler()(policy, policy_states, policy_kwargs)

        # ------------------------------------------------------------------
        # Obtain entropy safely. Some `TransformedDistribution` instances do
        # not implement an analytic entropy; fall back to the base
        # distribution in that case.
        # ------------------------------------------------------------------
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            if hasattr(dist, 'base_dist'):
                entropy = dist.base_dist.entropy()
            else:
                raise

        # Validate that each action produces *exactly* one log-prob and one
        # entropy scalar.
        assert log_probs.dim() == 1 or (log_probs.dim() == 2 and log_probs.shape[1] == 1), \
            f"log_probs shape {log_probs.shape} invalid; expected (B,) or (B,1)."
        assert entropy.dim() == 1 or (entropy.dim() == 2 and entropy.shape[1] == 1), \
            f"entropy shape {entropy.shape} invalid; expected (B,) or (B,1)."

        # Ensure entropy is a column vector (B,1) for consistent downstream
        # handling, **without** accidentally reducing across the batch.
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(-1)

        self.rl_add_transition_batch(policy_states, actions, log_probs, entropy)
        wm_states = {
            "sensor": state["sensor"].detach().to("cpu"),
            "heading_idx": state["heading_idx"].detach().to("cpu"),
            "location": state["location"].detach().to("cpu"),
            "policy_state": policy_states.detach().to("cpu"),
        }
        actions_cpu = actions.detach().to("cpu")
        log_probs_cpu = torch.zeros_like(log_probs).detach().to("cpu")
        entropy_cpu = torch.zeros_like(entropy).detach().to("cpu")
        self._wm_pool.add_transition_batch(wm_states, actions_cpu, log_probs_cpu, entropy_cpu)
        self.record_sampled_actions(actions)
        return actions
