import math

import torch
from torch import nn
from torch.distributions import Categorical, Dirichlet

from agent_group import AgentGroup
from policy_head import WMActionHeadPolicy, WMValueHeadPolicy
from ppo import PPO
from sample import ActionSampler
from util import detach as detach_tree, flatten_padded, to_device, tree_index, tree_map, tree_stack
from utils import masked_lr_metrics_logits
from wm_joint_agent import JointWMPPO


class SharedBodyPolicy(nn.Module):
    """Local Dirichlet vote and optional categorical message policy."""

    def __init__(self, head, num_actions, message_symbols, concentration=20.0):
        super().__init__()
        if not math.isfinite(concentration) or concentration <= 0:
            raise ValueError("Vote concentration must be finite and positive")
        self.head = head
        self.num_actions = num_actions
        self.message_symbols = message_symbols
        self.concentration = concentration

    def local_logits(self, features):
        logits = self.head(features)
        expected = self.num_actions + self.message_symbols
        assert logits.shape[-1] == expected, f"Expected {expected} local policy logits"
        return logits[..., :self.num_actions], logits[..., self.num_actions:]

    def forward(self, state, **kwargs):
        motor, message = self.local_logits(state["features"])
        probabilities = motor.softmax(-1)
        # Keep alpha positive even when softmax underflows, preserving sum(alpha) = kappa.
        probabilities = (probabilities + 1e-6) / (1 + self.num_actions * 1e-6)
        return {"concentration": self.concentration * probabilities, "message_logits": message}


class VoteMessageDistribution:
    """PPO tensor action: vote coordinates, then one message index when enabled."""

    def __init__(self, concentration, message_logits):
        self.vote = Dirichlet(concentration)
        self.message = Categorical(logits=message_logits) if message_logits.shape[-1] else None
        self.num_actions = concentration.shape[-1]

    def sample(self):
        vote = self.vote.sample()
        if self.message is None:
            return vote
        return torch.cat([vote, self.message.sample().to(vote.dtype).unsqueeze(-1)], dim=-1)

    def log_prob(self, actions):
        assert actions.shape[-1] == self.num_actions + int(self.message is not None)
        log_prob = self.vote.log_prob(actions[..., :self.num_actions])
        if self.message is not None:
            log_prob = log_prob + self.message.log_prob(actions[..., self.num_actions])
        return log_prob

    def entropy_parts(self):
        parts = [self.vote.entropy()]
        if self.message is not None:
            parts.append(self.message.entropy())
        return torch.stack(parts, dim=-1)

    def entropy(self):
        return self.entropy_parts().sum(-1)


class VoteMessageSampler(ActionSampler):
    def __call__(self, params, actions=None, return_distribution=False):
        distribution = VoteMessageDistribution(params["concentration"], params["message_logits"])
        if actions is None:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution


class SharedBodyPPO(PPO):
    def __init__(self, policy, *args, **kwargs):
        symmetric = torch.full((policy.num_actions,), policy.concentration / policy.num_actions)
        self.entropy_targets = [Dirichlet(symmetric).entropy().item()]
        if policy.message_symbols:
            self.entropy_targets.append(0.5 * math.log(policy.message_symbols))
        super().__init__(policy, *args, target_entropy=self.entropy_targets[0], **kwargs)

    def compute_distribution_params(self, observations, actions, key_padding_mask):
        # Padded zero vectors are outside the Dirichlet simplex; exclude them before log_prob.
        states = {name: flatten_padded(value, key_padding_mask) for name, value in observations.items()
                  if name != "key_padding_mask"}
        actions = flatten_padded(actions, key_padding_mask)
        _, log_prob, distribution = self.sampler.sample_policy(self.policy, states, actions=actions)
        return log_prob, distribution.entropy_parts(), None

    def compute_entropy_loss(self, entropy, device, dtype, target_entropy=None, return_parts=False):
        targets = self.entropy_targets if target_entropy is None else target_entropy
        targets = torch.as_tensor(targets, device=device, dtype=dtype)
        entropy = entropy.to(device=device, dtype=dtype)
        if entropy.ndim == 1:
            entropy = entropy.unsqueeze(-1)
        assert entropy.shape[-1:] == targets.shape
        error = (entropy - targets) / targets.abs().clamp_min(1.0)
        loss_items = error.square().sum(-1)
        for index, name in enumerate(("vote", "message")[:len(self.entropy_targets)]):
            self.logger.log_scalar(f"entropy/{name}", entropy[:, index].mean().item())
        loss = loss_items.mean()
        return (loss, loss_items) if return_parts else loss


class LocalValueHead(WMValueHeadPolicy):
    def forward(self, state, **kwargs):
        return super().forward(state["features"], **kwargs)


class LocalBodyAgent(JointWMPPO):
    """PPO learns local votes/messages; the WM conditions on the complete transition."""

    def __init__(self, *args, wm_body_action=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_body_action = wm_body_action
        self.action_dim = self.wm_model.action_dim

    def _encode_transition(self, transition):
        received = transition["received_message"].masked_fill(~transition["message_valid"].unsqueeze(-1), 0)
        parts = [transition["own_vote"], transition["own_message"], received.flatten(-2),
                 transition["message_valid"].float()]
        if self.wm_body_action:
            parts.insert(0, transition["body_action"])
        encoded = torch.cat(parts, dim=-1).float().detach()
        assert encoded.shape[-1] == self.action_dim, "WM transition width does not match model action_dim"
        return encoded

    def _wm_episode_action_inputs(self, episode, action_indices):
        transitions = tree_stack([record[0]["transition"] for record in episode[:-1]])
        return self._encode_transition(transitions)

    def episode_start(self):
        super().episode_start()
        self._pending = None
        self._next_observation = None
        self.sensor_predictions = None

    def _compute_step_sensor_error_from_preds(self, wm_model, *, pred_sensor, targets):
        error = super()._compute_step_sensor_error_from_preds(wm_model, pred_sensor=pred_sensor, targets=targets)
        # Retain the latest evaluation on new episodes, not independently sampled replay.
        self.sensor_predictions = detach_tree({"prediction": pred_sensor, "target": targets["y_sensor"]["local"],
                                               "mask": targets["key_padding_mask"]})
        return error

    @torch.no_grad()
    def propose(self, observation, episode_start, channel):
        if self._pending is not None:
            raise RuntimeError("Previous shared-body proposal has not been recorded")
        self.wm_model.eval()
        policy = self.get_policy_for_action()
        policy.eval()
        features = self.process_states(observation, episode_start.to(self.device)).detach()
        state = {"features": features}
        action, log_prob, distribution = self.agent.sampler.sample_policy(policy, state)
        self._pending = {"observation": detach_tree(observation), "state": state, "action": action,
                         "log_prob": log_prob, "entropy": distribution.entropy()}
        if channel == "learned":
            message = action[..., policy.num_actions].long()
            payload = torch.nn.functional.one_hot(message, policy.message_symbols).to(features.dtype)
        elif channel == "state":
            payload = features
        elif channel == "none":
            payload = features[:, :0]
        else:
            raise ValueError(f"Unknown channel: {channel}")
        self._pending["outgoing_message"] = payload.detach()
        return action[..., :policy.num_actions], payload.detach()

    @torch.no_grad()
    def record_body_step(self, body_action, next_observation):
        if self._pending is None:
            raise RuntimeError("record_body_step requires a preceding proposal")
        pending = self._pending
        self.rl_add_transition_batch(pending["state"], pending["action"], pending["log_prob"],
                                     pending["entropy"].unsqueeze(-1))
        transition = {"own_vote": pending["action"][..., :self.agent.policy.num_actions],
                      "own_message": pending["outgoing_message"],
                      "received_message": next_observation["sensor"]["message"],
                      "message_valid": next_observation["sensor"]["message_valid"]}
        if self.wm_body_action:
            transition["body_action"] = self.action_idx_to_val(body_action)
        self._prev_actions = self._encode_transition(transition)
        wm_state = to_device(detach_tree({**pending["observation"], "transition": transition}), torch.device("cpu"))
        self._wm_pool.add_transition_batch(wm_state, body_action.cpu(), pending["log_prob"].cpu(),
                                          pending["entropy"].cpu().unsqueeze(-1))
        self._next_observation = to_device(detach_tree(next_observation), torch.device("cpu"))
        self._pending = None

    def process_dones(self, dones):
        if self._next_observation is None:
            raise RuntimeError("Missing physical next observation for shared-body transition")
        for env_idx in dones.nonzero(as_tuple=False).flatten().tolist():
            self._wm_pool.episodes[env_idx].append((tree_index(self._next_observation, env_idx),))
        self._next_observation = None
        return super().process_dones(dones)


class AgentLogger:
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = prefix

    def log_scalar(self, name, value, step=None):
        self.logger.log_scalar(f"{self.prefix}/{name}", value, step)

    def __getattr__(self, name):
        return getattr(self.logger, name)


class SharedBodyAgents(AgentGroup):
    """Trainer-facing container with independent learners or one shared learner."""

    def __init__(self, agents, env, channel, shared_weights, logger, concentration=20.0, wm_body_action=True):
        self.agents = agents
        self.num_envs = env.num_envs
        self.num_agents = env.num_agents
        self.device = env.device
        self.channel = channel
        self.shared_weights = shared_weights
        self.logger = logger
        self.message_dim = env.message_dim
        self.num_actions = env.num_actions
        self.concentration = concentration
        self.wm_body_action = wm_body_action
        self._awaiting_step = False
        assert len(agents) == (1 if shared_weights else self.num_agents)
        expected_batch = self.num_envs * self.num_agents if shared_weights else self.num_envs
        assert all(agent.num_envs == expected_batch for agent in agents)

    @property
    def version(self):
        versions = [agent.version for agent in self.agents]
        assert len(set(versions)) == 1, "Local agent updates must stay synchronized"
        return versions[0]

    @version.setter
    def version(self, value):
        for agent in self.agents:
            agent.version = value

    def episode_start(self):
        self.start_agents()
        self._awaiting_step = False

    def _local_batch(self, tree, agent_idx):
        if self.shared_weights:
            return tree_map(tree, lambda value: value.flatten(0, 1))
        return tree_map(tree, lambda value: value[:, agent_idx])

    @torch.no_grad()
    def get_action(self, observation, episode_start):
        if self._awaiting_step:
            raise RuntimeError("update must follow every shared-body get_action")
        assert episode_start.shape == (self.num_envs,)
        votes, messages = [], []
        for agent_idx, agent in enumerate(self.agents):
            local_obs = self._local_batch(observation, agent_idx)
            starts = episode_start.repeat_interleave(self.num_agents) if self.shared_weights else episode_start
            vote, message = agent.propose(local_obs, starts, self.channel)
            votes.append(vote)
            messages.append(message)
        self._awaiting_step = True
        if self.shared_weights:
            return {"votes": votes[0].reshape(self.num_envs, self.num_agents, self.num_actions),
                    "messages": messages[0].reshape(self.num_envs, self.num_agents, self.message_dim)}
        return {"votes": torch.stack(votes, dim=1), "messages": torch.stack(messages, dim=1)}

    def update(self, rewards, dones, info=None, **kwargs):
        if not self._awaiting_step or info is None or "shared_body" not in info:
            raise RuntimeError("Shared-body update requires env.step feedback for the pending proposal")
        feedback = info["shared_body"]
        batch = {
            "action": feedback["action"][:, None].expand(-1, self.num_agents),
            "next_observation": feedback["next_observation"]}
        for agent_idx, agent in enumerate(self.agents):
            local = self._local_batch(batch, agent_idx)
            agent.record_body_step(local["action"], local["next_observation"])
        self._awaiting_step = False
        updated = []
        for agent in self.agents:
            local_rewards = rewards.repeat_interleave(self.num_agents) if self.shared_weights else rewards
            local_dones = dones.repeat_interleave(self.num_agents) if self.shared_weights else dones
            updated.append(agent.update(local_rewards, local_dones))
        if len(set(updated)) != 1:
            raise RuntimeError("Local agents reached different rollout update boundaries")
        if updated[0]:
            self._log_body_sensor_metrics()
            per_env = info["per_env"]
            for name in ("coverage", "effective_coverage", "wall_coverage", "walls_explored", "walls_total"):
                self.logger.log_scalar(name, sum(float(item[name]) for item in per_env) / len(per_env))
        self.logger.log_scalar("body/entropy", Categorical(probs=feedback["body_probs"]).entropy().mean().item())
        for agent_idx in range(self.num_agents):
            votes = feedback["votes"][:, agent_idx]
            self.logger.log_scalar(f"body/agent_{agent_idx}/vote_entropy", Categorical(probs=votes).entropy().mean().item())
        return updated[0]

    def _log_body_sensor_metrics(self):
        if self.shared_weights:
            predictions = self.agents[0].sensor_predictions
            # Completed episodes follow body/agent order, including asynchronous resets.
            left = tree_map(predictions, lambda value: value[0::self.num_agents])
            right = tree_map(predictions, lambda value: value[1::self.num_agents])
            left_config = right_config = self.agents[0].wm_model.config
        else:
            left, right = (agent.sensor_predictions for agent in self.agents[:2])
            left_config, right_config = (agent.wm_model.config for agent in self.agents[:2])
        assert torch.equal(left["mask"], right["mask"]), "Body predictions must align by episode and step"
        lr_rmse, lr_acc = masked_lr_metrics_logits(
            pred_left=left["prediction"][0], pred_right=right["prediction"][1],
            target_left=left["target"][..., 0], target_right=right["target"][..., 1], mask=left["mask"],
            min_left=float(left_config.sensor_min_idx[0]), min_right=float(right_config.sensor_min_idx[1]))
        self.logger.log_scalar("wm/metric/lr_acc", lr_acc.item())
        self.logger.log_scalar("wm/metric/lr_rmse", lr_rmse.item())
        for agent in self.agents:
            agent.sensor_predictions = None

    def _configuration(self):
        return {"num_agents": self.num_agents, "shared_weights": self.shared_weights,
                "channel": self.channel, "message_dim": self.message_dim, "num_actions": self.num_actions,
                "action_distribution": "dirichlet", "vote_concentration": self.concentration,
                "wm_transition": "own_vote_messages", "wm_body_action": self.wm_body_action}

    def get_state_dict(self):
        return {"shared_body_config": self._configuration(),
                "agents": [agent.get_state_dict() for agent in self.agents]}

    def load_state_dict(self, state):
        if state["shared_body_config"] != self._configuration():
            raise ValueError("Shared-body checkpoint configuration does not match this experiment")
        if len(state["agents"]) != len(self.agents):
            raise ValueError("Shared-body checkpoint has the wrong number of learners")
        for agent, agent_state in zip(self.agents, state["agents"]):
            agent.load_state_dict(agent_state)


def create_shared_body_agents(args, env, model_args, logger):
    from agent_utils_wm import create_world_model

    if args.wm_divergence_novelty_coef != 0:
        raise ValueError("Shared-body local rewards require --wm-divergence-novelty-coef 0")
    if args.env_reward_scale != 0:
        raise ValueError("Shared-body intrinsic experiments require --env-reward-scale 0")
    if args.wm_load_path:
        raise ValueError("Use a shared-body --checkpoint; full-sensor WM weights are not compatible")
    if args.max_steps < 3:
        raise ValueError("Shared-body WM training requires --max-steps >= 3")
    if not args.wm_body_action and (args.wm_turn_weight != 0 or args.wm_step_weight != 0):
        raise ValueError("Without body-action access, WM turn/step auxiliary losses must be disabled")
    agents = []
    transition_dim = env.num_actions + env.message_dim * (1 + env.peer_count) + env.peer_count
    if args.wm_body_action:
        transition_dim += 2
    learner_count = 1 if args.shared_weights else args.body_agents
    message_symbols = args.message_symbols if args.communication == "learned" else 0
    for agent_idx in range(learner_count):
        local_logger = AgentLogger(logger, "shared" if args.shared_weights else f"agent_{agent_idx}")
        model = create_world_model(
            model_args=model_args, device=torch.device(args.device), observation_type="shared-maze",
            observation_space=env.local_observation_space, maze_dim=env.max_dim,
            action_dim=transition_dim,
            turn_bins=len({turn for turn, _ in env.env.action_table}),
            step_bins=len({step for _, step in env.env.action_table}),
            contrastive_temp=args.wm_contrastive_temp, contrastive_horizon_discount=args.wm_contrastive_discount,
            contrastive_uncertainty_weight=args.wm_contrastive_uncertainty_weight,
            sensor_weight=args.wm_sensor_weight, loc_weight=args.wm_loc_weight,
            head_weight=args.wm_head_weight, turn_weight=args.wm_turn_weight, step_weight=args.wm_step_weight,
            sensor_sigma=args.wm_sensor_sigma, pos_sigma=args.wm_pos_sigma,
            heading_smoothing=args.wm_heading_smoothing, sensor_max_bin=args.wm_sensor_max_bin, logger=local_logger)
        feature_dim = model.get_feature_size()
        if args.communication == "state" and feature_dim != env.message_dim:
            raise ValueError("State-channel width must equal the WM feature size")
        head = WMActionHeadPolicy(env.num_actions + message_symbols, env.device, feature_dim)
        policy = SharedBodyPolicy(head, env.num_actions, message_symbols, args.vote_concentration)
        value = LocalValueHead(env.device, feature_dim)
        ppo = SharedBodyPPO(policy=policy, value=value, sampler=VoteMessageSampler(), joint=True,
                  policy_lr=args.policy_lr, num_envs=env.num_envs * (env.num_agents if args.shared_weights else 1),
                  discount=args.discount, logger=local_logger, device=env.device,
                  entropy_coef=args.entropy_coef)
        agent = LocalBodyAgent(
            agent=ppo, logger=local_logger, wm_model=model, action_table=env.env.action_table,
            wm_body_action=args.wm_body_action,
            intrinsic_reward_scale=args.intrinsic_reward_scale, env_reward_scale=0,
            wm_updates_per_policy=args.wm_updates_per_policy, wm_replay_capacity=args.wm_replay_capacity,
            wm_train_episodes=args.wm_train_episodes, wm_sensor_lp_reward_coef=args.wm_sensor_lp_reward_coef,
            wm_divergence_novelty_coef=0, wm_fixed=args.wm_fixed, sensor_max_bin=args.wm_sensor_max_bin,
            maze_dim=env.max_dim, wm_weight_decay=args.wm_weight_decay, wm_lr=args.wm_lr)
        agents.append(agent)
    return SharedBodyAgents(agents, env, args.communication, args.shared_weights, logger,
                            args.vote_concentration, args.wm_body_action)
