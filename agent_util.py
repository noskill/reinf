import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces.box import Box
from reinforce import Reinforce
from vpg import VPG
from ppo import PPO
from sample import DiscreteActionSampler, NormalActionSampler

from networks import PolicyNetwork, ValueNetwork as Value
from transformer import LlamaConfig


def create_networks_with_transformer(
    obs_space,
    action_dim: int,
    *,
    hidden_dim=256,
    device='cpu',
    skill_dim: int = None,
    discrete_skills: bool = True,
):
    """Create transformer-based policy and value networks.

    - Policy: IsaacLabPolicy or IsaacLabSkillPolicy (DIAYN) outputs 2 * base_action_dim.
    - Value: IsaacLabValue mirrors the encoder stack without temporal decoding.
    """
    from tr_policy import IsaacLabPolicy, IsaacLabSkillPolicy, IsaacLabValue

    # Derive base action dimension for continuous actions (policy outputs 2*base)
    base_action_dim = action_dim // 2 if (action_dim % 2 == 0 and action_dim > 1) else action_dim

    # Infer n_cubes from observation space when possible
    n_cubes = 3
    try:
        if hasattr(obs_space, 'spaces') and 'cube_positions' in obs_space.spaces:
            cp = obs_space.spaces['cube_positions']
            if hasattr(cp, 'shape') and cp.shape[-1] % 3 == 0:
                n_cubes = cp.shape[-1] // 3
    except Exception:
        pass

    attention_window = 10
    head_dim = hidden_dim // 4
    cfg = LlamaConfig(
        input_size=hidden_dim,
        hidden_size=hidden_dim,
        intermediate_size=hidden_dim * 4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=True,
        head_dim=head_dim,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        use_rope=True,
        attention_window=attention_window,
        _attn_implementation="eager",
    )
    if skill_dim is not None:
        policy = IsaacLabSkillPolicy(
            cfg,
            n_cubes=n_cubes,
            action_dim=base_action_dim,
            skill_dim=skill_dim,
            discrete=discrete_skills,
            use_cube_id_emb=False,
            attention_window=attention_window,
        ).to(device)
    else:
        policy = IsaacLabPolicy(
            cfg,
            n_cubes=n_cubes,
            action_dim=base_action_dim,
            use_cube_id_emb=False,
            attention_window=attention_window,
        ).to(device)

    # Transformer-based value network without temporal decoding
    value = IsaacLabValue(cfg, n_cubes=n_cubes, use_cube_id_emb=False).to(device)

    return policy, value

# Utility functions for creating networks and agents
def create_networks(
    obs_dim: int,
    action_dim: int,
    hidden_dim=256,
    device='cpu',
    hidden_dims=None,
    value_hidden_dims=None,
    activation=nn.ReLU,
    value_activation=None,
    layer_norm=False,
    value_layer_norm=None,
    dropout=0.0,
    value_dropout=None,
):
    if hidden_dims is None:
        hidden_dims = [hidden_dim, hidden_dim]
    if value_hidden_dims is None:
        value_hidden_dims = hidden_dims
    if value_activation is None:
        value_activation = activation
    if value_layer_norm is None:
        value_layer_norm = layer_norm
    if value_dropout is None:
        value_dropout = dropout

    policy = PolicyNetwork(
        n_obs=obs_dim,
        n_action=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        layer_norm=layer_norm,
        dropout=dropout,
    ).to(device)

    value = Value(
        n_obs=obs_dim,
        hidden_dims=value_hidden_dims,
        activation=value_activation,
        layer_norm=value_layer_norm,
        dropout=value_dropout,
    ).to(device)

    return policy, value


def create_sampler(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscreteActionSampler()
    elif isinstance(action_space, gym.spaces.Box):
        shape = action_space.shape
        if len(shape) != 1:
            raise RuntimeError("unexpected action space size " + str(shape))
        return NormalActionSampler(shape[0], a_min=-20, a_max=20, transform=False)
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")


def create_agent(args_cli, env_cfg, env, logger):
    num_envs = env.num_envs
    # Determine observation and action dimensions
    policy_obs_space = env.unwrapped.single_observation_space["policy"]
    obs_space = env.observation_space
    action_space = env.action_space
    obs_dim = gym.spaces.flatdim(policy_obs_space)
    state_extractor = None

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    else:
        action_dim = action_space.shape[0] * 2  # For mean and std in continuous actions

    device = env_cfg.sim.device
    diayn_algorithms = ('ppodr', 'ppod', 'ppod_novel', 'ppodr_novel')
    uses_diayn = args_cli.algorithm in diayn_algorithms

    base_hidden_dims = (512, 512, 256, 256)
    diayn_hidden_dims = (1024, 1024, 512, 512)
    base_activation = nn.ReLU
    diayn_activation = nn.SiLU

    def build_networks(use_skill: bool = False, skill_dim: int = None, discrete_skills: bool = True):
        if isinstance(obs_space, Box):
            return create_networks(
                obs_dim,
                action_dim=action_dim,
                device=device,
                hidden_dim=256,
                value_hidden_dims=256,
                activation=base_activation,
                value_activation=base_activation
                )
        else:
            return create_networks_with_transformer(
                obs_space=policy_obs_space,
                action_dim=action_dim,
                device=device,
                hidden_dim=diayn_hidden_dims[0] if uses_diayn else base_hidden_dims[0],
                skill_dim=skill_dim if use_skill else None,
                discrete_skills=discrete_skills,
            )

    # Create sampler
    sampler = create_sampler(action_space)

    # Define hyperparameters
    policy_lr_default = 3e-4 if uses_diayn else 0.001
    value_lr_default = 3e-4 if uses_diayn else 0.001
    disc_lr_default = 1e-3 if uses_diayn else None

    policy_lr = args_cli.policy_lr if args_cli.policy_lr is not None else policy_lr_default
    value_lr = args_cli.value_lr if args_cli.value_lr is not None else value_lr_default
    disc_lr = args_cli.disc_lr if args_cli.disc_lr is not None else disc_lr_default

    discount = 0.99
    entropy_coef = 0.01

    common_args = dict(state_extractor=state_extractor)
    if args_cli.algorithm == "ppo":
        policy, value = build_networks()
        num_learning_epochs = 4
        agent = PPO(
            policy=policy,
            value=value,
            num_envs=num_envs,
            policy_lr=policy_lr,
            value_lr=value_lr,
            disc_lr=disc_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            num_learning_epochs=num_learning_epochs,
            sampler=sampler,
            logger=logger,
            **common_args
        )
    elif args_cli.algorithm == "vpg":
        policy, value = build_networks()
        agent = VPG(
            policy=policy,
            value=value,
            num_envs=num_envs,
            policy_lr=policy_lr,
            value_lr=value_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            sampler=sampler,
            logger=logger,
            **common_args
        )
    elif args_cli.algorithm in ('ppodr', "ppod", "ppod_novel", "ppodr_novel"):
        reward_scale = 10
        continious = args_cli.continious_skills
        if args_cli.algorithm == 'ppodr':
            from ppod import PPODRunning as PPOD
        elif args_cli.algorithm == 'ppod_novel':
            from ppod_novel import PPODNovel as PPOD
            from clustering import SmartClusteringNovelty
            common_args['novelty'] = SmartClusteringNovelty(reward_scale=reward_scale)
        elif args_cli.algorithm == 'ppodr_novel':
            from ppod_novel import PPODNovelRunning as PPOD
            from clustering import SmartClusteringNovelty
            common_args['novelty'] = SmartClusteringNovelty(reward_scale=reward_scale)
        else:
            from ppod import PPOD
        embedding_dim = args_cli.embedding_dim
        skill_dim = args_cli.skill_dim
        embedding_dim = max(embedding_dim * 2, 64)
        args_cli.embedding_dim = embedding_dim
        policy, value = build_networks(
            use_skill=True,
            skill_dim=skill_dim,
            discrete_skills=not continious,
        )
        num_learning_epochs = 4
        discard_steps = 20
        desc_fields = ['cube_positions', 'cube_orientations']
        from types import SimpleNamespace
        from tr_policy import IsaacLabDiscriminator
        cfg = SimpleNamespace(hidden_size=diayn_hidden_dims[0])
        n_cubes = 3
        if hasattr(policy_obs_space, 'spaces') and 'cube_positions' in policy_obs_space.spaces:
            cp = policy_obs_space.spaces['cube_positions']
            if hasattr(cp, 'shape') and cp.shape[-1] % 3 == 0:
                n_cubes = cp.shape[-1] // 3
        discriminator = IsaacLabDiscriminator(
            cfg,
            n_cubes=n_cubes,
            skill_dim=skill_dim,
            fields=desc_fields,
            use_cube_id_emb=False,
        ).to(device)
        agent = PPOD(
            policy=policy,
            value=value,
            obs_dim=obs_dim,
            num_envs=num_envs,
            policy_lr=policy_lr,
            embedding_dim=embedding_dim,
            value_lr=value_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            num_learning_epochs=num_learning_epochs,
            sampler=sampler,
            logger=logger,
            continious=continious,
            discriminator=discriminator,
            skill_dim=skill_dim,
            discriminator_fields=desc_fields,
            desc_discard_steps=discard_steps,
            disc_lr=disc_lr,
            **common_args
        )
    else:
        # Default to REINFORCE
        # For REINFORCE, the value network is not used
        # Adjust the action_dim accordingly
        if not isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.shape[0] * 2
        else:
            action_dim = action_space.n
        policy, _ = build_networks()
        agent = Reinforce(
            policy=policy,
            sampler=sampler,
            num_envs=num_envs,
            policy_lr=policy_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            logger=logger,
            **common_args
        )
    return agent
