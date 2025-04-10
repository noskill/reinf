import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from reinforce import Reinforce
from vpg import VPG
from ppo import PPO
from sample import DiscreteActionSampler, NormalActionSampler

from agent import Agent
from reinforce_main import PolicyNetwork
from vpg_main import Value
from util import StateExtractor


# Utility functions for creating networks and agents
def create_networks(obs_dim: int, action_dim: int, hidden_dim=256, device='cpu'):
    # Use the imported PolicyNetwork
    policy = PolicyNetwork(
        n_obs=obs_dim,
        n_action=action_dim,
        hidden_dim=hidden_dim
    ).to(device)
        
    # Use the imported Value network (named Value)
    value = Value(
        n_obs=obs_dim,
        hidden_dim=hidden_dim
    ).to(device)
        
    return policy, value


def create_sampler(action_space):
    shape = action_space.shape
    if len(shape) != 1:
        raise RuntimeError("unexpected action space size " + str(shape))
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscreteActionSampler()
    elif isinstance(action_space, gym.spaces.Box):
        return NormalActionSampler(shape[0], a_min=-20, a_max=20, transform=False)
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")


def create_agent(args_cli, env_cfg, env, logger):
    num_envs = env.num_envs
    # Determine observation and action dimensions
    obs_space = env.observation_space
    action_space = env.action_space
    obs_dim = gym.spaces.flatdim(obs_space)
    state_extractor = StateExtractor.from_dict_observation(obs_space)
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    else:
        action_dim = action_space.shape[0] * 2  # For mean and std in continuous actions

    # Create networks using imported classes
    device = env_cfg.sim.device
    hidden_dim = 556  # You can adjust this value
    policy, value = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device,
    )

    # Create sampler
    sampler = create_sampler(action_space)

    # Define hyperparameters
    policy_lr = 0.0001
    value_lr = 0.001
    discount = 0.99
    entropy_coef = 0.01

    common_args = dict(state_extractor=state_extractor)
    if args_cli.algorithm == "ppo":
        num_learning_epochs = 4
        agent = PPO(
            policy=policy,
            value=value,
            num_envs=num_envs,
            policy_lr=policy_lr,
            value_lr=value_lr,
            discount=discount,
            device=device,
            entropy_coef=entropy_coef,
            num_learning_epochs=num_learning_epochs,
            sampler=sampler,
            logger=logger,
            **common_args
        )
    elif args_cli.algorithm == "vpg":
        value_clip = 0.2
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
    elif args_cli.algorithm in ('ppodr', "ppod"):
        if args_cli.algorithm == 'ppodr':
            from ppod import PPODRunning as PPOD
        else:
            from ppod import PPOD
        from ppod import SkillDiscriminator
        embedding_dim = args_cli.embedding_dim
        continious = args_cli.continious_skills
        skill_dim = args_cli.skill_dim
        policy, value = create_networks(
            obs_dim=obs_dim + embedding_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        num_learning_epochs = 4
        state_extractor.add_field_at_end("skill", shape=(skill_dim if continious else 1,))
        
        sk_size = state_extractor.get_fields_size('skill')
        desc_fields = ['cube_positions', 'cube_orientations', 'eef_pos']
        desc_input_size = state_extractor.get_fields_size(desc_fields)

        # Discriminator supports continuous or discrete
        discriminator = SkillDiscriminator(
            desc_input_size, skill_dim, continuous=continious).to(device)
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
            desc_discard_steps=100,
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
