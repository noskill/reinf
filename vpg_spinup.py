import gymnasium as gym

from spinup import vpg_pytorch
import torch

# Create the environment
env_fn = lambda : gym.make('Pendulum-v1')

# Set the hyperparameters
ac_kwargs = dict(hidden_sizes=[64,64])

# Train the agent
vpg_pytorch(
    env_fn=env_fn,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=4000,
    epochs=250,
    gamma=0.99,
    pi_lr=3e-4,
    vf_lr=1e-3,
    max_ep_len=200,
    seed=0
)
