import torch


def cube_positions_centered(env, cube_cfgs=None, name=None):
    if cube_cfgs is None:
        cube_cfgs = getattr(env, "cube_cfgs", [])
    if not cube_cfgs:
        cube_cfgs = [f"cube_{i}" for i in range(1, 4)]

    origins = env.scene.env_origins
    positions = []
    for cfg in cube_cfgs:
        name = cfg if isinstance(cfg, str) else cfg.name
        cube = env.scene[name]
        positions.append(cube.data.root_pos_w - origins)
    return torch.cat(positions, dim=1)


def cube_velocities_from_history(env, name=None):
    """Return cached cube velocities for each environment.

    This relies on :class:`CustomFrankaStackEnv` populating ``env._cube_velocities``
    during reset/step via finite differencing of cube positions.
    """

    velocities = getattr(env, "_cube_velocities", None)
    if velocities is None:
        cube_count = len(getattr(env, "cube_cfgs", [])) or 3
        return torch.zeros((env.num_envs, cube_count * 3), device=env.device)
    return velocities.clone()
