"""
Utility functions for custom Isaac environment configurations.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from dataclasses import dataclass, field

@dataclass
class SceneEntityPoseCfg(SceneEntityCfg):
    """Extension of SceneEntityCfg that includes a target position for resetting objects."""
    pos: list[float] = field(default_factory=list)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
):
    """Set specified objects to given absolute positions with identity orientation and zero velocity."""
    if env_ids is None:
        return

    for cur_env in env_ids.tolist():
        for asset_cfg in asset_cfgs:
            asset = env.scene[asset_cfg.name]
            # Position relative to environment origin
            pose = asset_cfg.pos
            pose_tensor = torch.tensor([pose], device=env.device)
            positions = pose_tensor[:, :3] + env.scene.env_origins[cur_env, :3]
            # Identity orientation quaternion (w, x, y, z)
            orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(1, device=env.device),
                torch.zeros(1, device=env.device),
                torch.zeros(1, device=env.device),
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )