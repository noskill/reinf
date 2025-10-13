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


# -----------------------------------------------------------------------------
#  Grasp detection using contact sensors
# -----------------------------------------------------------------------------


def _object_grasped_via_sensors(
    env: "ManagerBasedEnv",
    *,
    object_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
    sensor_proximity_thresh: float = 0.05,
) -> torch.Tensor:
    """Return boolean mask indicating if *object* is firmly grasped.

    The object is considered grasped when both finger contact sensors register
    forces exceeding *force_threshold* and the object lies within
    *diff_threshold* distance from the end-effector frame.

    The scene **must** contain the sensors referenced by *left_sensor_cfg* and
    *right_sensor_cfg*. A RuntimeError is raised otherwise — this is intentional
    to surface configuration errors early.
    """

    # Resolve sensor entities -------------------------------------------------
    try:
        left_sensor = env.scene[left_sensor_cfg.name]
        right_sensor = env.scene[right_sensor_cfg.name]
    except KeyError as exc:
        raise RuntimeError(
            "Required gripper contact sensors are missing from the scene. "
            f"Expected sensors named '{left_sensor_cfg.name}' and "
            f"'{right_sensor_cfg.name}'."
        ) from exc

    obj = env.scene[object_cfg.name]
    ee_frame = env.scene["ee_frame"]  # available in all envs using this util

    # 1. Contact forces -------------------------------------------------------
    left_force = left_sensor.data.net_forces_w[:, 0, :]
    right_force = right_sensor.data.net_forces_w[:, 0, :]

    left_contact = torch.linalg.vector_norm(left_force, dim=1) > force_threshold
    right_contact = torch.linalg.vector_norm(right_force, dim=1) > force_threshold

    # 2. Proximity between object and end-effector ----------------------------
    obj_pos = obj.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    close_enough = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1) < diff_threshold

    # 3. Object proximity to each finger --------------------------------------
    # Sensors must be configured with `track_pose=True` to expose pos_w field.
    left_sensor_pos = left_sensor.data.pos_w[:, 0, :]
    right_sensor_pos = right_sensor.data.pos_w[:, 0, :]

    near_left = torch.linalg.vector_norm(obj_pos - left_sensor_pos, dim=1) < sensor_proximity_thresh
    near_right = torch.linalg.vector_norm(obj_pos - right_sensor_pos, dim=1) < sensor_proximity_thresh

    return left_contact & right_contact & close_enough & near_left & near_right


# Public alias without leading underscore, in case other modules want it
object_grasped_via_sensors = _object_grasped_via_sensors
