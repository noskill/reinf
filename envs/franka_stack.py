import time
import torch
import gymnasium as gym
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import object_stacked
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
# Re-exported env config class
from isaaclab.envs import ManagerBasedRLEnvCfg

# util for sensor-based grasp detection
from .utils import _object_grasped_via_sensors

# Sensors
from isaaclab.sensors import ContactSensor, ContactSensorCfg


class CustomFrankaStackEnv(ManagerBasedRLEnv):
    # Reward weights as class attributes
    grasp_reward_weight = 2.0
    stack_reward_weight = 10.0
    distance_reward_weight = 0.5

    def __init__(self, cfg: ManagerBasedRLEnvCfg = None, render_mode=None, env_cfg_entry_point=None,
                         reward_on: bool = True,
                         **kwargs):
        if cfg is None and env_cfg_entry_point is not None:
            cfg = env_cfg_entry_point()
        super().__init__(cfg, render_mode=render_mode)
        # ------------------------------------------------------------------
        #  Instantiate contact sensors on both gripper fingers
        # ------------------------------------------------------------------
        # Create configs (history_length=1 to minimize memory). We keep update_period=0
        # to refresh every physics step.
        left_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            history_length=1,
            update_period=0.0,
            debug_vis=False,
            track_pose=True,
        )
        right_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            history_length=1,
            update_period=0.0,
            debug_vis=False,
            track_pose=True,
        )

        # Instantiate sensors and register into the scene (under keys matching
        # the SceneEntityCfg names used elsewhere).
        self._left_finger_sensor = ContactSensor(left_sensor_cfg)
        self._right_finger_sensor = ContactSensor(right_sensor_cfg)

        # Explicitly initialize sensors to allocate required buffers before the
        # environment reset logic accesses them. Normally, initialization is
        # triggered on simulator *PLAY* via internal callbacks, but the
        # ManagerBasedRLEnv workflow queries sensor data during the initial
        # reset that occurs prior to the first play event. Calling the
        # protected initialization routine here ensures the sensors are ready.
        self._left_finger_sensor._initialize_impl()
        self._right_finger_sensor._initialize_impl()

        # Expose via scene for easy access through SceneEntityCfg
        self.scene.sensors["left_finger_sensor"] = self._left_finger_sensor
        self.scene.sensors["right_finger_sensor"] = self._right_finger_sensor

        # ------------------------------------------------------------------
        #  Scene-entity configurations used throughout the environment
        # ------------------------------------------------------------------
        # Pre-define commonly accessed entities. These map directly to
        #   - rigid-body robot model
        #   - end-effector coordinate frame
        #   - the three colored stacking cubes
        # Keeping these configs in one place avoids re-creating them in the
        # hot-loop of reward computation.
        #
        # Note: Previously we attempted to manually apply the
        # `PhysxContactReportAPI` to the finger bodies at runtime. This is no
        # longer necessary because the dedicated contact–sensor prims already
        # author the required API schemas inside the USD. Removing the dynamic
        # authoring step eliminates an `AttributeError` that occurred in
        # setups where the PhysX extension is unavailable during import time.

        self.robot_cfg = SceneEntityCfg("robot")
        self.ee_frame_cfg = SceneEntityCfg("ee_frame")
        self.cube_cfgs = [SceneEntityCfg(f"cube_{i}") for i in range(1,4)]
        # contact sensor configurations for the two gripper fingers (must exist in scene)
        # Note: The scene is expected to include contact sensors attached to each finger
        # accessible via these entity names. No fallback logic is provided – if sensors
        # are missing, an explicit exception is raised to highlight the mis-configuration.
        self.left_finger_sensor_cfg = SceneEntityCfg("left_finger_sensor")
        self.right_finger_sensor_cfg = SceneEntityCfg("right_finger_sensor")
        self.episode_flags = {}
        self.reward_on = reward_on
        self.prev_cube_positions = None
        self._cube_velocities = None
        self.initial_cube_positions = None

    def reset(self, env_ids=None, seed=None, options=None):
        obs, info = super().reset(env_ids=env_ids, seed=seed, options=options)

        num_envs = self.num_envs if env_ids is None else len(env_ids)
        device = self.device

        # Reset tracking flags at each episode reset
        self.episode_flags['grasped_cubes'] = torch.zeros((num_envs, 3), dtype=torch.bool, device=device)
        self.episode_flags['stacked_cubes'] = torch.zeros((num_envs, 3, 3), dtype=torch.bool, device=device)
        self.episode_flags['previous_cube_to_cube_dist'] = torch.full((num_envs, 3, 3), float('inf'), device=device)
        self.episode_flags['previous_min_dist'] = torch.full((num_envs,), float('inf'), device=device)

        # Initialize event tensors in info
        info['grasp_events'] = torch.zeros((num_envs, 3), dtype=torch.bool, device=device)
        info['stack_events'] = torch.zeros((num_envs, 3, 3), dtype=torch.bool, device=device)

        cube_positions = obs['policy']['cube_positions']
        self.prev_cube_positions = cube_positions.clone()
        cube_velocities = torch.zeros_like(cube_positions)
        obs['policy']['cube_velocities'] = cube_velocities
        self._cube_velocities = cube_velocities.clone()

        if self.initial_cube_positions is None or self.initial_cube_positions.shape != cube_positions.shape:
            self.initial_cube_positions = cube_positions.clone()
        else:
            if env_ids is None or cube_positions.shape[0] == self.initial_cube_positions.shape[0]:
                self.initial_cube_positions.copy_(cube_positions)
            else:
                if isinstance(env_ids, torch.Tensor):
                    target_ids = env_ids.to(dtype=torch.long, device=cube_positions.device)
                else:
                    target_ids = torch.as_tensor(env_ids, dtype=torch.long, device=cube_positions.device)
                self.initial_cube_positions[target_ids] = cube_positions.clone()

        return obs, info

    def compute_reward(self, obs, info):
        """Update event flags and info dict with events using tensors."""
        reward = torch.zeros(self.num_envs, device=self.device)

        # Initialize event tensors if they don't exist in info
        if 'grasp_events' not in info:
            # Shape: [num_envs, num_cubes]
            info['grasp_events'] = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)

            # Shape: [num_envs, upper_cube, lower_cube]
            info['stack_events'] = torch.zeros((self.num_envs, 3, 3), dtype=torch.bool, device=self.device)

        # Extract needed observations
        gripper_to_cube_1 = obs['policy']['object'][:, 21:24]
        gripper_to_cube_2 = obs['policy']['object'][:, 24:27]
        gripper_to_cube_3 = obs['policy']['object'][:, 27:30]

        # Compute min distance to any cube
        dist1 = torch.norm(gripper_to_cube_1, dim=1)
        dist2 = torch.norm(gripper_to_cube_2, dim=1)
        dist3 = torch.norm(gripper_to_cube_3, dim=1)
        min_dist = torch.min(torch.stack([dist1, dist2, dist3]), dim=0).values

        # Reward for reducing distance to any cube if not grasped
        getting_closer = min_dist < self.episode_flags['previous_min_dist']
        not_grasped_any = ~self.episode_flags['grasped_cubes'].any(dim=1)
        distance_reward = self.distance_reward_weight * getting_closer.float() * not_grasped_any.float()
        reward += distance_reward

        # Update min distance tracker
        self.episode_flags['previous_min_dist'] = torch.minimum(self.episode_flags['previous_min_dist'], min_dist)

        # ------ Grasp Events and Rewards -------
        for cube_idx, cube_cfg in enumerate(self.cube_cfgs):
            # Determine grasp via dedicated contact sensors
            currently_grasped = _object_grasped_via_sensors(
                self,
                object_cfg=cube_cfg,
                left_sensor_cfg=self.left_finger_sensor_cfg,
                right_sensor_cfg=self.right_finger_sensor_cfg,
            )
            first_time_grasp = currently_grasped & ~self.episode_flags['grasped_cubes'][:, cube_idx]
            # (Debug breakpoint removed)

            # Add reward
            reward += first_time_grasp.float() * self.grasp_reward_weight

            # Update event tensor
            info['grasp_events'][:, cube_idx] |= first_time_grasp

            # Update flags
            self.episode_flags['grasped_cubes'][:, cube_idx] |= currently_grasped

        # ------ Stacking Events and Rewards -------
        for upper_idx, upper_cfg in enumerate(self.cube_cfgs):
            for lower_idx, lower_cfg in enumerate(self.cube_cfgs):
                if upper_idx == lower_idx:
                    continue
                is_stacked = object_stacked(self, robot_cfg=self.robot_cfg, upper_object_cfg=upper_cfg, lower_object_cfg=lower_cfg)
                first_time_stacked = is_stacked & ~self.episode_flags['stacked_cubes'][:, upper_idx, lower_idx]

                # Add reward
                reward += first_time_stacked.float() * self.stack_reward_weight

                # Update event tensor
                info['stack_events'][:, upper_idx, lower_idx] |= first_time_stacked

                # Update flags
                self.episode_flags['stacked_cubes'][:, upper_idx, lower_idx] |= is_stacked

        # ------ Distance Reward for Moving Grasped Cube closer to another Cube -------
        cube_positions = obs['policy']['cube_positions'].reshape(self.num_envs, 3, 3)  # num_envs, num_cubes, xyz
        for grasped_idx in range(3):  # Grasped cube
            grasped_mask = self.episode_flags['grasped_cubes'][:, grasped_idx]
            if grasped_mask.any():
                for target_idx in range(3):  # Cube to stack upon
                    if target_idx == grasped_idx:
                        continue
                    # Compute pairwise distance
                    pos_grasped_cube = cube_positions[:, grasped_idx, :]
                    pos_target_cube = cube_positions[:, target_idx, :]
                    dist = torch.norm(pos_grasped_cube - pos_target_cube, dim=1)

                    # Check improvement over previous minimal
                    improved = dist < self.episode_flags['previous_cube_to_cube_dist'][:, grasped_idx, target_idx]
                    distance_reward = grasped_mask & improved & (~self.episode_flags['stacked_cubes'][:, grasped_idx, target_idx])

                    reward += distance_reward.float() * self.distance_reward_weight

                    # Update previous minimal distance
                    new_min_dist = torch.minimum(dist, self.episode_flags['previous_cube_to_cube_dist'][:, grasped_idx, target_idx])
                    self.episode_flags['previous_cube_to_cube_dist'][:, grasped_idx, target_idx] = new_min_dist

        # Add success information
        info['success'] = self.episode_flags['stacked_cubes'].any(dim=(1,2))

        # Add aggregate statistics
        info['grasp_success_rate'] = self.episode_flags['grasped_cubes'].any(dim=1).float().mean().item()
        info['stack_success_rate'] = self.episode_flags['stacked_cubes'].any(dim=(1,2)).float().mean().item()

        return reward, info

    def step(self, action: torch.Tensor):
        """Override step to compute events and apply rewards conditionally."""
        obs, rewards, terminated, truncated, info = super().step(action)
        cube_positions = obs['policy']['cube_positions']
        if self.prev_cube_positions is None or self.prev_cube_positions.shape != cube_positions.shape:
            cube_velocities = torch.zeros_like(cube_positions)
        else:
            delta = cube_positions - self.prev_cube_positions
            cube_velocities = delta / self.step_dt
            reset_mask = (self.episode_length_buf == 0).unsqueeze(-1)
            cube_velocities = torch.where(reset_mask, torch.zeros_like(cube_velocities), cube_velocities)
        self.prev_cube_positions = cube_positions.clone()
        obs['policy']['cube_velocities'] = cube_velocities
        self._cube_velocities = cube_velocities.clone()

        if self.initial_cube_positions is not None and self.initial_cube_positions.shape == cube_positions.shape:
            num_cubes = cube_positions.shape[1] // 3
            displacement = (cube_positions - self.initial_cube_positions).view(cube_positions.shape[0], num_cubes, 3).norm(dim=2)
            info['cube_displacement'] = displacement
        else:
            num_cubes = cube_positions.shape[1] // 3
            info['cube_displacement'] = torch.zeros((cube_positions.shape[0], num_cubes), device=cube_positions.device)

        if hasattr(self, "episode_length_buf"):
            info['episode_length_buf'] = self.episode_length_buf.clone()
        rewards_calc, info = self.compute_reward(obs, info)
        if self.reward_on:
            rewards = rewards_calc
        return obs, rewards, terminated, truncated, info
