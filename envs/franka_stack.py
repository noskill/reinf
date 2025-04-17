import torch
import gymnasium as gym
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import object_grasped, object_stacked
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnvCfg


class CustomFrankaStackEnv(ManagerBasedRLEnv):
    # Reward weights as class attributes
    grasp_reward_weight = 2.0
    stack_reward_weight = 5.0
    distance_reward_weight = 0.5
    
    def __init__(self, cfg: ManagerBasedRLEnvCfg = None, render_mode=None, env_cfg_entry_point=None, **kwargs):
        if cfg is None and env_cfg_entry_point is not None:
            cfg = env_cfg_entry_point()
        super().__init__(cfg, render_mode=render_mode)
        self.episode_flags = {}

    def reset(self, env_ids=None, seed=None, options=None):
        obs, info = super().reset(env_ids=env_ids, seed=seed, options=options)

        num_envs = self.num_envs if env_ids is None else len(env_ids)
        device = self.device
        # reset tracking flags at each episode reset
        self.episode_flags['grasped_cubes'] = torch.zeros((num_envs, 3), dtype=torch.bool, device=device) # 3 cubes
        self.episode_flags['stacked_cubes'] = torch.zeros((num_envs, 3, 3), dtype=torch.bool, device=device) # pairs of cubes
        self.episode_flags['previous_cube_to_cube_dist'] = torch.full((num_envs, 3, 3), float('inf'), device=device) 
        self.episode_flags['previous_min_dist'] = torch.full((num_envs,), float('inf'), device=device)

        return obs, info

    def compute_reward(self, obs):
        reward = torch.zeros(self.num_envs, device=self.device)  # Changed env to self

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
        not_grasped_any = ~self.episode_flags['grasped_cubes'].any(dim=1)  # Fixed reference to grasped_cubes
        distance_reward = self.distance_reward_weight * getting_closer.float() * not_grasped_any.float()
        reward += distance_reward
        
        # Update min distance tracker
        self.episode_flags['previous_min_dist'] = torch.minimum(self.episode_flags['previous_min_dist'], min_dist)

        # Convenience cfgs
        robot_cfg = SceneEntityCfg("robot")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        cube_cfgs = [SceneEntityCfg(f"cube_{i}") for i in range(1,4)]

        # ------ Grasp Rewards using built-in `mdp.object_grasped` -------
        for cube_idx, cube_cfg in enumerate(cube_cfgs):
            currently_grasped = object_grasped(self, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=cube_cfg)
            first_time_grasp = currently_grasped & ~self.episode_flags['grasped_cubes'][:, cube_idx]
            reward += first_time_grasp.float() * self.grasp_reward_weight
            self.episode_flags['grasped_cubes'][:, cube_idx] |= currently_grasped

        # ------ Stacking Rewards using built-in `mdp.object_stacked` -------
        for upper_idx, upper_cfg in enumerate(cube_cfgs):
            for lower_idx, lower_cfg in enumerate(cube_cfgs):
                if upper_idx == lower_idx:
                    continue
                is_stacked = object_stacked(self, robot_cfg=robot_cfg, upper_object_cfg=upper_cfg, lower_object_cfg=lower_cfg)
                first_time_stacked = is_stacked & ~self.episode_flags['stacked_cubes'][:, upper_idx, lower_idx]
                reward += first_time_stacked.float() * self.stack_reward_weight
                self.episode_flags['stacked_cubes'][:, upper_idx, lower_idx] |= is_stacked

        # ------ Distance Reward for Moving Grasped Cube closer to another Cube -------
        cube_positions = obs['policy']['cube_positions'].reshape(self.num_envs, 3, 3) # num_envs, num_cubes, xyz
        for grasped_idx in range(3): # Grasped cube
            grasped_mask = self.episode_flags['grasped_cubes'][:, grasped_idx]
            if grasped_mask.any():
                for target_idx in range(3): # Cube to stack upon
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

        return reward
    
    def step(self, action: torch.Tensor):
        """Override step to use our custom reward calculation."""
        # Call the parent step method but ignore its reward calculation
        obs, _, terminated, truncated, info = super().step(action)
        
        # Calculate rewards using your custom method
        rewards = self.compute_reward(obs)
        
        # Return with your custom rewards
        return obs, rewards, terminated, truncated, info
