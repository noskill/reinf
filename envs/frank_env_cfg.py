from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg, EventCfg
from isaaclab.utils import configclass


# deterministic EventCfg: no randomization on reset
@configclass
class DeterministicEventCfg(EventCfg):
    # inherit init_franka_arm_pose, but drop the two reset‐time randomizers
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    reset_cube_positions  = EventTerm(
        func=franka_stack_events.set_object_poses,
        mode="reset",
        params={
            "asset_cfgs": [
               SceneEntityCfg("cube_1", pos=[0.4,0,0.0203]),
               SceneEntityCfg("cube_2", pos=[0.55,0.05,0.0203]),
               SceneEntityCfg("cube_3", pos=[0.60,-0.1,0.0203]),
            ]
        }
    )



# subclass the original env cfg and swap in our deterministic events
@configclass
class DeterministicFrankaCubeStackEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # run all of the original setup
        super().__post_init__()
        # replace the events with our deterministic ones
        self.events = DeterministicEventCfg()


