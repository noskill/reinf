from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg, EventCfg

# deterministic EventCfg: no randomization on reset
class DeterministicEventCfg(EventCfg):
    # inherit init_franka_arm_pose, but drop the two reset‐time randomizers
    randomize_franka_joint_state = None
    randomize_cube_positions   = None


# subclass the original env cfg and swap in our deterministic events
class DeterministicFrankaCubeStackEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # run all of the original setup
        super().__post_init__()
        # replace the events with our deterministic ones
        self.events = DeterministicEventCfg()


