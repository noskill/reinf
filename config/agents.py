import os
from dataclasses import dataclass, asdict
from isaaclab.utils import configclass
from typing import List, Optional
from datetime import datetime
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@configclass
class LoggingCfg:
    root_dir: str = "logs"
    run_name: Optional[str] = None
    
    def get_experiment_dir(self, algorithm_type: str, experiment_name: str) -> str:
        """Creates and returns the experiment directory path."""
        log_root_path = os.path.abspath(os.path.join(
            self.root_dir, 
            algorithm_type, 
            experiment_name
        ))
        
        # Create timestamp-based directory name
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.run_name:
            log_dir += f"_{self.run_name}"
            
        return os.path.join(log_root_path, log_dir)


@configclass
class PolicyNetworkCfg:
    hidden_dims: List[int] = [256, ]
    activation: str = "relu"
    init_noise_std: float = 1.0

@configclass
class ValueNetworkCfg:
    hidden_dims: List[int] = [256, ]
    activation: str = "relu"


@configclass
class BaseAlgorithmCfg:
    policy_lr: float = 1.0e-3
    discount: float = 0.99
    max_grad_norm: float = 1.0
    num_envs: int = 8
    n_episodes: int = 5000


@configclass
class ReinforceAlgorithmCfg(BaseAlgorithmCfg):
    type: str = "reinforce"
    entropy_coef: float = 0.01

@configclass
class VPGAlgorithmCfg(ReinforceAlgorithmCfg):
    type: str = "vpg"
    entropy_coef: float = 0.01

@configclass
class PPOAlgorithmCfg(ReinforceAlgorithmCfg):
    type: str = "ppo"
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    lam: float = 0.95
    desired_kl: float = 0.01

    
@configclass
class SamplerCfg:
    type: str = "normal"  # or "discrete"
    action_min: float = -2.0
    action_max: float = 2.0


@configclass
class TrainingCfg:
    experiment_name: str = "default"
    save_interval: int = 50
    algorithm: BaseAlgorithmCfg = PPOAlgorithmCfg()
    policy: PolicyNetworkCfg = PolicyNetworkCfg()
    value: Optional[ValueNetworkCfg] = ValueNetworkCfg()
    logging: LoggingCfg = LoggingCfg()  # Add logging configuration
    sampler: SamplerCfg = SamplerCfg()  # Add sampler config

    def to_omegaconf(self):
        """Convert the config to an OmegaConf object"""
        return OmegaConf.create(asdict(self))

    @classmethod
    def from_omegaconf(cls, conf):
        """Create a config instance from an OmegaConf object"""
        return cls(**OmegaConf.to_container(conf))


@configclass
class CartpoleReinforceCfg(TrainingCfg):
    experiment_name: str = "cartpole_reinforce"
    algorithm: ReinforceAlgorithmCfg = ReinforceAlgorithmCfg(
        policy_lr=1.0e-3,
        entropy_coef=0.01,
    )
    policy: PolicyNetworkCfg = PolicyNetworkCfg(
        hidden_dims=[64],
        activation="elu",
    )
    sampler: SamplerCfg = SamplerCfg(
        type="normal",
        action_min=-2.0,
        action_max=2.0
    )


# Example specific configurations
@configclass
class CartpolePPOCfg(CartpoleReinforceCfg):
    experiment_name: str = "cartpole_ppo"
    algorithm: PPOAlgorithmCfg = PPOAlgorithmCfg(
        policy_lr=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
    )

    value: ValueNetworkCfg = ValueNetworkCfg(
        hidden_dims=[64],
        activation="elu",
    )
    sampler: SamplerCfg = SamplerCfg(
        type="normal",
        action_min=-2.0,
        action_max=2.0
    )


@configclass
class CartpoleVGPCfg(CartpolePPOCfg):
    experiment_name: str = "cartpole_vpg"
    algorithm: VPGAlgorithmCfg = VPGAlgorithmCfg(
        policy_lr=1.0e-3,
        entropy_coef=0.01,
    )


# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainingCfg)
cs.store(name="cartpole_ppo", node=CartpolePPOCfg)
cs.store(name="cartpole_reinforce", node=CartpoleReinforceCfg)

