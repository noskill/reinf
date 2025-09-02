# RL Code-Base – High-Level Architecture  

This repository contains two closely related reinforcement-learning
code-bases:

1.  **A generic on-policy RL framework** (files in the repo root).  
    It implements REINFORCE, VPG, PPO and utilities such as logging,
    data storage and environment wrappers.
2.  **Task-specific sub-packages** (directories `mine/`, `envs/`,
    `grab/`, …) that provide environments, specialised agents and
    runnable training scripts.

The design follows the familiar “**env ↔ agent ↔ trainer**” loop while
keeping hard experiment-specific code contained in its own folder.

## 1. Core packages / modules

| Location               | Responsibility | Most relevant classes |
|------------------------|----------------|-----------------------|
| `reinforce.py`         | Lightweight on-policy **REINFORCE** baseline. Provides common helper code that more complex algorithms inherit from | `ReinforceBase` |
| `vpg.py`               | **VPG** (REINFORCE + baseline) built on top of `ReinforceBase` | `VPGBase`, `VPG` |
| `ppo.py`               | **PPO** implementation (clipped-ratio) | `PPOBase`, `PPO` |
| `mine/empowerment_ppo.py` | PPO variant that adds an inverse model and intrinsic empowerment rewards | `EmpowermentPPO`, `InverseModel` |
| `sample.py`            | Action samplers for discrete / Gaussian / transformed distributions | `DiscreteActionSampler`, `NormalActionSampler`, … |
| `pool.py`              | Episode buffer & replay helpers; mix-in used by all on-policy agents | `EpisodesPoolMixin`, `EpisodesOldPoolMixin` |
| `log.py`               | Thin wrapper around TensorBoard’s `SummaryWriter` with an  `episode_count` counter | `Logger` |
| `on_policy_train.py`, `mine/comm_trainer.py` | Trainers that drive the  *env ↔ agent* interaction loop, checkpointing, seeding, video capture | `OnPolicyTrainer`, `CommTrainer` |
| `envs/`, `mine/`, `grab/` | Task definitions (gym-style environments), tiny grid-worlds, IsaacGym tasks, etc. | e.g. `VecCommEnv`, `CustomFrankaStackEnv` |


## 2. Data flow (step-by-step)

```
┌──────┐      obs       ┌──────────┐  action  ┌─────────────┐
│ Env  │ ─────────────▶ │  Agent   │─────────▶│  Sampler    │
│      │ ◀───────────── │ (Policy) │ log π(a) │ (dist.sample)
└──────┘   r, done      └────┬─────┘          └────┬────────┘
                             │                   entropy
                             │ transitions        │
                             ▼                   ▼
                        Episodes buffer   (optional) inverse-model
                             │                   │
                             ▼                   ▼
                      Trainer `.should_learn()`  Empowerment reward
                             │
                             ▼
                     Compute returns / advantages
                             │
                             ▼
                        Optimise policy & value
```

Key points:

* **Sampler** decouples distribution logic from the policy network.
  All policies output *logits / means*; the sampler handles
  `dist = Categorical(logits)` etc.  This keeps algorithms agnostic to
  action space type.
* **EpisodesPoolMixin** collects `(s, a, log π, entropy, r)` tuples per
  environment.  Once *num_envs* episodes are complete `should_learn()`
  triggers a policy update.
* **EmpowermentPPO** augments rewards with

  ```text
  r_int = log q_ψ(a|s,s′)  +  β · H[π(·|s)]
  ```

  where `q_ψ` is the inverse model.  β is controlled by
  `--entropy-coef`.


## 3. Extending the code-base

1. **New environment**  
   Add a Gym-style env file (e.g. `envs/my_env.py`).  Ensure it exposes
   `observation_space` and `action_space`.

2. **New algorithm**  
   Derive from `ReinforceBase` (for Monte-Carlo policy-gradient family)
   or `PPOBase`.  Implement:
   * `learn_from_episodes()` – create batches, call `_prepare_batches` …
   * Optionally override `get_action` if you need special handling.

3. **Custom trainer / experiment script**  
   Copy `mine/train_comm_empower.py`, wire up env, networks, agent,
   `CommTrainer` (or your own), add CLI flags.


## 4. Directory overview

```
./envs/            IsaacGym & gymnasium environments
./mine/            Communication / empowerment toy tasks
./grab/            Object-manipulation task variants
./config/          Yaml / python hyper-parameter configs
./logs/            TensorBoard & checkpoints (git-ignored)
```


## 5. Important design choices

* **No global state** – agent has its own `Logger`, episode buffer and
  RNG seed.  Trainers just orchestrate.
* **Pure PyTorch** – no dependency on RL libraries (stable-baselines,
  RLlib), making the code easy to hack.
* **Explicit checkpoint dicts** – `get_state_dict` / `load_state_dict`
  implemented for every agent so that new fields can be added without
  breaking backward compatibility.
