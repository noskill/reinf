# RL Code-Base – High-Level Architecture

This document describes the reinforcement-learning stack. It mixes a lightweight on-policy RL framework with IsaacLab
task integrations and DIAYN-style diversity exploration.

Two layers make up the project:

1. **On-policy baseline algorithms** – REINFORCE, VPG, PPO, and shared
   utilities (`reinforce.py`, `vpg.py`, `ppo.py`, `sample.py`, `pool.py`).
2. **Skill-based extensions and experiments** – DIAYN/PPOD agents,
   novelty clustering, IsaacLab trainers, and task-specific code.

The training loop still follows the classic **env ⇄ agent ⇄ trainer**
structure, but DIAYN adds an auxiliary discriminator and skill sampling
path inside the agent.

## 1. Core Packages and Modules

| Location | Responsibility | Key classes / functions |
|----------|----------------|-------------------------|
| `reinforce.py`, `vpg.py`, `ppo.py` | Baseline on-policy agents with EpisodeBatch padding/flattening and sequence-aware PPO updates | `Reinforce`, `VPG`, `PPO` |
| `ppod.py` | DIAYN-enabled PPO variants with skill sampling, dict-obs passthrough, and discriminator training | `PPOD`, `PPODRunning` |
| `ppod_novel.py` | Adds novelty rewards on top of PPOD via a reusable reward provider interface | `NoveltyReward`, `AdditionalRewardMixin`, `PPODNovel`, `PPODNovelRunning` |
| `agent_util.py` | Factory that builds transformer policies/values/discriminators and instantiates the right agent based on CLI flags | `create_agent`, `create_networks_with_transformer` |
| `sample.py` | Action distribution wrappers decoupled from policies | `DiscreteActionSampler`, `NormalActionSampler` |
| `pool.py` | Episode buffers with skill bookkeeping for DIAYN | `EpisodesPoolMixin`, `PPODPool` |
| `clustering.py` | Novelty score computation and clustering utilities | `SmartClusteringNovelty` |
| `util.py` | Shared helpers for observation extraction and experiment hygiene | `StateExtractor`, `copy_python_sources` |
| `log.py` | Thin TensorBoard/console logger with episode tracking | `Logger` |
| `on_policy_train.py`, `reinforce_main_is.py` | IsaacLab/Hydra entrypoint, trainer loop, checkpointing, and experiment orchestration | `OnPolicyTrainer`, `main()` |
| `envs/`, `mine/`, `grab/` | Task registrations and gym-style environments | e.g. IsaacLab manager configs |


## 2. Training and Data Flow

```
┌────────┐   obs    ┌────────┐  skill idx/vector ┌────────────┐
│  Env   │ ───────▶ │ Agent  │ ◀─────────────────│ Skill pool │
│        │ ◀────────│ Policy │─── action ───────▶│  sampler   │
└────────┘    r     └──┬─────┘                   └────┬───────┘
                       │ episodes + descriptors      │
                       ▼                             ▼
                  Episodes buffer             DIAYN discriminator
                       │                             │
                       ▼                             ▼
               Trainer & optimiser ⟹ policy/value updates
                                        │
                                        ▼
                          Novelty clustering & diagnostics
```

Highlights:

- **Observation flow** – Dict observations are passed through as-is to
  policy/value/discriminator. Episode boundaries are preserved via
  `EpisodeBatch` padding + `key_padding_mask` for sequence models.
- **Skill management** – `PPOD` samples skills per environment and attaches
  them under the `skills` key during rollout; skill-aware transformer
  policies embed them internally.
- **DIAYN discriminator** – Receives dict observation fields (configurable
  via `discriminator_fields`) and predicts the skill, providing intrinsic
  rewards and accuracy diagnostics. Target/online networks stay synchronised
  when checkpoints resume.
- **Novelty branch** – `PPODNovel*` variants use `NoveltyReward` as an
  additional reward provider layered on top of DIAYN and update novelty
  state after training.
- **Transformer history window** – Transformer models support a tunable
  attention window for training masks and inference cache, keeping only the
  most recent steps when enabled.


## 3. Extending the Code-Base

1. **Add a new environment** – Create a Gym/IsaacLab definition under
   `envs/` (or an experiment-specific subfolder) and ensure it exposes
   dict observations compatible with transformer policies.
2. **Create a new algorithm** – Derive from `PPOBase`/`Reinforce` or reuse
   `PPOD` mixins. Implement `learn_from_episodes()` and wire any auxiliary
   modules (e.g. new intrinsic rewards) inside `agent_util.create_agent`.
3. **Experiment entrypoint** – Update `reinforce_main_is.py` or add a new
   CLI wrapper. Use `copy_python_sources()` so each run captures source
   snapshots alongside checkpoints and logs.


## 4. Directory Overview

```
agent*.py, ppod*.py     Core agents and DIAYN extensions
clustering/             Novelty-based reward helpers
config/                 IsaacLab/Hydra experiment presets
envs/, grab/, mine/     Environment implementations and task suites
logs/, runs/            TensorBoard and checkpoints (git-ignored)
utils/                  Ancillary tooling (e.g. visualisation)
```


## 5. Design Principles

- **Modular agents** – Policies, value functions, discriminators, and
  samplers remain interchangeable via `create_agent`.
- **Skill flexibility** – Discrete and continuous skills are supported via
  `IsaacLabSkillPolicy` embedding and DIAYN reward computation.
- **Experiment reproducibility** – CLI arguments are snapshotted and
  relevant sources are copied into each run directory.
- **Checkpoint robustness** – Novelty and discriminator networks sync their
  target/online weights during save/load to avoid divergence after resume.

This summary should help new contributors locate the right extension points
and understand how DIAYN novelty augments the underlying on-policy baselines.
