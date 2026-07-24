# Repository Instructions

These instructions apply to the reinforcement-learning experimentation codebase.

## Architecture

- Preserve the existing lean, mostly flat structure of core algorithms such as PPO, VPG, and REINFORCE.
- Avoid unnecessary abstractions, but introduce shared modules or base APIs when they remove real duplication or unify behavior across the codebase.
- Keep infrastructure components in the appropriate existing areas such as `wm/`, `envs/`, and `networks.py`.

## Tensor and Failure Handling

- Fail early when required state, parameters, indices, or tensors are missing or malformed. Raise an informative exception or assert the expected shape.
- Do not use zero tensors, dummy values, or fallback branches to hide missing or invalid inputs. Intentional zero initialization is allowed when it is part of the algorithm.
- Do not add fallbacks supporting multiple guessed tensor shapes. Assert the single expected shape and keep backend APIs consistent.
- Handle padding and invalid sequence steps with Boolean indexing or attention masks rather than arithmetic zero fallbacks.

## Code Structure

- Do not create parallel wrapper functions that duplicate an existing computation with a minor mathematical change.
- Prefer one shared implementation. Extend an existing pipeline, helper, or base class when that keeps the API clear.
- Do not add configuration flags to a function when a small shared helper or cleaner interface would be simpler.

## Workflow

- Before a non-trivial algorithm, API, architecture, or gradient-flow change, present a concise design describing the affected variables, mathematics, and files, and wait for approval.
- An explicit user request to apply a specific fix counts as approval. Small local fixes do not require a separate design phase.
- When changing a differentiable path, verify that intended gradients remain connected and that detach operations are deliberate.
