# Multi-Agent Communication with Empowerment-Based Reinforcement Learning

## Overview
This project explores multi-agent reinforcement learning (RL) systems where agents communicate to optimize their empowerment, 
defined as the mutual information (MI) between an agent’s actions (or messages) and the subsequent state of the environment, 
given the current state. Specifically, we use MI-based intrinsic rewards—( MI(S', A | s) ) for agents with environmental actions
and ( MI(S', M | s) ) for agents with communication-only actions. 
The goal is to investigate how emergent communication protocols arise among agents to maximize their collective or 
individual control over the environment, potentially augmented by sparse extrinsic rewards from the
environment to guide task-specific behavior.

## Key Concepts

Multi-Agent Setup: 
Multiple agents operate in a shared environment, each with partial observability of the state space. Agents can take actions (e.g., moving in a grid) and/or send messages to one another, enabling coordination through bidirectional communication.

Empowerment as Intrinsic Motivation: Each agent aims to maximize its empowerment, measured as the MI between its actions/messages and the next state. This encourages agents to explore actions that increase their influence over the environment or other agents’ behaviors via communication.

Communication: 

Agents exchange discrete messages (e.g., directional signals or "null" for silence) to share information or coordinate actions. Communication is resource-constrained, with an energy cost to encourage efficiency and prevent redundant signaling.
Optional Extrinsic Rewards: To align empowerment with task-specific goals, experiments may incorporate small extrinsic rewards (e.g., for achieving a cooperative objective like reaching a goal or capturing a target), allowing us to compare purely intrinsic vs. hybrid reward strategies.

Resource Constraints: Actions and messages cost energy, with success rates tied to an agent’s energy level. This mimics real-world limitations and encourages strategic decision-making, reducing fixation on trivial or repetitive behaviors.

Objectives
Study how MI-based rewards lead to emergent communication protocols in multi-agent systems.
Investigate the impact of partial observability on agents’ ability to coordinate and maximize empowerment.
Evaluate the role of energy constraints in shaping efficient communication and action strategies.
Explore the effect of combining intrinsic (MI) and extrinsic (task-specific) rewards on agent behavior and cooperation.

Scope
The experiments are designed to be flexible across various environments, such as grid worlds, cooperative tasks, or dynamic pursuit scenarios. Unlike setups where only one agent can act, all agents may have the ability to take environmental actions, send messages, or both, depending on the specific environment. The focus is on understanding general principles of emergent communication and empowerment-driven learning in multi-agent RL, with applications to cooperative and competitive scenarios.

