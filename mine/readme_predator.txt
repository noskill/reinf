# Predator-Prey Particle Environment with Communicating Predators


## Environment Description

Base Setup: 
A 2D space, either continuous (e.g., [0,1]²) or discrete (e.g., 10x10 grid),  with Three Predator Agents
Each can observe nearby entities (partial observability), 
move to chase the prey, and communicate with the others.
One Prey: Moves randomly (e.g., random walk) or evasively (e.g., moves away from nearby predators).

Objective: 

Predators aim to catch the prey (e.g., move within a capture radius, like 0.1 units, or land on the same grid cell).
Key Feature: Communication between predators should make prey capture significantly easier than without it, incentivizing emergent communication protocols.

Observation Space
Each Predator:
    Observes entities within a limited radius (e.g., 0.2 units in continuous space or 2 grid cells).
    Within this range, they see Prey presence Binary (prey in range or not).
    Prey attributes:
        Richer info, like relative position (“prey to the right”), velocity direction (“prey moving up”), or distance (“prey near/far”).
    Other predators’ positions: If within range, to enable coordination.

    Receives messages from the other two predators.
    Example: Predator 1 at (0.4, 0.6) with prey at (0.5, 0.7) might observe “prey present, to the right, moving up, near” and get messages like “prey left” from Predator 2.

Prey: 
    No observation (follows a simple policy, e.g., random walk or flee if a predator is within 0.3 units).

Action Space
Predators:
    Movement: Discrete (up, down, left, right) or continuous (velocity vector in 2D).
Communication: 
    Send discrete messages (e.g., 4 options: “prey near,” “prey far,” “prey up,” “prey left”) to the other predators.
    or 
    Send vector messages encoding necessary information.

Prey: 
    Random walk or evasive movement (e.g., move away from the nearest predator with some probability).

Communication
    Messages: Discrete for simplicity (e.g., 4-8 options like “prey near,” “prey up”). 
    Continuous messages are possible but complicate MI estimation.
Channel: Each predator broadcasts messages to the others. Messages are part of the state ( s ) for empowerment calculations.

Why Communication Helps: Without communication, each predator relies only on its limited observation radius, making it hard to track a moving prey. With communication, they can share info (e.g., “prey up” from Predator 1 informs Predator 2 to move up), enabling coordinated strategies like surrounding the prey.

Reward
Empowerment: 
    For each predator, compute ( MI(S', A | s) ), where:( S' ): Next state (e.g., predators’ positions, prey’s position, capture status).
    s : Current state (own position, local observations, messages from others).

    This rewards actions (movement + messages) that maximize control over the next state, ideally leading to prey capture.

Energy Constraint: each predator has an energy level (0 to 1).
Actions (moves, messages) cost energy (e.g., 0.05), and success probability scales with energy e.g., P(success) = energy.
 
Energy replenishes slowly (e.g., 0.01 per step) or fully on prey capture.

Optional Shared Reward: 
A small extrinsic reward for prey capture (e.g., +1 for all predators) could align their goals, but pure empowerment can test if communication emerges intrinsically.

Framework: 
    Use PettingZoo’s Multi-Agent Particle Environment (MPE), specifically Simple Tag, modified for three predators with partial observability.Install: pip install pettingzoo[mpe].
    Base code is Python-based, lightweight, and supports communication.

