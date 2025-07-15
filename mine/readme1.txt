Two Observer Agents: These agents can see parts of the observation space but can’t act directly on the environment. They can send messages to each other and to the third agent.
One Actor Agent: This agent can take actions that affect the environment and can communicate with the two observers.
Goal: Test how communication emerges and whether intrinsic motivation (like empowerment) drives useful behavior.




Grid: 10x10 grid with a moving prey (random walk, one step per turn) and an actor (predator) trying to catch it.

Agents:
Observer 1: Sees prey’s x-coordinate.
Observer 2: Sees prey’s y-coordinate.
Actor: Sees its own position (x, y) and moves to catch the prey.


State Space:

Observer 1: s1=(x_prey, m_actor, m2)
Observer 2: s2=(y_prey, m_actor, m1)
Actor: s_actor=(x_actor, y_actor, m1, m2)


Next state ( S' ): Includes new positions (prey, actor) and new messages (m1′,m2′,mactor′).



Action Space:

Observers: Observers: Send a message m_i ∈ {“left”,“right”,“up”,“down”,“stay”,“null”}. “Null” means no message is sent.


Actor: A=(move,m_actor)
where move ∈ {up,down,left,right}
m_actor ∈ {“left”,“right”,“up”,“down”,“stay”,“null”}.



Communication:

Bidirectional: Observers send to actor and each other; actor sends to both observers.
“Null” message means no signal is transmitted, and receivers see a default value (e.g., a zero vector in the state).

Energy Constraint:
Message cost: 0.05 energy (except “null,” which costs 0).
Move cost (actor only): 0.1 energy.
Success rate: 100% if energy > 0.25, else 50%.
Recharge: 0.02 per step.
catching prey increases energy to 100%

Reward:
Actor: ( MI(s_actor', A | s) ), where A=(move, m_actor)

Observers: ( MI(S_i', M | s) ), where ( M ) is their message.


Energy Check:Sending a non-null message costs 0.05 energy. “Null” costs 0, making it a viable choice when energy is low or communication isn’t needed.
If energy is below 0.25, there’s a 50% chance the message (or move) fails, replaced by “null” or no action.

Empowerment Reward:

For the actor: ( MI(a_actor', A | s) ) rewards choosing moves and messages (including “null”) that maximize control over ( S' ). Sending “null” could be optimal if it prompts useful observer responses or conserves energy for a critical move.
For observers: ( MI(s_i', M | s) ) rewards messages (or “null”) that influence ( S' ), e.g., by guiding the actor to the prey. “Null” might be chosen if the observer’s info isn’t relevant (e.g., prey’s x-coordinate hasn’t changed).


