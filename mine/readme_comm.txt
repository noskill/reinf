===========  
Two-Agent Empowerment Toy Environment  
------------------------------------

Purpose  
-------  
Provide a minimal reference implementation in which **two agents discover an
informative communication protocol by maximising empowerment** (intrinsic
reward only).  Each agent may send one symbol per turn from a small,
fixed alphabet **Α**.

Files  
-----  
env.py      ‑ alternating–turn environment (see timing below)  
model.py    ‑ policy net  πθ(a | s)  and inverse model  qψ(a | s,s′)  
README.txt  ‑ this file  

Alphabet  
--------  
Α = { 0 … K-1 } (K set in `config.yaml`, default 4)

Timing diagram (alternating-turn mode)  
--------------------------------------

t            t+1            t+2           …  
┌──────────┬──────────┬──────────┬──────────┐  
│Agent-1   │Agent-2   │Agent-1   │Agent-2   │  
│turn      │turn      │turn      │turn      │  
│          │          │          │          │  
│sees mₜ₋₁ │sees mₜ   │sees mₜ₊₁│sees mₜ₊₂│  
│sends aₜ  │sends aₜ₊₁│sends aₜ₊₂│sends … │  
└──────────┴──────────┴──────────┴──────────┘  

 Environment writes `m_t = a_t`, so the partner receives it next step.

Empowerment definition  
----------------------  
For each agent and current **state s** (“it is my turn and I have just seen
m”), horizon **H = 2 physical steps**:

    A  = symbol I send now  (a_t)
    S′ = symbol I will hear after partner acts (m_{t+1})

Empowerment(s) = maxₚ(A|s) I(A ; S′ | s)  
We *approximate* I by the inverse-model trick **and include an entropy term**
that encourages a high-capacity (near-uniform) communication channel.  This
results in the following *intrinsic* reward used during RL optimisation:

    r_int(s, a, s′) = log qψ(A=a | s , S′=s′) + H[πθ(·|s)]

The first term (inverse model) rewards actions that are easily predictable from
the successor state, whereas the second term adds the *entropy* of the policy
at that state – encouraging a high-capacity (near-uniform) channel without
needing to recompute individual action log-probabilities.
Intuitively, the agent is encouraged to act so that its messages remain
recoverable **and** its action distribution stays as diverse as possible.

where qψ tries to reconstruct the agent’s own action from the successor
state.

Implementation recipe  
---------------------

1. Observation vector  
   [ one-hot(m),  agent_id ]                        (length K+1)

2. Action space  
   Discrete(K)  → symbol a ∈ Α.

3. Networks  
   • πθ : obs ↦ softmax over K symbols  
   • qψ : concat(obsₜ , obsₜ₊₁) ↦ softmax over K symbols

4. Interaction loop  
   for each episode  
       reset env  
       while not done  
           a  = πθ(o)          # agent whose turn it is  
           o′, _ = env.step(a)  
           store (o, a, o′, agent_id)  
           o = o′  
   end

5. Optimisation (e.g. PPO)  
   • Intrinsic reward  r_int = log qψ(a | o , o′) + H[πθ(·|o)]  
   • Advantage        = r_int + γ V(o′) – V(o)  
   • Policy loss      = –Advantage * log πθ(a|o)  
   • Value loss       = ‖r_int + γ V(o′) – V(o)‖²  
   • Inverse-model loss  = –log qψ(a | o , o′)  

   Update θ, φ, ψ jointly every N steps.

Default hyper-parameters (config.yaml)  
--------------------------------------  
K:            4                 # alphabet size  
γ:            0.99              # discount (irrelevant for pure intrinsic)  
lr_policy:    3e-4  
lr_inverse:   3e-4  
# Note: the entropy bonus is now part of *r_int* above, so a separate
# ``entropy_coef`` in the PPO loss is no longer required (set to 0).  
batch_size:   2048  
entropy_coef: 0.0               # already handled in reward  
clip_r_e:     [-20, 20]         # numerical stability  

Quick test  
----------  
$ python train.py --episodes 20000  

Success criterion  
-----------------  
After training, sample 1000 turns and measure empirical MI:

    Î ≈ Ĥ(A) – Ĥ(A | S′)  → should converge to log K (≈1.39 for K=4).

Both agents should settle on an (almost) bijective deterministic reply rule
(echo, +1 mod K, etc.) and choose their own symbols uniformly.  

Variants  
--------  
•  Fused-step mode: call partner’s policy **inside** env.step(); then horizon
   H = 1 and code simplifies (set `FUSED=True` in env.py).  
•  Add Gaussian noise to messages to study error-correcting protocols.  
•  Introduce energy cost per symbol to create potential competition.  

Reference papers  
----------------  
Klyubin et al., “Empowerment: An Introduction” (2008)  
Gregor et al., “Variational Intrinsic Control” (2016)  
