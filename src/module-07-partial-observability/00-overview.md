# Module 7: Partial Observability

## Where this module fits

Every module so far has assumed the agent can see the full state of the environment. That assumption is false in almost every real SSA scenario. A ground telescope sees a two-dimensional angular position at a single moment in time, not a six-dimensional orbital state vector. An operator knows the locations and behaviors of their own satellites but not the adversary's. A conjunction risk assessment is based on an uncertain covariance estimate propagated forward from an old observation. The game is almost always played with incomplete information.

Partial observability introduces a qualitatively different challenge: the agent must simultaneously decide what to do and infer what it cannot see. The optimal action depends on the unknown state; the unknown state must be estimated from a history of noisy observations. This two-level inference-and-decision problem is what this module addresses.

The module covers the formal framework (POMDPs), the computational tools for maintaining uncertainty over the hidden state (belief states and particle filters), the game-theoretic extension (imperfect information games with multiple strategic agents), and the practical question of how to model and respond to an opponent whose type and strategy are unknown.

## What we cover

**POMDPs (lesson 1)**: the Partially Observable Markov Decision Process — the single-agent extension of MDPs to hidden states. Observation functions, belief states, the belief MDP, and why exact POMDP solution is intractable for large state spaces. The point-based approximation methods (PBVI, SARSOP) that make it tractable.

**Belief state representation (lesson 2)**: three concrete representations for the hidden state distribution. Kalman filters (for linear-Gaussian dynamics), particle filters (for nonlinear, non-Gaussian cases), and LSTM-based implicit belief (the deep RL approach). Particle deprivation, effective sample size, and how to detect when your filter is failing.

**Imperfect-information games (lesson 3)**: the multi-agent extension. What changes when multiple players each have private information and strategic incentives. Information sets, the distinction between POMDPs and imperfect-information games, and the value of information — how much you would pay to learn a hidden variable.

**Opponent modeling (lesson 4)**: how to build and use a probabilistic model of an opponent's type or strategy. Bayesian type inference from observed actions, exploiting a predictable opponent vs. playing Nash, and using KL divergence to detect when an opponent model has gone stale.

## Lessons

1. [POMDPs](01-pomdps.md)
2. [Belief state representation](02-belief-state-representation.md)
3. [Imperfect-information games](03-imperfect-information-games.md)
4. [Opponent modeling](04-opponent-modeling.md)

## Module project: particle-filter belief tracker

You will implement a particle filter that tracks the orbital state of an uncooperative RSO under intermittent, noisy ground-based observations. The scenario: a ground telescope sees the RSO once every few orbital periods, generating a noisy RA/Dec measurement each time. Between observations, the RSO propagates under a simplified two-body model plus a small unknown drag perturbation. Your particle filter maintains a distribution over the full orbital state and updates it each time an observation arrives.

You will instrument the filter to detect particle deprivation (via effective sample size), implement roughening to recover from it, and visualize how the uncertainty ellipsoid shrinks with each observation. The project connects the belief-state theory from lesson 2 to a concrete SSA tracking problem and provides the belief-propagation infrastructure you will need for the capstone game in Module 8.
