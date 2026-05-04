# Module 6: Multi-Agent Reinforcement Learning

## Where this module fits

Modules 3–5 built a progression: single-agent RL, then planning with search, then game-theoretic equilibrium computation. All of that assumed we could either solve the game directly (CFR) or train a single neural network against fixed opponents. Real SSA scenarios break both assumptions: there are multiple agents learning simultaneously, the strategy space is too large for CFR, and the notion of a "fixed opponent" is exactly what we are trying to move beyond.

Multi-agent RL (MARL) is what happens when you run RL in a multi-agent environment. This sounds simple, but it introduces a non-trivial problem: each agent's environment is non-stationary because the other agents are also learning. A strategy that works against today's opponent may fail against tomorrow's. Convergence guarantees that hold in single-agent RL break down. MARL requires new concepts and new algorithms.

This module covers the most important tools for practical MARL: how to reason about the non-stationarity problem, how to search the joint policy space systematically (PSRO), how to evaluate entire populations of policies (Alpha-rank), and how to train cooperative agents efficiently via centralized training with decentralized execution (CTDE).

## What we cover

**The multi-agent problem (lesson 1)**: what changes when you add a second learning agent. Non-stationarity of the environment, joint action spaces, the difference between cooperative, competitive, and mixed settings. Why single-agent RL fails and what replaces it.

**Fictitious play (lesson 2)**: the oldest multi-agent learning algorithm. Each agent best-responds to the historical average policy of the opponents. Simple, convergent in two-player zero-sum games, and a direct conceptual precursor to PSRO.

**Policy-Space Response Oracles (PSRO) (lesson 3)**: generalizes fictitious play to neural-network policies. Maintain a growing population of policies. Use RL to compute best responses to the current Nash mixture over the population. Solve the resulting meta-game for a new Nash. Repeat. This is the algorithm that produced AlphaStar and the most robust multi-agent strategies for complex games.

**Alpha-rank (lesson 4)**: an alternative to Nash for evaluating policy populations. Instead of asking "what mixture is unexploitable," it asks "which policies dominate in an evolutionary sense?" Alpha-rank is more tractable for large populations and produces a ranking rather than a mixture, which is often more useful in practice.

**Centralized training, decentralized execution (lesson 5)**: the CTDE paradigm. During training, give each agent access to the full joint state and other agents' actions. At execution time, each agent acts only on its own observations. This includes MAPPO (the practical cooperative MARL algorithm) and QMIX (value decomposition for cooperative agents). CTDE is how you train the ally coalition in the recommended SSA wargame architecture.

## Lessons

1. [The multi-agent problem](01-multi-agent-problem.md)
2. [Fictitious play](02-fictitious-play.md)
3. [Policy-Space Response Oracles (PSRO)](03-psro.md)
4. [Alpha-rank](04-alpha-rank.md)
5. [Centralized training, decentralized execution](05-ctde-mappo.md)

## Module project: PSRO for satellite constellation coverage

You will implement a two-player PSRO loop for a satellite constellation coverage game. The scenario: two operators compete over sensor coverage of a shared orbital region. Each controls a subset of satellites and can task them to observe different orbital slots. Payoff is coverage area minus overlap penalty. You will build the meta-game payoff matrix from simulated policy rollouts, solve the 2-player Nash at each PSRO iteration, and watch the policy population evolve from random tasking toward coordinated coverage strategies. The project demonstrates PSRO's core loop — oracle training, meta-game construction, Nash solve, repeat — at a scale that runs on a laptop in minutes.
