# Curriculum Layout

This document maps the full arc from foundations to a working SSA simulation. Each module builds on the previous ones. The final module project is the payoff: a Rust implementation of a small game-theoretic SSA scenario using everything learned.

## The arc in one sentence per module

1. **Foundations**: probability, vectors, and gradients, the three tools every later algorithm uses.
2. **Neural networks**: MLPs as function approximators, PyTorch mechanics, training loops.
3. **Reinforcement learning**: MDPs, value functions, Q-learning, and policy gradients.
4. **Search and planning**: MCTS rollouts, AlphaZero self-play, and neural-guided search.
5. **Game theory**: extensive-form games, Nash equilibria, CFR, and MCCFR.
6. **Multi-agent RL**: self-play, PSRO, fictitious play, and alpha-rank.
7. **Partial observability**: POMDPs, belief states, imperfect-information extensions.
8. **OpenSpiel and capstone**: a Rust SSA simulation tying everything together.

---

## Module 1: Foundations
**Builds toward**: a Monte Carlo conjunction probability estimator.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | Probability, distributions, and expectation | Random variables, categorical and Gaussian distributions, E[X] |
| 2 | Conditional probability and Bayes' rule | P(A\|B), prior/likelihood/posterior, sequential updates |
| 3 | Sampling and Monte Carlo estimation | The 1/√N convergence, unbiasedness, variance reduction preview |
| 4 | Entropy, cross-entropy, and KL divergence | Surprise, H(P), H(P,Q), KL(P‖Q), asymmetry |
| 5 | Vectors and dot products | State vectors, norms, alignment, cosine similarity |
| 6 | Matrices and matrix-vector multiplication | Row-as-dot-product, shapes, bias, why nonlinearity is needed |
| 7 | Derivatives, gradients, and the chain rule | Slope, partial derivatives, ∇f, chain rule, autograd |

**Project**: Monte Carlo conjunction probability estimator in Python.

---

## Module 2: Neural Networks as Function Approximators
**Builds toward**: a trained MLP that predicts conjunction risk from orbital features, the value function approximator pattern used in every later RL module.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | Activation functions | Why nonlinearity is needed, ReLU, tanh, softmax |
| 2 | Building an MLP | Stacking layers, `nn.Sequential`, forward pass by hand |
| 3 | Loss functions and what we are optimizing | MSE, cross-entropy loss, what minimizing loss means geometrically |
| 4 | The training loop | Datasets and batches, forward/backward/step, overfitting and validation |

**Project**: train a small MLP to approximate a conjunction-risk scoring function from simulated orbital feature data. Lays the groundwork for the value network in Module 4.

---

## Module 3: Reinforcement Learning Fundamentals
**Builds toward**: a tabular Q-learning agent playing a simple satellite resource allocation game.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | Markov Decision Processes | States, actions, transitions, rewards, discount factor γ |
| 2 | Value functions | V(s), Q(s,a), Bellman equations, bootstrapping |
| 3 | Tabular Q-learning | TD error, ε-greedy exploration, convergence |
| 4 | Deep Q-Networks (DQN) | Function approximation for Q, experience replay, target networks |
| 5 | Policy gradient methods | REINFORCE, the score function estimator, entropy regularization |
| 6 | Actor-critic | Advantage functions, baseline subtraction, the A2C structure |

**Project**: a DQN agent that learns to allocate sensor dwell time across a set of tracked objects to maximize conjunction-detection reward. First OpenSpiel touchpoint: the game is defined as an OpenSpiel environment.

---

## Module 4: Search and Planning
**Builds toward**: an MCTS agent that plans in a two-satellite maneuver decision game.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | Tree search fundamentals | Game trees, minimax, alpha-beta pruning |
| 2 | Monte Carlo Tree Search | UCB1, selection/expansion/simulation/backpropagation, PUCT |
| 3 | Neural-guided MCTS | Policy network for priors, value network replacing rollouts |
| 4 | AlphaZero self-play | Self-play data generation, MCTS as policy improvement operator |

**Project**: an AlphaZero-lite agent trained by self-play on a small pursuit-evasion game between two spacecraft. Uses an OpenSpiel game definition and PyTorch policy/value networks. Rust translation: the MCTS tree structure.

---

## Module 5: Game Theory and Equilibrium Computation
**Builds toward**: a CFR solver for a small orbital negotiation game (who maneuvers to avoid conjunction?).

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | Normal-form and extensive-form games | Strategy profiles, Nash equilibrium, information sets |
| 2 | Extensive-form games in detail | Game trees, information sets, strategies vs. policies, reach probabilities |
| 3 | Counterfactual Regret Minimization (CFR) | Counterfactual values, regret matching, convergence to Nash |
| 4 | Monte Carlo CFR (MCCFR) | Outcome sampling, external sampling, variance vs. speed tradeoff |
| 5 | Deep CFR | Neural network as regret buffer, traversal sampling |

**Project**: a vanilla CFR and MCCFR solver for the "who maneuvers?" conjunction game defined in OpenSpiel. Rust translation: the CFR data structures (information set table, regret vector). This is the most Rust-relevant lesson in the curriculum.

---

## Module 6: Multi-Agent RL and Self-Play
**Builds toward**: a PSRO solver for a multi-operator satellite-constellation game.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | The multi-agent problem | Non-stationarity, simultaneous vs. sequential, cooperative vs. competitive |
| 2 | Fictitious play | Best response to empirical distribution, convergence in zero-sum games |
| 3 | Policy Space Response Oracles (PSRO) | Meta-game, restricted Nash, oracle computation |
| 4 | Alpha-rank | Markov chain over strategy profiles, stationary distribution, eigenvectors |

**Project**: a PSRO loop for a 2-player satellite constellation coverage game using OpenSpiel. Alpha-rank used to analyze which strategies dominate. First lesson that needs eigenvalues (handled inline, in context).

---

## Module 7: Partial Observability and Imperfect Information
**Builds toward**: a belief-state POMDP agent for an SSA sensor-tasking problem under adversarial conditions.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | POMDPs | Observation functions, belief states, PBVI, point-based methods |
| 2 | Belief state representation | Particle filters, Gaussian belief, exact Bayes updating |
| 3 | Imperfect-information game solving | Information sets revisited, blueprint strategies |
| 4 | Opponent modeling | Recursive reasoning, level-k, Bayesian opponent models |

**Project**: a particle-filter belief state tracker for a pursuit-evasion POMDP. The evader's state is hidden; the tracker maintains a particle approximation of the posterior and uses it to guide sensor tasking. Python, using OpenSpiel's POMDP support.

---

## Module 8: OpenSpiel Deep Dive and Rust Capstone
**Builds toward**: a self-contained Rust crate implementing a small SSA game-theoretic simulation end to end.

| # | Lesson | Key concepts |
|---|--------|--------------|
| 1 | OpenSpiel architecture | Game API, algorithm API, bots, observers |
| 2 | Implementing a custom game | Extending `pyspiel.Game`, state transitions, information states |
| 3 | Rust and burn: the production gap | What exists, what does not, how to bridge |
| 4 | Designing the SSA game | State representation, action space, reward structure for the capstone |

**Project (capstone)**: a Rust crate implementing:
- A two-player extensive-form SSA game (attacker tries to mask a maneuver; defender allocates sensors to detect it)
- A vanilla CFR solver over the game tree using native Rust data structures
- A `burn` neural network trained to approximate regret values (replacing tabular CFR for larger state spaces)
- A simple CLI that runs self-play and prints the Nash equilibrium strategy profile

This is the artifact you could drop into a thesis simulation. It references every concept built in modules 1 through 7 and fills the gap left by the absence of a Rust-native OpenSpiel.