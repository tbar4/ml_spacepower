# Curriculum Layout

This curriculum builds from orbital domain knowledge through mathematical foundations to a production-grade AI system for space domain awareness (SDA) wargaming. Each module introduces a new layer of the recommended architecture and connects it to SSA/SDA applications throughout.

## Module 0: Orbital Mechanics and the SDA Data Ecosystem

The domain foundation required before any SDA ML work. Covers the TLE format and Keplerian orbital elements, coordinate reference frames (ECI/ECEF/TEME/RTN), SGP4 propagation, conjunction analysis and the CCSDS CDM format, and the commercial SDA data ecosystem (Space-Track, CelesTrak, LeoLabs, and commercial providers). Every concept is tied to practical data engineering: parsing TLEs, batch propagation with python-sgp4, reading CDM covariance matrices, and building a conjunction screening pipeline.

**Key outcomes:** Parse and ingest TLE and OMM data from Space-Track and CelesTrak; propagate orbital states with python-sgp4; understand the TEME → ECI frame conversion; interpret CDM fields including covariance matrices in the RTN frame; build a 7-day conjunction screening pipeline; understand the SSA vs. SDA distinction and the commercial provider landscape.

## Module SP: Spacepower Theory and Strategic Context

The strategic theory foundation for wargame design. Covers foundational spacepower theory (Dolman, Lutes, the USSF Space Capstone Publication, Ziarnick, Chinese theory from Carlson), the counterspace operations taxonomy (kinetic/non-kinetic, reversible/irreversible, attributable/non-attributable), deterrence stability in space, Krepinevich's domain expansion theory and RMA/MTR distinction, and the mapping from strategic frameworks to game structures. Connects Dolman's sanctuary debate, PLA space doctrine, and gray zone operations to the specific game-theoretic tools (CFR, PSRO, IS-MCTS, particle filters) built in subsequent modules. No code.

**Key outcomes:** State the Lutes definition of spacepower; explain Dolman's Mackinder analogy and the strongest counterargument; name the seven USSF spacepower disciplines; classify counterspace capabilities by kinetic/non-kinetic and reversible/irreversible; explain the stability-instability paradox in space; describe the MTR vs. RMA distinction and apply it to U.S. SDA ML products; explain why CFR is the correct solver for the conjunction-masking game (private information about intent); describe when PSRO is required instead of two-player Nash equilibrium (multi-actor heterogeneous actors).

## Module 1: Foundations

Mathematical foundations required for all subsequent modules. Covers probability, Bayesian reasoning, linear algebra (including SVD and Cholesky decomposition), multivariate calculus, the multivariate Gaussian distribution, and constrained optimization. Every concept is introduced in the context of SSA problems: orbital state estimation, conjunction probability, radar measurement uncertainty, and sensor scheduling.

**Key outcomes:** Understand and implement Bayesian belief update, Monte Carlo estimation, SVD-based dimensionality reduction, covariance matrix manipulation, Mahalanobis distance, and Lagrange multiplier optimization.

## Module 2: Neural Networks as Function Approximators

Builds PyTorch neural networks from scratch: activation functions, forward passes, loss functions (with their MLE/MAP probabilistic interpretations), and the full training loop with gradient descent and backpropagation. The emphasis is on understanding what a neural network *is* mathematically — a parameterized function — before treating it as a black box.

**Key outcomes:** Implement a working MLP in PyTorch, choose the correct loss function for regression vs. classification, understand why MSE is MLE under Gaussian noise and cross-entropy is MLE for categorical distributions, and debug a training loop.

## Module 3: Reinforcement Learning Fundamentals

Sequential decision-making under uncertainty. Markov Decision Processes, value functions, Bellman equations, Q-learning, Deep Q-Networks, policy gradient methods, and actor-critic. Extends to hierarchical RL for decomposing complex multi-scale decisions (strategic/operational/tactical), and IMPALA for distributed large-scale training. Every algorithm is motivated by the sensor-tasking and orbital-maneuver-decision problems in SSA.

**Key outcomes:** Implement DQN and actor-critic in PyTorch, understand policy gradient variance reduction via baselines and GAE, design a hierarchical RL agent with sub-goal decomposition, and understand IMPALA's actor-learner decoupling for training thousands of parallel environments.

## Module 4: Search and Planning

Tree search as an alternative to pure function approximation. Minimax, alpha-beta pruning, MCTS, neural-guided MCTS (AlphaZero-style), and Information Set MCTS for fog-of-war games. IS-MCTS is the inference-time planner in the recommended production architecture: it uses the trained neural network as a prior and handles hidden state by sampling determinizations.

**Key outcomes:** Implement MCTS and understand UCB exploration, understand how AlphaZero combines MCTS with neural network training, implement IS-MCTS with determinization sampling for an imperfect-information SSA scenario.

## Module 5: Game Theory and Equilibrium Computation

Formal game theory: normal-form and extensive-form games, Nash equilibrium, information sets, and imperfect information. CFR (Counterfactual Regret Minimization) is the primary algorithm — the one that produced superhuman poker play. Monte Carlo CFR scales CFR to large games via sampling. Deep CFR replaces tabular regret storage with a neural network, making CFR applicable to games too large for exact tabular methods.

**Key outcomes:** Understand Nash equilibrium and why it is the right solution concept for adversarial multi-agent problems, implement vanilla CFR for a small extensive-form game, understand MCCFR's sampling strategies and their variance-speed tradeoff, understand Deep CFR's neural network approximation.

## Module 6: Multi-Agent Reinforcement Learning

Running RL in multi-agent environments: non-stationarity, joint policy search via PSRO, population evaluation via Alpha-rank, and the CTDE (Centralized Training, Decentralized Execution) paradigm. CTDE is the foundation of MAPPO and QMIX — the practical algorithms for training cooperative multi-agent systems. In the recommended SSA wargame architecture, CTDE trains the ally coalition while PSRO/self-play trains adversarial agents.

**Key outcomes:** Understand why multi-agent non-stationarity breaks single-agent RL convergence guarantees, implement the PSRO outer loop with meta-game Nash solving, implement MAPPO with a centralized critic and decentralized execution, understand QMIX's value decomposition for cooperative reward sharing.

## Module 7: Partial Observability

Hidden state inference combined with decision-making. POMDPs as the formal framework, belief states as sufficient statistics, particle filters for nonlinear/non-Gaussian state estimation, and imperfect-information games when multiple strategic agents each have private information. Opponent modeling: Bayesian type inference and KL divergence-based model drift detection.

**Key outcomes:** Implement a particle filter for orbital state estimation, understand why the belief state is a sufficient statistic (the Markov property applied to beliefs), connect POMDP belief updating to the Kalman filter, model an opponent's type from observed actions using Bayes' rule.

## Module 8: OpenSpiel and the Rust Capstone

Production engineering of the full stack. OpenSpiel's C++ game architecture and Python bindings, implementing custom games, the PettingZoo/shimmy/Ray RLlib integration pipeline (OpenSpiel → PettingZoo → MARLlib → distributed training), and the Rust/burn production gap. The module also covers the business on-ramp: SBIR/SpaceWERX contracting for uncleared solo founders, and LLM-in-the-loop wargame adjudication using locally-deployed models. The capstone implements a Rust CFR solver for an SSA conjunction-masking game.

**Key outcomes:** Implement a custom game in OpenSpiel's C++ API, wire it to PettingZoo via the shimmy compatibility wrapper, configure Ray RLlib for multi-agent training with thousands of parallel environments, understand SBIR eligibility requirements and the commercial-first vs. SBIR-first trade-off, build an LLM-adjudicated wargame with local models meeting FedRAMP constraints, and complete the capstone CFR solver.

## Module 9: Applied SDA ML

The curriculum's commercial product module. Takes the ML foundations from Modules 1–8 and applies them to the highest-value SDA AI product a solo uncleared founder can build from public data: maneuver detection from TLE history. Covers the label scarcity problem and synthetic data generation, feature engineering for orbital sequences (time-normalized delta features, J2 drift removal, BSTAR caveats), LSTM architecture and training, and operational evaluation metrics (detection latency, false alarm rate) that matter for deployment.

**Key outcomes:** Build a full maneuver detection pipeline on Space-Track TLE history; engineer physically meaningful orbital sequence features; train an LSTM classifier using synthetic label injection; evaluate against real ISS reboost test events with detection latency and false alarm rate metrics; understand the competitive landscape for TLE-based SDA AI products.
