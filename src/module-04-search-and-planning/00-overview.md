# Module 4: Search and Planning


<!-- toc -->

## Where this module fits

Module 3 gave you model-free RL: the agent learns from experience without explicitly reasoning about the future. That is powerful but limited. When you can simulate possible futures (in a game, you can imagine the consequences of moves), explicit search through those futures is often dramatically better than learned policies alone. This module covers Monte Carlo Tree Search (MCTS), the most important search algorithm in modern game AI, and AlphaZero, which combines MCTS with neural networks trained by self-play.

The skills here directly enable the rest of the curriculum. MCTS as a planning method shows up in CFR variants (Module 5). The AlphaZero pattern (network-guided search, network trained on search results) is one template for solving multi-agent problems (Module 6). The capstone (Module 8) builds a custom MCTS-flavored solver in Rust.

## What we cover

**Tree search fundamentals (lesson 1)**: minimax for two-player perfect-information games, alpha-beta pruning. The classical foundation. We do not dwell here because alpha-beta does not scale to large branching factors, but the vocabulary and structure carry over to MCTS.

**Monte Carlo Tree Search (lesson 2)**: the four-phase MCTS loop (selection, expansion, simulation, backpropagation). UCB1 for the exploration-exploitation tradeoff during selection. Pure MCTS (no neural network) on a small game.

**Neural-guided MCTS (lesson 3)**: replace random rollouts with a value network's prediction; use a policy network to bias the selection phase. PUCT (Predictor + UCT). This is the architecture inside AlphaGo Zero and AlphaZero.

**AlphaZero self-play (lesson 4)**: how to train the value and policy networks from games the agent plays against itself. The training loop is the actor-critic structure from Module 3 with MCTS as the policy improvement operator.

## Lessons

1. [Tree search fundamentals](01-tree-search-fundamentals.md)
2. [Monte Carlo Tree Search](02-monte-carlo-tree-search.md)
3. [Neural-guided MCTS](03-neural-guided-mcts.md)
4. [AlphaZero self-play](04-alphazero-self-play.md)

## Module project: an AlphaZero-lite agent for a pursuit-evasion game

You will train an AlphaZero-style agent on a simple 2-spacecraft pursuit-evasion game defined in OpenSpiel. The defender (you) tries to maintain coverage of an orbital region; the evader (also you, in self-play) tries to avoid detection. The agent learns by playing against itself: MCTS guided by a small policy/value network, network trained on the resulting games, MCTS using the improved network, and so on.

This is the canonical pattern that produced superhuman Go, Chess, and Shogi players. We are doing it on a much smaller game so it actually trains in reasonable time on a laptop.
