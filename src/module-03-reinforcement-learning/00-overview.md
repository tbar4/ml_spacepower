# Module 3: Reinforcement Learning Fundamentals

## Where this module fits

Modules 1 and 2 gave you the mathematical and computational tools. This module is where we start using those tools to make decisions over time. Reinforcement learning is the framework for "an agent acts in an environment, receives feedback, and learns to act better." That is the structural skeleton of every algorithm in the rest of this curriculum: MCTS planning (Module 4), CFR equilibrium computation (Module 5), multi-agent self-play (Module 6), and POMDP planning (Module 7) are all variations on this theme.

The single most important conceptual jump in this module is from **prediction** (Module 2: given features, predict a number) to **decision-making** (Module 3: given a state, choose an action that affects the future). Decision-making over time introduces problems that prediction does not have: temporal credit assignment (which earlier action caused this later reward?), the exploration-exploitation tradeoff (do I take the best-known action or try something new?), and bootstrapping (using my current value estimates to improve themselves).

This module covers the foundational algorithms in roughly the order they were historically developed. We start tabular (no neural networks) so the algorithms are clearly visible. Then we add neural network function approximation. By the end you will have implemented Q-learning, DQN, REINFORCE, and an actor-critic algorithm, all on small SSA-flavored problems.

## What we cover

**MDPs (lesson 1)**: the formal language for sequential decision problems. States, actions, transitions, rewards, discount factors. Every later algorithm assumes this structure. We frame an SSA sensor allocation problem as an MDP and use it as the running example throughout the module.

**Value functions (lesson 2)**: the mathematical object every value-based algorithm computes. V(s) is "how good is this state?", Q(s,a) is "how good is taking this action in this state?". The Bellman equations relate them recursively, which is the foundation for everything that follows.

**Tabular Q-learning (lesson 3)**: the simplest possible value-based RL algorithm. Each state-action pair gets its own table entry; the algorithm updates entries as it gains experience. Convergence is guaranteed in tabular settings under mild conditions. This lesson is where the temporal difference (TD) learning idea becomes concrete.

**Deep Q-Networks (lesson 4)**: replacing the table with a neural network. This is the algorithm that achieved superhuman performance on Atari games in 2013-2015. We cover experience replay and target networks, the two engineering tricks that make it work.

**Policy gradient methods (lesson 5)**: a fundamentally different approach. Instead of learning a value function and deriving a policy from it, learn the policy directly. REINFORCE is the simplest form. The score function estimator (which we build from scratch) is the mathematical engine.

**Actor-critic (lesson 6)**: combine value and policy methods. The "critic" learns a value function; the "actor" learns a policy and is trained using the critic to reduce variance. This is the architecture used by AlphaZero (Module 4) and most modern deep RL.

## Lessons

1. [Markov Decision Processes](01-markov-decision-processes.md)
2. [Value functions and Bellman equations](02-value-functions-and-bellman.md)
3. [Tabular Q-learning](03-tabular-q-learning.md)
4. [Deep Q-Networks](04-deep-q-networks.md)
5. [Policy gradient methods](05-policy-gradient-methods.md)
6. [Actor-critic](06-actor-critic.md)

## Module project: a DQN sensor allocation agent

You will build a DQN agent that learns to allocate sensor dwell time across a set of tracked space objects, with the goal of maximizing the expected detection of high-priority conjunctions. The environment is defined as an OpenSpiel game (your first OpenSpiel touchpoint), the value network is the conjunction-risk approximator from Module 2 (refactored as a Q-network), and the training loop ties together everything from Modules 1 through 3.

By the end of this project, you will have an agent that learns from scratch (no domain knowledge programmed in) to make sensible sensor scheduling decisions in a simplified SSA scenario.

## What we are deliberately skipping

We are not covering: trust region methods (TRPO, PPO) in detail, off-policy actor-critic methods (DDPG, SAC), distributional RL, hierarchical RL, model-based RL. PPO is mentioned briefly in the actor-critic lesson because it is what most modern policy gradient implementations use, but we do not implement it ourselves. These topics are important for a broad RL education; they are not load-bearing for the OpenSpiel multi-agent algorithms we are heading toward.
