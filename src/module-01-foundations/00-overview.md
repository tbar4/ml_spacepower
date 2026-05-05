# Module 1: Foundations


<!-- toc -->

## Where this module fits

Everything you'll build later in this curriculum (value functions, policy gradients, MCTS rollouts, CFR regret updates, neural network forward passes) reduces to three operations: pushing numbers through matrices, taking derivatives of those numbers with respect to other numbers, and reasoning about what those numbers mean when they're random. This module gives you working intuition for those three things and nothing else.

This is not a math course. We are picking exactly the pieces of probability, linear algebra, and calculus that show up when you read OpenSpiel source code, and skipping everything else. If a topic feels truncated, that's because the rest doesn't matter for our goal. When we hit a later algorithm that genuinely needs more (eigenvectors for alpha-rank, for instance), we'll handle it then, in context.

## What we cover

**Probability (lessons 1-4).** Distributions, expectation, conditional probability, Monte Carlo sampling, and the information-theoretic quantities (entropy, cross-entropy, KL divergence) that show up everywhere from policy gradients to regret matching. The MCTS, MCCFR, and policy gradient methods later in the curriculum are, deep down, ways of making smart estimates from random samples. If you internalize "expectation under a distribution, estimated by sampling," you've already got the shape of half the algorithms in OpenSpiel.

**Linear algebra (lessons 5-6).** Vectors as state representations (an orbital state vector is, mechanically, just a vector). Dot products as similarity and projection. Matrix-vector multiplication as the operation that defines a single neural network layer. That's it for now. We're skipping eigendecomposition, determinants, matrix inverses, and rank, because they don't show up until later (and some never do).

**Calculus (lesson 7).** Derivatives as slopes, partial derivatives as slopes-along-one-axis, gradients as the vector pointing uphill, and the chain rule. The chain rule is the entire mathematical content of backpropagation; if you can see it visually, the rest of "how neural nets learn" is bookkeeping.

## Lessons

1. [Probability, distributions, and expectation](01-probability-distributions-expectation.md)
2. [Conditional probability and Bayes' rule](02-conditional-probability-bayes.md)
3. [Sampling and Monte Carlo estimation](03-sampling-and-monte-carlo.md)
4. [Entropy, cross-entropy, and KL divergence](04-entropy-and-kl-divergence.md)
5. [Vectors and dot products](05-vectors-and-dot-products.md)
6. [Matrices and matrix-vector multiplication](06-matrices-and-matrix-vector-multiplication.md)
7. [Derivatives, gradients, and the chain rule](07-derivatives-gradients-chain-rule.md)

## Module project: Monte Carlo conjunction probability

You'll write a small Python program that estimates the probability of a collision between two satellites whose positions and velocities are known only to within some uncertainty. It uses every concept in this module: state vectors (linear algebra), uncertainty distributions (probability), Monte Carlo sampling (expectation under randomness), and a small sensitivity analysis that previews what gradients are good for.

This is a real problem in your field. JSpOC and the commercial conjunction services do something more sophisticated, but the bones are the same: simulate possible futures, average over them, use the average to make a decision. It is also a microcosm of the rest of the curriculum: every algorithm we build later is, in some way, doing exactly this with more structure on top.

## What this module is not

We are not doing epsilon-delta proofs. We are not deriving Cauchy-Schwarz. We are not classifying matrices into normal forms. We are not covering measure-theoretic probability. If you came in hoping this would be the time you finally Get linear algebra, this module will frustrate you. It exists to make you fluent enough to read RL and game-theory code without bouncing off the notation, and that is the entire bar.

## How to read the lessons

Every lesson follows the same shape: where it fits, the concept (intuition first), the math (only when load-bearing), code, a worked example small enough to verify by hand, and a quiz. If you find yourself stuck on math notation, that's a signal to reread the symbol-decoding paragraph rather than to push through. The notation is a compression of the intuition; if the intuition isn't there, the notation will not magically install it.
