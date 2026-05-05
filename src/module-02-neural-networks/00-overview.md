# Module 2: Neural Networks as Function Approximators


<!-- toc -->

## Where this module fits

Module 1 gave you three tools: probability (to reason about uncertainty), linear algebra (to represent states and compute scores), and calculus (to find the direction that improves a score). This module assembles those tools into a working machine learning system.

The goal is not to deeply understand every corner of deep learning. The goal is to understand neural networks well enough to use them as function approximators in reinforcement learning and game theory. In Module 3, a neural network will approximate a value function (what is this game state worth?). In Module 4, one network will approximate a policy (what action should I take here?) and another will approximate the outcome of playing from a position. In Module 5, a network will approximate regret values. The neural network is infrastructure; the algorithms that use it are the point.

So this module is deliberately compressed. Four lessons, then a project. Classification gets one lesson's worth of attention, not because it is unimportant, but because its only job here is to motivate cross-entropy loss and softmax, which we will need later. We do not build image classifiers. We build function approximators.

## What we cover

**Activation functions (lesson 1)**: The final missing piece from Module 1's linear algebra discussion. Without nonlinear activation functions, stacking layers does nothing. With them, networks can approximate arbitrarily complex functions. We cover ReLU (the workhorse), tanh (the older workhorse), and softmax (how you turn raw scores into a probability distribution over actions).

**Building an MLP (lesson 2)**: How to assemble layers into a multi-layer perceptron in PyTorch. Forward pass from scratch, then using `nn.Sequential`. We trace exactly what happens to a state vector as it flows through the network.

**Loss functions (lesson 3)**: What the network is actually trying to minimize. Mean squared error for regression (approximating a continuous value function). Cross-entropy loss for classification (approximating a policy). The connection between these losses and the concepts from Module 1 (expectation, cross-entropy between distributions).

**The training loop (lesson 4)**: The complete gradient descent cycle in PyTorch: load a batch, forward pass, compute loss, backward pass, optimizer step. Overfitting, validation, and why you need both. This lesson is largely mechanical; the concepts from Module 1 lessons 3 and 7 do the heavy lifting.

**Recurrent networks: LSTM and GRU (lesson 5)**: The extension from fixed-input MLPs to sequential data. Vanilla RNNs fail on long sequences because of vanishing gradients; LSTMs solve this by separating long-term memory (cell state) from short-term memory (hidden state) via learned forget, input, and output gates. GRU is a simpler alternative. This lesson is the direct prerequisite for Module 9's maneuver detection pipeline, which processes 30-day TLE histories as LSTM inputs.

**Regularization and model evaluation (lesson 6)**: The tools that prevent overfitting and the evaluation practices that detect it. Covers train/val/test splits, dropout, L2 weight decay, batch normalization, early stopping with checkpoint restoration, and the evaluation metrics appropriate for imbalanced classification. In Module 9's label-scarce setting — a few hundred real maneuver labels supplemented by synthetic injection — these practices are not optional extras; they are what separates a deployable model from one that memorized its training set.

## Lessons

1. [Activation functions: giving networks their power](01-activation-functions.md)
2. [Building an MLP in PyTorch](02-building-an-mlp.md)
3. [Loss functions and what we are optimizing](03-loss-functions.md)
4. [The training loop](04-the-training-loop.md)
5. [Recurrent networks: LSTM and GRU](05-recurrent-networks.md)
6. [Regularization and model evaluation](06-regularization-and-evaluation.md)

## Module project: approximating a conjunction-risk value function

You will train a small MLP to predict a conjunction risk score from orbital feature inputs. The training data is synthetically generated from the Monte Carlo estimator you built in Module 1. This connects the two modules directly: the Monte Carlo estimator provides training labels, and the neural network learns to predict those labels quickly without running the Monte Carlo simulation each time.

This is exactly the pattern used in deep RL and deep CFR: generate data by simulation, train a neural net to approximate the result, use the neural net to make fast predictions during the actual algorithm. Module 3 will build on this.

## What this module is not

We are not building an image classifier or training GPT. We are not covering convolutional layers, attention mechanisms (attention is introduced in Module 9 as a contrast to LSTMs), or advanced training techniques like gradient checkpointing and mixed-precision training. These are important topics for a broader ML education; they are not on the path to OpenSpiel and SSA simulations. If you want to go deeper into deep learning fundamentals after finishing this curriculum, the fast.ai course and Andrej Karpathy's "Neural Networks: Zero to Hero" series are excellent.
