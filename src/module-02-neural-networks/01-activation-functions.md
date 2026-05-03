# Lesson 1: Activation Functions

## Where this fits

At the end of Module 1, lesson 6, we noted a problem: stacking linear layers produces another linear layer. No matter how deep you make the network, if every layer is just \\(W\mathbf{x} + \mathbf{b}\\), the whole thing is equivalent to a single linear transformation. It can only learn straight-line relationships between inputs and outputs.

That is a fatal limitation. Real value functions in RL are not linear. Real policy distributions are not linear functions of the game state. The conjunction risk for a satellite does not increase linearly with approach velocity, because the risk profile is nonlinear: there are safe regimes, transition zones, and high-risk regimes that a line cannot capture.

**Activation functions** are the fix. After each linear layer, you apply a simple nonlinear function to every output. This breaks the "composition of linears is linear" problem and gives networks the ability to approximate any continuous function, given enough capacity.

## The problem, made concrete

Suppose you want a network to learn: "if conjunction risk is above 0.7, return HIGH priority; otherwise return LOW priority." This is a threshold decision, a step function. No single linear function can make a hard decision like this. A line either keeps going up, keeps going down, or stays flat. It cannot bend.

Here is what a linear function can and cannot do:

```python
import torch

def linear_decision(conjunction_risk):
    """A linear function trying to distinguish high vs. low risk."""
    # This can approximate the threshold for ONE input value,
    # but will be wrong on both sides of the threshold.
    return 2.0 * conjunction_risk - 1.0

# The best a linear function can do is a ramp
risks = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
outputs = linear_decision(risks)
print(outputs.tolist())
# [-0.8, -0.4, 0.0, 0.4, 0.8]
# This goes from negative to positive... but gradually. No sharp decision.
```

What we actually want is something more like: below 0.7, output is 0. Above 0.7, output is 1. That requires a bend in the function, which requires nonlinearity.

## ReLU: the workhorse activation

**ReLU** stands for Rectified Linear Unit. The function is:

\\[ \text{ReLU}(x) = \max(0, x) \\]

In plain English: if the input is negative, output 0. If the input is positive, output it unchanged.

That is the whole function. Graphically, it is a flat line at zero for negative inputs, then a straight ramp upward for positive inputs. There is exactly one bend, at x = 0.

| Input x | ReLU(x) |
|---------|---------|
| -5.0    | 0.0     |
| -1.0    | 0.0     |
| -0.001  | 0.0     |
| 0.0     | 0.0     |
| 0.001   | 0.001   |
| 1.0     | 1.0     |
| 5.0     | 5.0     |

Why does this simple function solve the problem? When you apply ReLU after each layer, each neuron acts as a gating mechanism: it either passes its input through (if the weighted sum was positive) or blocks it (if the weighted sum was negative). Different neurons gate on different conditions. The combination of many such gates, applied after each layer, can carve up the input space into arbitrarily complex regions.

The deep theorem here (the Universal Approximation Theorem) says: a neural network with at least one hidden layer and a nonlinear activation function can approximate any continuous function to arbitrary precision, given enough neurons. ReLU is one of the activation functions that makes this true.

**Why ReLU specifically?** It is fast to compute (just a max operation), its gradient is simple (0 for negative inputs, 1 for positive), and it avoids the "vanishing gradient" problem that plagued earlier activations. When neural networks get very deep, gradients can shrink to nearly zero as they backpropagate through many layers. ReLU's gradient is either 0 or 1, not a small fraction, so deep networks train faster.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
output = F.relu(x)
print(output.tolist())  # [0.0, 0.0, 0.0, 1.0, 3.0]
```

## What ReLU does to a layer output

Recall from lesson 6 of Module 1 that a layer computes \\(\mathbf{y} = W\mathbf{x} + \mathbf{b}\\). The output \\(\mathbf{y}\\) is a vector, one value per neuron. Applying ReLU to it means applying max(0, ·) to each element independently:

```python
import torch
import torch.nn.functional as F

# Simulate a layer output (the result of W @ x + b)
layer_output = torch.tensor([-1.2, 0.5, -0.3, 2.1, -0.8, 1.4])

# Apply ReLU: negative values get zeroed, positive values pass through
after_relu = F.relu(layer_output)
print(f"Before ReLU: {layer_output.tolist()}")
print(f"After ReLU:  {after_relu.tolist()}")
# Before: [-1.2, 0.5, -0.3, 2.1, -0.8, 1.4]
# After:  [ 0.0, 0.5,  0.0, 2.1,  0.0, 1.4]
```

Three neurons got zeroed out. They were "inactive" for this input. The other three pass their values through. Different inputs will activate different subsets of neurons. This selectivity is what lets the network learn different behaviors for different parts of the input space.

## Tanh: an older alternative

**tanh** (hyperbolic tangent) is an S-shaped (sigmoid) curve that squashes any input into the range (−1, +1):

\\[ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \\]

You do not need to memorize this formula. What matters:

| Input x | tanh(x) |
|---------|---------|
| -∞      | -1.0    |
| -2.0    | -0.964  |
| -1.0    | -0.762  |
| 0.0     | 0.0     |
| 1.0     | 0.762   |
| 2.0     | 0.964   |
| +∞      | 1.0     |

Tanh is smooth (no kink at zero), bounded (always between -1 and +1), and centered at zero. For problems where you want outputs in a bounded range, it can work well.

The downside: for large positive or negative inputs, tanh gets very close to +1 or -1 and its gradient becomes nearly zero (the curve flattens out). This "saturation" causes the vanishing gradient problem in deep networks. ReLU avoids saturation on the positive side. For most modern architectures, ReLU or its variants are preferred over tanh, but you will see tanh in some game-playing contexts and in recurrent networks.

```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.tanh(x).tolist())
# [-0.9640, -0.7616, 0.0, 0.7616, 0.9640]
```

## Softmax: turning scores into a probability distribution

ReLU and tanh are activation functions for hidden layers (the intermediate layers inside the network). For the final output layer, when you want a probability distribution over discrete choices (like action probabilities in a policy network), you use **softmax**.

Softmax takes a vector of raw scores (called **logits**) and converts them into a valid probability distribution: all values positive, all values summing to 1.

For an input vector \\(\mathbf{z} = (z_1, z_2, \ldots, z_n)\\), the softmax output for component \\(i\\) is:

\\[ \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} \\]

**Decoding each piece:**

**\\(e^{z_i}\\)**: The number \\(e\\) (approximately 2.718, Euler's number) raised to the power \\(z_i\\). This is the exponential function. Its important property: exponentials are always positive, so softmax outputs are always positive. Also, larger inputs give exponentially larger outputs, so softmax amplifies differences between logits.

**\\(\sum_{j=1}^{n} e^{z_j}\\)**: Sum of all the exponentials, over all n outputs. This is the normalizing constant that makes everything add up to 1.

**Reading in English**: "Exponentiate each score, then divide each by the sum of all the exponentiated scores."

### Walking through a softmax calculation by hand

Suppose your policy network outputs raw scores (logits) for 4 possible actions:

```
Logits: z = [1.0, 2.0, 0.5, -1.0]
```

**Step 1**: Compute the exponential of each logit.

| Action | Logit \\(z_i\\) | \\(e^{z_i}\\) |
|--------|----------------|----------------|
| 0      | 1.0            | e¹ ≈ 2.718     |
| 1      | 2.0            | e² ≈ 7.389     |
| 2      | 0.5            | e⁰·⁵ ≈ 1.649   |
| 3      | -1.0           | e⁻¹ ≈ 0.368    |

**Step 2**: Sum all the exponentials.

2.718 + 7.389 + 1.649 + 0.368 = **12.124**

**Step 3**: Divide each exponential by the sum.

| Action | \\(e^{z_i}\\) | Probability |
|--------|----------------|-------------|
| 0      | 2.718          | 2.718 / 12.124 ≈ **0.224** |
| 1      | 7.389          | 7.389 / 12.124 ≈ **0.609** |
| 2      | 1.649          | 1.649 / 12.124 ≈ **0.136** |
| 3      | 0.368          | 0.368 / 12.124 ≈ **0.030** |
| **Sum**|               | **1.000** ✓ |

Action 1 had the highest logit (2.0) and gets the highest probability (61%). Action 3 had the lowest logit (-1.0) and gets the lowest probability (3%). All probabilities are positive and sum to 1.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, 2.0, 0.5, -1.0])
probs = F.softmax(logits, dim=0)
print(probs.tolist())
# [0.2241, 0.6093, 0.1359, 0.0306]  (sums to 1.0)
print(probs.sum().item())  # 1.0
```

### Why softmax for action distributions?

In a policy network, the output is a probability distribution over actions. Softmax gives you this naturally: whatever raw scores the network produces, softmax converts them into valid probabilities. You can then sample from this distribution (using `Categorical(probs=...)` from lesson 1), or take the argmax for a deterministic greedy policy.

Softmax also has a nice training property: it is differentiable everywhere, so gradients flow through it cleanly during backpropagation.

## SSA application: a risk-level classifier

Let us put ReLU and softmax together in a minimal example. Suppose you want a network that takes a 3D conjunction feature vector and classifies the risk level as low, medium, or high.

```python
import torch
import torch.nn.functional as F

# A tiny two-layer network (manually, before using nn.Sequential)
# Input: [approach_speed_kms, miss_distance_km, time_to_closest_approach_hrs]
# Output: logits for [low_risk, medium_risk, high_risk]

torch.manual_seed(42)

# Layer 1: 3 inputs -> 8 hidden neurons
W1 = torch.randn(8, 3) * 0.5
b1 = torch.zeros(8)

# Layer 2: 8 hidden neurons -> 3 outputs (one per risk level)
W2 = torch.randn(3, 8) * 0.5
b2 = torch.zeros(3)

# A conjunction feature vector
x = torch.tensor([7.5, 0.5, 2.0])  # 7.5 km/s approach, 0.5 km miss distance, 2 hrs out

# Forward pass
h = F.relu(W1 @ x + b1)   # Layer 1: linear + ReLU
logits = W2 @ h + b2       # Layer 2: linear (no ReLU before softmax)
probs = F.softmax(logits, dim=0)  # Softmax to get probabilities

print(f"Hidden layer (after ReLU): {h.tolist()[:4]}...")  # first 4 of 8
print(f"Logits:      {logits.tolist()}")
print(f"Probs:       {[f'{p:.3f}' for p in probs.tolist()]}")
print(f"Predicted risk level: {['Low', 'Medium', 'High'][probs.argmax().item()]}")
```

The weights \\(W_1, W_2\\) and biases \\(b_1, b_2\\) are random right now. In lessons 3 and 4, we will train them from data. The structure of the forward pass is what matters here: linear → ReLU → linear → softmax.

## What we do not use on the final layer

A common mistake: applying ReLU to the final layer before softmax. Do not do this. ReLU zeroes out negative values, which would distort the probability computation. The final layer produces logits (raw scores, can be any sign), and softmax handles the conversion to probabilities directly. ReLU is for hidden layers only.

Summary of where each activation goes:

| Layer type | Activation | Reason |
|------------|-----------|--------|
| Hidden layers | ReLU (default) | Fast, avoids vanishing gradients |
| Hidden layers | tanh (sometimes) | Bounded outputs, smooth gradients |
| Output (classification) | Softmax | Converts logits to probabilities |
| Output (regression) | None | Raw linear output is fine |

## Quiz

{{#quiz 01-activation-functions.toml}}
