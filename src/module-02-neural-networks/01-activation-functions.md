# Lesson 1: Activation Functions


<!-- toc -->

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

## Leaky ReLU and ELU: fixing the dying neuron problem

Plain ReLU has a subtle failure mode called the **dying ReLU problem**. Because ReLU outputs exactly zero for any negative pre-activation, a neuron whose weights are initialized in a bad region — where the weighted sum is almost always negative — never fires. Its gradient is exactly zero, so gradient descent never updates those weights. The neuron is permanently dead.

In large networks, it is not unusual to find 10–40% of neurons permanently inactive after training. They contribute nothing. For a 64-neuron hidden layer, that might mean only 40 neurons are actually doing work.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Simulate a neuron whose incoming weights happened to start negative
# (This happens more often than you'd think with random init)
pre_activations = torch.tensor([-2.5, -1.8, -3.1, -0.9, -2.2])

relu_out = F.relu(pre_activations)
print(f"ReLU output:        {relu_out.tolist()}")
# [0.0, 0.0, 0.0, 0.0, 0.0]
# The gradient through ReLU is 0 for all of these.
# These neurons are dead — gradient descent cannot update them.

# Leaky ReLU: max(0.01x, x)
leaky_out = F.leaky_relu(pre_activations, negative_slope=0.01)
print(f"Leaky ReLU output:  {leaky_out.tolist()}")
# [-0.025, -0.018, -0.031, -0.009, -0.022]
# Small but nonzero! Gradient still flows. Neurons can recover.
```

### Leaky ReLU

**Leaky ReLU** lets a small fraction of the negative signal through:

\\[ \text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\\\ 0.01 \cdot x & \text{if } x \leq 0 \end{cases} \\]

The slope for negative inputs (0.01 by default) is the "leak." It is small enough not to dominate but large enough to keep gradients nonzero, so the neuron can recover during training if the weights shift.

### ELU: Exponential Linear Unit

**ELU** uses an exponential curve for negative inputs rather than a fixed linear slope:

\\[ \text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\\\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases} \\]

where \\(\alpha\\) is typically 1.0.

**Decoding:** For negative \\(x\\), the exponential \\(e^x\\) is between 0 and 1, so \\(e^x - 1\\) is between -1 and 0. ELU smoothly saturates near \\(-\alpha\\) for large negative values, which can help with training stability. Unlike Leaky ReLU (which is linear everywhere), ELU has a curved negative region that brings the mean activation of layers closer to zero — a property known to speed up learning.

```python
import torch.nn as nn

elu = nn.ELU(alpha=1.0)
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(elu(x).tolist())
# [-0.9502, -0.6321, 0.0, 1.0, 3.0]
# Negative values saturate toward -1; positive values pass through unchanged.
```

### Which activation to use when

| Situation | Recommended activation | Why |
|-----------|----------------------|-----|
| Default hidden layer | ReLU | Fast, works well, no hyperparameters |
| Seeing many dead neurons (check with `(activations == 0).float().mean()`) | Leaky ReLU | Keeps gradient flow for negative pre-activations |
| Want zero-mean activations without vanishing gradient | ELU | Smooth, negative saturation near -α |
| Bounded output needed | tanh | Always between -1 and +1 |
| Output is a probability (classification) | Softmax | Converts logits to valid distribution |

For the conjunction risk network in SSA applications, ReLU is the right default. If you notice training stalls or large fractions of inactive neurons, switch to Leaky ReLU — it requires no other changes to the architecture.

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

## Temperature in softmax: controlling sharpness

Vanilla softmax has an implicit "temperature" of 1. We can generalize it with an explicit temperature parameter T:

\\[ \text{softmax}_T(\mathbf{z})_i = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}} \\]

**Decoding:** Dividing each logit by T before exponentiating scales all the values. This controls how "sharp" or "spread out" the resulting distribution is.

- **High T (e.g., T = 10)**: Dividing by a large number flattens the logits, making the distribution more uniform. All actions get similar probability. The agent explores more.
- **Low T (e.g., T = 0.1)**: Dividing by a small number amplifies the differences between logits, making the distribution sharper. The highest-scoring action dominates. The agent exploits more.
- **T → 0**: Approaches a one-hot distribution — all probability on the best action. Pure greedy.
- **T → ∞**: Approaches a uniform distribution — equal probability for all actions. Pure random.

This parameter directly controls the **exploration vs. exploitation tradeoff** in reinforcement learning. Early in training, a high temperature encourages the agent to try many actions and gather experience. As training progresses, lowering the temperature makes the agent increasingly commit to the actions it has learned are best.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, 2.0, 0.5, -1.0])

temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]
print(f"{'T':>6}  {'p(a0)':>8}  {'p(a1)':>8}  {'p(a2)':>8}  {'p(a3)':>8}")
print("-" * 50)
for T in temperatures:
    probs = F.softmax(logits / T, dim=0)
    p = [f"{x:.4f}" for x in probs.tolist()]
    print(f"{T:>6.1f}  {p[0]:>8}  {p[1]:>8}  {p[2]:>8}  {p[3]:>8}")
```

Sample output:

| T     | p(a0)  | p(a1)  | p(a2)  | p(a3)  |
|-------|--------|--------|--------|--------|
| 0.1   | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.5   | 0.1192 | 0.8756 | 0.0049 | 0.0000 |
| 1.0   | 0.2241 | 0.6093 | 0.1359 | 0.0306 |
| 2.0   | 0.2592 | 0.4223 | 0.1934 | 0.1251 |
| 10.0  | 0.2462 | 0.2718 | 0.2387 | 0.2433 |

At T=0.1, action 1 has probability ≈ 1.0 (pure exploitation). At T=10.0, all four actions have nearly equal probability (near-uniform exploration). In RL implementations, you will often see temperature schedules that start high and decay over training epochs.

## Sigmoid: binary output decisions

Softmax is designed for choosing among multiple competing outputs. When you have a single yes/no output — is there a conjunction alert? is this track anomalous? — you use **sigmoid** instead.

\\[ \sigma(x) = \frac{1}{1 + e^{-x}} \\]

**Decoding:** The exponential \\(e^{-x}\\) is always positive, so the denominator \\(1 + e^{-x}\\) is always greater than 1. Therefore \\(\sigma(x)\\) is always in the interval (0, 1). For large positive x, \\(e^{-x} \approx 0\\) and \\(\sigma(x) \approx 1\\). For large negative x, \\(e^{-x}\\) is large and \\(\sigma(x) \approx 0\\).

| Input x | σ(x)  | Interpretation |
|---------|-------|----------------|
| -5.0    | 0.007 | Very unlikely  |
| -2.0    | 0.119 | Unlikely       |
| 0.0     | 0.500 | Uncertain      |
| 2.0     | 0.881 | Likely         |
| 5.0     | 0.993 | Very likely    |

### Sigmoid vs. softmax: a critical distinction

Many beginners conflate sigmoid and softmax. They are different tools for different jobs:

- **Sigmoid**: one output, one independent binary decision. "Is this conjunction event a true positive?" The output is a probability from 0 to 1 for that single question.
- **Softmax**: multiple outputs that must sum to 1, representing competing alternatives. "Which of these 5 satellites should I observe?" Each output is a share of a single probability budget.

If you apply softmax to a 2-output network for binary classification, you get the same probabilities as sigmoid — but with an extra redundant output. Sigmoid is cleaner and is the standard choice.

```python
import torch
import torch.nn.functional as F

# Binary conjunction alert classifier output (a single logit)
logit = torch.tensor(2.3)  # raw network output for "is this a real alert?"

# Sigmoid gives probability of alert
prob_alert = torch.sigmoid(logit)
print(f"Logit: {logit.item():.1f}  ->  P(alert) = {prob_alert.item():.4f}")
# Logit: 2.3  ->  P(alert) = 0.9090

# Compare: applying softmax to [logit, -logit] gives the same probs
# but wastes an output
two_class_logits = torch.tensor([logit.item(), -logit.item()])
two_class_probs = F.softmax(two_class_logits, dim=0)
print(f"Softmax equivalent: {two_class_probs.tolist()}")
# [0.9090, 0.0910]  -- same probability for the positive class, but redundant
```

In the SSA context, sigmoid is the right output activation for a network that predicts "probability that satellite pair X will have a conjunction within 72 hours." Softmax is the right output activation for a policy that must allocate observation time across 5 satellites.

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

## Common mistakes: activation function cheat sheet

Getting the activation function wrong is a subtle bug — the network will often still train, just slowly or to a suboptimal solution. Here is a lookup table for the most frequent errors:

| Layer type | Wrong activation | Right activation | Why it matters |
|------------|-----------------|-----------------|----------------|
| Hidden layer | sigmoid | ReLU | Sigmoid saturates, causing vanishing gradients in deep networks; training slows or stalls |
| Output (multi-class) | ReLU | Softmax | ReLU outputs can be negative or exceed 1; they are not valid probabilities |
| Output (binary) | Softmax (2 outputs) | Sigmoid (1 output) | Two-class softmax is redundant; sigmoid is the canonical binary output |
| Output (regression) | Any activation | None (linear) | Any activation function bounds or warps the output range; regression targets can be any real number |
| Output (multi-label) | Softmax | Sigmoid (per output) | Softmax enforces outputs sum to 1, which is wrong when multiple labels can be true simultaneously |

The SSA context makes the output-layer mistakes especially costly. If you put ReLU before softmax in a conjunction risk classifier, negative logits get zeroed before normalization, making the predicted distribution systematically wrong in ways that may not surface until the system misses a real event.

## Key Takeaways

- **Activation functions are what make neural networks nonlinear.** Without them, no matter how many layers you stack, the whole network is equivalent to a single linear transformation — it cannot learn thresholds, risk regimes, or any curved decision boundary.
- **ReLU is the default choice for hidden layers.** It is computationally cheap (just a max), avoids vanishing gradients, and works well across a wide range of architectures. Start with ReLU; switch to a variant only if you observe a specific problem.
- **Dying neurons are a real failure mode.** If a neuron's pre-activation is always negative, its gradient is exactly zero, and gradient descent cannot recover it. Monitor the fraction of zero activations during training; if it exceeds ~40%, switch to Leaky ReLU or ELU.
- **Softmax is for competing outputs; sigmoid is for independent binary outputs.** Using softmax for binary classification adds a redundant output. Using sigmoid for multi-class classification is wrong because outputs do not sum to 1. Match the activation to the output structure.
- **Temperature controls how peaked or spread out a softmax distribution is.** High temperature encourages exploration (uniform-like distribution); low temperature encourages exploitation (near-greedy distribution). This is a primary knob in RL algorithms for managing the exploration-exploitation tradeoff.
- **The final layer's activation must match the task.** Regression outputs need no activation. Classification outputs need softmax (or sigmoid). Applying ReLU to the final layer is almost always a bug — it throws away information about which logits were negative, distorting the output.

## Quiz

{{#quiz 01-activation-functions.toml}}
