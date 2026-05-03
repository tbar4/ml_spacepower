# Lesson 3: Loss Functions and What We Are Optimizing

## Where this fits

A neural network with random weights is useless. Training makes it useful. But what does training mean, precisely? It means adjusting the weights to minimize a **loss function**: a single number that measures how wrong the network's current outputs are. Gradient descent (from Module 1, lesson 7) steps the weights in the direction that reduces the loss. The loss function determines what "wrong" means, and choosing the right one is as important as choosing the right architecture.

This lesson covers two loss functions that cover the vast majority of our use cases: mean squared error (when the network outputs a continuous value) and cross-entropy loss (when the network outputs a probability distribution over categories). Both connect directly to concepts from Module 1.

## What a loss function does

A loss function takes two inputs:
1. **The network's prediction**: what the network currently outputs for a given input
2. **The target**: the correct answer for that input

It returns a single non-negative number: the **loss**. A loss of 0 means the prediction is perfect. A larger loss means the prediction is further from the target.

Training loops over examples, computes the loss on each batch, uses backpropagation to get the gradient of the loss with respect to all the weights, and takes a small step to reduce the loss. After many iterations, the weights settle into values that produce low loss on the training data.

The key question is: what should "how wrong" mean for your specific problem?

## Mean Squared Error: for continuous value prediction

### The scenario

Your SSA sensor system generates a conjunction risk score for each tracked pair of objects. That score is a continuous number between 0 and 1. You have 1,000 historical examples of (feature vector, risk score) pairs and you want a neural network to predict the risk score from the feature vector.

This is a **regression** problem: predicting a continuous output. The natural loss function is **Mean Squared Error (MSE)**.

### Building the formula from scratch

Suppose your network outputs a prediction \\(\hat{y}\\) for an example whose true label is \\(y\\).

The **error** for this example is how far off the prediction is: \\(\hat{y} - y\\).

The **squared error** is \\((\hat{y} - y)^2\\). We square it for two reasons:
1. It makes negative and positive errors contribute equally (being 0.3 too high is as bad as 0.3 too low)
2. It penalizes large errors more than small ones (being off by 0.6 is four times worse than being off by 0.3, not twice as bad)

For a batch of N examples, the **mean squared error** is:

\\[ \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 \\]

**Decoding:**
- **\\(\hat{y}_i\\)**: the network's prediction for example i (\\(\hat{}\\) is the "hat" notation for estimates)
- **\\(y_i\\)**: the true label for example i
- **\\((\hat{y}_i - y_i)^2\\)**: the squared error for example i
- **\\(\frac{1}{N} \sum_{i=1}^{N}\\)**: average over all N examples in the batch

### Walking through an example by hand

Suppose you have a batch of 4 examples:

| Example | True risk \\(y_i\\) | Predicted \\(\hat{y}_i\\) | Error \\(\hat{y}_i - y_i\\) | Squared error |
|---------|---------------------|---------------------------|------------------------------|---------------|
| 1       | 0.80                | 0.72                      | -0.08                        | 0.0064        |
| 2       | 0.20                | 0.35                      | +0.15                        | 0.0225        |
| 3       | 0.95                | 0.91                      | -0.04                        | 0.0016        |
| 4       | 0.45                | 0.60                      | +0.15                        | 0.0225        |

MSE = (0.0064 + 0.0225 + 0.0016 + 0.0225) / 4 = 0.0530 / 4 = **0.01325**

The loss is 0.01325. After training, we want this number to be much smaller.

```python
import torch
import torch.nn.functional as F

y_true = torch.tensor([0.80, 0.20, 0.95, 0.45])
y_pred = torch.tensor([0.72, 0.35, 0.91, 0.60])

# By hand
squared_errors = (y_pred - y_true) ** 2
mse_manual = squared_errors.mean()
print(f"MSE (manual):  {mse_manual.item():.6f}")

# PyTorch built-in
mse_pytorch = F.mse_loss(y_pred, y_true)
print(f"MSE (PyTorch): {mse_pytorch.item():.6f}")
```

Both should give the same answer: 0.013250.

### What MSE minimization looks like geometrically

Imagine plotting the loss as a surface over the space of all possible weight values. MSE loss creates a bowl-shaped landscape (approximately, for linear models exactly). Gradient descent rolls the weights downhill toward the minimum. At the minimum, the predictions are as close to the targets as possible.

MSE penalizes large errors quadratically: being off by 0.3 contributes 0.09, being off by 0.6 contributes 0.36 (four times more, not twice). This makes the network pay particular attention to reducing its worst errors.

## Cross-Entropy Loss: for probability predictions

### The scenario

Now suppose instead of a continuous risk score, you want to classify a conjunction event into one of three priority levels: **low** (0), **medium** (1), **high** (2). Your network should output a probability distribution over these three classes. The loss should measure how well that probability distribution matches the true class.

This is a **classification** problem, and the right loss function is **cross-entropy loss**.

### The connection to Module 1

In lesson 4 of Module 1, you learned that cross-entropy \\(H(P, Q)\\) measures how surprised a model using distribution Q would be when the true distribution is P.

For classification, the true distribution P is one-hot: probability 1.0 on the correct class, probability 0.0 on all others. The network's output Q is the softmax probability distribution. Cross-entropy loss is:

\\[ H(P, Q) = -\sum_{c=1}^{C} P(c) \log Q(c) \\]

But since P is one-hot (only one class has nonzero probability), all terms in the sum except the true class drop out:

\\[ \text{Cross-entropy loss} = -\log Q(\text{true class}) \\]

In plain English: **cross-entropy loss is just the negative log probability that the network assigned to the correct answer.**

- If the network says the correct class has probability 0.99: loss = −log(0.99) ≈ 0.01 (small, good prediction)
- If the network says the correct class has probability 0.50: loss = −log(0.50) ≈ 0.693 (moderate)
- If the network says the correct class has probability 0.01: loss = −log(0.01) ≈ 4.605 (large, terrible prediction)

The loss grows rapidly as the network's confidence in the correct class decreases.

### Walking through an example by hand

Your network outputs logits for three classes. After softmax:

| Example | True class | P(low) | P(medium) | P(high) | Loss = -log(P(true class)) |
|---------|-----------|--------|-----------|---------|---------------------------|
| 1       | high (2)  | 0.05   | 0.10      | 0.85    | -log(0.85) = 0.163        |
| 2       | low (0)   | 0.70   | 0.20      | 0.10    | -log(0.70) = 0.357        |
| 3       | medium (1)| 0.30   | 0.35      | 0.35    | -log(0.35) = 1.050        |

Mean cross-entropy loss = (0.163 + 0.357 + 1.050) / 3 = **0.523**

Example 3 drives the loss up: the network is nearly equally unsure between all three classes for a medium-priority event.

```python
import torch
import torch.nn.functional as F

# True class labels (integers: 0=low, 1=medium, 2=high)
y_true = torch.tensor([2, 0, 1])  # high, low, medium

# Raw logits from the network (before softmax)
logits = torch.tensor([
    [-2.0, -1.0,  2.5],  # example 1: strongly predicts high
    [ 2.0,  0.5, -0.5],  # example 2: strongly predicts low
    [ 0.3,  0.5,  0.4],  # example 3: nearly uniform (uncertain)
])

# PyTorch's cross-entropy takes logits (NOT softmax probabilities)
# It applies softmax internally before computing the loss
loss = F.cross_entropy(logits, y_true)
print(f"Cross-entropy loss: {loss.item():.4f}")

# See what the softmax probabilities look like
probs = F.softmax(logits, dim=1)
print("\nPredicted probabilities:")
for i, (p, label) in enumerate(zip(probs, ["high", "low", "medium"])):
    print(f"  Example {i+1} (true={label}): "
          f"low={p[0]:.3f}, med={p[1]:.3f}, high={p[2]:.3f}")
```

**Important**: PyTorch's `F.cross_entropy` takes raw **logits**, not softmax probabilities. It applies softmax internally. This is more numerically stable than applying softmax yourself and then passing the probabilities. Do not apply softmax before `cross_entropy`.

### Why negative log probability?

Minimizing the negative log probability of the correct class is equivalent to maximizing the probability the network assigns to the correct class. It is the likelihood of the training data under the model, which is a natural objective.

The logarithm also prevents vanishing gradient problems: the gradient of −log(p) is −1/p, which gets very large as p approaches 0. This means the gradient is large when the prediction is badly wrong (p close to 0), which produces a strong correction signal. The gradient is small when the prediction is good (p close to 1), which produces a gentle nudge. This is the right behavior: large corrections when wrong, small corrections when right.

## Choosing the right loss function

| Problem type | Output | Loss function | PyTorch function |
|-------------|--------|---------------|------------------|
| Predicting a continuous value | Single number | MSE | `F.mse_loss(pred, target)` |
| Classifying into N categories | N probabilities | Cross-entropy | `F.cross_entropy(logits, target)` |
| Policy (action distribution) | N probabilities | Cross-entropy (or policy gradient) | depends on algorithm |
| Value function approximation | Single number | MSE | `F.mse_loss(pred, target)` |

In RL, the value network uses MSE loss (we are approximating a continuous expected return). The policy network in REINFORCE uses a policy gradient loss that is more complex (covered in Module 3). For deep CFR, the regret network uses MSE loss (approximating a continuous regret value). The pattern is: continuous target → MSE, categorical target → cross-entropy.

## The loss landscape and local minima

MSE and cross-entropy loss are not convex for neural networks. This means gradient descent is not guaranteed to find the global minimum. Instead, it will find a local minimum, or more commonly in practice, a "good enough" region of the loss landscape that generalizes well to new data.

In practice, this is usually fine. Modern neural networks trained with stochastic gradient descent tend to find solutions that work well even though they are not globally optimal. The theoretical reasons are still an active research area. For our purposes: define a loss that measures what you want to optimize, minimize it with gradient descent, and evaluate on held-out test data to check that it generalized.

## Quiz

{{#quiz 03-loss-functions.toml}}
