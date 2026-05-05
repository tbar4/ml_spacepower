# Lesson 3: Loss Functions and What We Are Optimizing


<!-- toc -->

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

## Huber Loss: robustness to outliers

### Why MSE can hurt you

MSE's quadratic penalty is a double-edged sword. It does make the network attend to its worst errors — but it also means a single corrupted label or measurement outlier can dominate the entire loss. Imagine your SSA data pipeline occasionally mis-tags a benign object as a high-risk conjunction (sensor dropout, coordinate transform bug, stale catalog entry). That one corrupted label has a squared error that might be 10× larger than any real example. Gradient descent will spend enormous energy chasing it.

**Huber loss** solves this by being quadratic for small errors and linear for large ones. Below the threshold \\(\delta\\), it behaves exactly like MSE. Above \\(\delta\\), it grows linearly — the outlier still contributes to the loss, but its influence is bounded.

### The formula

\\[
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\\\
\delta \cdot \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{otherwise}
\end{cases}
\\]

**Decoding:**
- **\\(\delta\\)** (delta): the threshold that separates "small error" from "large error." Common default is 1.0. A smaller \\(\delta\\) transitions to linear sooner (more robust, but less sensitive to genuine large errors). A larger \\(\delta\\) stays quadratic longer (behaves more like MSE).
- **\\(\frac{1}{2}(y - \hat{y})^2\\)**: the quadratic region — identical to MSE (with a ½ factor for clean derivative math).
- **\\(\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta)\\)**: the linear region — grows at rate \\(\delta\\) per unit of additional error, not quadratically.
- The two pieces meet smoothly at \\(|y - \hat{y}| = \delta\\), so there is no sharp kink in the loss surface.

### DQN and TD error stability

In Deep Q-Networks (DQN), the loss is computed on the **temporal difference (TD) error**: the difference between the Q-network's current estimate and the TD target (reward + discounted next-state Q-value). Early in training, Q-estimates can be wildly off, and TD errors can be enormous. MSE on a TD error of 50 produces a gradient of 100 — a weight update large enough to destabilize the network.

Huber loss clips this: a TD error of 50 with \\(\delta=1\\) produces a gradient of magnitude 1, not 100. Training stabilizes. This is why the original DQN paper (Mnih et al., 2015) used Huber loss rather than MSE.

```python
import torch
import torch.nn.functional as F

# Conjunction risk predictions and targets, with one outlier
y_true = torch.tensor([0.80, 0.20, 0.95, 0.45,  0.10])
y_pred = torch.tensor([0.72, 0.35, 0.91, 0.60,  0.98])  # last one is badly wrong

mse   = F.mse_loss(y_pred, y_true)
huber = F.huber_loss(y_pred, y_true, delta=1.0)

print(f"MSE loss:   {mse.item():.6f}")    # dominated by the outlier
print(f"Huber loss: {huber.item():.6f}")  # outlier's influence is bounded

# In a DQN training loop:
# q_values = online_net(states).gather(1, actions)
# with torch.no_grad():
#     td_targets = rewards + gamma * target_net(next_states).max(1).values
#
# td_loss = F.huber_loss(q_values.squeeze(), td_targets, delta=1.0)
# td_loss.backward()
```

### When to use Huber loss

Use Huber loss when:
- Your training labels come from a noisy source (sensor readings, human annotations, simulated environments with occasional bugs)
- You are training a value function in RL where early TD errors can be arbitrarily large
- You suspect your dataset has a small fraction of corrupted or mislabeled examples

Use MSE when:
- Your labels are clean and accurate
- You want the network to aggressively minimize its largest errors (not just get close)
- The label-generating process is Gaussian with small variance (MSE is the maximum likelihood estimator for Gaussian noise)

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

## Numerical stability: never manually compute log(softmax)

### The problem with naive computation

It is tempting to apply softmax yourself, then pass probabilities to a log. Here is why that is a mistake:

```python
import torch
import torch.nn.functional as F

# Logits with one very dominant class (common in early training)
logits = torch.tensor([[10.0, 0.0, 0.0]])
true_class = torch.tensor([0])

# WRONG: manual softmax then log — unstable for extreme logits
probs = F.softmax(logits, dim=1)
manual_loss = -torch.log(probs[0, true_class]).mean()

# RIGHT: use F.cross_entropy directly — applies log-sum-exp trick internally
stable_loss = F.cross_entropy(logits, true_class)

print(f"Manual (unsafe):  {manual_loss.item():.6f}")
print(f"Stable (correct): {stable_loss.item():.6f}")

# Now try with extreme logits that cause underflow:
extreme_logits = torch.tensor([[0.001, 0.001, 0.001]])  # nearly uniform, tiny values
probs_extreme = F.softmax(extreme_logits, dim=1)
# probs_extreme values are ~0.333 — fine so far

# But imagine the reverse: large negative logits
very_negative = torch.tensor([[-100.0, -100.0, -100.0]])
probs_neg = F.softmax(very_negative, dim=1)
log_probs_neg = torch.log(probs_neg)
print(f"\nLog of softmax (manual, extreme): {log_probs_neg}")
# May produce -inf or nan depending on the platform

log_probs_stable = F.log_softmax(very_negative, dim=1)
print(f"Log-softmax (stable):             {log_probs_stable}")
# Numerically correct even for extreme inputs
```

### What goes wrong

Softmax computes \\(\exp(x_i) / \sum_j \exp(x_j)\\). When logits are very large, `exp(x)` overflows to `inf`. When logits are very small (large negative), `exp(x)` underflows to `0.0`, and `log(0.0)` is `-inf`. Either way, your loss and gradients are corrupted.

The solution uses the **log-sum-exp trick**: subtract the maximum logit before exponentiating, compute in log-space, then add back. PyTorch implements this in `F.log_softmax` and `F.cross_entropy`.

**Rule:** always use `F.cross_entropy(logits, targets)` — never `F.nll_loss(F.softmax(logits).log(), targets)` or anything equivalent. The former takes raw logits and handles numerical stability internally. This is not an optimization detail: on real SSA classification data, where one orbit class can have logits 10× larger than others, the unstable version will silently produce `nan` losses and corrupt your weights.

### log_softmax vs cross_entropy

```python
# These three are equivalent; prefer the first:
loss1 = F.cross_entropy(logits, targets)                        # preferred
loss2 = F.nll_loss(F.log_softmax(logits, dim=1), targets)       # equivalent, verbose
loss3 = -F.log_softmax(logits, dim=1)[range(N), targets].mean() # equivalent, manual

# F.cross_entropy is why the API takes logits, not probabilities.
# If you pass probabilities by mistake:
wrong_input = F.softmax(logits, dim=1)          # already probabilities
F.cross_entropy(wrong_input, targets)           # silently produces wrong answer
# The function treats them as logits and applies softmax *again*.
```

## Gradient magnitudes: why these loss functions work

Understanding the gradient of the loss with respect to the prediction helps explain why these loss functions are well-suited to their tasks.

### MSE gradient

\\[ \text{MSE} = (\hat{y} - y)^2 \quad \Rightarrow \quad \frac{\partial \text{MSE}}{\partial \hat{y}} = 2(\hat{y} - y) \\]

**Decoding:** The gradient is zero when \\(\hat{y} = y\\) (perfect prediction) and grows linearly as the error grows. This is the right behavior: no update needed when correct, proportionally larger update when wrong.

### Cross-entropy gradient

For the softmax-cross-entropy combination, the gradient with respect to the logit for the true class \\(z_c\\) is:

\\[ \frac{\partial \text{CE}}{\partial z_c} = p_c - 1 \\]

where \\(p_c\\) is the predicted probability for the true class.

**Decoding:** The gradient is \\(-1 + p_c\\). When \\(p_c \approx 0\\) (network is confidently wrong), the gradient is close to \\(-1\\) — a large correction. When \\(p_c \approx 1\\) (network is correct), the gradient is close to \\(0\\) — a tiny nudge. This is exactly the right signal.

Compare to what you would get from MSE on probabilities (\\((\hat{p} - 1)^2\\)):

| Predicted probability | CE gradient | MSE gradient | Which is bigger? |
|----------------------|-------------|--------------|-----------------|
| p = 0.01 (very wrong) | -0.99 | -1.98 | MSE (slightly) |
| p = 0.50 (uncertain)  | -0.50 | -1.00 | MSE |
| p = 0.90 (close)      | -0.10 | -0.20 | MSE |
| p = 0.99 (correct)    | -0.01 | -0.02 | Equal (both ~0) |

For classification, cross-entropy is preferred not because the gradients are larger, but because the loss landscape is smoother and the gradient near zero is correct — the network gets only a small nudge once it is already confident and right.

```python
import torch

# Manually compute gradients for both loss functions
p = torch.linspace(0.01, 0.99, 10)  # predicted probabilities

ce_gradient  = p - 1.0              # d(CE)/d(logit) = p - 1
mse_gradient = 2 * (p - 1.0)        # d(MSE)/d(p) = 2*(p - y), y=1

print(f"{'p':>6} | {'CE grad':>10} | {'MSE grad':>10}")
print("-" * 32)
for pi, ce, mse in zip(p, ce_gradient, mse_gradient):
    print(f"{pi.item():>6.2f} | {ce.item():>10.4f} | {mse.item():>10.4f}")
```

## The probabilistic interpretation of loss functions

Every standard loss function is secretly a maximum likelihood estimator. Understanding this connection gives you a principled way to derive new loss functions when your problem is non-standard, and it explains why L2 regularization and Gaussian priors are the same thing.

### MSE = MLE under Gaussian noise

Suppose each training label \\(y_i\\) is generated by the true function plus independent Gaussian noise:

\\[ y_i = f(x_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2) \\]

This means the likelihood of observing label \\(y_i\\) given prediction \\(\hat{y}_i = f_\theta(x_i)\\) is:

\\[ p(y_i \mid x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right) \\]

**Decoding:**
- The model says \\(y_i\\) is Gaussian-distributed around \\(\hat{y}_i\\)
- A label close to the prediction has high likelihood; a label far away has low likelihood
- \\(\sigma^2\\) is the assumed noise variance

The **log-likelihood** over all \\(N\\) training examples is:

\\[ \log p(\mathcal{D} \mid \theta) = \sum_{i=1}^N \log p(y_i \mid x_i, \theta) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \hat{y}_i)^2 \\]

**Maximizing** this log-likelihood is equivalent to **minimizing** \\(\sum_i (y_i - \hat{y}_i)^2\\), which is exactly MSE (up to a constant \\(1/N\\) scaling).

> Conclusion: **MSE is MLE under a Gaussian likelihood.** Choosing MSE implicitly assumes your labels are corrupted by Gaussian noise. If your noise is actually heavy-tailed (outliers), a more appropriate likelihood gives Huber or absolute-error loss.

### Cross-entropy = MLE under categorical likelihood

For classification, the label \\(y_i \in \{1, \ldots, C\}\\) is drawn from a categorical distribution parameterized by the network's softmax output \\(\hat{p}_i \in \mathbb{R}^C\\):

\\[ p(y_i = c \mid x_i, \theta) = \hat{p}_{i,c} \\]

The log-likelihood is:

\\[ \log p(\mathcal{D} \mid \theta) = \sum_{i=1}^N \log \hat{p}_{i, y_i} = -\sum_{i=1}^N \text{CE}(y_i, \hat{p}_i) \\]

**Minimizing cross-entropy loss equals maximizing the categorical log-likelihood.** This explains why cross-entropy is the right loss for any problem where the network is trying to predict a probability distribution: it is the natural MLE objective for that output type.

### L2 regularization = MAP with a Gaussian prior

Plain MLE can overfit: the weights grow large to memorize training data. The fix is to add a prior over the weights and compute the **maximum a posteriori (MAP)** estimate instead.

Choose a Gaussian prior \\(p(\theta) = \mathcal{N}(0, (1/\lambda) \cdot I)\\). The log-posterior is:

\\[ \log p(\theta \mid \mathcal{D}) = \log p(\mathcal{D} \mid \theta) + \log p(\theta) + \text{const} \\]

\\[ = \underbrace{\log p(\mathcal{D} \mid \theta)}_{\text{log-likelihood}} - \underbrace{\frac{\lambda}{2} \|\theta\|^2}_{\text{Gaussian log-prior}} + \text{const} \\]

Maximizing this is equivalent to minimizing:

\\[ \mathcal{L}_\text{MAP} = -\log p(\mathcal{D} \mid \theta) + \frac{\lambda}{2} \|\theta\|^2 \\]

The second term is **L2 regularization** (weight decay). The regularization strength \\(\lambda\\) is the precision (inverse variance) of the prior: larger \\(\lambda\\) means a tighter prior that pulls weights closer to zero.

> This is why the lesson on constrained optimization (Module 1, Lesson 10) discusses weight decay as a Lagrangian penalty: you are computing MAP with a Gaussian prior, and \\(\lambda\\) is the Lagrange multiplier for the norm constraint.

```python
import torch
import torch.nn as nn

# Two ways to express the same MAP objective for an MSE regression model

# --- Option 1: explicit Gaussian MAP ---
def map_loss(model, x, y, lam=1e-3):
    y_pred = model(x).squeeze()
    nll = torch.mean((y - y_pred) ** 2)           # negative log-likelihood (MSE)
    log_prior = sum(p.pow(2).sum() for p in model.parameters())
    return nll + lam * log_prior                   # MAP = NLL + prior penalty

# --- Option 2: PyTorch optimizer weight_decay (identical math) ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# weight_decay adds lambda * ||theta||^2 to the gradient automatically

# Both are equivalent; weight_decay is the standard choice in practice.
```

### Choosing a loss function from first principles

| Noise model for labels | Likelihood | Loss function |
|------------------------|-----------|---------------|
| Gaussian \\(\mathcal{N}(y \mid \hat{y}, \sigma^2)\\) | MLE | MSE |
| Laplace \\(\text{Laplace}(y \mid \hat{y}, b)\\) | MLE | MAE (L1) |
| Huber (Gaussian + heavy tails) | MLE | Huber loss |
| Categorical \\(\text{Cat}(y \mid \hat{p})\\) | MLE | Cross-entropy |
| Gaussian + Gaussian weight prior | MAP | MSE + L2 |
| Gaussian + Laplace weight prior | MAP | MSE + L1 (sparsity) |

For SSA applications: if your conjunction-risk labels come from a physics-based simulator with well-characterized Gaussian output noise, MSE is the principled choice. If labels come from human analysts who occasionally disagree wildly, Huber loss is appropriate. If you are classifying RSO maneuver intent into categories, cross-entropy is correct.

## Loss functions for reinforcement learning

Standard supervised learning uses MSE and cross-entropy. RL introduces additional loss formulations that appear throughout Modules 3–5.

### Value function loss (DQN)

The Q-network estimates \\(Q(s, a)\\): the expected cumulative reward for taking action \\(a\\) in state \\(s\\). Training uses MSE between the Q-estimate and the TD target:

\\[ \mathcal{L}_\text{DQN} = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s, a)\right)^2\right] \\]

In practice, Huber loss is used instead of MSE for stability (see earlier section). For SSA applications, the "state" might be a vector of conjunction features and the "action" might be which sensor to task next for follow-up observation.

### Policy gradient loss (REINFORCE)

The policy gradient loss is not a loss in the supervised sense — you do not have a target to compare against. Instead, you maximize the expected reward by pushing up the log-probability of actions that led to high advantage:

\\[ \mathcal{L}_\text{PG} = -\log \pi(a \mid s) \cdot A(s, a) \\]

**Decoding:**
- **\\(\pi(a \mid s)\\)**: the policy network's probability of taking action \\(a\\) in state \\(s\\)
- **\\(A(s, a)\\)**: the **advantage** — how much better action \\(a\\) was compared to the average action in state \\(s\\)
- **Negative sign**: we flip the sign because PyTorch minimizes, but we want to maximize reward
- If the advantage is positive (action was better than average), we decrease the loss by increasing \\(\log \pi(a \mid s)\\), making the action more likely
- If the advantage is negative (action was worse than average), we increase the loss, making the action less likely

### Entropy bonus

Pure policy gradient tends to converge prematurely to deterministic policies — the network becomes overconfident in one action and stops exploring. The entropy bonus adds a term that rewards maintaining uncertainty:

\\[ \mathcal{L}_\text{total} = \mathcal{L}_\text{PG} - \beta \cdot H(\pi(\cdot \mid s)) \\]

where \\(H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)\\) is the entropy of the policy and \\(\beta\\) is a small coefficient (typically 0.01–0.1). Subtracting entropy means reducing the loss by having a high-entropy (exploratory) policy.

```python
import torch
import torch.nn.functional as F

# Policy network output (logits for 3 sensor-tasking actions)
logits = torch.tensor([[1.5, 0.5, -0.3]])
log_probs = F.log_softmax(logits, dim=1)
probs     = log_probs.exp()

# Advantage estimate for the selected action (action index 0)
action     = torch.tensor([0])
advantage  = torch.tensor([0.8])   # this action was better than average

# Policy gradient loss
pg_loss = -(log_probs[0, action] * advantage).mean()

# Entropy bonus (we want to maximize entropy, so subtract it from the loss)
entropy    = -(probs * log_probs).sum(dim=1).mean()
beta       = 0.01
total_loss = pg_loss - beta * entropy

print(f"PG loss:     {pg_loss.item():.4f}")
print(f"Entropy:     {entropy.item():.4f}")
print(f"Total loss:  {total_loss.item():.4f}")
```

### Regret network loss (Deep CFR)

Deep Counterfactual Regret Minimization (Deep CFR, covered in Module 5) trains a neural network to predict the **cumulative regret** for each action at each information set. This is a regression target — use MSE:

\\[ \mathcal{L}_\text{CFR} = \mathbb{E}\left[\left(R_\text{predicted}(a) - R_\text{actual}(a)\right)^2\right] \\]

The regret values can range widely (they accumulate over many iterations), making Huber loss an option if they become unstable.

### Summary table

| Problem type | Output | Loss function | PyTorch function |
|-------------|--------|---------------|------------------|
| Predicting a continuous value | Single number | MSE | `F.mse_loss(pred, target)` |
| Regression with noisy/outlier labels | Single number | Huber | `F.huber_loss(pred, target, delta=1.0)` |
| Classifying into N categories | N probabilities | Cross-entropy | `F.cross_entropy(logits, target)` |
| DQN value function | Single Q-value | Huber (on TD error) | `F.huber_loss(q_est, td_target)` |
| Policy gradient (REINFORCE) | Action log-prob | Policy gradient loss | `-(log_pi * advantage).mean()` |
| Entropy bonus | Policy entropy | Negative entropy | `-(probs * log_probs).sum()` |
| Deep CFR regret network | Regret per action | MSE | `F.mse_loss(pred_regret, actual_regret)` |

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

## Key Takeaways

- **MSE is for regression; cross-entropy is for classification.** The loss function encodes what "wrong" means for your problem. Using the wrong one produces training that technically runs but converges to a poor model.
- **MSE penalizes outliers quadratically.** A prediction that is 3 units off contributes 9× more to the loss than one that is 1 unit off. In SSA datasets with occasional sensor artifacts or mislabeled events, this can dominate training.
- **Huber loss gives you the best of both worlds** for noisy data and RL value functions. It is quadratic near zero (sensitive to small errors) and linear far from zero (robust to outliers). DQN uses Huber loss on TD error because early Q-estimates can be wildly off.
- **Never compute log(softmax(x)) manually.** Use `F.log_softmax` or `F.cross_entropy` (which takes raw logits and handles stability internally). Manual softmax followed by log produces `-inf` and `nan` for extreme logits, silently corrupting your weights.
- **Cross-entropy's gradient is well-behaved for classification:** close to 1.0 when the network is confidently wrong, close to 0.0 when correct. This gives strong correction signals where they are needed and gentle nudges where they are not.
- **RL introduces additional loss formulations** beyond MSE and cross-entropy: policy gradient loss pushes up the probability of high-advantage actions, entropy bonus keeps the policy exploratory, and regret network loss (Deep CFR) is regression over accumulated regret values.

---

## Quiz

{{#quiz 03-loss-functions.toml}}
