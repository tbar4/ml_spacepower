# Lesson 4: Entropy, Cross-Entropy, and KL Divergence

**Module:** Foundations — M01: Mathematical Foundations for ML and Game Theory
**Source:** *Elements of Information Theory* — Cover & Thomas, Chapters 2 and 3 (Entropy, Relative Entropy, and Mutual Information); *Pattern Recognition and Machine Learning* — Bishop, Chapter 1.6 (Information Theory); *Deep Learning* — Goodfellow, Bengio, Courville, Chapter 3.13 (Information Theory)

---

## Where this fits

Three quantities, all measuring something about probability distributions. They show up in specific and important places downstream. Policy gradient methods add entropy bonuses to encourage exploration. PPO and TRPO constrain policy updates using KL divergence. Cross-entropy is the training loss for nearly every classification network. If you have ever seen a training log print "cross-entropy loss = 0.43" or a paper say "we constrain the KL between old and new policy," this lesson is where those terms become concrete.

The good news: all three quantities reduce to one idea, which we will build up from an SSA scenario.

---

## Starting from scratch: what is surprise?

Your space operations center receives automated alerts whenever the catalog detects a significant event. These alerts are categorized:

| Alert type                        | Probability | Your reaction |
|-----------------------------------|-------------|---------------|
| Routine conjunction warning       | 0.70        | Expected, handled by procedure |
| Debris cloud update               | 0.20        | Notable, moderate attention |
| Uncontrolled reentry warning      | 0.08        | Significant, escalate |
| Adversarial maneuver detected     | 0.02        | Urgent, emergency response |

When a routine conjunction warning comes in (probability 0.70), you are barely surprised. This happens all the time. When an adversarial maneuver alert comes in (probability 0.02), you are very surprised. This almost never happens.

**Surprise is inversely related to probability.** Common events are unsurprising. Rare events are surprising.

We can make this precise. The mathematical definition of surprise for an event with probability \\(p\\) is:

\\[ \text{surprise}(p) = -\log(p) \\]

Let us compute surprise for each alert type and see if it matches our intuition:

| Alert type                    | Probability | \\(-\log(p)\\) (natural log) |
|-------------------------------|-------------|-------------------------------|
| Routine conjunction warning   | 0.70        | 0.357 (not very surprising)   |
| Debris cloud update           | 0.20        | 1.609 (somewhat surprising)   |
| Uncontrolled reentry warning  | 0.08        | 2.526 (quite surprising)      |
| Adversarial maneuver          | 0.02        | 3.912 (very surprising)       |

Rare events (low probability) get high surprise scores. Common events (high probability) get low surprise scores. A guaranteed event (probability 1.0) gets a surprise of −log(1) = 0 (no surprise at all).

**Why the negative sign?** Because log(p) is negative when p < 1 (log of a fraction is negative), and we want surprise to be a positive number. The negative sign flips it to positive.

**Why a logarithm?** Two reasons. First, independent events should have additive surprise. If two independent events each occur, you should be exactly as surprised as the sum of surprises for each individually. Logarithms turn multiplication into addition: −log(p₁ × p₂) = −log(p₁) + (−log(p₂)). Second, the logarithm grows slowly at first then quickly, capturing the intuition that going from 50% to 10% feels less dramatic than going from 2% to 0.1%, even though both are 5× reductions in probability.

---

## Bits vs. nats: two choices of logarithm base

The surprise formula \\(-\log(p)\\) leaves one choice open: which base of logarithm? This choice determines the unit of information.

**Bits (base 2):** If you use \\(\log_2\\), surprise is measured in **bits**. This is the unit from classical information theory and digital communication. A fair coin flip carries exactly 1 bit of information: \\(-\log_2(0.5) = 1\\). A message drawn uniformly from an alphabet of 8 symbols carries 3 bits per symbol: \\(-\log_2(1/8) = 3\\). Bits are the natural unit when you think about how many binary digits you need to encode something.

**Nats (natural log):** If you use \\(\ln\\) (the natural logarithm, base \\(e\\)), surprise is measured in **nats**. This is the unit used in machine learning, statistics, and physics. The reason ML uses nats is practical: gradient computation is cleaner with natural logarithm because \\(\frac{d}{dx}\ln(x) = 1/x\\), without any prefactor. When you compute \\(\nabla_\theta \ln \pi_\theta(a|s)\\) in the policy gradient theorem, you want that clean derivative.

**The conversion:** 1 nat = \\(\log_2(e) \approx 1.4427\\) bits. A fair coin flip in nats: \\(-\ln(0.5) = \ln(2) \approx 0.693\\) nats. Multiplying by 1.4427 gives 1 bit, as expected.

**Which should you use?** Always use the same base consistently within a calculation. When reading papers: if entropy values are around 0.69 for a fair coin, they are in nats. If entropy values are 1.0 for a fair coin, they are in bits. In PyTorch: `torch.log` is natural log (nats); `torch.log2` gives bits.

```python
import torch
import math

# Alert distribution from the SSA operations center
probs = torch.tensor([0.70, 0.20, 0.08, 0.02])
labels = ["Routine conj.", "Debris cloud", "Reentry", "Adversarial"]

print(f"{'Alert type':<18} {'p':>6}  {'Surprise (nats)':>16}  {'Surprise (bits)':>16}")
print("-" * 62)
for label, p in zip(labels, probs.tolist()):
    surprise_nats = -math.log(p)
    surprise_bits = -math.log2(p)
    print(f"{label:<18} {p:>6.2f}  {surprise_nats:>16.3f}  {surprise_bits:>16.3f}")

# Entropy in both units
entropy_nats = -(probs * torch.log(probs)).sum().item()
entropy_bits = -(probs * torch.log2(probs)).sum().item()
print(f"\nEntropy in nats: {entropy_nats:.4f}")
print(f"Entropy in bits: {entropy_bits:.4f}")
print(f"Conversion check: {entropy_nats * math.log2(math.e):.4f} bits  (= nats × log2(e))")

# Fair coin: should be 1 bit, ln(2) nats
print(f"\nFair coin entropy: {math.log(2):.4f} nats = {1.0:.4f} bit")
```

---

## Entropy: average surprise

Now here is the key question: how surprising is your alert system, on average?

You do not know which alert will come next. But you know the probabilities. The expected surprise is the average amount of surprise per alert, weighted by how often each alert type occurs.

Using the expectation formula from lesson 1:

\\[ \text{average surprise} = \sum_{\text{all types}} P(\text{type}) \times \text{surprise}(\text{type}) \\]

Let us compute it:

| Alert type                    | Prob \\(p\\) | Surprise \\(-\log(p)\\) | Contribution \\(p \times (-\log p)\\) |
|-------------------------------|------------|-------------------------|---------------------------------------|
| Routine conjunction           | 0.70       | 0.357                   | 0.250                                 |
| Debris cloud                  | 0.20       | 1.609                   | 0.322                                 |
| Uncontrolled reentry          | 0.08       | 2.526                   | 0.202                                 |
| Adversarial maneuver          | 0.02       | 3.912                   | 0.078                                 |
| **Total**                     |            |                         | **0.852**                             |

The average surprise is 0.852. This quantity is the **entropy** of the alert distribution.

**Entropy measures how uncertain a distribution is.** High entropy means you are often surprised (the distribution is spread out, unpredictable). Low entropy means you are rarely surprised (one or a few outcomes dominate and you almost always know what is coming).

---

## The entropy formula

Entropy of a distribution P is written:

\\[ H(P) = -\sum_{x} P(x) \log P(x) \\]

**Decoding each symbol:**

**\\(H(P)\\)**: The entropy of distribution P. H stands for "Hartley" (an early information theorist), and P is the distribution. The parentheses just mean "the entropy of P."

**\\(-\\)**: The negative sign. Without it, the expression would be negative (since log of a probability < 1 is negative). The negative sign makes entropy positive.

**\\(\sum_{x}\\)**: Sum over all possible outcomes x. In our alert example, x ranges over the four alert types.

**\\(P(x)\\)**: The probability of outcome x under distribution P.

**\\(\log P(x)\\)**: The logarithm of that probability.

**\\(P(x) \log P(x)\\)**: Probability times log-probability. Note that this is different from the surprise calculation: surprise is \\(-\log P(x)\\), but the contribution to entropy is \\(P(x) \times (-\log P(x))\\), the surprise weighted by how often it occurs.

**Reading in English**: "For each possible outcome, multiply its probability by its log-probability, sum all those products, and negate the result."

This is just the expectation of surprise: \\(H(P) = \mathbb{E}[-\log P(X)]\\).

---

## Maximum and minimum entropy

**Minimum entropy** (zero) occurs when one outcome has probability 1 and all others have probability 0. A completely determined distribution. A deterministic policy has zero entropy; you know exactly which action it will take.

**Maximum entropy** occurs when the distribution is uniform: all outcomes equally likely. For four outcomes, maximum entropy would be \\(-4 \times (0.25 \times \log 0.25) = \log 4 \approx 1.386\\). A uniform policy over actions is maximally uncertain; you have no idea which action the agent will take.

Your alert distribution (entropy ≈ 0.852) is between these extremes, closer to the minimum. You are not completely surprised on average, because most alerts are routine.

### The maximum entropy principle

Why does a uniform distribution maximize entropy, and what does this mean in practice?

Entropy is maximized by the distribution that is "as spread out as possible" subject to known constraints. If you have no information beyond "there are four alert types," the uniform distribution is the honest representation of your ignorance: it encodes no preference for one outcome over another.

This is the **maximum entropy principle** in Bayesian reasoning: among all distributions consistent with the constraints you actually know, use the one with highest entropy. Using a lower-entropy distribution means claiming knowledge you do not have.

The principle gives concrete answers for common constraint types:

- **No constraints beyond valid probability**: use the uniform distribution.
- **Known mean (e.g., average alert rate = 10/hour)**: if alerts arrive as a Poisson process, the maximum entropy distribution for inter-arrival times subject to "known mean rate" is the **Exponential distribution**. In SSA terms: if you only know that your sensor generates 10 conjunction alerts per hour on average, and you want the least-informative model of the time between consecutive alerts, use Exponential(rate=10). Any other distribution would be claiming structure you do not have.
- **Known mean and variance**: the maximum entropy distribution is the Gaussian.

```python
import torch
from torch.distributions import Categorical, Exponential

# Maximum entropy for N outcomes = log(N) (uniform)
for n_outcomes in [2, 4, 8, 16]:
    max_h = math.log(n_outcomes)
    uniform_h = Categorical(probs=torch.ones(n_outcomes) / n_outcomes).entropy().item()
    print(f"N={n_outcomes}: max entropy = {max_h:.4f} nats, "
          f"uniform entropy = {uniform_h:.4f} nats")

print()
# Maximum entropy distribution for inter-arrival times with known mean rate
# = Exponential. Verify it has higher entropy than a same-mean truncated distribution.
rate = 10.0  # alerts per hour
exp_dist = Exponential(rate=torch.tensor(rate))
print(f"Exponential(rate=10) entropy: {exp_dist.entropy().item():.4f} nats")
print(f"  (This is the max-entropy distribution for inter-arrival times with mean=0.1 hr)")
```

In RL, the maximum entropy principle motivates **entropy regularization**: adding \\(\alpha H(\pi)\\) to the reward to encourage exploration. The agent is pushed toward the maximum entropy policy that still achieves high expected reward — uncertain unless it has a good reason to be certain.

---

## Cross-entropy: surprise when using the wrong model

Now suppose a new analyst joins your team. Based on their prior experience at a different space ops center, they have a different model of alert probabilities:

| Alert type                    | True probability \\(P\\) | Analyst's model \\(Q\\) |
|-------------------------------|--------------------------|--------------------------|
| Routine conjunction           | 0.70                     | 0.40                     |
| Debris cloud                  | 0.20                     | 0.30                     |
| Uncontrolled reentry          | 0.08                     | 0.20                     |
| Adversarial maneuver          | 0.02                     | 0.10                     |

The analyst thinks adversarial maneuvers are much more common than they actually are (10% vs 2%), and underestimates routine conjunctions (40% vs 70%).

When alerts actually arrive (following the true distribution P), how surprised will the analyst be on average?

The analyst's surprise when alert type x occurs is \\(-\log Q(x)\\), because they are using their model Q to form expectations. The actual frequency of each alert type follows P. So the analyst's average surprise is:

\\[ \sum_{x} P(x) \times (-\log Q(x)) \\]

This is the **cross-entropy** of P and Q:

\\[ H(P, Q) = -\sum_{x} P(x) \log Q(x) \\]

Let us compute it for your analyst:

| Alert type           | True prob \\(P(x)\\) | Analyst surprise \\(-\log Q(x)\\) | Contribution |
|----------------------|----------------------|-----------------------------------|--------------|
| Routine conjunction  | 0.70                 | \\(-\log(0.40)\\) = 0.916         | 0.641        |
| Debris cloud         | 0.20                 | \\(-\log(0.30)\\) = 1.204         | 0.241        |
| Uncontrolled reentry | 0.08                 | \\(-\log(0.20)\\) = 1.609         | 0.129        |
| Adversarial maneuver | 0.02                 | \\(-\log(0.10)\\) = 2.303         | 0.046        |
| **Total**            |                      |                                   | **1.057**    |

The analyst's average surprise is 1.057, compared to 0.852 for someone who knows the true distribution. The analyst experiences more surprise than necessary because their model is wrong.

Notice: when Q = P (the analyst's model matches reality perfectly), cross-entropy equals entropy. The cross-entropy is always at least as large as the entropy, and the gap tells you how much extra surprise the wrong model causes.

---

## Binary cross-entropy: the special case for classification

The most common loss function in ML is a special case of cross-entropy for two-class (binary) problems: **binary cross-entropy (BCE)**.

When there are only two outcomes — conjunction risk above threshold (positive) or below (negative) — every true label is a degenerate distribution: either 100% probability on the positive class, or 100% on the negative class. The neural network outputs a scalar \\(p \in (0, 1)\\) predicting the probability of the positive class.

The BCE loss for a single example with true label \\(y \in \{0, 1\}\\) and predicted probability \\(\hat{p}\\) is:

\\[ L_{\text{BCE}} = -\left[y \log \hat{p} + (1 - y) \log(1 - \hat{p})\right] \\]

**Decoding:**

**\\(y \log \hat{p}\\)**: When \\(y = 1\\) (positive class), this term is \\(\log \hat{p}\\) — the log-probability the model assigned to the correct class. We want this large (close to 0), which means we want \\(\hat{p}\\) close to 1.

**\\((1 - y) \log(1 - \hat{p})\\)**: When \\(y = 0\\) (negative class), this term is \\(\log(1 - \hat{p})\\) — the log-probability of the negative class. We want \\(\hat{p}\\) close to 0.

**Why only one term is active at a time**: when \\(y = 1\\), the \\((1-y)\\) factor zeroes out the second term. When \\(y = 0\\), the \\(y\\) factor zeroes out the first. You are always computing the cross-entropy between the degenerate true distribution and the model's prediction.

**Why this is cross-entropy:** The true label \\(y = 1\\) represents a degenerate distribution \\(P = [0, 1]\\) over {negative, positive}. The model's output represents \\(Q = [1-\hat{p}, \hat{p}]\\). The cross-entropy is \\(H(P, Q) = -[0 \cdot \log(1-\hat{p}) + 1 \cdot \log \hat{p}] = -\log \hat{p}\\), which is the BCE formula with \\(y = 1\\). The full two-term formula handles both cases compactly.

In SSA terms: a conjunction-risk binary classifier predicts whether a given RSO pair poses a collision risk above the 1-in-10,000 threshold. The BCE loss is the natural training objective — it penalizes the model proportionally to how surprised it would be by the true label, given its prediction.

```python
import torch
import torch.nn as nn

# Manual BCE for SSA conjunction-risk prediction
# Scenario: a batch of 6 RSO pairs with true risk labels
y_true = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])  # 1 = high-risk conjunction
y_pred = torch.tensor([0.85, 0.10, 0.60, 0.40, 0.05, 0.92])  # model predictions

# Manual BCE
eps = 1e-8  # avoid log(0)
bce_manual = -(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))
print("Per-sample BCE loss (manual):")
for i, (yt, yp, loss) in enumerate(zip(y_true.tolist(), y_pred.tolist(), bce_manual.tolist())):
    print(f"  Pair {i+1}: y={yt:.0f}, p_hat={yp:.2f}, BCE={loss:.4f}")

print(f"\nMean BCE loss (manual): {bce_manual.mean().item():.4f}")

# PyTorch's BCELoss should give the same result
bce_torch = nn.BCELoss(reduction='none')(y_pred, y_true)
print(f"\nPer-sample BCE (torch.nn.BCELoss):")
print(f"  {bce_torch.tolist()}")
print(f"Mean BCE (torch):        {bce_torch.mean().item():.4f}")

# Verify they match (up to numerical precision)
print(f"\nMax difference: {(bce_manual - bce_torch).abs().max().item():.2e}")

# Note: conjunction risk 3 (high-risk, predicted 0.60) contributes more loss
# than conjunction risk 1 (high-risk, predicted 0.85) — the model is less
# confident about a true positive, so it is penalized more.
```

---

## KL divergence: the extra surprise from being wrong

The extra surprise caused by using model Q instead of the true distribution P is:

\\[ \text{KL}(P \| Q) = H(P, Q) - H(P) \\]

Or expanded:

\\[ \text{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \\]

For your analyst: KL = 1.057 − 0.852 = **0.205**. The analyst experiences 0.205 extra units of surprise per alert because their model is miscalibrated.

**Decoding the expanded formula:**

**\\(\text{KL}(P \| Q)\\)**: The KL divergence from P to Q. The double bars and the order matter. \\(\text{KL}(P \| Q)\\) asks: "if reality is P, how much extra surprise does using model Q cause?"

**\\(\sum_x\\)**: Sum over all outcomes.

**\\(P(x)\\)**: Weight by the actual frequency (what really happens).

**\\(\log \frac{P(x)}{Q(x)}\\)**: The log-ratio. When \\(Q(x) = P(x)\\), this is \\(\log 1 = 0\\): no extra surprise for that outcome. When \\(Q(x) < P(x)\\), you underestimated how often x occurs, and your surprise for that outcome is higher than it should be.

**Key properties:**

- KL divergence is always ≥ 0. It equals 0 only when P and Q are identical.
- **KL divergence is asymmetric**: \\(\text{KL}(P \| Q) \neq \text{KL}(Q \| P)\\) in general. "How surprised is the analyst when reality is P and model is Q" is a different question from "how surprised is the analyst when reality is Q and model is P."

---

## Forward vs. reverse KL: mode-covering and mode-seeking

The asymmetry of KL divergence is not just a mathematical curiosity — it has profound practical consequences for how an approximating distribution \\(Q\\) behaves when fitted to a target \\(P\\).

### Forward KL: KL(P || Q) — mode-covering

\\(\text{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}\\)

This averages the log-ratio **weighted by P**. Wherever \\(P(x) > 0\\), any terms with \\(Q(x) \approx 0\\) contribute enormous positive values (since \\(\log P/Q \to \infty\\)). To minimize \\(\text{KL}(P \| Q)\\), the approximation \\(Q\\) **must cover all modes of P** — it cannot afford to assign zero probability to any region where P is significant.

The result: minimizing forward KL produces a Q that is **spread out** (over-dispersed relative to any single mode of P). If P is bimodal, Q tries to cover both modes, which may mean Q is high between the modes even where P is low. This is the "mode-covering" (or zero-avoiding) behavior.

### Reverse KL: KL(Q || P) — mode-seeking

\\(\text{KL}(Q \| P) = \sum_x Q(x) \log \frac{Q(x)}{P(x)}\\)

This averages the log-ratio **weighted by Q**. Now, wherever \\(P(x) \approx 0\\), having \\(Q(x) > 0\\) contributes large positive values (since \\(\log Q/P \to \infty\\)). To minimize \\(\text{KL}(Q \| P)\\), the approximation Q **avoids placing mass where P is small**. Q concentrates on regions where P is large — one mode at a time.

The result: minimizing reverse KL produces a Q that is **concentrated** (under-dispersed, hugging one mode of P). If P is bimodal, Q typically collapses onto whichever mode it found first. This is the "mode-seeking" (or zero-forcing) behavior.

### Why this matters for RL: how PPO uses KL

PPO (Proximal Policy Optimization) uses **forward KL** as its trust-region constraint — or equivalently, a clipped surrogate that approximates it. The constraint is:

\\[ \text{KL}(\pi_\text{old} \| \pi_\text{new}) \leq \delta \\]

With forward KL, the old policy \\(\pi_\text{old}\\) plays the role of P. The constraint penalizes any region where \\(\pi_\text{new}\\) assigns near-zero probability to actions that \\(\pi_\text{old}\\) would take. This means the new policy **must still cover all actions the old policy would consider**, preventing catastrophic collapse in any direction of the action space.

If PPO used reverse KL instead (\\(\text{KL}(\pi_\text{new} \| \pi_\text{old}) \leq \delta\\)), the new policy could freely collapse toward a single action as long as it matched \\(\pi_\text{old}\\)'s top action well. Forward KL is the right choice for policy stability because it enforces broad coverage, not just fidelity at the mode.

### Visual demonstration: fitting a bimodal distribution

```python
import torch
import torch.optim as optim
import torch.distributions as dist

torch.manual_seed(42)

# P: a bimodal distribution — a mixture of two Gaussians.
# We represent P as a discrete distribution over 200 evenly-spaced points.
x = torch.linspace(-5, 5, 200)
dx = x[1] - x[0]

# True bimodal target P
mode1 = dist.Normal(-2.0, 0.6)
mode2 = dist.Normal(2.0, 0.6)
p_unnorm = 0.5 * mode1.log_prob(x).exp() + 0.5 * mode2.log_prob(x).exp()
P = p_unnorm / (p_unnorm.sum() * dx)          # normalized density
P_probs = (P * dx).clamp(min=1e-8)            # discrete probabilities
P_probs = P_probs / P_probs.sum()              # ensure sums to 1

def kl_forward(p, q_logits):
    """KL(P || Q): minimizing this forces Q to cover all modes of P."""
    q_probs = torch.softmax(q_logits, dim=0).clamp(min=1e-8)
    return (p * (torch.log(p) - torch.log(q_probs))).sum()

def kl_reverse(p, q_logits):
    """KL(Q || P): minimizing this lets Q concentrate on one mode."""
    q_probs = torch.softmax(q_logits, dim=0).clamp(min=1e-8)
    return (q_probs * (torch.log(q_probs) - torch.log(p))).sum()

# --- Fit Q by minimizing forward KL ---
q_logits_fwd = torch.zeros(200, requires_grad=True)
opt_fwd = optim.Adam([q_logits_fwd], lr=0.05)
for step in range(800):
    opt_fwd.zero_grad()
    loss = kl_forward(P_probs, q_logits_fwd)
    loss.backward()
    opt_fwd.step()

q_fwd = torch.softmax(q_logits_fwd.detach(), dim=0)
fwd_mean = (q_fwd * x).sum().item()
fwd_std  = ((q_fwd * (x - fwd_mean)**2).sum()**0.5).item()
print(f"Forward KL minimization:")
print(f"  Final KL(P||Q):  {kl_forward(P_probs, q_logits_fwd.detach()).item():.4f}")
print(f"  Q mean: {fwd_mean:.2f}, Q std: {fwd_std:.2f}")
print(f"  Q is spread between modes (mode-covering): std should be ~2")

# --- Fit Q by minimizing reverse KL ---
q_logits_rev = torch.zeros(200, requires_grad=True)
opt_rev = optim.Adam([q_logits_rev], lr=0.05)
for step in range(800):
    opt_rev.zero_grad()
    loss = kl_reverse(P_probs, q_logits_rev)
    loss.backward()
    opt_rev.step()

q_rev = torch.softmax(q_logits_rev.detach(), dim=0)
rev_mean = (q_rev * x).sum().item()
rev_std  = ((q_rev * (x - rev_mean)**2).sum()**0.5).item()
print(f"\nReverse KL minimization:")
print(f"  Final KL(Q||P):  {kl_reverse(P_probs, q_logits_rev.detach()).item():.4f}")
print(f"  Q mean: {rev_mean:.2f}, Q std: {rev_std:.2f}")
print(f"  Q collapsed onto one mode (mode-seeking): std should be ~0.6")

# SSA interpretation:
# If P represents the space of plausible satellite maneuver policies,
# forward KL forces our learned Q to consider all plausible maneuvers
# (important for safety: we do not want the policy to rule out a
# safety-critical action just because it is infrequent).
# Reverse KL would let our policy collapse to the single most common
# maneuver type, ignoring rarer but necessary maneuvers.
print(f"\nSSA interpretation:")
print(f"  Forward KL → Q covers both modes → {abs(fwd_mean):.2f} from center (should be ~1-2)")
print(f"  Reverse KL → Q collapses to one mode → {abs(rev_mean):.2f} from center (should be ~2)")
```

---

## Code

```python
import torch
from torch.distributions import Categorical

# True alert distribution P
P_probs = torch.tensor([0.70, 0.20, 0.08, 0.02])
P = Categorical(probs=P_probs)

# Analyst's model Q
Q_probs = torch.tensor([0.40, 0.30, 0.20, 0.10])
Q = Categorical(probs=Q_probs)

# Entropy of P (average surprise under the true distribution)
# Using the exact formula: -sum(p * log(p))
entropy_P = -(P_probs * torch.log(P_probs)).sum()
print(f"Entropy of P:         {entropy_P.item():.4f}")  # 0.852

# PyTorch also computes it directly
print(f"Entropy via PyTorch:  {P.entropy().item():.4f}")  # same answer

# Cross-entropy: average surprise analyst experiences
# Formula: -sum(P(x) * log(Q(x)))
cross_entropy = -(P_probs * torch.log(Q_probs)).sum()
print(f"Cross-entropy H(P,Q): {cross_entropy.item():.4f}")  # 1.057

# KL divergence: the extra surprise from the wrong model
kl_PQ = torch.distributions.kl_divergence(P, Q)
print(f"KL(P || Q):           {kl_PQ.item():.4f}")  # 0.205

# Verify: cross-entropy - entropy = KL
print(f"Verification:         {(cross_entropy - entropy_P).item():.4f}")  # 0.205

# KL is asymmetric: Q || P is different from P || Q
kl_QP = torch.distributions.kl_divergence(Q, P)
print(f"KL(Q || P):           {kl_QP.item():.4f}")  # different number
```

---

## Entropy of a sensor allocation policy

Let us look at entropy through an SSA operations lens: your sensor allocation policy over five candidate target objects.

```python
import torch
from torch.distributions import Categorical

# Policy A: uniform allocation (maximum uncertainty about which target gets sensor time)
policy_A = Categorical(probs=torch.ones(5) / 5)

# Policy B: focused primarily on target 1 (satellite in highest-risk conjunction)
policy_B = Categorical(probs=torch.tensor([0.80, 0.05, 0.05, 0.05, 0.05]))

# Policy C: deterministic (always sensor target 1)
policy_C = Categorical(probs=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

print(f"H(uniform policy A):        {policy_A.entropy().item():.4f}")  # 1.6094 = log(5)
print(f"H(focused policy B):        {policy_B.entropy().item():.4f}")  # lower
print(f"H(deterministic policy C):  {policy_C.entropy().item():.4f}")  # 0.0
```

Policy A has the maximum possible entropy for five targets: you have no idea which target will get sensor time, and you are equally uncertain about all of them. Policy C has zero entropy: you always know exactly which target will be observed. Policy B is in between.

In RL, an agent that has learned a near-deterministic policy has low entropy: it reliably takes the action it has determined is best. An agent still in early exploration has high entropy: its policy is spread across many actions.

---

## Why KL divergence appears in policy gradient methods

When training a neural network policy, you want to update the policy to improve expected reward. But if you take a very large gradient step, the new policy might be dramatically different from the old one, to the point where your reward estimates (which were based on the old policy) are no longer valid.

PPO and TRPO solve this by adding a constraint: the new policy should not diverge too far from the old policy, as measured by KL divergence. Specifically, they constrain:

\\[ \text{KL}(\pi_\text{old} \| \pi_\text{new}) \leq \delta \\]

where \\(\delta\\) is a small threshold (like 0.01). This says: after the update, the average extra surprise under the old policy's expectations should not exceed \\(\delta\\). This keeps updates stable and prevents the policy from collapsing after a lucky or unlucky batch of experience.

The choice of **forward KL** (old || new) rather than reverse KL (new || old) is deliberate. Forward KL forces the new policy to still cover all actions the old policy would take — preventing catastrophic forgetting of any action direction. Reverse KL would allow the new policy to collapse to a single action as long as that action matched the old policy's mode. In a satellite collision avoidance context, you never want to entirely rule out a class of avoidance maneuvers just because they were infrequent in the last training batch.

Now that you know what KL divergence measures, this constraint makes intuitive sense: "update the policy, but not so much that the new policy would surprise the old policy."

---

## Common pitfalls

**Pitfall 1: Confusing bits and nats.** If your entropy computation gives values around 1.0 for a fair coin instead of 0.693, you are accidentally using log base 2. PyTorch's `torch.log` is natural log; `torch.log2` is base 2. Pick one and be consistent.

**Pitfall 2: KL is not a distance.** KL divergence is not symmetric and does not satisfy the triangle inequality, so it is not a metric in the mathematical sense. "KL from P to Q" and "KL from Q to P" are different quantities measuring different things. Do not interchange them.

**Pitfall 3: Cross-entropy is not KL divergence.** Cross-entropy includes the entropy of P. When P is fixed (as in supervised learning), minimizing cross-entropy and minimizing KL divergence are equivalent — but the numerical values are different, and it matters when comparing across different tasks.

**Pitfall 4: log(0) is undefined.** If any probability is exactly 0, the entropy computation \\(0 \cdot \log(0)\\) requires the convention \\(0 \log 0 = 0\\) (by continuity). PyTorch's Categorical handles this, but manual computations with `torch.log` on zero-probability tensors will give `-inf`. Add a small epsilon or use `torch.nan_to_num`.

**Pitfall 5: Using reverse KL where forward KL is appropriate.** For policy constraints and trust regions, forward KL \\(\text{KL}(\pi_\text{old} \| \pi_\text{new})\\) is almost always correct. Reverse KL allows mode-seeking collapse, which is usually bad for policy stability.

---

## Key Takeaways

- **Surprise** is \\(-\log p\\): rare events are surprising, common events are not. The choice of log base determines the unit — log base 2 gives bits (information theory), natural log gives nats (ML). They differ by a factor of \\(\log_2(e) \approx 1.443\\). PyTorch uses nats.
- **Entropy** \\(H(P) = \mathbb{E}_P[-\log P(X)]\\) is average surprise: high entropy means a spread-out, uncertain distribution; zero entropy means a deterministic one. The **maximum entropy principle** says to use the highest-entropy distribution consistent with your known constraints — for known mean inter-arrival rate, that is the Exponential distribution.
- **Binary cross-entropy** \\(-[y \log \hat{p} + (1-y)\log(1-\hat{p})]\\) is the natural loss for binary classifiers (like a conjunction-risk predictor). It is cross-entropy between the degenerate true label distribution and the model's prediction, penalizing the model in proportion to how surprised it would be by the correct answer.
- **KL divergence** \\(\text{KL}(P \| Q) = H(P, Q) - H(P)\\) measures extra surprise from using the wrong model. It is always ≥ 0, asymmetric, and not a metric. Minimizing \\(\text{KL}(P \| Q)\\) over Q is equivalent to minimizing cross-entropy H(P, Q) when P is fixed — which is exactly what supervised learning does.
- **Forward KL** \\(\text{KL}(P \| Q)\\) produces mode-covering behavior (Q must cover all modes of P); **reverse KL** \\(\text{KL}(Q \| P)\\) produces mode-seeking behavior (Q collapses onto one mode). PPO uses forward KL for its trust-region constraint, ensuring the new policy cannot completely abandon any action direction the old policy used.
- All three quantities — entropy, cross-entropy, KL — are faces of the same idea: **measuring information under mismatched models**. Every place they appear in ML (training loss, exploration bonus, policy constraint) is an instance of that one idea applied to a specific problem.

---

{{#quiz 04-entropy-and-kl-divergence.toml}}
