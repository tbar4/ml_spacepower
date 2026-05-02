# Lesson 4: Entropy, cross-entropy, and KL divergence

## Where this fits

Three quantities, all built from logarithms of probabilities, all measuring something subtly different about distributions. They show up everywhere downstream. Policy gradient methods add an entropy bonus to encourage exploration. PPO and TRPO constrain the KL divergence between old and new policies to keep updates safe. Cross-entropy is the loss function for almost every classification network you'll encounter. CFR's regret matching is built around a quantity (the negative KL of a uniform distribution against the action probabilities) that mathematically reduces to entropy. If you ever skim an RL or game-theory paper and see "entropy regularization" or "KL constraint," you're seeing this lesson.

## Concept

**Entropy** measures how uncertain a distribution is. A uniform distribution over 4 actions has high entropy: you have no idea what the agent will do. A distribution that puts probability 0.99 on one action has low entropy: you basically know what's coming. A distribution that puts probability 1.0 on one action has zero entropy: you know with certainty.

That's it. Entropy is a single number. Bigger means more uncertain.

**Cross-entropy** between two distributions \\(P\\) and \\(Q\\) measures how surprised you'd be, on average, if you used \\(Q\\) to predict outcomes that actually came from \\(P\\). When \\(Q = P\\), cross-entropy equals entropy (you're not surprised by anything more than is structurally inevitable). When \\(Q\\) is very different from \\(P\\), cross-entropy is large.

**KL divergence** between \\(P\\) and \\(Q\\) is exactly the "extra surprise" from using \\(Q\\) instead of \\(P\\). It's cross-entropy minus entropy:

\\[ \text{KL}(P \| Q) = H(P, Q) - H(P) \\]

KL is zero when \\(P = Q\\) (no extra surprise from using the right distribution). It's positive otherwise. It is asymmetric: \\(\text{KL}(P \| Q) \neq \text{KL}(Q \| P)\\) in general, and this matters more than it looks.

## Why logarithms?

Both entropy and KL involve \\(-\log p\\) or \\(\log(p/q)\\). The reason is one of those design choices that turns out to be exactly right.

The "surprise" of an outcome with probability \\(p\\) should have two properties: rare events should be more surprising than common events, and the surprise of independent events should add. The function that does both is \\(-\log p\\). Probability 0.5: surprise of \\(\log 2 \approx 0.69\\). Probability 0.01: surprise of \\(\log 100 \approx 4.6\\). Probability 1.0: surprise of \\(\log 1 = 0\\) (no surprise). And if two independent events have probability \\(p_1\\) and \\(p_2\\), the surprise of both happening is \\(-\log(p_1 p_2) = -\log p_1 - \log p_2\\), which is just the sum of the individual surprises. The log makes this work.

Entropy is then the average surprise: \\(\mathbb{E}[-\log p(X)]\\). Cross-entropy is the average surprise that a wrong predictor experiences. KL is the gap between them.

## The math

For discrete distributions over a set of outcomes:

**Entropy** of \\(P\\):

\\[ H(P) = -\sum_x P(x) \log P(x) \\]

**Cross-entropy** of \\(P\\) relative to \\(Q\\):

\\[ H(P, Q) = -\sum_x P(x) \log Q(x) \\]

**KL divergence** from \\(P\\) to \\(Q\\):

\\[ \text{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P) \\]

Decoding the symbols:

- \\(P(x)\\) is the probability of outcome \\(x\\) under distribution \\(P\\). Same for \\(Q(x)\\).
- \\(\log\\) here is usually natural log in machine learning (base \\(e\\)). Some textbooks use \\(\log_2\\) (which gives "bits" instead of "nats"). The choice is a units choice; it doesn't change the math.
- \\(\sum_x\\) loops over every possible outcome.
- The minus sign in entropy and cross-entropy is there because \\(\log P(x) \leq 0\\) for probabilities (since \\(P(x) \leq 1\\)), and we want a positive number out.

A sanity check: if \\(P(x) = 1\\) for some specific outcome (the distribution is a sure thing), then \\(\log 1 = 0\\) and the entropy is 0. No uncertainty.

## A warning: KL is asymmetric

\\(\text{KL}(P \| Q)\\) is the extra surprise from using \\(Q\\) when reality is \\(P\\). \\(\text{KL}(Q \| P)\\) is the extra surprise from using \\(P\\) when reality is \\(Q\\). These are different numbers, and they have different practical implications.

In particular, \\(\text{KL}(P \| Q)\\) blows up when \\(P\\) puts probability on outcomes that \\(Q\\) thinks are impossible (small \\(Q(x)\\) makes \\(\log P(x)/Q(x)\\) huge). So minimizing \\(\text{KL}(P \| Q)\\) over choices of \\(Q\\) makes \\(Q\\) "cover" everything \\(P\\) does (mode-covering). Minimizing \\(\text{KL}(Q \| P)\\) over choices of \\(Q\\) makes \\(Q\\) avoid placing probability where \\(P\\) doesn't (mode-seeking). Same letters, different rearrangement, opposite behavior. This bites people often enough to be worth flagging now even though we won't lean on it heavily until later.

## Code

PyTorch has all three built in.

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Two policies over 4 actions.
P = Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]))  # uniform
Q = Categorical(probs=torch.tensor([0.70, 0.10, 0.10, 0.10]))  # peaky

# Entropy: how uncertain is each distribution?
print(f"H(P) = {P.entropy().item():.4f}")  # ~1.3863 (= log 4, max entropy for 4 outcomes)
print(f"H(Q) = {Q.entropy().item():.4f}")  # ~0.9405 (lower; more concentrated)

# KL divergence: how different are they?
kl_PQ = torch.distributions.kl_divergence(P, Q)
kl_QP = torch.distributions.kl_divergence(Q, P)
print(f"KL(P || Q) = {kl_PQ.item():.4f}")  # 0.4463
print(f"KL(Q || P) = {kl_QP.item():.4f}")  # 0.4458
```

These two KL values are close to each other but not equal. (For very different distributions, they can differ by orders of magnitude.) Notice also that the entropy of \\(P\\) is exactly \\(\log 4 \approx 1.386\\): the uniform distribution over 4 outcomes is the maximum-entropy distribution over 4 outcomes. This is a useful fact to know.

For cross-entropy, the most common form you'll see is the one used as a loss for classification. Given a one-hot true distribution and predicted probabilities, the cross-entropy collapses to \\(-\log Q(\text{correct class})\\):

```python
# True class is index 2; predicted distribution is Q.
predicted_log_probs = torch.log(torch.tensor([0.1, 0.2, 0.5, 0.2]))
true_class = torch.tensor([2])
loss = F.nll_loss(predicted_log_probs.unsqueeze(0), true_class)
print(loss.item())  # 0.6931 = -log(0.5)
```

`F.nll_loss` is "negative log likelihood loss," which is the same thing as cross-entropy when the true distribution is one-hot.

## Worked example: entropy of two sensor allocation policies

Suppose you're allocating sensor time across 5 candidate targets.

**Policy A (uniform)**: 0.20 to each. Maximum entropy.
**Policy B (focused)**: 0.80 to target 1, 0.05 to each of the others. Low entropy.

By hand for Policy A:

\\[ H(A) = -5 \cdot (0.2 \cdot \log 0.2) = -\log 0.2 = \log 5 \approx 1.6094 \\]

For Policy B:

\\[ H(B) = -(0.8 \log 0.8 + 4 \cdot 0.05 \log 0.05) \\]

\\[ = -(0.8 \cdot (-0.2231) + 4 \cdot 0.05 \cdot (-2.9957)) \\]

\\[ = -(-0.1785 - 0.5991) = 0.7776 \\]

So policy A has entropy ~1.61 and policy B has entropy ~0.78. Policy A "spreads" its uncertainty maximally; policy B is much more committed.

```python
import torch
from torch.distributions import Categorical

A = Categorical(probs=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
B = Categorical(probs=torch.tensor([0.8, 0.05, 0.05, 0.05, 0.05]))
print(f"H(A) = {A.entropy().item():.4f}")  # 1.6094
print(f"H(B) = {B.entropy().item():.4f}")  # 0.7776

print(f"KL(A || B) = {torch.distributions.kl_divergence(A, B).item():.4f}")
print(f"KL(B || A) = {torch.distributions.kl_divergence(B, A).item():.4f}")
```

The KL values give you a measure of how "different" these allocation policies are. In an RL setting, if you're constraining policy updates to small KL changes, you're saying "don't let the new policy disagree too much with the old one in expectation." That's all PPO and TRPO are doing in their inner loops.

## Why this matters going forward

Three concrete payoffs.

In policy gradient methods, you'll see an "entropy bonus" added to the loss: subtract \\(\beta \cdot H(\pi)\\) from your minimization objective and the policy stays more random, which encourages exploration. The intuition is just "don't let the policy get too sharp too fast."

In PPO and TRPO, you'll see a "KL constraint" or "KL penalty" between the old and new policies. The intuition is "don't let one update step move the policy too far in distribution space." This prevents the catastrophic policy collapses that early policy gradient methods suffered from.

In classification (which we'll touch briefly in the next module to motivate neural network training), the cross-entropy loss \\(-\log Q(\text{true class})\\) is what gradient descent is minimizing. Every time you've seen `nn.CrossEntropyLoss` in PyTorch tutorials, this is what was happening.

## Quiz

{{#quiz 04-entropy-and-kl-divergence.toml}}
