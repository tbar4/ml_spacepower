# Lesson 4: Entropy, Cross-Entropy, and KL Divergence

## Where this fits

Three quantities, all measuring something about probability distributions. They show up in specific and important places downstream. Policy gradient methods add entropy bonuses to encourage exploration. PPO and TRPO constrain policy updates using KL divergence. Cross-entropy is the training loss for nearly every classification network. If you have ever seen a training log print "cross-entropy loss = 0.43" or a paper say "we constrain the KL between old and new policy," this lesson is where those terms become concrete.

The good news: all three quantities reduce to one idea, which we will build up from an SSA scenario.

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

### Maximum and minimum entropy

**Maximum entropy** occurs when the distribution is uniform: all outcomes equally likely. For four outcomes, maximum entropy would be −4 × (0.25 × log 0.25) = log 4 ≈ 1.386. A uniform policy over actions is maximally uncertain; you have no idea which action the agent will take.

**Minimum entropy** (zero) occurs when one outcome has probability 1 and all others have probability 0. A completely determined distribution. A deterministic policy has zero entropy; you know exactly which action it will take.

Your alert distribution (entropy ≈ 0.852) is between these extremes, closer to the minimum. You are not completely surprised on average, because most alerts are routine.

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
- The asymmetry matters in ML. Minimizing \\(\text{KL}(P \| Q)\\) over choices of Q produces a different result than minimizing \\(\text{KL}(Q \| P)\\). The first makes Q try to cover all regions where P is large. The second makes Q try to concentrate wherever P is large.

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

## Why KL divergence appears in policy gradient methods

When training a neural network policy, you want to update the policy to improve expected reward. But if you take a very large gradient step, the new policy might be dramatically different from the old one, to the point where your reward estimates (which were based on the old policy) are no longer valid.

PPO and TRPO solve this by adding a constraint: the new policy should not diverge too far from the old policy, as measured by KL divergence. Specifically, they constrain:

\\[ \text{KL}(\pi_\text{old} \| \pi_\text{new}) \leq \delta \\]

where \\(\delta\\) is a small threshold (like 0.01). This says: after the update, the average extra surprise under the old policy's expectations should not exceed \\(\delta\\). This keeps updates stable and prevents the policy from collapsing after a lucky or unlucky batch of experience.

Now that you know what KL divergence measures, this constraint makes intuitive sense: "update the policy, but not so much that the new policy would surprise the old policy."

## Quiz

{{#quiz 04-entropy-and-kl-divergence.toml}}
