# Lesson 1: Probability, Distributions, and Expectation

## Where this fits

Every algorithm in this curriculum, from MCTS rollouts to CFR regret minimization to policy gradient training, is answering one core question: given that I am uncertain about the world, what should I expect? That question has a precise mathematical answer, and this lesson is that answer. Once you understand expectation, you understand the conceptual shape of half the algorithms in OpenSpiel. The rest is detail.

## A space scenario to motivate everything

Imagine you are working a conjunction assessment shift at a space operations center. Your ground radar just detected a new Resident Space Object (RSO) in low Earth orbit. Based on the radar cross-section measurement and preliminary orbital elements, your analyst has assigned probabilities to what this object might be:

| Object type       | Probability |
|-------------------|-------------|
| Active satellite  | 0.80        |
| Debris            | 0.15        |
| Dead satellite    | 0.05        |
| **Total**         | **1.00**    |

You cannot wait for perfect information. You need to decide right now how much sensor time to allocate to continued tracking, which operators to notify, and how urgently to treat this object. Different object types require different responses. This situation, having to reason and act when you do not know the truth for certain, is exactly the problem probability is designed for.

## What is a random variable?

A **random variable** is a number (or category) whose value you do not know yet, but where you know something about what values it could take.

The object type in your scenario is a random variable. It could be active satellite, debris, or dead satellite. Right now, before you have more sensor data, you are uncertain which it is. The "random" part just means you are uncertain. The "variable" part means it is a slot waiting to be filled with a value.

You will constantly encounter random variables in this curriculum:

- The action an adversary's satellite takes during a conjunction (uncertain because you cannot read their intentions)
- The reward an RL agent receives after making a move (uncertain because the environment is stochastic)
- The object type of a newly detected RSO (uncertain because your sensor is imperfect)
- The exact position of a satellite given imperfect tracking data (uncertain because measurement errors exist)

They are all the same idea. A quantity that depends on something you have not fully observed yet.

## What is a distribution?

A **distribution** is the complete description of your uncertainty. It lists every possible value the random variable could take, and the probability of each one.

Your RSO analysis produced a distribution:

| Object type       | Probability |
|-------------------|-------------|
| Active satellite  | 0.80        |
| Debris            | 0.15        |
| Dead satellite    | 0.05        |

Notice the probabilities sum to 1.00. This is always required. The distribution has to account for all possibilities. One of these outcomes will happen (or already has happened, you just do not know which yet). The probabilities just describe how likely each one is.

### Two distributions you will see constantly

**Categorical distribution**: a distribution over a finite list of named categories. The object-type example above is categorical. In reinforcement learning, your policy is a categorical distribution over actions: "there are four possible moves, with these probabilities."

**Gaussian (Normal) distribution**: a distribution over all real numbers, shaped like a bell curve. The position of a satellite you are tracking is often modeled as Gaussian: you have a best estimate of where it is, and uncertainty spreads symmetrically around that estimate. A satellite with position uncertainty of 0.2 km is much more precisely tracked than one with uncertainty of 5 km, even if both have the same best estimate.

## What is expectation? Building it from arithmetic

Now suppose each object type has an associated sensor priority score:

| Object type       | Probability | Priority score |
|-------------------|-------------|----------------|
| Active satellite  | 0.80        | 30             |
| Debris            | 0.15        | 90             |
| Dead satellite    | 0.05        | 80             |

**Question**: what is the average priority score for this object, given your uncertainty about its type?

Here is how to think about it without any formulas yet. Suppose you processed 1,000 similar radar detections using this same probability model. Based on those probabilities, you would expect:

- About 800 to be active satellites (priority score 30)
- About 150 to be debris (priority score 90)
- About 50 to be dead satellites (priority score 80)

To find the average priority score across all 1,000 detections:

```
Total priority points from active satellites: 800 × 30 = 24,000
Total priority points from debris:            150 × 90 = 13,500
Total priority points from dead satellites:    50 × 80 =  4,000
                                              ─────────────────
Total priority points:                                   41,500

Average = 41,500 / 1,000 = 41.5
```

Now look at what those 800, 150, and 50 are. Divide each by 1,000 and you get 0.80, 0.15, and 0.05. Those are the probabilities. So the exact same arithmetic can be written more directly:

```
(0.80 × 30) + (0.15 × 90) + (0.05 × 80)
= 24.0 + 13.5 + 4.0
= 41.5
```

That is expectation. **Multiply each value by its probability, then add up the products.** The result is the probability-weighted average, called the expected value or expectation.

A few things to notice:
- The expected value (41.5) is not one of the possible values (30, 90, 80). That is fine. Expectation is a property of the distribution, not a prediction of any single outcome.
- If you actually processed that radar contact, you would see a priority score of 30, 90, or 80, nothing else. The 41.5 is what you should plan around on average, before you know which one you got.
- If the debris probability were much higher (say, 0.95), the expected priority would be much higher too. The expected value follows the probability mass.

## The formula, built from the arithmetic you just did

Here is the same calculation written compactly. Let us use symbols to represent the quantities:

- Let \\(n\\) be the number of possible outcomes (in our case, 3)
- Let \\(p_i\\) be the probability of outcome \\(i\\)
- Let \\(x_i\\) be the value (priority score) for outcome \\(i\\)

The expected value is written:

\\[ \mathbb{E}[X] = \sum_{i=1}^{n} p_i \cdot x_i \\]

**Decoding every symbol, one at a time:**

**\\(\mathbb{E}[X]\\)**: Read this as "the expected value of X" or just "E of X." The double-struck capital E is a conventional notation for "take the expectation of." X is the random variable (our priority score). The brackets just mean we are asking for the expectation of that particular thing.

**\\(\sum_{i=1}^{n}\\)**: This is the capital Greek letter sigma, used here as a summation sign. Read it as "add up the following thing for every i, starting at i = 1 and ending at i = n." It is literally a for loop:

```python
total = 0
for i in range(1, n + 1):
    total += p_i * x_i
```

**\\(p_i\\)**: The probability of outcome i. The subscript i (written below and slightly to the right of p) connects this probability to the i-th outcome. When i = 1, this is the probability of outcome 1 (active satellite, 0.80). When i = 2, it is the probability of outcome 2 (debris, 0.15). And so on.

**\\(x_i\\)**: The value (priority score) for outcome i. Same subscript convention: x with subscript 1 is the priority score when outcome 1 occurs (30), x with subscript 2 is the score when outcome 2 occurs (90), and so on.

**\\(p_i \cdot x_i\\)**: The dot means multiplication. This is "probability of outcome i times value of outcome i."

**Reading the whole formula in English**: "For each possible outcome (from i = 1 to i = n), multiply its probability by its value. Add up all those products. That total is the expected value."

That is the calculation you already did by hand.

### Expectation of a function

One more version you will see often. Instead of taking the expectation of a raw value, sometimes you take the expectation of a function applied to the outcome:

\\[ \mathbb{E}[f(X)] = \sum_{i=1}^{n} p_i \cdot f(x_i) \\]

Here \\(f(x_i)\\) means "apply the function f to outcome i, then use that result." For example, if \\(f(x) = x^2\\), then \\(f(x_i)\\) is the priority score squared.

In RL, \\(f\\) is usually "the total reward you collect starting from this state." In CFR, \\(f\\) is "the regret from taking this action." The structure is always the same: for each outcome, compute f of that outcome, weight by probability, sum up.

## Variance: how spread out is the distribution?

Expectation gives you the average. But two distributions can have the same average while behaving very differently.

**Scenario A**: You always track active satellites, every single contact. Priority score is always 30. Expected priority: 30. Variance: zero. Your planning is perfectly predictable.

**Scenario B**: 50% chance of a priority-10 object, 50% chance of a priority-50 object. Expected priority: (0.5 × 10) + (0.5 × 50) = 30. Same average, but your actual experience swings between 10 and 50.

**Variance** measures the average squared distance from the expected value. "Squared distance" means you take the difference between an outcome and the expected value, then square it.

For Scenario B:
- Outcome 1 is priority 10. Distance from expected (30) is 10 - 30 = -20. Squared: 400.
- Outcome 2 is priority 50. Distance from expected (30) is 50 - 30 = +20. Squared: 400.
- Expected squared distance: (0.5 × 400) + (0.5 × 400) = 400.

So variance is 400. The square root of variance is the **standard deviation**: √400 = 20. A typical sample lands about 20 priority points away from the mean. In Scenario A, standard deviation is zero: you always land exactly on the mean.

Variance will come back in lesson 3 when it determines how noisy your Monte Carlo estimates are. High variance means you need more samples to get a reliable estimate.

## Code

```python
import torch
from torch.distributions import Categorical

# Our RSO probability estimate.
probs = torch.tensor([0.80, 0.15, 0.05])
dist = Categorical(probs=probs)

# Sample from the distribution: returns 0 (active sat), 1 (debris), or 2 (dead sat).
sample = dist.sample()
print(f"Sampled object type index: {sample.item()}")

# Sample many times to see the frequencies.
many_samples = dist.sample(sample_shape=(10_000,))
for i, label in enumerate(["Active sat", "Debris", "Dead sat"]):
    freq = (many_samples == i).float().mean()
    print(f"  {label}: {freq:.3f}  (expected: {probs[i]:.3f})")
```

Computing expected priority directly:

```python
import torch

probs           = torch.tensor([0.80, 0.15, 0.05])
priority_scores = torch.tensor([30.0, 90.0, 80.0])

# E[priority] = sum of (p_i * x_i).
# Step 1: multiply each probability by its priority score.
products = probs * priority_scores
print(f"Products:        {products.tolist()}")  # [24.0, 13.5, 4.0]

# Step 2: sum the products.
expected = products.sum()
print(f"Expected priority: {expected.item()}")  # 41.5
```

Notice how the Python arithmetic directly mirrors the formula. `probs * priority_scores` is the elementwise multiplication of all the \\(p_i \cdot x_i\\) terms. `.sum()` is the \\(\sum\\) symbol.

## Worked example: dwell time planning across two RSOs

You are planning sensor dwell time for two simultaneously tracked RSOs on an upcoming pass. Each object type requires different dwell times:

| Object type       | Dwell time needed (seconds) |
|-------------------|-----------------------------|
| Active satellite  | 5                           |
| Debris            | 15                          |
| Dead satellite    | 10                          |

Your current probability estimates:

| Object type       | RSO Alpha | RSO Beta |
|-------------------|-----------|----------|
| Active satellite  | 0.70      | 0.10     |
| Debris            | 0.20      | 0.80     |
| Dead satellite    | 0.10      | 0.10     |

**Expected dwell for RSO Alpha:**

Step 1, for each object type, multiply probability by dwell time:
- Active satellite: 0.70 × 5 = 3.50 seconds
- Debris: 0.20 × 15 = 3.00 seconds
- Dead satellite: 0.10 × 10 = 1.00 second

Step 2, sum:
- 3.50 + 3.00 + 1.00 = **7.5 seconds expected dwell**

**Expected dwell for RSO Beta:**

Step 1:
- Active satellite: 0.10 × 5 = 0.50 seconds
- Debris: 0.80 × 15 = 12.00 seconds
- Dead satellite: 0.10 × 10 = 1.00 second

Step 2:
- 0.50 + 12.00 + 1.00 = **13.5 seconds expected dwell**

**Total expected dwell**: 7.5 + 13.5 = **21 seconds** for this pass.

If your radar has 30 seconds of dwell capacity, you are comfortable. If it has 15 seconds, you have a prioritization problem to solve. Expectation gives you the planning number.

```python
import torch

dwell_times = torch.tensor([5.0, 15.0, 10.0])
alpha_probs = torch.tensor([0.70, 0.20, 0.10])
beta_probs  = torch.tensor([0.10, 0.80, 0.10])

alpha_dwell = (alpha_probs * dwell_times).sum()
beta_dwell  = (beta_probs  * dwell_times).sum()

print(f"Alpha expected dwell: {alpha_dwell.item():.1f}s")  # 7.5s
print(f"Beta expected dwell:  {beta_dwell.item():.1f}s")   # 13.5s
print(f"Total:                {(alpha_dwell + beta_dwell).item():.1f}s")  # 21.0s
```

## Quiz

{{#quiz 01-probability-distributions-expectation.toml}}
