# Lesson 1: Probability, Distributions, and Expectation

**Module:** ML and Game Theory for Space Power — M01: Foundations
**Source:** *Math for Deep Learning* — Ronald T. Kneusel, Chapters 2–3 (Probability Fundamentals); *Bayesian Statistics the Fun Way* — Will Kurt, Chapters 1–2 (Bayesian Thinking and Probability)

---


<!-- toc -->

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

You cannot wait for perfect information. You need to decide right now how much sensor time to allocate to continued tracking, which operators to notify, and how urgently to treat this object. Different object types require different responses. This situation — having to reason and act when you do not know the truth for certain — is exactly the problem probability is designed for.

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

---

## The rules of probability

Will Kurt opens *Bayesian Statistics the Fun Way* with a deceptively simple point: before you can update beliefs with evidence, you need to know the two rules that govern how probabilities combine. These rules are the grammar of the language. Everything else — Bayes' rule, belief updates, joint distributions — is written in this grammar.

### The sum rule: combining probabilities of alternatives

**When events are mutually exclusive** (at most one can happen at a time):

\[ P(A \cup B) = P(A) + P(B) \]

**Decoding:**

- \(P(A \cup B)\): "the probability that A or B occurs." The \(\cup\) symbol is the set-theory union: "either A, or B, or both."
- For mutually exclusive events, "both" is impossible, so the formula simplifies to straight addition.

**SSA example**: RSO Alpha is either an active satellite (0.80) or debris (0.15) or a dead satellite (0.05). These categories are mutually exclusive — an object cannot be two types at once. So the probability it is "active satellite or debris" is simply 0.80 + 0.15 = 0.95.

**When events are not mutually exclusive** (both can happen simultaneously):

\[ P(A \cup B) = P(A) + P(B) - P(A \cap B) \]

**Decoding:**

- \(P(A \cap B)\): the probability that both A and B occur. The \(\cap\) symbol is the intersection: "both A and B."
- You subtract \(P(A \cap B)\) because you would otherwise count the overlap twice — once when you add \(P(A)\) and once when you add \(P(B)\).

**SSA example**: You want to know the probability that *at least one* of two RSOs — Alpha or Beta — experiences a conjunction event in the next 24 hours. Say \(P(\text{Alpha conjunction}) = 0.30\) and \(P(\text{Beta conjunction}) = 0.25\). If the two RSOs are in different orbital planes and their events are independent, \(P(\text{both}) = 0.30 \times 0.25 = 0.075\). So:

\[ P(\text{Alpha or Beta}) = 0.30 + 0.25 - 0.075 = 0.475 \]

Without subtracting the overlap, you would overcount the scenarios where both have conjunctions.

### The product rule: combining probabilities of simultaneous events

For **independent** events (the outcome of one does not change the probability of the other):

\[ P(A \cap B) = P(A) \times P(B) \]

**Decoding:**

- This says: if A and B have nothing to do with each other, the probability they both occur is the product of their individual probabilities.
- Independence is a model assumption. It is often approximately right (two RSOs in different planes) and sometimes dangerously wrong (two sensors sharing the same atmospheric perturbation).

```python
import torch

# Sum rule: mutually exclusive (object type categories)
p_active = torch.tensor(0.80)
p_debris = torch.tensor(0.15)
p_active_or_debris = p_active + p_debris
print(f"P(active or debris): {p_active_or_debris.item():.2f}")  # 0.95

# Sum rule: non-mutually-exclusive (conjunction events for two RSOs)
p_alpha_conj = torch.tensor(0.30)
p_beta_conj  = torch.tensor(0.25)

# Assuming independence, compute the overlap first
p_both_conj  = p_alpha_conj * p_beta_conj  # product rule
p_either_conj = p_alpha_conj + p_beta_conj - p_both_conj
print(f"P(Alpha or Beta conjunction): {p_either_conj.item():.3f}")  # 0.475

# Verify: P(neither) = (1 - 0.30) * (1 - 0.25) = 0.525, so P(at least one) = 0.475
p_neither = (1 - p_alpha_conj) * (1 - p_beta_conj)
print(f"P(neither): {p_neither.item():.3f}")          # 0.525
print(f"Sum check:  {(p_either_conj + p_neither).item():.3f}")  # 1.000
```

---

## Joint and marginal probability

Distributions over single variables are only the beginning. In SSA, most interesting questions involve two or more variables together: "what type of object is it *and* what orbital regime is it in?" The tools for this are joint and marginal probability.

### Joint probability

**Joint probability** \(P(X = x, Y = y)\) is the probability that two random variables simultaneously take specific values. The comma means "and."

Here is a joint distribution over object type (X) and orbit regime (Y) for the objects in a hypothetical SSA catalog:

| | **LEO** | **MEO** | **GEO** | Row total |
|---|---|---|---|---|
| **Active satellite** | 0.18 | 0.09 | 0.23 | 0.50 |
| **Debris** | 0.28 | 0.06 | 0.01 | 0.35 |
| **Dead satellite** | 0.10 | 0.03 | 0.02 | 0.15 |
| **Column total** | 0.56 | 0.18 | 0.26 | 1.00 |

Each cell is a joint probability. \(P(\text{active satellite, GEO}) = 0.23\): 23% of tracked objects are active satellites in GEO. The sum of the entire table equals 1.

### Marginal probability

**Marginal probability** is what you get when you collapse (sum out) one variable to focus on the other.

The row totals give you the marginal distribution over object type: \(P(\text{active satellite}) = 0.50\), \(P(\text{debris}) = 0.35\), \(P(\text{dead satellite}) = 0.15\). These are called "marginals" because, in printed tables, they traditionally appear in the margins.

The column totals give you the marginal distribution over orbit regime: \(P(\text{LEO}) = 0.56\), \(P(\text{MEO}) = 0.18\), \(P(\text{GEO}) = 0.26\).

The relationship is:

\[ P(X = x) = \sum_{y} P(X = x, Y = y) \]

"Sum over all values of Y to get the probability that X takes value x."

```python
import torch

# Joint distribution as a 2D tensor: rows = object type, cols = orbit regime
# Rows: [active satellite, debris, dead satellite]
# Cols: [LEO, MEO, GEO]
joint = torch.tensor([
    [0.18, 0.09, 0.23],   # active satellite
    [0.28, 0.06, 0.01],   # debris
    [0.10, 0.03, 0.02],   # dead satellite
])

# Verify it sums to 1
print(f"Total probability mass: {joint.sum().item():.2f}")  # 1.00

# Marginal over object type: sum across orbit regimes (dim=1 collapses columns)
marginal_type = joint.sum(dim=1)
print(f"P(object type): {marginal_type.tolist()}")
# [0.50, 0.35, 0.15] — active sat, debris, dead sat

# Marginal over orbit regime: sum across object types (dim=0 collapses rows)
marginal_orbit = joint.sum(dim=0)
print(f"P(orbit regime): {marginal_orbit.tolist()}")
# [0.56, 0.18, 0.26] — LEO, MEO, GEO

# Conditional distribution: P(orbit | object type = debris)
# = joint[debris, :] / P(debris)
p_orbit_given_debris = joint[1, :] / marginal_type[1]
print(f"P(orbit | debris): {p_orbit_given_debris.tolist()}")
# Mostly LEO — debris concentrates in low orbits
```

You will use joint distributions heavily when you study POMDPs: the joint distribution over (hidden state, observation) is the raw material from which belief states are computed.

---

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

- Let \(n\) be the number of possible outcomes (in our case, 3)
- Let \(p_i\) be the probability of outcome \(i\)
- Let \(x_i\) be the value (priority score) for outcome \(i\)

The expected value is written:

\[ \mathbb{E}[X] = \sum_{i=1}^{n} p_i \cdot x_i \]

**Decoding every symbol, one at a time:**

**\(\mathbb{E}[X]\)**: Read this as "the expected value of X" or just "E of X." The double-struck capital E is a conventional notation for "take the expectation of." X is the random variable (our priority score). The brackets just mean we are asking for the expectation of that particular thing.

**\(\sum_{i=1}^{n}\)**: This is the capital Greek letter sigma, used here as a summation sign. Read it as "add up the following thing for every i, starting at i = 1 and ending at i = n." It is literally a for loop:

```python
total = 0
for i in range(1, n + 1):
    total += p_i * x_i
```

**\(p_i\)**: The probability of outcome i. The subscript i (written below and slightly to the right of p) connects this probability to the i-th outcome. When i = 1, this is the probability of outcome 1 (active satellite, 0.80). When i = 2, it is the probability of outcome 2 (debris, 0.15). And so on.

**\(x_i\)**: The value (priority score) for outcome i. Same subscript convention: x with subscript 1 is the priority score when outcome 1 occurs (30), x with subscript 2 is the score when outcome 2 occurs (90), and so on.

**\(p_i \cdot x_i\)**: The dot means multiplication. This is "probability of outcome i times value of outcome i."

**Reading the whole formula in English**: "For each possible outcome (from i = 1 to i = n), multiply its probability by its value. Add up all those products. That total is the expected value."

That is the calculation you already did by hand.

### Expectation of a function

One more version you will see often. Instead of taking the expectation of a raw value, sometimes you take the expectation of a function applied to the outcome:

\[ \mathbb{E}[f(X)] = \sum_{i=1}^{n} p_i \cdot f(x_i) \]

Here \(f(x_i)\) means "apply the function f to outcome i, then use that result." For example, if \(f(x) = x^2\), then \(f(x_i)\) is the priority score squared.

In RL, \(f\) is usually "the total reward you collect starting from this state." In CFR, \(f\) is "the regret from taking this action." The structure is always the same: for each outcome, compute f of that outcome, weight by probability, sum up.

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

---

## The Law of Large Numbers

The connection between probability and long-run frequency is formalized by the **Law of Large Numbers** (LLN). Kneusel emphasizes this in *Math for Deep Learning* Chapter 2 as the justification for why sample-based methods work at all.

**Formal statement**: Let \(X_1, X_2, \ldots, X_N\) be independent, identically distributed random variables, each with expected value \(\mu\). Define the sample mean:

\[ \bar{X}_N = \frac{1}{N} \sum_{i=1}^{N} X_i \]

As \(N \to \infty\), the sample mean converges to the true mean:

\[ \bar{X}_N \to \mu \]

**Decoding:**

- \(\bar{X}_N\): the average of N actual observed values. This is a number you can compute from data.
- \(\mu\): the true expected value \(\mathbb{E}[X]\). This is a property of the distribution.
- The arrow means: as you draw more samples, the sample mean gets closer and closer to the true mean.

**Why this matters for SSA**: if you run a simulation of 100 conjunction events sampled from your uncertainty distribution, the average outcome (say, expected dwell time) will be close to but not exactly equal to the true expected dwell time. If you run 100,000 simulations, it will be much closer. The LLN guarantees convergence; the convergence rate (which depends on variance) determines how many samples you actually need.

**Important contrast with Monte Carlo**: the LLN tells you that the sample mean converges. It does not tell you how fast. Lesson 3 will show that Monte Carlo estimates converge at rate \(1/\sqrt{N}\) — halving your error requires quadrupling your sample count. This slow convergence rate is both the limitation and the operational reality of simulation-based planning.

```python
import torch

# Demonstrate LLN: priority score samples converging to the true expectation
probs           = torch.tensor([0.80, 0.15, 0.05])
priority_scores = torch.tensor([30.0, 90.0, 80.0])

# True expected value
true_mean = (probs * priority_scores).sum()
print(f"True expected priority: {true_mean.item():.2f}")  # 41.50

# Draw increasingly large samples and watch the sample mean converge
from torch.distributions import Categorical
dist = Categorical(probs=probs)

for n in [10, 100, 1_000, 10_000, 100_000]:
    indices = dist.sample((n,))                    # sample n object type indices
    scores  = priority_scores[indices]             # map indices to priority scores
    sample_mean = scores.mean()
    error = (sample_mean - true_mean).abs()
    print(f"  N={n:>7}: sample mean = {sample_mean.item():.3f}, "
          f"error = {error.item():.3f}")

# Typical output (results vary):
# N=     10: sample mean = 38.000, error = 3.500
# N=    100: sample mean = 40.200, error = 1.300
# N=  1,000: sample mean = 41.620, error = 0.120
# N= 10,000: sample mean = 41.491, error = 0.009
# N=100,000: sample mean = 41.502, error = 0.002
```

The sample mean gets steadily closer to 41.50 as N grows. Note that the improvement is not linear — going from N=10 to N=100 (10×) does not give 10× the accuracy. That \(1/\sqrt{N}\) convergence rate is the reason large-scale simulations are expensive.

---

## Continuous distributions: the Gaussian in depth

The Gaussian distribution is the dominant model for continuous uncertainty in SSA. Every sensor has measurement noise; every satellite position estimate comes with an uncertainty covariance. Understanding the Gaussian's structure is not optional.

### The 68-95-99.7 rule

For a Gaussian with mean \(\mu\) and standard deviation \(\sigma\):

- **68%** of outcomes fall within \(\mu \pm 1\sigma\)
- **95%** of outcomes fall within \(\mu \pm 2\sigma\)
- **99.7%** of outcomes fall within \(\mu \pm 3\sigma\)

This rule lets you translate between "standard deviations" and "probability." If your conjunction analysis says the miss distance is Gaussian with mean 5.0 km and \(\sigma = 1.5\) km, then:

- 68% of conjunction geometries will have miss distance between 3.5 and 6.5 km
- 95% will be between 2.0 and 8.0 km
- 99.7% will be between 0.5 and 9.5 km

If the hard-body radius of the two objects sums to 0.01 km, the probability of a collision is the probability mass below that threshold — far out in the left tail. That is a numerical computation, but the 68-95-99.7 rule gives you the right intuition before you do the math.

### Satellite position uncertainty as a Gaussian covariance

In three-dimensional space, position uncertainty is not captured by a single number. A satellite tracked by two radar sites will have better cross-track than along-track precision, and better range precision than angular precision. The full description is a **covariance matrix** — a 3×3 symmetric positive-definite matrix where the diagonal entries give per-axis variance and the off-diagonal entries capture correlations between axes.

When you read the standard conjunction message format (CCSDS CDM), the position covariance is one of the first data fields. The 1-sigma ellipsoid defined by that covariance is the Gaussian uncertainty region. The probability of collision is the integral of the combined position uncertainty distribution over the combined hard-body volume — a Gaussian integral in 6D position space.

For a 1D version, the Gaussian probability density function is:

\[ f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \]

**Decoding:**

- \(\mu\): the mean (center of the bell curve). For a satellite position, this is the best-estimated position.
- \(\sigma\): the standard deviation. Larger \(\sigma\) means wider uncertainty.
- \(\exp(\cdot)\): the exponential function \(e^{(\cdot)}\). It is always positive, which keeps the density positive.
- \(\frac{1}{\sigma \sqrt{2\pi}}\): a normalizing constant that ensures the total area under the curve equals 1.

```python
import torch
from torch.distributions import Normal

# Model along-track position uncertainty for a tracked RSO
# mu: best-estimated along-track position offset from predicted (km)
# sigma: 1-sigma position uncertainty (km)
mu    = torch.tensor(0.0)    # centered on the prediction
sigma = torch.tensor(1.5)    # 1.5 km 1-sigma uncertainty

dist = Normal(loc=mu, scale=sigma)

# Demonstrate 68-95-99.7 rule by sampling
samples = dist.sample((100_000,))

within_1sigma = ((samples - mu).abs() <= sigma).float().mean()
within_2sigma = ((samples - mu).abs() <= 2 * sigma).float().mean()
within_3sigma = ((samples - mu).abs() <= 3 * sigma).float().mean()

print(f"Fraction within 1σ: {within_1sigma.item():.3f}  (expected: 0.683)")
print(f"Fraction within 2σ: {within_2sigma.item():.3f}  (expected: 0.954)")
print(f"Fraction within 3σ: {within_3sigma.item():.3f}  (expected: 0.997)")

# Compute log-probability (useful for training neural networks)
# A position measurement of 1.0 km from the predicted position:
measurement = torch.tensor(1.0)
log_prob = dist.log_prob(measurement)
print(f"\nLog-probability of 1.0 km offset: {log_prob.item():.3f}")
print(f"Probability density at 1.0 km:    {log_prob.exp().item():.4f}")

# Compare two objects with different uncertainty levels
tight_dist = Normal(0.0, 0.2)   # well-tracked object, 0.2 km 1-sigma
loose_dist = Normal(0.0, 5.0)   # poorly tracked, 5 km 1-sigma

# Probability of being within 0.1 km of predicted position (rough collision zone)
from torch.distributions import Normal
tight_prob = tight_dist.cdf(torch.tensor(0.1)) - tight_dist.cdf(torch.tensor(-0.1))
loose_prob = loose_dist.cdf(torch.tensor(0.1)) - loose_dist.cdf(torch.tensor(-0.1))
print(f"\nP(within 0.1 km): tight track = {tight_prob.item():.4f}, "
      f"loose track = {loose_prob.item():.4f}")
# The tightly tracked object has much higher probability density at any specific point
```

The `torch.distributions.Normal` class is the building block for many ML loss functions — Gaussian negative log-likelihood is the MSE loss in disguise. When you later train a neural network to predict position estimates with uncertainty, you will be maximizing exactly this log-probability.

---

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

Notice how the Python arithmetic directly mirrors the formula. `probs * priority_scores` is the elementwise multiplication of all the \(p_i \cdot x_i\) terms. `.sum()` is the \(\sum\) symbol.

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

---

## Key Takeaways

- **Probability is a complete description of uncertainty, not just a single number.** A distribution over object types, orbit regimes, or sensor readings tells you the full range of what could be true and how likely each possibility is. Every algorithm in this curriculum manipulates distributions, not point guesses.
- **The sum and product rules are the foundation.** \(P(A \cup B) = P(A) + P(B) - P(A \cap B)\) and \(P(A \cap B) = P(A) \times P(B)\) for independent events. Before you can do Bayes' rule or compute likelihoods, you need these two rules working fluently.
- **Joint distributions capture correlations between variables.** \(P(\text{object type, orbit regime})\) is more informative than either marginal alone. Marginalizing (summing over one variable) recovers the individual distributions. In SSA, ignoring joint structure means treating orbital regime and object type as independent when they may not be.
- **Expectation is a planning tool, not a prediction.** The expected priority score (41.5) is not a value you will ever observe. It is the average you should plan around when facing many decisions under the same uncertainty. RL value functions are expectations; so are the cost estimates in dwell time planning.
- **The Law of Large Numbers guarantees that sample averages converge to expectations, but slowly.** Error shrinks at rate \(1/\sqrt{N}\): to halve the error you must quadruple the sample count. This is why Monte Carlo methods require careful variance management — lesson 3 picks this up directly.
- **Gaussian uncertainty is the default model for satellite position and sensor noise.** The 68-95-99.7 rule gives you fast intuition about what any \(\sigma\) value means in practice. When you see a conjunction probability computed from a covariance matrix, it is computing the integral of a Gaussian over the collision zone — the same math as `Normal.cdf()` in PyTorch.

---

## Quiz

{{#quiz 01-probability-distributions-expectation.toml}}
