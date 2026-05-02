# Lesson 1: Probability, distributions, and expectation

## Where this fits

Almost everything you'll read in this curriculum is, at its core, a sentence about expected value. Reinforcement learning maximizes expected return. CFR minimizes expected regret. MCTS rollouts estimate the expected value of a game node. Policy gradient methods compute expected gradients. The word "expected" in all those phrases is the same word and it means the same thing, and once you've internalized what it means, you've cleared the conceptual hurdle for half of OpenSpiel's source code.

This module is about tools. This lesson is about the tool we'll use most.

## Concept

A **random variable** is a number whose value isn't known until something happens. Roll a die: the result is a random variable. Decide an attacker's first action in a satellite-jamming scenario: their action is a random variable (assuming they're not deterministic). Sample from a neural network's policy output: the action is a random variable.

A **distribution** is the rule that says how often each possible value of a random variable shows up. For a fair six-sided die, the distribution is "each of {1, 2, 3, 4, 5, 6} with probability 1/6." For a biased coin that lands heads 70% of the time, the distribution is "{heads: 0.7, tails: 0.3}." Distributions can be over discrete sets (like dice rolls) or continuous ranges (like a satellite's exact position, which is some real number with infinite possibilities).

The **expectation** of a random variable is its long-run average. If you rolled a fair die a million times and averaged the results, you'd get something extremely close to 3.5. That's because 3.5 is the expectation of a fair die roll, even though no actual roll ever produces 3.5. Expectation is a summary statistic, not a prediction of any single outcome.

That's the entire conceptual content of this lesson. Everything below is making the formula match the intuition.

## The math

For a discrete random variable \\(X\\) taking values \\(x_1, x_2, \ldots, x_n\\) with probabilities \\(p_1, p_2, \ldots, p_n\\), the expectation is:

\\[ \mathbb{E}[X] = \sum_{i=1}^{n} p_i \cdot x_i \\]

Decoding the symbols:

- \\(\mathbb{E}[X]\\) is "the expected value of \\(X\\)." Think of \\(\mathbb{E}\\) as a verb: "average."
- \\(\sum_{i=1}^{n}\\) is "add up the following thing for \\(i\\) going from 1 to \\(n\\)." It's a `for` loop with an accumulator.
- \\(p_i \cdot x_i\\) is "this outcome's value times this outcome's probability."

So the whole expression reads as: "loop over every possible outcome, multiply each value by how likely it is, sum the results." That is just a weighted average.

For a function of a random variable, \\(\mathbb{E}[f(X)]\\), the formula has the same shape:

\\[ \mathbb{E}[f(X)] = \sum_{i=1}^{n} p_i \cdot f(x_i) \\]

This is the form we actually care about. In RL, \\(f\\) is "return I get if I play action \\(X\\)." In CFR, \\(f\\) is "regret from this action." We almost never care about the expectation of a raw random variable; we care about the expectation of some function of it.

For continuous distributions, the sum becomes an integral and you'll see it written as \\(\int p(x) f(x) dx\\). We won't need this much, because sampling will rescue us from integrals (lesson 3).

**Variance** measures how spread out a distribution is around its mean. The formal definition is \\(\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]\\), but the intuition is: "if I sample from this distribution a bunch, how far do my samples typically land from the average?" High-variance distributions are unpredictable; low-variance distributions cluster tightly around their mean. We'll come back to variance in lesson 3, where it's going to bite us.

## Code

PyTorch ships a `distributions` module that's exactly the right level of abstraction. You define a distribution, then sample from it or compute probabilities of specific outcomes.

```python
import torch
from torch.distributions import Bernoulli, Categorical, Normal

# A biased coin (Bernoulli distribution).
# `probs` is the probability of "1" (heads).
coin = Bernoulli(probs=torch.tensor(0.7))
print(coin.sample(sample_shape=(10,)))  # 10 coin flips, each 0 or 1

# A categorical distribution: pick one of N options with given probabilities.
# This is the natural shape for action distributions in RL.
action_probs = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 4 possible actions
policy = Categorical(probs=action_probs)
print(policy.sample(sample_shape=(5,)))  # 5 sampled actions, each 0..3

# A Gaussian distribution: parameterized by mean and standard deviation.
# This is what we'd use for continuous quantities like sensor noise.
position = Normal(loc=0.0, scale=1.0)  # mean 0, std 1
print(position.sample(sample_shape=(5,)))
```

Two API gotchas worth flagging now. PyTorch calls the mean of a Normal `loc` and the standard deviation `scale`. And `Categorical`'s `probs` argument doesn't strictly have to sum to 1 (PyTorch will normalize it for you), but it's good hygiene to make it sum to 1 yourself so that bugs are visible.

To compute an expectation analytically (without sampling), you just transliterate the formula:

```python
import torch

# Expected value of a fair die roll.
values = torch.arange(1, 7).float()  # tensor([1., 2., 3., 4., 5., 6.])
probs = torch.ones(6) / 6              # uniform
expected_value = (probs * values).sum()
print(expected_value.item())  # 3.5
```

That is literally `sum(p_i * x_i)`.

## Worked example: expected mission value

Suppose a small satellite has three possible operational outcomes for a given week:

| Outcome   | Probability | Value (data units) |
|-----------|-------------|--------------------|
| Nominal   | 0.80        | 100                |
| Degraded  | 0.15        | 30                 |
| Lost      | 0.05        | 0                  |

What's the expected weekly value?

By hand:

\\[ \mathbb{E}[V] = 0.80 \cdot 100 + 0.15 \cdot 30 + 0.05 \cdot 0 = 80 + 4.5 + 0 = 84.5 \\]

In code:

```python
import torch

probs  = torch.tensor([0.80, 0.15, 0.05])
values = torch.tensor([100.0, 30.0, 0.0])

expected_value = (probs * values).sum()
print(expected_value.item())  # 84.5
```

Now suppose we want \\(\mathbb{E}[V^2]\\) (this will be useful in a moment). Same shape, with \\(f(x) = x^2\\):

```python
expected_value_squared = (probs * values**2).sum()
print(expected_value_squared.item())  # 8135.0
```

Variance falls out as \\(\mathbb{E}[V^2] - (\mathbb{E}[V])^2 = 8135 - 84.5^2 \approx 994\\). Standard deviation is \\(\sqrt{994} \approx 31.5\\) data units. So our weekly value is "about 84.5, plus or minus 30-ish in any given week." Both numbers matter for decision-making, and both fall out of the same distribution. The expectation tells you the long-run yield; the variance tells you how much you should hate any individual bad week.

## Why this matters going forward

When you read "the agent maximizes the expected discounted return," translate it as: "the agent considers all possible futures (weighted by how likely each is) and picks the action whose probability-weighted average outcome is best." That single translation will get you through most of the RL literature. The same translation works for "expected counterfactual regret," "expected gradient," and so on. The structure is always the same: enumerate outcomes, weight by probability, sum.

The catch, which we'll deal with in lesson 3, is that we usually can't enumerate all the outcomes because there are too many of them. So we sample.

## Quiz

{{#quiz 01-probability-distributions-expectation.toml}}
