# Lesson 3: Sampling and Monte Carlo estimation

## Where this fits

This is the lesson where the algorithms in your future actually start making sense. MCTS is Monte Carlo. The first M in MCCFR (Monte Carlo Counterfactual Regret Minimization) is the same Monte Carlo. The REINFORCE policy gradient estimator is Monte Carlo. The conjunction-probability project at the end of this module is Monte Carlo. When the next module talks about "stochastic gradient descent," the "stochastic" part is also Monte Carlo. The entire reason this curriculum spends a lesson on sampling is that ninety percent of what we're going to build is, structurally, "estimate an expectation by drawing samples and averaging."

## Concept

In lesson 1 we computed expectations by enumerating outcomes and weighting by probability. That's fine when there are six outcomes (a die) or three outcomes (a mission status). It is hopeless when there are \\(10^{50}\\) possible game trajectories or when the distribution is over a continuous space (orbital states).

Monte Carlo estimation rescues you. The idea is two sentences long:

1. Draw \\(N\\) samples from your distribution.
2. Compute the average of whatever quantity you care about over those samples.

That average is a noisy but unbiased estimate of the true expectation. As \\(N\\) grows, the estimate gets closer to the true value (this is the Law of Large Numbers, and it is the only thing letting any of this work).

The deep reason this is useful: for distributions where direct computation is intractable, sampling is often easy. You don't need to know the analytic form of an integral if you can just draw from the distribution and observe what happens.

The price you pay is noise. Your estimate is correct on average, but any particular run gives a slightly wrong answer. The amount of wrongness shrinks as you take more samples, but it shrinks slowly.

## The math

For a function \\(f\\) of a random variable \\(X\\) drawn from distribution \\(p\\), the true expectation is:

\\[ \mathbb{E}_{X \sim p}[f(X)] \\]

(The notation \\(X \sim p\\) reads as "\\(X\\) drawn from distribution \\(p\\).")

The Monte Carlo estimate, given \\(N\\) samples \\(x_1, \ldots, x_N\\) drawn from \\(p\\), is:

\\[ \hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(x_i) \\]

The hat on \\(\hat{\mu}\\) is universal notation for "estimate of." Decoding the rest: draw \\(N\\) samples, apply \\(f\\) to each, average.

Two facts about this estimator that matter:

**Unbiased**: \\(\mathbb{E}[\hat{\mu}] = \mathbb{E}_{X \sim p}[f(X)]\\). On average across many runs of the estimator, you get the right answer. No systematic error.

**Standard error scales as \\(1/\sqrt{N}\\)**: if your samples have variance \\(\sigma^2\\), the standard error of the estimator is \\(\sigma / \sqrt{N}\\). To halve your error, you need 4x more samples. To get 10x more accuracy, you need 100x more samples.

That \\(1/\sqrt{N}\\) is the central tradeoff of Monte Carlo. It's why MCTS does many rollouts. It's why MCCFR struggles with very deep games. And it's why "just sample more" is sometimes a fine answer and sometimes hopelessly impractical.

## Code

A canonical worked example: estimate \\(\pi\\) by sampling points uniformly in the unit square and counting how many fall inside the unit quarter-circle.

```python
import torch

N = 100_000
points = torch.rand(N, 2)              # N points uniform in [0,1] x [0,1]
distances_squared = (points ** 2).sum(dim=1)
inside_circle = distances_squared <= 1
pi_estimate = 4 * inside_circle.float().mean()
print(pi_estimate.item())              # something close to 3.14159...
```

The logic: the quarter-circle has area \\(\pi/4\\), the unit square has area 1, so the fraction of points inside the quarter-circle estimates \\(\pi/4\\). Multiply by 4 to get \\(\pi\\).

If you run this with `N = 100`, you'll get something like 3.0 or 3.2, randomly off. With `N = 100_000`, you'll get something like 3.140 to 3.144. With `N = 100_000_000`, you'll get something like 3.14156 to 3.14163. The number of correct digits scales with the logarithm of \\(N\\), which is the cruel meaning of \\(1/\sqrt{N}\\) in disguise.

A more curriculum-relevant example: estimating expected return.

```python
import torch
from torch.distributions import Categorical

# A tiny "policy" over 4 actions, each with a known reward (deterministic for now).
action_probs = torch.tensor([0.10, 0.20, 0.30, 0.40])
rewards      = torch.tensor([5.0, 1.0, -2.0, 3.0])
policy = Categorical(probs=action_probs)

# Sample N actions, look up the reward for each, average.
N = 10_000
sampled_actions = policy.sample(sample_shape=(N,))
sampled_rewards = rewards[sampled_actions]
mc_estimate = sampled_rewards.mean()

# Compare to the analytic answer.
analytic = (action_probs * rewards).sum()
print(f"MC estimate: {mc_estimate.item():.4f}")
print(f"Analytic:    {analytic.item():.4f}")
```

These will agree to about 2 or 3 digits with `N = 10_000`. They wouldn't agree to 6 digits unless you cranked `N` way up.

This is, structurally, exactly what REINFORCE does. The only differences are: (a) the rewards aren't constants, they come from running the environment; and (b) we'll multiply each reward by a gradient term. The averaging step is the same.

## Worked example: convergence rate

Let's actually watch the \\(1/\sqrt{N}\\) curve. Estimate \\(\pi\\) for \\(N = 10, 100, 1\\,000, 10\\,000, 100\\,000\\), repeating 10 times for each \\(N\\) so we can see the spread.

```python
import torch

def estimate_pi(N):
    points = torch.rand(N, 2)
    inside = (points ** 2).sum(dim=1) <= 1
    return 4 * inside.float().mean().item()

for N in [10, 100, 1_000, 10_000, 100_000]:
    runs = [estimate_pi(N) for _ in range(10)]
    runs_t = torch.tensor(runs)
    print(f"N={N:>6}: mean={runs_t.mean():.4f}, std={runs_t.std():.4f}")
```

You'll see something like:

```
N=    10: mean=3.1200, std=0.5103
N=   100: mean=3.1400, std=0.1714
N=  1000: mean=3.1380, std=0.0510
N= 10000: mean=3.1407, std=0.0166
N=100000: mean=3.1414, std=0.0049
```

Two things to notice. First, the mean across runs is very close to \\(\pi\\) at every \\(N\\); the estimator is unbiased. Second, the spread of individual runs (the std column) drops by roughly a factor of \\(\sqrt{10} \approx 3.16\\) each time \\(N\\) increases tenfold. That is the \\(1/\sqrt{N}\\) scaling, in the wild.

## Variance reduction

This is a brief preview, not something to memorize now. Because Monte Carlo's error scales as \\(\sigma / \sqrt{N}\\), you have two ways to make your estimate better: take more samples, or shrink \\(\sigma\\). Most clever Monte Carlo techniques (importance sampling, control variates, antithetic sampling) are tricks for shrinking \\(\sigma\\) without changing what you're estimating. We'll meet this idea again in MCCFR (which uses outcome sampling and external sampling to control variance) and in policy gradient methods (which use baselines for the same reason). Right now, just file away that "sample more" and "sample smarter" are both real options.

## Why this matters going forward

The phrase "Monte Carlo" in MCTS, MCCFR, and elsewhere means exactly what it means here. Each rollout in MCTS is a sample. Each trajectory in MCCFR is a sample. Each episode in REINFORCE is a sample. The estimator's job is always the same: average over samples to estimate an expectation. The clever part of each algorithm is what they do with the estimate, but the estimate itself is the thing you just wrote.

When you encounter a paper that talks about "high variance estimators" being a problem, that's the \\(\sigma / \sqrt{N}\\) thing biting. When a paper says "we use a baseline to reduce variance," they're shrinking \\(\sigma\\). You now have the conceptual hooks for both.

## Quiz

{{#quiz 03-sampling-and-monte-carlo.toml}}
