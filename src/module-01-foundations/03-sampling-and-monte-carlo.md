# Lesson 3: Sampling and Monte Carlo Estimation

## Where this fits

In lesson 1 you learned that expectation is a weighted sum over all possible outcomes. That works perfectly when there are three object types or six dice faces. It becomes completely impossible when there are millions of possible game trajectories, or when the quantity you want to average does not have a tidy formula. Monte Carlo estimation is how you compute expectations when direct computation is hopeless. MCTS, MCCFR, and the REINFORCE policy gradient estimator are all, at their core, Monte Carlo estimation with extra structure on top. This lesson is where those algorithms get their conceptual foundation.

## The problem: some expectations cannot be computed directly

Consider a simple version of an SSA planning problem. You are deciding whether to maneuver a satellite now or wait. The outcome depends on:

- Whether an approaching RSO turns out to be debris or an active satellite (two possibilities)
- What orbital regime it settles into after the next atmospheric drag update (many possibilities)
- What other operators in the region decide to do (unknown)
- Small stochastic perturbations from solar radiation pressure (continuous, infinite possibilities)

The exact expected cost of maneuvering versus not maneuvering involves averaging over all combinations of these factors. If each factor has only 10 possible values, you have 10 × 10 × 10 × 10 = 10,000 combinations to sum over. With 20 factors, you have 10^20 combinations, which is more than the number of atoms in a gram of carbon. With continuous values, it is literally infinite.

In game theory the situation is similar. A chess game has roughly 10^120 possible game states. Computing the exact expected value of a move by summing over all of them is not possible in any practical sense.

**Monte Carlo estimation** is the answer: instead of summing over all possibilities, you draw a random sample, compute the quantity for that sample, and repeat. The average of many samples is a reliable estimate of the true expectation.

## The core idea: sampling and averaging

Here is the key insight, stated plainly:

**If you cannot enumerate all outcomes, simulate some of them and average the results.**

The average of your simulated outcomes will be close to the true expectation. The more samples you take, the closer it will be.

Let us see this concretely with a simple SSA-flavored example before we look at any formulas.

**Scenario**: Your satellite is about to pass through a debris field. You estimate there is a 30% chance of a collision with a piece of debris large enough to cause damage. If there is a collision, the mission cost is 1,000 (arbitrary units). If there is no collision, the cost is 0.

The expected cost is straightforward here: 0.30 × 1,000 + 0.70 × 0 = 300. You can compute it directly.

But suppose you did not know how to compute it directly, and instead you simulated 10 passes through the debris field:

- Pass 1: no collision (cost 0)
- Pass 2: no collision (cost 0)
- Pass 3: **collision** (cost 1,000)
- Pass 4: no collision (cost 0)
- Pass 5: no collision (cost 0)
- Pass 6: no collision (cost 0)
- Pass 7: **collision** (cost 1,000)
- Pass 8: no collision (cost 0)
- Pass 9: no collision (cost 0)
- Pass 10: no collision (cost 0)

Average cost: (0 + 0 + 1000 + 0 + 0 + 0 + 1000 + 0 + 0 + 0) / 10 = 2000 / 10 = **200**.

That is not exactly 300, but it is in the right ballpark. With 10 samples, you got 2 collisions instead of the "expected" 3. With 1,000 samples you would typically get something much closer to 300.

## The formula for Monte Carlo estimation

Suppose you want to estimate \\(\mathbb{E}[f(X)]\\), the expected value of a function \\(f\\) applied to a random variable \\(X\\).

Instead of computing the infinite (or intractable) sum, you:

1. Draw \\(N\\) samples \\(x_1, x_2, \ldots, x_N\\) from the distribution of \\(X\\)
2. Compute \\(f(x_i)\\) for each sample
3. Average the results

The Monte Carlo estimate is written:

\\[ \hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(x_i) \\]

**Decoding the symbols:**

**\\(\hat{\mu}\\)**: Read as "mu hat." The Greek letter mu (\\(\mu\\)) is conventional notation for a mean or expected value. The hat (\\(\hat{}\\)) means "estimated." So \\(\hat{\mu}\\) is "our estimate of the true mean." The hat distinguishes the estimate (which we computed from samples) from the true value (which we might never know exactly).

**\\(\frac{1}{N}\\)**: Divide by \\(N\\), the number of samples. This is just computing an average.

**\\(\sum_{i=1}^{N}\\)**: Add up the following thing for i from 1 to N. Same summation sign as in lesson 1, now looping over samples rather than outcomes.

**\\(f(x_i)\\)**: Apply function \\(f\\) to the i-th sample. In the debris example, f(x) = the cost of that pass (1,000 if collision, 0 if not).

**Reading it in English**: "Draw N samples, compute f for each one, add them all up, divide by N to get the average." That is the entire thing.

## Two properties that make this useful

**Property 1: The estimate is unbiased.**

"Unbiased" means that if you ran your Monte Carlo estimator many times (each time drawing a fresh set of N samples), the average of your estimates would equal the true expectation. There is no systematic error in one direction or the other. Individual runs might be too high or too low, but they are wrong in a random, symmetric way.

**Property 2: The error shrinks as \\(1/\sqrt{N}\\).**

The standard error (a measure of how wrong a typical estimate is) follows this formula:

\\[ \text{standard error} = \frac{\sigma}{\sqrt{N}} \\]

Where \\(\sigma\\) (sigma, the Greek lowercase letter for standard deviation) describes how spread out your samples are, and \\(N\\) is the number of samples.

**Decoding**: the standard error gets smaller as N gets bigger. But it shrinks by a square root factor. To get twice as accurate, you need four times as many samples. To get ten times as accurate, you need one hundred times as many samples.

Let us see what that looks like numerically for the debris example:

| Samples (N) | Typical error in Pc estimate |
|-------------|------------------------------|
| 10          | ±0.145 (huge)                |
| 100         | ±0.046                       |
| 1,000       | ±0.014                       |
| 10,000      | ±0.005 (about ±0.5%)         |
| 100,000     | ±0.0014 (about ±0.1%)        |

At 10 samples, your estimate of a 30% probability might range from 15% to 45%. At 100,000 samples, it is accurate to within a tenth of a percent. The cost of that accuracy is 10,000 times more computation.

This 1/√N trade-off is fundamental. It is why MCTS does many rollouts to improve its value estimates. It is why MCCFR accumulates regret over many iterations rather than converging in a handful. Samples are cheap compared to exact computation, but they are not free.

## Watching the convergence happen

Here is a Monte Carlo estimation of a simple orbital probability: the fraction of a 95-minute low Earth orbit that a satellite spends in eclipse (behind Earth's shadow). We do not need to derive the analytic answer; we just simulate orbital positions and count how many are in shadow.

We will use an extremely simplified model: a circular orbit at 400 km altitude. A position is "in eclipse" if the angle from the Sun direction is more than about 90 + a few degrees (accounting for Earth's radius). We will approximate this with a random position on a circle.

```python
import torch

def estimate_eclipse_fraction(N):
    """Estimate fraction of orbit spent in eclipse via Monte Carlo."""
    # Sample N random positions along a circular orbit (uniform angles).
    angles = torch.rand(N) * 2 * torch.pi  # uniform in [0, 2*pi]
    
    # An extremely simplified eclipse model: in eclipse if angle
    # from sun direction (0 radians) is between 110 and 250 degrees.
    # (This is a rough approximation; real eclipse geometry is more complex.)
    angle_deg = torch.rad2deg(angles)
    in_eclipse = (angle_deg >= 110) & (angle_deg <= 250)
    
    return in_eclipse.float().mean().item()

# True fraction for this simplified model: (250 - 110) / 360 ≈ 0.389
true_fraction = (250 - 110) / 360
print(f"True fraction (simplified model): {true_fraction:.4f}")
print()

# Watch convergence with increasing N
for N in [10, 100, 1_000, 10_000, 100_000]:
    # Run 5 times to see the spread
    runs = [estimate_eclipse_fraction(N) for _ in range(5)]
    runs_t = torch.tensor(runs)
    mean = runs_t.mean().item()
    std  = runs_t.std().item()
    error = abs(mean - true_fraction)
    print(f"N={N:>6}: mean={mean:.4f}, std={std:.4f}, error={error:.4f}")
```

When you run this, you will see the error and standard deviation shrink roughly by a factor of 3 each time N increases by a factor of 10 (because √10 ≈ 3.16). That is the 1/√N convergence, made visible.

## The canonical Monte Carlo example: estimating pi

This example appears in virtually every introduction to Monte Carlo methods because it makes the sampling process visually obvious.

Imagine a unit square (width 1, height 1) with a quarter-circle of radius 1 inscribed in it. The area of the square is 1. The area of the quarter-circle is π/4. So the fraction of random points that fall inside the quarter-circle is π/4, and we can estimate π by multiplying that fraction by 4.

```python
import torch

torch.manual_seed(42)  # makes results reproducible

def estimate_pi(N):
    # Draw N random points uniformly in the unit square [0,1] x [0,1].
    points = torch.rand(N, 2)
    
    # A point (x, y) is inside the quarter-circle if x^2 + y^2 <= 1.
    # points**2 squares each coordinate. .sum(dim=1) sums x^2 + y^2 per point.
    distance_squared = (points ** 2).sum(dim=1)
    inside = distance_squared <= 1.0
    
    # Fraction inside * 4 estimates pi.
    return 4 * inside.float().mean().item()

print(f"True pi:  {torch.pi:.6f}")
print()
for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
    estimate = estimate_pi(N)
    error = abs(estimate - torch.pi.item())
    print(f"N={N:>7}: estimate={estimate:.5f}, error={error:.5f}")
```

With N = 100, you might get 3.08 or 3.24, off by a noticeable amount. With N = 1,000,000, you will reliably get something like 3.14163, accurate to five decimal places.

Notice that going from 100 samples to 1,000,000 samples is a factor of 10,000 increase in computation, but the accuracy only improved from roughly ±0.05 to roughly ±0.002, a factor of 25. That is the 1/√N scaling at work. Getting two more decimal places of pi costs 10,000 times more samples.

## How this connects to game-playing algorithms

When MCTS evaluates a game position, it cannot sum over all possible continuations of the game (there are too many). Instead, it runs random **rollouts**: simulations of a complete game from that position to the end, following some approximate policy. The fraction of those rollouts that end in a win is the Monte Carlo estimate of the winning probability from that position.

Each rollout is one sample. The winning probability estimate improves as more rollouts are run. The estimate is noisy with few rollouts and reliable with many. That is exactly the convergence behavior you just saw with pi and eclipse fraction.

The first M in MCCFR (Monte Carlo Counterfactual Regret Minimization) refers to the same idea. Instead of computing counterfactual regret over all possible game trajectories, MCCFR samples trajectories and estimates the regret from those samples. It converges to the correct solution as the number of samples grows, at the 1/√N rate.

The REINFORCE policy gradient algorithm computes the gradient of expected return by running episodes of the agent's policy and averaging the resulting gradients. Each episode is a sample. The gradient estimate improves with more episodes, again at 1/√N.

The pattern is universal: whenever "exact computation is intractable," the answer involves some variant of "sample and average."

## Variance reduction: getting more accuracy without more samples

Because standard error = σ/√N, there are two ways to get a more accurate estimate:

1. Take more samples (increase N)
2. Reduce how spread out the samples are (decrease σ)

Techniques that reduce σ without changing what you are estimating are called **variance reduction** methods. You will meet several of them later:

- **Baselines in policy gradient methods**: subtract a fixed value from each reward before computing the gradient estimate. Does not change the expected gradient but reduces how much it varies from estimate to estimate.
- **Outcome sampling in MCCFR**: choose which game trajectories to sample based on their importance rather than uniformly at random.
- **Importance sampling**: sample from a different distribution that has lower variance, then correct for the bias.

You do not need to understand these yet. Just file away that "variance reduction" means "same answer, less noise per sample," and it is an active area of research precisely because 1/√N is expensive.

## Quiz

{{#quiz 03-sampling-and-monte-carlo.toml}}
