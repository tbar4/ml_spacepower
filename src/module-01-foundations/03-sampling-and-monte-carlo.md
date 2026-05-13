# Lesson 3: Sampling and Monte Carlo Estimation

**Module:** Foundations — M01: Mathematical Foundations for ML and Game Theory
**Source:** *Reinforcement Learning: An Introduction* — Sutton & Barto, Chapter 5 (Monte Carlo Methods); *Probabilistic Theory of Pattern Recognition* — Devroye, Györfi, Lugosi, Chapter 2 (Concentration Inequalities); *Monte Carlo Statistical Methods* — Robert & Casella, Chapter 3 (Monte Carlo Integration)

---


<!-- toc -->

## Where this fits

In lesson 1 you learned that expectation is a weighted sum over all possible outcomes. That works perfectly when there are three object types or six dice faces. It becomes completely impossible when there are millions of possible game trajectories, or when the quantity you want to average does not have a tidy formula. Monte Carlo estimation is how you compute expectations when direct computation is hopeless. MCTS, MCCFR, and the REINFORCE policy gradient estimator are all, at their core, Monte Carlo estimation with extra structure on top. This lesson is where those algorithms get their conceptual foundation.

---

## The problem: some expectations cannot be computed directly

Consider a simple version of an SSA planning problem. You are deciding whether to maneuver a satellite now or wait. The outcome depends on:

- Whether an approaching RSO turns out to be debris or an active satellite (two possibilities)
- What orbital regime it settles into after the next atmospheric drag update (many possibilities)
- What other operators in the region decide to do (unknown)
- Small stochastic perturbations from solar radiation pressure (continuous, infinite possibilities)

The exact expected cost of maneuvering versus not maneuvering involves averaging over all combinations of these factors. If each factor has only 10 possible values, you have 10 × 10 × 10 × 10 = 10,000 combinations to sum over. With 20 factors, you have 10^20 combinations, which is more than the number of atoms in a gram of carbon. With continuous values, it is literally infinite.

In game theory the situation is similar. A chess game has roughly 10^120 possible game states. Computing the exact expected value of a move by summing over all of them is not possible in any practical sense.

**Monte Carlo estimation** is the answer: instead of summing over all possibilities, you draw a random sample, compute the quantity for that sample, and repeat. The average of many samples is a reliable estimate of the true expectation.

---

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

---

## The formula for Monte Carlo estimation

Suppose you want to estimate \(\mathbb{E}[f(X)]\), the expected value of a function \(f\) applied to a random variable \(X\).

Instead of computing the infinite (or intractable) sum, you:

1. Draw \(N\) samples \(x_1, x_2, \ldots, x_N\) from the distribution of \(X\)
2. Compute \(f(x_i)\) for each sample
3. Average the results

The Monte Carlo estimate is written:

\[ \hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(x_i) \]

**Decoding the symbols:**

**\(\hat{\mu}\)**: Read as "mu hat." The Greek letter mu (\(\mu\)) is conventional notation for a mean or expected value. The hat (\(\hat{}\)) means "estimated." So \(\hat{\mu}\) is "our estimate of the true mean." The hat distinguishes the estimate (which we computed from samples) from the true value (which we might never know exactly).

**\(\frac{1}{N}\)**: Divide by \(N\), the number of samples. This is just computing an average.

**\(\sum_{i=1}^{N}\)**: Add up the following thing for i from 1 to N. Same summation sign as in lesson 1, now looping over samples rather than outcomes.

**\(f(x_i)\)**: Apply function \(f\) to the i-th sample. In the debris example, f(x) = the cost of that pass (1,000 if collision, 0 if not).

**Reading it in English**: "Draw N samples, compute f for each one, add them all up, divide by N to get the average." That is the entire thing.

---

## Two properties that make this useful

**Property 1: The estimate is unbiased.**

"Unbiased" means that if you ran your Monte Carlo estimator many times (each time drawing a fresh set of N samples), the average of your estimates would equal the true expectation. There is no systematic error in one direction or the other. Individual runs might be too high or too low, but they are wrong in a random, symmetric way.

**Property 2: The error shrinks as \(1/\sqrt{N}\).**

The standard error (a measure of how wrong a typical estimate is) follows this formula:

\[ \text{standard error} = \frac{\sigma}{\sqrt{N}} \]

Where \(\sigma\) (sigma, the Greek lowercase letter for standard deviation) describes how spread out your samples are, and \(N\) is the number of samples.

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

---

## The Central Limit Theorem: why uncertainty quantification works

The 1/√N trade-off tells us the error shrinks. But what is the *shape* of that error? Could estimates be wildly skewed in one direction? The Central Limit Theorem (CLT) answers this — and the answer is what makes MC estimation not just useful, but scientifically rigorous.

**The CLT, stated plainly:**

Take N independent random variables \(X_1, X_2, \ldots, X_N\), all drawn from the same distribution with mean \(\mu\) and variance \(\sigma^2\). Their average

\[ \bar{X}_N = \frac{1}{N} \sum_{i=1}^N X_i \]

is itself a random variable. As N grows, the distribution of \(\bar{X}_N\) approaches a Normal distribution centered at \(\mu\) with standard deviation \(\sigma/\sqrt{N}\), regardless of what the original distribution looks like.

**Decoding:**

**\(\bar{X}_N\)**: The sample mean of N draws — our Monte Carlo estimate \(\hat{\mu}\). It is random because different draws produce different averages.

**"Approaches a Normal distribution"**: The bell curve. Even if each individual sample comes from a wildly non-Normal distribution (like a Bernoulli, which only takes values 0 or 1), the average of many such samples looks Normal.

**"Regardless of the original distribution shape"**: This is the remarkable part. It does not matter whether you are averaging collision indicators (Bernoulli), damage costs (heavy-tailed), or orbital period perturbations (roughly Normal). The sample mean is always approximately Normal for large N.

**Why this validates MC estimation.** Our MC estimate \(\hat{\mu}\) is the sample mean. The CLT tells us that \(\hat{\mu}\) is approximately Normally distributed around the true mean \(\mu\). This means we can write:

\[ \hat{\mu} \approx \mathcal{N}\!\left(\mu,\; \frac{\sigma^2}{N}\right) \]

And because Normal distributions are well-understood, we can compute confidence intervals directly:

\[ \mu \in \left[\hat{\mu} - 1.96\frac{\hat{\sigma}}{\sqrt{N}},\;\; \hat{\mu} + 1.96\frac{\hat{\sigma}}{\sqrt{N}}\right] \quad \text{with 95\% confidence} \]

where \(\hat{\sigma}\) is the sample standard deviation, estimated from the same N samples. You do not need to know the true \(\sigma\) in advance.

**Practical consequence**: after running your MC estimator, you can quote not just the estimate but its uncertainty. In SSA terms: "our MC estimate of conjunction probability is 0.0043, with a 95% CI of [0.0038, 0.0048] based on 10,000 simulations." That kind of statement is only possible because of the CLT.

```python
import torch

torch.manual_seed(0)

# Demonstrate the CLT: sample means from a Bernoulli(0.3) distribution
# are approximately Normal, regardless of the binary shape of individual samples.

p_true = 0.30          # true collision probability
N_per_estimate = 500   # samples per MC estimate
n_estimates = 10_000   # how many estimates to draw

# Each row: one MC experiment of N_per_estimate coin flips
samples = torch.bernoulli(torch.full((n_estimates, N_per_estimate), p_true))

# Each MC estimate is the mean of one row
sample_means = samples.mean(dim=1)   # shape: (10_000,)

# Expected distribution: Normal(mu=0.30, sigma=sqrt(p*(1-p)/N))
theoretical_std = (p_true * (1 - p_true) / N_per_estimate) ** 0.5

print(f"True mean:                {p_true:.4f}")
print(f"Mean of MC estimates:     {sample_means.mean().item():.4f}")  # ≈ 0.30
print(f"Std of MC estimates:      {sample_means.std().item():.6f}")   # ≈ theoretical_std
print(f"Theoretical std (CLT):    {theoretical_std:.6f}")

# Verify normality: check that 95% of estimates fall within 1.96 std of the mean
z_scores = (sample_means - p_true) / theoretical_std
within_95 = ((z_scores.abs() <= 1.96).float().mean().item())
print(f"Fraction within 1.96σ:    {within_95:.4f}")  # should be ≈ 0.95

# Compute a confidence interval for one MC run
single_run = samples[0]
estimate = single_run.mean().item()
std_est = single_run.std().item()
ci_lo = estimate - 1.96 * std_est / N_per_estimate ** 0.5
ci_hi = estimate + 1.96 * std_est / N_per_estimate ** 0.5
print(f"\nSingle run estimate:      {estimate:.4f}")
print(f"95% CI:                   [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"True value inside CI:     {ci_lo <= p_true <= ci_hi}")
```

The Bernoulli distribution could not look less Normal — it only ever takes values 0 or 1. Yet when you average 500 of them, the distribution of those averages is essentially a bell curve. That is the CLT in action.

The same demonstration in Rust. Cargo dependencies for every Rust example in this lesson (versions chosen to match the Rust Playground's catalog so the mdbook "play" button works):

```toml
[dependencies]
ndarray = "0.17"
rand = "0.10"
rand_distr = "0.6"
```

A wrinkle if you copy the code into a fresh playground tab outside mdbook: the playground has these crates built and ready, but they aren't pre-declared in `Cargo.toml`, so `use ndarray::...` on its own won't resolve. You need a stray `extern crate ndarray;` (and the same for `rand`, `rand_distr`) at the top of the file. mdbook's `#` line-prefix trick hides those declarations from the rendered page but still includes them when the "play" button ships the code to the playground, which is what the leading hidden lines in each block below are doing.

```rust
# extern crate ndarray;
# extern crate rand;
use ndarray::{Array1, Array2, Axis};
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let p_true = 0.30_f64;
    let n_per_estimate = 500;
    let n_estimates = 10_000;

    // Each row: one MC experiment of n_per_estimate Bernoulli(p_true) flips.
    // 1.0 = collision, 0.0 = no collision. rand_distr has a Bernoulli type,
    // but rolling it by hand here keeps the example self-contained.
    let samples = Array2::<f64>::from_shape_fn(
        (n_estimates, n_per_estimate),
        |_| if rng.random::<f64>() < p_true { 1.0 } else { 0.0 },
    );

    // Each MC estimate is the mean of one row.
    let sample_means: Array1<f64> = samples.mean_axis(Axis(1)).unwrap();

    let theoretical_std = (p_true * (1.0 - p_true) / n_per_estimate as f64).sqrt();

    // ndarray's .std(ddof) takes the degrees-of-freedom correction. Pass 1.0
    // to match PyTorch and NumPy (Bessel's correction). 0.0 gives population std.
    let mc_mean = sample_means.mean().unwrap();
    let mc_std  = sample_means.std(1.0);
    println!("True mean:                {p_true:.4}");
    println!("Mean of MC estimates:     {mc_mean:.4}");
    println!("Std of MC estimates:      {mc_std:.6}");
    println!("Theoretical std (CLT):    {theoretical_std:.6}");

    let within_95 = sample_means
        .iter()
        .filter(|&&m| ((m - p_true) / theoretical_std).abs() <= 1.96)
        .count() as f64
        / n_estimates as f64;
    println!("Fraction within 1.96σ:    {within_95:.4}");

    // 95% confidence interval from a single MC run.
    let single_run = samples.row(0);
    let estimate = single_run.mean().unwrap();
    let std_est  = single_run.std(1.0);
    let ci_lo = estimate - 1.96 * std_est / (n_per_estimate as f64).sqrt();
    let ci_hi = estimate + 1.96 * std_est / (n_per_estimate as f64).sqrt();
    println!("\nSingle run estimate:      {estimate:.4}");
    println!("95% CI:                   [{ci_lo:.4}, {ci_hi:.4}]");
    println!("True value inside CI:     {}", ci_lo <= p_true && p_true <= ci_hi);
}
```

One thing worth flagging about the imports: `rand::Rng` is the base RNG trait (formerly `RngCore`), and `rand::RngExt` is the extension trait that gives you `.random()`, `.random_range()`, and friends. You import both: `Rng` to use as a trait bound in function signatures, `RngExt` to call the convenience methods. This is a recent rearrangement (rand 0.10); older code you'll find online uses `rng.gen()` directly.

---

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

And in Rust:

```rust
# extern crate ndarray;
# extern crate rand;
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::StdRng;
use std::f64::consts::PI;

fn estimate_eclipse_fraction(n: usize, rng: &mut impl Rng) -> f64 {
    // Sample n random positions along a circular orbit (uniform angles in [0, 2π)).
    let angles = Array1::<f64>::from_shape_fn(n, |_| rng.random::<f64>() * 2.0 * PI);
    let angle_deg = angles.mapv(|a| a.to_degrees());
    let in_eclipse = angle_deg.mapv(|d| if (110.0..=250.0).contains(&d) { 1.0 } else { 0.0 });
    in_eclipse.mean().unwrap()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let true_fraction = (250.0_f64 - 110.0) / 360.0;
    println!("True fraction (simplified model): {true_fraction:.4}");
    println!();

    for n in [10, 100, 1_000, 10_000, 100_000] {
        // Run 5 times to see the spread.
        let runs: Vec<f64> = (0..5).map(|_| estimate_eclipse_fraction(n, &mut rng)).collect();
        let runs = Array1::from(runs);
        let mean  = runs.mean().unwrap();
        let std   = runs.std(1.0);
        let error = (mean - true_fraction).abs();
        println!("N={n:>6}: mean={mean:.4}, std={std:.4}, error={error:.4}");
    }
}
```

The pattern `from_shape_fn(n, |_| ...)` is the workhorse for building random arrays without pulling in `ndarray-rand`. You'll see it again in every subsequent example.

---

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

```rust
# extern crate ndarray;
# extern crate rand;
use ndarray::{Array2, Axis};
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::StdRng;
use std::f64::consts::PI;

fn estimate_pi(n: usize, rng: &mut impl Rng) -> f64 {
    // n points uniform in [0,1] x [0,1]. Each call to rng draws one coordinate.
    let points = Array2::<f64>::from_shape_fn((n, 2), |_| rng.random::<f64>());
    let distance_squared = points.mapv(|x| x * x).sum_axis(Axis(1));
    let inside = distance_squared.mapv(|d| if d <= 1.0 { 1.0 } else { 0.0 });
    4.0 * inside.mean().unwrap()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    println!("True pi:  {PI:.6}");
    println!();
    for n in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let estimate = estimate_pi(n, &mut rng);
        let error = (estimate - PI).abs();
        println!("N={n:>7}: estimate={estimate:.5}, error={error:.5}");
    }
}
```

`sum_axis(Axis(1))` is the direct equivalent of PyTorch's `sum(dim=1)`. The `Axis(1)` newtype is ndarray being explicit about which dimension you mean, so you can't accidentally collapse the wrong one. You'll see this everywhere from here on.

---

## Variance reduction: getting more accuracy without more samples

Because standard error = σ/√N, there are two ways to get a more accurate estimate:

1. Take more samples (increase N)
2. Reduce how spread out the samples are (decrease σ)

Techniques that reduce σ without changing what you are estimating are called **variance reduction** methods. You will meet several of them later:

- **Baselines in policy gradient methods**: subtract a fixed value from each reward before computing the gradient estimate. Does not change the expected gradient but reduces how much it varies from estimate to estimate.
- **Outcome sampling in MCCFR**: choose which game trajectories to sample based on their importance rather than uniformly at random.
- **Importance sampling**: sample from a different distribution that has lower variance, then correct for the bias.

You do not need to understand these yet. Just file away that "variance reduction" means "same answer, less noise per sample," and it is an active area of research precisely because 1/√N is expensive.

---

## Importance sampling: sampling where it matters

Variance reduction techniques all ask the same question: can we gather information more efficiently than uniform random sampling? Importance sampling is the most conceptually fundamental answer.

### The problem: rare events under uniform sampling

Suppose you want to estimate the probability of a close-approach conjunction event between two satellites where the miss distance is below 100 meters — a rare but catastrophically dangerous scenario. Under a realistic orbital uncertainty distribution, such close approaches might occur with probability 1 in 10,000. To estimate this probability to 10% relative accuracy, you need on the order of 1,000,000 samples. That is expensive.

The problem is structural: you are drawing most of your samples from the vast region of orbital state space where nothing interesting happens, and only a tiny fraction of samples land in the dangerous region you actually care about.

### The intuition: sample more where the function is large

If the integrand (the thing you are averaging) is concentrated in a small region, draw more samples from that region and then correct for having over-sampled it. The correction is a weight that accounts for how much more often you sampled each point than you would have under the original distribution.

### The formula

Suppose you want to estimate \(\mathbb{E}_p[f(X)] = \int f(x)\, p(x)\, dx\), but sampling from \(p\) is inefficient because \(f(x)\) is large only in a small region.

Choose a proposal distribution \(q\) that assigns higher probability to the region where \(f(x)\) is large. Then:

\[ \mathbb{E}_p[f(X)] = \int f(x)\, p(x)\, dx = \int f(x)\, \frac{p(x)}{q(x)}\, q(x)\, dx = \mathbb{E}_q\!\left[f(X)\, \frac{p(x)}{q(x)}\right] \]

The importance sampling estimator is:

\[ \hat{\mu}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^{N} f(x_i) \cdot \frac{p(x_i)}{q(x_i)}, \quad x_i \sim q \]

**Decoding the symbols:**

**\(q(x)\)**: The proposal distribution — the distribution you actually sample from. You choose \(q\); the art of importance sampling is choosing it well.

**\(p(x)\)**: The original distribution you want to average under.

**\(p(x_i)/q(x_i)\)**: The importance weight for sample \(x_i\). If \(q\) over-samples a region relative to \(p\), the weight is less than 1 (down-weighting the over-sampled points). If \(q\) under-samples a region, the weight is greater than 1.

**Reading in English**: "Draw samples from a smarter distribution \(q\), then multiply each sample's contribution by a correction factor that accounts for how \(q\) distorts the sampling."

The estimator is still unbiased: \(\mathbb{E}_q[\hat{\mu}_{\text{IS}}] = \mathbb{E}_p[f(X)]\).

### SSA example: estimating conjunction probability for rare close approaches

```python
import torch
import torch.distributions as dist

torch.manual_seed(7)

# --- Problem setup ---
# Two satellites. The relative position uncertainty is modeled as a 1D Gaussian
# with mean 500m and std 200m (separation in the conjunction plane).
# We want P(miss distance < 100m) — a "dangerous conjunction."
#
# True answer: P(X < 100) where X ~ Normal(500, 200^2)
# = Phi((100 - 500) / 200) = Phi(-2.0) ≈ 0.0228

mu_sep = 500.0    # mean separation in meters
sigma_sep = 200.0 # std of separation
threshold = 100.0 # dangerous miss distance

p_dist = dist.Normal(mu_sep, sigma_sep)
true_prob = p_dist.cdf(torch.tensor(threshold)).item()
print(f"True conjunction probability: {true_prob:.6f}")

# --- Naive Monte Carlo ---
# Sample from p (the real distribution) and check if < threshold.
N = 50_000
samples_naive = p_dist.sample((N,))
mc_naive = (samples_naive < threshold).float().mean().item()
print(f"\nNaive MC (N={N}):   estimate={mc_naive:.6f}")
print(f"  Relative error: {abs(mc_naive - true_prob)/true_prob:.2%}")
# With P ≈ 0.023, most samples are misses. Very few useful samples.

# --- Importance Sampling ---
# Proposal q: Normal centered on the dangerous region (mean = 0, std = 50).
# This distribution almost entirely generates samples below the threshold.
q_dist = dist.Normal(0.0, 50.0)

samples_is = q_dist.sample((N,))
# Importance weights: p(x) / q(x)
log_w = p_dist.log_prob(samples_is) - q_dist.log_prob(samples_is)
w = torch.exp(log_w)
# f(x) = indicator that x < threshold
f_x = (samples_is < threshold).float()
mc_is = (f_x * w).mean().item()
print(f"\nImportance sampling (N={N}):   estimate={mc_is:.6f}")
print(f"  Relative error: {abs(mc_is - true_prob)/true_prob:.2%}")

# --- Compare variance ---
# Run each estimator 200 times and measure std of estimates
n_runs = 200
naive_runs = []
is_runs = []
for _ in range(n_runs):
    s = p_dist.sample((N,))
    naive_runs.append((s < threshold).float().mean().item())
    s_q = q_dist.sample((N,))
    lw = p_dist.log_prob(s_q) - q_dist.log_prob(s_q)
    is_runs.append(((s_q < threshold).float() * torch.exp(lw)).mean().item())

print(f"\nStd of naive MC estimates:   {torch.tensor(naive_runs).std().item():.6f}")
print(f"Std of IS estimates:         {torch.tensor(is_runs).std().item():.6f}")
# IS should show substantially lower variance
```

The importance sampling estimator concentrates its computational budget on the dangerous region, dramatically reducing variance for the same N.

The Rust translation is more interesting than the others because `rand_distr::Normal` gives you sampling but not log-density. You have to roll the log-PDF yourself, which is a useful exercise:

```rust
# extern crate ndarray;
# extern crate rand;
# extern crate rand_distr;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

// Log-PDF of a univariate Normal(mu, sigma). PyTorch's Normal.log_prob does
// this; rand_distr deliberately separates "sample from this distribution"
// (its job) from "evaluate this density" (statrs's job or yours).
fn normal_log_prob(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * PI).ln()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);

    let mu_sep    = 500.0_f64;
    let sigma_sep = 200.0_f64;
    let threshold = 100.0_f64;

    // True answer: Φ((threshold - μ)/σ) = Φ(-2.0). Computing the Normal CDF
    // in Rust requires the error function, which isn't in std. statrs would
    // give it, but we already know the answer for this example.
    let true_prob = 0.02275_f64;
    println!("True conjunction probability: {true_prob:.6}");

    // --- Naive Monte Carlo: sample from p, check below threshold. ---
    let n = 50_000;
    let p_dist = Normal::new(mu_sep, sigma_sep).unwrap();
    let samples_naive = Array1::<f64>::from_shape_fn(n, |_| p_dist.sample(&mut rng));
    let mc_naive = samples_naive
        .mapv(|x| if x < threshold { 1.0 } else { 0.0 })
        .mean()
        .unwrap();
    println!("\nNaive MC (N={n}):   estimate={mc_naive:.6}");
    println!("  Relative error: {:.2}%", 100.0 * (mc_naive - true_prob).abs() / true_prob);

    // --- Importance sampling: draw from q centered on the dangerous region. ---
    let q_mu    = 0.0_f64;
    let q_sigma = 50.0_f64;
    let q_dist  = Normal::new(q_mu, q_sigma).unwrap();
    let samples_is = Array1::<f64>::from_shape_fn(n, |_| q_dist.sample(&mut rng));

    // Importance weights: p(x) / q(x), computed in log-space for stability.
    let log_w = samples_is.mapv(|x|
        normal_log_prob(x, mu_sep, sigma_sep) - normal_log_prob(x, q_mu, q_sigma));
    let w   = log_w.mapv(f64::exp);
    let f_x = samples_is.mapv(|x| if x < threshold { 1.0 } else { 0.0 });
    let mc_is = (&f_x * &w).mean().unwrap();
    println!("\nImportance sampling (N={n}):   estimate={mc_is:.6}");
    println!("  Relative error: {:.2}%", 100.0 * (mc_is - true_prob).abs() / true_prob);

    // --- Compare variance across many independent runs. ---
    let n_runs = 200;
    let mut naive_runs = Vec::with_capacity(n_runs);
    let mut is_runs    = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let s = Array1::<f64>::from_shape_fn(n, |_| p_dist.sample(&mut rng));
        naive_runs.push(s.mapv(|x| if x < threshold { 1.0 } else { 0.0 }).mean().unwrap());

        let sq = Array1::<f64>::from_shape_fn(n, |_| q_dist.sample(&mut rng));
        let lw = sq.mapv(|x| normal_log_prob(x, mu_sep, sigma_sep) - normal_log_prob(x, q_mu, q_sigma));
        let f  = sq.mapv(|x| if x < threshold { 1.0 } else { 0.0 });
        is_runs.push((&f * &lw.mapv(f64::exp)).mean().unwrap());
    }
    let naive = Array1::from(naive_runs);
    let is    = Array1::from(is_runs);
    println!("\nStd of naive MC estimates:   {:.6}", naive.std(1.0));
    println!("Std of IS estimates:         {:.6}", is.std(1.0));
}
```

Two things worth flagging here. First, `&f_x * &w` (using references) is how you do elementwise multiplication of two ndarrays without consuming them. Drop the references and the arrays get moved into the operation, which usually isn't what you want. Second, hand-rolling `normal_log_prob` made the math explicit: this is just `-z²/2 - log σ - ½ log(2π)`, with `z = (x - μ)/σ`. PyTorch's `log_prob` is doing exactly that arithmetic for you.

### The critical warning: coverage

Importance sampling can fail catastrophically if the proposal \(q\) does not cover the full support of \(p\). Specifically, wherever \(p(x) > 0\) but \(q(x) = 0\), the weight \(p(x)/q(x)\) is infinite — those samples can never be drawn, so their contribution is permanently lost, and the estimator is **biased** (no longer corrects to the true expectation).

The practical rule: \(q\) should have heavier tails than \(p\), not lighter. A proposal with lighter tails will create regions where \(p(x)/q(x)\) is astronomically large — and the few samples that land there will dominate the estimate, causing high variance and instability. In SSA terms: if your proposal distribution only covers "near-nominal" conjunction geometries, you will miss the contribution of extreme-approach scenarios entirely.

---

## How this connects to game-playing algorithms

When MCTS evaluates a game position, it cannot sum over all possible continuations of the game (there are too many). Instead, it runs random **rollouts**: simulations of a complete game from that position to the end, following some approximate policy. The fraction of those rollouts that end in a win is the Monte Carlo estimate of the winning probability from that position.

Each rollout is one sample. The winning probability estimate improves as more rollouts are run. The estimate is noisy with few rollouts and reliable with many. That is exactly the convergence behavior you just saw with pi and eclipse fraction.

### The MCTS value estimate as a Monte Carlo average

In MCTS, each node in the search tree tracks two quantities: \(W\) (total wins accumulated from rollouts through this node) and \(N\) (total visits). The value estimate is simply:

\[ \hat{V}(\text{node}) = \frac{W}{N} \]

This is exactly a Monte Carlo average of rollout outcomes (each rollout returns 1 for a win and 0 for a loss). By the CLT, this estimate is approximately Normal around the true win probability, with standard deviation \(\sim 1/(2\sqrt{N})\) (for win probabilities near 0.5). With 40 rollouts, the standard error is about 0.079; with 400 rollouts it falls to 0.025 — a 3× improvement for a 10× cost.

### The UCB exploration-exploitation tradeoff

MCTS does not distribute rollouts uniformly across all child nodes. It uses the Upper Confidence Bound (UCB) formula to balance exploration and exploitation:

\[ \text{UCB}(\text{node}) = \frac{W_i}{N_i} + c \sqrt{\frac{\ln N_{\text{parent}}}{N_i}} \]

**Decoding:**

**\(W_i / N_i\)**: The exploitation term — the current value estimate for child \(i\). Prefer nodes with high estimated value.

**\(c \sqrt{\ln N_{\text{parent}} / N_i}\)**: The exploration term. When \(N_i\) is small, this term is large — under-visited nodes get a bonus. The \(\ln N_{\text{parent}}\) in the numerator grows slowly, preventing the exploration bonus from dominating forever.

**\(c\)**: A hyperparameter controlling the exploration-exploitation trade-off. Common values are \(\sqrt{2}\) or 1.0; in practice it is tuned per domain.

The UCB formula is optimal in the bandit setting (proven by Auer, Cesa-Bianchi, and Fischer, 2002): it minimizes regret (the gap between UCB's cumulative reward and that of always choosing the best arm) at a logarithmic rate. In a game tree, this means MCTS concentrates rollouts on the most promising lines while guaranteeing no subtree is permanently neglected.

```python
import torch
import math

torch.manual_seed(3)

# A toy MCTS on a single parent node with 3 child nodes.
# True win probabilities for the children (unknown to the algorithm).
true_win_probs = torch.tensor([0.45, 0.60, 0.35])
n_children = len(true_win_probs)

W = torch.zeros(n_children)   # wins per child
N = torch.zeros(n_children)   # visits per child
N_total = 0
c = math.sqrt(2)

def ucb_score(w, n, n_total, c):
    if n == 0:
        return float('inf')  # unvisited nodes get infinite priority
    return w / n + c * math.sqrt(math.log(n_total) / n)

print(f"True win probs: {true_win_probs.tolist()}")
print(f"Best child: {true_win_probs.argmax().item()} (prob={true_win_probs.max().item():.2f})")
print()

for rollout in range(1, 401):
    # Select child with highest UCB score
    scores = [ucb_score(W[i].item(), N[i].item(), N_total + 1, c)
              for i in range(n_children)]
    selected = scores.index(max(scores))
    
    # Simulate a rollout: draw outcome from true win probability
    outcome = torch.bernoulli(true_win_probs[selected]).item()
    W[selected] += outcome
    N[selected] += 1
    N_total += 1
    
    if rollout in [10, 40, 100, 200, 400]:
        estimates = [W[i].item() / N[i].item() if N[i] > 0 else 0.0
                     for i in range(n_children)]
        best = estimates.index(max(estimates))
        print(f"After {rollout:>3} rollouts: estimates={[f'{e:.3f}' for e in estimates]}, "
              f"visits={N.int().tolist()}, best={best}")
```

After 40 rollouts the estimates are noisy; after 400, they have converged close to the true values and the algorithm reliably identifies child 1 as the best. This is the 1/√N improvement made concrete in a game-tree context.

The Rust version. Note how the per-node stats become a couple of `Vec<f64>`s; you'll see this same shape (visit counts, accumulated values) in the real MCTS implementation in Module 4, just wrapped in a proper tree.

```rust
# extern crate rand;
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;

fn ucb_score(w: f64, n: f64, n_total: f64, c: f64) -> f64 {
    if n == 0.0 {
        f64::INFINITY            // unvisited nodes get infinite priority
    } else {
        w / n + c * (n_total.ln() / n).sqrt()
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(3);

    let true_win_probs = [0.45_f64, 0.60, 0.35];
    let n_children = true_win_probs.len();

    let mut w = vec![0.0_f64; n_children];
    let mut n = vec![0.0_f64; n_children];
    let mut n_total = 0_u32;
    let c = 2.0_f64.sqrt();

    let (best_idx, &best_prob) = true_win_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("True win probs: {:?}", true_win_probs);
    println!("Best child: {best_idx} (prob={best_prob:.2})");
    println!();

    let checkpoints = [10, 40, 100, 200, 400];

    for rollout in 1..=400 {
        // Select child with highest UCB score.
        let (selected, _) = (0..n_children)
            .map(|i| (i, ucb_score(w[i], n[i], (n_total + 1) as f64, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        // Rollout: Bernoulli draw from this child's true win probability.
        let outcome = if rng.random::<f64>() < true_win_probs[selected] { 1.0 } else { 0.0 };
        w[selected] += outcome;
        n[selected] += 1.0;
        n_total += 1;

        if checkpoints.contains(&rollout) {
            let estimates: Vec<f64> = (0..n_children)
                .map(|i| if n[i] > 0.0 { w[i] / n[i] } else { 0.0 })
                .collect();
            let (best, _) = estimates
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let visits: Vec<i64> = n.iter().map(|&x| x as i64).collect();
            let est_str: Vec<String> = estimates.iter().map(|e| format!("{:.3}", e)).collect();
            println!(
                "After {rollout:>3} rollouts: estimates=[{}], visits={:?}, best={best}",
                est_str.join(", "),
                visits
            );
        }
    }
}
```

The repeated `partial_cmp(...).unwrap()` is the Rust tax for working with floats: `f64` doesn't implement `Ord` because of NaN, so you can't just call `.max()` on an iterator of floats. In production code you'd wrap this in a helper. For now, see it for what it is and move on.

The first M in MCCFR (Monte Carlo Counterfactual Regret Minimization) refers to the same idea. Instead of computing counterfactual regret over all possible game trajectories, MCCFR samples trajectories and estimates the regret from those samples. It converges to the correct solution as the number of samples grows, at the 1/√N rate.

---

## Numerical stability: log-space Monte Carlo

So far we have been accumulating MC estimates in linear probability space. For many SSA applications this is fine. But consider a conjunction probability estimate that involves multiplying together many independent uncertain factors — orbital uncertainty, atmospheric drag uncertainty, sensor accuracy uncertainty — each with probability < 1. The product of 50 small probabilities will underflow to exactly zero in standard 32-bit floating point, even if the true product is a meaningful number like 10^{-15}.

### The log-sum-exp trick

The standard fix is to work in log-space and use the log-sum-exp identity to safely aggregate:

\[ \log\left(\sum_i e^{a_i}\right) = a_{\max} + \log\left(\sum_i e^{a_i - a_{\max}}\right) \]

**Decoding:**

**\(a_{\max}\)**: The maximum of the \(a_i\) values. Shifting all values down by \(a_{\max}\) before exponentiating keeps the numbers in a safe range — the largest term becomes \(e^0 = 1\) and all others are smaller.

**Why it works**: The two forms are mathematically equal; the second just avoids numerical overflow/underflow by keeping all exponentials near 1.

For Monte Carlo in log-space, the pattern is: compute \(\log f(x_i)\) for each sample, then use `torch.logsumexp` to aggregate.

### When to accumulate in log-space vs. linear space

- **Use linear space** when individual sample values are not extremely small or large (roughly in the range 10^{-6} to 10^6 in float32). Most MCTS value estimates, policy entropy computations, and reward averaging fall here.
- **Use log-space** when multiplying many probabilities together, computing likelihoods of long sequences, or working with probabilities below roughly 10^{-7}.

```python
import torch
import math

torch.manual_seed(11)

# Scenario: estimate the probability that a satellite survives 50 consecutive
# orbital passes through a debris field, each with survival probability 0.97.
# True answer: 0.97^50 ≈ 0.2181

p_survive_one_pass = 0.97
n_passes = 50
n_simulations = 10_000

# --- Naive approach: multiply probabilities together in linear space ---
# Each simulation: draw 50 Bernoulli(0.97) samples, check all survived
outcomes = torch.bernoulli(
    torch.full((n_simulations, n_passes), p_survive_one_pass)
)
# Product of all outcomes per simulation (1 if all survived, 0 if any collision)
# But: even the product in float32 won't underflow here because individual
# values are 0 or 1. The underflow problem arises when multiplying small floats.
survived_all = outcomes.prod(dim=1)   # 1.0 or 0.0 per simulation
mc_linear = survived_all.mean().item()

# --- Demonstrate the underflow problem ---
# Instead, multiply the *probabilities* directly (simulating a different
# quantity: the probability-weighted integral, not a Bernoulli draw).
# With 100 factors of 0.97, in float32:
probs_100 = torch.full((100,), 0.97, dtype=torch.float32)
product_linear = probs_100.prod().item()
log_sum_true = 100 * math.log(0.97)

# With 500 factors:
probs_500 = torch.full((500,), 0.97, dtype=torch.float32)
product_500 = probs_500.prod().item()    # This will underflow to 0.0 in float32

print(f"True 50-pass survival probability: {p_survive_one_pass**n_passes:.6f}")
print(f"MC estimate (Bernoulli draws):     {mc_linear:.6f}")
print()
print(f"100-factor product in float32:     {product_linear:.8f}")
print(f"100-factor log-sum (exact):        {math.exp(log_sum_true):.8f}")
print()
print(f"500-factor product in float32:     {product_500}")  # → 0.0 (underflow!)
log_sum_500 = 500 * math.log(0.97)
print(f"500-factor log-space result:       {math.exp(log_sum_500):.8e}")  # ≈ 5.2e-7

# --- Log-space MC accumulation ---
# When each sample contributes a log-probability, use logsumexp to aggregate
log_contributions = torch.full((n_simulations,), math.log(p_survive_one_pass) * n_passes)
# Add noise to simulate MC variation
log_contributions += torch.randn(n_simulations) * 0.05
log_mean_estimate = torch.logsumexp(log_contributions, dim=0) - math.log(n_simulations)
print(f"\nLog-space MC estimate:             {math.exp(log_mean_estimate):.6f}")
```

The underflow from 500 factors of 0.97 to exactly 0.0 is the silent-OOM equivalent for numerical computation: the program runs, returns an answer, and the answer is completely wrong. Log-space accumulation prevents this.

The Rust version. There's no `torch.logsumexp` here, but writing one yourself is six lines and shows exactly what the trick is doing:

```rust
# extern crate ndarray;
# extern crate rand;
# extern crate rand_distr;
use ndarray::{Array1, Array2, Axis};
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

// Numerically safe log(sum(exp(x_i))). Subtract the max before exponentiating
// so the largest term is e^0 = 1 and everything else is smaller. PyTorch's
// torch.logsumexp does the same thing under the hood.
fn logsumexp(xs: &Array1<f64>) -> f64 {
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = xs.iter().map(|&x| (x - max).exp()).sum();
    max + s.ln()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(11);

    let p_survive = 0.97_f64;
    let n_passes = 50_usize;
    let n_simulations = 10_000;

    // --- Naive Bernoulli MC: 1 if all passes survived, 0 if any collision. ---
    let outcomes = Array2::<f64>::from_shape_fn(
        (n_simulations, n_passes),
        |_| if rng.random::<f64>() < p_survive { 1.0 } else { 0.0 },
    );
    let survived_all = outcomes.map_axis(Axis(1), |row| row.iter().product::<f64>());
    let mc_linear = survived_all.mean().unwrap();

    // --- Demonstrate underflow in f32. ---
    // 100 factors of 0.97 is fine; 500 factors collapses to exactly 0.0.
    let product_100: f32 = (0..100).map(|_| 0.97_f32).product();
    let product_500: f32 = (0..500).map(|_| 0.97_f32).product();
    let log_sum_100 = 100.0 * 0.97_f64.ln();
    let log_sum_500 = 500.0 * 0.97_f64.ln();

    println!("True 50-pass survival probability: {:.6}", p_survive.powi(n_passes as i32));
    println!("MC estimate (Bernoulli draws):     {mc_linear:.6}");
    println!();
    println!("100-factor product in f32:         {product_100:.8}");
    println!("100-factor log-sum (exact):        {:.8}", log_sum_100.exp());
    println!();
    println!("500-factor product in f32:         {product_500}");          // 0.0
    println!("500-factor log-space result:       {:.8e}", log_sum_500.exp());

    // --- Log-space MC accumulation with the logsumexp helper. ---
    // Each "sample" is a log-probability that 50 passes survived, plus a bit
    // of Gaussian noise to simulate MC variation across simulations.
    let log_contributions = Array1::<f64>::from_shape_fn(n_simulations, |_| {
        p_survive.ln() * n_passes as f64 + StandardNormal.sample(&mut rng) * 0.05
    });
    let log_mean = logsumexp(&log_contributions) - (n_simulations as f64).ln();
    println!("\nLog-space MC estimate:             {:.6}", log_mean.exp());
}
```

The `f64::max` fold pattern (`fold(f64::NEG_INFINITY, f64::max)`) is the canonical Rust way to take a max over floats; the same NaN-ordering issue from the MCTS example is why this exists. You will see this fold throughout numerical Rust code.

---

## Common pitfalls

**Pitfall 1: Treating a single MC run as reliable.** A single estimate from N=100 samples has a standard error that may be 30-50% of the true value for low-probability events. Always run multiple independent estimates and report the spread.

**Pitfall 2: Forgetting the 1/√N cost.** Going from ±5% error to ±0.5% error requires 100× more samples, not 10×. MC is powerful but the variance reduction is sublinear.

**Pitfall 3: Importance sampling with thin-tailed proposals.** If \(q\) has lighter tails than \(p\), importance weights in the tails blow up. The estimator becomes dominated by a handful of extreme samples. Always verify that \(q\) covers the support of \(p\) and has heavier tails.

**Pitfall 4: Ignoring numerical underflow.** Multiplying together many probabilities in linear float32 will silently underflow to 0. Use log-space accumulation whenever you are multiplying more than ~50 independent probabilities.

**Pitfall 5: Using MC where direct computation is feasible.** MC is for intractable expectations. If your distribution has 10 outcomes, just enumerate them. The 1/√N convergence is strictly worse than direct summation for small, discrete problems.

---

## Key Takeaways

- **Monte Carlo estimation** replaces an intractable sum or integral with an average over N random samples: \(\hat{\mu} = \frac{1}{N}\sum f(x_i)\). The estimate is unbiased and converges at rate \(\sigma/\sqrt{N}\).
- **The Central Limit Theorem** guarantees that the sample mean is approximately Normally distributed around the true mean for large N, regardless of the underlying distribution shape. This is why confidence intervals on MC estimates work, and why you can quote uncertainty alongside any MC result.
- **Importance sampling** corrects for sampling from a non-uniform proposal \(q\) by reweighting each sample by \(p(x_i)/q(x_i)\). It is a powerful variance reducer for rare-event estimation (like dangerous conjunctions) but catastrophically fails if \(q\) does not cover the support of \(p\).
- **MCTS value estimates** are Monte Carlo averages of rollout outcomes. The UCB formula (exploitation + exploration bonus) directs rollouts efficiently across the tree, providing logarithmic regret in the bandit sense. More rollouts give better estimates at the 1/√N rate.
- **Log-space accumulation** (via the log-sum-exp trick) is essential when multiplying together many small probabilities. Naive linear-space products silently underflow to 0 in float32, producing wrong answers with no error signal.
- The 1/√N trade-off is **fundamental and unavoidable** for plain MC. All variance reduction techniques (baselines, importance sampling, control variates) reduce \(\sigma\), not the √N denominator — they get you more accuracy per sample, but the rate of improvement remains sublinear.

---

{{#quiz 03-sampling-and-monte-carlo.toml}}