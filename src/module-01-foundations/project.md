# Module 1 Project: Monte Carlo Conjunction Probability

## What you're building

You're going to write a small Python program that estimates the probability that two satellites will pass within some unsafe distance of each other, given that we know their states only with some uncertainty. This is a real problem in your field. The 18th Space Defense Squadron does an industrial-strength version of it for every conjunction screen they publish, and commercial services like LeoLabs and ComSpOC do dressed-up versions for their customers.

We are using a simplified version: linear motion, position-only uncertainty, isotropic Gaussian noise. That's not realistic. It is, however, the right level of complexity to exercise everything we learned in this module without drowning in orbital mechanics we haven't covered.

## What this exercises

- **Vectors and matrices** (lessons 5-6): satellite states and velocity propagation.
- **Probability distributions** (lesson 1): Gaussian uncertainty in initial position.
- **Sampling and Monte Carlo** (lesson 3): the actual estimator.
- **Bayes intuition** (lesson 2): conditioning on the observed nominal trajectory.
- **Variance and convergence** (lesson 3 again): the sensitivity analysis.
- **Gradient intuition** (lesson 7): we'll do a numerical sensitivity analysis that mirrors what gradients give you.

## Setup

Two satellites at time \\(t = 0\\), both with known nominal positions and velocities. Each has uncertainty in its initial position, modeled as an isotropic 3D Gaussian (the same standard deviation in each axis, no correlations between axes). Velocities are assumed known exactly. (This is the unrealistic part; in reality, velocity uncertainty matters a lot. We'll fix this in a later module when we have proper covariance propagation.)

```
Satellite A:
  nominal position (km):   [0, 0, 0]
  nominal velocity (km/s): [7.5, 0.0, 0.0]
  position uncertainty:    sigma = 0.10 km in each axis

Satellite B:
  nominal position (km):   [100, 0.5, 0]
  nominal velocity (km/s): [-7.5, 0.0, 0.0]
  position uncertainty:    sigma = 0.10 km in each axis

Safety threshold: 1.0 km
Time window:     [0, 20] seconds, sampled at 0.1 s intervals
```

The two satellites are moving directly toward each other, with a half-kilometer cross-track offset, and meet in the middle of the window. Their minimum distance is going to be small. The question is: given the position uncertainty, how often will they actually come within 1 km of each other?

## Step-by-step plan

### Step 1: Encode the nominal scenario

Use vectors. Don't use individual `x`, `y`, `z` variables; that defeats the point.

```python
import torch

# Nominals
r0_A = torch.tensor([  0.0, 0.0, 0.0])  # km
r0_B = torch.tensor([100.0, 0.5, 0.0])  # km
v_A  = torch.tensor([ 7.5, 0.0, 0.0])   # km/s
v_B  = torch.tensor([-7.5, 0.0, 0.0])   # km/s

# Uncertainty
sigma = 0.10  # km in each axis, both satellites

# Time grid
dt = 0.1
t = torch.arange(0.0, 20.0 + dt, dt)  # shape: (T,) where T = 201
```

### Step 2: Compute the nominal minimum distance

Before adding noise, do the deterministic version. This is a sanity check and gives you something to compare your Monte Carlo estimate against. Propagate linearly: \\(\mathbf{r}(t) = \mathbf{r}_0 + \mathbf{v} \cdot t\\). For each \\(t\\), compute \\(|\mathbf{r}_A(t) - \mathbf{r}_B(t)|\\). Find the minimum over the time window.

Hint: PyTorch broadcasting will let you do this without a loop. `t.unsqueeze(1)` gives shape `(T, 1)`, and `v.unsqueeze(0)` gives shape `(1, 3)`, and their product broadcasts to `(T, 3)`.

You should find a nominal minimum distance of about 0.5 km (the cross-track offset, which the two satellites can't close given their parallel-but-opposite velocities along x).

### Step 3: Add uncertainty and sample

Now the Monte Carlo part. For each of \\(N\\) trials:

1. Sample a perturbation \\(\delta_A \sim \mathcal{N}(0, \sigma^2 I)\\) and add it to \\(\mathbf{r}_0^A\\).
2. Sample \\(\delta_B \sim \mathcal{N}(0, \sigma^2 I)\\) and add it to \\(\mathbf{r}_0^B\\).
3. Propagate both linearly over the time window.
4. Find the minimum distance over the window.
5. Record whether that minimum was below the safety threshold.

The probability of conjunction \\(P_c\\) is then the fraction of trials with a min-distance below threshold.

```python
def estimate_pc(N, sigma=0.10, threshold=1.0):
    # Sample perturbations: shape (N, 3)
    deltas_A = sigma * torch.randn(N, 3)
    deltas_B = sigma * torch.randn(N, 3)
    
    # Perturbed initial positions: shape (N, 3)
    r0A = r0_A + deltas_A
    r0B = r0_B + deltas_B
    
    # Propagate. We want positions of shape (N, T, 3).
    # r(t) = r0 + v*t
    # t has shape (T,), v has shape (3,), so v*t.unsqueeze(1) is (T, 3)
    trajA = r0A.unsqueeze(1) + (v_A.unsqueeze(0) * t.unsqueeze(1)).unsqueeze(0)
    trajB = r0B.unsqueeze(1) + (v_B.unsqueeze(0) * t.unsqueeze(1)).unsqueeze(0)
    # trajA, trajB: shape (N, T, 3)
    
    # Distances at each timestep: shape (N, T)
    diffs = trajA - trajB
    dists = torch.linalg.norm(diffs, dim=2)
    
    # Min distance per trial: shape (N,)
    min_dists = dists.min(dim=1).values
    
    # Probability estimate
    pc = (min_dists < threshold).float().mean()
    return pc.item(), min_dists
```

If the broadcasting is making your head hurt, write it with a `for` loop first, get correct numbers, then refactor to vectorized form. Vectorized PyTorch will be 10 to 100 times faster, which matters when we crank \\(N\\) up.

### Step 4: Convergence study

Run the estimator with \\(N \in \{100, 1\,000, 10\,000, 100\,000\}\\), repeating 10 times for each \\(N\\). Plot or print the mean and standard deviation of \\(P_c\\) across the 10 runs. You should see the standard deviation shrink as roughly \\(1/\sqrt{N}\\), exactly as lesson 3 promised.

```python
import torch

for N in [100, 1_000, 10_000, 100_000]:
    runs = [estimate_pc(N)[0] for _ in range(10)]
    runs_t = torch.tensor(runs)
    print(f"N={N:>6}: Pc mean = {runs_t.mean():.4f}, std = {runs_t.std():.4f}")
```

Your absolute \\(P_c\\) value will depend on your scenario and threshold. The point is the convergence behavior, not any specific number.

### Step 5: Sensitivity analysis

This is the part that previews gradients without using them yet. We want to know: how sensitive is \\(P_c\\) to the uncertainty level \\(\sigma\\)?

Compute \\(P_c\\) for \\(\sigma \in \{0.05, 0.10, 0.20, 0.50, 1.00\}\\) km, all with \\(N = 50\,000\\). In this geometry you should see \\(P_c\\) **decrease** as \\(\sigma\\) grows. The reason: the nominal minimum distance (0.5 km) is already below the 1.0 km threshold, so the baseline scenario is almost always a conjunction. Wider uncertainty scatters samples further from the nominal, pushing more of them above threshold and lowering \\(P_c\\).

The direction flips if you move the nominal above threshold. Change `r0_B[1]` from 0.5 to 1.5 (a 1.5 km cross-track offset, giving a nominal miss of 1.5 km, safely outside the 1.0 km threshold) and rerun. Now \\(P_c\\) increases with \\(\sigma\\): uncertainty is occasionally bridging the gap into the danger zone.

The lesson: "more uncertainty means more Pc" is not a law. The direction of \\(dP_c/d\sigma\\) depends on which side of the threshold the nominal sits. This matters operationally: a maneuver that nudges the nominal further from the threshold can either increase or decrease the sensitivity of Pc to state uncertainty, depending on the geometry. This is, conceptually, a finite-difference approximation to \\(dP_c/d\sigma\\): you're seeing how the output changes as the input parameter changes. If you wrote the entire pipeline in PyTorch with `requires_grad=True` on \\(\sigma\\), you could get this gradient analytically with `.backward()`. We're doing it the slow way for now, but the fact that "sensitivity of an output to an input" is exactly what gradients give you is the bridge to module 2.

### Step 6: Reflect

At the end of your script (or in a comment block), write down answers to:

1. How does the standard deviation of your \\(P_c\\) estimates scale with \\(N\\)?
2. What's the smallest \\(N\\) at which you'd trust the answer to two decimal places?
3. If \\(P_c\\) for the nominal scenario is, say, 0.04, and you have a "decision threshold" of 0.001 (the value above which JSpOC might issue a maneuver recommendation), how many samples do you need before your estimator's noise is small compared to that threshold? (Hint: the standard error needs to be much smaller than 0.001.)
4. What's missing from this model that you'd want to add to make it realistic? (Velocity uncertainty? Correlated position uncertainty? Nonlinear dynamics? Time-varying covariance?)

These questions are not graded; they are the things you should be able to answer after doing the project, and they are the things that distinguish "I ran the code" from "I understood the code."

## Stretch goals (optional)

Pick one or two if you want extra reps:

- **Different geometries.** Modify the scenario so the nominal trajectories cross exactly (head-on with no offset). What does \\(P_c\\) do as a function of \\(\sigma\\) here, vs. the near-miss case?
- **Velocity uncertainty.** Add Gaussian noise to the initial velocities too. Does this change the convergence rate of your estimator? (Spoiler: no, it's still \\(1/\sqrt{N}\\). What changes is the **variance** of individual trial outcomes, which affects the multiplicative constant.)
- **A baseline comparison.** Implement a "delta-r" approximation: assume the relative position vector at the nominal time of closest approach is Gaussian, and use the standard analytic formula for probability of falling inside a sphere of radius `threshold` around the origin. Compare to your Monte Carlo estimate. They should agree to within Monte Carlo noise. This is a useful sanity check.

## What you should hand yourself afterward

A single Python file (or notebook) that:

- Imports torch.
- Defines the scenario as named variables.
- Has a function `estimate_pc(N, sigma, threshold)` that returns the estimate.
- Runs the convergence study and prints results.
- Runs the sensitivity analysis and prints results.
- Has a comment block at the bottom with your answers to the reflection questions.

Don't make this fancy. The code should be 100 to 200 lines, including comments. The point isn't a polished tool; it's that you've now used every concept from this module on a single small problem and seen them work together.

## What's next

Module 2 will build neural networks: stacks of the matrix-vector multiplications you saw in lesson 6, with the gradients you saw in lesson 7 doing the training. The Monte Carlo machinery you built here will come back when we get to RL in module 3, where each "rollout" of a policy is exactly the same kind of sample-and-average loop you just wrote.
