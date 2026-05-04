# Lesson 9: The Multivariate Gaussian

**Module:** ML and Game Theory for Space Power — M01: Foundations
**Source:** *Mathematics for Machine Learning* — Deisenroth, Faisal & Ong (2020), Chapter 6.5; *Bayesian Statistics the Fun Way* — Will Kurt

---

## Where this fits

Lesson 1 introduced the Gaussian for scalar quantities: a single mean and a single variance describe your uncertainty about one number. But a satellite's orbital state is six-dimensional — position in three axes (x, y, z) and velocity in three axes (vx, vy, vz) — and those six components are correlated. A single variance cannot capture the fact that "position uncertainty is larger in the radial direction than cross-track," or that a large along-track position error often comes with a correspondingly large along-track velocity error.

The multivariate Gaussian is the tool for correlated, multi-dimensional uncertainty. It is a distribution over vectors, not scalars, and it can represent the full geometry of an uncertainty cloud in any number of dimensions.

This lesson builds directly on the covariance intuition from Lesson 6, the matrix multiplication from Lesson 6, and the eigenvalue intuition introduced there. It feeds forward into several critical later topics:

- **SSA conjunction probability**: the probability of collision is computed by integrating a bivariate Gaussian in the conjunction plane.
- **Particle filter initialization (Module 07)**: particles are drawn from a multivariate Gaussian centered on the prior belief.
- **Neural network weight initialization**: the default initialization in `nn.Linear` draws weights from a distribution related to the Gaussian.
- **Kalman filter mechanics**: the Kalman update is the closed-form solution for conditioning a Gaussian prior on a Gaussian observation. Understanding marginals and conditionals of the multivariate Gaussian is the same as understanding why the Kalman filter works.

---

## The covariance matrix

Start with the 2D case. Suppose you are tracking the position of an RSO in a cross-sectional plane (cross-track and radial, for example), and you have two uncertain measurements: \\(X_1\\) (cross-track position error, km) and \\(X_2\\) (radial position error, km).

For each variable, you already know the concept of variance from Lesson 1:

\\[ \text{Var}(X_1) = \mathbb{E}[(X_1 - \mu_1)^2] \\]
\\[ \text{Var}(X_2) = \mathbb{E}[(X_2 - \mu_2)^2] \\]

But when you have two variables, there is a third quantity: the **covariance**, which measures how much the two variables move together:

\\[ \text{Cov}(X_1, X_2) = \mathbb{E}[(X_1 - \mu_1)(X_2 - \mu_2)] \\]

**Decoding:**

- \\((X_1 - \mu_1)\\): how far \\(X_1\\) deviates from its mean on a given trial.
- \\((X_2 - \mu_2)\\): how far \\(X_2\\) deviates from its mean on the same trial.
- Multiplied together and averaged: if they tend to deviate in the same direction (both high or both low at the same time), the product is positive on average, so covariance is positive. If they tend to deviate in opposite directions, the product is negative on average, so covariance is negative. If they are uncorrelated, the positive and negative products cancel, giving covariance near zero.

The **covariance matrix** \\(\Sigma\\) assembles all variances and covariances into a single matrix. For a 2D random vector \\(\mathbf{x} = [X_1, X_2]^T\\):

\\[
\Sigma = \begin{pmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) \\\\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) \end{pmatrix}
= \begin{pmatrix} \sigma_1^2 & \sigma_{12} \\\\ \sigma_{12} & \sigma_2^2 \end{pmatrix}
\\]

**Decoding the structure:**

- **Diagonal entries** \\(\Sigma_{ii} = \sigma_i^2\\): the variance of variable \\(i\\). These are always non-negative.
- **Off-diagonal entries** \\(\Sigma_{ij} = \text{Cov}(X_i, X_j)\\) for \\(i \neq j\\): how much variable \\(i\\) and variable \\(j\\) move together. Positive means they increase together; negative means they move oppositely; zero means uncorrelated.
- **Symmetry**: \\(\Sigma_{ij} = \Sigma_{ji}\\) always. Covariance of \\(X_i\\) with \\(X_j\\) is the same as covariance of \\(X_j\\) with \\(X_i\\).
- **Positive semi-definiteness**: for any vector \\(\mathbf{v}\\), \\(\mathbf{v}^T \Sigma \mathbf{v} \geq 0\\). Geometrically this means the uncertainty ellipse cannot have negative volume. All eigenvalues of \\(\Sigma\\) are non-negative.

**SSA example**: In an orbital slot, the cross-track and radial position errors of an RSO often have non-zero covariance. When the estimated orbital inclination is uncertain, the object can appear anywhere along a tilted arc in the cross-track/radial plane. If the inclination is too low, both the radial position (perigee too close) and cross-track position (below the equatorial plane at a longitude where you expected the object to be above it) will be off simultaneously in the same direction. That correlation is exactly what a positive \\(\sigma_{12}\\) encodes.

```python
import torch

# 3x3 covariance matrix for (x, y, z) position uncertainty (km^2)
# Diagonal: variances for each axis
# Off-diagonal: cross-axis covariances
Sigma = torch.tensor([
    [9.0,  2.1, -0.5],   # x variance=9, cross-covariance with y=2.1, with z=-0.5
    [2.1,  4.0,  0.8],   # y variance=4, cross-covariance with z=0.8
    [-0.5, 0.8,  2.25],  # z variance=2.25
], dtype=torch.float64)

# Verify symmetry
print(f"Symmetric: {torch.allclose(Sigma, Sigma.T)}")  # True

# Verify positive semi-definiteness: all eigenvalues >= 0
eigenvalues = torch.linalg.eigvalsh(Sigma)  # eigvalsh is for symmetric matrices
print(f"Eigenvalues: {eigenvalues.tolist()}")
print(f"All non-negative (PSD): {(eigenvalues >= 0).all().item()}")  # True

# Standard deviations along each axis
stds = Sigma.diag().sqrt()
print(f"Std dev x: {stds[0].item():.2f} km, "
      f"y: {stds[1].item():.2f} km, "
      f"z: {stds[2].item():.2f} km")

# Correlation matrix (normalize covariances by std devs)
# corr_ij = Sigma_ij / (sigma_i * sigma_j)
std_outer = stds.unsqueeze(1) * stds.unsqueeze(0)
corr = Sigma / std_outer
print(f"\nCorrelation matrix (off-diagonals are in [-1, 1]):")
print(corr.round(decimals=3))
```

Note that `torch.linalg.eigvalsh` is the right function here: it is specialized for symmetric matrices, returns real eigenvalues in ascending order, and is numerically more stable than the general `torch.linalg.eig`. A covariance matrix with a negative eigenvalue indicates a numerical or construction error — it is not a valid covariance matrix.

---

## The multivariate Gaussian PDF

For a \\(d\\)-dimensional random vector \\(\mathbf{x} \in \mathbb{R}^d\\), the multivariate Gaussian distribution with mean \\(\boldsymbol{\mu}\\) and covariance \\(\Sigma\\) has probability density:

\\[
p(\mathbf{x}) = (2\pi)^{-d/2} \, |\Sigma|^{-1/2} \, \exp\!\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
\\]

**Decoding each piece:**

**\\((2\pi)^{-d/2}\\)**: a normalization constant that grows with dimension. It ensures the density integrates to 1 over all of \\(\mathbb{R}^d\\). In 1D (d=1), this is \\(\frac{1}{\sqrt{2\pi}}\\), which you recognize from the 1D Gaussian.

**\\(|\Sigma|^{-1/2}\\)**: the inverse square root of the determinant of \\(\Sigma\\). The determinant \\(|\Sigma|\\) measures the "volume" of the uncertainty ellipsoid. A large determinant (spread-out distribution) makes the density lower overall; a small determinant (tight distribution) makes the density higher, concentrating probability mass more sharply. Dividing by this ensures the total probability is 1 regardless of how spread out \\(\Sigma\\) is.

**\\(\exp(\cdots)\\)**: the exponential is always positive and equals 1 at its maximum (when \\(\mathbf{x} = \boldsymbol{\mu}\\)), decaying toward zero as \\(\mathbf{x}\\) moves away from \\(\boldsymbol{\mu}\\).

**\\((\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\\)**: this is the **Mahalanobis distance squared**. It is the scalar quantity inside the exponent, and it is the key to understanding how the multivariate Gaussian differs from a simple product of independent Gaussians.

### The Mahalanobis distance

The **Mahalanobis distance** of a point \\(\mathbf{x}\\) from the mean \\(\boldsymbol{\mu}\\) is:

\\[
d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
\\]

**Decoding:**

- If \\(\Sigma = I\\) (identity matrix, all dimensions independent with unit variance), then \\(\Sigma^{-1} = I\\) and \\(d_M = \|\mathbf{x} - \boldsymbol{\mu}\|_2\\): the ordinary Euclidean distance.
- When \\(\Sigma\\) is not the identity, \\(\Sigma^{-1}\\) rescales and rotates the difference vector so that dimensions with larger variance are "shrunk" before computing the distance. An observation 2 km away along a direction with 4 km standard deviation is "closer" (in Mahalanobis terms) than one 2 km away along a direction with 1 km standard deviation.
- The Mahalanobis distance answers: "How many standard deviations (accounting for the full correlation structure) is \\(\mathbf{x}\\) from the mean?" It is the multivariate generalization of "how many sigmas away is this?"

**SSA example**: your RSO tracking system reports a mean position \\(\boldsymbol{\mu}\\) and covariance \\(\Sigma\\) for an object in GEO. A ground telescope reports a candidate detection at position \\(\mathbf{x}_A\\) (2 km from \\(\boldsymbol{\mu}\\) in the radial direction) and another candidate at \\(\mathbf{x}_B\\) (2 km from \\(\boldsymbol{\mu}\\) in the along-track direction). Euclidean distance calls these equal. But if radial uncertainty is 5 km (large, common in GEO) while along-track uncertainty is 0.5 km (tight), then \\(\mathbf{x}_A\\) is only 0.4 Mahalanobis sigmas away while \\(\mathbf{x}_B\\) is 4 Mahalanobis sigmas away. Candidate \\(\mathbf{x}_B\\) is a much more surprising observation; it is far less likely to be the same object.

```python
import torch
from torch.distributions import MultivariateNormal

torch.manual_seed(42)

# RSO position estimate in ECI (km): mean and covariance
mu = torch.tensor([7000.0, 0.0, 0.0], dtype=torch.float64)  # km from Earth center

# Elongated uncertainty: large radial (x-axis here), tight cross-track/z
Sigma = torch.tensor([
    [25.0,  0.0,  0.0],   # 5 km 1-sigma in x (radial-ish)
    [ 0.0,  0.25, 0.0],   # 0.5 km 1-sigma in y (cross-track)
    [ 0.0,  0.0,  0.25],  # 0.5 km 1-sigma in z
], dtype=torch.float64)

dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma)

# Three candidate observations:
x_A = torch.tensor([7002.0, 0.0, 0.0], dtype=torch.float64)  # 2 km in radial (easy direction)
x_B = torch.tensor([7000.0, 2.0, 0.0], dtype=torch.float64)  # 2 km cross-track (tight direction)
x_C = torch.tensor([7001.0, 0.3, 0.1], dtype=torch.float64)  # a realistic noisy observation

# Euclidean distance (ignores covariance shape)
for name, x in [("A", x_A), ("B", x_B), ("C", x_C)]:
    eucl = torch.norm(x - mu).item()

    # Mahalanobis distance: sqrt( (x-mu)^T Sigma^{-1} (x-mu) )
    diff = (x - mu).unsqueeze(1)                        # column vector
    Sigma_inv = torch.linalg.inv(Sigma)
    mahal_sq = (diff.T @ Sigma_inv @ diff).squeeze().item()
    mahal = mahal_sq ** 0.5

    log_p = dist.log_prob(x).item()
    print(f"Candidate {name}: Euclidean={eucl:.2f} km, "
          f"Mahalanobis={mahal:.2f} sigma, log_prob={log_p:.2f}")

# Output shows A is 0.4 Mahalanobis sigma (plausible), B is 4.0 (suspicious),
# even though both are 2 km Euclidean. log_prob reflects this ranking.
```

---

## The uncertainty ellipse and ellipsoid

The Mahalanobis distance gives a natural way to describe the shape of a multivariate Gaussian. The set of all points \\(\mathbf{x}\\) at Mahalanobis distance exactly \\(k\\) from \\(\boldsymbol{\mu}\\) satisfies:

\\[
(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) = k^2
\\]

In 2D, this is an **ellipse**. In 3D, it is an **ellipsoid**. The axes of this ellipse/ellipsoid are the eigenvectors of \\(\Sigma\\), and the half-lengths of the axes are proportional to \\(\sqrt{\lambda_i}\\) where \\(\lambda_i\\) are the eigenvalues. A large eigenvalue means the distribution is spread far in that eigenvector direction.

**Decoding:** The eigenvectors of \\(\Sigma\\) point in the "natural axes" of the uncertainty. If the covariance matrix is diagonal, those axes align with the coordinate axes. If \\(\Sigma\\) has off-diagonal entries, the ellipse is tilted — the natural axes of uncertainty are rotated relative to the coordinate frame.

### The 68-95-99.7 rule does not directly generalize to multiple dimensions

In 1D, 68% of probability mass falls within 1 sigma of the mean. In multiple dimensions, the 1-sigma ellipse (Mahalanobis distance ≤ 1) does not contain 68%:

- In **2D**: the 1-sigma ellipse contains approximately **39%** of the probability mass.
- In **3D**: the 1-sigma ellipsoid contains approximately **20%** of the probability mass.
- In **d** dimensions, the fraction inside the k-sigma ellipsoid is the chi-squared CDF with d degrees of freedom evaluated at \\(k^2\\).

The reason: in higher dimensions, most of the probability mass concentrates in a shell away from the center (the "curse of dimensionality" for Gaussians). The 95% containment ellipse in 2D has Mahalanobis radius \\(\sqrt{5.99} \approx 2.45\\), not 2.

**SSA example**: a conjunction message reports the combined position uncertainty covariance of two RSOs in the conjunction plane. The reported 1-sigma ellipse encloses only about 39% of possible relative-position outcomes. When analysts speak of "the 3-sigma ellipse" they typically mean the ellipse with Mahalanobis radius 3, which in 2D encloses about 98.9% of probability mass. Conflating this with the 1D rule (where 3-sigma captures 99.7%) leads to underestimates of conjunction risk.

```python
import torch
from torch.distributions import MultivariateNormal, Chi2

torch.manual_seed(0)

# 2D position uncertainty in the conjunction plane (km^2)
mu_2d = torch.tensor([0.0, 0.0], dtype=torch.float64)
Sigma_2d = torch.tensor([
    [4.0, 2.4],   # tilted covariance: strong correlation
    [2.4, 2.0],
], dtype=torch.float64)

dist_2d = MultivariateNormal(loc=mu_2d, covariance_matrix=Sigma_2d)

# Sample many points and check Mahalanobis distance fractions
n = 200_000
samples = dist_2d.sample((n,))               # shape (n, 2)

# Mahalanobis distance for each sample
Sigma_inv = torch.linalg.inv(Sigma_2d)
diff = samples - mu_2d                       # (n, 2)
# (n, 2) @ (2, 2) @ (2, n) but we want (n,) -- use einsum
mahal_sq = torch.einsum('ni,ij,nj->n', diff, Sigma_inv, diff)

for k in [1.0, 2.0, 3.0]:
    frac_inside = (mahal_sq <= k**2).float().mean().item()
    # Compare to chi-squared CDF with d=2 degrees of freedom
    chi2_cdf = Chi2(df=torch.tensor(2.0)).cdf(torch.tensor(k**2)).item()
    print(f"k={k:.0f}: sample fraction inside = {frac_inside:.4f}, "
          f"chi2 CDF = {chi2_cdf:.4f}")

# Expected:
# k=1: ~0.393  (not 0.683 -- 2D changes the rule)
# k=2: ~0.865  (not 0.954)
# k=3: ~0.989  (close to 0.997 by coincidence at k=3 in 2D)
```

---

## Marginals and conditionals of a Gaussian

One of the most important properties of the multivariate Gaussian is that it is **closed under marginalization and conditioning**: both operations produce Gaussian results.

### Marginalizing out dimensions

Suppose \\(\mathbf{x} = [\mathbf{x}_a, \mathbf{x}_b]^T \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)\\) where we partition the vector into two parts. The marginal distribution over \\(\mathbf{x}_a\\) is:

\\[
p(\mathbf{x}_a) = \mathcal{N}(\mathbf{x}_a \mid \boldsymbol{\mu}_a, \Sigma_{aa})
\\]

where \\(\boldsymbol{\mu}_a\\) is the subvector of \\(\boldsymbol{\mu}\\) corresponding to the \\(\mathbf{x}_a\\) dimensions, and \\(\Sigma_{aa}\\) is the corresponding submatrix of \\(\Sigma\\). You literally just extract the relevant rows and columns — no integration required.

### Conditioning on observations

Now suppose you observe \\(\mathbf{x}_b = \mathbf{b}\\) (you measure part of the state). The conditional distribution of \\(\mathbf{x}_a\\) given this observation is:

\\[
p(\mathbf{x}_a \mid \mathbf{x}_b = \mathbf{b}) = \mathcal{N}(\mathbf{x}_a \mid \boldsymbol{\mu}_{a|b}, \Sigma_{a|b})
\\]

where the conditional mean and covariance are:

\\[
\boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \Sigma_{ab} \Sigma_{bb}^{-1} (\mathbf{b} - \boldsymbol{\mu}_b)
\\]

\\[
\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}
\\]

**Decoding the conditional mean:**

- \\((\mathbf{b} - \boldsymbol{\mu}_b)\\): the innovation — how far the observed \\(\mathbf{b}\\) is from what you expected.
- \\(\Sigma_{bb}^{-1}(\mathbf{b} - \boldsymbol{\mu}_b)\\): the innovation normalized by the prior uncertainty in \\(\mathbf{b}\\).
- \\(\Sigma_{ab} \Sigma_{bb}^{-1}(\mathbf{b} - \boldsymbol{\mu}_b)\\): "how much does observing a deviation in \\(\mathbf{b}\\) tell me to shift my estimate of \\(\mathbf{a}\\)?" The cross-covariance \\(\Sigma_{ab}\\) propagates the information.
- If \\(\Sigma_{ab} = 0\\) (the two parts are uncorrelated), the observation of \\(\mathbf{b}\\) tells you nothing about \\(\mathbf{a}\\) and the mean does not shift.

**Decoding the conditional covariance:**

- \\(\Sigma_{aa}\\): your prior uncertainty about \\(\mathbf{a}\\).
- \\(\Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}\\): the uncertainty reduction from observing \\(\mathbf{b}\\). This is always non-negative (the subtracted term is positive semi-definite), so the posterior is always at least as certain as the prior. Observing correlated variables can only reduce uncertainty.

**SSA example**: you have a 4D state uncertainty over (range, range-rate, azimuth, elevation) for an RSO. Your telescope reports a measurement of azimuth and elevation. Conditioning the 4D Gaussian on the observed (azimuth, elevation) = \\(\mathbf{b}\\) gives you an updated 2D distribution over (range, range-rate). This is precisely the Kalman filter measurement update step — the formulas above are the Kalman update in disguise when the measurement model is linear.

```python
import torch
from torch.distributions import MultivariateNormal

torch.manual_seed(1)

# 4D state: [range (km), range_rate (km/s), azimuth (deg), elevation (deg)]
mu_full = torch.tensor([1200.0, -0.8, 45.0, 30.0], dtype=torch.float64)

# Full 4x4 covariance (range/range-rate correlated; az/el correlated;
# cross-correlations between range group and angle group)
Sigma_full = torch.tensor([
    [100.0,  2.0,  0.5,  0.2],
    [  2.0,  0.04, 0.01, 0.005],
    [  0.5,  0.01, 0.25, 0.05],
    [  0.2,  0.005, 0.05, 0.09],
], dtype=torch.float64)

# Partition indices: a = range/range-rate (0,1), b = azimuth/elevation (2,3)
a_idx = [0, 1]
b_idx = [2, 3]

mu_a     = mu_full[a_idx]                     # (2,)
mu_b     = mu_full[b_idx]                     # (2,)
Sigma_aa = Sigma_full[a_idx][:, a_idx]        # (2,2)
Sigma_bb = Sigma_full[b_idx][:, b_idx]        # (2,2)
Sigma_ab = Sigma_full[a_idx][:, b_idx]        # (2,2)
Sigma_ba = Sigma_ab.T                         # (2,2)

# Observation: telescope reports azimuth=45.3 deg, elevation=29.8 deg
b_obs = torch.tensor([45.3, 29.8], dtype=torch.float64)
innovation = b_obs - mu_b                     # (2,)

# Conditional mean: mu_a + Sigma_ab @ Sigma_bb^{-1} @ innovation
Sigma_bb_inv = torch.linalg.inv(Sigma_bb)
gain = Sigma_ab @ Sigma_bb_inv                # (2,2) -- the Kalman gain matrix
mu_a_given_b = mu_a + gain @ innovation

# Conditional covariance: Sigma_aa - Sigma_ab @ Sigma_bb^{-1} @ Sigma_ba
Sigma_a_given_b = Sigma_aa - Sigma_ab @ Sigma_bb_inv @ Sigma_ba

print("Prior (range, range-rate):")
print(f"  mean = {mu_a.tolist()}")
print(f"  std  = {Sigma_aa.diag().sqrt().tolist()}")

print("\nPosterior (range, range-rate) given az/el observation:")
print(f"  mean = {mu_a_given_b.tolist()}")
print(f"  std  = {Sigma_a_given_b.diag().sqrt().tolist()}")

# Posterior uncertainty should be less than or equal to prior uncertainty
prior_det = torch.linalg.det(Sigma_aa).item()
post_det  = torch.linalg.det(Sigma_a_given_b).item()
print(f"\nPrior covariance determinant: {prior_det:.4f}")
print(f"Post  covariance determinant: {post_det:.4f}")
print(f"Observation reduced volume by factor: {prior_det / post_det:.2f}x")

# Verify: posterior covariance is still PSD
evals = torch.linalg.eigvalsh(Sigma_a_given_b)
print(f"\nPosterior eigenvalues (all >= 0): {evals.tolist()}")
```

---

## Sampling via Cholesky decomposition

To draw samples from \\(\mathcal{N}(\boldsymbol{\mu}, \Sigma)\\), the standard approach uses the **Cholesky decomposition** of \\(\Sigma\\): find the lower triangular matrix \\(L\\) such that \\(LL^T = \Sigma\\). This is the matrix "square root" of \\(\Sigma\\).

The sampling algorithm is:
1. Compute \\(L = \text{cholesky}(\Sigma)\\)
2. Draw \\(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)\\) (a vector of independent standard normals — trivial to sample)
3. Return \\(\mathbf{x} = L\mathbf{z} + \boldsymbol{\mu}\\)

**Why this works — decoding the linear transformation rule:**

If \\(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)\\) and \\(\mathbf{x} = L\mathbf{z} + \boldsymbol{\mu}\\), then:
- Mean of \\(\mathbf{x}\\): \\(\mathbb{E}[\mathbf{x}] = L \cdot \mathbf{0} + \boldsymbol{\mu} = \boldsymbol{\mu}\\). Correct.
- Covariance of \\(\mathbf{x}\\): \\(\text{Cov}(\mathbf{x}) = L \cdot I \cdot L^T = LL^T = \Sigma\\). Correct.

So \\(\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)\\), exactly as desired. The Cholesky factor \\(L\\) stretches and rotates the isotropic (spherical) samples from \\(\mathcal{N}(\mathbf{0}, I)\\) into the correct elongated, correlated shape.

The Cholesky decomposition is covered in Deisenroth et al. Chapter 4.3. Computationally, it is much faster than forming \\(\Sigma^{1/2}\\) via eigendecomposition, and it is numerically stable for well-conditioned covariance matrices. PyTorch exposes it as `torch.linalg.cholesky`.

```python
import torch
from torch.distributions import MultivariateNormal

torch.manual_seed(7)

def sample_multivariate_gaussian(
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    n_samples: int
) -> torch.Tensor:
    """
    Sample from N(mu, Sigma) using Cholesky decomposition.

    Args:
        mu:       mean vector, shape (d,)
        Sigma:    covariance matrix, shape (d, d), symmetric PSD
        n_samples: number of samples to draw

    Returns:
        samples: shape (n_samples, d)
    """
    d = mu.shape[0]
    L = torch.linalg.cholesky(Sigma)             # lower triangular, L @ L.T == Sigma
    z = torch.randn(n_samples, d, dtype=Sigma.dtype)  # z ~ N(0, I)
    # x = z @ L.T + mu  (equivalent to (L @ z.T).T + mu, broadcast-friendly)
    return z @ L.T + mu

# Target distribution: 3D position uncertainty in RSW frame (km)
mu_rsw = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
Sigma_rsw = torch.tensor([
    [4.00, 1.20, 0.00],
    [1.20, 1.00, 0.00],
    [0.00, 0.00, 0.25],
], dtype=torch.float64)

n = 50_000
samples = sample_multivariate_gaussian(mu_rsw, Sigma_rsw, n)

# Verify: sample mean ≈ mu
sample_mean = samples.mean(dim=0)
print("Sample mean (should be near [0, 0, 0]):")
print(sample_mean.tolist())

# Verify: sample covariance ≈ Sigma
# Unbiased sample covariance: 1/(N-1) * sum (x_i - xbar)(x_i - xbar)^T
diff = samples - sample_mean
sample_cov = (diff.T @ diff) / (n - 1)
print("\nSample covariance (should be close to Sigma_rsw):")
print(sample_cov.round(decimals=3))
print("\nTarget Sigma_rsw:")
print(Sigma_rsw)

# Compare to PyTorch's built-in sampler (which also uses Cholesky internally)
dist = MultivariateNormal(loc=mu_rsw, covariance_matrix=Sigma_rsw)
samples_builtin = dist.sample((n,))
builtin_cov = ((samples_builtin - samples_builtin.mean(0)).T @
               (samples_builtin - samples_builtin.mean(0))) / (n - 1)
print(f"\nMax absolute difference between manual and builtin sample covariances: "
      f"{(sample_cov - builtin_cov).abs().max().item():.4f}")
# Should be very small -- both are Monte Carlo estimates of the same quantity
```

**Connection to Module 07**: when the particle filter is initialized, it draws N particles from the prior belief distribution \\(\mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)\\). The Cholesky sampling algorithm above is exactly how that initialization works. Each particle is one sample from the prior — a plausible initial state for the tracked object, consistent with the initial uncertainty.

---

## Linear transformations of a Gaussian

The Cholesky argument generalized: if \\(\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)\\) and \\(\mathbf{y} = A\mathbf{x} + \mathbf{b}\\) for some matrix \\(A\\) and vector \\(\mathbf{b}\\), then:

\\[
\mathbf{y} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, \, A\Sigma A^T)
\\]

**Decoding:**

- **Mean transforms linearly**: \\(\mathbb{E}[\mathbf{y}] = A\boldsymbol{\mu} + \mathbf{b}\\). The mean just gets the same transformation as any individual point.
- **Covariance transforms as \\(A\Sigma A^T\\)**: the \\(A\\) on the left and \\(A^T\\) on the right "wrap around" the original covariance. The transpose appears because covariance is a quadratic object — it involves products of deviations, and each deviation gets transformed by \\(A\\).
- **The bias \\(\mathbf{b}\\) does not affect the covariance**: shifting every sample by the same constant does not change how spread out they are.

**SSA application — frame transformation**: conjunction probability is computed in the conjunction plane frame (the RSW or B-plane frame), not in the ECI frame where orbital state is propagated. To convert a covariance from ECI to RSW frame, you apply a rotation matrix \\(R\\). Since rotation matrices are orthogonal (\\(R^T = R^{-1}\\)), the transformed covariance is \\(R\Sigma R^T\\).

This is the standard preprocessing step in any conjunction probability computation: propagate the state in ECI with its full 6×6 covariance, then rotate to the conjunction plane to get the 2D covariance that governs the collision geometry.

```python
import torch

torch.manual_seed(3)

# 3x3 position covariance in ECI frame (km^2)
# Represents uncertainty that is elongated in the x-direction
Sigma_eci = torch.tensor([
    [16.0,  2.0,  0.5],
    [ 2.0,  2.25, 0.3],
    [ 0.5,  0.3,  1.0],
], dtype=torch.float64)

# Rotation matrix: ECI -> RSW (radial-along-track-cross-track) frame
# For a satellite at a specific orbital position, RSW is a rotation of ECI
# Here we use a simple 45-degree rotation in the x-y plane as illustration
theta = torch.tensor(0.7854, dtype=torch.float64)  # 45 degrees in radians
R = torch.tensor([
    [ torch.cos(theta).item(), torch.sin(theta).item(), 0.0],
    [-torch.sin(theta).item(), torch.cos(theta).item(), 0.0],
    [ 0.0,                    0.0,                      1.0],
], dtype=torch.float64)

# Transform covariance from ECI to RSW frame: Sigma_rsw = R @ Sigma_eci @ R.T
Sigma_rsw = R @ Sigma_eci @ R.T

print("Sigma ECI:")
print(Sigma_eci.round(decimals=3))
print("\nSigma RSW (after rotation):")
print(Sigma_rsw.round(decimals=3))

# Verify rotation preserves PSD: all eigenvalues still non-negative
evals_eci = torch.linalg.eigvalsh(Sigma_eci)
evals_rsw = torch.linalg.eigvalsh(Sigma_rsw)
print(f"\nECI eigenvalues: {evals_eci.tolist()}")
print(f"RSW eigenvalues: {evals_rsw.tolist()}")
# Eigenvalues are preserved under rotation (rotation is orthogonal),
# so both sets should be identical up to floating-point noise

# Verify: rotation preserves total variance (trace is invariant)
print(f"\nTrace ECI: {Sigma_eci.trace().item():.4f}")
print(f"Trace RSW: {Sigma_rsw.trace().item():.4f}")

# Verify: rotation preserves determinant
print(f"\nDet ECI: {torch.linalg.det(Sigma_eci).item():.4f}")
print(f"Det RSW: {torch.linalg.det(Sigma_rsw).item():.4f}")
# Determinant is also preserved under orthogonal transformation
```

The rotation-invariance of eigenvalues, trace, and determinant is a useful sanity check: if any of these change significantly during a frame transformation, you have introduced a numerical error.

---

## Connection to Bayesian updating and the Kalman filter

The marginal/conditional formulas from Section 5 are the heart of the Kalman filter. To see this, write the Kalman setup in Gaussian terms.

**Prior**: your current belief about the state is:

\\[
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{prior}}, \Sigma_{\text{prior}})
\\]

**Observation model**: the measurement \\(\mathbf{y}\\) is a noisy linear function of the state:

\\[
\mathbf{y} = H\mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, R)
\\]

where \\(H\\) is the measurement matrix and \\(R\\) is the measurement noise covariance.

**Posterior**: after observing \\(\mathbf{y} = \mathbf{y}_\text{obs}\\), the posterior is also Gaussian:

\\[
\mathbf{x} \mid \mathbf{y}_\text{obs} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \Sigma_{\text{post}})
\\]

with the **Kalman update equations**:

\\[
K = \Sigma_{\text{prior}} H^T (H \Sigma_{\text{prior}} H^T + R)^{-1}
\\]

\\[
\boldsymbol{\mu}_{\text{post}} = \boldsymbol{\mu}_{\text{prior}} + K(\mathbf{y}_\text{obs} - H\boldsymbol{\mu}_{\text{prior}})
\\]

\\[
\Sigma_{\text{post}} = (I - KH)\Sigma_{\text{prior}}
\\]

The matrix \\(K\\) is the **Kalman gain**: it controls how much the observation shifts the estimate. Compare this to the conditional mean formula from Section 5 — they are the same update, written in terms of the cross-covariance \\(\Sigma_{\text{prior}} H^T\\) and the innovation variance \\(H\Sigma_{\text{prior}} H^T + R\\).

**Why the Gaussian is special**: it is the only continuous distribution that stays Gaussian under two operations simultaneously:

1. **Linear transformations**: \\(A\mathbf{x} + \mathbf{b}\\) is Gaussian if \\(\mathbf{x}\\) is Gaussian (shown above).
2. **Gaussian likelihoods**: multiplying a Gaussian prior by a Gaussian likelihood (as in Bayesian updating with additive Gaussian noise) gives a Gaussian posterior.

This "closed under linear-Gaussian operations" property is precisely why the Kalman filter has exact analytical solutions. If either the dynamics or the noise were non-Gaussian, you would need numerical approximations (particle filters, unscented Kalman filters, etc.) — exactly what Module 07 covers.

**Connecting to Kurt's *Bayesian Statistics the Fun Way***: Kurt emphasizes that Bayesian updating is just multiplying probabilities and renormalizing. The Kalman filter is this principle applied to Gaussians: multiply the Gaussian prior density by the Gaussian likelihood, and the result is a new Gaussian. The Kalman gain is the normalizing factor in that multiplication. No numerical integration required.

**Forward reference**: Module 07 covers the belief state representation for POMDPs. For linear-Gaussian systems, the belief state is exactly a multivariate Gaussian — a mean vector and covariance matrix. The Kalman update equations are how the belief state is updated after each observation. For nonlinear or non-Gaussian systems, particles replace the Gaussian parameters, and the Cholesky sampling from Section 6 is how the particle cloud is initialized.

---

## Key Takeaways

- **The covariance matrix encodes the full correlation structure of a multivariate distribution.** Diagonal entries are per-dimension variances; off-diagonal entries capture how dimensions move together. A valid covariance matrix is always symmetric and positive semi-definite (all eigenvalues non-negative). In SSA, the covariance matrix of an orbital state is the authoritative description of tracking uncertainty — it tells you not just how uncertain each coordinate is, but how those uncertainties are linked.

- **The Mahalanobis distance is the right measure of "how surprising is this observation."** It accounts for the shape of the uncertainty ellipsoid, unlike Euclidean distance. An observation that is 3 km away in a direction with 5 km standard deviation is closer (in Mahalanobis terms) than one 3 km away in a direction with 0.5 km standard deviation. Any data association task in SSA — matching sensor observations to catalog objects — should use Mahalanobis distance, not Euclidean distance.

- **The uncertainty ellipsoid is the geometric picture of the covariance.** Its axes are the eigenvectors of \\(\Sigma\\); its axis half-lengths are \\(\sqrt{\lambda_i}\\). The 68-95-99.7 rule for 1D Gaussians does not transfer directly to multiple dimensions: the 1-sigma ellipse in 2D contains only about 39% of probability mass. In d dimensions, containment probabilities follow the chi-squared distribution with d degrees of freedom.

- **Marginals and conditionals of a Gaussian are Gaussian.** Marginalizing out dimensions is trivially done by extracting the relevant submatrix of \\(\Sigma\\). Conditioning on observations applies the Gaussian conditioning formulas and reduces uncertainty in the remaining dimensions. This is the mathematical core of the Kalman filter: Bayesian updating with a linear observation model and Gaussian noise has a closed-form Gaussian solution.

- **Cholesky decomposition is the standard way to sample from a multivariate Gaussian.** Factor \\(\Sigma = LL^T\\), draw \\(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)\\), return \\(L\mathbf{z} + \boldsymbol{\mu}\\). The linear transformation rule — \\(\text{Cov}(L\mathbf{z}) = LL^T = \Sigma\\) — explains why this works. In Module 07, this is the exact algorithm used to initialize particle clouds around the prior belief state.

- **Linear transformations map Gaussians to Gaussians via the \\(A\Sigma A^T\\) rule.** The mean transforms linearly; the covariance "wraps around" the transformation matrix. Rotating a covariance from ECI to RSW frame, propagating uncertainty through a linear dynamics model, or projecting a 3D covariance onto the 2D conjunction plane all follow this rule. It is the single most-used formula in the computational pipeline for SSA conjunction probability.

---

{{#quiz 09-multivariate-gaussian.toml}}
