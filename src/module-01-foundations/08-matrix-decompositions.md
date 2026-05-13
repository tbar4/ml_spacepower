# Lesson 8: Matrix Decompositions

**Module:** ML and Game Theory for Space Power — M01: Foundations
**Source:** *Mathematics for Machine Learning* — Deisenroth, Faisal & Ong (2020), Chapters 4.3–4.6

---


<!-- toc -->

## Where this fits

Lessons 05–07 built up three capabilities: representing observations as vectors, applying weight matrices to those vectors, and computing gradients to train the weights. Those tools treat matrices as flat objects — grids of numbers you multiply through. But many of the most powerful ML and SSA algorithms depend on understanding the structure hidden inside a matrix: which directions does it stretch? Which directions are most important? What is the signal versus the noise? How do you sample efficiently from a multivariate distribution over orbital states?

Matrix decompositions answer these questions by factoring a matrix into simpler pieces, each piece carrying a specific geometric or statistical meaning.

Three decompositions dominate this curriculum:

**Cholesky decomposition** (Σ = L Lᵀ) appears whenever you need to sample from a multivariate Gaussian or solve linear systems involving a covariance matrix. In Module 07, the particle filter draws samples from a state-uncertainty distribution at every time step — it uses Cholesky to do so efficiently and numerically stably.

**Eigendecomposition** (A = Q Λ Qᵀ) reveals how a square matrix stretches space along its natural axes. The eigenvalues of a value-iteration update operator in Module 03 determine whether repeated application converges, diverges, or cycles. The eigenvectors of a covariance matrix are the principal components — the directions of maximum variance in the error distribution of a tracked object.

**Singular Value Decomposition** (A = U Σ Vᵀ) is the most general of the three: it works for any matrix, rectangular or square. It is the engine behind Principal Component Analysis (PCA) for compressing high-dimensional sensor data, the pseudoinverse for least-squares orbit determination from noisy radar measurements, and low-rank approximation for compressing a large catalog of orbital element time series. SVD is sometimes called the "fundamental theorem of linear algebra" — once you understand it, many seemingly unrelated algorithms reveal themselves as special cases.

This lesson builds the geometric intuition and PyTorch mechanics for all three, with SSA examples throughout.

---

## Eigendecomposition

### The factorization

For a square matrix A ∈ ℝⁿˣⁿ that has n linearly independent eigenvectors, the eigendecomposition is:

\[ A = Q \Lambda Q^{-1} \]

**Decoding:**

**Q**: An n × n matrix whose columns are the eigenvectors of A. Column i of Q is the eigenvector corresponding to the i-th eigenvalue.

**Λ** (capital Lambda): A diagonal matrix. The diagonal entry Λᵢᵢ is the i-th eigenvalue λᵢ. Off-diagonal entries are zero.

**Q⁻¹**: The inverse of Q. For general (non-symmetric) matrices, Q⁻¹ is distinct from Qᵀ.

**Reading in English**: "A can be written as: rotate to eigenvector axes, scale each axis by the corresponding eigenvalue, then rotate back." The decomposition exposes A's stretching behavior along its natural directions.

### The symmetric case: covariance matrices

Symmetric matrices — where A = Aᵀ — have two additional guarantees:
1. All eigenvalues are real (not complex).
2. The eigenvectors are orthogonal to each other.

When eigenvectors are orthogonal and unit-length, Q is an orthogonal matrix: Q⁻¹ = Qᵀ. The decomposition simplifies to:

\[ \Sigma = Q \Lambda Q^T \]

This is exactly the form of a covariance matrix. In SSA, a 3 × 3 position-uncertainty covariance matrix Σ (entries in km²) encodes how uncertain we are about where an RSO (Resident Space Object) actually is. Its eigendecomposition reveals the principal axes and magnitudes of that uncertainty ellipsoid:

- The eigenvectors of Σ point along the axes of the uncertainty ellipsoid.
- The eigenvalues are the variances along those axes (large eigenvalue = large uncertainty in that direction).

```python
import torch

torch.manual_seed(0)

# 3x3 orbital position uncertainty covariance matrix (units: km^2)
# This represents a realistic LEO track where along-track error dominates
Sigma = torch.tensor([
    [25.0,  8.0,  2.0],   # x-x, x-y, x-z
    [ 8.0, 16.0,  1.5],   # y-x, y-y, y-z
    [ 2.0,  1.5,  4.0],   # z-x, z-y, z-z
], dtype=torch.float64)

# eigh is for symmetric (Hermitian) matrices — more stable than eig
eigenvalues, Q = torch.linalg.eigh(Sigma)

print("Eigenvalues (variances along principal axes, km^2):")
print(eigenvalues)
# eigh returns eigenvalues in ascending order

print("\nEigenvectors (columns are principal axes):")
print(Q)

# Verify Q is orthogonal: Q^T @ Q should be identity
QtQ = Q.T @ Q
print(f"\nQ^T Q close to I: {torch.allclose(QtQ, torch.eye(3, dtype=torch.float64), atol=1e-10)}")

# Verify the decomposition: Q Λ Q^T = Σ
Lambda = torch.diag(eigenvalues)
Sigma_reconstructed = Q @ Lambda @ Q.T
print(f"Q Λ Q^T close to Σ: {torch.allclose(Sigma_reconstructed, Sigma, atol=1e-10)}")

# Physical interpretation: square root of largest eigenvalue
# gives the standard deviation along the most uncertain direction (km)
print(f"\nLargest uncertainty std dev: {eigenvalues[-1].sqrt().item():.3f} km")
print(f"Smallest uncertainty std dev: {eigenvalues[0].sqrt().item():.3f} km")
```

### Why eigendecomposition requires square matrices

The formula A = Q Λ Q⁻¹ requires Q to be invertible, which requires Q to be square. If A is m × n with m ≠ n, you cannot form a complete set of eigenvectors — the shape mismatch breaks the decomposition. This is the core limitation that motivates SVD, which generalizes the idea to any matrix by using two separate orthogonal matrices (one for each dimension).

---

## Cholesky decomposition

### The factorization

For any symmetric positive-definite (SPD) matrix Σ, there exists a unique lower-triangular matrix L such that:

\[ \Sigma = L L^T \]

**Decoding:**

**L**: A lower-triangular matrix — all entries above the diagonal are zero. The diagonal entries of L are strictly positive.

**Lᵀ**: The transpose of L, which is upper-triangular.

**Σ = L Lᵀ**: The outer product structure means every vector xᵀ Σ x = xᵀ L Lᵀ x = ‖Lᵀx‖² ≥ 0. Positive definiteness is built in.

**Reading in English**: "Cholesky is the matrix square root. L is the unique lower-triangular matrix whose product with its own transpose recovers Σ." In the same way that any positive number c can be written as √c × √c, any SPD matrix can be written as L × Lᵀ.

### What "positive definite" means physically

A covariance matrix Σ describes uncertainty over a vector of quantities. The condition xᵀ Σ x > 0 for all nonzero x means that variance is always positive when you project onto any direction — there is no direction of zero uncertainty. If a sensor glitch produces a matrix that is only positive semi-definite (some zero eigenvalues), Cholesky will fail at that zero-variance direction. That failure is informative: it signals a degenerate covariance that must be fixed before downstream operations proceed.

### An SSA covariance example

Consider the 3 × 3 position-uncertainty covariance matrix Σ from above. The uncertainty is described in Cartesian (x, y, z) coordinates, with cross-correlations because along-track, cross-track, and radial errors are not independent.

```python
import torch

torch.manual_seed(0)

Sigma = torch.tensor([
    [25.0,  8.0,  2.0],
    [ 8.0, 16.0,  1.5],
    [ 2.0,  1.5,  4.0],
], dtype=torch.float64)

# Cholesky decomposition: Sigma = L @ L^T
L = torch.linalg.cholesky(Sigma)

print("Lower-triangular factor L:")
print(L)

# Verify reconstruction
Sigma_reconstructed = L @ L.T
print(f"\nL @ L^T close to Sigma: {torch.allclose(Sigma_reconstructed, Sigma, atol=1e-10)}")

# L is the "square root" of uncertainty: its diagonal entries (km) are
# related to the standard deviations of the marginal distributions
print("\nDiagonal of L (km):")
print(L.diagonal())
```

### Why Cholesky is essential in SSA

Three distinct use cases make Cholesky indispensable:

**1. Sampling from a multivariate Gaussian.** Given a mean vector μ and covariance Σ, you want to draw samples from N(μ, Σ). The recipe is:
1. Draw z ~ N(0, I) — a standard normal vector (independent components, unit variance).
2. Compute x = L z + μ.

Then x ~ N(μ, Σ). This works because Cov(Lz) = L Cov(z) Lᵀ = L I Lᵀ = L Lᵀ = Σ.

```python
import torch

torch.manual_seed(42)

mu = torch.tensor([0.0, 500.0, 6871.0], dtype=torch.float64)   # mean position (km)
Sigma = torch.tensor([
    [25.0,  8.0,  2.0],
    [ 8.0, 16.0,  1.5],
    [ 2.0,  1.5,  4.0],
], dtype=torch.float64)

L = torch.linalg.cholesky(Sigma)

# Draw 1000 samples from N(mu, Sigma)
n_samples = 1000
z = torch.randn(3, n_samples, dtype=torch.float64)   # shape (3, 1000)
samples = L @ z + mu.unsqueeze(1)                     # shape (3, 1000)

# Verify: sample covariance should recover Sigma
# Center the samples
centered = samples - samples.mean(dim=1, keepdim=True)
sample_cov = (centered @ centered.T) / (n_samples - 1)

print("True covariance Sigma:")
print(Sigma)
print("\nSample covariance from 1000 draws:")
print(sample_cov.round(decimals=1))
# With 1000 samples, the sample covariance should approximate Sigma reasonably well
```

**2. Solving linear systems without inverting Σ.** Computing Σ⁻¹ b directly is numerically unstable and expensive. Instead, factor Σ = L Lᵀ and solve two triangular systems:
- Forward substitution: solve L y = b for y.
- Back substitution: solve Lᵀ x = y for x.

Triangular systems are solved in O(n²) operations rather than the O(n³) of full inversion, and they accumulate less numerical error.

**3. Validating a covariance matrix.** If `torch.linalg.cholesky` raises an error, the matrix is not positive definite — it is not a valid covariance matrix. This check is built into Cholesky, at no extra cost.

### Common pitfall: numerical jitter

A matrix that is theoretically positive definite can fail Cholesky in floating-point arithmetic. Floating-point round-off can make eigenvalues appear slightly negative. The standard fix is to add a small multiple of the identity matrix before factoring:

```python
import torch

def safe_cholesky(Sigma: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """
    Numerically stable Cholesky that adds jitter if needed.
    Common in Gaussian process and Kalman filter implementations.
    """
    n = Sigma.shape[0]
    try:
        L = torch.linalg.cholesky(Sigma)
        return L
    except torch.linalg.LinAlgError:
        # Add diagonal jitter and retry
        Sigma_jittered = Sigma + jitter * torch.eye(n, dtype=Sigma.dtype)
        return torch.linalg.cholesky(Sigma_jittered)

# Example: a near-singular covariance matrix
Sigma_tricky = torch.tensor([
    [1.0, 0.999],
    [0.999, 1.0],
], dtype=torch.float64)

L = safe_cholesky(Sigma_tricky)
print("Cholesky succeeded with jitter-protected function")
print(L)
```

The jitter ε effectively says "add ε variance in all directions," which moves eigenvalues away from zero without meaningfully changing the distribution for ε much smaller than the true eigenvalues.

---

## Singular Value Decomposition

### The theorem

Any matrix A ∈ ℝᵐˣⁿ — regardless of shape — can be written as:

\[ A = U \Sigma V^T \]

where:

- **U ∈ ℝᵐˣᵐ**: An orthogonal matrix whose columns are the **left singular vectors**.
- **Σ ∈ ℝᵐˣⁿ**: A "diagonal" matrix (with the diagonal possibly being rectangular) whose nonzero entries σ₁ ≥ σ₂ ≥ ... ≥ 0 are the **singular values**, ordered from largest to smallest.
- **Vᵀ ∈ ℝⁿˣⁿ**: The transpose of an orthogonal matrix V whose columns are the **right singular vectors**.

Note: the Σ in A = U Σ Vᵀ is the singular value matrix (not a covariance matrix, despite the same symbol). This overloading of notation is standard and context-disambiguates them.

**Decoding each piece:**

**Vᵀ rotates the input.** Because V is orthogonal, Vᵀ is a pure rotation (or reflection) in the n-dimensional input space. It rotates the standard basis directions into the "natural input directions" of A — the right singular vectors.

**Σ scales (and reshapes).** After the rotation, Σ scales each component: the first component by σ₁, the second by σ₂, and so on. If m < n, extra input directions are dropped (compressed). If m > n, extra output dimensions receive zero contribution.

**U rotates the output.** U is a pure rotation in the m-dimensional output space. It rotates the scaled components back into the "natural output directions" — the left singular vectors.

**Reading in English**: "Any linear transformation can be broken into three steps: a rotation of the input space, a rescaling along each axis, and a rotation of the output space." SVD exposes the three pure components of any linear map.

### Why SVD generalizes eigendecomposition

Eigendecomposition requires A to be square and to have n independent eigenvectors. SVD has no such requirement. Any m × n matrix — with m ≠ n, or with rank less than min(m, n) — has an SVD. This is why SVD is called the "fundamental theorem of linear algebra" in some texts: it is the most complete factorization available.

When A is symmetric positive definite, its SVD and eigendecomposition coincide: U = V = Q and Σ = Λ (singular values equal eigenvalues). SVD is the natural generalization.

### PyTorch implementation

```python
import torch

torch.manual_seed(7)

# Synthetic 5x3 sensor data matrix
# 5 rows = observations from 5 ground stations
# 3 cols = signal levels for 3 orbital slots
A = torch.tensor([
    [3.2, 1.1, 0.4],
    [2.9, 1.0, 0.5],
    [0.3, 2.8, 1.9],
    [0.4, 2.6, 2.0],
    [1.5, 1.8, 1.2],
], dtype=torch.float64)

# Full SVD: U (5x5), S (singular values, length 3), Vh (3x3, = V^T)
U, S, Vh = torch.linalg.svd(A, full_matrices=True)

print(f"A shape:  {A.shape}")
print(f"U shape:  {U.shape}")    # (5, 5) - full left singular vectors
print(f"S shape:  {S.shape}")    # (3,)   - min(5, 3) singular values
print(f"Vh shape: {Vh.shape}")   # (3, 3) - full right singular vectors (transposed)

print(f"\nSingular values: {S.tolist()}")

# Reconstruct A from U, S, Vh
# Need to form the (5, 3) Sigma matrix from the (3,) vector S
# U_k = U[:, :3], S_k = diag(S), Vh_k = Vh[:3, :]
A_reconstructed = U[:, :3] @ torch.diag(S) @ Vh[:3, :]
print(f"\nReconstructed close to A: {torch.allclose(A_reconstructed, A, atol=1e-10)}")

# Verify orthogonality
print(f"U^T U ≈ I: {torch.allclose(U.T @ U, torch.eye(5, dtype=torch.float64), atol=1e-10)}")
print(f"Vh Vh^T ≈ I: {torch.allclose(Vh @ Vh.T, torch.eye(3, dtype=torch.float64), atol=1e-10)}")
```

---

## What singular values tell you

### Singular values as importance scores

The singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 measure how much "energy" or "information" flows through each dimension of the transformation.

- **σ₁** is the largest scaling factor: the most important direction, carrying the most variance from input to output.
- **σᵣ** (the last nonzero singular value) defines the matrix rank: rank r means r independent directions of information.
- **σ_i ≈ 0** for large i: those directions carry negligible information and are dominated by noise.

### The condition number

The ratio σ₁ / σₙ is the **condition number** of the matrix. A large condition number means the matrix nearly collapses some directions to zero while expanding others enormously. Small perturbations in those near-zero directions get amplified in the output — the system is ill-conditioned and sensitive to measurement noise.

In orbit determination, an ill-conditioned design matrix (high condition number) means that small radar measurement errors produce large errors in the estimated orbital elements. Understanding the condition number tells the analyst which parameters are well-determined and which are poorly constrained by the available observations.

### SSA example: ground station measurements

Imagine 5 ground stations each observing 3 orbital slots. A large first singular value means one dominant pattern explains most of the variation — perhaps all stations see the same orbital behavior in all slots (a space weather event affecting all RSOs). Small later singular values indicate correlated noise or low-information secondary patterns.

```python
import torch

torch.manual_seed(3)

# Synthetic 5x3 measurement matrix
# 5 ground stations, 3 orbital slots
# True signal: a dominant shared pattern plus noise
true_signal = torch.outer(
    torch.tensor([1.0, 0.95, 0.3, 0.28, 0.6], dtype=torch.float64),   # station sensitivity
    torch.tensor([4.0, 3.0, 1.5], dtype=torch.float64)                   # slot brightness
)
noise = 0.3 * torch.randn(5, 3, dtype=torch.float64)
M = true_signal + noise

U, S, Vh = torch.linalg.svd(M, full_matrices=False)

print("Singular values:")
for i, s in enumerate(S):
    print(f"  sigma_{i+1} = {s.item():.4f}")

# Condition number
cond = S[0] / S[-1]
print(f"\nCondition number sigma_1 / sigma_r: {cond.item():.2f}")

# Fraction of variance explained by each singular value
variance_fractions = S**2 / (S**2).sum()
print("\nVariance fraction per singular value:")
for i, frac in enumerate(variance_fractions):
    cumulative = variance_fractions[:i+1].sum()
    print(f"  sigma_{i+1}: {frac.item():.3f}  (cumulative: {cumulative.item():.3f})")

# Effective rank: count singular values above a threshold
threshold = 1e-2 * S[0]  # 1% of largest singular value
effective_rank = (S > threshold).sum().item()
print(f"\nEffective rank (threshold = 1% of sigma_1): {effective_rank}")
```

---

## Low-rank approximation

### The Eckart-Young theorem

The **Eckart-Young theorem** states that the best rank-k approximation of A in the Frobenius norm is:

\[ A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T \]

where Uₖ keeps only the first k columns of U, Σₖ is the k × k upper-left block of Σ, and Vₖᵀ keeps only the first k rows of Vᵀ.

**Decoding:**

**σᵢ uᵢ vᵢᵀ**: A rank-1 matrix — the outer product of the i-th left and right singular vectors, scaled by σᵢ. Each such term is a single "pattern": left singular vector uᵢ describes which output dimensions are active in this pattern, right singular vector vᵢ describes which input dimensions activate it, and σᵢ is the strength.

**The sum over k terms**: The rank-k approximation Aₖ retains the k strongest patterns and discards the rest.

**"Best" in Frobenius norm**: ‖A - Aₖ‖²_F = σ²_{k+1} + σ²_{k+2} + ... The reconstruction error equals the sum of squares of the discarded singular values. No other rank-k matrix does better.

### SSA application: compressing orbital element time series

Suppose you have a 100 × 20 matrix: 100 time steps of measurements for 20 RSOs, each row recording six orbital elements. Storing and transmitting this full matrix requires 2000 numbers. A rank-k approximation requires storing only k left vectors (100 entries each), k singular values, and k right vectors (20 entries each) — a total of k × (100 + 1 + 20) numbers. For k = 3, that is 363 numbers instead of 2000, a compression ratio of about 5.5×, while capturing most of the variance.

```python
import torch

torch.manual_seed(11)

# Simulate 100 time steps x 20 RSO orbital element measurements
# True data has a low-rank structure: a few shared orbital patterns
n_times, n_rsos = 100, 20
rank_true = 4

# Low-rank ground truth + noise
U_true = torch.randn(n_times, rank_true, dtype=torch.float64)
V_true = torch.randn(rank_true, n_rsos, dtype=torch.float64)
A = U_true @ V_true + 0.5 * torch.randn(n_times, n_rsos, dtype=torch.float64)

# Full SVD
U, S, Vh = torch.linalg.svd(A, full_matrices=False)

print("Top 10 singular values:")
print(S[:10].round(decimals=2).tolist())

# Frobenius norm of original matrix
A_norm = torch.linalg.norm(A, 'fro').item()

# Reconstruct with increasing rank and measure error
print(f"\n{'Rank':>5} | {'Recon error (Frob)':>20} | {'Error / ||A||':>15} | {'Storage ratio':>15}")
print("-" * 65)

for k in [1, 3, 5, 10, 20]:
    # Rank-k approximation
    A_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    error = torch.linalg.norm(A - A_k, 'fro').item()
    relative_error = error / A_norm
    # Original storage: n_times * n_rsos entries
    # Rank-k storage: k * (n_times + 1 + n_rsos) entries
    storage_original = n_times * n_rsos
    storage_rank_k   = k * (n_times + 1 + n_rsos)
    storage_ratio    = storage_rank_k / storage_original
    print(f"{k:>5} | {error:>20.4f} | {relative_error:>15.4f} | {storage_ratio:>15.3f}")

# Compute the Eckart-Young bound: ||A - A_k||_F = sqrt(sum of sigma_{k+1}^2 ... sigma_r^2)
k = 5
ey_bound = S[k:].pow(2).sum().sqrt().item()
actual_error = torch.linalg.norm(A - U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :], 'fro').item()
print(f"\nEckart-Young bound for k=5: {ey_bound:.4f}")
print(f"Actual reconstruction error: {actual_error:.4f}")
print(f"Match: {abs(ey_bound - actual_error) < 1e-8}")
```

The table shows a characteristic pattern: for data with true rank-4 structure, the error drops sharply through k = 4 and then falls much more slowly as you add higher components that capture only noise.

---

## SVD for the pseudoinverse and linear regression

### The pseudoinverse

For a general (possibly rectangular) matrix A ∈ ℝᵐˣⁿ, the **Moore-Penrose pseudoinverse** is:

\[ A^+ = V \Sigma^+ U^T \]

where Σ⁺ is obtained from Σ by taking the reciprocal of each nonzero singular value and leaving zero entries as zero.

**Decoding:**

**Σ⁺**: If Σ has singular values (σ₁, σ₂, ..., σᵣ, 0, ..., 0), then Σ⁺ has entries (1/σ₁, 1/σ₂, ..., 1/σᵣ, 0, ..., 0). Directions with zero singular value (rank-deficient directions) are not inverted — projecting onto them would amplify noise infinitely.

**A⁺ b**: The minimum-norm least-squares solution to the system Ax ≈ b. When A is tall (more equations than unknowns, m > n) and the system is overdetermined, A⁺ b gives the solution that minimizes ‖Ax - b‖². When A is wide (more unknowns than equations), it gives the minimum-norm solution.

### Connection to orbit determination

Orbit determination from radar measurements is a classic overdetermined system. A spacecraft is observed at multiple times, producing measurements of range, range-rate, and angles. Each measurement contributes one or more equations in a linearized system A δx ≈ δz, where δx is the correction to the orbital state estimate and δz is the measurement residual. The system typically has many more measurements than the 6 state variables — it is overdetermined and solved by least-squares via the pseudoinverse or its equivalent.

```python
import torch

torch.manual_seed(5)

# Overdetermined system: 12 radar measurements, 6 orbital state parameters
n_measurements = 12
n_params       = 6

# Design matrix A (measurement Jacobians — how each measurement depends on each state)
A_well = torch.randn(n_measurements, n_params, dtype=torch.float64)

# True state perturbation
x_true = torch.tensor([0.5, -0.3, 0.2, 0.01, -0.02, 0.005], dtype=torch.float64)

# Noisy measurements
noise = 0.1 * torch.randn(n_measurements, dtype=torch.float64)
b = A_well @ x_true + noise

# --- Method 1: least-squares via torch.linalg.lstsq ---
result = torch.linalg.lstsq(A_well, b.unsqueeze(1))
x_lstsq = result.solution.squeeze()
print("Least-squares solution (torch.linalg.lstsq):")
print(x_lstsq.tolist())

# --- Method 2: manual pseudoinverse via SVD ---
U, S, Vh = torch.linalg.svd(A_well, full_matrices=False)
# S has shape (n_params,) since n_params < n_measurements
S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
A_pinv = Vh.T @ torch.diag(S_inv) @ U.T    # V Sigma^+ U^T
x_pinv = A_pinv @ b
print("\nLeast-squares solution (manual pseudoinverse via SVD):")
print(x_pinv.tolist())

print(f"\nTwo methods agree: {torch.allclose(x_lstsq, x_pinv, atol=1e-8)}")
print(f"Residual norm: {torch.linalg.norm(A_well @ x_pinv - b).item():.6f}")

# --- Ill-conditioned system: poorly observed geometry ---
# Make one column nearly parallel to another (two parameters nearly unobservable)
A_ill = A_well.clone()
A_ill[:, 1] = A_ill[:, 0] + 1e-3 * torch.randn(n_measurements, dtype=torch.float64)

U_ill, S_ill, Vh_ill = torch.linalg.svd(A_ill, full_matrices=False)
print(f"\nWell-conditioned system — condition number: {(S[0] / S[-1]).item():.1f}")
print(f"Ill-conditioned system — condition number:  {(S_ill[0] / S_ill[-1]).item():.1f}")
print("High condition number: solution is sensitive to measurement noise")
```

The condition number comparison shows the practical risk: an ill-conditioned geometry (two nearly-parallel baselines, or a ground station network that all lie on the same great circle) can produce a condition number thousands of times larger than a well-designed network, meaning small measurement errors become large state estimate errors.

---

## Key Takeaways

* **Cholesky (Σ = L Lᵀ) is the workhorse for Gaussian operations.** For any symmetric positive-definite covariance matrix, Cholesky provides the unique lower-triangular "square root" needed to sample from N(μ, Σ) (via x = Lz + μ with z ~ N(0, I)), to solve linear systems without inverting Σ, and to validate that a matrix is a legal covariance. Add a small jitter term (+ ε I) to protect against floating-point precision failures on near-singular matrices.

* **SVD (A = U Σ Vᵀ) works for any matrix.** Unlike eigendecomposition, SVD imposes no shape or symmetry requirements. It decomposes any linear transformation into three geometrically pure steps: a rotation of inputs (Vᵀ), a scaling along independent axes (Σ), and a rotation of outputs (U). It is the most complete factorization available and a foundation for a large fraction of practical ML algorithms.

* **Singular values are importance scores for the transformation.** The i-th singular value σᵢ measures how much "energy" or variance flows through the i-th independent direction of the matrix. Large singular values correspond to signal; small singular values correspond to noise or near-redundant dimensions. The ratio σ₁/σₙ (the condition number) measures how numerically sensitive the system is to perturbations.

* **The Eckart-Young theorem justifies low-rank approximation.** Keeping only the top-k singular values and vectors produces the best possible rank-k approximation of A in the Frobenius norm, with reconstruction error equal to √(σ²_{k+1} + ... + σ²_r). This justifies PCA for sensor data compression, low-rank factorization of policy value tables, and compact representations of orbital element catalogs.

* **The pseudoinverse (A⁺ = V Σ⁺ Uᵀ) solves overdetermined systems.** By inverting only the nonzero singular values, A⁺ gives the minimum-norm least-squares solution to Ax ≈ b. This is the correct tool for orbit determination from many measurements, for fitting linear observation models, and for any system with more equations than unknowns. Use `torch.linalg.lstsq` in practice; understanding the SVD derivation clarifies what it is doing and when it will fail (high condition number).

* **These decompositions underpin the algorithms in every subsequent module.** Eigendecomposition controls convergence of value iteration (Module 03) and reveals principal error axes in orbit covariances. Cholesky enables multivariate Gaussian sampling in the particle filter (Module 07). SVD powers PCA for high-dimensional sensor data, low-rank approximations in neural network analysis, and the pseudoinverse in Kalman filter measurement updates (Module 07). Recognizing which decomposition an algorithm relies on is the key to understanding why it works — and diagnosing when it fails.

---

{{#quiz 08-matrix-decompositions.toml}}
