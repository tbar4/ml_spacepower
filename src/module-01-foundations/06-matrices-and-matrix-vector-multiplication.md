# Lesson 6: Matrices and Matrix-Vector Multiplication

**Module:** ML Foundations — M01: Mathematical Foundations
**Source:** *Math for Deep Learning* — Ronald T. Kneusel, Ch. 6–7 (Matrices and Matrix Operations); *Bayesian Statistics the Fun Way* — Will Kurt, Ch. 14 (covariance and correlation); PyTorch documentation

---


<!-- toc -->

## Where this fits

In lesson 5, you saw that the dot product of a weight vector and an observation vector scores how well the observation matches that weight vector's "interest." A neural network layer does that simultaneously for many weight vectors at once, producing a score for each one. That simultaneous scoring is matrix-vector multiplication. Once you understand what \\(W\mathbf{x} + \mathbf{b}\\) computes, you know what a neural network layer does. Every modern deep learning architecture, from the policy networks in AlphaZero to the value networks in deep CFR, is built by stacking this operation repeatedly with nonlinearities in between.

---

## What is a matrix?

A **matrix** is a rectangular grid of numbers arranged in rows and columns.

When we say a matrix is "m by n" (written m × n), we mean it has:
- **m rows** (horizontal lines of numbers)
- **n columns** (vertical lines of numbers)

Here is a 3 × 4 matrix (3 rows, 4 columns):

\\[
W = \begin{pmatrix}
1 & 0 & -1 & 2 \\\\
0 & 1 &  1 & 0 \\\\
-1 & 1 &  0 & 1
\end{pmatrix}
\\]

Each row is a list of 4 numbers. There are 3 such rows. In total, the matrix contains 3 × 4 = 12 numbers.

We refer to individual entries using row and column indices. The notation \\(W_{ij}\\) means "the entry in row \\(i\\), column \\(j\\)." Row index first, column index second.

From the matrix above:
- \\(W_{11} = 1\\) (row 1, column 1)
- \\(W_{13} = -1\\) (row 1, column 3)
- \\(W_{23} = 1\\) (row 2, column 3)
- \\(W_{31} = -1\\) (row 3, column 1)

**The key insight**: each row of a matrix is a vector. A 3 × 4 matrix contains three row-vectors, each of length 4. Matrix-vector multiplication uses each of those row vectors to compute a dot product.

---

## Matrix-vector multiplication: the core idea

Suppose we have a weight matrix \\(W\\) with shape m × n and an input vector \\(\mathbf{x}\\) of length n. The matrix-vector product \\(W\mathbf{x}\\) produces an output vector \\(\mathbf{y}\\) of length m.

**The rule**: each entry of the output \\(\mathbf{y}\\) is the dot product of one row of \\(W\\) with the input \\(\mathbf{x}\\).

Specifically:
- \\(y_1\\) = (row 1 of W) · x
- \\(y_2\\) = (row 2 of W) · x
- \\(y_m\\) = (row m of W) · x

In formula form:

\\[ y_i = \sum_{j=1}^{n} W_{ij} \cdot x_j \\]

**Decoding:**

**\\(y_i\\)**: The i-th component of the output vector.

**\\(\sum_{j=1}^{n}\\)**: Sum over j from 1 to n. This loops through the columns.

**\\(W_{ij}\\)**: The entry in row i, column j of the weight matrix.

**\\(x_j\\)**: The j-th component of the input vector.

**\\(W_{ij} \cdot x_j\\)**: Multiply the matrix entry by the input component.

**Reading in English**: "The i-th output is computed by taking each entry in row i of W, multiplying it by the corresponding entry in x, and adding all those products up." That is a dot product.

---

## Step-by-step example

Let us work through a complete example by hand.

**Scenario**: You have a sensor processing pipeline. Your sensor returns a 4-dimensional observation:

\\[ \mathbf{x} = \begin{pmatrix} 0.8 \\ 0.2 \\ 0.1 \\ 0.6 \end{pmatrix} \\]

These represent [conjunction_risk, debris_density, solar_activity, comms_window].

You want to compute scores for 3 possible operational responses. Your scoring matrix (one row per response, one column per observation feature) is:

\\[ W = \begin{pmatrix}
1.0 & 0.5 & 0.0 & 0.2 \\\\
0.1 & 0.0 & 0.0 & 1.0 \\\\
0.3 & 0.8 & 0.2 & 0.1
\end{pmatrix} \\]

Row 1 weights for Response A (conjunction-focused).
Row 2 weights for Response B (comms-focused).
Row 3 weights for Response C (debris-monitoring).

**Computing y = Wx:**

**Output y₁** (score for Response A):
Dot product of row 1 with x:

| Row 1 entry | × | x entry | = | Product |
|-------------|---|---------|---|---------|
| 1.0 (col 1) | × | 0.8 (x₁) | = | 0.80 |
| 0.5 (col 2) | × | 0.2 (x₂) | = | 0.10 |
| 0.0 (col 3) | × | 0.1 (x₃) | = | 0.00 |
| 0.2 (col 4) | × | 0.6 (x₄) | = | 0.12 |
| **Sum** | | | | **1.02** |

**Output y₂** (score for Response B):
Dot product of row 2 with x:

| Row 2 entry | × | x entry | = | Product |
|-------------|---|---------|---|---------|
| 0.1 (col 1) | × | 0.8 (x₁) | = | 0.08 |
| 0.0 (col 2) | × | 0.2 (x₂) | = | 0.00 |
| 0.0 (col 3) | × | 0.1 (x₃) | = | 0.00 |
| 1.0 (col 4) | × | 0.6 (x₄) | = | 0.60 |
| **Sum** | | | | **0.68** |

**Output y₃** (score for Response C):
Dot product of row 3 with x:

| Row 3 entry | × | x entry | = | Product |
|-------------|---|---------|---|---------|
| 0.3 (col 1) | × | 0.8 (x₁) | = | 0.24 |
| 0.8 (col 2) | × | 0.2 (x₂) | = | 0.16 |
| 0.2 (col 3) | × | 0.1 (x₃) | = | 0.02 |
| 0.1 (col 4) | × | 0.6 (x₄) | = | 0.06 |
| **Sum** | | | | **0.48** |

**Result**:

\\[ \mathbf{y} = W\mathbf{x} = \begin{pmatrix} 1.02 \\ 0.68 \\ 0.48 \end{pmatrix} \\]

Response A scores highest (1.02), Response B is second (0.68), Response C is lowest (0.48). Given the high conjunction risk (0.8) in the input, the conjunction-focused response dominates. Makes operational sense.

In code:

```python
import torch

W = torch.tensor([
    [1.0, 0.5, 0.0, 0.2],
    [0.1, 0.0, 0.0, 1.0],
    [0.3, 0.8, 0.2, 0.1]
])

x = torch.tensor([0.8, 0.2, 0.1, 0.6])

y = W @ x  # @ is the matrix-vector multiplication operator in Python
print(y.tolist())  # [1.02, 0.68, 0.48]

# Verify by computing row 1's dot product manually
row1_dot = torch.dot(W[0], x)
print(f"Row 1 dot product: {row1_dot.item()}")  # 1.02
```

The `@` operator is Python's matrix multiplication operator. For a matrix times a vector, it does exactly the row-by-row dot products you just computed by hand.

---

## Shape rules: why dimensions must match

A matrix-vector multiplication \\(W\mathbf{x}\\) is only defined when the number of columns in \\(W\\) matches the length of \\(\mathbf{x}\\).

- If \\(W\\) is m × n and \\(\mathbf{x}\\) has length n, the result \\(\mathbf{y} = W\mathbf{x}\\) has length m.
- The "inner" dimension (columns of W, length of x) must match.
- The "outer" dimensions (rows of W, length of output) determine the result's shape.

In our example: W is 3 × 4, x has length 4. The 4s match (column dimension of W equals length of x). The result has length 3 (the number of rows).

This matters practically: if you have an observation of length 4 and want to compute scores for 3 responses, your weight matrix must be 3 × 4. Not 4 × 3. Not 3 × 3. The dimensions encode the data flow.

---

## Transpose

The **transpose** of a matrix swaps its rows and columns. If \\(A\\) is m × n, then its transpose \\(A^T\\) is n × m. The element that was at row i, column j moves to row j, column i:

\\[ (A^T)_{ij} = A_{ji} \\]

**Decoding:** The superscript T on a matrix (or sometimes a prime symbol, or `.T` in code) means "transpose." The key rule is that the shape flips: m × n becomes n × m.

A concrete example:

\\[
A = \begin{pmatrix} 1 & 2 & 3 \\\\ 4 & 5 & 6 \end{pmatrix} \quad \text{shape: } 2 \times 3
\qquad
A^T = \begin{pmatrix} 1 & 4 \\\\ 2 & 5 \\\\ 3 & 6 \end{pmatrix} \quad \text{shape: } 3 \times 2
\\]

### When you need the transpose

**Backpropagation** — During the backward pass through a linear layer \\(y = Wx\\), the gradient of the loss with respect to x is \\(W^T \delta\\), where \\(\delta\\) is the gradient flowing back. The transpose appears because the backward pass reverses the direction of information flow. PyTorch handles this automatically via autograd.

**Attention in transformers** — The scaled dot-product attention computes \\(QK^T / \sqrt{d}\\). The transpose of K turns the columns of K into the rows that are used for dot products against each query. This gives a similarity score for every (query, key) pair in a single matrix multiply.

**Symmetric matrices** — A matrix where \\(A = A^T\\) is called symmetric. Covariance matrices are always symmetric: the covariance of x with y equals the covariance of y with x. This symmetry has important computational consequences (symmetric eigendecompositions, positive semidefinite guarantees).

```python
import torch

W = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print(f"W shape:    {W.shape}")     # torch.Size([2, 3])

# Two equivalent ways to transpose
W_T_verbose = torch.transpose(W, 0, 1)   # explicit: swap dim 0 and dim 1
W_T_short   = W.T                         # shorthand property

print(f"W.T shape:  {W_T_short.shape}")   # torch.Size([3, 2])
print(W_T_short)

# Symmetric matrix example: covariance of a 3D orbital state estimate
# A covariance matrix C satisfies C = C.T
# Simple example: diagonal covariance (uncorrelated)
cov = torch.tensor([
    [4.0, 0.5, 0.0],
    [0.5, 2.0, 0.1],
    [0.0, 0.1, 1.0]
])
print(f"\nCovariance matrix is symmetric: {torch.allclose(cov, cov.T)}")  # True

# Verify the transpose identity: (AB)^T = B^T A^T
A = torch.randn(3, 4)
B = torch.randn(4, 5)
lhs = (A @ B).T
rhs = B.T @ A.T
print(f"(AB)^T = B^T A^T: {torch.allclose(lhs, rhs)}")  # True
```

---

## Matrix-matrix multiplication

So far we have multiplied a matrix by a vector. We can also multiply two matrices together.

If \\(A\\) is m × k and \\(B\\) is k × n, then \\(C = AB\\) is m × n.

**The rule**: each entry \\(C_{ij}\\) equals the dot product of row i of A with column j of B.

\\[ C_{ij} = \sum_{l=1}^{k} A_{il} \cdot B_{lj} \\]

**Decoding:**

**\\(C_{ij}\\)**: Entry at row i, column j of the output matrix.

**\\(\sum_{l=1}^{k}\\)**: Sum over the shared inner dimension of length k.

**\\(A_{il}\\)**: Entry in row i, column l of A (one row of A, scanned left to right).

**\\(B_{lj}\\)**: Entry in row l, column j of B (one column of B, scanned top to bottom).

**In plain English**: "Row i of A, dotted with column j of B, gives entry [i,j] of C." The dimensions must be compatible: the number of columns in A must equal the number of rows in B. The output has the number of rows from A and the number of columns from B.

### Worked example: two-layer scoring

Suppose you have 2 observations, each characterized by 3 features — threat scores from a preliminary risk assessment:

\\[
X = \begin{pmatrix} 0.9 & 0.4 & 0.1 \\\\ 0.2 & 0.8 & 0.7 \end{pmatrix} \quad \text{(2 observations × 3 features)}
\\]

You apply a 3 × 2 output layer that combines those features into 2 final response scores:

\\[
W_2 = \begin{pmatrix} 1.0 & 0.5 & 0.0 \\\\ 0.0 & 0.3 & 1.0 \end{pmatrix} \quad \text{(2 outputs × 3 inputs)}
\\]

The result \\(Y = W_2 X^T\\) gives shape (2 outputs) × (2 observations):

```python
import torch

X = torch.tensor([
    [0.9, 0.4, 0.1],   # observation 1
    [0.2, 0.8, 0.7],   # observation 2
])

W2 = torch.tensor([
    [1.0, 0.5, 0.0],
    [0.0, 0.3, 1.0]
])

# Score each observation against each output weight vector
Y = W2 @ X.T   # shape: (2, 2) = (outputs, observations)
print("Scores (output x observation):")
print(Y)
# Row 0: conjunction-response scores for obs1, obs2
# Row 1: debris-response scores for obs1, obs2

# More commonly in ML, X is (batch, features) and W is (out, in)
# so you do W @ X.T or equivalently X @ W.T
Y_per_obs = X @ W2.T   # shape: (2, 2) = (observations, outputs)
print("\nScores (observation x output):")
print(Y_per_obs)
```

### Matrix multiplication is NOT commutative

This is a critical difference from scalar multiplication. For scalars, ab = ba. For matrices, **AB ≠ BA in general** — and often the product is not even defined in both orders.

```python
import torch

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

AB = A @ B
BA = B @ A
print("AB ="); print(AB)
print("BA ="); print(BA)
print(f"AB == BA: {torch.allclose(AB, BA)}")  # False — they differ

# Practical consequence: W2 @ W1 is not the same as W1 @ W2
# The order of matrix multiplication encodes the order of the layers
W1 = torch.randn(8, 4)   # first layer: 4 inputs -> 8 hidden
W2 = torch.randn(3, 8)   # second layer: 8 hidden -> 3 outputs

combined = W2 @ W1        # shape (3, 4): entire two-layer network as one matrix
x = torch.randn(4)

# These two computations give the same result:
y_sequential  = W2 @ (W1 @ x)
y_combined    = combined @ x
print(f"\nSequential equals combined: {torch.allclose(y_sequential, y_combined, atol=1e-5)}")
# True — but note: W1 @ W2 would be nonsense (shape mismatch)
```

The non-commutativity of matrix multiplication is not just a mathematical curiosity — it encodes the directionality of data flow in a neural network. Layer 1 comes before layer 2, and \\(W_2 W_1\\) is not the same transformation as \\(W_1 W_2\\).

---

## Adding a bias: the full neural network layer

In a real neural network layer, matrix multiplication is followed by adding a **bias vector** \\(\mathbf{b}\\):

\\[ \mathbf{y} = W\mathbf{x} + \mathbf{b} \\]

The bias vector \\(\mathbf{b}\\) has length m (same as the output). Adding the bias shifts each output score by a fixed amount, regardless of the input. This lets the network set a baseline level for each output even when the input is zero.

Extending the example:

\\[ \mathbf{b} = \begin{pmatrix} -0.5 \\ 0.3 \\ -0.1 \end{pmatrix} \\]

\\[ \mathbf{y} = W\mathbf{x} + \mathbf{b} = \begin{pmatrix} 1.02 \\ 0.68 \\ 0.48 \end{pmatrix} + \begin{pmatrix} -0.5 \\ 0.3 \\ -0.1 \end{pmatrix} = \begin{pmatrix} 0.52 \\ 0.98 \\ 0.38 \end{pmatrix} \\]

Now Response B scores highest. The bias shifted the scores, making Response B look more attractive even though its raw dot product was second. In a learned network, the bias values are adjusted during training to capture the prior attractiveness of each output independently of the input.

---

## PyTorch's nn.Linear: what it does internally

PyTorch's `nn.Linear` module is a pre-packaged version of \\(W\mathbf{x} + \mathbf{b}\\):

```python
import torch
import torch.nn as nn

# Create a linear layer: input dimension 4, output dimension 3
layer = nn.Linear(in_features=4, out_features=3)

# What does it contain?
print(f"Weight shape: {layer.weight.shape}")  # torch.Size([3, 4]) = 3 rows, 4 columns
print(f"Bias shape:   {layer.bias.shape}")    # torch.Size([3])   = 3 entries

# Apply it to an input
x = torch.tensor([0.8, 0.2, 0.1, 0.6])
y = layer(x)
print(f"Output shape: {y.shape}")  # torch.Size([3])

# Verify it is computing W @ x + b
y_manual = layer.weight @ x + layer.bias
print(f"Manual matches: {torch.allclose(y, y_manual)}")  # True
```

The weight matrix is stored as shape (out_features, in_features), meaning rows correspond to output dimensions and columns correspond to input dimensions. This is the same convention we have been using: each row is a weight vector for one output neuron, and its dot product with the input gives that neuron's pre-activation value.

---

## Tensors in ML: batched matrix operations

In the single-vector examples above, we multiplied a weight matrix by one input vector. In real training and inference, you almost never process one example at a time. You process a **batch** of examples simultaneously.

Why batching? Two reasons:

1. **GPU parallelism.** GPUs contain thousands of small arithmetic units that can perform the same operation on different data simultaneously (Single Instruction Multiple Data). Processing a batch of 32 inputs in one matrix multiply is far faster than 32 separate multiplications on GPU.

2. **Better gradient estimates.** Computing a gradient from a single example is noisy — that one example might not be representative. Averaging the gradient over 32-256 examples gives a much more reliable estimate of which direction to update the weights.

Concretely: if your input vector has 4 features and you want to process 32 examples at once, you stack them into a matrix with shape (32, 4). Each row is one example.

```python
import torch
import torch.nn as nn

# A realistic forward pass with batching
batch_size = 32
input_dim  = 4    # [conjunction_risk, debris_density, solar_activity, comms_window]
hidden_dim = 16
output_dim = 3    # scores for 3 response strategies

# Define a two-layer network
layer1 = nn.Linear(input_dim, hidden_dim)
layer2 = nn.Linear(hidden_dim, output_dim)
relu   = nn.ReLU()

# Simulate a batch of 32 SSA observations (normally loaded from a dataset)
torch.manual_seed(42)
observations = torch.randn(batch_size, input_dim)
print(f"Input batch shape:   {observations.shape}")   # (32, 4)

# Forward pass
hidden = relu(layer1(observations))
scores = layer2(hidden)

print(f"Hidden activations:  {hidden.shape}")         # (32, 16)
print(f"Output scores:       {scores.shape}")         # (32, 3)
# Each of the 32 rows is the output for one SSA observation

# nn.Linear internally handles the batching:
# For a single vector x of shape (4,):   output = W @ x + b
# For a batch X of shape (32, 4):       output = X @ W.T + b  (broadcasted)
# The shapes work out: (32,4) @ (4,3) + (3,) = (32,3)

# Manual verification for one example in the batch
x_single = observations[0]  # shape (4,)
h_manual = relu(layer1.weight @ x_single + layer1.bias)
y_manual = layer2.weight @ h_manual + layer2.bias
print(f"\nManual (example 0):  {y_manual.tolist()}")
print(f"Batched (example 0): {scores[0].tolist()}")
print(f"Match: {torch.allclose(y_manual, scores[0], atol=1e-5)}")  # True
```

The shape arithmetic generalizes cleanly: if `W` has shape (out, in) and `X` has shape (batch, in), then `X @ W.T` has shape (batch, out) — one output row per input example. PyTorch's `nn.Linear` handles this automatically, which is why you can define a layer for a single example and feed it batches without changing any code.

---

## Why stacking layers requires nonlinearities

You might wonder: if each layer is just \\(W\mathbf{x} + \mathbf{b}\\), what happens when you stack two layers?

```
y = W₂(W₁x + b₁) + b₂
  = W₂W₁x + W₂b₁ + b₂
  = W'x + b'
```

Where \\(W' = W_2 W_1\\) and \\(b' = W_2 b_1 + b_2\\). Stacking two linear layers gives you... another linear layer. The composition of linear functions is still linear.

This means that without anything else, a deep network with many layers would be no more powerful than a single layer. It could only learn linear transformations of the input.

What breaks this is the **activation function**: a nonlinear function applied elementwise to the output of each layer before passing it to the next. The most common is ReLU (Rectified Linear Unit): max(0, x). It is literally just: if the value is negative, set it to zero. If positive, leave it alone.

With a nonlinearity between layers, the composition is no longer equivalent to a single linear layer. The network can represent curved decision boundaries, complex patterns, and sophisticated functions of its input. That is what makes deep neural networks powerful.

In module 2 you will see this in full, including how the weights W are learned from data using the gradients from lesson 7. For now, just hold onto the idea: a layer is \\(W\mathbf{x} + \mathbf{b}\\), you can compute it as row-by-row dot products, and the W and b are what the network learns.

---

## Eigenvalues and eigenvectors (intuition only)

Most matrix operations transform a vector — they change both its direction and its magnitude. But certain special vectors, called **eigenvectors**, only get scaled when multiplied by a matrix. Their direction does not change.

Formally: **v** is an eigenvector of matrix A if:

\\[ A\mathbf{v} = \lambda \mathbf{v} \\]

where \\(\lambda\\) (Greek letter lambda) is a scalar called the **eigenvalue** — the factor by which A scales **v**.

**Decoding:** \\(A\mathbf{v} = \lambda \mathbf{v}\\) says "multiplying A by v gives back v, scaled by λ." If λ = 3, the matrix triples the vector's length without changing its direction. If λ = -1, the matrix flips the vector to point in the opposite direction. If λ = 0, the matrix collapses the vector to zero.

### Why eigenvalues matter for ML

**Principal Component Analysis (PCA).** The covariance matrix of a dataset encodes the variance and correlation of its features. Its eigenvectors point in the directions of greatest variance — these are the principal components. Its eigenvalues tell you how much variance each direction captures. PCA projects data onto the top-k eigenvectors, keeping the dimensions with the most information. In SSA, PCA on orbital state errors can reveal which error directions dominate across the catalog.

**Markov chains in RL and game theory.** A Markov chain is described by a transition matrix T, where T[i,j] is the probability of moving from state i to state j. The **stationary distribution** π satisfies \\(\pi = T\pi\\), which is the eigenvector equation with λ = 1. Finding the stationary distribution finds the long-run behavior of the system — in a game, this tells you what fraction of time a player spends in each state under equilibrium play.

**Stability analysis.** In dynamical systems (like orbital mechanics), whether a system's behavior grows, shrinks, or stays bounded depends on whether the eigenvalues of its state-transition matrix are larger than 1, smaller than 1, or exactly 1 in absolute value.

```python
import torch

# Simple 2x2 matrix
A = torch.tensor([[3.0, 1.0],
                  [0.0, 2.0]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
# tensor([3.+0.j, 2.+0.j])  -- eigenvalues are 3 and 2

print("Eigenvectors (columns):")
print(eigenvectors)
# Each column is an eigenvector

# Verify: A @ v = lambda * v for the first eigenvector
v0 = eigenvectors[:, 0].real   # first eigenvector (take real part)
lam0 = eigenvalues[0].real      # first eigenvalue

Av0 = (A.to(torch.complex64) @ eigenvectors[:, 0]).real
lam_v0 = (eigenvalues[0] * eigenvectors[:, 0]).real

print(f"\nA @ v0:      {Av0.tolist()}")
print(f"lambda * v0: {lam_v0.tolist()}")
# These should be equal (up to floating point)

# Covariance matrix example: eigenvectors point along principal error axes
# For a 2D position uncertainty ellipse:
cov_pos = torch.tensor([[4.0, 1.5],
                        [1.5, 1.0]])  # correlated position errors (km^2)

evals, evecs = torch.linalg.eig(cov_pos)
print(f"\nPosition covariance eigenvalues (variances along principal axes):")
print(evals.real.tolist())
print(f"First principal axis direction: {evecs[:, 0].real.tolist()}")
# The first eigenvector points in the direction of maximum position uncertainty
```

You do not need to compute eigenvalues by hand. PyTorch's `torch.linalg.eig` handles it. The key takeaway is what they mean: eigenvectors reveal the "natural axes" of a matrix, and eigenvalues tell you how much that matrix stretches or compresses along each axis.

---

## Key Takeaways

* **Matrix-vector multiplication is many dot products in parallel.** Each row of the weight matrix is a weight vector for one output neuron. The output is a vector of scores, one per row. This is what every neural network layer computes at its core.

* **Shape rules are non-negotiable.** For \\(W\mathbf{x}\\), the columns of W must equal the length of x. For \\(AB\\), the columns of A must equal the rows of B. The output shape is (rows of A) × (columns of B). Reading shapes in PyTorch — before debugging anything else — will resolve most dimension errors in 60 seconds.

* **The transpose flips rows and columns.** It is required in backpropagation (\\(W^T\\) in the gradient computation), in attention (\\(QK^T\\)), and in going from column vectors to row vectors. Symmetric matrices (\\(A = A^T\\)) arise naturally as covariance matrices in orbit estimation and as metric tensors.

* **Matrix multiplication is not commutative.** AB ≠ BA in general. The order encodes the data flow direction. Reversing two layers in a network is not the same network.

* **Batched operations are how real training works.** Processing 32–256 examples simultaneously as a matrix — rather than one at a time as a vector — is what makes GPU training fast. `nn.Linear` handles the batch dimension automatically. Kneusel's *Math for Deep Learning* Ch. 7 covers batched matrix operations in depth.

* **Eigenvectors reveal the natural axes of a matrix.** For covariance matrices, they point in the directions of greatest variance (PCA). For transition matrices, the eigenvector with eigenvalue 1 is the stationary distribution. These appear throughout RL and Bayesian inference as described in Kurt's *Bayesian Statistics the Fun Way* Ch. 14.

---

{{#quiz 06-matrices-and-matrix-vector-multiplication.toml}}
