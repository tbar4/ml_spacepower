# Lesson 6: Matrices and Matrix-Vector Multiplication

## Where this fits

In lesson 5, you saw that the dot product of a weight vector and an observation vector scores how well the observation matches that weight vector's "interest." A neural network layer does that simultaneously for many weight vectors at once, producing a score for each one. That simultaneous scoring is matrix-vector multiplication. Once you understand what \\(W\mathbf{x} + \mathbf{b}\\) computes, you know what a neural network layer does. Every modern deep learning architecture, from the policy networks in AlphaZero to the value networks in deep CFR, is built by stacking this operation repeatedly with nonlinearities in between.

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

## Shape rules: why dimensions must match

A matrix-vector multiplication \\(W\mathbf{x}\\) is only defined when the number of columns in \\(W\\) matches the length of \\(\mathbf{x}\\).

- If \\(W\\) is m × n and \\(\mathbf{x}\\) has length n, the result \\(\mathbf{y} = W\mathbf{x}\\) has length m.
- The "inner" dimension (columns of W, length of x) must match.
- The "outer" dimensions (rows of W, length of output) determine the result's shape.

In our example: W is 3 × 4, x has length 4. The 4s match (column dimension of W equals length of x). The result has length 3 (the number of rows).

This matters practically: if you have an observation of length 4 and want to compute scores for 3 responses, your weight matrix must be 3 × 4. Not 4 × 3. Not 3 × 3. The dimensions encode the data flow.

## Adding a bias: the full neural network layer

In a real neural network layer, matrix multiplication is followed by adding a **bias vector** \\(\mathbf{b}\\):

\\[ \mathbf{y} = W\mathbf{x} + \mathbf{b} \\]

The bias vector \\(\mathbf{b}\\) has length m (same as the output). Adding the bias shifts each output score by a fixed amount, regardless of the input. This lets the network set a baseline level for each output even when the input is zero.

Extending the example:

\\[ \mathbf{b} = \begin{pmatrix} -0.5 \\ 0.3 \\ -0.1 \end{pmatrix} \\]

\\[ \mathbf{y} = W\mathbf{x} + \mathbf{b} = \begin{pmatrix} 1.02 \\ 0.68 \\ 0.48 \end{pmatrix} + \begin{pmatrix} -0.5 \\ 0.3 \\ -0.1 \end{pmatrix} = \begin{pmatrix} 0.52 \\ 0.98 \\ 0.38 \end{pmatrix} \\]

Now Response B scores highest. The bias shifted the scores, making Response B look more attractive even though its raw dot product was second. In a learned network, the bias values are adjusted during training to capture the prior attractiveness of each output independently of the input.

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

## Quiz

{{#quiz 06-matrices-and-matrix-vector-multiplication.toml}}
