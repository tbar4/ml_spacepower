# Lesson 6: Matrices and matrix-vector multiplication

## Where this fits

A neural network layer is, mechanically, the operation \\(\mathbf{y} = W \mathbf{x} + \mathbf{b}\\), where \\(W\\) is a matrix and \\(\mathbf{x}, \mathbf{y}, \mathbf{b}\\) are vectors. That's it. Add an "activation function" applied elementwise to the result and you have what `nn.Linear` does in PyTorch. If you understand what \\(W \mathbf{x}\\) computes, you understand the inner loop of every feedforward neural network. The next module will stack a bunch of these and call it deep learning, but the operation underneath is exactly what you're about to see.

## Concept

A **matrix** is a rectangular grid of numbers. We say a matrix is "\\(m\\) by \\(n\\)" (or \\(m \times n\\)) when it has \\(m\\) rows and \\(n\\) columns. The order matters: rows first, then columns.

There are two equally valid ways to think about a matrix, and you should hold both in your head:

1. **As a stack of row vectors.** A 3-by-4 matrix is three row vectors of length 4 stacked on top of each other.
2. **As a stack of column vectors.** That same 3-by-4 matrix is four column vectors of length 3 stacked side by side.

Different operations lend themselves to different views. For matrix-vector multiplication, the row-vector view is the easier one.

## Matrix-vector multiplication

Take a matrix \\(W\\) of shape \\(m \times n\\) and a vector \\(\mathbf{x}\\) of length \\(n\\). The product \\(\mathbf{y} = W \mathbf{x}\\) is a new vector of length \\(m\\), where each entry is:

\\[ y_i = \sum_{j=1}^{n} W_{ij} \, x_j \\]

Decoding the symbols:

- \\(W_{ij}\\) is the matrix entry in row \\(i\\), column \\(j\\).
- \\(y_i\\) is the \\(i\\)-th entry of the output vector.
- The sum runs over \\(j\\), the column index, which corresponds to the entries of \\(\mathbf{x}\\).

The clean way to read this: **\\(y_i\\) is the dot product of row \\(i\\) of \\(W\\) with \\(\mathbf{x}\\).**

That's the whole operation, and it's exactly the lesson 5 dot product, applied repeatedly. A matrix-vector multiplication is just \\(m\\) dot products done in parallel. Each row of \\(W\\) asks "how much does \\(\mathbf{x}\\) look like me?" and emits a single number. The output vector collects those answers.

The shapes have to line up. If \\(W\\) is \\(m \times n\\), then \\(\mathbf{x}\\) must have length \\(n\\) (matching the number of columns of \\(W\\)), and the result \\(\mathbf{y}\\) has length \\(m\\) (matching the number of rows). The "inner dimensions" must agree.

## A neural network layer

A linear layer in a neural network adds a bias and (usually) an activation function:

\\[ \mathbf{y} = \sigma(W \mathbf{x} + \mathbf{b}) \\]

where \\(\mathbf{b}\\) is a vector of biases (length \\(m\\), one per output) and \\(\sigma\\) is some elementwise nonlinearity (like ReLU, which is just \\(\max(0, x)\\) applied elementwise). The activation function is what makes the network nonlinear and capable of learning complicated things; we'll cover it properly in module 2. For now, just know that the matrix-vector multiplication is the structural core, and the bias and activation are decorations on top.

When PyTorch defines `nn.Linear(in_features=4, out_features=3)`, what it creates internally is:

- A learnable weight matrix \\(W\\) of shape \\(3 \times 4\\).
- A learnable bias vector \\(\mathbf{b}\\) of length 3.

When you call this layer on an input vector \\(\mathbf{x}\\) of length 4, it computes \\(W \mathbf{x} + \mathbf{b}\\) and returns a vector of length 3. The "learnable" part means that during training, the entries of \\(W\\) and \\(\mathbf{b}\\) get nudged to make the network's outputs more correct. We'll see how that nudging happens in lesson 7.

## Code

```python
import torch
import torch.nn as nn

# Define a linear layer: 4-dimensional input, 3-dimensional output.
layer = nn.Linear(in_features=4, out_features=3)

# What's inside:
print(layer.weight.shape)  # torch.Size([3, 4])  — 3 rows, 4 columns
print(layer.bias.shape)    # torch.Size([3])     — 3 entries

# Apply it to an input.
x = torch.tensor([1.0, 0.5, -1.0, 2.0])
y = layer(x)
print(y.shape)  # torch.Size([3])

# Verify by hand: y should equal W @ x + b
y_manual = layer.weight @ x + layer.bias
print(torch.allclose(y, y_manual))  # True
```

The `@` operator in `layer.weight @ x` does the matrix-vector multiplication. The `nn.Linear` class is a convenience wrapper that bundles the weight matrix, the bias vector, and some bookkeeping for gradient tracking.

A subtle naming convention: PyTorch stores weights with the **output dimension first** (rows = outputs), so a layer mapping length-4 to length-3 has a `weight` of shape `(3, 4)`. This is exactly the row-major view we just described.

## Worked example: hand-computed forward pass

Let's do one by hand. Suppose:

\\[ W = \begin{pmatrix} 1 & 0 & -1 & 2 \\ 0 & 1 & 1 & 0 \\ -1 & 1 & 0 & 1 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 0 \\ -1 \\ 2 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 2 \\ 1 \\ 3 \\ 1 \end{pmatrix} \\]

Compute \\(\mathbf{y} = W \mathbf{x} + \mathbf{b}\\).

Row 1 of \\(W\\) dotted with \\(\mathbf{x}\\): \\((1)(2) + (0)(1) + (-1)(3) + (2)(1) = 2 + 0 - 3 + 2 = 1\\).

Row 2 of \\(W\\) dotted with \\(\mathbf{x}\\): \\((0)(2) + (1)(1) + (1)(3) + (0)(1) = 0 + 1 + 3 + 0 = 4\\).

Row 3 of \\(W\\) dotted with \\(\mathbf{x}\\): \\((-1)(2) + (1)(1) + (0)(3) + (1)(1) = -2 + 1 + 0 + 1 = 0\\).

So \\(W \mathbf{x} = (1, 4, 0)\\). Add the bias to get \\(\mathbf{y} = (1, 3, 2)\\).

Verify in code:

```python
import torch

W = torch.tensor([
    [ 1.0,  0.0, -1.0,  2.0],
    [ 0.0,  1.0,  1.0,  0.0],
    [-1.0,  1.0,  0.0,  1.0],
])
b = torch.tensor([0.0, -1.0, 2.0])
x = torch.tensor([2.0, 1.0, 3.0, 1.0])

y = W @ x + b
print(y)  # tensor([1., 3., 2.])
```

That's a forward pass through a linear layer. No machine learning, just arithmetic. The "learning" part will come from finding good values of \\(W\\) and \\(\mathbf{b}\\), which is the next lesson.

## Why this matters going forward

When the next module talks about a "two-layer MLP," what it means is: take an input vector, multiply by a matrix, apply a nonlinearity, multiply by another matrix, apply another nonlinearity. That is the entire architecture. Three layers means three matrices. Deep networks are stacks of these, with various tricks to keep training stable. Once you internalize "neural network = matrix-vector products with nonlinearities sprinkled in," nearly all of deep learning becomes much less mysterious.

Same story in OpenSpiel: when an algorithm uses a neural net to approximate a value function or a policy, the value function is `state_vector -> matrix -> nonlinearity -> matrix -> nonlinearity -> output`, and the policy is the same shape with a softmax at the end. We'll get to softmax in module 2.

## A note on what we skipped

Matrix-matrix multiplication is just matrix-vector multiplication done multiple times in parallel (each column of the result vector matrix is the matrix multiplied by the corresponding column of the input matrix). PyTorch handles batched inputs by doing exactly this, and you'll see expressions like `Y = X @ W.T` for batched forward passes. The mechanics are the same.

We skipped matrix inverses, determinants, ranks, eigenvectors, and eigenvalues. Inverses and determinants don't show up in our path. Eigenvectors will come back when we hit alpha-rank in module 6, and we'll handle them then with proper motivation. Skipping them now is the correct move.

## Quiz

{{#quiz 06-matrices-and-matrix-vector-multiplication.toml}}
