# Lesson 5: Vectors and Dot Products

**Module:** ML Foundations — M01: Mathematical Foundations
**Source:** *Math for Deep Learning* — Ronald T. Kneusel, Ch. 5–6 (Vectors and Vector Operations); *Bayesian Statistics the Fun Way* — Will Kurt, Ch. 2 (distributions as vectors of probability); PyTorch documentation

---


<!-- toc -->

## Where this fits

Every state in a reinforcement learning system, every observation your agent receives, every action embedding, every intermediate representation inside a neural network, is a vector. The dot product is the single most common operation performed on those vectors: it is how a neural network layer evaluates whether its input "matches" a learned pattern. If you can look at a vector and say what it represents, and look at a dot product and say what it is measuring, you have the geometric intuition for 90% of what deep learning does internally.

---

## Scalars, vectors, matrices, and tensors

Before we go any further, we need to get the vocabulary right. These four terms appear in every deep learning paper and codebase, and they are often used loosely.

**Scalars** are single numbers — a dimensionless quantity. A loss value, a learning rate, a single orbital period: all scalars.

**Vectors** are one-dimensional arrays of numbers. The key thing is the ordering: position 0 is always the same kind of value, position 1 is always the same kind of value, and so on. An orbital state vector, a probability distribution over actions, a gradient computed during backpropagation — all vectors.

**Matrices** are two-dimensional arrays. Rows and columns. A weight matrix in a neural network layer is the canonical example. A covariance matrix describing uncertainty in an orbit estimate is another.

**Tensors** are n-dimensional arrays. A scalar is a 0-dimensional tensor. A vector is a 1-dimensional tensor. A matrix is a 2-dimensional tensor. An image is typically a 3-dimensional tensor (height × width × channels). A batch of images is a 4-dimensional tensor (batch_size × height × width × channels). Attention score matrices for multiple heads across a batch are 4-dimensional tensors.

PyTorch's `torch.Tensor` is the unified object that represents all of these. The number of dimensions is called the **rank** or **ndim**, and the size along each dimension is the **shape**.

```python
import torch

# Scalar: rank 0, shape ()
loss = torch.tensor(3.0)
print(f"Scalar:  ndim={loss.ndim}, shape={loss.shape}")
# ndim=0, shape=torch.Size([])

# Vector: rank 1, shape (5,)
state = torch.zeros(5)
print(f"Vector:  ndim={state.ndim}, shape={state.shape}")
# ndim=1, shape=torch.Size([5])

# Matrix: rank 2, shape (3, 4)
weights = torch.zeros(3, 4)
print(f"Matrix:  ndim={weights.ndim}, shape={weights.shape}")
# ndim=2, shape=torch.Size([3, 4])

# 3D tensor: rank 3, shape (2, 3, 4)
# Imagine: 2 time steps, each with a 3x4 feature map
features = torch.zeros(2, 3, 4)
print(f"3D Tensor: ndim={features.ndim}, shape={features.shape}")
# ndim=3, shape=torch.Size([2, 3, 4])

# Shape inspection is a habit worth building
orbital_state = torch.tensor([6371.0, 500.0, -200.0, 7.2, 0.3, -0.1])
print(f"Orbital state: shape={orbital_state.shape}, dtype={orbital_state.dtype}")
# shape=torch.Size([6]), dtype=torch.float32
```

In SSA, you will regularly encounter all four. An individual threat score is a scalar. A six-element orbital state is a vector. A batch of orbital states for 32 tracked objects is a matrix. A sequence of observation batches over time is a 3D tensor.

Kneusel's *Math for Deep Learning* (Ch. 5) works through the vector and matrix operations that underlie all of this. What PyTorch adds is the ability to operate on entire tensors in parallel on GPU hardware — but the math is the same as working component-by-component.

---

## What is a vector? Start with the orbital state vector

Suppose you are tracking a satellite. At any given moment, you need to know two things to describe its complete dynamical state: where it is and how fast it is moving. In three-dimensional space, position requires three numbers and velocity requires three numbers. You need six numbers total.

You could write these numbers separately:

```
Position x: 6,371 km     Velocity x: 7.5 km/s
Position y: 0 km          Velocity y: 0.0 km/s
Position z: 0 km          Velocity z: 0.0 km/s
```

Or you could write them as an ordered list:

```
state = [6371, 0, 0, 7.5, 0.0, 0.0]
```

That ordered list is a **vector**. It contains six numbers, so we say it is a "six-dimensional vector" or a "vector of length 6." The order matters: the first number is always the x-position, the fourth is always the x-velocity, and so on.

A vector is nothing more than an ordered list of numbers. The numbers can represent anything: positions, velocities, sensor readings, action probabilities, learned features. The abstract mathematical concept does not care what the numbers mean; it just provides tools for working with lists of numbers.

Here are some vectors that appear constantly in our work:

**A 6D orbital state vector** (position + velocity in Earth-centered inertial frame):
```
[x, y, z, vx, vy, vz]
[6371.0, 500.0, -200.0, 7.2, 0.3, -0.1]
```

**A 4D action probability distribution** for an RL agent with 4 possible actions:
```
[P(action 0), P(action 1), P(action 2), P(action 3)]
[0.10, 0.20, 0.30, 0.40]
```

**A 3D observation vector** from a tracking sensor (range, angle, angular rate):
```
[range_km, azimuth_deg, elevation_deg]
[850.3, 42.1, 15.6]
```

Each of these is just a list of numbers. The mathematics of vectors applies equally to all of them.

---

## Visualizing vectors as arrows

In two dimensions, a vector [a, b] can be drawn as an arrow starting at the origin and ending at the point (a, b). The length of the arrow is the magnitude of the vector, and the direction the arrow points captures the relationship between the two components.

This geometric picture generalizes to higher dimensions even though we cannot draw a 6D vector. The key properties of vectors as arrows:

- **Longer arrows** represent vectors with larger magnitude (we will make this precise shortly with norms)
- **Direction** captures the ratio and sign relationship between components
- **Two arrows pointing in the same direction** represent vectors with the same ratios between components, even if one is longer

For two velocity vectors representing satellites in similar orbits:
```
v1 = [7.5, 0.0, 0.1]   # moving mostly in +x direction
v2 = [7.2, 0.3, 0.0]   # also mostly in +x, slightly in +y
```
These arrows point in nearly the same direction. Both satellites are moving primarily along the x-axis with small components in other directions. This "similarity of direction" is exactly what the dot product measures.

---

## The length of a vector: norms

Before we talk about dot products, we need to know how to measure the length of a vector.

For a 2D vector [a, b], the Pythagorean theorem gives the length: √(a² + b²). A vector [3, 4] has length √(9 + 16) = √25 = 5.

For a vector of any length [v₁, v₂, ..., vₙ], the same idea extends:

\[ \|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2} \]

**Decoding each symbol:**

**\(\|\mathbf{v}\|_2\)**: The "L2 norm" or "Euclidean norm" of vector v. The subscript 2 distinguishes it from other norms (L1, L∞) we will cover shortly. The double vertical bars mean "length of." The bold v indicates a vector.

**\(\sqrt{\ldots}\)**: Square root.

**\(v_i^2\)**: The i-th component of the vector, squared.

**\(\sum_{i=1}^{n}\)**: Sum all n components.

**In plain English**: "Square every component, add them all up, take the square root." This is the Euclidean distance from the origin to the point represented by the vector.

For the orbital state vector example:

```
v = [6371.0, 500.0, -200.0, 7.2, 0.3, -0.1]
||v||₂ = sqrt(6371^2 + 500^2 + 200^2 + 7.2^2 + 0.3^2 + 0.1^2)
       = sqrt(40,589,641 + 250,000 + 40,000 + 51.84 + 0.09 + 0.01)
       ≈ sqrt(40,879,693)
       ≈ 6,394 (a mix of km and km/s, so not physically meaningful,
                but mathematically valid)
```

In code:
```python
import torch
v = torch.tensor([6371.0, 500.0, -200.0, 7.2, 0.3, -0.1])
norm = torch.linalg.norm(v)
print(norm.item())  # approximately 6394
```

Rust uses `ndarray` for vector operations. Cargo dependencies for every Rust block in this lesson (matched to the Playground catalog so the mdbook "play" button works):

```toml
[dependencies]
ndarray = "0.17"
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn main() {
    let v = Array1::from_vec(vec![6371.0_f64, 500.0, -200.0, 7.2, 0.3, -0.1]);
    let norm = v.mapv(|x| x * x).sum().sqrt();
    println!("{norm:.0}"); // approximately 6394
}
```

`.mapv(|x| x * x)` applies the closure element-wise and returns a new array; `.sum()` collapses it to a scalar `f64`; `.sqrt()` is the standard float method.

---

## L1 norm and other norms

The L2 norm is the most common, but it is not the only way to measure vector length. Different norms have different properties that make them useful in different parts of machine learning.

### The L1 norm

\[ \|\mathbf{v}\|_1 = |v_1| + |v_2| + \ldots + |v_n| = \sum_{i=1}^{n} |v_i| \]

**Decoding:** The L1 norm is simply the sum of the absolute values of all components. The subscript 1 indicates this is the "1-norm." Unlike L2, it does not square the components — large and small components are treated more equally.

The L1 norm is used in **LASSO regularization** (L1 penalty on weights). Because it does not square the values, it has a tendency to push small weights all the way to zero — producing sparse weight vectors where many entries are exactly 0. This is useful when you want a model that ignores most of its inputs and focuses on a few key features.

### The L∞ norm (maximum norm)

\[ \|\mathbf{v}\|_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|) \]

**Decoding:** The L∞ norm is the largest absolute component. The subscript ∞ reflects that this is the limit of the p-norm as p → ∞. It answers the question: "what is the single worst-case component?"

The L∞ norm appears in robust bounding problems. If you are computing a guaranteed error bound on a state estimate, the L∞ norm tells you the maximum error in any single dimension — often the operationally relevant quantity.

### When to use each norm

| Norm | Formula | ML Use Case | SSA Use Case |
|------|---------|------------|-------------|
| L1 | sum of abs values | LASSO regularization, sparse features | Robustness to outlier measurements |
| L2 | sqrt of sum of squares | Weight decay regularization, Euclidean distance | Orbital distance, conjunction metric |
| L∞ | max abs component | Worst-case bounds, robust optimization | Maximum position error bound |

```python
import torch

# Satellite state vector: position error in km
state_error = torch.tensor([0.5, -2.1, 0.8, 0.002, -0.004, 0.001])

# L2 norm: the "size" of the error in Euclidean sense
l2 = torch.linalg.norm(state_error, ord=2)
print(f"L2 norm (Euclidean):  {l2.item():.4f} km")  # ~2.27

# L1 norm: sum of absolute deviations
l1 = torch.linalg.norm(state_error, ord=1)
print(f"L1 norm (sum abs):    {l1.item():.4f} km")  # ~3.41

# L-inf norm: maximum single-component deviation
linf = torch.linalg.norm(state_error, ord=float('inf'))
print(f"L∞ norm (max abs):    {linf.item():.4f} km")  # 2.1

# In an SSA context:
# L2 norm: overall state estimation error magnitude
# L∞ norm: "the worst single coordinate is 2.1 km off"
# L1 norm: used in sparse sensor selection regularization
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn main() {
    let err = Array1::from_vec(vec![0.5_f64, -2.1, 0.8, 0.002, -0.004, 0.001]);

    let l2 = err.mapv(|x| x * x).sum().sqrt();
    let l1: f64 = err.mapv(f64::abs).sum();
    let linf = err.mapv(f64::abs).iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("L2 norm (Euclidean): {l2:.4}"); // ~2.27
    println!("L1 norm (sum abs):   {l1:.4}"); // ~3.41
    println!("L∞ norm (max abs):   {linf:.4}"); // 2.1
}
```

`.mapv(f64::abs)` uses `f64::abs` as a function pointer (signature `fn(f64) -> f64`). `.iter().cloned().fold(f64::NEG_INFINITY, f64::max)` walks the array finding the maximum; `f64::max` is a two-argument function `fn(f64, f64) -> f64` that returns the larger value.

---

## Unit vectors and normalization

A **unit vector** has norm 1. It represents a pure direction — no magnitude information, just which way a vector points.

To normalize a vector (convert it to a unit vector), divide every component by the vector's norm:

\[ \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \]

**Decoding:** The hat symbol ˆ (called "hat" notation) over a vector conventionally indicates a unit vector. The formula divides the entire vector by its scalar norm. Every component is scaled by the same factor, so the direction is preserved while the length becomes exactly 1.

**Verification:** \(\|\hat{\mathbf{v}}\|_2 = \|\mathbf{v}\|_2 / \|\mathbf{v}\|_2 = 1\). Correct.

### When normalization is essential

**Cosine similarity** — If you want to compare directions without being confused by magnitudes, normalize first. Two satellites moving at the same angle but different speeds should have direction-similarity 1.0, not be penalized for the speed difference.

**Attention in transformers** — The query-key dot product is scaled by 1/√d where d is the dimension. This is similar in spirit to normalization: it prevents the dot products from getting arbitrarily large as the vector dimension grows, which would cause vanishing gradients through the softmax.

**Orbit determination** — When you have a position vector and want to describe the direction to a satellite (the "look vector" for a sensor), you normalize the position vector. The resulting unit vector is the pointing direction, independent of how far away the satellite is.

```python
import torch

# Position vector to a satellite in ECI frame (km)
position = torch.tensor([4500.0, 3200.0, -1100.0])

# The look vector is the unit vector in the direction of position
norm = torch.linalg.norm(position)
look_vector = position / norm

print(f"Position magnitude: {norm.item():.2f} km")
print(f"Look vector:        {look_vector.tolist()}")
print(f"Look vector norm:   {torch.linalg.norm(look_vector).item():.6f}")
# Should be exactly 1.0

# In PyTorch, F.normalize is a convenience function that does the same
import torch.nn.functional as F
look_vector_v2 = F.normalize(position, dim=0)
print(f"Using F.normalize:  {look_vector_v2.tolist()}")

# Comparing two satellite directions by cosine similarity
# (without needing to know their actual distances)
pos_sat2 = torch.tensor([4450.0, 3250.0, -1050.0])
look_sat2 = F.normalize(pos_sat2, dim=0)

cos_sim = torch.dot(look_vector, look_sat2)
print(f"\nCosine similarity between pointing directions: {cos_sim.item():.6f}")
# Near 1.0: satellites are in nearly the same direction from the observer
```

> **Warning: normalizing a zero vector causes division by zero.**  
> If the input vector is all zeros, its norm is 0, and dividing by it produces `nan` or `inf` values silently in PyTorch. Always guard against this in production code:

```python
def safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a vector, returning a zero vector if the input norm is below eps."""
    norm = torch.linalg.norm(v)
    if norm < eps:
        # Could also raise an error — depends on whether zero input is expected
        return torch.zeros_like(v)
    return v / norm

# Defensive usage in SSA:
relative_velocity = torch.tensor([0.0, 0.0, 0.0])  # satellites at rest relative to each other
safe_dir = safe_normalize(relative_velocity)
print(f"Safe normalize of zero: {safe_dir.tolist()}")  # [0.0, 0.0, 0.0]
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn safe_normalize(v: &Array1<f64>, eps: f64) -> Array1<f64> {
    let norm = v.mapv(|x| x * x).sum().sqrt();
    if norm < eps {
        Array1::zeros(v.len())
    } else {
        v.mapv(|x| x / norm)
    }
}

fn main() {
    let position = Array1::from_vec(vec![4500.0_f64, 3200.0, -1100.0]);
    let norm = position.mapv(|x| x * x).sum().sqrt();
    let look_vector = position.mapv(|x| x / norm);

    println!("Position magnitude: {norm:.2} km");
    println!("Look vector norm:   {:.6}", look_vector.mapv(|x| x * x).sum().sqrt()); // 1.000000

    // Cosine similarity of two satellite pointing directions
    let pos_sat2 = Array1::from_vec(vec![4450.0_f64, 3250.0, -1050.0]);
    let norm2 = pos_sat2.mapv(|x| x * x).sum().sqrt();
    let look_sat2 = pos_sat2.mapv(|x| x / norm2);

    let cos_sim: f64 = (&look_vector * &look_sat2).sum();
    println!("Cosine similarity:  {cos_sim:.6}");

    // Safe normalize handles the zero-vector edge case
    let zero = Array1::<f64>::zeros(3);
    let safe = safe_normalize(&zero, 1e-8);
    println!("Safe normalize of zero: {:?}", safe.as_slice().unwrap());
}
```

`v.mapv(|x| x / norm)` divides every element by the scalar `norm` — ndarray does not overload `/` between `&Array` and `f64` directly, so `.mapv` is the idiomatic path. `&Array1 * &Array1` is element-wise; `.sum()` collapses to `f64`.

---

## Projection

The **projection** of vector **v** onto vector **w** is the component of **v** that lies in the direction of **w**. Geometrically: if you shone a light perpendicular to **w** and cast the shadow of **v** onto the line defined by **w**, the shadow is the projection.

The formula:

\[ \text{proj}_{\mathbf{w}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{w}\|^2} \cdot \mathbf{w} \]

**Decoding:**

**\(\mathbf{v} \cdot \mathbf{w}\)**: The dot product of v and w.

**\(\|\mathbf{w}\|^2\)**: The squared norm of w. This normalizes so that scaling w does not change the projection result.

**\(\frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{w}\|^2}\)**: A scalar — the "amount" of v in the w direction.

**\(\frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{w}\|^2} \cdot \mathbf{w}\)**: That scalar times the direction vector **w**, giving the projected vector (in the direction of **w**, with the appropriate magnitude).

The equivalent formulation using the unit vector \(\hat{\mathbf{w}}\) is cleaner:

\[ \text{proj}_{\hat{\mathbf{w}}}(\mathbf{v}) = (\mathbf{v} \cdot \hat{\mathbf{w}}) \cdot \hat{\mathbf{w}} \]

### SSA application: radial approach component

In conjunction analysis, the relative approach velocity between two satellites can be decomposed into:
- The **radial component**: approach velocity along the line connecting their positions (directly toward or away from each other)
- The **tangential component**: approach velocity perpendicular to that line

The radial component is what determines how quickly the miss distance is changing. A high relative speed with a small radial component means the satellites are passing each other, not approaching head-on.

```python
import torch

# Satellite 1 position (km) in ECI
r1 = torch.tensor([6800.0, 0.0, 0.0])
# Satellite 2 position (km) in ECI
r2 = torch.tensor([6790.0, 50.0, 10.0])

# Relative approach velocity of satellite 2 w.r.t. satellite 1 (km/s)
dv = torch.tensor([-0.5, 7.2, 0.1])  # mainly in y-direction (tangential)

# Radial direction: unit vector from r1 to r2
radial_vec = r2 - r1
radial_unit = radial_vec / torch.linalg.norm(radial_vec)

print(f"Vector between satellites: {radial_vec.tolist()}")
print(f"Radial unit vector:        {radial_unit.tolist()}")

# Projection of approach velocity onto the radial direction
radial_speed = torch.dot(dv, radial_unit)  # scalar: speed along radial
radial_component = radial_speed * radial_unit  # vector: radial part of velocity

# Tangential component: what's left after subtracting the radial part
tangential_component = dv - radial_component

print(f"\nApproach velocity:          {dv.tolist()}")
print(f"Radial approach speed:      {radial_speed.item():.4f} km/s")
print(f"  (negative = approaching, positive = separating)")
print(f"Radial velocity component:  {radial_component.tolist()}")
print(f"Tangential velocity:        {tangential_component.tolist()}")

# Verify: radial + tangential = original
reconstructed = radial_component + tangential_component
print(f"\nReconstruction check (should be {dv.tolist()}):")
print(f"  {reconstructed.tolist()}")
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn main() {
    let r1 = Array1::from_vec(vec![6800.0_f64, 0.0, 0.0]);
    let r2 = Array1::from_vec(vec![6790.0_f64, 50.0, 10.0]);
    let dv = Array1::from_vec(vec![-0.5_f64, 7.2, 0.1]);

    // Radial unit vector from r1 to r2
    let radial_vec = &r2 - &r1;
    let radial_norm = radial_vec.mapv(|x| x * x).sum().sqrt();
    let radial_unit = radial_vec.mapv(|x| x / radial_norm);

    // Project dv onto the radial direction
    let radial_speed: f64 = (&dv * &radial_unit).sum();
    let radial_component = radial_unit.mapv(|x| x * radial_speed);
    let tangential_component = &dv - &radial_component;

    println!("Radial approach speed: {radial_speed:.4} km/s");
    println!("  (negative = approaching, positive = separating)");

    // Verify reconstruction
    let reconstructed = &radial_component + &tangential_component;
    println!("Reconstruction: {:?}", reconstructed.as_slice().unwrap());
}
```

`&r2 - &r1` is element-wise subtraction between two array references — ndarray infers the shape from the operands and returns an owned `Array1<f64>`.

The radial speed here is small, meaning the primary motion is tangential (the satellites are mostly moving past each other, not closing directly). This is the geometric content of the projection.

---

## The dot product: measuring alignment

The **dot product** of two vectors of the same length is computed by:

1. Multiplying corresponding components together
2. Adding all the products

For vectors **v** = [v₁, v₂, ..., vₙ] and **w** = [w₁, w₂, ..., wₙ]:

\[ \mathbf{v} \cdot \mathbf{w} = v_1 w_1 + v_2 w_2 + \ldots + v_n w_n = \sum_{i=1}^{n} v_i w_i \]

**Decoding:**

**\(\mathbf{v} \cdot \mathbf{w}\)**: "The dot product of v and w." The bold letters indicate vectors. The centered dot is the dot product operation (not regular multiplication, which would give a vector).

**\(v_i w_i\)**: Component i of v times component i of w. Subscripts connect corresponding components.

**\(\sum_{i=1}^{n} v_i w_i\)**: Add up all those pairwise products.

Let us compute it step by step for two small vectors:

```
v = [2, 3, -1]
w = [4, -2, 5]

Step 1: Multiply corresponding components
  2 × 4  = 8
  3 × (-2) = -6
  (-1) × 5 = -5

Step 2: Add the products
  8 + (-6) + (-5) = -3

Dot product: -3
```

In code:
```python
import torch
v = torch.tensor([2.0, 3.0, -1.0])
w = torch.tensor([4.0, -2.0, 5.0])

# Three equivalent ways to compute the dot product
print((v * w).sum().item())      # -3.0
print(torch.dot(v, w).item())    # -3.0
print((v @ w).item())            # -3.0
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn main() {
    let v = Array1::from_vec(vec![2.0_f64, 3.0, -1.0]);
    let w = Array1::from_vec(vec![4.0_f64, -2.0, 5.0]);

    // Element-wise product then sum — the definition of dot product
    let dot: f64 = (&v * &w).sum();
    println!("{dot}"); // -3
}
```

`&v * &w` is element-wise multiplication of two array references; `.sum()` collapses the result to a scalar. ndarray has no dedicated `.dot()` for 1D arrays, so this pattern is the standard idiom.

---

## What the dot product is measuring: alignment

The arithmetic definition is straightforward. But what does the dot product actually tell us?

The dot product has a geometric interpretation:

\[ \mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \cdot \|\mathbf{w}\| \cdot \cos(\theta) \]

where \(\theta\) (the Greek letter theta) is the angle between the two vectors.

**Decoding:**

**\(\|\mathbf{v}\|\)**: The norm (length) of v.
**\(\|\mathbf{w}\|\)**: The norm (length) of w.
**\(\cos(\theta)\)**: The cosine of the angle between them.

You do not need to remember the details of cosine, but you need to know these key facts:
- cos(0°) = 1: vectors pointing in exactly the same direction
- cos(90°) = 0: vectors that are perpendicular (at right angles)
- cos(180°) = -1: vectors pointing in exactly opposite directions

So what does the dot product tell you?

- **Large positive dot product**: the vectors point in roughly the same direction
- **Zero dot product**: the vectors are perpendicular (completely "unrelated" in direction)
- **Large negative dot product**: the vectors point in roughly opposite directions

**In SSA terms**: if two satellites have velocity vectors with a large positive dot product, they are moving in roughly the same direction. Their relative speed is low and they will not approach each other quickly. If their dot products are large negative, they are moving toward each other on nearly head-on trajectories: higher collision risk. This is not how real conjunction analysis works, but the intuition is correct.

---

## An SSA example: comparing approach geometries

Two satellites are on potential collision courses. You want a quick sense of how "head-on" versus "overtaking" the geometry is.

```python
import torch

# Satellite 1 moving in +x direction at orbital velocity
v1 = torch.tensor([7.5, 0.0, 0.0])  # km/s

# Case A: satellite 2 moving in -x direction (head-on collision geometry)
v2_headon = torch.tensor([-7.5, 0.0, 0.0])

# Case B: satellite 2 moving in +x direction but slower (overtaking geometry)
v2_overtake = torch.tensor([6.8, 0.1, 0.0])

# Case C: satellite 2 on an inclined orbit (crossing geometry)
v2_cross = torch.tensor([0.0, 7.5, 0.0])

# Dot products
dot_headon   = torch.dot(v1, v2_headon)
dot_overtake = torch.dot(v1, v2_overtake)
dot_cross    = torch.dot(v1, v2_cross)

print(f"Head-on geometry:    dot product = {dot_headon.item():.1f}")    # -56.25 (strongly negative)
print(f"Overtaking geometry: dot product = {dot_overtake.item():.1f}")  # +51.0  (positive)
print(f"Crossing geometry:   dot product = {dot_cross.item():.1f}")     # 0.0    (perpendicular)

# Cosine similarity: normalize out the lengths to get just the direction
norm_v1 = torch.linalg.norm(v1)
norm_headon   = torch.linalg.norm(v2_headon)
norm_overtake = torch.linalg.norm(v2_overtake)
norm_cross    = torch.linalg.norm(v2_cross)

cos_headon   = dot_headon   / (norm_v1 * norm_headon)
cos_overtake = dot_overtake / (norm_v1 * norm_overtake)
cos_cross    = dot_cross    / (norm_v1 * norm_cross)

print(f"\nCosine similarity:")
print(f"Head-on:    {cos_headon.item():.3f}   (angle: {torch.rad2deg(torch.acos(cos_headon)).item():.1f}°)")
print(f"Overtaking: {cos_overtake.item():.3f}   (angle: {torch.rad2deg(torch.acos(cos_overtake)).item():.1f}°)")
print(f"Crossing:   {cos_cross.item():.3f}   (angle: {torch.rad2deg(torch.acos(cos_cross)).item():.1f}°)")
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a * b).sum()
}

fn norm(a: &Array1<f64>) -> f64 {
    a.mapv(|x| x * x).sum().sqrt()
}

fn main() {
    let v1         = Array1::from_vec(vec![7.5_f64, 0.0, 0.0]);
    let v2_headon  = Array1::from_vec(vec![-7.5_f64, 0.0, 0.0]);
    let v2_overtake = Array1::from_vec(vec![6.8_f64, 0.1, 0.0]);
    let v2_cross   = Array1::from_vec(vec![0.0_f64, 7.5, 0.0]);

    let dot_headon   = dot(&v1, &v2_headon);
    let dot_overtake = dot(&v1, &v2_overtake);
    let dot_cross    = dot(&v1, &v2_cross);

    println!("Head-on:    {dot_headon:.1}");   // -56.25
    println!("Overtaking: {dot_overtake:.1}"); // 51.0
    println!("Crossing:   {dot_cross:.1}");    // 0.0

    let norm_v1 = norm(&v1);
    let cos_headon   = dot_headon   / (norm_v1 * norm(&v2_headon));
    let cos_overtake = dot_overtake / (norm_v1 * norm(&v2_overtake));
    let cos_cross    = dot_cross    / (norm_v1 * norm(&v2_cross));

    println!("Head-on cosine:    {cos_headon:.3}  ({:.1}°)", cos_headon.acos().to_degrees());
    println!("Overtaking cosine: {cos_overtake:.3}  ({:.1}°)", cos_overtake.acos().to_degrees());
    println!("Crossing cosine:   {cos_cross:.3}  ({:.1}°)", cos_cross.acos().to_degrees());
}
```

`.acos().to_degrees()` is available directly on `f64` — no extra import needed. Note that `cos_cross` will be exactly `0.0`, and `f64::acos(0.0) = π/2` radians = 90°.

The head-on geometry gives a highly negative cosine (angle ≈ 180°), the crossing geometry gives zero (exactly 90°), and the overtaking geometry gives a high positive value (small angle, similar direction).

---

## Dot products as scoring: the bridge to neural networks

Here is the connection to machine learning that makes the dot product so important.

Suppose you want to score how much an observation "favors" a particular action. For example, you are operating a sensor, and based on the current observation vector (describing the state of the space environment), you want to score each possible pointing action.

You define a **weight vector** \(\mathbf{w}\) for each action. The weight vector describes what kind of observation the action is best suited for. The score for taking that action given observation \(\mathbf{o}\) is the dot product \(\mathbf{w} \cdot \mathbf{o}\).

If the observation looks like what the weight vector describes (same direction, high alignment), the score is high. If the observation is perpendicular or opposite to the weight vector, the score is low or negative.

This is exactly what a single neuron in a neural network computes. The neuron has a learned weight vector. Its output is the dot product of the weight vector and the input. The network learns weight vectors that give high scores to the kinds of inputs that should lead to good outputs.

In the next lesson, we will stack many neurons in parallel. That will give us matrix-vector multiplication: the operation that defines a neural network layer.

---

## Worked example: hand-computing a dot product for a sensor scoring task

Your sensor has a 4D observation vector describing the current environment:

```
o = [conjunction_risk, debris_density, solar_activity, comms_window_fraction]
o = [0.8, 0.2, 0.1, 0.6]
```

You have two candidate sensor pointing strategies, each represented by a weight vector that describes what conditions each strategy "cares about":

```
Strategy A (conjunction-focused): w_A = [1.0, 0.3, 0.0, 0.2]
  (heavily weights conjunction risk, somewhat weights debris)

Strategy B (comms-window-focused): w_B = [0.1, 0.0, 0.0, 1.0]
  (mainly weights communications window availability)
```

**Score for Strategy A:**

Step 1: Multiply corresponding components:
- 0.8 × 1.0 = 0.80
- 0.2 × 0.3 = 0.06
- 0.1 × 0.0 = 0.00
- 0.6 × 0.2 = 0.12

Step 2: Add:
- 0.80 + 0.06 + 0.00 + 0.12 = **0.98**

**Score for Strategy B:**

Step 1:
- 0.8 × 0.1 = 0.08
- 0.2 × 0.0 = 0.00
- 0.1 × 0.0 = 0.00
- 0.6 × 1.0 = 0.60

Step 2:
- 0.08 + 0.00 + 0.00 + 0.60 = **0.68**

Strategy A scores 0.98, Strategy B scores 0.68. Given the current high conjunction risk (0.8) and available comms window (0.6), the conjunction-focused strategy is more strongly indicated by the dot-product scoring.

```python
import torch

o   = torch.tensor([0.8, 0.2, 0.1, 0.6])
w_A = torch.tensor([1.0, 0.3, 0.0, 0.2])
w_B = torch.tensor([0.1, 0.0, 0.0, 1.0])

score_A = torch.dot(o, w_A)
score_B = torch.dot(o, w_B)
print(f"Strategy A score: {score_A.item():.2f}")  # 0.98
print(f"Strategy B score: {score_B.item():.2f}")  # 0.68
```

```rust
# extern crate ndarray;
use ndarray::Array1;

fn main() {
    let o   = Array1::from_vec(vec![0.8_f64, 0.2, 0.1, 0.6]);
    let w_a = Array1::from_vec(vec![1.0_f64, 0.3, 0.0, 0.2]);
    let w_b = Array1::from_vec(vec![0.1_f64, 0.0, 0.0, 1.0]);

    let score_a: f64 = (&o * &w_a).sum();
    let score_b: f64 = (&o * &w_b).sum();
    println!("Strategy A score: {score_a:.2}"); // 0.98
    println!("Strategy B score: {score_b:.2}"); // 0.68
}
```

In a neural network, the weight vectors are learned from data rather than hand-designed. But the scoring mechanism is exactly this dot product. When we stack many weight vectors in the next lesson, we compute scores for many strategies simultaneously. That is a matrix-vector multiplication.

---

## Key Takeaways

* **Scalars, vectors, matrices, and tensors are the same object at different ranks.** PyTorch's `torch.Tensor` represents all of them. The `.shape` attribute is your first diagnostic when something goes wrong — most dimension errors in deep learning code are caught by reading shapes.

* **The L2 norm is the default "length" of a vector**, but L1 and L∞ norms are important in regularization and robust bounds respectively. Kneusel's *Math for Deep Learning* Ch. 5 covers all three in depth. In SSA, L∞ gives you worst-case position error; L1 promotes sparsity in learned feature representations.

* **Unit vectors (norm = 1) represent pure direction.** Normalizing a vector before taking dot products removes the confounding effect of magnitude. The cosine similarity — dot product of two unit vectors — is the standard direction-comparison metric in ML. Always guard against normalizing a zero vector.

* **Projection decomposes a vector into components.** The projection of **v** onto **w** gives the part of **v** aligned with **w**. In SSA, this separates radial from tangential approach velocity — the operationally relevant decomposition for conjunction geometry.

* **The dot product measures alignment.** Positive means same direction, zero means perpendicular, negative means opposite. The neural network's core operation — scoring an input against a weight vector — is a dot product. Everything else (matrix-vector multiply, attention, convolution) reduces to dot products.

* **Cosine similarity is the scale-invariant version of the dot product.** Dividing the dot product by the product of norms removes the magnitude dependence. Two orbital state vectors can have the same direction (same type of orbit) but different magnitudes. Cosine similarity catches the directional similarity; raw dot product does not.

---

{{#quiz 05-vectors-and-dot-products.toml}}
