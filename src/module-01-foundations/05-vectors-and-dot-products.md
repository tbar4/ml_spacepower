# Lesson 5: Vectors and Dot Products

## Where this fits

Every state in a reinforcement learning system, every observation your agent receives, every action embedding, every intermediate representation inside a neural network, is a vector. The dot product is the single most common operation performed on those vectors: it is how a neural network layer evaluates whether its input "matches" a learned pattern. If you can look at a vector and say what it represents, and look at a dot product and say what it is measuring, you have the geometric intuition for 90% of what deep learning does internally.

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

## The length of a vector: norms

Before we talk about dot products, we need to know how to measure the length of a vector.

For a 2D vector [a, b], the Pythagorean theorem gives the length: √(a² + b²). A vector [3, 4] has length √(9 + 16) = √25 = 5.

For a vector of any length [v₁, v₂, ..., vₙ], the same idea extends:

\\[ \|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2} \\]

**Decoding each symbol:**

**\\(\|\mathbf{v}\|\\)**: The "norm" or "magnitude" of vector v. The double vertical bars mean "length of." The bold v (as opposed to a plain v) indicates that v is a vector rather than a single number.

**\\(\sqrt{\ldots}\\)**: Square root.

**\\(v_i^2\\)**: The i-th component of the vector, squared. The subscript i selects one component.

**\\(\sum_{i=1}^{n}\\)**: Sum all n components.

**In plain English**: "Square every component, add them all up, take the square root." This is the Euclidean distance from the origin to the point represented by the vector.

For the orbital state vector example:

```
v = [6371.0, 500.0, -200.0, 7.2, 0.3, -0.1]
||v|| = sqrt(6371^2 + 500^2 + 200^2 + 7.2^2 + 0.3^2 + 0.1^2)
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

## The dot product: measuring alignment

The **dot product** of two vectors of the same length is computed by:

1. Multiplying corresponding components together
2. Adding all the products

For vectors **v** = [v₁, v₂, ..., vₙ] and **w** = [w₁, w₂, ..., wₙ]:

\\[ \mathbf{v} \cdot \mathbf{w} = v_1 w_1 + v_2 w_2 + \ldots + v_n w_n = \sum_{i=1}^{n} v_i w_i \\]

**Decoding:**

**\\(\mathbf{v} \cdot \mathbf{w}\\)**: "The dot product of v and w." The bold letters indicate vectors. The centered dot is the dot product operation (not regular multiplication, which would give a vector).

**\\(v_i w_i\\)**: Component i of v times component i of w. Subscripts connect corresponding components.

**\\(\sum_{i=1}^{n} v_i w_i\\)**: Add up all those pairwise products.

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

## What the dot product is measuring: alignment

The arithmetic definition is straightforward. But what does the dot product actually tell us?

The dot product has a geometric interpretation:

\\[ \mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \cdot \|\mathbf{w}\| \cdot \cos(\theta) \\]

where \\(\theta\\) (the Greek letter theta) is the angle between the two vectors.

**Decoding:**

**\\(\|\mathbf{v}\|\\)**: The norm (length) of v.
**\\(\|\mathbf{w}\|\\)**: The norm (length) of w.
**\\(\cos(\theta)\\)**: The cosine of the angle between them.

You do not need to remember the details of cosine, but you need to know these key facts:
- cos(0°) = 1: vectors pointing in exactly the same direction
- cos(90°) = 0: vectors that are perpendicular (at right angles)
- cos(180°) = -1: vectors pointing in exactly opposite directions

So what does the dot product tell you?

- **Large positive dot product**: the vectors point in roughly the same direction
- **Zero dot product**: the vectors are perpendicular (completely "unrelated" in direction)
- **Large negative dot product**: the vectors point in roughly opposite directions

**In SSA terms**: if two satellites have velocity vectors with a large positive dot product, they are moving in roughly the same direction. Their relative speed is low and they will not approach each other quickly. If their dot products are large negative, they are moving toward each other on nearly head-on trajectories: higher collision risk. This is not how real conjunction analysis works, but the intuition is correct.

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

The head-on geometry gives a highly negative cosine (angle ≈ 180°), the crossing geometry gives zero (exactly 90°), and the overtaking geometry gives a high positive value (small angle, similar direction).

## Dot products as scoring: the bridge to neural networks

Here is the connection to machine learning that makes the dot product so important.

Suppose you want to score how much an observation "favors" a particular action. For example, you are operating a sensor, and based on the current observation vector (describing the state of the space environment), you want to score each possible pointing action.

You define a **weight vector** \\(\mathbf{w}\\) for each action. The weight vector describes what kind of observation the action is best suited for. The score for taking that action given observation \\(\mathbf{o}\\) is the dot product \\(\mathbf{w} \cdot \mathbf{o}\\).

If the observation looks like what the weight vector describes (same direction, high alignment), the score is high. If the observation is perpendicular or opposite to the weight vector, the score is low or negative.

This is exactly what a single neuron in a neural network computes. The neuron has a learned weight vector. Its output is the dot product of the weight vector and the input. The network learns weight vectors that give high scores to the kinds of inputs that should lead to good outputs.

In the next lesson, we will stack many neurons in parallel. That will give us matrix-vector multiplication: the operation that defines a neural network layer.

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

In a neural network, the weight vectors are learned from data rather than hand-designed. But the scoring mechanism is exactly this dot product. When we stack many weight vectors in the next lesson, we compute scores for many strategies simultaneously. That is a matrix-vector multiplication.

## Quiz

{{#quiz 05-vectors-and-dot-products.toml}}
