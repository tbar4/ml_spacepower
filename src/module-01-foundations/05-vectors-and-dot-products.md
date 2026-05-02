# Lesson 5: Vectors and dot products

## Where this fits

We're switching gears from probability to linear algebra. The reason is short: every input to a neural network, every state representation in RL, every action embedding, every observation in OpenSpiel, is a vector. And the single operation a neural network does most often is the dot product. If you understand what a vector means and what a dot product computes, you've understood the inner loop of nearly every deep learning library on Earth.

This lesson and the next are deliberately compressed. We are skipping eigenvalues, matrix inverses, determinants, ranks, and most of what a linear algebra course covers. We pick that up later if we ever need it (alpha-rank in module 6 will force us to learn one extra thing).

## Concept

A **vector** is an ordered list of numbers. That's the whole definition. A vector of length 3 is a triple of numbers. A vector of length 100 is a list of 100 numbers. The numbers themselves can mean anything.

Some examples that show up in our work:

- An orbital state vector: \\((x, y, z, v_x, v_y, v_z)\\), six numbers describing position and velocity.
- A neural network input layer: a vector of feature values describing the current state of the world.
- A neural network output layer for a policy: a vector of action logits, one entry per action.
- An action probability distribution: a vector of probabilities summing to 1.
- A "belief state" over which information set you're in: another vector of probabilities.

The fact that all these things are vectors means we can apply the same operations to all of them. That generality is the whole reason linear algebra is the lingua franca of ML.

## The dot product

Given two vectors \\(\mathbf{v}\\) and \\(\mathbf{w}\\) of the same length \\(n\\), the **dot product** is:

\\[ \mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^{n} v_i w_i \\]

Decoding: pair up the entries, multiply each pair, sum. That's it.

The dot product has a geometric interpretation that's worth carrying around even if you never compute one by hand again:

\\[ \mathbf{v} \cdot \mathbf{w} = |\mathbf{v}| \, |\mathbf{w}| \cos \theta \\]

where \\(|\mathbf{v}|\\) means the **length** (or "norm") of the vector \\(\mathbf{v}\\), and \\(\theta\\) is the angle between the two vectors.

What this tells you:

- If \\(\mathbf{v}\\) and \\(\mathbf{w}\\) point in the same direction (\\(\theta = 0\\), \\(\cos\theta = 1\\)), the dot product is the product of their lengths. Maximally positive.
- If they're perpendicular (\\(\theta = 90°\\), \\(\cos\theta = 0\\)), the dot product is zero. They don't "share direction" at all.
- If they point in opposite directions (\\(\theta = 180°\\), \\(\cos\theta = -1\\)), the dot product is negative.

So the dot product, intuitively, measures **alignment**. Same direction: big positive. Perpendicular: zero. Opposite: big negative. The magnitude is also affected by how long the vectors are.

If you divide the dot product by the lengths to remove that magnitude effect, you get \\(\cos\theta\\) directly, which is called **cosine similarity**:

\\[ \cos\theta = \frac{\mathbf{v} \cdot \mathbf{w}}{|\mathbf{v}| \, |\mathbf{w}|} \\]

Cosine similarity is the version you usually want when you actually care about "how similar are these two things in direction" without caring about magnitude. Embedding-based search, recommender systems, and a lot of NLP machinery rely on it. (We don't, much, but it's good to recognize.)

## Norms

The **norm** of a vector is its length. The most common one is the Euclidean (or "L2") norm:

\\[ |\mathbf{v}| = \sqrt{\sum_i v_i^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}} \\]

Notice that \\(|\mathbf{v}|^2 = \mathbf{v} \cdot \mathbf{v}\\), which is sometimes a useful shortcut. In code, it's just `torch.linalg.norm(v)`.

Other norms exist (L1 is sum of absolute values, L-infinity is max absolute value), but L2 is the default and the one we'll use.

## Code

PyTorch makes vectors very easy. They're just 1-d tensors.

```python
import torch

v = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([4.0, 5.0, 6.0])

# Dot product. Three equivalent ways:
print((v * w).sum().item())     # 32.0
print(torch.dot(v, w).item())   # 32.0
print((v @ w).item())           # 32.0  (matrix-mul operator works for vectors)

# Norms
print(torch.linalg.norm(v).item())  # ~3.7417 = sqrt(1 + 4 + 9)
print(torch.linalg.norm(w).item())  # ~8.7750

# Cosine similarity
cos_sim = torch.dot(v, w) / (torch.linalg.norm(v) * torch.linalg.norm(w))
print(cos_sim.item())  # ~0.9746  (very aligned; small angle)
```

The `@` operator is Python's matrix multiplication operator. For two vectors of the same length, it does a dot product. For other shapes, it does what you'd expect from matrix multiplication. Get used to seeing it; it's everywhere.

## Worked example: relative motion direction

Two satellites have velocity vectors (in some inertial frame, in km/s):

\\[ \mathbf{v}_1 = (7.5, 0.0, 0.0) \quad \mathbf{v}_2 = (5.3, 5.3, 0.0) \\]

Question: what's the angle between their velocity vectors? Are they roughly parallel, roughly perpendicular, or roughly opposed?

By hand:

\\[ \mathbf{v}_1 \cdot \mathbf{v}_2 = 7.5 \cdot 5.3 + 0 \cdot 5.3 + 0 \cdot 0 = 39.75 \\]

\\[ |\mathbf{v}_1| = 7.5, \quad |\mathbf{v}_2| = \sqrt{5.3^2 + 5.3^2} = 5.3 \sqrt{2} \approx 7.495 \\]

\\[ \cos\theta = \frac{39.75}{7.5 \cdot 7.495} \approx 0.7071 \\]

\\(\arccos(0.7071) \approx 45°\\). So the second satellite is moving at roughly 45° to the first. They're both moving in roughly the +x direction (the dot product is positive), but at a substantial angle.

In code:

```python
import torch

v1 = torch.tensor([7.5, 0.0, 0.0])
v2 = torch.tensor([5.3, 5.3, 0.0])

cos_theta = torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))
angle_deg = torch.rad2deg(torch.acos(cos_theta))
print(angle_deg.item())  # ~45.0
```

This kind of "are these motions aligned?" question shows up in basic conjunction analysis (objects on similar trajectories converge slowly; objects on perpendicular or opposing ones converge fast). It's also the same operation that a neural network does internally when it asks "how much does this input look like the pattern stored in this row of weights?"

## Vectors as features, dot products as scores

Here's the bridge to neural networks, which we'll cross properly in the next lesson.

Suppose we want to score states based on how "good" they are. A simple way is to define a weight vector \\(\mathbf{w}\\) of the same length as our state representation \\(\mathbf{s}\\), and define the score as their dot product:

\\[ \text{score}(\mathbf{s}) = \mathbf{w} \cdot \mathbf{s} \\]

This is called a **linear function** of the state, and you can read it as: "each feature contributes some amount to the score, and the weights say how much." If \\(w_i\\) is positive, feature \\(i\\) increases the score when it's positive. If \\(w_i\\) is negative, that feature drags the score down. The dot product just adds up all those contributions.

This is, exactly and literally, what a single neuron in a neural network computes (before adding a bias and an activation function). When you stack many neurons in parallel, you get a matrix-vector product, which is the next lesson.

## Quiz

{{#quiz 05-vectors-and-dot-products.toml}}
