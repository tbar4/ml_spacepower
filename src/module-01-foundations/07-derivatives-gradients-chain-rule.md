# Lesson 7: Derivatives, Gradients, and the Chain Rule

## Where this fits

This is the final foundational lesson before we start building neural networks. Every learning algorithm in the rest of this curriculum trains by gradient descent: compute a loss function, figure out which direction to adjust the parameters to reduce that loss, take a small step in that direction. The gradient is the mathematical object that tells you which direction that is. The chain rule is what makes computing the gradient tractable when the loss function involves a long composition of operations (which it always does in a neural network). Backpropagation is the chain rule applied systematically to a computational graph. If you understand this lesson, the training of neural networks is bookkeeping, not magic.

## What is a derivative? Starting from slope

Suppose you are controlling a satellite's orbit-raising thruster. The altitude of the satellite (in km) after a burn of duration t seconds is given by some function:

```
altitude = h(t)
```

If you increase the burn duration by a tiny amount, how much does the altitude change?

That is the question a derivative answers. The **derivative** of h with respect to t is the rate of change of h as t changes. It tells you:

"If I change t by a tiny amount Δt (delta-t, a small change), the altitude changes by approximately h'(t) × Δt."

The notation \\(h'(t)\\) (read "h prime of t") is one way to write the derivative. Another common notation is \\(\frac{dh}{dt}\\) (read "dh by dt"), which emphasizes that we are asking how much h changes per unit change in t.

### A concrete simple example

Let the altitude function be \\(h(t) = t^2\\) (a simple made-up example for illustration).

At \\(t = 3\\) seconds:
- Current altitude: \\(h(3) = 3^2 = 9\\) km
- One second later: \\(h(4) = 4^2 = 16\\) km
- Change over 1 second: 16 - 9 = 7 km

If we zoom in to a much smaller interval (0.01 seconds):
- \\(h(3.01) = 3.01^2 = 9.0601\\) km
- Change over 0.01 seconds: 9.0601 - 9 = 0.0601 km
- Rate of change: 0.0601 / 0.01 = 6.01 km/s

If we zoom in even more (0.001 seconds):
- \\(h(3.001) = 3.001^2 = 9.006001\\) km
- Change over 0.001 seconds: 9.006001 - 9 = 0.006001 km
- Rate of change: 0.006001 / 0.001 = 6.001 km/s

As we take smaller and smaller intervals, the rate of change converges to exactly 6 km/s. **The derivative of \\(h(t) = t^2\\) at \\(t = 3\\) is 6.**

This derivative can be computed using calculus rules that you do not need to derive yourself. For the function \\(h(t) = t^2\\), the derivative is \\(h'(t) = 2t\\). At \\(t = 3\\): \\(h'(3) = 2 \times 3 = 6\\). This matches what we computed numerically.

**The derivative as slope**: if you plotted \\(h(t)\\) on a graph and drew a tangent line at \\(t = 3\\), the slope of that tangent line would be 6. That is all a derivative is: the slope of the function at a specific point.

Key interpretations of the derivative:

- **Positive derivative**: the function is increasing at this point. Moving t in the positive direction increases h.
- **Negative derivative**: the function is decreasing. Moving t in the positive direction decreases h.
- **Zero derivative**: you are at a flat spot (a local minimum, local maximum, or saddle point).

## Partial derivatives: functions of multiple inputs

In lesson 5 you saw that a sensor scoring function might take a 4-dimensional input:

```
score = f(conjunction_risk, debris_density, solar_activity, comms_window)
score = f(x₁, x₂, x₃, x₄)
```

The output depends on all four inputs simultaneously. A derivative asks "how does the output change if I vary one input?" When there are multiple inputs, we need to specify which input we are varying. That is what a **partial derivative** does.

The partial derivative of \\(f\\) with respect to \\(x_1\\) is written \\(\frac{\partial f}{\partial x_1}\\) (using the curly ∂ instead of d to indicate "partial"). It means: "how does f change if I vary \\(x_1\\) while holding \\(x_2, x_3, x_4\\) fixed?"

The curly ∂ symbol (called "del" or "partial") is just a stylistic convention to distinguish partial derivatives from regular derivatives. It means the same thing: rate of change, but with respect to one specific variable.

### A concrete partial derivative example

Let us use a simple two-variable function: the combined risk score:

\\[ \text{risk} = f(c, d) = c^2 + 2cd + d \\]

where c = conjunction_risk and d = debris_density.

At the point (c = 0.5, d = 0.3):
- \\(f(0.5, 0.3) = 0.5^2 + 2(0.5)(0.3) + 0.3 = 0.25 + 0.30 + 0.30 = 0.85\\)

**Partial derivative with respect to c** (∂f/∂c):

Treat d as a constant and differentiate with respect to c:
- \\(c^2\\) differentiates to \\(2c\\)
- \\(2cd\\) differentiates to \\(2d\\) (since d is treated as a constant)
- \\(d\\) differentiates to \\(0\\) (no c dependence)

Result: \\(\frac{\partial f}{\partial c} = 2c + 2d\\)

At our point: \\(2(0.5) + 2(0.3) = 1.0 + 0.6 = 1.6\\)

This means: if I increase conjunction_risk by a small amount while keeping debris_density fixed, the risk score increases by approximately 1.6 times that amount.

**Partial derivative with respect to d** (∂f/∂d):

Treat c as a constant:
- \\(c^2\\) → 0 (no d dependence)
- \\(2cd\\) → 2c (c is constant)
- \\(d\\) → 1

Result: \\(\frac{\partial f}{\partial d} = 2c + 1\\)

At our point: \\(2(0.5) + 1 = 2.0\\)

This means: if I increase debris_density by a small amount while keeping conjunction_risk fixed, the risk score increases by approximately 2.0 times that amount.

## The gradient: all partial derivatives together

The **gradient** collects all the partial derivatives of a function into a single vector. For a function \\(f(x_1, x_2, \ldots, x_n)\\), the gradient is:

\\[ \nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right) \\]

**Decoding:**

**\\(\nabla f\\)**: The gradient of f. The symbol ∇ is called "nabla" or "del." Read it as "the gradient of f" or "grad f."

**\\(\left( \ldots \right)\\)**: A vector (the parentheses enclose the components of the gradient vector).

**\\(\frac{\partial f}{\partial x_i}\\)**: The partial derivative with respect to the i-th input. This is one component of the gradient vector.

**Reading in English**: "The gradient is a vector containing the partial derivative of f with respect to each of its inputs."

For our risk function at (c = 0.5, d = 0.3):

\\[ \nabla f = \left( \frac{\partial f}{\partial c}, \frac{\partial f}{\partial d} \right) = (1.6, 2.0) \\]

**What the gradient tells you**: the gradient points in the direction that increases f most steeply. Each component of the gradient tells you how sensitive f is to changes in that input. A large gradient component means f is very sensitive to that input. A small component means f barely changes when that input changes.

For gradient descent (the training algorithm for neural networks), you want to minimize f (the loss). So you move in the direction **opposite** to the gradient: decrease each parameter by a small multiple of the gradient component for that parameter. That small multiple is the learning rate.

## The chain rule: derivatives through compositions

Neural networks are compositions of functions. The input goes through layer 1, then layer 2, then layer 3, and so on. Each layer applies \\(W\mathbf{x} + \mathbf{b}\\) followed by a nonlinearity. If you want to compute how the final output (the loss) changes as you change a weight in layer 1, you need to trace the effect all the way through every subsequent layer.

The **chain rule** tells you how to do this.

**Simple case: two composed functions**

If \\(y = f(u)\\) and \\(u = g(x)\\), so overall \\(y = f(g(x))\\), then:

\\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \\]

**Decoding:**

**\\(\frac{dy}{dx}\\)**: How much does y change per unit change in x? This is what we want.

**\\(\frac{dy}{du}\\)**: How much does y change per unit change in u? We can compute this from the definition of f.

**\\(\frac{du}{dx}\\)**: How much does u change per unit change in x? We can compute this from the definition of g.

**The multiplication**: the rate of change of y with respect to x is the product of these two rates.

**Intuition with a pipeline analogy**: imagine water flowing through two pipes in series. The first pipe takes in x and outputs u, with flow rate 3 (meaning 3 units of u per unit of x). The second pipe takes u and outputs y, with flow rate 2 (2 units of y per unit of u). If x increases by 1, u increases by 3, and then y increases by 3 × 2 = 6. The total rate from x to y is the product of the individual rates: 3 × 2 = 6.

### Working through an example

Let \\(y = (2x + 1)^2\\).

We can split this into two operations:
1. \\(u = 2x + 1\\) (the inner function g)
2. \\(y = u^2\\) (the outer function f)

**Step 1: find du/dx**

\\(u = 2x + 1\\). The derivative of 2x is 2 (constant times x), and the derivative of 1 is 0. So:

\\[ \frac{du}{dx} = 2 \\]

**Step 2: find dy/du**

\\(y = u^2\\). The derivative of \\(u^2\\) with respect to u is \\(2u\\):

\\[ \frac{dy}{du} = 2u \\]

**Step 3: apply the chain rule**

\\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 2 = 4u = 4(2x + 1) \\]

**Step 4: verify with a numerical check**

At \\(x = 1\\): \\(y = (2 \cdot 1 + 1)^2 = 9\\)
At \\(x = 1.001\\): \\(y = (2 \cdot 1.001 + 1)^2 = (3.002)^2 = 9.012\\)

Rate of change ≈ (9.012 - 9) / 0.001 = 12.0

Analytic answer at x = 1: \\(4(2 \cdot 1 + 1) = 4 \times 3 = 12\\). They match.

```python
import torch

x = torch.tensor(1.0, requires_grad=True)
y = (2 * x + 1) ** 2

y.backward()  # PyTorch computes the derivative using the chain rule internally
print(f"dy/dx at x=1: {x.grad.item()}")  # 12.0
```

## How PyTorch computes gradients automatically

When you write `y.backward()`, PyTorch walks backward through the computational graph it recorded during the forward pass, applying the chain rule at each operation. This is **backpropagation**.

The graph for \\(y = (2x + 1)^2\\) looks like:

```
x → [multiply by 2] → [add 1] → u → [square] → y
```

Going backward (from right to left):
- Start at y. We want dy/dy = 1.
- Apply chain rule through "square": dy/du = 2u
- Apply chain rule through "add 1": du/(u_before_add) = 1 (adding a constant does not change the rate)
- Apply chain rule through "multiply by 2": d(u_before_add)/dx = 2
- Total: dy/dx = 1 × 2u × 1 × 2 = 4u = 4(2x+1) = 12 at x=1

Every neural network, regardless of how many layers, uses this same backward walk through the computational graph. The graph is more complex (involving matrices and nonlinearities), but the principle is identical.

## A complete training step

Here is a full gradient descent step on a simple problem, showing every part:

**Problem**: find the value of x that minimizes \\(L(x) = (x - 3)^2\\). The minimum is clearly at x = 3, but we will find it by gradient descent.

```python
import torch

# Start with an initial guess
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.2

print("Starting gradient descent to minimize L(x) = (x - 3)^2")
print(f"{'Step':>5} | {'x':>8} | {'L(x)':>8} | {'dL/dx':>8}")
print("-" * 40)

for step in range(10):
    # Forward pass: compute the loss
    L = (x - 3) ** 2
    
    # Backward pass: compute dL/dx using the chain rule
    L.backward()
    
    # Read the gradient
    gradient = x.grad.item()
    
    # Print the current state
    print(f"{step:>5} | {x.item():>8.4f} | {L.item():>8.4f} | {gradient:>8.4f}")
    
    # Update step: move x in the opposite direction of the gradient
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # Reset the gradient for the next iteration
    x.grad.zero_()
```

What you will see:

At step 0: x = 0, L = 9, gradient = -6. The gradient is negative, meaning increasing x decreases L. So we add a positive amount to x: x += 0.2 × 6 = 1.2. New x = 1.2.

At step 1: x = 1.2, L = 3.24, gradient = -3.6. Still moving toward x = 3. New x = 1.2 + 0.2 × 3.6 = 1.92.

Each step, x gets closer to 3 and L gets closer to 0. By step 10, x is very close to 3.

**Why `with torch.no_grad():`?** When we update x, we do not want PyTorch to record this update as part of the computational graph. That context manager tells PyTorch to pause its graph-recording temporarily.

**Why `x.grad.zero_()`?** PyTorch accumulates gradients by default (adds new gradients to existing ones). In a training loop, you almost always want a fresh gradient each step, so you clear it before the next forward pass.

## The learning rate: why not take the full gradient step?

Notice we multiplied the gradient by `learning_rate = 0.2` instead of just subtracting the gradient directly. Why?

The gradient tells you the slope at your current location. It is a local approximation: it is accurate close to where you are, but the function might curve away from the linear approximation if you move too far.

If you take too large a step, you might overshoot the minimum and end up on the other side, potentially further away than you started. A small learning rate keeps you in the regime where the linear approximation is trustworthy.

Choosing the learning rate is one of the most practically important decisions in training a neural network. Too large and training oscillates or diverges. Too small and training is painfully slow. We will discuss this more in module 2 when we actually train networks.

## Why this matters for the rest of the curriculum

Every algorithm from here on trains by gradient descent:

- **Policy gradient methods** (module 3): compute the gradient of expected return with respect to policy parameters, step parameters in the positive direction (we want to maximize, not minimize).
- **Q-learning with neural networks** (module 3): compute the gradient of a temporal-difference error with respect to value function parameters, step to reduce the error.
- **Deep CFR** (module 5): compute gradients of a regret prediction loss and step to make regret predictions more accurate.

In each case, you will write a forward pass (compute the loss from the current parameters), call `.backward()` (chain rule through the computational graph), and update the parameters (subtract learning_rate × gradient). The specific loss function and what you are minimizing will differ. The gradient descent structure will be the same.

## Quiz

{{#quiz 07-derivatives-gradients-chain-rule.toml}}
