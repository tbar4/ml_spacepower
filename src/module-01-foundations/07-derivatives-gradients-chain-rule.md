# Lesson 7: Derivatives, gradients, and the chain rule

## Where this fits

This is the bridge to module 2. Every learning algorithm we'll meet in the rest of this curriculum trains by some form of gradient descent: compute a loss, compute the gradient of that loss with respect to the parameters, take a small step in the negative gradient direction. The chain rule is the only thing that makes "compute the gradient" tractable when the loss involves a forty-layer composition of operations. Backpropagation is, mathematically, the chain rule applied systematically to a computational graph. If you understand the chain rule visually, you have understood the conceptual content of how neural networks learn. The rest is engineering.

## Concept

A **derivative** of a function \\(f(x)\\) at a point is the slope of \\(f\\) at that point. If you nudge \\(x\\) by a tiny amount \\(\Delta x\\), \\(f(x)\\) changes by approximately \\(f'(x) \cdot \Delta x\\). That's it. The derivative answers the question: "if I move \\(x\\) a little, how much does \\(f\\) change?"

If \\(f'(x)\\) is positive, increasing \\(x\\) increases \\(f\\). Negative, the opposite. Zero, you're at a flat spot (a minimum, maximum, or saddle).

A **partial derivative** generalizes this to functions of multiple variables. For \\(f(x, y)\\), the partial \\(\partial f / \partial x\\) is the slope of \\(f\\) when you nudge \\(x\\) and hold \\(y\\) fixed. Same idea, just one variable at a time.

The **gradient** of a scalar function \\(f(\mathbf{x})\\) of a vector \\(\mathbf{x}\\) is the vector of all the partial derivatives:

\\[ \nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right) \\]

The gradient points in the direction of steepest **increase** of \\(f\\). To minimize \\(f\\), you walk in the direction \\(-\nabla f\\). This is gradient descent, and it's the entire training algorithm of essentially every neural network.

## The chain rule

Here's the operation that makes neural network training possible.

If \\(y = f(u)\\) and \\(u = g(x)\\), so \\(y = f(g(x))\\), then:

\\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \\]

Decoding: the rate at which \\(y\\) changes as \\(x\\) changes is the rate at which \\(y\\) changes as \\(u\\) changes (the "outer derivative") times the rate at which \\(u\\) changes as \\(x\\) changes (the "inner derivative").

The chain rule generalizes naturally to longer compositions. For \\(y = f(g(h(x)))\\):

\\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dv} \cdot \frac{dv}{dx} \\]

where \\(u = g(v)\\) and \\(v = h(x)\\). It just keeps multiplying. A neural network with 50 layers gives you 50 factors in the chain. Backpropagation is the algorithm for computing all those factors efficiently and combining them.

There's a visual intuition that helps. Imagine the function \\(y(x)\\) as a chain of pipes. Each pipe transforms its input into its output, and each pipe has a "sensitivity" (how much its output changes per unit of input). The total sensitivity of the chain is the **product** of the sensitivities of the individual pipes. That's the chain rule.

## Computational graphs

A computational graph is just a way of drawing out a complicated function as a chain (or web) of simple operations. Each node is an operation. Each edge carries a value. Backpropagation traverses the graph backward, applying the chain rule at each node.

Consider \\(y = (2x + 1)^2\\). We can break this into two operations:

1. \\(u = 2x + 1\\) (multiply by 2, add 1).
2. \\(y = u^2\\).

The derivatives of each piece:

- \\(du/dx = 2\\).
- \\(dy/du = 2u\\).

By the chain rule:

\\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 2 = 4u = 4(2x + 1) \\]

Sanity check at \\(x = 1\\): \\(y = (2 \cdot 1 + 1)^2 = 9\\), and \\(dy/dx = 4 \cdot (2 + 1) = 12\\). If we perturb \\(x\\) by 0.001, \\(y\\) becomes \\((2 \cdot 1.001 + 1)^2 = 9.024\ldots\\), a change of about 0.024. And indeed \\(12 \cdot 0.001 = 0.012\\). Wait, that's off by a factor of 2. Let me redo: \\((3.002)^2 = 9.012\\), a change of 0.012. Yes. Chain rule confirmed.

(That kind of "compute analytically, verify numerically by perturbing" is a debugging technique you'll use later when you suspect a custom gradient is wrong. It's called gradient checking.)

## Code: PyTorch autograd

PyTorch's `autograd` system computes derivatives automatically by building the computational graph as you do operations on tensors. To turn on gradient tracking, use `requires_grad=True`.

```python
import torch

x = torch.tensor(1.0, requires_grad=True)

# Forward pass: compute y = (2x + 1)^2
y = (2 * x + 1) ** 2
print(y.item())  # 9.0

# Backward pass: compute dy/dx
y.backward()
print(x.grad.item())  # 12.0
```

That's the entire core of how every deep learning library works. You build the forward computation; the framework records the operations; calling `.backward()` walks the graph in reverse and applies the chain rule at every step. The gradients show up in the `.grad` attribute of each tensor that had `requires_grad=True`.

Multivariate version:

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

# A scalar function of x: f(x) = (x_0 + x_1)^2
f = (x.sum()) ** 2
print(f.item())  # 9.0

f.backward()
print(x.grad)  # tensor([6., 6.])
```

Each entry of `x.grad` is a partial derivative. \\(\partial f / \partial x_0 = 2(x_0 + x_1) \cdot 1 = 6\\) and similarly for \\(x_1\\). The gradient \\(\nabla f = (6, 6)\\) tells you that to make \\(f\\) bigger fastest, you should move both components of \\(x\\) in the positive direction at equal rates.

## Worked example: a one-step gradient descent

Here's a complete (if tiny) training loop. We have a function \\(f(x) = (x - 3)^2\\), which is minimized at \\(x = 3\\). Start at \\(x = 0\\) and walk downhill.

```python
import torch

x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

for step in range(20):
    # Forward: compute the loss.
    loss = (x - 3) ** 2
    
    # Backward: compute dloss/dx.
    loss.backward()
    
    # Step: update x in the direction of -gradient.
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # Reset grad for the next iteration (PyTorch accumulates by default).
    x.grad.zero_()
    
    print(f"step {step:2d}: x = {x.item():.4f}, loss = {loss.item():.4f}")
```

You'll watch \\(x\\) march from 0 toward 3, with the loss shrinking each step. By step 20 you'll be very close to 3. Same algorithm scales, with no conceptual change, to functions of millions of parameters; the only differences are that the gradient is computed by autograd (which is the chain rule, applied to whatever computational graph you built) and the parameters are matrices not scalars.

The `with torch.no_grad():` block tells PyTorch not to track this update as part of the computational graph. The `.grad.zero_()` is a quirk: PyTorch accumulates gradients by default (which is useful for some advanced cases), so you have to clear them between iterations. The optimizer classes (`torch.optim.SGD`, etc.) do this for you in real training code.

## Why "learning rate"?

Notice the `learning_rate = 0.1` in the loop. We don't take a step of size `x.grad`, we take a step of size `learning_rate * x.grad`. Why?

Because the gradient is only a **local linear approximation** to the loss. It tells you the slope right where you are, not what the loss looks like a long way away. If you take too big a step, you may overshoot the minimum and bounce around. If you take too small a step, training is slow but stable.

Choosing learning rates is a black art. There are theoretical guidelines, lots of heuristics, and entire papers about adaptive learning rate schedules. For now: it's a hyperparameter, and you tune it.

## Why this matters going forward

Module 2 will train neural networks. Training a neural network means: define a loss, do a forward pass to compute it, call `.backward()` to get gradients of the loss with respect to all the parameters, and step the parameters in the negative gradient direction. The math is exactly what you just did, scaled up. The chain rule is what makes the gradient computation feasible.

In RL, "policy gradient" methods compute the gradient of expected return with respect to policy parameters, then step in that direction. "Q-learning" methods compute the gradient of a TD error with respect to value function parameters and step. "Actor-critic" methods do both. In CFR with neural net function approximation (deep CFR), gradients show up again. The pattern is universal.

If you remember nothing else from this lesson: gradient descent walks downhill, the chain rule is what lets you compute "downhill" through a long composition of operations, and PyTorch does it for you. The rest of the curriculum is, in a real sense, variations on that theme.

## Quiz

{{#quiz 07-derivatives-gradients-chain-rule.toml}}
