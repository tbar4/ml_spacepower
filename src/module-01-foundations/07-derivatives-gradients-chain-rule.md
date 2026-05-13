# Lesson 7: Derivatives, Gradients, and the Chain Rule

**Module:** ML Foundations — M01: Mathematical Foundations
**Source:** *Math for Deep Learning* — Ronald T. Kneusel, Ch. 7–8 (Calculus and Automatic Differentiation); *Bayesian Statistics the Fun Way* — Will Kurt, Ch. 13 (prior-updating as gradient-like steps); PyTorch autograd documentation

---


<!-- toc -->

## Where this fits

This is the final foundational lesson before we start building neural networks. Every learning algorithm in the rest of this curriculum trains by gradient descent: compute a loss function, figure out which direction to adjust the parameters to reduce that loss, take a small step in that direction. The gradient is the mathematical object that tells you which direction that is. The chain rule is what makes computing the gradient tractable when the loss function involves a long composition of operations (which it always does in a neural network). Backpropagation is the chain rule applied systematically to a computational graph. If you understand this lesson, the training of neural networks is bookkeeping, not magic.

---

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

---

## The formal limit definition

The "zooming in" process in the example above has a formal mathematical expression. The derivative is defined as:

\\[ h'(t) = \lim_{\Delta t \to 0} \frac{h(t + \Delta t) - h(t)}{\Delta t} \\]

**Decoding:**

**\\(\lim_{\Delta t \to 0}\\)**: "The limit as Δt approaches zero." We are taking the ratio \\(\frac{h(t + \Delta t) - h(t)}{\Delta t}\\) and seeing what value it converges to as Δt gets arbitrarily small.

**\\(h(t + \Delta t) - h(t)\\)**: The change in h when t increases by Δt — the numerator of the ratio.

**\\(\Delta t\\)**: The change in t — the denominator.

**The whole fraction**: the average rate of change of h over a small interval [t, t + Δt]. As Δt → 0, this converges to the instantaneous rate of change: the derivative.

This formula is not just academic. It is exactly what **numerical gradient checking** computes — an approximation of the derivative by using a very small but finite Δt (called epsilon). The **central difference approximation** is more accurate than the one-sided formula above:

\\[ h'(t) \approx \frac{h(t + \varepsilon) - h(t - \varepsilon)}{2\varepsilon} \\]

**Decoding:** By using \\(t + \varepsilon\\) and \\(t - \varepsilon\\) symmetrically around \\(t\\), we cancel the first-order error term. The result is accurate to \\(O(\varepsilon^2)\\) rather than \\(O(\varepsilon)\\).

```python
import torch

def numerical_gradient(f, x: float, eps: float = 1e-5) -> float:
    """Central difference approximation of the derivative of f at x."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Test on a few known functions
def h(t):     return t ** 2          # true derivative: 2t
def g(t):     return t ** 3 - 2 * t  # true derivative: 3t^2 - 2
def sigmoid(t): return 1 / (1 + 2.718281828 ** (-t))  # true derivative: σ(t)(1-σ(t))

t0 = 3.0
print(f"h(t) = t^2 at t={t0}:")
print(f"  Numerical: {numerical_gradient(h, t0):.8f}")
print(f"  Analytic:  {2 * t0:.8f}")   # 6.0

t1 = 2.0
print(f"\ng(t) = t^3 - 2t at t={t1}:")
print(f"  Numerical: {numerical_gradient(g, t1):.8f}")
print(f"  Analytic:  {3 * t1**2 - 2:.8f}")   # 10.0

t2 = 0.5
s = sigmoid(t2)
print(f"\nσ(t) at t={t2}:")
print(f"  Numerical: {numerical_gradient(sigmoid, t2):.8f}")
print(f"  Analytic:  {s * (1 - s):.8f}")
```

The numerical gradient is pure arithmetic — no ML library needed. The Rust version requires no external crates:

```rust
fn numerical_gradient(f: impl Fn(f64) -> f64, x: f64, eps: f64) -> f64 {
    (f(x + eps) - f(x - eps)) / (2.0 * eps)
}

fn h(t: f64) -> f64 { t * t }                            // true derivative: 2t
fn g(t: f64) -> f64 { t * t * t - 2.0 * t }              // true derivative: 3t^2 - 2
fn sigmoid(t: f64) -> f64 { 1.0 / (1.0 + (-t).exp()) }   // true derivative: σ(t)(1-σ(t))

fn main() {
    let eps = 1e-5_f64;

    let t0 = 3.0_f64;
    println!("h(t) = t^2 at t={t0}:");
    println!("  Numerical: {:.8}", numerical_gradient(h, t0, eps));
    println!("  Analytic:  {:.8}", 2.0 * t0);                    // 6.0

    let t1 = 2.0_f64;
    println!("g(t) = t^3 - 2t at t={t1}:");
    println!("  Numerical: {:.8}", numerical_gradient(g, t1, eps));
    println!("  Analytic:  {:.8}", 3.0 * t1 * t1 - 2.0);         // 10.0

    let t2 = 0.5_f64;
    let s = sigmoid(t2);
    println!("σ(t) at t={t2}:");
    println!("  Numerical: {:.8}", numerical_gradient(sigmoid, t2, eps));
    println!("  Analytic:  {:.8}", s * (1.0 - s));
}
```

Functions are passed as `impl Fn(f64) -> f64` — any closure or named function fits. `(-t).exp()` computes \\(e^{-t}\\) using `f64::exp`.

Kneusel's *Math for Deep Learning* Ch. 8 covers numerical differentiation in depth and explains when the finite difference approximation can fail due to floating-point precision (when ε is too small, subtraction of nearly equal numbers loses precision).

---

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

---

## Differentiation rules cheat sheet

You do not need to re-derive derivatives from the limit definition every time. Calculus gives us a set of rules that cover all the common cases. Here are the ones you will encounter throughout this curriculum.

| Rule | Formula | Example |
|------|---------|---------|
| **Power rule** | \\(\frac{d}{dx}[x^n] = n x^{n-1}\\) | \\(\frac{d}{dx}[x^3] = 3x^2\\) |
| **Constant rule** | \\(\frac{d}{dx}[c] = 0\\) | \\(\frac{d}{dx}[5] = 0\\) |
| **Sum rule** | \\(\frac{d}{dx}[f + g] = f' + g'\\) | \\(\frac{d}{dx}[x^2 + x] = 2x + 1\\) |
| **Product rule** | \\(\frac{d}{dx}[f \cdot g] = f'g + fg'\\) | \\(\frac{d}{dx}[x^2 \cdot \sin x] = 2x\sin x + x^2 \cos x\\) |
| **Chain rule** | \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\) | \\(\frac{d}{dx}[(2x+1)^2] = 2(2x+1) \cdot 2\\) |
| **Exponential** | \\(\frac{d}{dx}[e^x] = e^x\\) | The exponential is its own derivative |
| **Natural log** | \\(\frac{d}{dx}[\ln x] = \frac{1}{x}\\) | \\(\frac{d}{dx}[\ln(wx)] = \frac{1}{x}\\) for constant w |

**Derivatives of common ML activation functions:**

| Function | Definition | Derivative |
|----------|-----------|-----------|
| **Sigmoid** \\(\sigma(x)\\) | \\(\frac{1}{1 + e^{-x}}\\) | \\(\sigma(x)(1 - \sigma(x))\\) |
| **ReLU** | \\(\max(0, x)\\) | \\(1\\) if \\(x > 0\\), else \\(0\\) |
| **Tanh** | \\(\frac{e^x - e^{-x}}{e^x + e^{-x}}\\) | \\(1 - \tanh^2(x)\\) |
| **Leaky ReLU** | \\(x\\) if \\(x > 0\\), else \\(\alpha x\\) | \\(1\\) if \\(x > 0\\), else \\(\alpha\\) |
| **Softplus** | \\(\ln(1 + e^x)\\) | \\(\sigma(x)\\) (sigmoid!) |

Two key observations: (1) The derivative of sigmoid is expressible in terms of sigmoid itself — very convenient for backpropagation, since you already have \\(\sigma(x)\\) from the forward pass. (2) ReLU's derivative is undefined exactly at 0, but in practice PyTorch returns 0 there, and it does not matter numerically.

```python
import torch

x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

# Sigmoid and its derivative
sigma = torch.sigmoid(x)
dsigma_dx = sigma * (1 - sigma)
print("Sigmoid values:    ", sigma.tolist())
print("Sigmoid gradients: ", dsigma_dx.tolist())

# ReLU and its derivative
relu_out  = torch.relu(x)
drelu_dx  = (x > 0).float()   # 1 where x > 0, 0 elsewhere
print("\nReLU values:    ", relu_out.tolist())
print("ReLU gradients: ", drelu_dx.tolist())

# Tanh and its derivative
tanh_out  = torch.tanh(x)
dtanh_dx  = 1 - tanh_out ** 2
print("\nTanh values:    ", tanh_out.tolist())
print("Tanh gradients: ", dtanh_dx.tolist())

# Verify: PyTorch autograd agrees with manual formulas
x_grad = torch.tensor(0.5, requires_grad=True)
s = torch.sigmoid(x_grad)
s.backward()
print(f"\nAutograd sigmoid'(0.5):  {x_grad.grad.item():.8f}")
manual_s = torch.sigmoid(torch.tensor(0.5))
print(f"Manual  sigmoid'(0.5):   {(manual_s * (1 - manual_s)).item():.8f}")
```

The derivative formulas are the same computation regardless of framework. Cargo dependency: `ndarray = "0.17"` (same as lessons 5 and 6).

```rust
# extern crate ndarray;
use ndarray::Array1;

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn relu(x: f64) -> f64 { x.max(0.0) }

fn main() {
    let x = Array1::from_vec(vec![-2.0_f64, -0.5, 0.0, 0.5, 2.0]);

    // Sigmoid and its derivative σ(x)(1 - σ(x))
    let sigma   = x.mapv(sigmoid);
    let dsigma  = sigma.mapv(|s| s * (1.0 - s));
    println!("Sigmoid values:    {:?}", sigma.as_slice().unwrap());
    println!("Sigmoid gradients: {:?}", dsigma.as_slice().unwrap());

    // ReLU and its derivative: 1 if x > 0, else 0
    let relu_out = x.mapv(relu);
    let drelu    = x.mapv(|v| if v > 0.0 { 1.0_f64 } else { 0.0 });
    println!("\nReLU values:    {:?}", relu_out.as_slice().unwrap());
    println!("ReLU gradients: {:?}", drelu.as_slice().unwrap());

    // Tanh and its derivative 1 - tanh²(x)
    let tanh_out = x.mapv(f64::tanh);
    let dtanh    = tanh_out.mapv(|t| 1.0 - t * t);
    println!("\nTanh values:    {:?}", tanh_out.as_slice().unwrap());
    println!("Tanh gradients: {:?}", dtanh.as_slice().unwrap());
}
```

The PyTorch autograd verification (`.backward()`) has no equivalent here — that is the point: these formulas are just math, not framework magic. The autograd system computes the same values by applying the same formulas automatically during the backward pass.

---

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

---

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

---

## Numerical gradient checking

When you implement a custom loss function or a custom neural network layer, there is a powerful way to verify that your analytic gradient is correct: compare it to a numerically approximated gradient.

The idea: compute the gradient analytically (via your code or autograd), then compute it numerically using the central difference approximation. If they match closely, your gradient is probably correct.

**The test statistic:**

\\[ \text{relative error} = \frac{|\text{grad}_{analytic} - \text{grad}_{numeric}|}{\max(|\text{grad}_{analytic}|, |\text{grad}_{numeric}|)} \\]

If this is below \\(10^{-5}\\), your gradients are almost certainly correct. Between \\(10^{-5}\\) and \\(10^{-3}\\), suspect a bug. Above \\(10^{-3}\\), you have a bug.

```python
import torch

def my_loss(x: torch.Tensor, target: float) -> torch.Tensor:
    """
    Custom loss: Huber loss variant (smooth L1).
    Acts like L2 for small errors, L1 for large errors.
    Useful in SSA for robust estimation against outlier observations.
    """
    delta = 1.0
    err = x - target
    return torch.where(
        err.abs() < delta,
        0.5 * err ** 2,
        delta * (err.abs() - 0.5 * delta)
    )

def grad_check(f, x_val: float, eps: float = 1e-5) -> dict:
    """Check analytic gradient against numerical gradient for a scalar function."""
    # Analytic gradient via autograd
    x_analytic = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
    loss = f(x_analytic)
    loss.backward()
    analytic = x_analytic.grad.item()

    # Numerical gradient via central difference
    x_plus  = torch.tensor(x_val + eps, dtype=torch.float64)
    x_minus = torch.tensor(x_val - eps, dtype=torch.float64)
    numeric = (f(x_plus).item() - f(x_minus).item()) / (2 * eps)

    # Relative error
    denom = max(abs(analytic), abs(numeric), 1e-8)
    rel_err = abs(analytic - numeric) / denom

    return {
        "analytic":      analytic,
        "numeric":       numeric,
        "relative_error": rel_err,
        "pass":          rel_err < 1e-5
    }

target = 2.0
for x_test in [-1.0, 0.5, 2.0, 2.5, 5.0]:
    f = lambda x: my_loss(x, target)
    result = grad_check(f, x_test)
    status = "PASS" if result["pass"] else "FAIL"
    print(f"x={x_test:5.1f}: analytic={result['analytic']:+.6f}, "
          f"numeric={result['numeric']:+.6f}, "
          f"rel_err={result['relative_error']:.2e}  [{status}]")
```

The gradient check itself is pure math — no autograd needed. The Rust version implements the Huber loss and its analytic gradient directly, then compares against the central difference:

```rust
fn huber_loss(x: f64, target: f64, delta: f64) -> f64 {
    let err = x - target;
    if err.abs() < delta { 0.5 * err * err } else { delta * (err.abs() - 0.5 * delta) }
}

fn huber_grad(x: f64, target: f64, delta: f64) -> f64 {
    let err = x - target;
    if err.abs() < delta { err } else { delta * err.signum() }
}

fn numerical_gradient(f: impl Fn(f64) -> f64, x: f64, eps: f64) -> f64 {
    (f(x + eps) - f(x - eps)) / (2.0 * eps)
}

fn main() {
    let target = 2.0_f64;
    let delta  = 1.0_f64;
    let eps    = 1e-5_f64;

    println!("{:>6} | {:>12} | {:>12} | {:>10} | pass",
             "x", "analytic", "numeric", "rel_err");
    println!("{}", "-".repeat(55));

    for &x_test in &[-1.0_f64, 0.5, 2.0, 2.5, 5.0] {
        let analytic = huber_grad(x_test, target, delta);
        let numeric  = numerical_gradient(|x| huber_loss(x, target, delta), x_test, eps);
        let denom    = analytic.abs().max(numeric.abs()).max(1e-8);
        let rel_err  = (analytic - numeric).abs() / denom;
        println!("{x_test:>6.1} | {analytic:>+12.6} | {numeric:>+12.6} | {rel_err:>10.2e} | {}",
                 if rel_err < 1e-5 { "PASS" } else { "FAIL" });
    }
}
```

`err.signum()` returns -1.0, 0.0, or 1.0 — Rust's built-in sign function for `f64`. The closure `|x| huber_loss(x, target, delta)` captures `target` and `delta` from the enclosing scope, making it a `Fn(f64) -> f64` that `numerical_gradient` accepts.

In practice, when implementing a new loss function or custom layer for SSA (for example, a conjunction probability loss that uses orbital mechanics), running gradient checks like this before training saves enormous debugging time. If the check fails, the analytical gradient in your code is wrong — not the numerical one.

> **When gradient checking fails:** check for (1) sign errors in the chain rule, (2) missing terms in a sum, (3) wrong branching in piecewise functions (like ReLU or Huber loss at the boundary), (4) operations that are intentionally not differentiable being included in the loss.

---

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

---

## Jacobians

So far we have taken derivatives of scalar-valued functions: one number in, one number out (or a vector in, one number out). But what if the function maps a vector to a vector?

For a function \\(f: \mathbb{R}^n \to \mathbb{R}^m\\) (n inputs, m outputs), the derivative is a matrix called the **Jacobian**:

\\[ J_{ij} = \frac{\partial f_i}{\partial x_j} \\]

The Jacobian is an m × n matrix. Row i corresponds to output i. Column j corresponds to input j. Entry [i, j] is the partial derivative of output i with respect to input j.

**Decoding:** The Jacobian generalizes "slope" to vector-valued functions. Where a scalar derivative tells you "how much does this one output change per unit change in this one input?", the Jacobian tells you "how does each output change per unit change in each input?" The gradient \\(\nabla f\\) is the special case where m = 1 (one output): it is a 1 × n Jacobian, which we normally write as a length-n vector.

In backpropagation, when a layer transforms a vector (not a scalar), the Jacobian appears in the gradient calculation. The gradient of the loss with respect to the layer's input is the layer's Jacobian transposed, times the gradient of the loss with respect to the layer's output.

```python
import torch

# Coordinate transformation from Cartesian to spherical (simplified 2D example)
# Input: [x, y]  (Cartesian position in km)
# Output: [r, theta]  (range and angle)
def cartesian_to_polar(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    r     = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack([r, theta])

# Compute the Jacobian at a specific point using torch.autograd.functional.jacobian
point = torch.tensor([3.0, 4.0])   # position in km, Cartesian

J = torch.autograd.functional.jacobian(cartesian_to_polar, point)
print("Jacobian of (r, theta) w.r.t. (x, y):")
print(J)
print(f"Shape: {J.shape}")   # (2, 2): 2 outputs, 2 inputs

# Verify one entry manually: d(r)/d(x) = x / sqrt(x^2 + y^2)
x_val, y_val = point
r_val = torch.sqrt(x_val**2 + y_val**2)
dr_dx_manual = x_val / r_val
print(f"\nJ[0,0] = d(r)/d(x): {J[0,0].item():.6f}")
print(f"Manual:              {dr_dx_manual.item():.6f}")
```

The Jacobian of coordinate transformations between reference frames (Cartesian to spherical, ECI to RSW, etc.) appears throughout orbit determination. When a Kalman filter propagates uncertainty through a nonlinear measurement model, it uses the Jacobian of the measurement function — this is the "H matrix" in the extended Kalman filter.

---

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

---

## SGD, full-batch, and mini-batch gradient descent

In the single-variable example above, we computed the gradient using the entire problem (one point). In realistic ML problems, you have a dataset of N examples and a loss that averages over them:

\\[ L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(\theta; x_i, y_i) \\]

There are three strategies for computing this gradient:

**Full-batch gradient descent:** compute the gradient using all N examples, then update the parameters once. The gradient is exact, but one update requires processing the entire dataset — slow for large N.

**Stochastic gradient descent (SGD):** pick one random example, compute its gradient, update. Fast (one example per update) but very noisy — a single example may not be representative of the whole dataset.

**Mini-batch SGD:** pick a random batch of 32–256 examples, compute the average gradient over that batch, update. This is what everyone uses in practice. It is fast (parallel computation on a batch), has manageable noise (averaged over many examples), and produces good gradient estimates.

| Method | Gradient quality | Speed per update | Memory | Used in practice? |
|--------|-----------------|-----------------|--------|------------------|
| Full-batch | Exact | Slow (scales with N) | High (entire dataset) | Rarely — only small datasets |
| SGD (batch=1) | Very noisy | Fast | Minimal | Sometimes for online learning |
| Mini-batch SGD | Good (low variance) | Fast (GPU-parallelized) | Moderate | Yes — the standard |

```python
import torch
import torch.nn as nn

# Simulated dataset: predict threat level from 4 orbital features
torch.manual_seed(7)
N = 1000        # total training examples
X_all = torch.randn(N, 4)             # orbital features
true_w = torch.tensor([2.0, -1.0, 0.5, 1.5])
y_all  = X_all @ true_w + 0.1 * torch.randn(N)  # true labels with noise

model  = nn.Linear(4, 1, bias=False)
optim  = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# --- Full-batch gradient descent ---
print("Full-batch gradient descent (1 update per epoch):")
for epoch in range(5):
    pred = model(X_all).squeeze()
    loss = loss_fn(pred, y_all)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}")

# Reset model
model  = nn.Linear(4, 1, bias=False)
optim  = torch.optim.SGD(model.parameters(), lr=0.01)

# --- Mini-batch gradient descent ---
batch_size = 32
print(f"\nMini-batch gradient descent (batch_size={batch_size}):")
for epoch in range(5):
    # Shuffle data
    perm  = torch.randperm(N)
    X_s   = X_all[perm]
    y_s   = y_all[perm]
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, N, batch_size):
        X_batch = X_s[start : start + batch_size]
        y_batch = y_s[start : start + batch_size]

        pred  = model(X_batch).squeeze()
        loss  = loss_fn(pred, y_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / n_batches
    print(f"  Epoch {epoch+1}: avg loss = {avg_loss:.4f}  ({n_batches} batches)")

# Key observation: mini-batch does N/batch_size=31 updates per epoch
# vs. full-batch's 1 update. More updates per epoch → faster convergence.
```

The mini-batch approach updates the model parameters \\(\lfloor N / \text{batch\_size} \rfloor\\) times per pass through the data. Those frequent updates — even though each one uses a noisy gradient estimate — typically produce faster overall convergence than the single precise update of full-batch GD. The noise also helps: noisy gradient descent tends to escape local minima more readily than exact gradient descent.

---

## The learning rate: why not take the full gradient step?

Notice we multiplied the gradient by `learning_rate = 0.2` instead of just subtracting the gradient directly. Why?

The gradient tells you the slope at your current location. It is a local approximation: it is accurate close to where you are, but the function might curve away from the linear approximation if you move too far.

If you take too large a step, you might overshoot the minimum and end up on the other side, potentially further away than you started. A small learning rate keeps you in the regime where the linear approximation is trustworthy.

Choosing the learning rate is one of the most practically important decisions in training a neural network. Too large and training oscillates or diverges. Too small and training is painfully slow. We will discuss this more in module 2 when we actually train networks.

---

## Why this matters for the rest of the curriculum

Every algorithm from here on trains by gradient descent:

- **Policy gradient methods** (module 3): compute the gradient of expected return with respect to policy parameters, step parameters in the positive direction (we want to maximize, not minimize).
- **Q-learning with neural networks** (module 3): compute the gradient of a temporal-difference error with respect to value function parameters, step to reduce the error.
- **Deep CFR** (module 5): compute gradients of a regret prediction loss and step to make regret predictions more accurate.

In each case, you will write a forward pass (compute the loss from the current parameters), call `.backward()` (chain rule through the computational graph), and update the parameters (subtract learning_rate × gradient). The specific loss function and what you are minimizing will differ. The gradient descent structure will be the same.

Kneusel's *Math for Deep Learning* Ch. 7–8 goes deeper on both the calculus and the PyTorch autograd mechanics. The Jacobian perspective from this lesson connects to the extended Kalman filter (EKF) used in orbit determination — the EKF is gradient-based estimation applied to dynamical systems, and understanding the Jacobian is the key to understanding why the EKF works.

---

## Key Takeaways

* **The derivative is the slope at a point.** The formal limit definition is what numerical gradient checking computes using a finite ε. The central difference approximation \\((f(x+\varepsilon) - f(x-\varepsilon)) / 2\varepsilon\\) is more accurate than one-sided differences and is the standard for gradient checking.

* **Partial derivatives hold all other inputs fixed.** The gradient vector collects all partial derivatives of a scalar function. It points in the direction of steepest ascent. Gradient descent steps in the opposite direction.

* **The chain rule multiplies local rates of change.** For a composition of functions, the overall derivative is the product of all the intermediate derivatives. PyTorch's autograd automates this using the computational graph recorded during the forward pass.

* **The differentiation rules cheat sheet is your constant companion.** Power rule, sum rule, product rule, chain rule, and the derivatives of sigmoid/ReLU/tanh are enough to analyze any standard network architecture analytically.

* **Gradient checking is your first debugging tool for custom components.** If the relative error between numerical and analytical gradients exceeds \\(10^{-5}\\), there is a bug in your gradient code. Use double precision (float64) for gradient checks to reduce numerical noise.

* **The Jacobian generalizes the gradient to vector-valued functions.** It is an m × n matrix of partial derivatives. In backprop, the Jacobian of each layer appears in the gradient calculation. In orbit determination, the Jacobian of the measurement model is the H matrix in the extended Kalman filter.

* **Mini-batch SGD is the default training algorithm.** It balances gradient quality (batch average reduces noise) against speed (many updates per epoch) and memory (only one batch in GPU memory at a time). Full-batch GD is theoretically cleaner but rarely used at scale; single-sample SGD is used for online learning but noisy for offline training.

---

{{#quiz 07-derivatives-gradients-chain-rule.toml}}
