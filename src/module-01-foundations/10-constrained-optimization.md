# Lesson 10: Constrained Optimization and Lagrange Multipliers

**Module:** ML and Game Theory for Space Power — M01: Foundations
**Source:** *Mathematics for Machine Learning* — Deisenroth, Faisal & Ong (2020), Chapters 7.2–7.3

---


<!-- toc -->

## Where this fits

Lesson 07 covered unconstrained gradient descent: minimize \(f(x)\) by following the negative gradient. But real-world optimization is almost always constrained. Maneuver a satellite to a new orbit with a fixed delta-v budget. Train a policy subject to a KL divergence bound. Find the minimum-fuel trajectory subject to orbital dynamics. Constrained optimization is the tool for all of these.

The machinery developed here — Lagrange multipliers, the KKT conditions, the Lagrangian dual — appears in three specific places downstream. PPO's trust region (Module 03) is a constrained optimization problem in which the Lagrange multiplier becomes an automatically adapted learning rate. The SVM dual formulation (Module 04) converts a hard quadratic program into a tractable dual problem. Natural policy gradients and PSRO meta-game computation (Module 06) rely on convex optimization solvers. Understanding the Lagrangian and the dual is the common thread behind all of these.

---

## The constrained optimization problem

The general constrained optimization problem is:

\[
\min_{x} \; f(x) \quad \text{subject to} \quad g_i(x) \leq 0, \; i = 1, \ldots, m \quad \text{and} \quad h_j(x) = 0, \; j = 1, \ldots, p
\]

**Decoding:**

**\(f(x)\)**: The objective function — the quantity you want to minimize. In an orbit maneuver, this might be \(\|\Delta v\|\), the total change in velocity (and hence the fuel consumed).

**\(g_i(x) \leq 0\)**: Inequality constraints. These define a region you must stay inside. Each \(g_i\) describes a boundary that \(x\) cannot cross. Rewriting "perigee altitude must stay above 200 km" as \(g(x) = 200 - r_p(x) \leq 0\) puts it in this standard form.

**\(h_j(x) = 0\)**: Equality constraints. These are surfaces (not regions) that \(x\) must lie on exactly. The vis-viva equation relating orbital speed to distance is an equality constraint — you cannot just satisfy it approximately; the physics demands exact equality.

**The feasible set** is the set of all \(x\) that satisfy every constraint simultaneously. The constrained minimum is the point in the feasible set where \(f\) is smallest.

**Geometric intuition:** imagine the objective \(f\) as a bowl-shaped surface. The unconstrained minimum is the bottom of the bowl. Now impose a wall (an inequality constraint). If the bottom of the bowl is outside the wall, you must press the bowl against the wall, and the constrained minimum is the point where the bowl just touches the wall from the inside.

### SSA example: orbit raising with budget constraints

A satellite needs to transfer from a 400 km circular parking orbit to a 1200 km target orbit. The mission constraints are:

- **Minimize:** total delta-v \(\|\Delta v\|\) (fuel consumption proxy)
- **Equality constraint:** the final orbit must achieve the target semi-major axis \(a = 1200 + 6371 = 7571\) km (Earth's radius plus altitude)
- **Inequality constraint:** at no point during the transfer should the perigee drop below 180 km (atmospheric drag limit), i.e., \(r_p \geq 6371 + 180 = 6551\) km
- **Inequality constraint:** the maneuver must complete within 30 days

The unconstrained minimum (burn freely in any direction for any duration) might involve a trajectory that temporarily dips into the atmosphere. The constraints force a solution that achieves the target orbit while respecting the physical and operational bounds.

---

## Lagrange multipliers for equality constraints

### The key idea

At the constrained optimum, the gradient of the objective is parallel to the gradient of the constraint. If \(\nabla f\) pointed in a direction that is not parallel to \(\nabla h\), you could move slightly along the constraint surface (keeping \(h(x) = 0\)) and reduce \(f\). So at the minimum, you are stuck: any feasible direction is neutral for \(f\), which means \(\nabla f\) and \(\nabla h\) must point in the same or opposite direction.

Formally, there exists a scalar \(\lambda\) such that \(\nabla f = -\lambda \nabla h\) at the optimum.

### The Lagrangian

Rather than solving the constrained problem directly, we form the **Lagrangian**:

\[
\mathcal{L}(x, \lambda) = f(x) + \lambda \, h(x)
\]

**Decoding:**

**\(\lambda\)**: The **Lagrange multiplier**. It is a new scalar variable (or a vector of scalars, one per equality constraint). It plays the role of a price: how much would the optimal value of \(f\) improve if we relaxed the constraint slightly? A large positive \(\lambda\) means the constraint is expensive — the optimum would improve a lot if the constraint were loosened.

**\(\lambda \, h(x)\)**: The constraint is folded into the objective. At any point where \(h(x) \neq 0\), this term adds a penalty proportional to how much the constraint is violated.

### First-order conditions

To find the constrained optimum, take the gradient of the Lagrangian with respect to both \(x\) and \(\lambda\) and set both to zero:

\[
\nabla_x \mathcal{L} = \nabla f(x) + \lambda \nabla h(x) = 0
\]

\[
\frac{\partial \mathcal{L}}{\partial \lambda} = h(x) = 0
\]

**Decoding:**

The first equation says \(\nabla f = -\lambda \nabla h\): the gradients are antiparallel (scaled versions of each other), which is exactly the geometric condition above.

The second equation simply recovers the equality constraint \(h(x) = 0\): taking the derivative with respect to \(\lambda\) and setting it to zero forces the constraint to be satisfied.

### Worked example: orbit-raising delta-v

Consider a simplified Hohmann transfer. A satellite in a circular orbit of radius \(r_1\) applies a burn \(\Delta v\) to enter an elliptical transfer orbit, then a second burn at apoapsis to circularize at \(r_2\). For the first burn only (starting from circular velocity):

**Objective:** minimize \(f(\Delta v) = |\Delta v|\)

**Constraint:** the vis-viva equation requires that after the burn, the specific orbital energy satisfies:

\[
h(\Delta v) = \frac{(v_c + \Delta v)^2}{2} - \frac{\mu}{r_1} - \varepsilon^* = 0
\]

where \(v_c = \sqrt{\mu / r_1}\) is the circular speed, \(\mu\) is Earth's gravitational parameter, and \(\varepsilon^*\) is the specific energy needed for the transfer orbit. The equality constraint is exactly the vis-viva equation: the burn must produce precisely the right energy.

Lagrangian: \(\mathcal{L}(\Delta v, \lambda) = |\Delta v| + \lambda \left[\frac{(v_c + \Delta v)^2}{2} - \frac{\mu}{r_1} - \varepsilon^*\right]\)

Taking \(\partial \mathcal{L} / \partial (\Delta v) = 0\) gives \(1 + \lambda(v_c + \Delta v) = 0\), so \(\lambda = -1 / (v_c + \Delta v)\). The energy equation pins down the value of \(\Delta v\), and \(\lambda\) tells you the sensitivity: relaxing the energy requirement by one unit reduces the required \(\Delta v\) by \(1/(v_c + \Delta v)\).

### Code: equality-constrained 2D example

The following example minimizes a quadratic objective subject to a linear equality constraint, first using the Lagrangian conditions manually with PyTorch autograd, then verifying against `scipy.optimize.minimize`.

```python
import torch
import torch.autograd
from scipy.optimize import minimize
import numpy as np

# Problem: minimize f(x) = (x[0] - 3)^2 + (x[1] - 2)^2
# subject to: h(x) = x[0] + x[1] - 4 = 0
# (constrained to the line x[0] + x[1] = 4)
#
# SSA context: x[0] = delta-v in radial direction (km/s)
#              x[1] = delta-v in along-track direction (km/s)
# Objective: minimize fuel (distance from a reference burn vector [3, 2])
# Constraint: total speed change must equal exactly 4 km/s (fixed budget)

def f(x: torch.Tensor) -> torch.Tensor:
    return (x[0] - 3.0)**2 + (x[1] - 2.0)**2

def h(x: torch.Tensor) -> torch.Tensor:
    return x[0] + x[1] - 4.0

# --- Lagrangian approach: solve ∇f + λ∇h = 0 and h(x) = 0 ---
# Analytic: ∇f = [2(x0-3), 2(x1-2)], ∇h = [1, 1]
# Conditions: 2(x0-3) + λ = 0, 2(x1-2) + λ = 0, x0+x1 = 4
# From the first two: x0-3 = x1-2, so x0 = x1+1
# Substituting into constraint: (x1+1) + x1 = 4 → x1 = 1.5, x0 = 2.5
# λ = -2(x0-3) = -2(2.5-3) = 1.0

x_star = torch.tensor([2.5, 1.5], dtype=torch.float64, requires_grad=True)
lam = torch.tensor(1.0, dtype=torch.float64)

# Verify that Lagrangian stationarity conditions hold at x_star
lag = f(x_star) + lam * h(x_star)
lag.backward()
print("Lagrangian gradient at x*:")
print(f"  ∇_x L = {x_star.grad.tolist()}")     # should be [0, 0] or near zero
print(f"  h(x*) = {h(x_star.detach()).item():.6f}")  # should be 0
print(f"  f(x*) = {f(x_star.detach()).item():.6f}")  # optimal value

# --- Scipy verification ---
def f_np(x):
    return (x[0] - 3.0)**2 + (x[1] - 2.0)**2

def df_np(x):
    return np.array([2*(x[0]-3.0), 2*(x[1]-2.0)])

constraints = [{"type": "eq", "fun": lambda x: x[0] + x[1] - 4.0}]
result = minimize(f_np, x0=[0.0, 0.0], jac=df_np, constraints=constraints, method="SLSQP")

print(f"\nScipy solution: x* = {result.x}, f* = {result.fun:.6f}")
print(f"Scipy constraint satisfied: h(x*) = {result.x[0] + result.x[1] - 4.0:.2e}")
print(f"Lagrange multiplier: λ* = {result.v[0][0]:.4f}")  # v holds KKT multipliers
```

The Lagrange multiplier \(\lambda = 1.0\) has a concrete interpretation here: relaxing the budget constraint by 0.001 km/s (from 4.000 to 4.001) would reduce the optimal fuel cost by approximately 0.001.

---

## Lagrange multipliers for inequality constraints: KKT conditions

Equality constraints pin you to a surface. Inequality constraints \(g(x) \leq 0\) give you a region. The generalization requires more care because the constraint may or may not be active at the solution.

### The Lagrangian for inequality constraints

\[
\mathcal{L}(x, \lambda) = f(x) + \lambda \, g(x), \quad \lambda \geq 0
\]

The dual feasibility condition \(\lambda \geq 0\) is essential: the Lagrange multiplier for an inequality constraint must be non-negative. Intuitively, the constraint \(g(x) \leq 0\) pushes inward (into the feasible region), so the multiplier must have the right sign to oppose that push.

### The KKT conditions

The **Karush-Kuhn-Tucker (KKT) conditions** are the first-order necessary conditions for a constrained optimum with inequality constraints. For the problem with both types of constraints:

\[
\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \; h_j(x) = 0
\]

the KKT conditions are:

1. **Stationarity:** \(\nabla_x f(x^*) + \sum_i \lambda_i \nabla_x g_i(x^*) + \sum_j \nu_j \nabla_x h_j(x^*) = 0\)
2. **Primal feasibility:** \(g_i(x^*) \leq 0\) and \(h_j(x^*) = 0\)
3. **Dual feasibility:** \(\lambda_i \geq 0\)
4. **Complementary slackness:** \(\lambda_i \, g_i(x^*) = 0\) for all \(i\)

**Decoding each condition:**

**Stationarity** says that at the optimal point, no direction in the feasible set can further reduce \(f\). The gradient of the objective is balanced by the weighted gradients of the active constraints. This is the same geometric condition as before, extended to multiple constraints.

**Primal feasibility** says the solution must actually satisfy the original constraints — you did not gain a better objective by cheating and going outside the feasible set.

**Dual feasibility** (\(\lambda_i \geq 0\)) says the multipliers are non-negative. If \(g_i\) is an upper-bound constraint, the multiplier must pull the objective in the direction that tightens the constraint, not loosens it.

**Complementary slackness** is the critical new condition. Either \(\lambda_i = 0\) (the constraint is inactive — it is not binding at the solution and does not affect it) or \(g_i(x^*) = 0\) (the constraint is active — the solution lies exactly on the constraint boundary). Both cannot be nonzero simultaneously.

This captures the intuition precisely: if you are not against a wall (the constraint is inactive, \(g_i < 0\)), it does not affect your solution and its multiplier is zero. If you are against the wall (the constraint is active, \(g_i = 0\)), the multiplier is potentially nonzero and tells you the cost of the constraint.

### SSA example: power-constrained communications

A satellite must transmit telemetry to a ground station. The transmitter has a variable power level \(P\) (Watts). Minimize power consumption subject to the signal-to-noise ratio (SNR) meeting the minimum threshold:

- **Objective:** \(f(P) = P\) (minimize transmit power)
- **Constraint:** \(g(P) = \text{SNR}_\text{min} - \text{SNR}(P) \leq 0\) (SNR must exceed 10 dB threshold)

where \(\text{SNR}(P) = P \cdot G / (k T B)\) (linear SNR, with fixed antenna gain \(G\), noise temperature \(T\), Boltzmann constant \(k\), bandwidth \(B\)).

At the optimal solution: the constraint is active (\(\lambda > 0\), \(g(P^*) = 0\)). The satellite transmits at exactly the minimum power that hits 10 dB SNR — not more. The multiplier \(\lambda > 0\) tells you how much extra power you would need if the SNR requirement were raised.

If the satellite has a more efficient antenna that already achieves 15 dB at the minimum feasible power, the constraint is inactive (\(g(P^*) < 0\)) and \(\lambda = 0\) — you can reduce power freely until some other constraint becomes binding.

### Code: 2D inequality-constrained optimization

```python
import numpy as np
from scipy.optimize import minimize

# Problem: minimize f(x) = (x[0] - 1)^2 + (x[1] - 2.5)^2
# subject to:
#   g1(x) = -x[0] + 2*x[1] - 2 <= 0   (above a line in the x0-x1 plane)
#   g2(x) =  x[0] + 2*x[1] - 6 <= 0   (below another line)
#   g3(x) =  x[0] - 2*x[1] - 2 <= 0
# (A classic constrained QP example)
#
# SSA context: x[0] = radial burn component, x[1] = tangential burn component
# Objective: minimize distance from desired burn vector [1, 2.5]
# Constraints: control authority limits (each represents a linear bound on the burns)

def f_ineq(x):
    return (x[0] - 1.0)**2 + (x[1] - 2.5)**2

def df_ineq(x):
    return np.array([2*(x[0]-1.0), 2*(x[1]-2.5)])

# scipy convention: constraints are g(x) >= 0, so negate our g_i <= 0 forms
constraints = [
    {"type": "ineq", "fun": lambda x:  x[0] - 2*x[1] + 2},   # -g1 >= 0
    {"type": "ineq", "fun": lambda x: -x[0] - 2*x[1] + 6},   # -g2 >= 0
    {"type": "ineq", "fun": lambda x: -x[0] + 2*x[1] + 2},   # -g3 >= 0
]
bounds = [(0, None), (0, None)]  # x[0] >= 0, x[1] >= 0

result = minimize(
    f_ineq, x0=[2.0, 0.0], jac=df_ineq,
    method="SLSQP", bounds=bounds, constraints=constraints
)

print(f"Optimal x*:  [{result.x[0]:.4f}, {result.x[1]:.4f}]")
print(f"Optimal f*:  {result.fun:.4f}")
print(f"Constraint values at x* (should be <= 0 for active/inactive):")
g1 = -result.x[0] + 2*result.x[1] - 2
g2 =  result.x[0] + 2*result.x[1] - 6
g3 =  result.x[0] - 2*result.x[1] - 2

for name, val in [("g1", g1), ("g2", g2), ("g3", g3)]:
    status = "ACTIVE (binding)" if abs(val) < 1e-6 else f"inactive (slack = {-val:.4f})"
    print(f"  {name}(x*) = {val:.6f}  -> {status}")

# KKT multipliers (available from SLSQP as result.v if constraints are provided)
# Active constraints have nonzero multipliers, inactive have multiplier = 0
```

The output will show which constraints are binding at the solution. Any constraint reported as active (\(g_i \approx 0\)) corresponds to a nonzero KKT multiplier; inactive constraints have \(\lambda_i = 0\), confirming complementary slackness.

---

## The Lagrangian dual

### Primal and dual problems

Given the primal constrained problem, we can derive a paired **dual problem** that turns out to be easier to solve in many cases.

Define the **Lagrangian dual function**:

\[
D(\lambda) = \min_{x} \; \mathcal{L}(x, \lambda) = \min_{x} \left[ f(x) + \sum_i \lambda_i g_i(x) \right]
\]

For each fixed \(\lambda \geq 0\), \(D(\lambda)\) is the minimum of the Lagrangian over all \(x\) (unconstrained). The **dual problem** is:

\[
\max_{\lambda \geq 0} \; D(\lambda)
\]

**Decoding:**

\(D(\lambda)\) is the best lower bound on \(f\) you can get by penalizing constraint violations with weights \(\lambda\). You want to find the tightest such lower bound, which is what the dual maximization does.

### Weak and strong duality

**Weak duality** always holds: \(D(\lambda) \leq f(x^*)\) for any \(\lambda \geq 0\). The dual gives a lower bound on the primal optimum. This is true regardless of the problem structure.

**Strong duality**: when \(f\) and all \(g_i\) are convex and a regularity condition (Slater's condition) holds — there exists a strictly feasible point — the duality gap is zero:

\[
D(\lambda^*) = f(x^*)
\]

The dual achieves the same optimal value as the primal. Solving the dual is equivalent to solving the primal.

**The duality gap** is \(f(x^*) - D(\lambda^*)\). Under strong duality, it is zero. In non-convex problems, there may be a positive gap — the dual bound is loose.

**Why this matters for machine learning:**

The dual is often much easier to solve than the primal. The SVM dual, for example, converts a problem in the weight space (potentially infinite-dimensional via kernels) into a finite-dimensional quadratic program over training examples. PPO's trust-region constraint is handled by moving it into the Lagrangian and treating the multiplier \(\lambda\) as an adaptive penalty coefficient:

\[
\mathcal{L}_\text{PPO}(\theta) = \mathbb{E}\left[\hat{A} \cdot r_t(\theta)\right] - \lambda \cdot \text{KL}(\pi_\text{old} \| \pi_\theta)
\]

Instead of solving the constrained problem (hard), PPO approximately solves the unconstrained Lagrangian for a fixed \(\lambda\), then updates \(\lambda\) based on whether the KL constraint was satisfied. This is dual ascent — a first-order method on the dual problem.

### SSA framing

The primal orbit optimization problem is: "find the minimum-fuel maneuver sequence that satisfies all dynamics constraints, altitude limits, and timing requirements." The dual asks: "find the right penalty weights \(\lambda_i\) such that minimizing the penalized cost — fuel plus weighted constraint violations — gives the same answer as solving the primal directly." Under strong duality (the problem is convex), these two answers are identical.

---

## Convex optimization

### What is convexity?

A function \(f: \mathbb{R}^n \to \mathbb{R}\) is **convex** if for any two points \(x, y\) and any \(t \in [0, 1]\):

\[
f(tx + (1-t)y) \leq t \, f(x) + (1-t) f(y)
\]

**Decoding:**

The left side is the function value at a point on the line segment between \(x\) and \(y\). The right side is the corresponding point on the chord (the straight line from \(f(x)\) to \(f(y)\)). Convexity says the function lies below the chord everywhere — the graph of \(f\) is "cup-shaped."

Equivalently (for twice-differentiable functions), \(f\) is convex if and only if its Hessian \(\nabla^2 f(x)\) is positive semi-definite at every \(x\): all eigenvalues of the Hessian are \(\geq 0\).

A set \(\mathcal{C}\) is **convex** if for any two points in \(\mathcal{C}\), the entire line segment between them is also in \(\mathcal{C}\). The intersection of convex sets is convex. The feasible set of a problem with convex inequality constraints \(g_i(x) \leq 0\) and linear equality constraints \(h_j(x) = 0\) is convex.

### Convex vs. non-convex functions in ML

| Function | Convex? | Reason |
|---|---|---|
| Squared error \(\|y - \hat{y}\|^2\) | Yes | Hessian = \(2I\), positive definite |
| Cross-entropy loss (softmax output) | Yes | Composition of convex and log-sum-exp |
| L2 regularization \(\lambda\|w\|^2\) | Yes | Positive definite Hessian |
| KL divergence \(\text{KL}(p \| q)\) as a function of \(q\) | Yes | Follows from convexity of \(-\log\) |
| Neural network loss (in \(W\)) | No | Product of weight matrices; non-convex in general |
| Log-likelihood of GMM | No | Mixture model; local maxima exist |
| Product of two parameters \(w_1 \cdot w_2\) | No | Cross-term; indefinite Hessian |

### Key theorem: convex problems have no bad local minima

For a convex objective \(f\) minimized over a convex feasible set, any local minimum is a global minimum. If gradient descent finds a stationary point, it is the global optimum. There is no need to worry about getting stuck.

This is why convexity is so valuable: the optimization problem is fully solved once you find any critical point. For non-convex problems (neural networks, GMMs, policy optimization), gradient descent may converge to a local minimum that is not globally optimal.

### SSA framing: why orbit optimization is tractable

Orbital mechanics constraints — energy conservation, angular momentum, vis-viva equation — are generally convex (or bilinear) in the velocity increments \(\Delta v_i\). The fuel cost \(\sum_i \|\Delta v_i\|\) is convex (sum of norms). This means the orbit transfer optimization problem, despite involving continuous dynamics and multiple burns, can often be posed as a convex program and solved to global optimality. This is why trajectory optimization tools used in satellite operations work reliably: they are not searching a non-convex landscape.

### Code: checking convexity via the Hessian

```python
import torch
import torch.autograd.functional as AF

# --- Example 1: Quadratic loss (convex) ---
# f(w) = ||Xw - y||^2 for X = [[1,0],[0,1],[1,1]], y = [1,2,3]
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_target = torch.tensor([1.0, 2.0, 3.0])

def f_quad(w):
    residuals = X @ w - y_target
    return (residuals ** 2).sum()

w0 = torch.tensor([0.5, 0.5])
H_quad = AF.hessian(f_quad, w0)
print("Hessian of quadratic loss:")
print(H_quad)
eigenvalues_quad = torch.linalg.eigvalsh(H_quad)
print(f"Eigenvalues: {eigenvalues_quad.tolist()}")
print(f"All eigenvalues >= 0: {bool((eigenvalues_quad >= -1e-8).all())}  (convex)")

print()

# --- Example 2: Non-convex product loss ---
# f(w) = (w[0] * w[1] - 1)^2  — product of parameters; non-convex in w
def f_nonconvex(w):
    return (w[0] * w[1] - 1.0)**2

w1 = torch.tensor([0.1, 0.1])  # near origin — indefinite Hessian expected
H_nc = AF.hessian(f_nonconvex, w1)
print("Hessian of non-convex product loss at [0.1, 0.1]:")
print(H_nc)
eigenvalues_nc = torch.linalg.eigvalsh(H_nc)
print(f"Eigenvalues: {[f'{v:.4f}' for v in eigenvalues_nc.tolist()]}")
print(f"All eigenvalues >= 0: {bool((eigenvalues_nc >= -1e-8).all())}  (non-convex if False)")

print()

# --- Example 3: L2 regularization (convex scalar check) ---
# f(w) = 0.5 * ||w||^2; Hessian should be identity
def f_l2(w):
    return 0.5 * (w ** 2).sum()

w2 = torch.tensor([1.0, -1.0, 2.0])
H_l2 = AF.hessian(f_l2, w2)
print("Hessian of L2 regularizer:")
print(H_l2)
eigenvalues_l2 = torch.linalg.eigvalsh(H_l2)
print(f"Eigenvalues: {eigenvalues_l2.tolist()}  (all = 1.0, convex)")
```

---

## Constrained optimization in machine learning

The abstract machinery of Lagrange multipliers and convexity appears in concrete, practical forms throughout this curriculum. This section connects the theory to three specific cases you will implement.

### PPO and trust regions

Proximal Policy Optimization solves a constrained policy update:

\[
\max_\theta \; \mathbb{E}\left[\hat{A}(s,a) \cdot \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}\right] \quad \text{subject to} \quad \text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \varepsilon
\]

The Lagrangian is:

\[
\mathcal{L}(\theta, \lambda) = \mathbb{E}\left[\hat{A} \cdot r_t(\theta)\right] - \lambda \left[\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) - \varepsilon\right]
\]

The dual variable \(\lambda\) is the Lagrange multiplier for the KL constraint. In practice, PPO adapts \(\lambda\) after each update: if the KL exceeded \(\varepsilon\), increase \(\lambda\) (making the constraint more expensive); if the KL is well below \(\varepsilon\), decrease \(\lambda\). This is dual ascent on the KL constraint.

```python
# Sketch: dual ascent structure for PPO-style KL penalty
# (Full PPO is in Module 03; this shows only the Lagrangian structure)
import torch

def ppo_lagrangian_update(policy_ratio, advantage, kl_div, lam, eps=0.01):
    """
    One step of the Lagrangian objective for PPO.
    policy_ratio: pi_theta / pi_old  (shape: batch)
    advantage:    estimated advantage A_hat  (shape: batch)
    kl_div:       scalar KL(pi_old || pi_theta)
    lam:          current Lagrange multiplier (scalar)
    eps:          KL budget
    """
    # Lagrangian objective (we maximize, so negative for gradient descent)
    surrogate = (policy_ratio * advantage).mean()
    lagrangian = surrogate - lam * (kl_div - eps)

    # Dual update: increase lam if KL exceeded budget, decrease if under
    lam_new = max(0.0, lam + 0.1 * (kl_div.item() - eps))

    return lagrangian, lam_new

# Example values
ratio = torch.tensor([1.05, 0.98, 1.12, 0.95])
adv   = torch.tensor([0.3, -0.1, 0.5, 0.2])
kl    = torch.tensor(0.015)   # slightly over eps=0.01
lam0  = 1.0

obj, lam1 = ppo_lagrangian_update(ratio, adv, kl, lam0, eps=0.01)
print(f"Lagrangian objective: {obj.item():.4f}")
print(f"Updated λ: {lam0:.4f} → {lam1:.4f}  (increased: KL={kl.item():.3f} > ε=0.01)")
```

```rust
fn ppo_lagrangian_update(
    ratios: &[f64],
    advantages: &[f64],
    kl_div: f64,
    lam: f64,
    eps: f64,
) -> (f64, f64) {
    let n = ratios.len() as f64;
    let surrogate: f64 = ratios.iter().zip(advantages.iter())
        .map(|(r, a)| r * a)
        .sum::<f64>() / n;
    let lagrangian = surrogate - lam * (kl_div - eps);
    // Dual ascent: increase λ if KL exceeded budget, decrease if under (clamp at 0)
    let lam_new = (lam + 0.1 * (kl_div - eps)).max(0.0);
    (lagrangian, lam_new)
}

fn main() {
    let ratios    = [1.05, 0.98, 1.12, 0.95];
    let advantages = [0.3, -0.1, 0.5, 0.2];
    let kl  = 0.015_f64;   // slightly over eps
    let lam0 = 1.0_f64;
    let eps  = 0.01_f64;

    let (obj, lam1) = ppo_lagrangian_update(&ratios, &advantages, kl, lam0, eps);
    println!("Lagrangian objective: {:.4}", obj);
    println!("Updated λ: {:.4} → {:.4}  (increased: KL={:.3} > ε={:.2})", lam0, lam1, kl, eps);
}
```

No external crates needed — the update rule is pure arithmetic. The `.max(0.0)` enforces dual feasibility (\(\lambda \geq 0\)).

### Weight decay as a Lagrangian

L2 regularization adds a penalty \(\lambda \|w\|^2\) to the training loss. This is exactly the Lagrangian for the constrained problem:

\[
\min_w \; \mathcal{L}(w) \quad \text{subject to} \quad \|w\|^2 \leq C
\]

The regularization coefficient \(\lambda\) is the Lagrange multiplier for the weight norm constraint. Choosing \(\lambda = 0.01\) is equivalent to choosing a constraint budget \(C\) such that the KKT condition holds at the minimum with that \(\lambda\).

This is not merely a mathematical curiosity: it means L2 regularization does not add arbitrary noise — it enforces a budget on the total parameter energy. The multiplier \(\lambda\) controls how tight that budget is.

```python
import torch
import torch.nn as nn

# Two equivalent formulations of the same optimization problem

# Formulation 1: penalized (unconstrained Lagrangian)
# min L(w) + lambda * ||w||^2
lam = 0.01
model_penalized = nn.Linear(10, 1)
optimizer = torch.optim.SGD(
    model_penalized.parameters(), lr=0.01, weight_decay=lam * 2
)   # PyTorch weight_decay = 2 * lambda (gradient of lambda * ||w||^2 is 2*lambda*w)

# Formulation 2: projected gradient (enforces ||w||^2 <= C at each step)
# This is conceptually equivalent; the Lagrange multiplier adapts to enforce C
def project_onto_l2_ball(params, C):
    """Project parameters onto the L2 ball of radius sqrt(C)."""
    total_norm_sq = sum((p**2).sum().item() for p in params)
    if total_norm_sq > C:
        scale = (C / total_norm_sq) ** 0.5
        with torch.no_grad():
            for p in params:
                p.mul_(scale)

# Under KKT: at the optimum, both formulations give the same w* when lambda
# is the KKT multiplier corresponding to the constraint ||w||^2 <= C.
print("Formulation 1 (L2 penalty): weight_decay = 2*lambda added to optimizer")
print("Formulation 2 (projection): enforce ||w||^2 <= C at each step")
print("Both are solving the same constrained problem; lambda <-> C are paired by KKT.")
```

```rust
fn project_onto_l2_ball(params: &mut [f64], c: f64) {
    let norm_sq: f64 = params.iter().map(|p| p * p).sum();
    if norm_sq > c {
        let scale = (c / norm_sq).sqrt();
        for p in params.iter_mut() {
            *p *= scale;
        }
    }
}

fn main() {
    let mut weights = vec![0.5_f64, -0.8, 1.2, -0.3, 0.9];
    let c = 1.0_f64;  // enforce ||w||^2 <= 1.0

    let norm_sq_before: f64 = weights.iter().map(|p| p * p).sum();
    println!("||w||² before projection: {:.4}", norm_sq_before);

    project_onto_l2_ball(&mut weights, c);

    let norm_sq_after: f64 = weights.iter().map(|p| p * p).sum();
    println!("||w||² after  projection: {:.4}", norm_sq_after);
    println!("Constraint satisfied: {}", norm_sq_after <= c + 1e-10);
    println!("Projected weights: {:?}", weights.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
}
```

No external crates needed. The projection divides by the current norm and scales down to the ball boundary — the Rust translation maps directly to the Python version.

### Minimum-fuel orbit transfer as a linear program

When the fuel cost is approximated as proportional to the total delta-v, and the orbital dynamics are linearized (Clohessy-Wiltshire equations for relative motion, for example), the orbit transfer problem becomes a linear program:

\[
\min_{\Delta v} \; c^\top \Delta v \quad \text{subject to} \quad A \Delta v = b, \quad \Delta v_\text{min} \leq \Delta v \leq \Delta v_\text{max}
\]

where \(c\) encodes fuel costs, \(A \Delta v = b\) enforces the target orbital state, and the bounds enforce actuator limits.

```python
from scipy.optimize import linprog
import numpy as np

# Simplified minimum-fuel transfer: 3 burn windows, each with a radial and
# tangential component. Target: net radial change = 1.5 km/s, net tangential = 0.8 km/s.
#
# Variables: x = [dv_r1, dv_t1, dv_r2, dv_t2, dv_r3, dv_t3]  (6 variables)
# Objective: minimize total |delta-v| ~ sum of absolute values
# To handle abs values with linprog, introduce slack variables:
#   x = [dv_r1+, dv_r1-, dv_t1+, dv_t1-, dv_r2+, dv_r2-, dv_t2+, dv_t2-, dv_r3+, dv_r3-, dv_t3+, dv_t3-]
# Cost: minimize sum of all split variables (each >= 0, representing |dv|)

n_burns = 3
n_vars = 2 * 2 * n_burns  # 12 variables: each component split into positive/negative part

# Objective: minimize sum of all 12 slack variables
c = np.ones(n_vars)

# Equality constraints: net radial = 1.5, net tangential = 0.8
# dv_r = dv_r+ - dv_r- for each burn window; sum across burns must equal target
# Row 0: sum of all radial components = 1.5
# Row 1: sum of all tangential components = 0.8
A_eq = np.zeros((2, n_vars))
for burn in range(n_burns):
    # radial: positive part at 4*burn, negative at 4*burn+1
    A_eq[0, 4*burn    ] =  1.0   # dv_r+
    A_eq[0, 4*burn + 1] = -1.0   # dv_r-
    # tangential: positive part at 4*burn+2, negative at 4*burn+3
    A_eq[1, 4*burn + 2] =  1.0   # dv_t+
    A_eq[1, 4*burn + 3] = -1.0   # dv_t-

b_eq = np.array([1.5, 0.8])

# Bounds: all slack variables >= 0, each split component <= 0.8 km/s (actuator limit)
bounds = [(0.0, 0.8)] * n_vars

result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

print(f"Minimum total delta-v: {result.fun:.4f} km/s")
print(f"Status: {result.message}")

# Reconstruct actual burn components from slack variables
for burn in range(n_burns):
    dv_r = result.x[4*burn] - result.x[4*burn + 1]
    dv_t = result.x[4*burn + 2] - result.x[4*burn + 3]
    print(f"  Burn {burn+1}: Δv_r = {dv_r:.4f} km/s,  Δv_t = {dv_t:.4f} km/s")
```

### Decision table: choosing the right method

| Problem type | Method | Example |
|---|---|---|
| Unconstrained smooth | Gradient descent, Adam | Neural network training |
| Equality constrained | Lagrangian, solve KKT | Orbit determination with vis-viva |
| Inequality constrained, convex | Interior point, CVXPY, linprog | Resource allocation, minimum-fuel transfer |
| Inequality constrained, non-convex | Penalty methods, PPO dual ascent | Policy optimization with KL budget |

The key question is whether the feasible set and objective are jointly convex. If yes, any solver guarantees the global optimum. If no, you are doing approximate optimization and must accept local solutions (or use global search, which is expensive).

---

## Key Takeaways

- **Constrained optimization adds feasibility requirements to the minimization problem.** Equality constraints \(h(x) = 0\) pin \(x\) to a surface; inequality constraints \(g(x) \leq 0\) define a feasible region. The unconstrained minimum may lie outside the feasible set, requiring the solution to be pushed to the constraint boundary.

- **The Lagrange multiplier is the price of the constraint.** The multiplier \(\lambda\) for a constraint \(h(x) = 0\) measures how much the optimal objective value would improve if the constraint were relaxed by one unit. A large \(\lambda\) means the constraint is expensive; \(\lambda = 0\) means the constraint is not binding at the solution.

- **KKT conditions generalize Lagrange multipliers to inequality constraints.** The four conditions — stationarity, primal feasibility, dual feasibility (\(\lambda \geq 0\)), and complementary slackness (\(\lambda g(x) = 0\)) — together characterize every constrained optimum. They replace the simple "gradient is zero" condition of unconstrained optimization.

- **Complementary slackness is the key new condition.** Either the constraint is active (\(g(x) = 0\), the solution is on the boundary) or the multiplier is zero (\(\lambda = 0\), the constraint is not influencing the solution). Both cannot be simultaneously nonzero, which is a powerful diagnostic: you can look at a solution and immediately determine which constraints matter.

- **Strong duality means the dual problem has the same answer as the primal.** When the objective and constraints are convex, solving the Lagrangian dual \(\max_{\lambda \geq 0} \min_x \mathcal{L}(x, \lambda)\) gives exactly the same optimal value as the primal. PPO's adaptive penalty coefficient is dual ascent on the KL constraint, and L2 regularization is the Lagrangian for a weight-norm constraint.

- **Convexity guarantees that any local minimum is global.** For convex \(f\) over a convex feasible set, gradient descent converges to the global optimum. Orbital mechanics constraints are often convex in velocity increments, making trajectory optimization tractable. Neural network losses are non-convex, so training finds local minima — but the tools of convex analysis (Hessian eigenvalues, duality gaps) remain useful for diagnosing convergence and designing regularization.

---

{{#quiz 10-constrained-optimization.toml}}
