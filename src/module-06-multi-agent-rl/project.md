# Module 6 Project: PSRO for Satellite Constellation Coverage

## What you are building

You will implement the PSRO outer loop for a two-player satellite constellation coverage game. Each player controls a set of satellites that can observe one orbital slot per turn. Players compete for coverage of a shared region. The project builds the complete PSRO pipeline: simulating policy rollouts to fill a payoff matrix, solving the meta-game Nash with linear programming, training RL best-response oracles, and iterating until the population converges. By the end you will have a working PSRO loop and a policy population that plays a non-trivial coverage strategy.

## The game

**Players:** Two operators, each controlling 3 satellites.

**State:** Each satellite has a current orbital slot assignment (slots 0–7, arranged in a ring). The full state is 6 slot assignments (3 per player).

**Actions:** Each player simultaneously assigns each of their 3 satellites to a slot. Assignments are made without observing the opponent's assignments first (simultaneous-move game).

**Payoff:** Computed after both players commit:
- Each uniquely covered slot scores +1 for the covering player
- If both players cover the same slot, neither scores (contested, +0 each)
- Each player's score is the count of uniquely covered slots minus 0.5 × contested slots

This is a zero-sum game: total payoff sums to the number of non-contested covered slots.

## Step 1: the environment

```python
import numpy as np
from itertools import combinations

N_SLOTS = 8
N_SATS = 3

def compute_payoff(assign_p1: list[int], assign_p2: list[int]) -> tuple[float, float]:
    set1 = set(assign_p1)
    set2 = set(assign_p2)
    contested = set1 & set2
    unique1 = set1 - contested
    unique2 = set2 - contested
    p1 = len(unique1) - 0.5 * len(contested)
    p2 = len(unique2) - 0.5 * len(contested)
    return p1, p2

# All possible assignments: choose N_SATS distinct slots from N_SLOTS
ALL_ACTIONS = list(combinations(range(N_SLOTS), N_SATS))
N_ACTIONS = len(ALL_ACTIONS)  # C(8,3) = 56
ACTION_INDEX = {a: i for i, a in enumerate(ALL_ACTIONS)}
print(f"Action space size: {N_ACTIONS}")
```

## Step 2: build the full payoff matrix

```python
def build_payoff_matrix() -> np.ndarray:
    M = np.zeros((N_ACTIONS, N_ACTIONS))
    for i, a1 in enumerate(ALL_ACTIONS):
        for j, a2 in enumerate(ALL_ACTIONS):
            p1, _ = compute_payoff(list(a1), list(a2))
            M[i, j] = p1
    return M

M_full = build_payoff_matrix()
print(f"Payoff matrix shape: {M_full.shape}")
```

## Step 3: Nash solver for the meta-game

```python
from scipy.optimize import linprog

def solve_nash_zero_sum(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a zero-sum normal-form game via LP. Returns (sigma_1, sigma_2)."""
    n, m = M.shape

    # Player 1: maximize min expected payoff
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-M.T, np.ones((m, 1))])
    b_ub = np.zeros(m)
    A_eq = np.ones((1, n + 1)); A_eq[0, -1] = 0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n + [(None, None)]
    r1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    sigma_1 = np.maximum(r1.x[:n], 0); sigma_1 /= sigma_1.sum()

    # Player 2: minimize max expected payoff
    c2 = np.zeros(m + 1); c2[-1] = 1.0
    A_ub2 = np.hstack([M, -np.ones((n, 1))])
    b_ub2 = np.zeros(n)
    A_eq2 = np.ones((1, m + 1)); A_eq2[0, -1] = 0
    b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * m + [(None, None)]
    r2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2)
    sigma_2 = np.maximum(r2.x[:m], 0); sigma_2 /= sigma_2.sum()

    return sigma_1, sigma_2
```

## Step 4: best-response oracle

```python
def best_response_p1(sigma_2: np.ndarray, population_indices: list[int]) -> int:
    expected_payoffs = np.zeros(N_ACTIONS)
    for k, j in enumerate(population_indices):
        expected_payoffs += sigma_2[k] * M_full[:, j]
    return int(np.argmax(expected_payoffs))

def best_response_p2(sigma_1: np.ndarray, population_indices: list[int]) -> int:
    expected_payoffs = np.zeros(N_ACTIONS)
    for k, i in enumerate(population_indices):
        expected_payoffs += sigma_1[k] * M_full[i, :]
    return int(np.argmin(expected_payoffs))
```

## Step 5: the PSRO outer loop

```python
def run_psro(n_iterations: int = 20, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    pop1 = [int(rng.integers(N_ACTIONS))]
    pop2 = [int(rng.integers(N_ACTIONS))]
    history = []

    for iteration in range(n_iterations):
        M_restricted = M_full[np.ix_(pop1, pop2)]
        sigma_1, sigma_2 = solve_nash_zero_sum(M_restricted)
        meta_nash_value = float(sigma_1 @ M_restricted @ sigma_2)

        br1 = best_response_p1(sigma_2, pop2)
        br2 = best_response_p2(sigma_1, pop1)

        br1_value = float(M_full[br1, :][pop2] @ sigma_2)
        br2_value = float(M_full[:, br2][pop1] @ sigma_1)
        exploitability = (br1_value - meta_nash_value) + (meta_nash_value - (-br2_value))

        history.append({
            "iteration": iteration,
            "pop_size": len(pop1),
            "meta_nash_value": meta_nash_value,
            "exploitability": exploitability,
        })

        print(f"Iter {iteration:2d} | pop={len(pop1):2d} | "
              f"value={meta_nash_value:+.3f} | exploit={exploitability:.4f}")

        if br1 not in pop1:
            pop1.append(br1)
        if br2 not in pop2:
            pop2.append(br2)

        if exploitability < 1e-4:
            print(f"Converged at iteration {iteration}.")
            break

    return {"history": history, "pop1": pop1, "pop2": pop2}

results = run_psro()
```

## Step 6: analyze results

```python
import matplotlib.pyplot as plt

history = results["history"]
iters    = [h["iteration"] for h in history]
exploits = [h["exploitability"] for h in history]
values   = [h["meta_nash_value"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.semilogy(iters, exploits, "b-o", markersize=4)
ax1.set_xlabel("PSRO iteration"); ax1.set_ylabel("Exploitability (log scale)")
ax1.set_title("Convergence to Nash"); ax1.grid(True, alpha=0.3)

ax2.plot(iters, values, "r-o", markersize=4)
ax2.set_xlabel("PSRO iteration"); ax2.set_ylabel("Meta-Nash value (player 1)")
ax2.set_title("Nash equilibrium value over iterations")
ax2.axhline(0, color="k", linestyle="--", alpha=0.5, label="zero-sum balance")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("psro_convergence.png", dpi=150)
plt.show()

# Print final Nash strategies
final = history[-1]
print("\nFinal Nash mixture (player 1):")
for idx, w in zip(results["pop1"], results["history"][-1]["meta_nash_value"] * np.ones(len(results["pop1"]))):
    slots = ALL_ACTIONS[idx]
    print(f"  slots={slots}")
```

## What to observe

1. **Exploitability drops to near zero** within 5–10 PSRO iterations for this game. The restricted Nash converges to the full-game Nash because the best-response oracle quickly identifies the dominant pure strategies.

2. **The Nash value is near zero** — in this symmetric game, neither player can guarantee more than equal expected coverage at equilibrium.

3. **Population size at convergence** is typically 4–8 pure strategies. Inspect which slot assignments appear at high weight: they will tend to spread satellites evenly to minimize conflict with the opponent.

4. **Deviation from Nash is costly**: compute the expected payoff when player 1 plays a fixed assignment (e.g., slots 0, 1, 2) against the Nash mixture of player 2. Compare to the Nash value. The gap measures how exploitable a predictable player is.

5. **Extension**: replace the analytical best-response with the RL oracle from lesson 3 to see how the loop behaves when the oracle is approximate.
