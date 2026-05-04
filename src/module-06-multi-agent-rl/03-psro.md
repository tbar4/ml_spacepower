# Lesson 3: Policy-Space Response Oracles (PSRO)

## Where this fits

Fictitious play tracks a frequency count over a small finite set of actions. When the game is "choose channel A or channel B," tracking counts is easy. When the game is "command a constellation of 12 satellites across 6 orbital planes for a 72-hour coverage window," the action space is not a handful of discrete choices — it is a continuous high-dimensional space of satellite tasking sequences. Fictitious play cannot represent this.

PSRO (Policy-Space Response Oracles) generalizes fictitious play by replacing individual actions with neural-network policies and replacing the argmax best-response with a full RL training run. The empirical frequency over past actions becomes a mixture distribution over a growing population of policies. The algorithm uses this structure to converge to Nash equilibrium in games too complex for tabular methods.

This lesson builds on the double-oracle concept, which is the theoretical foundation for PSRO, and on fictitious play's empirical-frequency idea. It also sets up the next lesson on Alpha-rank, which provides an alternative to Nash for evaluating the policy populations PSRO builds.

## Double Oracle: the theoretical core

The **double oracle** algorithm is a general framework for solving large games by iteratively expanding the strategy set. Instead of solving the game in its full strategy space (which may be infinite), it maintains a small set of strategies for each player and grows it only when necessary.

The algorithm:

1. Start with a small initial strategy set for each player (e.g., one random policy each).
2. Solve the **restricted game**: the normal-form game played over only the current strategy sets.
3. Find a **best response** to the restricted-game Nash equilibrium for each player.
4. Add the best responses to the respective strategy sets.
5. Repeat until no player's best response improves their payoff by more than some tolerance ε.

The key property: the restricted game is much smaller than the full game (a 5×5 matrix instead of a continuous strategy space), so it can be solved efficiently. The best response computation is the expensive step — it requires exploring the full game. But because we only compute best responses to the current Nash, not exhaustively, the total computation is manageable.

PSRO makes this concrete for neural-network policies. The "restricted game" is a finite payoff matrix where entries are computed by running the policies against each other. The "best response oracle" is a full RL training run.

## The PSRO algorithm structure

PSRO maintains two data structures:

1. **Policy population**: a set of policies \\(\Pi_i = \{\pi_i^1, \pi_i^2, \ldots, \pi_i^k\}\\) for each player \\(i\\). Each policy is a neural network trained by RL.

2. **Meta-game payoff matrix**: a matrix \\(M\\) where \\(M_{ab}\\) contains the payoffs when player 1 uses policy \\(\pi_1^a\\) and player 2 uses policy \\(\pi_2^b\\). Each entry is estimated by running the two policies against each other and averaging the outcomes.

The PSRO loop:

```
Initialize with one policy per player (e.g., random or heuristic)
Construct initial meta-game payoff matrix M (1x1)
Solve meta-game for Nash equilibrium (σ*, one mixing weight per policy)

Repeat until convergence:
    For each player i:
        Train a best-response oracle: an RL agent that plays against
        the current meta-Nash mixture σ*_{-i} of the opponents
        Let π_new be the trained policy
    
    Add π_new to each player's population
    Extend the meta-game matrix M with new rows/columns
    Fill new entries by running new policies against all existing policies
    Solve the updated meta-game for a new Nash σ*

Output: the meta-Nash mixture σ* and the policy population Π
```

The PSRO outer loop is relatively simple; the complexity lives in the two subroutines: the oracle (RL training) and the meta-game solver (Nash computation on the matrix).

## The meta-game: a small tractable normal-form game

After k iterations, player 1 has k policies and player 2 has k policies. The meta-game is a k×k payoff matrix. For a typical PSRO run with 20-50 iterations, this is a matrix with at most 2500 entries — tiny by any measure.

Solving for Nash equilibrium in a 2-player zero-sum k×k matrix game is a linear program:

\\[ \min_{\sigma \in \Delta^k} \max_{j \in [k]} \sum_{i=1}^{k} \sigma_i \cdot M_{ij} \\]

**Decoding:**
- \\(\sigma \in \Delta^k\\): the mixing weights over player 1's k policies; they must be non-negative and sum to 1 (a simplex)
- \\(\Delta^k\\): the probability simplex over k strategies
- \\(\max_{j \in [k]}\\): the best response of player 2, who picks the column that maximizes their payoff
- \\(\sum_{i=1}^{k} \sigma_i \cdot M_{ij}\\): the expected payoff to player 2 when player 1 mixes with weights \\(\sigma\\)
- The outer minimization: player 1 wants the mixture that minimizes the damage from player 2's best response

For general-sum games, the meta-game Nash can be solved with support enumeration or the Lemke-Howson algorithm for small k. In practice, scipy's `linprog` handles zero-sum cases, and iterative solvers handle general-sum cases.

## Best response oracle: RL training against a mixture

The best response oracle for player i trains a new policy to maximize expected return against the current meta-Nash mixture of opponents.

Concretely: at each episode of RL training, sample an opponent policy \\(\pi_{-i}^j\\) from the meta-Nash distribution \\(\sigma^*_{-i}\\), then run a full episode of the game with the trainee policy against \\(\pi_{-i}^j\\). The RL gradient update optimizes the trainee against this sampled opponent.

This is equivalent to training against a **fixed mixed strategy** over the opponent population — exactly what fictitious play's best response computes, but now the "actions" are full neural-network policies and the "best response" is a gradient-descent training run.

```python
import numpy as np
from scipy.optimize import linprog

# ── Meta-game solver ───────────────────────────────────────────────────────────

def solve_zero_sum_nash(payoff_matrix):
    """
    Solve for the Nash equilibrium of a two-player zero-sum normal-form game.

    payoff_matrix: shape (n1, n2) — Player 1's payoffs (Player 2's are negatives)

    Returns (sigma1, sigma2): Nash equilibrium mixing weights for each player.
    """
    n1, n2 = payoff_matrix.shape

    # Player 1's LP: maximize the game value v subject to
    #   sum_i sigma1[i] * M[i, j] >= v  for all j
    #   sum_i sigma1[i] = 1, sigma1[i] >= 0
    #
    # Standard form: minimize -v
    # Variables: [sigma1[0], ..., sigma1[n1-1], v]

    # Inequality constraints: M.T @ sigma1 - v >= 0  <=>  -M.T @ sigma1 + v <= 0
    # A_ub x <= b_ub
    A_ub = np.hstack([-payoff_matrix.T, np.ones((n2, 1))])  # shape (n2, n1+1)
    b_ub = np.zeros(n2)

    # Equality constraint: sum(sigma1) = 1
    A_eq = np.hstack([np.ones((1, n1)), np.zeros((1, 1))])
    b_eq = np.array([1.0])

    # Objective: minimize -v (maximize v)
    c = np.zeros(n1 + 1)
    c[-1] = -1.0

    # Bounds: sigma1[i] in [0, 1], v unconstrained
    bounds = [(0, 1)] * n1 + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if not result.success:
        # Fall back to uniform if LP fails (e.g., degenerate payoff matrix)
        return np.ones(n1) / n1, np.ones(n2) / n2

    sigma1 = np.maximum(result.x[:n1], 0)
    sigma1 /= sigma1.sum()

    # Player 2's LP is symmetric: use the same approach on -M.T
    result2 = linprog(c[:n2+1],
                      A_ub=np.hstack([payoff_matrix, np.ones((n1, 1))]) * -1,
                      b_ub=np.zeros(n1),
                      A_eq=np.hstack([np.ones((1, n2)), np.zeros((1, 1))]),
                      b_eq=np.array([1.0]),
                      bounds=[(0, 1)] * n2 + [(None, None)],
                      method='highs')

    if not result2.success:
        sigma2 = np.ones(n2) / n2
    else:
        sigma2 = np.maximum(result2.x[:n2], 0)
        sigma2 /= sigma2.sum()

    return sigma1, sigma2


def exploitability_meta(payoff_matrix, sigma1, sigma2):
    """
    Compute exploitability of (sigma1, sigma2) in the meta-game.
    """
    ev1_per_action = payoff_matrix @ sigma2          # shape (n1,)
    ev2_per_action = (-payoff_matrix).T @ sigma1     # shape (n2,) for zero-sum

    exploit_1 = max(0, ev1_per_action.max() - sigma1 @ ev1_per_action)
    exploit_2 = max(0, ev2_per_action.max() - sigma2 @ ev2_per_action)
    return exploit_1 + exploit_2
```

## SSA application: constellation coverage game

Two satellite operators, Red and Blue, each manage constellations that observe overlapping orbital regimes. Each operator has a library of sensor-tasking policies:

- **Aggressive**: prioritize high-revisit on contested objects; accept gaps elsewhere
- **Distributed**: spread observations evenly across all objects
- **Reactive**: concentrate on objects that have been dark (unobserved) longest
- **Predictive**: observe objects before predicted maneuver windows
- **Random**: uniform random tasking (baseline)

PSRO grows this library by training new best-response policies. The meta-game payoff matrix tracks how each policy performs against each opponent policy.

The stub below shows the PSRO outer loop with placeholder oracle and evaluation functions. A real implementation would replace the stubs with an actual orbital simulation and RL training loop.

```python
import numpy as np
from typing import List, Callable, Tuple

# ── PSRO outer loop ────────────────────────────────────────────────────────────

class Policy:
    """Placeholder for a neural-network policy. In a real implementation,
    this would be a PyTorch module with a forward() method."""
    def __init__(self, name: str, weights=None):
        self.name = name
        self.weights = weights  # would be torch.nn.Module parameters

    def __repr__(self):
        return f"Policy({self.name})"


def evaluate_policies(policy1: Policy, policy2: Policy,
                      n_episodes: int = 100) -> Tuple[float, float]:
    """
    Stub: run policy1 vs policy2 for n_episodes and return mean payoffs.
    In a real implementation, this runs the orbital simulation.
    Returns (payoff_for_player1, payoff_for_player2).
    """
    # Placeholder: random payoffs for illustration
    np.random.seed(hash(policy1.name + policy2.name) % (2**31))
    r1 = np.random.randn() * 0.3
    r2 = -r1 + np.random.randn() * 0.1  # approximately zero-sum with noise
    return float(r1), float(r2)


def train_best_response_oracle(player_idx: int,
                                opponent_policies: List[Policy],
                                opponent_mixture: np.ndarray,
                                iteration: int) -> Policy:
    """
    Stub: train a new policy that best-responds to the opponent mixture.
    In a real implementation, this runs PPO or SAC for N steps.

    player_idx:       0 or 1 (which player is training)
    opponent_policies: the current opponent population
    opponent_mixture:  Nash mixing weights over opponent_policies
    iteration:         current PSRO iteration (for naming)
    """
    # In a real implementation:
    #   1. Create a new neural network policy
    #   2. Run RL training where each episode samples an opponent from
    #      the mixture and plays against it
    #   3. Return the trained policy
    name = f"p{player_idx}_iter{iteration}_oracle"
    return Policy(name)


def psro(initial_policies: List[List[Policy]],
         n_iterations: int = 10,
         n_eval_episodes: int = 50) -> Tuple[List[List[Policy]], np.ndarray, np.ndarray]:
    """
    PSRO outer loop for a two-player zero-sum game.

    initial_policies: [[p1_policies], [p2_policies]] — starting populations
    n_iterations:     number of PSRO rounds to run
    n_eval_episodes:  episodes to average per policy pair in the meta-game

    Returns:
        populations: final policy populations for each player
        sigma1, sigma2: Nash mixing weights over the final populations
    """
    populations = [list(p) for p in initial_policies]

    # Build initial meta-game payoff matrix
    def build_meta_game(pops):
        n1 = len(pops[0])
        n2 = len(pops[1])
        M = np.zeros((n1, n2))
        for i, p1 in enumerate(pops[0]):
            for j, p2 in enumerate(pops[1]):
                r1, r2 = evaluate_policies(p1, p2, n_eval_episodes)
                M[i, j] = r1   # zero-sum: M[i,j] for player 1
        return M

    M = build_meta_game(populations)
    sigma1, sigma2 = solve_zero_sum_nash(M)

    print(f"=== PSRO: {n_iterations} iterations ===")
    print(f"Initial meta-game: {M.shape[0]}x{M.shape[1]}")
    print(f"Initial Nash: {sigma1.round(3)} vs {sigma2.round(3)}")
    print(f"Initial exploitability: {exploitability_meta(M, sigma1, sigma2):.4f}")

    for iteration in range(n_iterations):
        # Train best-response oracles for each player
        new_policy_0 = train_best_response_oracle(
            player_idx=0,
            opponent_policies=populations[1],
            opponent_mixture=sigma2,
            iteration=iteration,
        )
        new_policy_1 = train_best_response_oracle(
            player_idx=1,
            opponent_policies=populations[0],
            opponent_mixture=sigma1,
            iteration=iteration,
        )

        # Add new policies to populations
        populations[0].append(new_policy_0)
        populations[1].append(new_policy_1)

        # Extend meta-game matrix with new rows and columns
        n1_new = len(populations[0])
        n2_new = len(populations[1])
        M_new = np.zeros((n1_new, n2_new))

        # Copy existing payoffs
        old_n1, old_n2 = M.shape
        M_new[:old_n1, :old_n2] = M

        # Fill new row (new_policy_0 vs all of player 2's policies)
        for j, p2 in enumerate(populations[1]):
            r1, _ = evaluate_policies(new_policy_0, p2, n_eval_episodes)
            M_new[n1_new - 1, j] = r1

        # Fill new column (all of player 1's policies vs new_policy_1)
        for i, p1 in enumerate(populations[0]):
            r1, _ = evaluate_policies(p1, new_policy_1, n_eval_episodes)
            M_new[i, n2_new - 1] = r1

        M = M_new

        # Solve updated meta-game
        sigma1, sigma2 = solve_zero_sum_nash(M)
        exploit = exploitability_meta(M, sigma1, sigma2)

        print(f"Iteration {iteration + 1:2d}: "
              f"populations ({n1_new}, {n2_new}), "
              f"exploitability = {exploit:.4f}")

    return populations, sigma1, sigma2


# ── Run with initial hand-crafted policies ──────────────────────────────────
initial = [
    [Policy("p0_aggressive"), Policy("p0_distributed")],
    [Policy("p1_reactive"), Policy("p1_predictive")],
]

final_pops, final_sigma1, final_sigma2 = psro(initial, n_iterations=6)

print(f"\nFinal population sizes: {len(final_pops[0])}, {len(final_pops[1])}")
print(f"Player 1 Nash weights: {final_sigma1.round(3)}")
print(f"Player 2 Nash weights: {final_sigma2.round(3)}")
print(f"Policies in Player 1 population:")
for w, p in zip(final_sigma1, final_pops[0]):
    print(f"  {w:.3f} x {p}")
```

## Practical implementation details

### Mixing strategies during RL training

When training the oracle for iteration k+1, the oracle must play against the meta-Nash mixture \\(\sigma^*\\) over the k existing opponent policies. The implementation is simple: at the start of each training episode, sample a policy index j with probability \\(\sigma^*_j\\), then run the episode against opponent policy j. The RL training loop sees a distribution of opponents rather than a single fixed one.

This is important for generalization: a policy trained against only the strongest opponent in the mixture might be brittle against the other opponents. Training against the full mixture produces a policy that is robust to all opponents in proportion to their Nash weight.

### Evaluating policy pairs: the payoff matrix

Each entry \\(M_{ab}\\) of the meta-game matrix requires simulating the two policies against each other. In an orbital simulation, this might mean running 50-100 episodes of a 24-hour tasking scenario and averaging the coverage scores. The entries need not be exact; PSRO is robust to noise in the payoff estimates. Standard error in the mean of 50-100 episodes is typically small enough.

One practical issue: as the population grows, the number of matrix entries grows quadratically. At iteration k, adding a new policy requires computing k+1 new entries (one row and one column). This is manageable for k up to a few hundred but becomes expensive for very large populations. **Reservoir sampling** addresses this: at each iteration, only evaluate the new policy against a random sample of the existing population rather than all of them, and impute missing entries.

### Initialization: seeding the population

PSRO converges faster when initialized with a diverse starting population. In practice:
- Include one or two domain heuristics (e.g., "always observe the most recently active object")
- Include a random baseline
- If domain knowledge suggests specific opponent strategies (e.g., aggressive blinding), include a counter-strategy in the initial population

A richer starting population means fewer PSRO iterations are needed to reach a good Nash approximation.

## Rectified Nash and variation selection

As the PSRO population grows, many policies in the population may have zero Nash weight — the meta-game solver assigns all weight to a subset of policies. Policies with zero weight contribute nothing to the mixture and should not be trained against.

**Rectified PSRO** (also called \\(\alpha\\)-PSRO in some papers) modifies the oracle training to only play against policies that have positive Nash weight. This focuses training on the relevant part of the strategy space and tends to produce more diverse final populations.

A related issue: the meta-game Nash may assign all weight to a single policy if one policy dominates all others in the current population. This is a degenerate case — the Nash reduces to a pure strategy. Rectified variants use additional exploration to force diversity in the population, ensuring that best-response oracles have something interesting to best-respond to.

## Convergence analysis and the exploitability diagnostic

As PSRO runs, the meta-game grows. A useful diagnostic is to track the exploitability of the meta-Nash at each iteration: how much could a newly trained best-response policy improve over the current Nash mixture? If exploitability is near zero, no new policy would offer a meaningful improvement — the population is approximately at a Nash equilibrium.

The connection to double oracle's convergence guarantee: if the best-response oracle always finds the true best response, the meta-game exploitability is guaranteed to decrease at each iteration. In practice, RL oracles are approximate, so exploitability may not decrease monotonically — but a consistent downward trend over 10-20 iterations is a reliable convergence signal.

One practical issue: the oracle's best response may be very similar to an existing policy in the population. When this happens, adding it to the population does not expand coverage of the strategy space and exploitability barely changes. This is the signal that PSRO has converged. A common stopping criterion is: stop if the new best-response policy improves exploitability by less than ε (e.g., 0.01) for several consecutive iterations.

```python
def psro_with_early_stopping(initial_policies, max_iterations=20,
                              convergence_eps=0.01, patience=3):
    """
    PSRO with early stopping based on exploitability improvement.

    Stops when exploitability improvement is below convergence_eps
    for `patience` consecutive iterations.
    """
    populations = [list(p) for p in initial_policies]

    def build_meta_game(pops):
        n1, n2 = len(pops[0]), len(pops[1])
        M = np.zeros((n1, n2))
        for i, p1 in enumerate(pops[0]):
            for j, p2 in enumerate(pops[1]):
                r1, _ = evaluate_policies(p1, p2)
                M[i, j] = r1
        return M

    M = build_meta_game(populations)
    sigma1, sigma2 = solve_zero_sum_nash(M)
    prev_exploit = exploitability_meta(M, sigma1, sigma2)

    no_improve_count = 0
    iteration = 0

    print(f"Initial exploitability: {prev_exploit:.4f}")

    while iteration < max_iterations:
        new_p0 = train_best_response_oracle(0, populations[1], sigma2, iteration)
        new_p1 = train_best_response_oracle(1, populations[0], sigma1, iteration)

        populations[0].append(new_p0)
        populations[1].append(new_p1)

        n1_new, n2_new = len(populations[0]), len(populations[1])
        M_new = np.zeros((n1_new, n2_new))
        old_n1, old_n2 = M.shape
        M_new[:old_n1, :old_n2] = M

        for j, p2 in enumerate(populations[1]):
            r1, _ = evaluate_policies(new_p0, p2)
            M_new[n1_new - 1, j] = r1

        for i, p1 in enumerate(populations[0]):
            r1, _ = evaluate_policies(p1, new_p1)
            M_new[i, n2_new - 1] = r1

        M = M_new
        sigma1, sigma2 = solve_zero_sum_nash(M)
        curr_exploit = exploitability_meta(M, sigma1, sigma2)

        improvement = prev_exploit - curr_exploit
        print(f"Iter {iteration + 1:2d}: exploit={curr_exploit:.4f}  "
              f"improvement={improvement:.4f}  "
              f"pop sizes=({n1_new},{n2_new})")

        if improvement < convergence_eps:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Converged after {iteration + 1} iterations "
                      f"(no improvement for {patience} consecutive rounds).")
                break
        else:
            no_improve_count = 0

        prev_exploit = curr_exploit
        iteration += 1

    return populations, sigma1, sigma2, M


initial = [
    [Policy("p0_heuristic_A"), Policy("p0_heuristic_B")],
    [Policy("p1_heuristic_A"), Policy("p1_heuristic_B")],
]
_, s1, s2, meta_M = psro_with_early_stopping(initial, max_iterations=15)
print(f"\nFinal meta-game size: {meta_M.shape}")
print(f"Final Nash weights P1: {s1.round(3)}")
print(f"Final Nash weights P2: {s2.round(3)}")
```

Early stopping prevents PSRO from continuing to train policies that provide no new strategic value, saving compute and preventing the policy population from growing unnecessarily large.

## When PSRO is overkill

PSRO is powerful but expensive. Each iteration requires training a full RL agent (potentially millions of gradient steps). For small games with a few dozen actions, fictitious play or CFR is strictly better: exact, fast, and with stronger convergence guarantees.

PSRO is the right choice when:
- The game has a large or continuous action space that neural networks must handle
- The game has complex structure (sequential, partially observable, long-horizon) that requires RL policies
- The equilibrium requires a mixture of qualitatively different strategies (aggressive, defensive, exploitative) that tabular methods cannot represent

For most SSA problems in this curriculum (sensor tasking, spectrum deconfliction), PSRO is appropriate when the problem is large enough to require neural network policies. The orbital coverage game above is a canonical example.

## Key Takeaways

- **PSRO generalizes fictitious play to neural-network policies.** The empirical-frequency table over actions becomes a population of RL-trained policies; the argmax best response becomes a full RL training run. The structure — maintain a population, compute best responses, update the meta-game, repeat — is identical to fictitious play at the abstract level.
- **The meta-game is the key computational shortcut.** Even when the underlying game is enormous (a continuous-action orbital simulation), the meta-game is a small finite matrix with at most k^2 entries for k policies. Solving this matrix for Nash is cheap, even as the policies themselves are large neural networks.
- **Best-response oracle training must target the meta-Nash mixture, not a single opponent.** Training against a mixture produces policies that are robust across all relevant opponents, weighted by their Nash importance. Training against only the strongest opponent produces brittle specialists.
- **Population diversity is required for convergence.** If the meta-Nash concentrates all weight on one policy, PSRO stagnates. Initialization with diverse heuristics and rectified variants that force exploration prevent this collapse.
- **The double oracle algorithm is the theoretical backbone.** PSRO inherits double oracle's guarantee: if the best-response oracle is exact (finds the true best response), the meta-game Nash converges to the true game Nash. In practice, RL oracles are approximate, which weakens but does not eliminate the convergence.
- **Exploitability in the meta-game tracks convergence.** After each PSRO iteration, compute exploitability of the meta-Nash: how much can any new policy improve by deviating? Declining exploitability over iterations is the clearest signal that PSRO is converging to a good equilibrium.

## Quiz

{{#quiz 03-psro.toml}}
