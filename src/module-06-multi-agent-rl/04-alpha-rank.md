# Lesson 4: Alpha-Rank


<!-- toc -->

## Where this fits

PSRO produces a population of policies and a Nash equilibrium mixture over them. But Nash equilibrium has practical limitations: it is not unique in general-sum games, it does not tell you which of multiple equilibria will actually emerge, and computing it in large populations is NP-hard. When a research team at DeepMind needed to evaluate and rank dozens of distinct agents trained by different algorithms — agents playing StarCraft II, Quake III, and other large games — Nash equilibrium was not a useful tool.

Alpha-rank provides a different answer: instead of asking "what is the equilibrium?", it asks "which strategies survive evolutionary competition?" The result is a unique ranking over strategies that is always well-defined, computationally efficient, and closely connected to the dynamics of how populations of agents actually evolve over time.

This lesson introduces Alpha-rank as a complement to PSRO. PSRO builds a policy population; Alpha-rank ranks the policies in that population and helps decide which to deploy.

## The problem with Nash for large populations

Nash equilibrium has three practical problems for policy evaluation in complex multi-agent settings.

**Non-uniqueness**: in general-sum games, there may be many Nash equilibria with very different character. A Nash equilibrium in a multi-player space-surveillance allocation game might involve operators focusing on different orbital regimes, but there could be dozens of such equilibria. Which one is actually predictive? Nash theory gives no answer.

**Computational hardness**: finding a Nash equilibrium in an n-player general-sum game is PPAD-complete — a complexity class that is believed to require exponential time in the worst case. For n = 2 (two players), linear programming finds Nash efficiently. For n > 2, there is no known polynomial-time algorithm. Constellations with multiple competing operators quickly leave the tractable regime.

**Lack of dynamics**: Nash equilibrium is a static concept. It tells you which strategy profiles are stable under rational deviation, but it does not tell you how a population of learning agents evolves toward or away from it, or which equilibrium is likely to emerge from a particular learning dynamic.

Alpha-rank addresses all three problems by grounding strategy evaluation in evolutionary game theory rather than rational-agent theory.

## Evolutionary game theory background

**Evolutionary game theory** studies how strategies spread through a population of agents that reproduce (or update) in proportion to their fitness (payoff). Unlike classical game theory, it does not assume that agents are fully rational or solve optimization problems. Instead, it assumes that successful strategies spread and unsuccessful ones die out.

The central equation is the **replicator dynamic**:

\\[ \dot{x}_i = x_i \left( f_i(\mathbf{x}) - \bar{f}(\mathbf{x}) \right) \\]

**Decoding:**
- \\(x_i\\): the fraction of the population currently using strategy \\(i\\)
- \\(\dot{x}_i\\): the rate of change of that fraction (derivative with respect to time)
- \\(f_i(\mathbf{x})\\): the expected fitness (payoff) of strategy \\(i\\) when the population distribution is \\(\mathbf{x}\\)
- \\(\bar{f}(\mathbf{x}) = \sum_j x_j f_j(\mathbf{x})\\): the mean fitness across the whole population
- The equation says: a strategy grows in the population if and only if its fitness exceeds the population mean

Replicator dynamics provide a natural way to think about which strategies flourish and which go extinct when agents copy successful neighbors. They are also deeply connected to gradient descent in policy space, which makes them relevant for RL.

The fixed points of replicator dynamics (where \\(\dot{x}_i = 0\\)) include all Nash equilibria, but not all fixed points are Nash equilibria. Evolutionarily stable strategies (ESS) are a refinement that selects for Nash equilibria that are robust to invasion by small numbers of mutants.

## Fixation probabilities: how a new strategy spreads

The key quantity in Alpha-rank is the **fixation probability**: given a population currently using strategy A, if one agent switches to strategy B, what is the probability that B eventually takes over the entire population?

If B beats A (i.e., a B-agent does better than an A-agent when playing against the current mixed population), the invasion will likely succeed and B will fix. If B loses, the invasion will likely fail and A will remain dominant.

For a population of size N, the fixation probability of strategy \\(j\\) invading a population of strategy \\(i\\) is:

\\[ \rho_{ij} = \frac{1}{\sum_{k=0}^{N-1} \prod_{m=1}^{k} \frac{f_i(m)}{f_j(m)}} \\]

**Decoding:**
- \\(\rho_{ij}\\): the probability that one individual using strategy \\(j\\) takes over a population of N individuals using strategy \\(i\\)
- \\(f_i(m)\\): the fitness of an \\(i\\)-strategist when there are \\(m\\) invaders (\\(j\\)-strategists) in the population
- \\(f_j(m)\\): the fitness of a \\(j\\)-strategist under the same conditions
- The denominator sums over all possible intermediate population states

This formula comes from the theory of stochastic processes in finite populations (the Moran process). For the purpose of Alpha-rank, we only need to evaluate whether \\(\rho_{ij}\\) is greater or less than the neutral fixation probability \\(1/N\\): is strategy \\(j\\) selected for (spreads faster than neutral) or selected against?

In the limit as the selection strength parameter \\(\alpha\\) grows large, the fixation probability simplifies to a step function: strategy \\(j\\) fixes with high probability if it beats \\(i\\), and with near-zero probability if it loses to \\(i\\). This is the "strong selection" limit that gives Alpha-rank its name.

## The Alpha-rank algorithm

Alpha-rank computes a ranking over strategies in a multi-player, multi-population game. The inputs are:

- A payoff matrix (or set of payoff matrices) from head-to-head evaluation of all strategy pairs
- A selection pressure parameter \\(\alpha > 0\\)

The output is a probability distribution over strategies — the stationary distribution of a Markov chain over the strategy space. Strategies with higher stationary probability are ranked higher.

The algorithm in four steps:

**Step 1: Compute pairwise payoffs.** Run all pairs of strategies against each other and record average payoffs. This produces a payoff matrix \\(A\\) where \\(A_{ij}\\) is the average payoff of strategy \\(i\\) against strategy \\(j\\).

**Step 2: Compute transition probabilities.** For each ordered pair \\((i, j)\\), compute the probability that the population transitions from "using strategy i" to "using strategy j." This transition is proportional to the fixation probability \\(\rho_{ij}\\), weighted by how often one agent switches to a new strategy. Under strong selection (large \\(\alpha\\)), strategies that beat the current population spread; strategies that lose do not.

The transition probability from state \\(i\\) to state \\(j \neq i\\) is:

\\[ T_{ij} = \frac{1}{n-1} \cdot \rho_{ij} \\]

where \\(n\\) is the number of strategies (not population size), and \\(1/(n-1)\\) reflects that any of the \\(n-1\\) alternative strategies might be introduced.

**Decoding:**
- \\(T_{ij}\\): probability that the population transitions from all-\\(i\\) to all-\\(j\\) in one step
- \\(1/(n-1)\\): uniform probability of selecting strategy \\(j\\) to introduce as a mutant
- \\(\rho_{ij}\\): fixation probability of \\(j\\) invading \\(i\\) (the evolutionary step)

The diagonal: \\(T_{ii} = 1 - \sum_{j \neq i} T_{ij}\\) (probability of staying in state \\(i\\)).

**Step 3: Find the stationary distribution.** Treat \\(T\\) as a Markov chain transition matrix. The stationary distribution \\(\pi\\) satisfies:

\\[ \pi T = \pi, \quad \sum_i \pi_i = 1 \\]

**Decoding:**
- \\(\pi\\): a row vector of probabilities, one per strategy
- \\(\pi T = \pi\\): left-multiplying the transition matrix by the stationary distribution gives the stationary distribution back (the chain does not move)
- The stationary distribution gives each strategy a probability proportional to how much time the population spends using it over the long run

**Step 4: Rank by stationary probability.** Strategies with higher \\(\pi_i\\) are ranked higher. The top-ranked strategy is the one that, under evolutionary competition with all other strategies, the population spends the most time using.

## SSA example: ranking sensor tasking strategies

Consider a space surveillance network with six candidate sensor-tasking strategies being evaluated for deployment:

- **Strategy 0 (Greedy)**: always observe the object with the highest current conjunction risk
- **Strategy 1 (Balanced)**: weighted blend of risk and staleness
- **Strategy 2 (Predictive)**: observe objects before predicted high-risk windows
- **Strategy 3 (Adversarial)**: focus on objects that an adversary might want unobserved
- **Strategy 4 (Uniform)**: equal revisit time for all objects (baseline)
- **Strategy 5 (Historical)**: prioritize objects with historically high conjunction frequencies

A head-to-head tournament runs each strategy pair for 100 simulated 24-hour coverage windows. The payoff matrix records the coverage differential: how many more high-priority events did strategy \\(i\\) detect than strategy \\(j\\)?

```python
import numpy as np

# ── Payoff matrix from head-to-head tournament ─────────────────────────────────
# A[i, j] = average coverage advantage of strategy i over strategy j
# (Positive means i beats j; negative means j beats i)
# In a purely zero-sum game, A[i,j] = -A[j,i], but with stochasticity
# we allow small asymmetries.

np.random.seed(0)
n_strategies = 6
strategy_names = [
    "Greedy", "Balanced", "Predictive",
    "Adversarial", "Uniform", "Historical"
]

# Simulated tournament results (hand-crafted to reflect reasonable domain logic)
# Rows: strategy i  Cols: strategy j
# A[i,j] > 0 means strategy i outperforms strategy j
A = np.array([
    [ 0.0,  -0.3,  -0.5,   0.8,   1.2,   0.4],   # Greedy
    [ 0.3,   0.0,  -0.2,   0.9,   1.4,   0.6],   # Balanced
    [ 0.5,   0.2,   0.0,   1.1,   1.6,   0.8],   # Predictive
    [-0.8,  -0.9,  -1.1,   0.0,   0.5,  -0.3],   # Adversarial
    [-1.2,  -1.4,  -1.6,  -0.5,   0.0,  -0.9],   # Uniform
    [-0.4,  -0.6,  -0.8,   0.3,   0.9,   0.0],   # Historical
])

# ── Alpha-rank computation ─────────────────────────────────────────────────────

def fixation_probability(payoff_ij, payoff_ii, alpha, N=100):
    """
    Compute fixation probability of strategy j invading a population of strategy i.

    payoff_ij: payoff of strategy j against strategy i (fitness of invader vs resident)
    payoff_ii: payoff of strategy i against itself (fitness of resident)
    alpha:     selection pressure
    N:         population size

    Uses the Moran process formula in the strong-selection approximation.
    """
    # Payoff advantage of j over i at intermediate population states
    # Approximation: fitness difference is constant at A[j,i] - A[i,i]
    # (This is the standard approximation used in Alpha-rank)
    delta = payoff_ij - payoff_ii   # advantage of the mutant over the resident

    if abs(delta) < 1e-10:
        return 1.0 / N   # neutral drift

    # Moran fixation probability under exponential fitness
    # rho = (1 - exp(-alpha * delta)) / (1 - exp(-alpha * N * delta))
    numerator = 1.0 - np.exp(-alpha * delta)
    denominator = 1.0 - np.exp(-alpha * N * delta)

    if abs(denominator) < 1e-15:
        return 1.0 / N

    rho = numerator / denominator
    return float(np.clip(rho, 0, 1))


def compute_transition_matrix(A, alpha, N=100):
    """
    Build the Alpha-rank Markov chain transition matrix.

    A:     payoff matrix, shape (n, n), A[i,j] = payoff of i vs j
    alpha: selection pressure
    N:     population size (controls strength of drift)

    Returns T of shape (n, n): T[i, j] = prob of transitioning from i to j.
    """
    n = A.shape[0]
    T = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Fixation probability of j invading population of i
            # Invader j plays against resident i: A[j, i]
            # Resident i plays against itself: A[i, i]
            rho = fixation_probability(
                payoff_ij=A[j, i],   # j playing against i
                payoff_ii=A[i, i],   # i playing against i
                alpha=alpha,
                N=N
            )
            T[i, j] = (1.0 / (n - 1)) * rho

        # Diagonal: probability of staying
        T[i, i] = 1.0 - T[i, :].sum()

    return T


def stationary_distribution_power(T, n_iters=10000, tol=1e-10):
    """
    Find the stationary distribution of Markov chain T by power iteration.
    Starts from a uniform distribution and repeatedly multiplies by T.

    T:       transition matrix, shape (n, n), rows sum to 1
    Returns: stationary distribution as a probability vector of length n
    """
    n = T.shape[0]
    pi = np.ones(n) / n

    for _ in range(n_iters):
        pi_new = pi @ T
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new

    # Normalize to ensure exact sum-to-one (numerical cleanup)
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi


def alpha_rank(A, alpha=100.0, N=100):
    """
    Compute Alpha-rank scores for an n-strategy game.

    A:     payoff matrix, shape (n, n)
    alpha: selection pressure (higher = stronger selection)
    N:     population size

    Returns:
        scores: stationary distribution (Alpha-rank scores), shape (n,)
        ranking: indices sorted by score descending
        T: the Markov transition matrix
    """
    T = compute_transition_matrix(A, alpha=alpha, N=N)
    scores = stationary_distribution_power(T)
    ranking = np.argsort(scores)[::-1]
    return scores, ranking, T


# ── Run Alpha-rank on the tournament results ───────────────────────────────────
alpha = 50.0
scores, ranking, T = alpha_rank(A, alpha=alpha, N=100)

print("=== Alpha-rank results (alpha={:.1f}) ===".format(alpha))
print(f"\n{'Rank':<6} {'Strategy':<15} {'Score':>10}")
print("-" * 35)
for rank_pos, strat_idx in enumerate(ranking):
    print(f"  {rank_pos + 1:<4} {strategy_names[strat_idx]:<15} {scores[strat_idx]:>10.4f}")

print("\nTransition matrix (rows = from, cols = to):")
header = "        " + "".join(f"{s[:4]:>8}" for s in strategy_names)
print(header)
for i, row in enumerate(T):
    print(f"  {strategy_names[i][:8]:<10}" + "".join(f"{v:>8.3f}" for v in row))
```

The output will show a clear ranking of the six strategies. Predictive and Balanced typically score highest because they consistently beat most other strategies in the tournament. Uniform (random) should score lowest because it loses to almost everything. The Alpha-rank scores are proportional to how long an evolving population spends using each strategy when strategies compete and spread according to their tournament performance.

## Connection to reinforcement learning

Alpha-rank is not a learning algorithm — it does not train policies. Its role is **evaluation and selection**: given a set of policies produced by any training method (RL, PSRO, hand-coding, random search), which ones are most robust?

This makes Alpha-rank a natural post-processing step for PSRO. After PSRO builds a population of policies over many iterations, Alpha-rank provides a principled ranking:

1. Run a tournament: evaluate all pairs of PSRO policies against each other and record payoffs.
2. Apply Alpha-rank: compute the stationary distribution over policies.
3. Select for deployment: deploy the top-ranked policy (highest stationary probability) or a mixture weighted by Alpha-rank scores.

This is more informative than Nash equilibrium for selecting a deployment policy because:
- Nash may assign weight to multiple policies, requiring a randomized deployment
- Alpha-rank provides a deterministic ranking that is easy to communicate to operators
- Alpha-rank captures evolutionary robustness, not just static optimality
- Alpha-rank is unique (for a given \\(\alpha\\)), avoiding the non-uniqueness problem of Nash

## Choosing alpha: selection pressure sensitivity

The parameter \\(\alpha\\) controls how strongly fitness differences translate into fixation probability differences.

- **Low \\(\alpha\\) (weak selection)**: the fixation probability of any strategy is close to \\(1/N\\) regardless of fitness differences. The Markov chain is nearly uniform; all strategies have similar Alpha-rank scores. Ranking is uninformative.

- **High \\(\alpha\\) (strong selection)**: strategies that beat the resident fix with high probability; strategies that lose fix with near-zero probability. The Alpha-rank scores concentrate on the strategies that consistently win head-to-head.

- **Very high \\(\alpha\\) (dominant selection)**: the ranking approaches a pure dominance ordering. A strategy gets a high score only if it beats most other strategies.

In practice, \\(\alpha\\) is chosen to be large enough to produce a clear ranking but not so large that the Markov chain becomes poorly conditioned. A common approach is to test a range and check that rankings are stable.

```python
import numpy as np

def sensitivity_analysis(A, strategy_names, alphas=None):
    """
    Compute Alpha-rank scores for multiple values of alpha and show how
    rankings change with selection pressure.
    """
    if alphas is None:
        alphas = [1.0, 5.0, 20.0, 100.0, 500.0]

    n = A.shape[0]
    print("=== Alpha sensitivity analysis ===")
    print(f"\n{'Alpha':>8}  " + "  ".join(f"{s[:10]:>10}" for s in strategy_names))
    print("-" * (12 + 12 * n))

    for alpha in alphas:
        scores, ranking, _ = alpha_rank(A, alpha=alpha, N=100)
        row = f"{alpha:>8.1f}  " + "  ".join(f"{scores[i]:>10.4f}" for i in range(n))
        print(row)

    print()
    print("Top-ranked strategy by alpha:")
    for alpha in alphas:
        scores, ranking, _ = alpha_rank(A, alpha=alpha, N=100)
        print(f"  alpha={alpha:>6.1f}: {strategy_names[ranking[0]]} (score={scores[ranking[0]]:.4f})")


sensitivity_analysis(A, strategy_names)
```

The output will show that at low alpha, all six strategies score near 1/6 (uniform). As alpha increases, the ranking sharpens: Predictive and Balanced pull ahead, Uniform falls to the bottom. At very high alpha, one or two strategies dominate the distribution entirely.

A useful diagnostic: if the ranking changes significantly between \\(\alpha = 50\\) and \\(\alpha = 500\\), there are strategies that perform only slightly better than others and the ranking is sensitive to noise. If the ranking is stable across a wide range of alpha, the dominance structure is robust.

## Multi-population Alpha-rank

The description so far assumes a single population where all strategies compete against each other. Many SSA games involve distinct populations: Red operators and Blue operators, or a sensor-allocation game between multiple satellite operators with different assets and objectives.

Multi-population Alpha-rank extends the single-population version by computing a separate Markov chain for each population, with fixation probabilities that depend on the current strategy of the other population.

For two populations with strategy sets of size \\(n_1\\) and \\(n_2\\), the joint state space has \\(n_1 \times n_2\\) states. The transition matrix is built similarly: the probability of population 1 transitioning from strategy \\(i\\) to strategy \\(j\\) depends on the current strategy of population 2, and vice versa.

The stationary distribution of this joint Markov chain gives a distribution over (strategy-for-population-1, strategy-for-population-2) pairs. Marginalizing gives the individual strategy rankings.

For the SSA coverage game with two operators, this would mean:
- Red has a population of sensor-tasking policies
- Blue has a population of sensor-tasking policies
- The joint ranking captures which Red-Blue strategy pairs dominate evolutionary competition

The computational cost scales as \\(O((n_1 n_2)^2)\\) for power iteration on the joint chain, which is manageable for populations of 10-20 strategies each.

## Full example: stationary distribution by power iteration

Power iteration is the simplest way to compute the stationary distribution. Starting from a uniform distribution, repeatedly multiply by the transition matrix. The distribution converges to the stationary distribution.

```python
import numpy as np

def visualize_markov_chain(T, strategy_names, scores):
    """
    Print a text visualization of which strategies have high transition
    probability into/out of them.
    """
    n = len(strategy_names)
    print("\n=== Markov chain summary ===")
    print(f"{'Strategy':<15} {'Score':>8}  {'Largest inflow from':<20} {'Largest outflow to':<20}")
    print("-" * 70)

    for i in range(n):
        # Inflow: T[j, i] for j != i, weighted by scores[j]
        inflow = np.array([T[j, i] * scores[j] if j != i else 0 for j in range(n)])
        outflow = np.array([T[i, j] if j != i else 0 for j in range(n)])

        main_in = strategy_names[np.argmax(inflow)] if inflow.sum() > 0 else "none"
        main_out = strategy_names[np.argmax(outflow)] if outflow.sum() > 0 else "none"

        print(f"{strategy_names[i]:<15} {scores[i]:>8.4f}  {main_in:<20} {main_out:<20}")


# Convergence behavior of power iteration
print("=== Power iteration convergence ===")
T_demo = compute_transition_matrix(A, alpha=50.0, N=100)
n = T_demo.shape[0]
pi = np.ones(n) / n

for step in [1, 5, 20, 100, 500, 2000]:
    pi_temp = np.ones(n) / n
    for _ in range(step):
        pi_temp = pi_temp @ T_demo
    pi_temp = np.maximum(pi_temp, 0)
    pi_temp /= pi_temp.sum()
    # Entropy as a measure of convergence (decreases as ranking sharpens)
    entropy = -np.sum(pi_temp * np.log(pi_temp + 1e-15))
    top_strat = strategy_names[np.argmax(pi_temp)]
    print(f"  After {step:5d} steps: entropy={entropy:.3f}, top strategy={top_strat}")

# Final Alpha-rank and chain visualization
final_scores, final_ranking, final_T = alpha_rank(A, alpha=50.0)
visualize_markov_chain(final_T, strategy_names, final_scores)
```

The entropy of the distribution decreases as power iteration converges, starting near \\(\log(6) \approx 1.79\\) (uniform over 6 strategies) and settling to a lower value as the distribution concentrates on the top strategies. The chain summary shows the dominant evolutionary pathways: which strategies are primarily replaced by which others.

## Key Takeaways

- **Alpha-rank provides a unique ranking of strategies that Nash equilibrium cannot.** Nash equilibria are non-unique in general-sum games and NP-hard to compute for multi-player games. Alpha-rank's stationary distribution is always unique and can be computed in polynomial time via power iteration.
- **The ranking is grounded in evolutionary dynamics, not rational-agent assumptions.** Alpha-rank asks: under evolutionary competition (better strategies spread; worse strategies die out), which strategies dominate the long run? This is more descriptive of how populations of RL agents actually evolve than Nash's rational-agent framing.
- **The selection pressure alpha controls the sharpness of the ranking.** Low alpha gives a near-uniform distribution (all strategies similar). High alpha concentrates probability on the dominant strategies. Sensitivity analysis across alpha values reveals whether the ranking is robust or depends on the exact choice of selection strength.
- **Alpha-rank is a natural complement to PSRO.** PSRO builds a population of policies; Alpha-rank ranks them. After PSRO produces a population, a tournament evaluation plus Alpha-rank identifies which policy is most evolutionarily robust and should be deployed.
- **Power iteration on the Markov chain is the simplest implementation.** The transition matrix has at most \\(n^2\\) entries for \\(n\\) strategies; power iteration converges in a few thousand steps for typical problems. No special solver is required.
- **Multi-population Alpha-rank handles games with distinct agent types.** When different players have structurally different strategy spaces (e.g., Red and Blue operators with different assets), the joint Markov chain over all population-strategy combinations provides a unified ranking that accounts for cross-population interactions.

## Quiz

{{#quiz 04-alpha-rank.toml}}
