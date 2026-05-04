# Lesson 4: Monte Carlo CFR (MCCFR)

**Module/Source:** Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in Extensive Games" (NeurIPS 2009) — the paper that introduced and analyzed outcome sampling and external sampling MCCFR. Gibson et al. (2012) "Generalized Sampling and Variance in Counterfactual Regret Minimization" for variance analysis. Brown and Sandholm (2019) "Solving Imperfect-Information Games via Discounted Regret Minimization" for discounted and linear CFR variants. Background on importance sampling: *Monte Carlo Statistical Methods* (Robert and Casella, 2004). Game theory foundations: Osborne (2004) Chapters 6–7; Zinkevich et al. (2007).

## Where this fits

Vanilla CFR (lesson 3) is correct but slow: every iteration traverses the entire game tree. For games beyond a certain size, this is hopeless. MCCFR replaces the full tree traversal with a sampled traversal, just like Monte Carlo (Module 1, lesson 3) replaces an intractable expectation with a sample-based estimate. The trade-off is the same: noisier per-iteration updates, but many more iterations are possible. MCCFR is the workhorse algorithm for medium-sized games and the foundation of deep CFR (next lesson).

## The bottleneck of vanilla CFR

For a game tree with N nodes, vanilla CFR does O(N) work per iteration. To converge to ε-Nash, it needs O(1/ε²) iterations. Total work: O(N/ε²).

Concrete numbers for poker:
- No-limit Hold'em: ~10^14 information sets
- One iteration: at least 10^14 operations
- For ε = 0.01: at least 10^14 × 10^4 = 10^18 operations total

This is infeasible. MCCFR aims to dramatically reduce per-iteration cost by sampling, accepting higher per-iteration variance in exchange.

## The two main MCCFR variants

### Outcome sampling

In **outcome sampling**, each iteration samples one complete trajectory (root to terminal) and updates regrets only along that trajectory. At each chance node, sample one outcome from the chance distribution. At each player decision, sample one action from the player's current strategy.

Per-iteration cost: O(D) where D is game depth (typically much less than N, the total tree size).

The variance is high because each iteration only updates one trajectory's worth of information sets. But you can run many more iterations per unit of compute.

### External sampling

In **external sampling** (the most popular MCCFR variant), at each iteration:
- For one player (the "traverser"), explore all of their actions at every information set
- For the other player and chance nodes, sample one action

This explores more of the tree than outcome sampling but less than vanilla CFR. The per-iteration cost is intermediate.

External sampling has lower variance than outcome sampling and converges faster in practice. It is what most production CFR implementations use.

## Outcome sampling in detail

Here is the algorithm for one iteration of outcome sampling:

```
1. Sample a complete trajectory by:
   - At chance nodes: sample one outcome from the chance distribution
   - At player nodes: sample one action from the player's current strategy
   
2. Walk back through the trajectory:
   At each information set I belonging to player i along the trajectory:
       Compute the regret for actions other than the one sampled
       (using counterfactual values estimated from the sample)
       Update R(I, a) by adding the regret divided by the sampling probability
       Update the strategy at I via regret matching
       Update the average strategy at I
```

The key trick is the **importance weighting**: divide the regret update by the probability of having sampled this trajectory. This ensures the estimator is unbiased: the expected update equals the vanilla CFR update.

## A simplified outcome-sampling implementation

```python
import numpy as np
from collections import defaultdict
import random

class OutcomeSamplingMCCFR:
    def __init__(self, game):
        self.game = game
        self.regrets       = defaultdict(lambda: np.zeros(game.num_actions()))
        self.strategy_sum  = defaultdict(lambda: np.zeros(game.num_actions()))
    
    def get_strategy(self, info_set, num_actions):
        regrets = self.regrets[info_set]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(num_actions) / num_actions
    
    def cfr_iteration(self, history, sample_prob, traversing_player):
        """
        Recursive outcome-sampling iteration.
        history: current state
        sample_prob: probability of having sampled this path
        traversing_player: which player we're updating regrets for this iteration
        Returns: (utility, sample_probability) tuple from this point onward
        """
        if history.is_terminal():
            return history.returns()[traversing_player], 1.0
        
        if history.is_chance_node():
            outcomes = history.chance_outcomes()
            actions, probs = zip(*outcomes)
            sampled_idx = np.random.choice(len(actions), p=probs)
            action = actions[sampled_idx]
            sampled_prob = probs[sampled_idx]
            next_history = history.apply(action)
            utility, downstream_prob = self.cfr_iteration(
                next_history, sample_prob * sampled_prob, traversing_player
            )
            return utility, sampled_prob * downstream_prob
        
        player = history.current_player()
        info_set = history.info_set()
        legal = history.legal_actions()
        strategy = self.get_strategy(info_set, len(legal))
        
        if player == traversing_player:
            # We need to estimate regrets for ALL actions, but only sample one.
            # Use importance weighting. For simplicity, this version just samples
            # one action and updates that one's regret estimate.
            sampled_idx = np.random.choice(len(legal), p=strategy)
            sampled_prob = strategy[sampled_idx]
            
            next_history = history.apply(legal[sampled_idx])
            utility, downstream_prob = self.cfr_iteration(
                next_history, sample_prob * sampled_prob, traversing_player
            )
            
            # Compute estimated regret for the sampled action
            # (in full outcome sampling, this is more sophisticated)
            estimated_regret = utility / (sample_prob * sampled_prob)
            
            # Simplified update: increment regret for sampled action, decrement for others
            for i in range(len(legal)):
                if i == sampled_idx:
                    self.regrets[info_set][i] += estimated_regret * (1 - strategy[i])
                else:
                    self.regrets[info_set][i] -= estimated_regret * strategy[i]
            
            # Average strategy
            for i in range(len(legal)):
                self.strategy_sum[info_set][i] += strategy[i]
            
            return utility, sampled_prob * downstream_prob
        else:
            # Other player: just sample their action and continue
            sampled_idx = np.random.choice(len(legal), p=strategy)
            sampled_prob = strategy[sampled_idx]
            next_history = history.apply(legal[sampled_idx])
            utility, downstream_prob = self.cfr_iteration(
                next_history, sample_prob * sampled_prob, traversing_player
            )
            return utility, sampled_prob * downstream_prob
    
    def run(self, iterations=100_000):
        for it in range(iterations):
            # Alternate which player we update each iteration
            player = it % self.game.num_players()
            initial = self.game.new_initial_state()
            self.cfr_iteration(initial, 1.0, player)
```

A note on this implementation: it is a simplified illustration. Real outcome-sampling MCCFR has additional bookkeeping for the importance weights and probability factorizations that make the estimator unbiased. The full algorithm is in the Lanctot et al. 2009 paper "Monte Carlo Sampling for Regret Minimization in Extensive Games."

## External sampling

External sampling is more commonly implemented because it has better empirical convergence. The structure:

```
For each iteration:
    For each player p (alternating each iteration):
        Traverse the tree
        At chance nodes: sample one outcome
        At nodes belonging to player p: explore all actions
        At nodes belonging to other players: sample one action
        Update regrets at all visited p-nodes
```

Per-iteration cost is O(branching_factor^depth_for_traverser × 1^depth_for_others).

External sampling has the advantage that the regret estimates are exact for the traverser at each information set visited (no importance weighting needed for the traverser's actions).

## The variance-vs-speed tradeoff

| Method | Per-iteration cost | Iterations needed | Total cost |
|--------|-------------------|-------------------|------------|
| Vanilla CFR | O(N) | O(1/ε²) | O(N/ε²) |
| External sampling | O(N/p^k) | O(C/ε²) for C > 1 | varies |
| Outcome sampling | O(D) | O(D/ε²) | O(D²/ε²) |

(N = tree size, D = depth, p = player branching factor, k = path length, ε = target accuracy.)

Outcome sampling is the cheapest per iteration but needs the most iterations. External sampling is in the middle. Vanilla is most expensive per iteration but converges with the fewest.

For most large games, external sampling wins overall because the per-iteration savings outweigh the additional iterations. For huge games, outcome sampling combined with deep neural networks (deep CFR) is the state of the art.

## When MCCFR works and when it does not

**Works well when:**
- The game tree is too large for vanilla CFR
- You can sample efficiently from chance distributions and strategies
- You have a CPU/GPU budget that supports many iterations

**Struggles when:**
- The game has very rare but high-value information sets that are unlikely to be sampled
- You need very tight convergence (ε very small)
- Memory for the regret table is the bottleneck (next lesson addresses this)

For our SSA conjunction game, MCCFR is overkill: vanilla CFR will converge fast enough. But understanding MCCFR is essential for understanding how the algorithm scales.

## Variants and improvements

**Probing variants** (CFR+): only update positive regret estimates, smoother convergence behavior.

**Discounted CFR**: weight more recent iterations more heavily, helps with non-stationary regret patterns.

**CFR with Linear weighting**: weight recent iterations linearly more than earlier ones. Often dramatically faster than vanilla.

For the project, you can stick with vanilla CFR. But know that production CFR implementations always use one of these improved variants in practice.

## Outcome sampling vs. external sampling: a closer look

### Which nodes are sampled

The two variants differ in exactly which nodes they visit per iteration:

**Outcome sampling** visits a single root-to-leaf path. At every node — whether chance, traverser, or opponent — a single action is sampled. The result is one complete play-through of the game per iteration.

- Visited nodes: \\(O(D)\\) where \\(D\\) is the maximum depth of the game tree
- Updated information sets: only the traverser's information sets that appear on the sampled path
- Unvisited information sets: receive no update this iteration, regardless of how important they are

**External sampling** visits more of the tree. At opponent and chance nodes, one action is sampled. But at the traverser's nodes, all actions are explored.

- Visited nodes: \\(O(b^{d_\text{traverser}})\\) where \\(b\\) is the traverser's branching factor and \\(d_\text{traverser}\\) is the depth of the traverser's decisions
- Updated information sets: all traverser information sets reachable under the sampled opponent/chance play
- Every traverser information set reachable in this trajectory is updated with an exact regret estimate

### Variance tradeoffs

Outcome sampling has high variance because a single trajectory is an extremely noisy estimate of the counterfactual values. An information set that is on the sampled path receives a large update; an information set just one step away receives nothing.

External sampling has lower variance for the traverser's estimates because all of the traverser's actions are explored exactly. The only noise comes from the sampled opponent/chance play. Across many iterations, the opponent play is sampled uniformly, providing an unbiased estimate of the counterfactual reach probabilities.

### When each is preferred

| Scenario | Preferred variant | Reason |
|----------|-------------------|--------|
| Very deep game trees (depth > 100) | Outcome sampling | External sampling's per-iteration cost grows with branching factor^depth |
| Many information sets per depth level | External sampling | Fewer iterations needed to cover all traverser info sets |
| High-variance game (rare high-payoff outcomes) | External sampling | Variance from rare outcomes is absorbed by exact traversal of traverser nodes |
| Memory bottleneck (can't store all info sets) | Outcome sampling (with Deep CFR) | Enables sampling-based neural network training |
| Real-time decision under a tight time budget | Outcome sampling | Constant per-iteration cost regardless of game size |

In SSA contexts: for the satellite-vs-jammer hide-and-seek game with many frequency bands but a short decision horizon (depth ~ 5–10), external sampling works well. For a long-horizon ISR sensor scheduling game (depth ~ 50–100), outcome sampling is more practical.

## The importance sampling correction

### Why sampled outcomes need reweighting

When outcome sampling visits only a subset of the game tree, it must correct for the fact that some trajectories are sampled more frequently than others. Without correction, the algorithm would overweight common paths and underweight rare-but-important ones.

The correction factor is the **importance weight**: the ratio of the actual probability of an outcome to the sampling probability used to select it.

### The w/q factor

In outcome sampling MCCFR, when a trajectory \\(\tau\\) is sampled with probability \\(q(\tau)\\) under the current sampling distribution, the regret estimate for action \\(a\\) at information set \\(I\\) on that trajectory is:

\\[ \hat{r}(I, a) = \frac{\pi_{-i}(I) \cdot (u_a - u_{\sigma})}{q(\tau)} \\]

**Decoding:**
- \\(\pi_{-i}(I)\\): counterfactual reach of information set \\(I\\) (opponents' and chance's contribution)
- \\(u_a\\): the utility at the sampled terminal node if we had taken action \\(a\\) at \\(I\\) (holding the rest of the trajectory constant)
- \\(u_\sigma\\): the utility at the sampled terminal under the current strategy
- \\(q(\tau)\\): the probability of sampling this particular trajectory

The division by \\(q(\tau)\\) is the **importance weight**. It corrects for the fact that if we sample a trajectory with probability \\(q\\) but it would naturally occur with probability \\(p\\), the update should be scaled by \\(p/q\\) to make it unbiased. In outcome sampling, \\(q(\tau)\\) includes the player's own strategy probabilities, so this factor partially cancels with the strategy probabilities in \\(\pi_{-i}\\).

### Variance can increase with importance weighting

Importance weighting is unbiased in expectation, but it can dramatically increase variance. Consider:

- A trajectory that naturally occurs with probability \\(p = 0.001\\) (very rare)
- Under the sampling distribution, it is sampled with probability \\(q = 0.0001\\) (10× undersampled)
- The importance weight is \\(p/q = 10\\)
- If this trajectory's terminal payoff is \\(+5\\), the weighted update is \\(+50\\) — much larger than the actual effect on expected payoff

This variance amplification is the fundamental tension in MCCFR: sampling rare trajectories infrequently is efficient per iteration, but the resulting high-variance updates mean more iterations are needed for the estimates to stabilize.

```python
import numpy as np

def importance_sampling_variance_demo(
    p_rare: float,
    q_rare: float,
    u_rare: float,
    u_common: float,
    n_samples: int = 10_000
):
    """
    Demonstrate variance amplification in importance-weighted estimators.

    We want to estimate E[U] = p_rare * u_rare + (1 - p_rare) * u_common.

    Two estimators:
    1. Direct sampling: sample from p, compute average.
    2. Importance-weighted sampling from q: correct with w = p/q.

    Args:
        p_rare: true probability of the rare event
        q_rare: sampling probability of the rare event (if q_rare < p_rare, undersampling)
        u_rare: utility of the rare event
        u_common: utility of the common event
        n_samples: number of samples
    """
    true_mean = p_rare * u_rare + (1 - p_rare) * u_common

    # Direct sampling
    samples_direct = np.where(
        np.random.random(n_samples) < p_rare,
        u_rare, u_common
    )
    direct_mean = samples_direct.mean()
    direct_var  = samples_direct.var()

    # Importance-weighted sampling from q
    is_samples = []
    for _ in range(n_samples):
        if np.random.random() < q_rare:
            # Sampled the rare event; importance weight p_rare / q_rare
            w = p_rare / q_rare
            is_samples.append(w * u_rare)
        else:
            w = (1 - p_rare) / (1 - q_rare)
            is_samples.append(w * u_common)
    is_samples = np.array(is_samples)
    is_mean = is_samples.mean()
    is_var  = is_samples.var()

    print(f"True mean: {true_mean:.4f}")
    print(f"Direct sampling:  mean={direct_mean:.4f}, variance={direct_var:.4f}")
    print(f"IS sampling:      mean={is_mean:.4f},    variance={is_var:.4f}")
    print(f"Variance ratio (IS / direct): {is_var / direct_var:.2f}x")


# SSA scenario: rare conjunction event (occurs 0.1% of time)
# Under IS, we sample it 10x less often (0.01%) to save computation
# Utility of conjunction if unhandled: -100 (catastrophic)
# Utility of nominal operations: +1
print("=== Rare conjunction event (IS undersamples by 10x) ===")
importance_sampling_variance_demo(
    p_rare=0.001, q_rare=0.0001, u_rare=-100.0, u_common=1.0, n_samples=50_000
)

print("\n=== Rare event (IS matches true probability, no correction needed) ===")
importance_sampling_variance_demo(
    p_rare=0.001, q_rare=0.001, u_rare=-100.0, u_common=1.0, n_samples=50_000
)
```

The output demonstrates that when \\(q \neq p\\), variance increases proportionally to \\((p/q)^2\\). Undersampling rare high-payoff events by 10× multiplies variance by up to 100×. This is why MCCFR practitioners are careful about the sampling distribution and why some variants use **ε-greedy sampling** (mixing the strategy with a small uniform component) to ensure all actions get sampled with a minimum probability.

## Convergence bounds for MCCFR

### The T^{-1/2} bound still holds

Like vanilla CFR, both outcome sampling and external sampling MCCFR converge at rate \\(O(T^{-1/2})\\):

\\[ \epsilon(T) \leq \frac{C_{\text{MC}}}{\sqrt{T}} \\]

The convergence rate exponent is the same. This might seem surprising — sampling introduces variance, which should slow convergence. The reason the rate exponent is preserved: regret matching already achieves \\(O(1/\sqrt{T})\\) convergence regardless of the noise level in the per-iteration update, as long as the estimates are unbiased and have finite variance.

### But the constant is larger

The crucial difference is in the constant \\(C_{\text{MC}}\\). For vanilla CFR, the constant depends on the game structure (payoff ranges, number of information sets). For MCCFR:

\\[ C_{\text{MC}} = C_{\text{vanilla}} \cdot \sqrt{\text{Var}_{\text{sampling}}} \\]

where \\(\text{Var}_{\text{sampling}}\\) is the variance introduced by the sampling procedure. This variance depends on:
- **Outcome sampling**: variance proportional to \\(\Delta^2 / q_{\min}\\), where \\(\Delta\\) is the payoff range and \\(q_{\min}\\) is the minimum probability of sampling any terminal node.
- **External sampling**: lower variance because the traverser's regrets are estimated exactly; variance only comes from the opponent's sampled actions.

### Practical implication for iteration count

Let \\(\rho = C_{\text{MC}} / C_{\text{vanilla}}\\) be the variance ratio. For a given target exploitability \\(\epsilon\\):

\\[ T_{\text{MCCFR}} = \rho^2 \cdot T_{\text{vanilla}} \\]

If outcome sampling has \\(\rho = 10\\) (estimated 10× higher constant due to variance), MCCFR needs 100× more iterations to match vanilla CFR's convergence. But since each MCCFR iteration is \\(O(D)\\) vs. \\(O(N)\\) for vanilla CFR:

\\[ \text{Total cost ratio} = \frac{T_{\text{MCCFR}} \cdot O(D)}{T_{\text{vanilla}} \cdot O(N)} = \rho^2 \cdot \frac{D}{N} \\]

For typical games, \\(D \ll N / \rho^2\\), so MCCFR wins overall. For shallow games with high branching (e.g., a 3-action game with depth 5, \\(N = 3^5 = 243\\), \\(D = 5\\)), the advantage is:

\\[ \frac{T_{\text{MCCFR}} \cdot 5}{T_{\text{vanilla}} \cdot 243} = 100 \cdot \frac{5}{243} \approx 2 \\]

Only a 2× improvement — vanilla CFR is competitive. For a deeper game (depth 20, \\(N = 3^{20} \approx 3.5 \times 10^9\\)):

\\[ 100 \cdot \frac{20}{3.5 \times 10^9} \approx 6 \times 10^{-7} \\]

A massive advantage for MCCFR. The deeper the game, the more MCCFR dominates.

```python
import numpy as np

def mccfr_vs_vanilla_cost_comparison(
    game_branching: int,
    game_depth: int,
    variance_ratio_rho: float,
    target_epsilon: float,
    vanilla_constant: float = 1.0,
):
    """
    Compare total computational cost of vanilla CFR vs. MCCFR.

    Args:
        game_branching: average branching factor
        game_depth: game tree depth
        variance_ratio_rho: C_MCCFR / C_vanilla (how much variance MCCFR adds)
        target_epsilon: desired exploitability
        vanilla_constant: the C_vanilla constant in epsilon = C/sqrt(T)
    """
    N = game_branching ** game_depth  # approximate tree size
    D = game_depth

    mccfr_constant = vanilla_constant * variance_ratio_rho

    T_vanilla = (vanilla_constant / target_epsilon) ** 2
    T_mccfr   = (mccfr_constant   / target_epsilon) ** 2

    cost_vanilla = T_vanilla * N
    cost_mccfr   = T_mccfr   * D

    print(f"Game: branching={game_branching}, depth={game_depth}")
    print(f"  Tree size N = {N:,.0f}")
    print(f"  Target epsilon = {target_epsilon}")
    print(f"  Vanilla CFR:  {T_vanilla:,.0f} iters × {N:,.0f} nodes = {cost_vanilla:.2e} ops")
    print(f"  MCCFR:        {T_mccfr:,.0f} iters × {D:,.0f} nodes = {cost_mccfr:.2e} ops")
    print(f"  MCCFR speedup: {cost_vanilla / cost_mccfr:.1f}x")
    print()


# Small SSA conjunction game (realistic for vanilla CFR)
mccfr_vs_vanilla_cost_comparison(
    game_branching=2, game_depth=6,
    variance_ratio_rho=5.0, target_epsilon=0.01, vanilla_constant=1.0
)

# Medium ISR sensor scheduling game
mccfr_vs_vanilla_cost_comparison(
    game_branching=4, game_depth=10,
    variance_ratio_rho=8.0, target_epsilon=0.01, vanilla_constant=2.0
)

# Large satellite-jammer frequency game (many frequency bands)
mccfr_vs_vanilla_cost_comparison(
    game_branching=16, game_depth=15,
    variance_ratio_rho=10.0, target_epsilon=0.05, vanilla_constant=5.0
)
```

The comparison shows that for the small SSA game, vanilla CFR is competitive or even preferred. For the large frequency game, MCCFR is orders of magnitude faster — the depth savings overwhelm the variance penalty.

## Key Takeaways

- MCCFR replaces the full game tree traversal of vanilla CFR with a sampled traversal, trading per-iteration accuracy for the ability to run far more iterations; the same \\(O(T^{-1/2})\\) convergence rate applies, but with a larger constant.
- **Outcome sampling** visits a single root-to-leaf trajectory per iteration (cost \\(O(D)\\)), while **external sampling** explores all traverser actions but samples opponent/chance play (cost \\(O(b^{d_\text{traverser}})\\)); external sampling has lower variance and is preferred for medium-sized games.
- **Importance weighting** (dividing regret updates by the sampling probability \\(q(\tau)\\)) makes the MCCFR estimator unbiased, but can amplify variance by \\((p/q)^2\\) when rare events are undersampled — practitioners mitigate this with ε-greedy sampling to ensure all actions get a minimum sampling probability.
- The \\(T^{-1/2}\\) convergence bound still holds for MCCFR, but the constant \\(C_{\text{MC}}\\) is larger than for vanilla CFR; the total cost advantage of MCCFR grows with game depth and is most dramatic for deep trees where \\(D \ll N / \rho^2\\).
- For very large games (poker, large SSA scheduling problems), even tabular MCCFR cannot store regrets for all information sets; this motivates Deep CFR (Lesson 5), which replaces the regret table with a neural network.
- In SSA applications, MCCFR is the practical algorithm for games with more than a few thousand information sets: the conjunction maneuver coordination game fits vanilla CFR, but a multi-satellite spectrum deconfliction game with many operators and frequency bands requires MCCFR or its deep variants.

## Quiz

{{#quiz 04-monte-carlo-cfr.toml}}
