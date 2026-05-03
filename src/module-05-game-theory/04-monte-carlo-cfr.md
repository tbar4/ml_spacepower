# Lesson 4: Monte Carlo CFR (MCCFR)

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

## Quiz

{{#quiz 04-monte-carlo-cfr.toml}}
