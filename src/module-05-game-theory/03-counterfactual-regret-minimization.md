# Lesson 3: Counterfactual Regret Minimization (CFR)

**Module/Source:** Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information" (NeurIPS 2007) — the original CFR paper. Tammelin et al. (2015) "Solving Large Imperfect Information Games Using CFR+" for the CFR+ / regret matching+ variant. Convergence analysis follows Bowling et al. (2015) and Brown and Sandholm (2019) "Solving Imperfect-Information Games via Discounted Regret Minimization." Background on online learning and regret bounds: Cesa-Bianchi and Lugosi (2006) *Prediction, Learning, and Games*, Chapter 4. The game theory foundations follow Osborne (2004) Chapters 1–3 and 6–7.

## Where this fits

This is the algorithm that the entire module builds toward. CFR is the workhorse of computational game theory: it solves extensive-form games (including imperfect-information ones) and converges to Nash equilibrium. Variants of CFR have produced superhuman poker bots (Cepheus solved limit Texas Hold'em; Libratus and Pluribus solved no-limit Hold'em). The mathematics behind CFR is more involved than what we have seen so far, but the algorithm itself is surprisingly simple once the conceptual pieces are in place.

This lesson introduces vanilla CFR. The next two lessons cover the variants needed for actual large games (MCCFR and deep CFR).

## The core idea: regret matching

Forget about extensive-form games for a moment. Suppose you have a single decision to make repeatedly with several actions available, and you do not know in advance which is best. After each decision, you observe the payoff for the action you took.

A natural question: how should you adapt your action choices over time?

**Regret matching** is one beautiful answer. For each action, you maintain a running tally of the **counterfactual regret**: the difference between what you would have earned from playing that action consistently and what you actually earned with your chosen strategy.

If an action has high accumulated regret (you would have done much better by playing it), increase its probability. If an action has low or negative regret, decrease its probability.

Specifically, the regret for action a at iteration t is:

\\[ R^t(a) = \sum_{\tau=1}^{t} \left( u^\tau(a) - u^\tau(\sigma^\tau) \right) \\]

**Decoding:**
- \\(R^t(a)\\): cumulative regret for action a after t iterations
- \\(u^\tau(a)\\): the payoff that would have been received from playing action a at iteration τ
- \\(u^\tau(\sigma^\tau)\\): the payoff actually received from playing the chosen strategy

Each iteration, you update your strategy using **regret matching**:

\\[ \sigma^{t+1}(a) = \frac{\max(0, R^t(a))}{\sum_{a'} \max(0, R^t(a'))} \\]

In English: the probability of action a is proportional to its positive regret, and zero if its regret is negative or zero. If all regrets are negative, the strategy is uniform.

The remarkable mathematical fact is that regret matching, when applied to all players in a game, converges to a Nash equilibrium of the game. (Specifically, the time-averaged strategies converge.)

## Counterfactual regret in extensive-form games

For extensive-form games, regret is computed per information set. The regret for an action at an information set is, intuitively, "how much extra payoff would I have gotten if I had taken this action at this information set, all else equal."

The technical definition uses counterfactual values:

\\[ v_i^\sigma(I) = \sum_{h \in I} \pi_{-i}^\sigma(h) \cdot v_i^\sigma(h) \\]

**Decoding:**
- \\(v_i^\sigma(I)\\): counterfactual value of information set I for player i
- \\(h \in I\\): histories (nodes) in the information set
- \\(\pi_{-i}^\sigma(h)\\): counterfactual reach probability (everyone except player i played to reach h)
- \\(v_i^\sigma(h)\\): expected utility from history h under strategy σ

This is "the expected payoff from this information set, weighted by how often we get to it through other players' play."

The counterfactual value of taking specific action a at information set I:

\\[ v_i^\sigma(I, a) = \sum_{h \in I} \pi_{-i}^\sigma(h) \cdot v_i^{\sigma_{I \to a}}(h \cdot a) \\]

This is the value if we always took action a at I instead of using our current strategy.

The counterfactual regret of action a at information set I:

\\[ r_i^t(I, a) = v_i^\sigma(I, a) - v_i^\sigma(I) \\]

In English: how much more value would I have gotten by always playing a at I, compared to playing my current strategy?

## The CFR algorithm

Vanilla CFR is the following loop:

```
Initialize all regrets to 0
Initialize strategies to uniform (probability 1/k for each of k actions)

Repeat for many iterations:
    For each player i:
        Traverse the game tree
        At each information set I belonging to player i:
            Compute counterfactual regret r(I, a) for each action a
            Add r(I, a) to the cumulative regret R(I, a)
        Update player i's strategy at each I using regret matching:
            σ(I, a) = max(0, R(I, a)) / sum of max(0, R(I, a'))
            (or uniform if all regrets non-positive)
        Update player i's average strategy:
            average_strategy(I, a) accumulates the strategy probabilities

After T iterations:
    Return the average strategy as the Nash equilibrium approximation
```

A few important details:

**Two strategies are tracked**: the current strategy σ (which is updated each iteration based on regrets) and the average strategy σ̄ (which accumulates σ over iterations). It is the **average strategy** that converges to Nash equilibrium, not the current strategy.

**Regrets accumulate**: do not reset them between iterations. The total accumulated regret over all iterations drives convergence.

**Both players update simultaneously**: at each iteration, you update Player 1's strategy AND Player 2's strategy. This is parallel best-response in disguise.

## A worked example by hand: Kuhn poker

Kuhn poker is a tiny imperfect-information game often used to illustrate CFR. Let us use a simplified version of our SSA conjunction game instead.

**The game**: Alice has a private mission state (high or low), assigned 50/50. Alice decides to maneuver (M) or hold (H). Then Bob (who only sees Alice's action, not her mission) decides M or H.

**Payoffs (cost form, lower is worse)**:

| Alice's mission | Alice's action | Bob's action | Alice payoff | Bob payoff |
|----------------|----------------|--------------|--------------|------------|
| High           | M              | M            | -2           | -1         |
| High           | M              | H            | -2           | -3         |
| High           | H              | M            | -3           | -1         |
| High           | H              | H            | -10          | -10        |
| Low            | M              | M            | -1           | -1         |
| Low            | M              | H            | -1           | -3         |
| Low            | H              | M            | -3           | -1         |
| Low            | H              | H            | -10          | -10        |

This game has:
- 2 information sets for Alice (one for each mission state)
- 2 information sets for Bob (one for each Alice action)
- 16 terminal nodes (we already enumerated payoffs above)

**Initial strategies**: uniform. Alice plays M with prob 0.5, H with prob 0.5 in each information set. Bob does the same.

**Initial regrets**: all zero.

**Iteration 1**: We compute counterfactual regrets for Alice. For Alice's "High" information set:

Probability of reaching it (counterfactual, i.e., chance only): 0.5.

Expected payoff under current strategy: average over Alice's actions and Bob's responses.
- (M, M): -2, prob = 0.5 × 0.5 × 0.5 = 0.125 (chance × Alice × Bob)
- (M, H): -2, prob = 0.5 × 0.5 × 0.5 = 0.125
- (H, M): -3, prob = 0.5 × 0.5 × 0.5 = 0.125
- (H, H): -10, prob = 0.5 × 0.5 × 0.5 = 0.125
- Wait, this isn't quite right. Let me redo with cleaner accounting.

Actually, for Alice's "High" information set (counterfactual reach 0.5 just from chance), Alice's choice of action affects the rest. Counterfactual value of action M for Alice at this info set:
- If Alice plays M for sure: outcome distribution depends on Bob (uniform).
- 0.5 × (-2) + 0.5 × (-2) = -2

Counterfactual value of action H for Alice:
- 0.5 × (-3) + 0.5 × (-10) = -6.5

Current strategy plays M with 0.5 and H with 0.5, so current strategy value is:
- 0.5 × (-2) + 0.5 × (-6.5) = -4.25

Counterfactual regrets (using counterfactual reach of 0.5):
- r(M) = 0.5 × ((-2) - (-4.25)) = 0.5 × 2.25 = 1.125
- r(H) = 0.5 × ((-6.5) - (-4.25)) = 0.5 × (-2.25) = -1.125

Updated cumulative regret for Alice's High info set: R(M) = 1.125, R(H) = -1.125.

Updated strategy: probability of M = 1.125 / 1.125 = 1.0 (since H's regret is negative, it is clipped to 0 in regret matching). So at iteration 2, Alice plays M with probability 1.0 in the High information set.

This makes sense: maneuvering is better than holding when you are high-priority (the alternative is risking a -10 collision).

We would similarly compute regrets for Alice's Low info set, Bob's "Alice maneuvered" info set, and Bob's "Alice held" info set, then move to iteration 2.

After many iterations, the cumulative regrets stabilize and the average strategy converges to a Nash equilibrium.

## Why this converges

The mathematical guarantee: regret matching produces strategies whose average regret converges to zero. By a theorem from online learning, the time-averaged strategy profile is then an ε-Nash equilibrium with ε that goes to zero as the number of iterations grows.

The convergence rate is \\(O(1/\sqrt{T})\\) where T is the number of iterations. Like Monte Carlo, you need 4× more iterations to halve the error. Like Monte Carlo, this scaling is why naive CFR is impractical for large games.

## The complete vanilla CFR implementation

```python
import numpy as np
from collections import defaultdict

class CFRSolver:
    def __init__(self, game):
        self.game = game
        self.regrets       = defaultdict(lambda: np.zeros(game.num_actions()))
        self.strategy_sum  = defaultdict(lambda: np.zeros(game.num_actions()))
    
    def get_strategy(self, info_set, num_actions):
        """Compute current strategy via regret matching."""
        regrets = self.regrets[info_set]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            strategy = positive / total
        else:
            strategy = np.ones(num_actions) / num_actions  # uniform
        return strategy
    
    def cfr(self, history, reach_probs):
        """
        Recursive CFR traversal.
        history: current game history (e.g., a state object)
        reach_probs: list of reach probabilities, one per player + chance
        Returns: utility for each player at this node
        """
        if history.is_terminal():
            return history.returns()  # array of payoffs, one per player
        
        if history.is_chance_node():
            # Sum over chance outcomes weighted by their probabilities
            outcomes = history.chance_outcomes()
            value = np.zeros(self.game.num_players())
            for action, prob in outcomes:
                next_history = history.apply(action)
                new_reach = reach_probs.copy()
                new_reach[-1] *= prob  # chance reach
                value += prob * self.cfr(next_history, new_reach)
            return value
        
        player = history.current_player()
        info_set = history.info_set()
        legal = history.legal_actions()
        
        strategy = self.get_strategy(info_set, len(legal))
        
        # Recursively get values for each action
        action_values = []
        for i, action in enumerate(legal):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_values.append(self.cfr(history.apply(action), new_reach))
        
        # Compute expected value of current strategy
        node_value = sum(strategy[i] * action_values[i] for i in range(len(legal)))
        
        # Compute regrets for each action
        # cf_reach: product of reach probabilities EXCLUDING player's
        cf_reach = np.prod([reach_probs[p] for p in range(self.game.num_players() + 1) if p != player])
        for i in range(len(legal)):
            regret = action_values[i][player] - node_value[player]
            self.regrets[info_set][i] += cf_reach * regret
            self.strategy_sum[info_set][i] += reach_probs[player] * strategy[i]
        
        return node_value
    
    def get_average_strategy(self, info_set):
        """Get the time-averaged strategy at an information set."""
        s = self.strategy_sum[info_set]
        total = s.sum()
        if total > 0:
            return s / total
        return np.ones(len(s)) / len(s)
    
    def run(self, iterations=10000):
        for it in range(iterations):
            initial_state = self.game.new_initial_state()
            initial_reach = np.ones(self.game.num_players() + 1)  # players + chance
            self.cfr(initial_state, initial_reach)
            
            if (it + 1) % 1000 == 0:
                print(f"Iteration {it + 1}/{iterations}")
        
        # Return average strategy over all information sets
        return {info_set: self.get_average_strategy(info_set) 
                for info_set in self.strategy_sum}
```

This is the complete vanilla CFR. It is short, but it is also slow: each iteration traverses the entire game tree. For a game with 10^9 nodes, one iteration might take hours.

## Limitations of vanilla CFR

**Tree traversal cost**: each iteration visits every node in the game tree. For poker (~10^14 information sets in no-limit Hold'em), this is hopeless. Even for medium games, vanilla CFR is too slow.

**Memory cost**: regrets and strategy sums must be stored for every information set. For huge games, this is too much memory.

The next lesson (MCCFR) fixes the speed problem by sampling. The lesson after that (deep CFR) fixes the memory problem by using a neural network to approximate the regret table.

## Regret decomposition: immediate regret

### Breaking total regret into per-decision-point components

One of CFR's key mathematical insights is that the total regret of a player can be **decomposed** into contributions from individual information sets. This decomposition is what makes CFR tractable.

Define the **immediate counterfactual regret** at iteration T as:

\\[ R_i^{T,\text{imm}}(I, a) = \frac{1}{T} \sum_{t=1}^{T} \pi_{-i}^{\sigma^t}(I) \left( v_i^{\sigma^t}(I, a) - v_i^{\sigma^t}(I) \right) \\]

**Decoding:**
- \\(R_i^{T,\text{imm}}(I, a)\\): the average instantaneous counterfactual regret for action \\(a\\) at information set \\(I\\) over \\(T\\) iterations
- \\(\pi_{-i}^{\sigma^t}(I)\\): at iteration \\(t\\), how often did the opponents' and chance's play lead to information set \\(I\\)?
- \\(v_i^{\sigma^t}(I, a) - v_i^{\sigma^t}(I)\\): at iteration \\(t\\), how much would player \\(i\\) have gained by always playing \\(a\\) at \\(I\\) compared to their actual strategy?

### Why this decomposition makes CFR tractable

The full regret of a player over a game is defined as the maximum gain they could have achieved over a sequence of iterations by committing to some fixed strategy profile \\(\sigma_i^*\\). This quantity is exponentially complex to compute directly: you would need to compare the actual play against all possible strategies simultaneously.

The decomposition theorem (Theorem 3 in Zinkevich et al. 2007) states:

\\[ R_i^T \leq \sum_{I \in \mathcal{I}_i} R_i^{T,+,\text{imm}}(I) \\]

where \\(R_i^{T,+,\text{imm}}(I) = \max_{a \in A(I)} R_i^{T,\text{imm}}(I, a)\\) is the positive part of the immediate regret at \\(I\\).

**In plain English:** the player's total regret is upper bounded by the sum of the per-information-set immediate regrets. This means:
- You never need to compare against all possible strategies globally.
- You only need to minimize regret at each information set locally.
- The local updates (regret matching at each information set) combine to control the global regret.

This is the mathematical heart of why CFR works: a problem that appears exponentially complex decomposes into a sum of polynomial-complexity subproblems. Each subproblem is a simple regret-matching update at one information set.

In our Alice–Bob conjunction game, Alice's total regret is bounded by the sum of four immediate regret terms (one per action at each of her two information sets). CFR drives each local term to zero by regret matching, which collectively drives Alice's global regret to zero.

## Convergence rate analysis

### The O(T^{-1/2}) bound

The convergence rate of CFR is:

\\[ \epsilon(T) = \frac{C}{\sqrt{T}} \\]

where \\(\epsilon(T)\\) is the **exploitability** of the average strategy after \\(T\\) iterations — the maximum gain any player could achieve by unilaterally deviating — and \\(C\\) is a constant that depends on the game structure.

**Decoding:**
- \\(T\\): number of CFR iterations
- \\(\epsilon(T)\\): how "far" the average strategy is from a Nash equilibrium; a Nash equilibrium has \\(\epsilon = 0\\)
- \\(C\\): roughly \\(\Delta \sqrt{|I_{\max}| \cdot |A_{\max}|}\\), where \\(\Delta\\) is the maximum payoff range, \\(|I_{\max}|\\) is the number of information sets, and \\(|A_{\max}|\\) is the maximum number of actions at any information set

### What ε means in practice

An ε-Nash equilibrium means no player can gain more than ε by deviating from the average strategy. In the SSA conjunction game where payoffs range from -10 to 0:
- ε = 1.0 means a player could gain at most 1 utility unit by deviating (out of a 10-unit payoff range). This is 10% exploitability.
- ε = 0.1 means 1% exploitability.

For practical SSA applications, ε around 0.01 to 0.05 is usually sufficient. For high-stakes domains (e.g., adversarial satellite-vs-jammer spectrum games), tighter convergence may be needed.

### How many iterations for 1% exploitability

From \\(\epsilon = C/\sqrt{T}\\), solving for T:

\\[ T = \left(\frac{C}{\epsilon}\right)^2 \\]

If \\(C = 1.0\\) (typical for small normalized games) and \\(\epsilon = 0.01\\):

\\[ T = (1.0 / 0.01)^2 = 10{,}000 \text{ iterations} \\]

If \\(C = 5.0\\) (larger games with wider payoff ranges):

\\[ T = (5.0 / 0.01)^2 = 250{,}000 \text{ iterations} \\]

### Comparison to gradient descent

Gradient descent on the expected utility (as in policy gradient from Module 3) converges at rate \\(O(1/T)\\) in convex settings. CFR's \\(O(1/\sqrt{T})\\) is slower. Why use CFR at all?

The crucial difference is the game-theoretic setting. Gradient ascent on expected utility is not a stable algorithm for multi-agent zero-sum games: the two players' gradients point in opposing directions. In practice, gradient ascent in zero-sum games **cycles** rather than converges. CFR's regret-matching update is specifically designed to handle this cycling and has provable convergence guarantees that gradient-based methods lack.

Required iterations at ε = 0.01 (from \\(T = (C/\epsilon)^2\\)):

| Game | C | Iterations |
|------|---|------------|
| Small SSA conjunction (2 players, 2 info sets each) | 1.0 | 10,000 |
| Medium ISR allocation game (many operators) | 5.0 | 250,000 |
| Large spectrum deconfliction (16 frequency bands) | 20.0 | 4,000,000 |

## Regret matching vs. regret matching+

### How RM+ floors regret at zero

Standard regret matching (RM) accumulates all regrets, including negative ones:

\\[ R^{T+1}(a) = R^T(a) + r^T(a) \\]

where \\(r^T(a)\\) is the instantaneous regret at iteration \\(T\\).

**Regret matching+** (RM+, introduced in CFR+) floors regret at zero after each update:

\\[ R^{T+1}(a) = \max\left(0, R^T(a) + r^T(a)\right) \\]

The strategy update is otherwise identical: \\(\sigma^{T+1}(a) \propto \max(0, R^T(a))\\).

### Why this speeds convergence empirically

The intuition: in standard RM, an action that was bad long ago can accumulate large negative regret. When the game dynamics change (because both players are adapting), that action might become good, but the large negative regret prevents it from being played until many iterations of positive regret cancel it out.

RM+ "forgets" negative regret by flooring at zero, making the strategy more responsive to recent game dynamics. In practice, CFR+ (which uses RM+) converges 10× to 100× faster than vanilla CFR on most games, while maintaining the same theoretical convergence guarantees.

```python
import numpy as np

def run_regret_matching(payoff_matrix, T, use_rm_plus=False):
    """RM or RM+ on a 2-player zero-sum game. Returns (avg_strategy, exploitabilities)."""
    n = payoff_matrix.shape[0]
    R1, R2 = np.zeros(n), np.zeros(n)
    S1, S2 = np.zeros(n), np.zeros(n)
    exploitabilities = []

    def rm(R):
        pos = np.maximum(R, 0)
        s = pos.sum()
        return pos / s if s > 0 else np.ones(n) / n

    for _ in range(T):
        s1, s2 = rm(R1), rm(R2)
        S1 += s1; S2 += s2
        ev = s1 @ payoff_matrix @ s2
        for a in range(n):
            dr1 = payoff_matrix[a, :] @ s2 - ev
            dr2 = -s1 @ payoff_matrix[:, a] + ev
            R1[a] = max(0.0, R1[a] + dr1) if use_rm_plus else R1[a] + dr1
            R2[a] = max(0.0, R2[a] + dr2) if use_rm_plus else R2[a] + dr2
        avg1, avg2 = S1 / S1.sum(), S2 / S2.sum()
        exploitabilities.append(np.max(payoff_matrix @ avg2) + np.max(-avg1 @ payoff_matrix))

    return S1 / S1.sum(), exploitabilities

# 3x3 satellite-frequency-vs-jammer game
freq_game = np.array([[-1,1,1],[1,-1,1],[1,1,-1]])
T = 10_000
_, exploit_rm   = run_regret_matching(freq_game, T, use_rm_plus=False)
_, exploit_rmp  = run_regret_matching(freq_game, T, use_rm_plus=True)

threshold = 0.01
rm_iters  = next((i for i, e in enumerate(exploit_rm)  if e < threshold), T)
rmp_iters = next((i for i, e in enumerate(exploit_rmp) if e < threshold), T)
print(f"Iterations to epsilon<0.01: RM={rm_iters:,}  RM+={rmp_iters:,}")
# Typically: RM+ reaches threshold ~10x faster than standard RM
```

In practice on this 3×3 spectrum game, RM+ typically reaches ε < 0.01 in roughly 1/10th the iterations of standard RM, because it does not need to "unlearn" the accumulated negative regret from early suboptimal rounds.

## Why CFR finds Nash, not just best response

### The self-play argument

A naive approach to finding good strategies in a two-player game is **gradient ascent on expected utility**: each player independently maximizes their own expected payoff gradient. This converges to a Nash equilibrium in some games but notoriously cycles or diverges in zero-sum games.

CFR takes a different approach grounded in the theory of no-regret learning. The key theorem:

> If both players use regret-minimizing algorithms (algorithms whose average regret goes to zero), then the joint average strategy profile converges to a Nash equilibrium.

This is the fundamental theorem connecting online learning to game theory (Theorem 2 in Zinkevich et al. 2007).

**Decoding the self-play argument:**
- Each player is minimizing their own average regret independently.
- Player 1's regret-minimizing algorithm guarantees: \\(\bar{R}_1^T / T \to 0\\) as \\(T \to \infty\\).
- Player 2's regret-minimizing algorithm guarantees: \\(\bar{R}_2^T / T \to 0\\) as \\(T \to \infty\\).
- The Nash gap (exploitability) is bounded: \\(\epsilon^T \leq (\bar{R}_1^T + \bar{R}_2^T) / T\\).
- Since both average regrets go to zero, the exploitability goes to zero.

### Why this differs from gradient ascent on expected utility

In gradient ascent, Player 1 updates: \\(\sigma_1^{t+1} = \sigma_1^t + \alpha \nabla_{\sigma_1} u_1(\sigma^t)\\).

The problem: when Player 1 improves their strategy, Player 2's best response changes. Player 2 then adapts, which changes Player 1's best response. In zero-sum games, this creates a feedback loop with no natural fixed point: the gradient updates drive the strategies around in a cycle.

CFR's regret matching does not follow the gradient of the current expected utility. Instead, it tracks the accumulated difference between what each action would have yielded historically and what was actually played. This historical averaging is what breaks the cycling: the average strategy converges even as the current strategy oscillates.

Analogy: in the SSA hide-and-seek game (satellite chooses frequency, jammer chooses which frequency to block), gradient ascent oscillates (satellite follows jammer to frequency X, jammer moves to X+1, satellite follows, ...). CFR builds a historical average that smooths out these oscillations, converging to the uniform random strategy where neither player can exploit the other.

If you run both algorithms on Rock-Paper-Scissors for 5000 iterations, gradient ascent's exploitability stays near 0.33 (close to worst-case, cycling perpetually), while CFR's average-strategy exploitability falls below 0.01. This is why CFR, not gradient ascent, is the standard algorithm for computing Nash equilibria in imperfect-information extensive-form games.

## Key Takeaways

- CFR decomposes the problem of minimizing a player's total game regret into a sum of **immediate counterfactual regrets** at individual information sets; this decomposition makes Nash equilibrium computation tractable via local regret-matching updates.
- The convergence rate is \\(O(T^{-1/2})\\): each doubling of iterations halves the exploitability, similar to Monte Carlo integration; for 1% exploitability in a medium game, expect 250,000+ iterations.
- **Regret matching+** (RM+) floors accumulated regrets at zero after each update, preventing old negative regrets from slowing adaptation; in practice this yields 10× to 100× faster convergence than standard regret matching.
- CFR finds Nash equilibrium through the self-play argument: when both players independently minimize their average regret, the joint average strategy profile converges to Nash — this is fundamentally different from gradient ascent, which cycles in zero-sum games.
- The key data structures are the **cumulative regret table** and the **strategy sum table**, both indexed by information set; the regret table drives the current strategy, and the strategy sum's average is the Nash approximation returned at the end.
- Vanilla CFR's memory and time costs grow linearly with the number of information sets; for games with \\(10^{14}\\) information sets (no-limit poker), variants like MCCFR (Lesson 4) and Deep CFR (Lesson 5) are required.

## Quiz

{{#quiz 03-counterfactual-regret-minimization.toml}}
