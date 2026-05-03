# Lesson 3: Counterfactual Regret Minimization (CFR)

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

## Quiz

{{#quiz 03-counterfactual-regret-minimization.toml}}
