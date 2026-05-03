# Module 5 Project: A CFR Solver for an SSA Negotiation Game

## What you are building

You will implement vanilla CFR for a small extensive-form imperfect-information game and use it to compute Nash equilibrium strategies for a satellite conjunction negotiation scenario. Optionally, you will also implement MCCFR and compare convergence rates. This is the Rust-most-relevant project in the curriculum: the data structures (information set table, regret vector) are simple and translate cleanly to Rust, and CFR is what your capstone (Module 8) will revolve around.

## The game

Two satellite operators, Alice and Bob, share a region of space. A potential conjunction has been detected. Each operator privately knows their own satellite's "operational priority" (high or low), assigned independently by chance with 50/50 probability. The operators have not communicated their priorities to each other.

The game proceeds:
1. Chance assigns Alice's priority and Bob's priority (independently, 50/50)
2. Alice (who only knows her own priority) decides M (maneuver) or H (hold)
3. Bob (who knows his own priority and observed Alice's action) decides M or H

**Cost structure** (cost is negative, so larger negative = worse):

The cost depends on the joint action and the operators' priorities. Maneuvering costs more for high-priority operators (interrupting their mission); holding when both hold causes a collision.

| Alice priority | Bob priority | A action | B action | Alice cost | Bob cost |
|----------------|--------------|----------|----------|------------|----------|
| H | H | M | M | -3 | -3 |
| H | H | M | H | -3 | -1 |
| H | H | H | M | -1 | -3 |
| H | H | H | H | -10 | -10 |
| H | L | M | M | -3 | -1 |
| H | L | M | H | -3 | -1 |
| H | L | H | M | -1 | -1 |
| H | L | H | H | -10 | -10 |
| L | H | M | M | -1 | -3 |
| L | H | M | H | -1 | -1 |
| L | H | H | M | -1 | -3 |
| L | H | H | H | -10 | -10 |
| L | L | M | M | -1 | -1 |
| L | L | M | H | -1 | -1 |
| L | L | H | M | -1 | -1 |
| L | L | H | H | -10 | -10 |

This is the cost from each operator's perspective. We will use these as **utilities** (so the values are negative; Nash equilibria will minimize cost, equivalently maximize utility).

**Information sets**:
- Alice's: 2 information sets (one per priority she observed)
- Bob's: 4 information sets (Bob's priority × Alice's observed action)

Alice's strategy is two probability distributions (over {M, H}). Bob's strategy is four probability distributions. Total strategy parameters: 2 + 4 = 6 free probabilities (one per info set; the other action's probability is 1 minus this).

## Step 1: define the game in Python

You can do this either as an OpenSpiel game (consistent with previous modules) or as a custom Python class (simpler and faster to iterate on). For learning CFR, the custom class is recommended:

```python
"""
conjunction_game.py: a small extensive-form game for CFR.
"""

import numpy as np
from copy import deepcopy

ALICE, BOB = 0, 1
HIGH, LOW = 0, 1
MANEUVER, HOLD = 0, 1
NUM_PRIORITIES = 2
NUM_ACTIONS = 2  # M or H

# Cost table indexed by [alice_priority, bob_priority, alice_action, bob_action, player]
COSTS = np.zeros((2, 2, 2, 2, 2))
def set_cost(ap, bp, a, b, alice_c, bob_c):
    COSTS[ap, bp, a, b, ALICE] = alice_c
    COSTS[ap, bp, a, b, BOB]   = bob_c

# H, H
set_cost(HIGH, HIGH, MANEUVER, MANEUVER, -3, -3)
set_cost(HIGH, HIGH, MANEUVER, HOLD,     -3, -1)
set_cost(HIGH, HIGH, HOLD,     MANEUVER, -1, -3)
set_cost(HIGH, HIGH, HOLD,     HOLD,     -10, -10)
# H, L
set_cost(HIGH, LOW, MANEUVER, MANEUVER, -3, -1)
set_cost(HIGH, LOW, MANEUVER, HOLD,     -3, -1)
set_cost(HIGH, LOW, HOLD,     MANEUVER, -1, -1)
set_cost(HIGH, LOW, HOLD,     HOLD,     -10, -10)
# L, H
set_cost(LOW, HIGH, MANEUVER, MANEUVER, -1, -3)
set_cost(LOW, HIGH, MANEUVER, HOLD,     -1, -1)
set_cost(LOW, HIGH, HOLD,     MANEUVER, -1, -3)
set_cost(LOW, HIGH, HOLD,     HOLD,     -10, -10)
# L, L
set_cost(LOW, LOW, MANEUVER, MANEUVER, -1, -1)
set_cost(LOW, LOW, MANEUVER, HOLD,     -1, -1)
set_cost(LOW, LOW, HOLD,     MANEUVER, -1, -1)
set_cost(LOW, LOW, HOLD,     HOLD,     -10, -10)


class ConjunctionGame:
    """
    Game state:
        - alice_priority: HIGH or LOW (chance-assigned)
        - bob_priority: HIGH or LOW (chance-assigned)
        - alice_action: None, MANEUVER, or HOLD
        - bob_action: None, MANEUVER, or HOLD
    """
    def __init__(self):
        self.alice_priority = None
        self.bob_priority = None
        self.alice_action = None
        self.bob_action = None
    
    def is_chance_node(self):
        return self.alice_priority is None or self.bob_priority is None
    
    def is_terminal(self):
        return self.alice_action is not None and self.bob_action is not None
    
    def current_player(self):
        if self.alice_priority is None or self.bob_priority is None:
            return -1  # chance
        if self.alice_action is None:
            return ALICE
        if self.bob_action is None:
            return BOB
        return -2  # terminal
    
    def chance_outcomes(self):
        """Return list of (action, probability) tuples for the next chance event."""
        if self.alice_priority is None:
            return [(HIGH, 0.5), (LOW, 0.5)]
        if self.bob_priority is None:
            return [(HIGH, 0.5), (LOW, 0.5)]
        return []
    
    def legal_actions(self):
        if self.is_chance_node() or self.is_terminal():
            return []
        return [MANEUVER, HOLD]
    
    def info_set(self):
        """Information set encoding for the current player."""
        player = self.current_player()
        if player == ALICE:
            # Alice knows her priority but not Bob's
            return f"alice_p{self.alice_priority}"
        elif player == BOB:
            # Bob knows his priority and Alice's action
            return f"bob_p{self.bob_priority}_a{self.alice_action}"
        return None
    
    def apply(self, action):
        """Return a new state with the action applied."""
        new_state = ConjunctionGame()
        new_state.alice_priority = self.alice_priority
        new_state.bob_priority = self.bob_priority
        new_state.alice_action = self.alice_action
        new_state.bob_action = self.bob_action
        
        if self.alice_priority is None:
            new_state.alice_priority = action
        elif self.bob_priority is None:
            new_state.bob_priority = action
        elif self.alice_action is None:
            new_state.alice_action = action
        else:
            new_state.bob_action = action
        return new_state
    
    def returns(self):
        if not self.is_terminal():
            return [0.0, 0.0]
        return [
            COSTS[self.alice_priority, self.bob_priority, 
                  self.alice_action, self.bob_action, ALICE],
            COSTS[self.alice_priority, self.bob_priority, 
                  self.alice_action, self.bob_action, BOB],
        ]
```

## Step 2: implement vanilla CFR

This is the algorithm from lesson 3, specialized to our two-player game:

```python
import numpy as np
from collections import defaultdict

class CFRSolver:
    def __init__(self):
        self.regrets = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    
    def get_strategy(self, info_set):
        """Compute current strategy via regret matching."""
        regrets = self.regrets[info_set]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS
    
    def get_average_strategy(self, info_set):
        """Compute time-averaged strategy."""
        s = self.strategy_sum[info_set]
        total = s.sum()
        if total > 0:
            return s / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS
    
    def cfr(self, state, reach_alice, reach_bob, reach_chance):
        """
        Recursive CFR.
        Returns: array of expected utilities, one per player.
        """
        if state.is_terminal():
            return np.array(state.returns())
        
        if state.is_chance_node():
            value = np.zeros(2)
            for action, prob in state.chance_outcomes():
                next_state = state.apply(action)
                value += prob * self.cfr(next_state, reach_alice, reach_bob, reach_chance * prob)
            return value
        
        player = state.current_player()
        info_set = state.info_set()
        strategy = self.get_strategy(info_set)
        
        # Compute action values
        action_values = []
        for i, action in enumerate(state.legal_actions()):
            next_state = state.apply(action)
            if player == ALICE:
                v = self.cfr(next_state, reach_alice * strategy[i], reach_bob, reach_chance)
            else:  # BOB
                v = self.cfr(next_state, reach_alice, reach_bob * strategy[i], reach_chance)
            action_values.append(v)
        
        # Expected value at this node
        node_value = sum(strategy[i] * action_values[i] for i in range(NUM_ACTIONS))
        
        # Update regrets for this player
        cf_reach = reach_bob * reach_chance if player == ALICE else reach_alice * reach_chance
        own_reach = reach_alice if player == ALICE else reach_bob
        for i in range(NUM_ACTIONS):
            regret = action_values[i][player] - node_value[player]
            self.regrets[info_set][i] += cf_reach * regret
            self.strategy_sum[info_set][i] += own_reach * strategy[i]
        
        return node_value
    
    def run(self, iterations=10000, verbose=True):
        for it in range(iterations):
            initial = ConjunctionGame()
            self.cfr(initial, 1.0, 1.0, 1.0)
            
            if verbose and (it + 1) % 1000 == 0:
                print(f"Iteration {it + 1}/{iterations}")
        
        # Return averaged strategies
        return {info_set: self.get_average_strategy(info_set) 
                for info_set in self.strategy_sum}
```

## Step 3: run and analyze

```python
solver = CFRSolver()
strategies = solver.run(iterations=20000)

print("\n=== Nash equilibrium strategies ===\n")

# Alice's strategies
print("Alice (when high priority):")
s = strategies['alice_p0']  # priority 0 = HIGH
print(f"  Maneuver: {s[MANEUVER]:.3f}, Hold: {s[HOLD]:.3f}")

print("\nAlice (when low priority):")
s = strategies['alice_p1']
print(f"  Maneuver: {s[MANEUVER]:.3f}, Hold: {s[HOLD]:.3f}")

# Bob's strategies
print("\nBob (when high priority, Alice maneuvered):")
s = strategies['bob_p0_a0']
print(f"  Maneuver: {s[MANEUVER]:.3f}, Hold: {s[HOLD]:.3f}")

print("\nBob (when high priority, Alice held):")
s = strategies['bob_p0_a1']
print(f"  Maneuver: {s[MANEUVER]:.3f}, Hold: {s[HOLD]:.3f}")

print("\nBob (when low priority, Alice maneuvered):")
s = strategies['bob_p1_a0']
print(f"  Maneuver: {s[MANEUVER]:.3f}, Hold: {s[HOLD]:.3f}")

print("\nBob (when low priority, Alice held):")
s = strategies['bob_p1_a1']
print(f"  Maneuver: {s[MANEUVER]:.3f}, Hold: {s[HOLD]:.3f}")
```

You should see (approximately):
- Alice (high priority): plays a mixed strategy or mostly holds, depending on costs
- Alice (low priority): mostly maneuvers
- Bob (Alice maneuvered): mostly holds (no need to also maneuver)
- Bob (Alice held): always maneuvers (must avoid collision)

The exact mixing probabilities depend on the cost structure. The interesting equilibrium behavior emerges from the imperfect information: Alice has to commit to an action without knowing Bob's priority, so she balances her own cost against the risk of forcing Bob into a bad position.

## Step 4: verify Nash equilibrium

To check that your computed strategies are actually a Nash equilibrium, compute the best response for each player to the other's strategy and verify that the player's current strategy is approximately a best response.

```python
def compute_best_response_value(solver, strategies, deviating_player):
    """
    Compute the maximum utility the deviating player could achieve
    by playing any strategy against the others' fixed strategies.
    """
    def br_value(state, prob):
        if state.is_terminal():
            return state.returns()[deviating_player]
        
        if state.is_chance_node():
            value = 0
            for action, p in state.chance_outcomes():
                value += p * br_value(state.apply(action), prob * p)
            return value
        
        player = state.current_player()
        info_set = state.info_set()
        legal = state.legal_actions()
        
        if player == deviating_player:
            # Pick the best action
            best = float('-inf')
            for action in legal:
                v = br_value(state.apply(action), prob)
                best = max(best, v)
            return best
        else:
            # Use the fixed strategy
            strat = strategies.get(info_set, np.ones(NUM_ACTIONS) / NUM_ACTIONS)
            value = 0
            for i, action in enumerate(legal):
                value += strat[i] * br_value(state.apply(action), prob * strat[i])
            return value
    
    return br_value(ConjunctionGame(), 1.0)


def compute_strategy_value(strategies):
    """Compute the actual expected utilities under the strategy profile."""
    def value(state, prob):
        if state.is_terminal():
            return np.array(state.returns())
        
        if state.is_chance_node():
            v = np.zeros(2)
            for action, p in state.chance_outcomes():
                v += p * value(state.apply(action), prob * p)
            return v
        
        info_set = state.info_set()
        strat = strategies.get(info_set, np.ones(NUM_ACTIONS) / NUM_ACTIONS)
        v = np.zeros(2)
        for i, action in enumerate(state.legal_actions()):
            v += strat[i] * value(state.apply(action), prob * strat[i])
        return v
    
    return value(ConjunctionGame(), 1.0)


actual_values = compute_strategy_value(strategies)
print(f"\nActual Nash strategy values: Alice = {actual_values[ALICE]:.4f}, "
      f"Bob = {actual_values[BOB]:.4f}")

alice_br = compute_best_response_value(solver, strategies, ALICE)
bob_br = compute_best_response_value(solver, strategies, BOB)

print(f"Best-response values:        Alice = {alice_br:.4f}, Bob = {bob_br:.4f}")
print(f"Exploitability (Alice):       {alice_br - actual_values[ALICE]:.6f}")
print(f"Exploitability (Bob):         {bob_br - actual_values[BOB]:.6f}")
```

The **exploitability** of a strategy profile is how much each player could gain by best-responding. A perfect Nash equilibrium has exploitability 0 for both players. CFR converges toward this asymptotically.

After 20,000 iterations, exploitability should be small (less than 0.05 or so).

## Step 5 (optional): implement MCCFR

If you want to compare convergence rates, implement outcome-sampling MCCFR following lesson 4. For this small game, vanilla CFR converges very fast and there is no need for sampling. But the implementation experience is valuable preparation for the capstone (which uses sampled methods).

## Step 6 (optional): Rust translation

This is where the curriculum's Rust focus pays off. CFR's data structures are simple:
- `regrets: HashMap<String, [f64; NUM_ACTIONS]>`
- `strategy_sum: HashMap<String, [f64; NUM_ACTIONS]>`

The recursive CFR function translates straightforwardly to Rust. Key challenges:
- Use `HashMap<String, _>` or interned strings for info sets
- Be careful about borrowing when recursing (use `Rc<RefCell<...>>` or pass immutable references and return updates)
- For large games, switch to a struct-of-arrays layout for cache efficiency

A Rust implementation of this game and CFR would be roughly 200-300 lines. Try it; the capstone in Module 8 builds on this directly.

## Step 7: reflect

1. What does the Nash equilibrium tell you about strategic behavior in conjunction-avoidance scenarios? Does it match your intuition about how operators should behave?
2. Modify the cost structure (make holding cheaper, or maneuvering more expensive). How does the equilibrium change?
3. What if you removed the imperfect information (Alice could see Bob's priority before deciding)? Does the equilibrium change? Why?
4. The exploitability after 20,000 iterations should be small but nonzero. How many iterations would you need to get exploitability below 0.001?
5. (Bonus) What would change in your CFR implementation if Alice and Bob each had 3 actions instead of 2 (e.g., maneuver-up, maneuver-down, hold)?

## What you have built

- A complete extensive-form game implementation in Python
- A working vanilla CFR solver
- A way to verify Nash equilibrium via exploitability
- (Optionally) An MCCFR implementation
- (Optionally) A Rust translation of the same algorithm

Module 6 introduces multi-agent RL methods (PSRO, fictitious play, alpha-rank) that extend the equilibrium-finding ideas of CFR to settings where best-response computation is itself an RL problem.
