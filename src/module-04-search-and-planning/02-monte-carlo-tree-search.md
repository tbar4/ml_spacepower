# Lesson 2: Monte Carlo Tree Search

## Where this fits

MCTS is the most important search algorithm in modern game AI. It powered AlphaGo (which beat Lee Sedol in 2016), AlphaZero (which mastered chess, shogi, and Go from self-play in 2017), and a long line of game-playing systems before and since. It is also used as the planning subroutine inside many RL systems and as the sampling method inside MCCFR (Module 5). Understanding MCTS gives you a versatile algorithm for any problem you can simulate forward.

The good news: MCTS is conceptually simpler than minimax once you internalize the four-phase loop. The trick is that it focuses computation on promising parts of the tree rather than exhaustively searching, which makes it scale to games where minimax cannot.

## The intuition

When facing a game position, you do not analyze every possible continuation exhaustively. You sample a few promising lines, see how they tend to turn out, focus more attention on the lines that look good, and after thinking for a while, pick the move that has produced the best results in your simulations.

That is essentially MCTS. The "Monte Carlo" part: estimate the value of moves by simulating forward to the end of the game and seeing what happens. The "Tree Search" part: use the simulation results to incrementally build a tree of statistics about which moves look promising.

## The four phases

MCTS proceeds in repeated iterations of four phases. Each iteration adds one new node to the tree and updates statistics on existing nodes.

### Phase 1: Selection

Starting from the root, traverse the tree by repeatedly selecting child nodes until you reach a node that has unexplored children. The selection rule must balance:
- **Exploitation**: visit children that have produced good results so far
- **Exploration**: visit children that have not been tried much yet

The standard selection rule is **UCB1** (Upper Confidence Bound), specifically the variant called **UCT** (UCB applied to Trees):

\\[ UCT(\text{child}) = \frac{W(\text{child})}{N(\text{child})} + c \cdot \sqrt{\frac{\ln N(\text{parent})}{N(\text{child})}} \\]

**Decoding:**
- \\(W(\text{child})\\): the total wins (or accumulated reward) from simulations that went through this child
- \\(N(\text{child})\\): the number of times this child has been visited
- \\(N(\text{parent})\\): the number of times the parent has been visited
- \\(c\\): an exploration constant (typically √2 ≈ 1.41 for binary win/loss games)
- \\(\ln\\): natural logarithm

The first term \\(W/N\\) is the average value (the win rate). The second term grows when the child has been visited rarely compared to its siblings, encouraging exploration of less-tried options.

At each step of selection, pick the child with the highest UCT value. This biases toward strong moves while still occasionally exploring weak-looking ones.

### Phase 2: Expansion

When you reach a node with unexplored children, add one of those children to the tree. Initialize its statistics: \\(N = 0\\), \\(W = 0\\).

### Phase 3: Simulation (also called "rollout" or "playout")

From the newly expanded node, play the game forward to a terminal state using a simple policy (often uniformly random move selection). Record the outcome.

The simulation does not add nodes to the tree; it just plays out one quick game to estimate the value of the new node.

### Phase 4: Backpropagation

Walk back up the tree from the new node to the root. Update the statistics on every node along the path:
- Increment \\(N\\) by 1
- Add the simulation outcome to \\(W\\)

This propagates the simulation result up through the path that led to it.

## A worked example

Consider a tiny 2-move game. After 1 MCTS iteration, the tree might look like:

```
Root: N=1, W=1
  └ Move A: N=1, W=1 (just expanded; rollout was a win)
```

After 2 iterations:

```
Root: N=2, W=1
  ├ Move A: N=1, W=1 (UCT selected this first iteration)
  └ Move B: N=1, W=0 (just expanded; rollout was a loss)
```

After 4 iterations:

```
Root: N=4, W=2
  ├ Move A: N=2, W=2 (UCT prefers it: 100% win rate)
  │   └ Move A.1: N=1, W=1 (deeper exploration after 3rd iteration)
  └ Move B: N=2, W=0
      └ Move B.1: N=1, W=0
```

After enough iterations, the tree's statistics converge: high-value moves have many visits and high win rates; low-value moves have low visits.

When it is time to actually play a move, you typically pick the move with the **most visits** (the most-explored move, which by UCT design should be the best one). Some implementations pick the highest average value, but visit count is more robust because high-visit nodes have more reliable statistics.

## A complete tabular MCTS implementation

```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action  # the action that led to this node from parent
        self.children = {}      # action -> MCTSNode
        self.N        = 0       # visit count
        self.W        = 0       # total reward
        self.untried_actions = list(state.legal_actions())
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_uct_child(self, c=1.41):
        """Pick child with highest UCT value."""
        best_action, best_score = None, float('-inf')
        for action, child in self.children.items():
            if child.N == 0:
                # Never visited; UCT score is infinite (always explore unvisited)
                return child
            exploit = child.W / child.N
            explore = c * math.sqrt(math.log(self.N) / child.N)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action
        return self.children[best_action]
    
    def expand(self):
        """Add a new child for one of the untried actions."""
        action = self.untried_actions.pop()
        next_state = self.state.apply(action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        return child


def random_rollout(state):
    """Play out the game with random moves until terminal. Return the outcome."""
    while not state.is_terminal():
        action = random.choice(list(state.legal_actions()))
        state = state.apply(action)
    return state.winner_reward()  # +1 if root player won, -1 if lost, 0 if draw


def mcts_search(root_state, num_iterations=1000):
    root = MCTSNode(root_state)
    
    for _ in range(num_iterations):
        # Phase 1: Selection
        node = root
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_uct_child()
        
        # Phase 2: Expansion
        if not node.state.is_terminal():
            node = node.expand()
        
        # Phase 3: Simulation
        outcome = random_rollout(node.state)
        
        # Phase 4: Backpropagation
        # Account for whose turn it was: outcome is from root player's perspective,
        # but each node along the path represents the perspective of the player to move.
        # In a 2-player zero-sum game, alternate the sign as you walk up.
        while node is not None:
            node.N += 1
            # If this node represents the same player as root, add outcome.
            # If opponent, subtract.
            node.W += outcome if node.state.player_to_move() == root_state.player_to_move() else -outcome
            node = node.parent
    
    # Pick the most-visited child of the root
    best_action = max(root.children, key=lambda a: root.children[a].N)
    return best_action, root  # return the tree too for inspection
```

The tricky bit is phase 4. In a two-player zero-sum game, the player at each node is either the "root player" or the "opponent." A win for one is a loss for the other. When backpropagating, the sign of the outcome flips at every level of the tree. If you forget this detail, MCTS can fail silently.

## What MCTS gives you

**Anytime behavior**: you can stop MCTS at any time and get a reasonable answer. More iterations → better answer. This is unlike minimax, which gives a definitive answer or no answer at all.

**Asymmetric exploration**: the tree grows deeper in promising directions and barely at all in unpromising ones. UCT automatically focuses computation where it matters most.

**No heuristic evaluation needed**: pure MCTS uses random rollouts to estimate value. No domain knowledge is required (though good heuristics can help).

**Convergence to optimal play**: with infinite iterations, MCTS converges to the minimax value. In practice, you stop when you run out of time, but more iterations always help.

## Limitations of pure MCTS

**Random rollouts can be poor estimators**: in games where most moves are bad and you need to play very specific sequences to win, random rollouts will almost never produce a meaningful win signal. The estimated values will be noisy and uninformative.

**No generalization across states**: a tabular MCTS treats each state independently. Two very similar positions get separate statistics; visiting one tells you nothing about the other.

**Cold start**: in any new position, MCTS has to start exploration from scratch. A trained network can immediately suggest good moves; pure MCTS has to discover them.

The next lesson fixes both problems by replacing the random rollout with a value network's prediction and biasing the selection phase with a policy network's recommendations. That is the AlphaGo Zero / AlphaZero architecture.

## UCB1 derivation intuition

**Module/Source:** Silver, D. et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search." *Nature* 529 (2016). Silver, D. et al. "A general reinforcement learning algorithm that masters chess, shogi and Go through self-play." *Science* 362 (2018).

### The multi-armed bandit origin

UCB1 was developed to solve the **multi-armed bandit problem**: you have k slot machines ("arms"), each with an unknown reward distribution. At each time step you pull one arm and observe a reward. Goal: maximize total reward over T pulls.

The tension is exploration vs. exploitation. Pull the arm that looks best so far (exploit) or try a less-tried arm that might actually be better (explore)?

**UCB1** (Auer et al., 2002) says: at time t, pull arm i that maximizes

\\[ \text{UCB1}(i) = \bar{X}_i + \sqrt{\frac{2 \ln t}{n_i}} \\]

**Decoding:**
- \\(\bar{X}_i\\): sample mean reward from arm i (the exploitation term)
- \\(t\\): total number of pulls so far across all arms
- \\(n_i\\): number of times arm i has been pulled
- \\(\sqrt{2 \ln t / n_i}\\): the exploration bonus — grows when arm i has been tried infrequently

**Why the log?** Auer et al. showed that UCB1 achieves regret (cumulative missed reward vs. omniscient play) of O(ln t). The log function grows slowly: ln(1000) ≈ 6.9, ln(1,000,000) ≈ 13.8. This means the exploration bonus shrinks relative to the exploitation term as total pulls grow — which is the right behavior. Early on, explore widely. Later, exploit the best arm.

**UCT** (Kocsis & Szepesvári, 2006) applies UCB1 to tree nodes: each child is treated as an arm, the parent's visit count plays the role of t, and wins/losses are the rewards. The constant 2 is absorbed into the tunable parameter c.

### Why log(N)/n grows slowly — a numerical example

For a node with parent visit count N = 100, the UCT exploration bonus (with c = 1.41) for children at varying visit counts:

| Child visits (n) | Exploit (W/N, assume W=n/2) | Explore term | UCT score |
|------------------|-----------------------------|--------------|-----------|
| 1 | 0.50 | 1.41 * sqrt(4.61/1) = 3.03 | 3.53 |
| 5 | 0.50 | 1.41 * sqrt(4.61/5) = 1.35 | 1.85 |
| 20 | 0.50 | 1.41 * sqrt(4.61/20) = 0.68 | 1.18 |
| 50 | 0.50 | 1.41 * sqrt(4.61/50) = 0.43 | 0.93 |
| 99 | 0.50 | 1.41 * sqrt(4.61/99) = 0.30 | 0.80 |

A child visited only once has a UCT score 4.4x higher than a child visited 99 times, even with identical win rates. This drives MCTS to eventually try every child — but the quickly-diminishing bonus means MCTS will not waste iterations on arms that have been clearly established as poor.

### Numerical UCT example: three children after varying visit counts

Suppose three children A, B, C. Parent N = 40.

- Child A: visited 20 times, won 14 (W=14). Exploit = 14/20 = 0.70.
- Child B: visited 15 times, won 6 (W=6). Exploit = 6/15 = 0.40.
- Child C: visited 5 times, won 4 (W=4). Exploit = 4/5 = 0.80.

UCT scores (c = 1.41, ln(40) ≈ 3.69):

- UCT(A) = 0.70 + 1.41 * sqrt(3.69 / 20) = 0.70 + 0.43 = **1.13**
- UCT(B) = 0.40 + 1.41 * sqrt(3.69 / 15) = 0.40 + 0.50 = **0.90**
- UCT(C) = 0.80 + 1.41 * sqrt(3.69 / 5) = 0.80 + 0.86 = **1.66**

Child C has the highest UCT score even though it has been visited least. Its 80% win rate plus the exploration bonus outweighs A's higher absolute win count. MCTS will select C next, collect more data, and eventually the estimates will converge.

---

## MCTS as approximate minimax

### Convergence theorem (informal)

Kocsis & Szepesvári (2006) proved that, for finite two-player zero-sum games, UCT converges to the minimax value as the number of iterations approaches infinity. Formally: the probability that UCT selects a non-optimal action at the root decreases polynomially in the number of iterations.

In practice: more iterations always produces a better approximation of the minimax value. MCTS with many iterations is playing essentially the same game as minimax, but through sampling rather than exhaustive enumeration.

### Why visit count beats win rate for final move selection

When you decide which move to actually play (not which node to expand next), you have two options:
1. Pick the child with the highest win rate W/N.
2. Pick the child with the highest visit count N.

**Option 2 is more robust.** Here is why: the child with the most visits is the one that UCT consistently judged worth revisiting. A child might have a temporarily inflated win rate from a small sample that includes lucky rollouts. Visit count integrates evidence over time; win rate alone can be noisy.

Additionally, in the presence of adversarial play: the minimax value of a position can differ significantly from the average outcome, and MCTS's visit count tracks something closer to the minimax estimate than the raw average.

### Code: logging visit counts and win rates for a 5-move game

```python
import math
import random

class SimpleMCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.untried = list(state.legal_actions())

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def uct_child(self, c=1.41):
        best, best_score = None, float('-inf')
        for action, child in self.children.items():
            exploit = child.W / child.N if child.N > 0 else 0
            explore = c * math.sqrt(math.log(self.N) / child.N) if child.N > 0 else float('inf')
            if exploit + explore > best_score:
                best_score = exploit + explore
                best = child
        return best

    def expand(self):
        action = self.untried.pop(random.randrange(len(self.untried)))
        next_state = self.state.apply(action)
        child = SimpleMCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        return child


def random_rollout(state):
    while not state.is_terminal():
        action = random.choice(list(state.legal_actions()))
        state = state.apply(action)
    return state.winner_reward()


def mcts_with_logging(root_state, num_iterations=200):
    root = SimpleMCTSNode(root_state)

    for i in range(num_iterations):
        # Selection
        node = root
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.uct_child()

        # Expansion
        if not node.state.is_terminal():
            node = node.expand()

        # Simulation
        outcome = random_rollout(node.state)

        # Backpropagation
        while node is not None:
            node.N += 1
            same_player = (node.state.current_player() ==
                           root_state.current_player())
            node.W += outcome if same_player else -outcome
            node = node.parent

    # Log final statistics
    print(f"{'Action':<12} {'Visits':>8} {'Win Rate':>10} {'Would Select?':>14}")
    print("-" * 48)
    best_visit_action = max(root.children, key=lambda a: root.children[a].N)
    best_winrate_action = max(root.children,
                              key=lambda a: root.children[a].W / root.children[a].N
                              if root.children[a].N > 0 else 0)
    for action, child in sorted(root.children.items()):
        by_visits = "YES (visit)" if action == best_visit_action else ""
        by_winrate = "YES (winrt)" if action == best_winrate_action else ""
        tag = by_visits or by_winrate
        print(f"{str(action):<12} {child.N:>8} {child.W/child.N:>10.3f} {tag:>14}")

    return best_visit_action, root
```

Typical output shows the two selection criteria usually agree, but diverge on low-visit arms where win-rate estimates are unreliable.

---

## Parallelization options

MCTS is inherently sequential — each iteration updates the same tree. Parallelization requires care to avoid race conditions and statistical corruption.

### Leaf parallelization

Run multiple rollouts from the same newly-expanded leaf node, in parallel threads or processes. Average the outcomes and use the average for a single backpropagation step.

**Tradeoff:** simple to implement; reduces variance of rollout estimates. But the number of parallel rollouts is bounded by the budget you want to spend on a single leaf, and deep search still proceeds serially.

### Root parallelization

Run completely independent MCTS trees in parallel, each starting from the root with its own random seed. At decision time, merge the visit counts across trees.

**Tradeoff:** trivially parallelizable; no shared state. Downside: no information sharing between trees. Tree A might spend thousands of iterations on a branch that Tree B quickly discovered was bad, wasting compute.

### Tree parallelization (with virtual loss)

Multiple threads share the same tree. Each thread locks nodes as it traverses, to prevent simultaneous writes. The challenge: two threads might both select the same promising node before either one updates its statistics.

The **virtual loss** technique addresses this: when a thread begins traversing through a node, immediately decrement that node's W by 1 (add a "virtual loss"). This makes the node look less attractive to other threads in the UCT formula, causing them to explore elsewhere. When the real outcome returns, add it back and remove the virtual loss.

```python
import threading

class ThreadSafeMCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.lock = threading.Lock()
        self.untried = list(state.legal_actions())

    def apply_virtual_loss(self):
        with self.lock:
            self.N += 1   # count visit immediately
            self.W -= 1   # add virtual loss: penalize to deter other threads

    def revert_virtual_loss(self, real_outcome: float):
        with self.lock:
            self.W += 1 + real_outcome  # remove virtual loss, add real outcome
```

**For GPU batching:** virtual loss is essential when batching evaluations. Without it, multiple threads select the same leaf, wasting the batch. With virtual loss, each thread takes a different path through the tree, building up a diverse batch of leaves for a single GPU forward pass. Once evaluations return, all threads backpropagate simultaneously.

---

## MCTS for SSA pursuit-evasion

### Game description

Two satellites in near-circular low Earth orbit:
- **Defender (D)**: wants to maintain a safe separation distance (>= 50 km in-track) from the attacker.
- **Attacker (A)**: wants to close the gap to within 10 km (proximity operations).

Each turn, both choose a maneuver simultaneously (or the game is modeled as turn-alternating to fit the MCTS 2-player framework). Maneuver options: `prograde` (+delta-v in velocity direction), `retrograde`, `radial`, `anti-radial`, `hold` (no maneuver). Fuel is finite; burning fuel costs 1 unit per maneuver from a budget of 10.

### State representation

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SSAPursuitState:
    """
    2D pursuit-evasion between two satellites using Hill-Clohessy-Wiltshire
    linearized relative dynamics. Positions in km, velocities in km/s.
    """
    rel_pos: np.ndarray   # [along-track, cross-track] km
    rel_vel: np.ndarray   # [along-track, cross-track] km/s
    defender_fuel: int    # 0-10 units
    attacker_fuel: int    # 0-10 units
    player_to_move: int   # 0 = defender, 1 = attacker
    turn: int
    max_turns: int = 20

    # Actions: prograde, retrograde, radial, anti-radial, hold
    DV_OPTIONS = [np.array([0.05,0]), np.array([-0.05,0]),
                  np.array([0,0.05]), np.array([0,-0.05]), np.array([0,0])]

    def legal_actions(self):
        fuel = self.defender_fuel if self.player_to_move == 0 else self.attacker_fuel
        return [4] if fuel == 0 else list(range(5))  # hold-only if out of fuel

    def apply(self, action_index):
        dv = self.DV_OPTIONS[action_index]
        cost = 0 if action_index == 4 else 1
        new_vel = self.rel_vel + (dv if self.player_to_move == 1 else -dv)
        new_pos = self.rel_pos + new_vel * (600 / 1000)  # 10-minute Euler step
        d_fuel = self.defender_fuel - (cost if self.player_to_move == 0 else 0)
        a_fuel = self.attacker_fuel - (cost if self.player_to_move == 1 else 0)
        return SSAPursuitState(new_pos, new_vel, d_fuel, a_fuel,
                               1 - self.player_to_move, self.turn + 1, self.max_turns)

    def is_terminal(self):
        sep = np.linalg.norm(self.rel_pos)
        return sep < 10 or sep > 200 or self.turn >= self.max_turns

    def winner_reward(self):
        """From attacker's perspective: +1=proximity achieved, -1=lost contact, 0=draw."""
        sep = np.linalg.norm(self.rel_pos)
        return +1.0 if sep < 10 else (-1.0 if sep > 200 else 0.0)

    def observation_tensor(self):
        return np.concatenate([self.rel_pos, self.rel_vel,
                               [self.defender_fuel/10, self.attacker_fuel/10,
                                self.player_to_move, self.turn/self.max_turns]])
```

### Why MCTS handles stochastic transitions better than minimax

In the real SSA scenario, drag perturbs the orbit at each time step: instead of the deterministic HCW propagation above, the transition is:

```
new_rel_pos = rel_pos + (rel_vel + drag_noise) * dt
```

where `drag_noise` is sampled from a distribution with standard deviation 0.1-1 km depending on orbital altitude and space weather.

Minimax's fix is expectimax: add "chance nodes" at each step that enumerate possible drag outcomes and average over them. With a continuous drag distribution, this requires discretizing into scenarios — multiplying the branching factor by the number of scenarios. For the SSA game above with 5 actions per player and 10 drag scenarios, the effective branching factor becomes 5 * 10 = 50 per player, 2500 per joint turn. Expectimax becomes intractable quickly.

MCTS's fix is free: during rollouts, sample a drag realization from the distribution at each step. The resulting rollout statistics automatically integrate over the stochastic transitions. No explicit chance node enumeration required. With enough rollouts, the statistics converge to the correct expected value under the transition distribution.

This is one of the most important practical advantages of MCTS for real-world planning under uncertainty.

---

## Key Takeaways

- **UCT** applies the multi-armed bandit formula UCB1 to tree nodes, balancing exploitation (high win rate) and exploration (few visits) via a slowly growing log term that ensures every child is eventually tried.
- **The exploration bonus shrinks as visits accumulate**, so MCTS naturally shifts from wide exploration early to focused exploitation later — without any hyperparameter tuning of the exploration schedule.
- **MCTS converges to minimax** in the limit of infinite iterations, but at any finite iteration count it provides a useful approximate answer, making it ideal for time-limited decision problems.
- **Visit count is a more robust final-move selector than win rate** because it aggregates statistical evidence over all iterations rather than reflecting possibly high-variance averages from small samples.
- **Parallelization via virtual loss** allows MCTS to batch leaf evaluations for GPU inference: by temporarily penalizing nodes under traversal, each parallel thread selects a different leaf, building a diverse evaluation batch.
- **MCTS handles stochastic transitions for free** by sampling outcomes during rollouts, avoiding the exponential blowup of expectimax chance nodes that makes minimax infeasible for real SSA pursuit-evasion scenarios.

## Quiz

{{#quiz 02-monte-carlo-tree-search.toml}}
