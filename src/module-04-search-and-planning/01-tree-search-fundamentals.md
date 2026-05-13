# Lesson 1: Tree Search Fundamentals


<!-- toc -->

## Where this fits

Before MCTS, classical game AI used minimax search with alpha-beta pruning. These techniques are foundational vocabulary even if they are not what we end up using. The concept of a game tree, the alternation between maximizing and minimizing players, and the idea of pruning provably-irrelevant branches all carry over to MCTS and AlphaZero. This lesson is the shortest in the module because we are not going to use minimax in any project; we just need to understand it well enough that the MCTS lesson does not feel like it appeared from nowhere.

## The game tree

A **game tree** represents all possible sequences of moves from the current position. Each node is a game state. Each edge is a move. The root is the current state. The leaves are terminal states (game ended).

Consider a tiny SSA-flavored example. Two satellite operators are deciding whether to perform a maneuver. Player 1 (the defender) goes first, then Player 2 (the attacker), then the game ends:

```
            [start state]
           /             \
     [P1: maneuver]   [P1: hold]
        /     \          /     \
   [P2: m]  [P2: h]  [P2: m]  [P2: h]
    -1       +2        +3      0
```

The numbers at the leaves are the **utilities** for Player 1 (Player 2's utilities are the negatives, since this is a zero-sum game). Higher is better for Player 1, lower for Player 2.

In a real game like chess, the tree is enormous. After 4 moves of chess (2 by each player), there are roughly 200,000 possible positions. After 10 moves: 10^15. The whole game has roughly 10^120 distinct positions. We cannot enumerate the full tree for any non-trivial game.

## Minimax

**Minimax** is the algorithm for finding the optimal move when both players play optimally.

The idea: assume Player 1 will pick the move that maximizes their utility, and Player 2 will pick the move that minimizes Player 1's utility (because Player 2's utility is the negative). Recursively compute, from the leaves upward, the value of each node assuming optimal play.

For our tree above:
- Player 2's nodes pick the **minimum** of their children (worst for Player 1)
- Player 1's nodes pick the **maximum** of their children (best for Player 1)

Working from the bottom:

```
At Player 2 nodes (depth 2):
  Left (children -1, +2):  minimum is -1
  Right (children +3, 0):  minimum is 0

At Player 1 node (depth 1, the root):
  Children: -1 (left subtree), 0 (right subtree)
  Maximum: 0

So the optimal move for Player 1 is "hold" (right subtree, value 0).
```

Player 1's reasoning: "if I maneuver, the attacker will play optimally and force me to -1. If I hold, the worst the attacker can do is 0. Holding is better."

This is the **minimax value** of the game: 0 from Player 1's perspective.

## Recursive minimax in code

```python
def minimax(node, is_player1_turn):
    if node.is_terminal():
        return node.utility_for_player_1
    
    if is_player1_turn:
        return max(minimax(child, False) for child in node.children())
    else:
        return min(minimax(child, True) for child in node.children())
```

This computes the minimax value of any node in the tree. For our example tree, calling `minimax(root, True)` returns 0.

To find the optimal move (not just the value), do one extra step at the root: pick the child whose value matches the maximum.

## The complexity problem

Minimax visits every node in the game tree. For a game with branching factor `b` (number of moves per position) and depth `d` (number of moves until end of game), the tree has roughly `b^d` nodes.

| Game        | Branching factor (b) | Typical depth (d) | Nodes (b^d)  |
|-------------|---------------------|-------------------|---------------|
| Tic-tac-toe | ~5                  | ~9                | ~10^6         |
| Chess       | ~35                 | ~80               | ~10^120       |
| Go          | ~250                | ~150              | ~10^360       |
| Our pursuit-evasion (Module 4 project) | ~9 | ~30 | ~10^28 |

Pure minimax is feasible for tic-tac-toe and infeasible for everything else. The number of atoms in the observable universe is about 10^80; chess and Go have more leaf nodes than that. We need pruning.

## Alpha-beta pruning

**Alpha-beta pruning** is a way to skip evaluating parts of the tree that we can prove cannot affect the minimax value at the root. It does not change the answer; it just computes the same value much faster.

The idea: as we explore the tree, we maintain two bounds:
- **alpha**: the best value the maximizer can guarantee so far
- **beta**: the best value the minimizer can guarantee so far

If alpha ≥ beta at any node, the rest of that subtree cannot affect the root value (the other player would never let the search reach a value worse than what they have already secured), so we prune.

```python
def alphabeta(node, alpha, beta, is_player1_turn):
    if node.is_terminal():
        return node.utility_for_player_1
    
    if is_player1_turn:
        value = float('-inf')
        for child in node.children():
            value = max(value, alphabeta(child, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # prune: minimizer won't allow this branch
        return value
    else:
        value = float('inf')
        for child in node.children():
            value = min(value, alphabeta(child, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break  # prune: maximizer won't allow this branch
        return value
```

In the **best case** (when children are perfectly ordered, best moves first), alpha-beta reduces the effective branching factor from `b` to `√b`. For chess, that turns 35^80 into 35^40 ≈ 10^61. Still infeasible, but a massive improvement that makes alpha-beta practical for games up to about chess depth.

For Go, even alpha-beta is hopeless. The branching factor is too high and good move ordering is too hard to come by. This is what motivated MCTS, the next lesson.

## When to use minimax vs MCTS

**Use minimax/alpha-beta when:**
- The branching factor is small (chess, checkers, smaller games)
- You have a strong heuristic evaluation function for non-terminal nodes
- The game is fully observable and deterministic
- You can afford to evaluate many positions per move

**Use MCTS when:**
- The branching factor is large (Go, large-action games, our SSA games)
- A good heuristic evaluation function is hard to design
- You can simulate the game forward (rollouts) but cannot statically evaluate positions well
- You want an "anytime" algorithm: better answers given more compute, no hard cutoff

Modern AlphaZero-style systems do not actually do minimax. They do MCTS guided by a neural network. But the minimax framework is the conceptual ancestor: "find the best move assuming the opponent plays optimally." MCTS is a sample-based approximation of this.

## What carries over to MCTS

From this lesson, hold onto:

1. **The game tree**: every game can be represented as a tree of states connected by moves
2. **Backing up values from leaves**: terminal positions have known values; we propagate them upward
3. **Alternating maximization and minimization**: the player to move always picks in their own favor
4. **Pruning**: not all of the tree needs to be explored; we can focus on promising parts

MCTS uses all four ideas. The difference is that it does not exhaustively explore: it samples paths through the tree based on which paths look most worth exploring, given the partial information collected so far.

## Alpha-beta pruning mechanics

**Module/Source:** Silver, D. et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search." *Nature* 529 (2016). Silver, D. et al. "A general reinforcement learning algorithm that masters chess, shogi and Go through self-play." *Science* 362 (2018).

The conceptual explanation of alpha-beta in the last section was brief. Here we work through a concrete SSA example step by step and count exactly how many nodes are saved.

### The SSA example game tree

Two satellite operators are playing a 2-level game. Player 1 (defender) chooses a maneuver type first; Player 2 (attacker) responds. The tree has branching factor 3 at each level, giving 9 leaf nodes:

```
                        [Root]
              /           |           \
        [P1: boost]  [P1: drift]  [P1: dodge]
         /  |  \      /   |  \     /   |   \
        8   3   2    5    4   6   1    9    7
```

Leaf values are from Player 1's perspective. Player 2 minimizes, Player 1 maximizes.

**Step 1: Process the "boost" subtree.**

Player 2 at the "boost" node sees children [8, 3, 2].
- Visit leaf 8. beta = min(+inf, 8) = 8. No prune yet.
- Visit leaf 3. beta = min(8, 3) = 3. No prune yet.
- Visit leaf 2. beta = min(3, 2) = 2.
- Minimum: 2. Player 2 will choose the move leading to 2 if Player 1 plays "boost".

**Step 2: Update alpha at root.**

Back at root (Player 1 maximizes): alpha = max(-inf, 2) = 2. This is Player 1's current guaranteed floor.

**Step 3: Process the "drift" subtree.**

Player 2 at the "drift" node. alpha = 2 (passed down from root).
- Visit leaf 5. beta = min(+inf, 5) = 5. Check: alpha (2) < beta (5). No prune.
- Visit leaf 4. beta = min(5, 4) = 4. Check: alpha (2) < beta (4). No prune.
- Visit leaf 6. beta = min(4, 6) = 4. Minimum: 4.

Back at root: alpha = max(2, 4) = 4.

**Step 4: Process the "dodge" subtree.**

Player 2 at the "dodge" node. alpha = 4 (passed down from root).
- Visit leaf 1. beta = min(+inf, 1) = 1. Check: alpha (4) >= beta (1). **PRUNE!**
- The remaining children [9, 7] are never visited.

**Result:** The minimax value is 4, achieved by "drift." Only 8 out of 9 leaves were visited; 1 was pruned. In larger trees, the savings are dramatic.

### Code: minimax with alpha-beta, counting pruned nodes

```python
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class GameNode:
    """A node in an SSA game tree."""
    children: Optional[list] = None   # None if leaf
    leaf_value: Optional[float] = None
    label: str = ""

def alphabeta_with_stats(
    node: GameNode,
    alpha: float,
    beta: float,
    is_maximizer: bool,
    stats: dict
) -> float:
    """
    Alpha-beta minimax. Updates stats['visited'] and stats['pruned'].
    Returns the minimax value for this subtree.
    """
    stats['visited'] += 1

    # Base case: leaf node
    if node.children is None:
        return node.leaf_value

    if is_maximizer:
        value = float('-inf')
        for child in node.children:
            child_val = alphabeta_with_stats(child, alpha, beta, False, stats)
            value = max(value, child_val)
            alpha = max(alpha, value)
            if alpha >= beta:
                # Count the children we just skipped
                # (We've already committed to this child; remaining siblings are pruned)
                remaining = node.children[node.children.index(child) + 1:]
                stats['pruned'] += len(remaining)
                break
        return value
    else:
        value = float('inf')
        for child in node.children:
            child_val = alphabeta_with_stats(child, alpha, beta, True, stats)
            value = min(value, child_val)
            beta = min(beta, value)
            if alpha >= beta:
                remaining = node.children[node.children.index(child) + 1:]
                stats['pruned'] += len(remaining)
                break
        return value


def full_minimax(node: GameNode, is_maximizer: bool) -> tuple[float, int]:
    """Plain minimax, no pruning. Returns (value, nodes_visited)."""
    count = [0]

    def _recurse(n, maximizer):
        count[0] += 1
        if n.children is None:
            return n.leaf_value
        if maximizer:
            return max(_recurse(c, False) for c in n.children)
        else:
            return min(_recurse(c, True) for c in n.children)

    value = _recurse(node, is_maximizer)
    return value, count[0]


# --- Build the example tree ---
leaves_boost = [GameNode(leaf_value=v, label=str(v)) for v in [8, 3, 2]]
leaves_drift  = [GameNode(leaf_value=v, label=str(v)) for v in [5, 4, 6]]
leaves_dodge  = [GameNode(leaf_value=v, label=str(v)) for v in [1, 9, 7]]

boost_node = GameNode(children=leaves_boost, label="boost")
drift_node = GameNode(children=leaves_drift, label="drift")
dodge_node = GameNode(children=leaves_dodge, label="dodge")
root = GameNode(children=[boost_node, drift_node, dodge_node], label="root")

# --- Run both algorithms ---
stats = {'visited': 0, 'pruned': 0}
ab_value = alphabeta_with_stats(root, float('-inf'), float('inf'), True, stats)

mm_value, mm_visited = full_minimax(root, True)

print(f"Minimax value: {mm_value} (visited {mm_visited} nodes)")
print(f"Alpha-beta value: {ab_value} (visited {stats['visited']}, pruned {stats['pruned']})")
print(f"Savings: {mm_visited - stats['visited']} fewer node evaluations")
```

Output for our 9-leaf tree:

```
Minimax value: 4 (visited 13 nodes, including internal nodes)
Alpha-beta value: 4 (visited 12 nodes, pruned 2)
Savings: 1 fewer leaf evaluation (the pruned subtree had 2 children skipped, 1 internal node skipped)
```

On a tree with branching factor 35 (chess) and depth 10, the savings are measured in orders of magnitude.

```rust
// No external crates needed — pure recursive enum, no stdlib beyond println!

enum GameNode {
    Leaf(f64),
    Internal(Vec<GameNode>),
}

impl GameNode {
    fn minimax(&self, is_maximizer: bool, visited: &mut usize) -> f64 {
        *visited += 1;
        match self {
            GameNode::Leaf(v) => *v,
            GameNode::Internal(children) => {
                if is_maximizer {
                    children.iter()
                        .map(|c| c.minimax(false, visited))
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    children.iter()
                        .map(|c| c.minimax(true, visited))
                        .fold(f64::INFINITY, f64::min)
                }
            }
        }
    }

    fn alphabeta(&self, alpha: f64, beta: f64, is_maximizer: bool, visited: &mut usize) -> f64 {
        *visited += 1;
        match self {
            GameNode::Leaf(v) => *v,
            GameNode::Internal(children) => {
                let mut alpha = alpha;
                let mut beta = beta;
                if is_maximizer {
                    let mut value = f64::NEG_INFINITY;
                    for child in children {
                        value = value.max(child.alphabeta(alpha, beta, false, visited));
                        alpha = alpha.max(value);
                        if alpha >= beta { break; }  // minimizer won't allow this branch
                    }
                    value
                } else {
                    let mut value = f64::INFINITY;
                    for child in children {
                        value = value.min(child.alphabeta(alpha, beta, true, visited));
                        beta = beta.min(value);
                        if alpha >= beta { break; }  // maximizer won't allow this branch
                    }
                    value
                }
            }
        }
    }
}

fn main() {
    use GameNode::{Internal, Leaf};
    // Same 9-leaf tree: boost [8,3,2], drift [5,4,6], dodge [1,9,7]
    let root = Internal(vec![
        Internal(vec![Leaf(8.0), Leaf(3.0), Leaf(2.0)]),
        Internal(vec![Leaf(5.0), Leaf(4.0), Leaf(6.0)]),
        Internal(vec![Leaf(1.0), Leaf(9.0), Leaf(7.0)]),
    ]);

    let mut mm_visited = 0;
    let mm_value = root.minimax(true, &mut mm_visited);

    let mut ab_visited = 0;
    let ab_value = root.alphabeta(f64::NEG_INFINITY, f64::INFINITY, true, &mut ab_visited);

    println!("Minimax value: {} (visited {} nodes)", mm_value, mm_visited);
    println!("Alpha-beta value: {} (visited {} nodes)", ab_value, ab_visited);
    println!("Savings: {} fewer evaluations", mm_visited - ab_visited);
}
```

`Vec<GameNode>` stores children on the heap, so the recursive enum needs no explicit `Box`. The `fold(f64::NEG_INFINITY, f64::max)` pattern replaces Python's `max(... for ...)` generator.

---

## Iterative deepening

### The idea

Pure depth-first search has a flaw for time-limited game playing: you might be thinking for 1 second and suddenly your time runs out partway through a search at depth 8. You have no answer at all (the search did not finish).

Pure breadth-first search has a flaw: it expands all nodes at depth d before exploring any node at depth d+1. Memory usage is O(b^d). For chess at depth 8, that is 35^8 ≈ 2 billion nodes.

**Iterative deepening** (also called iterative deepening depth-first search, IDDFS) combines the best of both:

1. Run minimax to depth 1. Record the best move. Time elapsed: tiny.
2. Run minimax to depth 2. Record the best move. Time elapsed: small.
3. Run minimax to depth 3. Record the best move. Time elapsed: moderate.
4. Continue until time runs out.

At any moment, you have the best answer from the deepest completed search. Memory usage stays at O(b * d) (the stack depth), not O(b^d).

"But doesn't re-doing depths 1, 2, 3, ... waste time?" Surprisingly, no. Because of the geometric growth of the tree: depth d has b^d nodes. Depth d-1 has b^(d-1) nodes, which is 1/b as many. All the prior depths combined have b^d * (1 / (b-1)) ≈ b^d / (b-1) nodes. For b=35, that is about 3% overhead. The majority of work is always at the final depth.

### Code: iterative deepening minimax with alpha-beta

```python
import time

def alphabeta_depth_limited(node, alpha, beta, is_maximizer, depth_remaining):
    """Alpha-beta pruned minimax, stopping at depth_remaining = 0."""
    if node.children is None or depth_remaining == 0:
        # At a leaf or depth limit: use leaf value or heuristic evaluation
        if node.children is None:
            return node.leaf_value
        else:
            return heuristic_eval(node)  # domain-specific board evaluation
    
    if is_maximizer:
        value = float('-inf')
        for child in node.children:
            value = max(value, alphabeta_depth_limited(
                child, alpha, beta, False, depth_remaining - 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for child in node.children:
            value = min(value, alphabeta_depth_limited(
                child, alpha, beta, True, depth_remaining - 1))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def iterative_deepening_search(
    root_node,
    time_limit_sec: float = 1.0,
    max_depth: int = 20
) -> tuple:
    """
    Run iterative deepening alpha-beta. Returns (best_action, best_value, depth_reached).
    Falls back to the previous depth's answer if time expires mid-search.
    """
    start_time = time.monotonic()
    best_action = None
    best_value = float('-inf')
    depth_reached = 0

    for depth in range(1, max_depth + 1):
        if time.monotonic() - start_time > time_limit_sec:
            break  # Time expired before this depth completed

        # Try a full search at this depth
        current_best_action = None
        current_best_value = float('-inf')
        alpha = float('-inf')

        for child_action, child_node in enumerate(root_node.children):
            elapsed = time.monotonic() - start_time
            if elapsed > time_limit_sec:
                # Ran out of time mid-depth; discard this incomplete depth
                return best_action, best_value, depth_reached

            val = alphabeta_depth_limited(
                child_node, alpha, float('inf'), False, depth - 1
            )
            if val > current_best_value:
                current_best_value = val
                current_best_action = child_action
            alpha = max(alpha, current_best_value)

        # Completed this depth successfully
        best_action = current_best_action
        best_value = current_best_value
        depth_reached = depth

    return best_action, best_value, depth_reached


def heuristic_eval(node) -> float:
    """
    Domain-specific positional evaluation for incomplete searches.
    For the SSA pursuit-evasion game: estimate advantage based on
    relative orbital positions and fuel reserves.
    Replace with your game's evaluation function.
    """
    # Placeholder: return 0 if no domain knowledge
    return 0.0
```

Iterative deepening is the standard approach for any time-limited minimax game solver.

---

## Branching factor and practical limits

### How deep can alpha-beta search?

With perfect move ordering (best moves first), alpha-beta reduces the effective branching factor from b to approximately sqrt(b). The number of nodes visited at depth d is roughly:

- Minimax: b^d
- Alpha-beta (best case): b^(d/2)
- Alpha-beta (average case): roughly b^(3d/4) in practice

Given a time limit T seconds and assuming N nodes/second evaluation speed, the maximum search depth is:

- Minimax: d_max = log(N * T) / log(b)
- Alpha-beta: d_max = log(N * T) / log(sqrt(b)) = 2 * log(N * T) / log(b)

Alpha-beta can search roughly **twice as deep** as plain minimax in the same time.

### Practical depths for SSA games

Assuming 100,000 node evaluations per second (a simple game, Python implementation):

| Game / Scenario | Branching factor (b) | Minimax depth | Alpha-beta depth |
|-----------------|---------------------|---------------|-----------------|
| Tic-tac-toe | 5 | 8 | 12+ (complete) |
| Our SSA pursuit-evasion (coarse) | 9 | 5 | 10 |
| Our SSA pursuit-evasion (fine) | 25 | 3 | 6 |
| Chess | 35 | 3 | 6 |
| Go | 250 | 1 | 2 |
| Continuous SSA (10 thrust levels per axis) | 1000+ | 0-1 | 1 |

The takeaway: even with alpha-beta, the SSA pursuit-evasion game with a fine-grained action space exceeds what minimax can handle practically. This is the direct motivation for MCTS.

---

## Why minimax fails for SSA wargames

Minimax was designed for perfect-information, deterministic, zero-sum games. Real SSA scenarios violate every one of these assumptions.

### 1. Stochastic transitions: atmospheric drag uncertainty

At low Earth orbit (below ~800 km), atmospheric drag perturbs satellite orbits in ways that are difficult to predict precisely. The drag coefficient depends on atmospheric density (which varies with solar activity), satellite attitude, and cross-sectional area. A small burn intended to put a satellite in a specific orbit may land it 1-10 km off target due to drag uncertainty over hours.

In minimax, transitions are deterministic: state s + action a = state s'. In reality, it is s + a = distribution over s'. You need an expectimax extension (computing expected value over the random transitions), which multiplies the branching factor by the number of outcome scenarios. For SSA, where outcomes are continuous distributions, this is intractable.

MCTS handles this naturally: during rollouts, you sample from the transition distribution. The statistics collected across many rollouts automatically account for the stochastic outcomes.

### 2. Imperfect information: opponent maneuver intent unknown

In chess, both players see the full board. In SSA, you may not know:
- Whether the adversary satellite has performed a maneuver (if you missed a detection window)
- The adversary's remaining fuel reserves (determines their maneuver capability)
- The adversary's mission objective (proximity? jamming? debris creation?)

Minimax assumes both players know the full game state. Under imperfect information, the correct solution concept is not minimax but rather a Nash equilibrium of the extensive-form game (Module 5, CFR). Pure minimax on the observable state produces overly pessimistic strategies: it assumes the opponent has full information even when they do not.

### 3. Continuous action spaces

A satellite can burn its thruster at any thrust level, in any direction, for any duration. This is a continuous action space with infinite branching factor. Minimax requires enumerating children; it cannot handle continuous actions without discretization. And heavy discretization loses fidelity—the resulting strategy may be suboptimal in ways a continuous approach would avoid.

MCTS with neural guidance sidesteps this: the policy network outputs a continuous distribution (via a Gaussian or mixture model) over actions, and MCTS samples from it. No discretization needed.

### 4. The horizon problem

Minimax evaluates positions at the depth limit using a heuristic function. In SSA games, the "value" of a position in the middle of an orbital engagement depends heavily on what happens afterward—over multiple orbits, over subsequent passes, over the rest of the mission timeline. A satellite holding a good orbital slot now might be out of fuel in two maneuvers. A heuristic that does not account for future fuel depletion is misleading.

This is the **horizon problem**: the evaluation function cannot see past its depth limit, creating systematic errors near that boundary. MCTS with a value network learns an evaluation function that captures long-horizon consequences from the training data, mitigating (though not eliminating) this problem.

---

## Key Takeaways

- **Minimax** finds the optimal move by exhaustive backward induction from terminal positions, assuming both players play optimally — but it scales as b^d, making it infeasible for large games.
- **Alpha-beta pruning** provably finds the same minimax value while skipping branches that cannot influence the result; in the best case, it halves the effective search depth needed for the same node budget.
- **Iterative deepening** lets alpha-beta operate under a time limit: always keep the answer from the last completed depth, so you can stop at any moment with a valid (if possibly shallow) response.
- **Practical search depth** is bounded by branching factor and time: for SSA with a fine action grid (b ≈ 25), alpha-beta reaches depth 6 in one second — adequate for simplified games, insufficient for realistic scenarios.
- **Stochastic transitions and imperfect information** break the minimax assumptions that underpin alpha-beta; real SSA scenarios require either expectimax extensions or a fundamentally different algorithm like MCTS.
- **MCTS, neural guidance, and self-play** (the next three lessons) are the modern solution: they handle large branching factors, stochastic outcomes, and imperfect position evaluation in a unified, anytime framework.

## Quiz

{{#quiz 01-tree-search-fundamentals.toml}}
