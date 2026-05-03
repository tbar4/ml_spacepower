# Lesson 1: Tree Search Fundamentals

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

## Quiz

{{#quiz 01-tree-search-fundamentals.toml}}
