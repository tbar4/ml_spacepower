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

## Quiz

{{#quiz 02-monte-carlo-tree-search.toml}}
