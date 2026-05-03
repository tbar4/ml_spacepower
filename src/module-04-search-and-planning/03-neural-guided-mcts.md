# Lesson 3: Neural-Guided MCTS

## Where this fits

Pure MCTS works but has two weaknesses: random rollouts produce noisy value estimates, and there is no way to inject prior knowledge about which moves are likely to be good. Neural networks fix both problems. A **value network** replaces the rollout: instead of playing random moves to a terminal state, just ask the network "what is the expected outcome from this position?" A **policy network** biases the selection phase: instead of UCT treating all children equally a priori, weight them by the network's prediction of which moves a strong player would consider.

This combination, neural-guided MCTS, is what powers AlphaGo Zero, AlphaZero, and MuZero. Once you understand it, AlphaZero (next lesson) is mostly about how to train these networks from self-play.

## The two networks

In neural-guided MCTS, you have one (or two) neural networks that take a game state as input.

**Value network V(s)**: outputs a single number, the predicted outcome from state s under expected play. Range typically [-1, +1] for two-player games (-1 = loss, 0 = draw, +1 = win).

**Policy network π(a|s)**: outputs a probability distribution over actions, predicting which actions a strong player would choose.

In AlphaZero, these are combined into a single network with two heads: a shared body of layers, then split into a policy head (softmax over actions) and a value head (single scalar). This is structurally identical to the actor-critic architecture from Module 3, lesson 6.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        policy_logits = self.policy_head(features)
        value         = torch.tanh(self.value_head(features)).squeeze(-1)
        return policy_logits, value
```

The value head's `tanh` activation constrains the output to (-1, +1), matching the expected range of game outcomes.

## PUCT: replacing UCT with a network-biased version

The selection phase of MCTS uses UCT (lesson 2):

\\[ UCT(\text{child}) = \frac{W}{N} + c \sqrt{\frac{\ln N_{parent}}{N}} \\]

Neural-guided MCTS uses a variant called **PUCT** (Predictor + UCT):

\\[ PUCT(\text{child}) = \frac{W}{N} + c \cdot P(\text{child}) \cdot \frac{\sqrt{N_{parent}}}{1 + N} \\]

**Decoding the changes:**
- \\(P(\text{child})\\): the prior probability of this child according to the policy network
- The exploration term is multiplied by the prior. Children the policy network thinks are good get more exploration bonus; children it ignores get less.
- The square root structure is slightly different (\\(\sqrt{N_{parent}}\\) in numerator, \\(1 + N\\) in denominator) but the spirit is the same.

This means: when the search has not yet visited a node many times, the policy network's prior dominates. As the visit count grows, the empirical win rate W/N takes over. The network gives a good starting guess; search refines it.

## Replacing rollouts with value network predictions

The simulation phase of pure MCTS plays out random moves to a terminal state. Neural-guided MCTS skips this entirely. When the search reaches a newly expanded node, it just asks the value network for an estimate of the outcome from that position.

```python
def evaluate_node(node, network):
    state_tensor = torch.tensor(node.state.observation_tensor(), dtype=torch.float32)
    with torch.no_grad():
        policy_logits, value = network(state_tensor)
    
    # Convert policy logits to probabilities, restricted to legal actions
    legal = node.state.legal_actions()
    legal_logits = policy_logits[legal]
    priors = F.softmax(legal_logits, dim=0)
    
    return priors.tolist(), value.item()
```

The value is used as the simulation outcome (backpropagated up the tree). The priors are stored on the new node and used by PUCT in future selection phases.

## A complete neural-guided MCTS implementation

```python
import math
import torch
import torch.nn.functional as F

class NeuralMCTSNode:
    def __init__(self, state, prior=0.0, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = {}    # action -> NeuralMCTSNode
        self.N        = 0
        self.W        = 0.0
        self.P        = prior  # prior probability from policy network
        self.expanded = False
    
    def is_leaf(self):
        return not self.expanded
    
    def best_puct_child(self, c=1.5):
        """Pick child with highest PUCT value."""
        best_action, best_score = None, float('-inf')
        for action, child in self.children.items():
            if child.N == 0:
                exploit = 0
            else:
                exploit = child.W / child.N
            explore = c * child.P * math.sqrt(self.N) / (1 + child.N)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action
        return self.children[best_action]


def neural_mcts_search(root_state, network, num_iterations=100, c=1.5):
    root = NeuralMCTSNode(root_state)
    
    # Expand the root once at the start
    priors, _ = evaluate_node(root, network)
    legal = root.state.legal_actions()
    for action, prior in zip(legal, priors):
        next_state = root.state.apply(action)
        root.children[action] = NeuralMCTSNode(
            next_state, prior=prior, parent=root, action=action
        )
    root.expanded = True
    
    for _ in range(num_iterations):
        # Phase 1: Selection
        node = root
        while not node.is_leaf() and not node.state.is_terminal():
            node = node.best_puct_child(c=c)
        
        # Phase 2: Expansion + Phase 3: Evaluation (replaces simulation)
        if node.state.is_terminal():
            value = node.state.terminal_value()  # actual outcome
        else:
            priors, value = evaluate_node(node, network)
            legal = node.state.legal_actions()
            for action, prior in zip(legal, priors):
                next_state = node.state.apply(action)
                node.children[action] = NeuralMCTSNode(
                    next_state, prior=prior, parent=node, action=action
                )
            node.expanded = True
        
        # Phase 4: Backpropagation
        # Value is from the perspective of the player at `node`.
        # Walk up, flipping sign for opponent nodes.
        while node is not None:
            node.N += 1
            node.W += value
            value = -value  # flip for parent (opponent)
            node = node.parent
    
    return root


def select_move(root, temperature=1.0):
    """Select a move from the root based on visit counts."""
    visits = [(action, child.N) for action, child in root.children.items()]
    if temperature == 0:
        # Deterministic: pick most-visited
        return max(visits, key=lambda x: x[1])[0]
    # Stochastic: sample proportional to visit counts (with temperature)
    actions, counts = zip(*visits)
    counts = torch.tensor(counts, dtype=torch.float32)
    probs = (counts ** (1 / temperature))
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, 1).item()
    return actions[idx]
```

The temperature parameter in `select_move` controls how stochastically moves are selected. Temperature 0 is greedy (always pick the most-visited move). Temperature 1 samples proportional to visit counts. During AlphaZero training, you use temperature > 0 in early moves to encourage diverse self-play games and temperature 0 in late moves and during evaluation.

## What the search produces

After running neural MCTS for some number of iterations, the visit counts at the root form an "improved policy": a probability distribution over moves that is generally better than the raw policy network output, because it has been refined by tree search.

\\[ \pi_{\text{search}}(a | s) = \frac{N(a)}{\sum_{a'} N(a')} \\]

This is the key insight that AlphaZero training relies on: **MCTS guided by a network produces a policy that is stronger than the network alone.** If we train the network to match the search's policy, the network gets better. Then the next round of search (using the better network) produces an even better policy. And so on.

This is the **policy improvement operator**: search makes the policy better. Training makes the network match the improved policy. Iterate.

## Why this works so well

Three factors:

**The value network gives clean rollouts.** In games where random rollouts are essentially random noise (most positions in Go are losing for both players under random play), a trained value network gives meaningful evaluations. The search results actually mean something.

**The policy network focuses search.** PUCT spends most of its iterations on plausible moves rather than wasting effort on obviously bad ones. The effective branching factor of the search shrinks, even though the network does not actually prune anything.

**Generalization across positions.** A trained network applies its learned knowledge to every position. Even positions never seen before get reasonable initial estimates, which the search refines. Pure MCTS has no such transfer.

The combination produces dramatically stronger play than either alone:
- Network alone: fast, but with mistakes the search would catch
- Search alone: thorough, but with poor evaluation in unusual positions
- Network + search: fast, thorough, and improving

## Hyperparameters that matter

- **Number of MCTS iterations per move**: more is better, with diminishing returns. AlphaZero used 800 per move during self-play, much more during evaluation. For our project, 50-200 will be plenty.
- **Exploration constant c**: typically 1.0 to 4.0. Too small: search overcommits to early-promising moves. Too large: search wastes iterations on unlikely moves.
- **Temperature for move selection**: 1.0 in early game (encourage variety), 0.0 in late game (play sharply).

## A note on "AlphaZero vs AlphaGo Zero vs MuZero"

These are all closely related:

- **AlphaGo Zero**: trained on Go from scratch, network and search as described here. Used a residual convolutional architecture for Go's 19x19 board.
- **AlphaZero**: same algorithm, but generalized to Chess, Go, and Shogi. Single architecture, learned from scratch in each game.
- **MuZero**: also learns the dynamics model. Does not need access to game rules during search. Lets you handle problems where you cannot easily simulate forward.

For our purposes, "AlphaZero" refers to the basic algorithm pattern: neural-guided MCTS + self-play training. We use that pattern in the next lesson and the project.

## Quiz

{{#quiz 03-neural-guided-mcts.toml}}
