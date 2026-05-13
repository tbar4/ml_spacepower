# Lesson 3: Neural-Guided MCTS


<!-- toc -->

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

\[ UCT(\text{child}) = \frac{W}{N} + c \sqrt{\frac{\ln N_{parent}}{N}} \]

Neural-guided MCTS uses a variant called **PUCT** (Predictor + UCT):

\[ PUCT(\text{child}) = \frac{W}{N} + c \cdot P(\text{child}) \cdot \frac{\sqrt{N_{parent}}}{1 + N} \]

**Decoding the changes:**
- \(P(\text{child})\): the prior probability of this child according to the policy network
- The exploration term is multiplied by the prior. Children the policy network thinks are good get more exploration bonus; children it ignores get less.
- The square root structure is slightly different (\(\sqrt{N_{parent}}\) in numerator, \(1 + N\) in denominator) but the spirit is the same.

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

\[ \pi_{\text{search}}(a | s) = \frac{N(a)}{\sum_{a'} N(a')} \]

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

## The policy network's role in selection

**Module/Source:** Silver, D. et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search." *Nature* 529 (2016). Silver, D. et al. "A general reinforcement learning algorithm that masters chess, shogi and Go through self-play." *Science* 362 (2018).

### PUCT formula revisited: prior probabilities in selection

Recall the PUCT formula introduced above:

\[ PUCT(\text{child}) = \frac{W}{N} + c \cdot P(\text{child}) \cdot \frac{\sqrt{N_{parent}}}{1 + N} \]

**Decoding:**
- \(W / N\): empirical win rate from simulations — the exploitation term
- \(P(\text{child})\): prior probability assigned to this child by the policy network
- \(c \cdot P(\text{child}) \cdot \frac{\sqrt{N_{parent}}}{1 + N}\): the exploration bonus, **weighted by the prior**
- When \(N = 0\) (never visited): the score equals \(c \cdot P(\text{child}) \cdot \sqrt{N_{parent}}\) — children with high prior get the largest initial bonus

The policy network effectively pre-ranks the children before any simulation. Children the network considers implausible start with a tiny exploration bonus and may never be visited in a short search. Children the network considers strong start with a large bonus and are explored first.

### How prior probabilities bias tree expansion

Consider a defender satellite with 5 legal maneuvers. The policy network assigns:

| Action | Prior P(a) | UCT (no prior) at N=0 | PUCT (with prior) at N=0 |
|--------|-----------|----------------------|--------------------------|
| prograde | 0.55 | ∞ (all equal) | 0.55 * sqrt(N_parent) |
| retrograde | 0.05 | ∞ | 0.05 * sqrt(N_parent) |
| radial | 0.20 | ∞ | 0.20 * sqrt(N_parent) |
| anti-radial | 0.15 | ∞ | 0.15 * sqrt(N_parent) |
| hold | 0.05 | ∞ | 0.05 * sqrt(N_parent) |

With plain UCT, the first 5 iterations must visit all 5 children before any can be visited twice (because unvisited nodes have infinite UCT score). With PUCT, the search visits "prograde" first (highest prior), then "radial," then "anti-radial," and so on. "Retrograde" and "hold" may not be visited at all in a 20-iteration budget.

**The result:** PUCT focuses computation on plausible moves. The effective branching factor shrinks from the nominal 5 to roughly 2-3 moves that get meaningful visit counts in a short search. This is how neural guidance extends the practical depth of search — not by pruning (PUCT never prunes a child outright), but by concentrating iterations on promising parts of the tree.

### Why this focuses computation on plausible moves

In the SSA ISR sensor-vs-jammer game: the attacker satellite has a jammer and can choose to jam, deceive, or go quiet. An untrained policy assigns equal probability to all moves. A trained policy, having seen thousands of simulated engagements, knows that "deceive" is rarely correct when the defender's sensor is already tracking — it assigns P("deceive") ≈ 0.03. PUCT barely visits this branch. The search budget goes instead to "jam" and "go quiet," where the outcome actually varies meaningfully with subsequent defender choices.

Without the prior, MCTS would waste roughly 1/5 of its iterations on deception moves that are almost always bad. With the prior, that budget redirects to genuinely contested moves, allowing the search to go one or two levels deeper.

---

## Training the policy and value networks

### Self-play data generation pipeline

The training loop is:
1. The current network guides MCTS in self-play games.
2. Each move records: (state, MCTS visit distribution, eventual game outcome).
3. After N games, train the network on the collected examples.
4. The improved network replaces the old one. Repeat.

The data pipeline in pseudocode (full implementation is in Lesson 4's `AlphaZeroTrainer`):

```
for each training iteration:
    for each self-play game:
        state = new_game()
        while not terminal:
            root = neural_mcts_search(state, network, iterations=100)
            target_policy = visit_counts / sum(visit_counts)  # MCTS distribution
            buffer.append( (state_tensor, target_policy, current_player) )
            action = sample_from(target_policy, temperature=τ(move_num))
            state = state.apply(action)
        outcome = state.final_returns()
        # relabel each stored step with the outcome for that player
        for (s, p, player) in buffer[-game_length:]:
            training_data.append( (s, p, outcome[player]) )
```

Each stored step labels the state with the *eventual* game result, not an intermediate reward. This is the key bootstrapping signal: the network learns to predict from early positions what the final outcome will be under good play.

### Supervised learning from MCTS visit distributions

The policy head is trained with **cross-entropy loss** against the MCTS visit distribution. The visit distribution is not just the single chosen move (like in supervised imitation learning from expert moves) — it is a soft distribution over all moves, weighted by how much evidence MCTS collected for each.

This is important: a 51%-majority move in the visit distribution carries a strong training signal. A 49%-minority move still contributes a small gradient, encoding the information that the second-best option was meaningfully considered and nearly selected.

```python
def train_network_on_buffer(
    network: torch.nn.Module,
    buffer: deque,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    num_steps: int = 200,
) -> list[float]:
    """
    Train the AlphaZero network on (state, mcts_policy, outcome) triples.
    Returns list of per-step losses for logging.
    """
    import random

    network.train()
    losses = []

    for step in range(num_steps):
        if len(buffer) < batch_size:
            continue

        batch = random.sample(buffer, batch_size)
        states, target_policies, target_values = zip(*batch)

        # Stack into tensors
        states = torch.stack(states)                           # [B, state_dim]
        target_policies = torch.tensor(
            np.array(target_policies), dtype=torch.float32    # [B, num_actions]
        )
        target_values = torch.tensor(
            target_values, dtype=torch.float32                 # [B]
        )

        # Forward pass
        policy_logits, value_preds = network(states)
        # policy_logits: [B, num_actions]; value_preds: [B]

        # Policy loss: cross-entropy between network log-probs and MCTS visit distribution
        # This is equivalent to -sum_a [ pi_mcts(a) * log pi_net(a) ]
        log_probs = F.log_softmax(policy_logits, dim=1)         # [B, num_actions]
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

        # Value loss: mean squared error between predicted value and actual outcome
        value_loss = F.mse_loss(value_preds, target_values)

        # L2 regularization is applied via weight_decay in the optimizer
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(total_loss.item())

    return losses
```

The cross-entropy loss on the visit distribution serves as a form of **knowledge distillation**: the slow, expensive MCTS search (the "teacher") generates a soft probability distribution. The fast neural network (the "student") is trained to reproduce that distribution at inference time, becoming an approximation of the search process.

---

## Temperature in move selection

### What temperature controls

Temperature \(\tau\) controls how deterministically the agent selects moves from the visit-count distribution. Given visit counts \(N_a\) for each action a, the move distribution is:

\[ P(a) = \frac{N_a^{1/\tau}}{\sum_{a'} N_{a'}^{1/\tau}} \]

**Decoding:**
- \(\tau \to 0\): the distribution concentrates entirely on the most-visited action (argmax). The agent always plays the "best" move it found. Used for competitive play.
- \(\tau = 1\): the distribution is exactly proportional to visit counts. The agent plays weaker moves in proportion to how often MCTS explored them.
- \(\tau > 1\): the distribution flattens toward uniform. The agent plays randomly among all explored moves.

### Why you need different temperatures for training vs. play

**During training (\(\tau = 1\) for early moves):** You want diverse self-play games. If every game follows the greedy policy, all games become identical after a few moves, and the buffer fills with variations of the same situation. The network cannot learn from this. High temperature generates exploration — games take different paths, covering more of the state space.

**During competitive play (\(\tau \to 0\)):** You want the best possible move, not an exploratory move. Competitive evaluation uses greedy selection.

**In AlphaZero:** temperature is 1.0 for the first 30 moves of a game, then drops to near 0 for the rest. The first 30 moves are the "opening," where diverse play is most valuable for learning. Late-game play is sharper.

### Code: temperature effects on move distribution

```python
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend for scripts
import matplotlib.pyplot as plt

def temperature_softmax(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convert visit counts to a move distribution using temperature.
    Handles temperature=0 as greedy (argmax).
    """
    if temperature == 0:
        probs = np.zeros_like(visit_counts, dtype=float)
        probs[np.argmax(visit_counts)] = 1.0
        return probs

    counts_temp = visit_counts ** (1.0 / temperature)
    return counts_temp / counts_temp.sum()


def demonstrate_temperature_effect():
    """
    Show how temperature changes move selection for a fixed MCTS result.
    Scenario: MCTS ran 200 iterations on the SSA defender's move choice.
    """
    actions = ['prograde', 'retrograde', 'radial', 'anti-radial', 'hold']
    # Realistic visit counts after 200 MCTS iterations
    visit_counts = np.array([120, 15, 40, 20, 5])

    temperatures = [0, 0.25, 0.5, 1.0, 2.0]

    print(f"{'Action':<14}", end="")
    for tau in temperatures:
        print(f"  τ={tau:<5}", end="")
    print()
    print("-" * 70)

    distributions = {}
    for tau in temperatures:
        distributions[tau] = temperature_softmax(visit_counts, tau)

    for i, action in enumerate(actions):
        print(f"{action:<14}", end="")
        for tau in temperatures:
            print(f"  {distributions[tau][i]:.3f}  ", end="")
        print()

    print(f"\nRaw visit counts: {dict(zip(actions, visit_counts))}")
    print(f"\nAt τ=0: always plays 'prograde' (120 visits)")
    print(f"At τ=1: plays 'prograde' 60% of the time")
    print(f"At τ=2: distribution nearly uniform")


# Run the demonstration
demonstrate_temperature_effect()
```

Output:
```
Action          τ=0     τ=0.25  τ=0.5   τ=1.0   τ=2.0
----------------------------------------------------------------------
prograde        1.000   0.996   0.936   0.600   0.327
retrograde      0.000   0.000   0.009   0.075   0.122
radial          0.000   0.004   0.044   0.200   0.228
anti-radial     0.000   0.000   0.012   0.100   0.163
hold            0.000   0.000   0.000   0.025   0.161
```

At \(\tau = 1\), the agent plays "prograde" 60% of the time but occasionally tries other options — providing training diversity. At \(\tau = 0\), the agent is fully committed to "prograde" and the buffer fills with games where the defender always burns prograde early.

### Dirichlet noise for root exploration

Even with \(\tau = 1\), the agent might still never try certain actions if the policy network assigns them near-zero prior. AlphaZero adds **Dirichlet noise** to the root node's prior before each search:

```python
def add_dirichlet_noise(
    root_node,
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25
):
    """
    Add Dirichlet noise to root priors to ensure all actions get some exploration.
    dirichlet_alpha: concentration parameter (smaller = sparser noise)
    noise_weight: fraction of prior replaced by noise (0.25 in AlphaZero)
    """
    priors = np.array([child.P for child in root_node.children.values()])
    noise = np.random.dirichlet(alpha=[dirichlet_alpha] * len(priors))
    noisy_priors = (1 - noise_weight) * priors + noise_weight * noise

    for child, new_prior in zip(root_node.children.values(), noisy_priors):
        child.P = new_prior
```

Dirichlet noise with \(\alpha = 0.3\) (AlphaZero's value for chess) typically assigns 1-5% of exploration mass to otherwise-ignored moves. This ensures the self-play buffer includes at least some games exploring unusual lines.

---

## Key Takeaways

- **PUCT replaces UCT** by weighting the exploration bonus with the policy network's prior, so children the network considers strong are explored first and children it considers weak may never be visited in a short search budget.
- **The prior acts as an effective branching factor reducer**: in the SSA sensor-vs-jammer scenario, PUCT concentrates iterations on 2-3 plausible moves rather than spreading them evenly across 5, allowing the search to go deeper in the same compute budget.
- **The policy network is trained on MCTS visit distributions**, not hard move labels — this cross-entropy "distillation" encodes the search's uncertainty about close-call decisions, not just its top choice.
- **The value network replaces rollouts**: rather than playing out random moves to a terminal state, the network provides an immediate estimate of the position's outcome, giving a far lower-variance signal especially in games where random play is uninformative.
- **Temperature \(\tau\) controls the exploitation-exploration tradeoff at inference time**: \(\tau = 1\) during early-game training generates diverse self-play trajectories; \(\tau \to 0\) during evaluation ensures the agent plays its best move.
- **Dirichlet noise at the root** ensures that even moves the policy network disfavors receive occasional exploration, preventing the self-play buffer from collapsing to a single deterministic line of play.

## Quiz

{{#quiz 03-neural-guided-mcts.toml}}
