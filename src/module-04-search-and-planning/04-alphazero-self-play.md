# Lesson 4: AlphaZero Self-Play


<!-- toc -->

## Where this fits

The previous lesson showed how MCTS guided by a network produces a stronger policy than the network alone. AlphaZero turns this into a learning algorithm: train the network to match the search's policy, then use the improved network to guide better searches, then train again. This iteration, **self-play training**, is the central conceptual contribution of AlphaZero. It produces dramatically strong game-playing agents starting from zero domain knowledge. The same pattern is used in Module 6 (PSRO is an analogous "best-response and update" loop) and is the conceptual model for the capstone in Module 8.

## The self-play training loop

The complete AlphaZero loop has three phases that repeat:

**Phase 1: Self-play game generation.** Use the current network and MCTS to play complete games against itself. At each move, run MCTS to produce an improved policy distribution \\(\pi_{\text{search}}(a | s)\\), then sample an action from it. Record the state, the search policy, and (when the game ends) the final outcome.

**Phase 2: Network training.** Use the recorded games as training data. The network's policy head is trained to match \\(\pi_{\text{search}}\\) (the MCTS-improved policy) using cross-entropy loss. The value head is trained to predict the final game outcome using MSE loss.

**Phase 3: Iterate.** Use the newly trained network for the next round of self-play. As the network improves, MCTS produces better policies. As MCTS produces better policies, the network has better training targets. The two improve together.

This is bootstrapping: the agent improves itself by treating its own (search-improved) decisions as ground truth.

## Why self-play works

Without external supervision, how can an agent know what good moves look like? Because MCTS is a policy improvement operator: given any policy, MCTS guided by it produces a stronger policy. If you train the network to match MCTS's output, the new network is stronger than the old one. Apply MCTS to the new network, get an even stronger policy, and so on.

The mathematical foundation is roughly: in two-player zero-sum games, MCTS converges (in the limit of infinite simulations) to a Nash equilibrium policy. Self-play with sufficient search depth iteratively closes the gap between the current network and that equilibrium.

In practice, you stop iterating when the agent stops improving (for example, when new networks no longer beat old networks in head-to-head play).

## The training data

For each game played in self-play, you store a list of training examples. Each example contains:

- **State** \\(s_t\\): the position at time t
- **Search policy** \\(\pi_t\\): the visit-count distribution from MCTS at this position
- **Outcome** \\(z\\): the final game result (+1 if root player won, -1 if lost, 0 if draw), with appropriate sign-flipping based on whose turn it was at \\(s_t\\)

A game of 50 moves produces 50 training examples. After many self-play games, you have a dataset of (state, target policy, target value) triples to train the network on.

## The loss function

The network is trained on a combined loss with three parts:

\\[ L = L_{\text{policy}} + L_{\text{value}} + L_{\text{regularization}} \\]

**Policy loss** (cross-entropy between network policy and search policy):

\\[ L_{\text{policy}} = -\sum_a \pi_t(a) \log \pi_{\text{net}}(a | s_t) \\]

This pushes the network's policy output to match the MCTS search policy.

**Value loss** (MSE between network value and game outcome):

\\[ L_{\text{value}} = (z - V_{\text{net}}(s_t))^2 \\]

This pushes the network's value output to match the actual game outcome.

**Regularization**: L2 penalty on the network weights, to prevent overfitting. Standard machine learning hygiene.

## A complete self-play training loop

Here is the structure of an AlphaZero training loop. The actual code is verbose; this shows the structure.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class AlphaZeroTrainer:
    def __init__(self, game_class, network, lr=1e-3, buffer_size=20_000):
        self.game_class = game_class
        self.network    = network
        self.optimizer  = torch.optim.Adam(network.parameters(), lr=lr,
                                          weight_decay=1e-4)  # L2 regularization
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def play_self_play_game(self, num_mcts_iterations=100, temperature_threshold=10):
        """Play one self-play game and store training examples."""
        state = self.game_class.new_initial_state()
        examples = []  # list of (state_tensor, search_policy, current_player)
        
        move_count = 0
        while not state.is_terminal():
            # Run MCTS from current position
            root = neural_mcts_search(state, self.network, 
                                       num_iterations=num_mcts_iterations)
            
            # Compute search policy from visit counts
            visits = np.zeros(state.num_distinct_actions())
            for action, child in root.children.items():
                visits[action] = child.N
            search_policy = visits / visits.sum()
            
            # Store this position
            state_tensor = torch.tensor(state.observation_tensor(), dtype=torch.float32)
            examples.append((state_tensor, search_policy, state.current_player()))
            
            # Sample a move (with temperature for exploration in early game)
            temperature = 1.0 if move_count < temperature_threshold else 0.01
            action = select_move(root, temperature=temperature)
            state.apply_action(action)
            move_count += 1
        
        # Game ended; assign outcome to each example based on whose turn it was
        outcome = state.returns()  # game-specific: returns from each player's perspective
        for state_t, policy, player in examples:
            value = outcome[player]
            self.replay_buffer.append((state_t, policy, value))
    
    def train_step(self, batch_size=64):
        """One gradient step on a batch of examples from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, policies, values = zip(*batch)
        
        states   = torch.stack(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values   = torch.tensor(values, dtype=torch.float32)
        
        # Forward pass
        policy_logits, value_preds = self.network(states)
        
        # Policy loss: cross-entropy between network policy and search policy
        # Use log_softmax + negative dot product (equivalent to KL divergence + entropy of policies)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        
        # Value loss: MSE between predicted value and actual outcome
        value_loss = F.mse_loss(value_preds, values)
        
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iterations=100, games_per_iteration=10, train_steps_per_iteration=100):
        for iteration in range(num_iterations):
            # Phase 1: self-play
            for _ in range(games_per_iteration):
                self.play_self_play_game()
            
            # Phase 2: training
            losses = []
            for _ in range(train_steps_per_iteration):
                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)
            
            avg_loss = np.mean(losses) if losses else 0
            print(f"Iteration {iteration}: avg loss = {avg_loss:.4f}, "
                  f"buffer size = {len(self.replay_buffer)}")
```

The full implementation has many engineering details (parallelizing self-play across processes, saving checkpoints, evaluating new vs. old networks, applying data augmentation through symmetries). For our project, this stripped-down version captures the essence.

## Comparing AlphaZero to DQN (Module 3)

Both are deep RL methods. They differ in fundamental ways.

| Aspect | DQN | AlphaZero |
|--------|-----|-----------|
| Action selection | Greedy w.r.t. Q-network | Sampled from MCTS visit counts |
| Network output | Q values for all actions | Policy + value |
| Policy improvement | None (just track Q*) | MCTS during self-play |
| Sample collection | Free play with ε-greedy | Self-play with MCTS at every move |
| Best for | Single-agent, learnable from limited compute | Two-player zero-sum games, willing to spend compute |

DQN is simpler and faster to train. AlphaZero is much stronger when you have the compute, especially for two-player games where MCTS can do real lookahead.

For our SSA-flavored multi-agent settings (Modules 5-7), AlphaZero-style self-play is closer to what we want than DQN. CFR (next module) and PSRO (Module 6) both have a "best response and aggregate" structure that mirrors AlphaZero's "search and train" loop.

## What can go wrong

**Self-play collapse**: if both players converge to a deterministic strategy, all games are identical and there is nothing to learn. Temperature in early moves and Dirichlet noise on the root prior (a small randomization added to the policy network's output) prevent this.

**Catastrophic forgetting**: as the network changes, it might forget how to play positions that were common with earlier networks but rare with the current one. Replay buffer size and a slow learning rate help.

**The network fails to improve**: sometimes self-play plateaus. Common fixes: more MCTS iterations per move (stronger search → better targets), more games per training iteration, larger network capacity.

**Computational cost**: AlphaZero on a real game (chess, Go) takes thousands of GPUs. On our small game it takes hours on a laptop. Make sure your game is small enough to make the loop tractable.

## The self-play data pipeline

**Module/Source:** Silver, D. et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search." *Nature* 529 (2016). Silver, D. et al. "A general reinforcement learning algorithm that masters chess, shogi and Go through self-play." *Science* 362 (2018).

### Game buffer, MCTS games, and training batches

AlphaZero's data pipeline has three distinct components that operate in a cycle:

1. **Self-play workers** run MCTS games using the current best network, writing (state, policy, outcome) triples into a shared **game buffer** (a replay buffer).
2. **The training process** samples random mini-batches from the game buffer and takes gradient steps on the network.
3. **The evaluator** periodically pits the latest trained network against the previous best network in head-to-head games. If the new network wins more than a threshold fraction of games, it becomes the new "best network" used for self-play.

The game buffer is a rolling window — old games are discarded as new ones arrive. This prevents the network from overfitting to games from early training, when play quality was low.

### Code skeleton: the outer training loop

The `AlphaZeroTrainer` class earlier in this lesson implements the inner loop. Here we show the game buffer and the outer evaluation harness that wraps it:

```python
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class GameBuffer:
    """
    Fixed-size circular buffer for (state, mcts_policy, outcome) training triples.
    Older examples are automatically discarded when capacity is exceeded.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, examples: list):
        self.buffer.extend(examples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, outcomes = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(np.array(policies), dtype=torch.float32),
            torch.tensor(outcomes, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def evaluate_networks(challenger, champion, game_class,
                      num_games: int = 40, mcts_iters: int = 100) -> float:
    """
    Head-to-head tournament. Returns win rate for challenger.
    Challenger alternates sides to eliminate first-mover advantage.
    """
    wins = 0
    for g in range(num_games):
        state = game_class.new_game()
        challenger_player = g % 2  # alternate sides each game
        while not state.is_terminal():
            current = state.current_player()
            net = challenger if current == challenger_player else champion
            root = neural_mcts_search(state, net, num_iterations=mcts_iters)
            action = max(root.children, key=lambda a: root.children[a].N)
            state = state.apply(action)
        if state.final_returns()[challenger_player] > 0:
            wins += 1
    return wins / num_games


def alphazero_outer_loop(
    game_class, network, games_per_iter=20, train_steps=100,
    eval_games=40, replace_threshold=0.55, num_iterations=200,
):
    """
    Outer loop: self-play → train → evaluate → maybe replace best network.
    Uses AlphaZeroTrainer from the lesson code for the inner loop.
    """
    import copy
    best_network = copy.deepcopy(network)
    trainer = AlphaZeroTrainer(game_class, network)
    elo_tracker = EloTracker()

    for iteration in range(num_iterations):
        # Phase 1: generate self-play data with best network
        trainer.network = copy.deepcopy(best_network)
        for _ in range(games_per_iter):
            trainer.play_self_play_game()

        # Phase 2: train candidate network on accumulated data
        for _ in range(train_steps):
            trainer.train_step()

        # Phase 3: evaluate candidate vs. current best
        win_rate = evaluate_networks(trainer.network, best_network,
                                     game_class, eval_games)
        elo_tracker.update(win_rate, iteration=iteration)

        if win_rate > replace_threshold:
            print(f"Iter {iteration}: {win_rate:.1%} wins — replacing best.")
            best_network = copy.deepcopy(trainer.network)
        else:
            print(f"Iter {iteration}: {win_rate:.1%} wins — keeping old best.")
```

---

## Temperature scheduling

### High temperature early, low temperature late

Early in a game, the position is rich with possibilities. Many moves lead to roughly equivalent positions — the game's outcome is determined more by mid- and end-game play than by the opening move. High temperature in the opening encourages the agent to vary its play, generating games that explore many different branches of the tree. The training buffer becomes diverse.

Late in a game, positions are concrete. There may be a clearly correct move and many losing moves. Low temperature ensures the agent finds and plays the correct move, generating informative wins and losses rather than random draws from a distribution of mediocre moves.

### Code: temperature annealing

```python
import numpy as np

def compute_temperature(move_number: int, schedule: str = "step",
                        cutoff: int = 15, decay_start: int = 10,
                        decay_end: int = 40, min_temp: float = 0.05) -> float:
    """
    schedule options:
    - "step"  : τ=1 until cutoff, then τ=0 (AlphaZero style)
    - "linear": linear decay from 1.0 to min_temp between decay_start and decay_end
    - "cosine": cosine annealing for smoother decay
    """
    if schedule == "step":
        return 1.0 if move_number < cutoff else 0.0
    if move_number < decay_start:
        return 1.0
    if move_number >= decay_end:
        return min_temp
    progress = (move_number - decay_start) / (decay_end - decay_start)
    if schedule == "linear":
        return 1.0 - progress * (1.0 - min_temp)
    if schedule == "cosine":
        return min_temp + 0.5 * (1 + np.cos(np.pi * progress)) * (1.0 - min_temp)
    raise ValueError(f"Unknown schedule: {schedule}")
```

| Move | Step | Linear | Cosine |
|------|------|--------|--------|
| 0  | 1.000 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 | 1.000 |
| 15 | 0.000 | 0.750 | 0.854 |
| 20 | 0.000 | 0.500 | 0.604 |
| 30 | 0.000 | 0.050 | 0.095 |
| 40 | 0.000 | 0.050 | 0.050 |

The step schedule (AlphaZero's original approach) is simple and effective. Cosine annealing provides a smoother transition that some implementations find trains more stably.

```rust
// No external crates — uses std::f64::consts::PI for cosine annealing.

fn compute_temperature(
    move_number: usize, schedule: &str,
    cutoff: usize, decay_start: usize, decay_end: usize, min_temp: f64,
) -> f64 {
    if schedule == "step" {
        return if move_number < cutoff { 1.0 } else { 0.0 };
    }
    if move_number < decay_start { return 1.0; }
    if move_number >= decay_end  { return min_temp; }
    let progress = (move_number - decay_start) as f64 / (decay_end - decay_start) as f64;
    match schedule {
        "linear" => 1.0 - progress * (1.0 - min_temp),
        "cosine"  => {
            min_temp + 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()) * (1.0 - min_temp)
        }
        other => panic!("Unknown schedule: {}", other),
    }
}

fn main() {
    let moves = [0_usize, 10, 15, 20, 30, 40];
    println!("{:>5}  {:>7}  {:>7}  {:>7}", "Move", "Step", "Linear", "Cosine");
    for &m in &moves {
        println!(
            "{:>5}  {:>7.3}  {:>7.3}  {:>7.3}",
            m,
            compute_temperature(m, "step",   15, 10, 40, 0.05),
            compute_temperature(m, "linear", 15, 10, 40, 0.05),
            compute_temperature(m, "cosine", 15, 10, 40, 0.05),
        );
    }
}
```

---

## Evaluating training progress

### The Elo rating system

**Elo rating** is a method for estimating relative skill between players from head-to-head game results. Originally developed for chess, it is used throughout competitive game AI including AlphaZero.

The expected score (probability of winning) for player A against player B is:

\\[ E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}} \\]

**Decoding:**
- \\(R_A, R_B\\): Elo ratings for players A and B
- The denominator \\(10^{(R_B - R_A) / 400}\\) maps rating differences to win probabilities
- A rating difference of 400 means the higher-rated player wins about 91% of the time
- A difference of 200 ≈ 75% win rate; 100 ≈ 64%

After a game with actual score \\(S_A\\) (1 for win, 0.5 for draw, 0 for loss), ratings update:

\\[ R_A \leftarrow R_A + K \cdot (S_A - E_A) \\]

where K is a sensitivity constant (typically 32 for fast learning, lower for stable established ratings).

### Tournament between old and new model

Rather than comparing a single game, AlphaZero evaluates using a tournament of N games (N=400 in the original paper, smaller in practice). The new network replaces the old one only if it wins more than a threshold fraction (e.g., 55%) of games.

This threshold prevents premature replacement: a network that wins 51% of games due to variance should not immediately supplant the old best network — the improvement might be noise from a lucky batch of training data.

### Code: Elo tracker

```python
import math
from dataclasses import dataclass, field

@dataclass
class EloTracker:
    """
    Tracks Elo ratings for a sequence of AlphaZero training checkpoints.
    Each checkpoint is a new 'player'; we track their ratings over time.
    """
    initial_rating: float = 1000.0
    k_factor: float = 32.0
    ratings: list = field(default_factory=list)
    history: list = field(default_factory=list)  # list of (iteration, rating) for plotting

    def __post_init__(self):
        self.ratings.append(self.initial_rating)
        self.history.append((0, self.initial_rating))

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score (win probability) for player A vs B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, win_rate: float, iteration: int = None):
        """
        Record result of new_network vs. previous_best_network.
        win_rate: fraction of games won by the new network.
        Updates both network ratings and appends to history.
        """
        if len(self.ratings) < 1:
            self.ratings.append(self.initial_rating)

        challenger_rating = self.initial_rating  # new network starts fresh
        champion_rating   = self.ratings[-1]     # previous best

        # Expected score for challenger against champion
        expected = self.expected_score(challenger_rating, champion_rating)

        # Update challenger's effective rating
        new_rating = challenger_rating + self.k_factor * (win_rate - expected)
        self.ratings.append(new_rating)

        iter_num = iteration if iteration is not None else len(self.ratings) - 1
        self.history.append((iter_num, new_rating))

        print(f"  Elo update: challenger {challenger_rating:.0f} vs champion "
              f"{champion_rating:.0f} | win_rate={win_rate:.2%} | "
              f"new rating={new_rating:.0f}")

        return new_rating

    def should_replace(self, win_rate: float, threshold: float = 0.55) -> bool:
        """Return True if the new model is strong enough to replace the best."""
        return win_rate > threshold


# Example usage: simulate 10 training iterations with improving win rates
tracker = EloTracker(initial_rating=1000, k_factor=32)
simulated_win_rates = [0.45, 0.48, 0.52, 0.56, 0.60, 0.62, 0.65, 0.58, 0.61, 0.63]
for i, win_rate in enumerate(simulated_win_rates):
    new_rating = tracker.update(win_rate, iteration=i+1)
    print(f"    => {'REPLACE' if tracker.should_replace(win_rate) else 'keep'} best network")
```

```rust
// No external crates — pure f64 math.

fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10_f64.powf((rating_b - rating_a) / 400.0))
}

fn main() {
    let k = 32.0_f64;
    let initial = 1000.0_f64;
    let win_rates = [0.45_f64, 0.48, 0.52, 0.56, 0.60, 0.62, 0.65, 0.58, 0.61, 0.63];

    let mut champion = initial;
    for (i, &wr) in win_rates.iter().enumerate() {
        // New network always starts from initial rating
        let expected = expected_score(initial, champion);
        let new_rating = initial + k * (wr - expected);
        let replace = wr > 0.55;
        println!(
            "Iter {:>2}: challenger {:.0}  champion {:.0}  wr={:.0}%  new={:.0}  {}",
            i + 1, initial, champion, wr * 100.0, new_rating,
            if replace { "REPLACE" } else { "keep" }
        );
        // champion rating tracks the most recent checkpoint
        champion = new_rating;
    }
}
```

`10_f64.powf(x)` is Rust's equivalent of Python's `10 ** x` for floating-point exponents.

### When to replace the best model

The replacement decision is policy-dependent, but common choices:

- **AlphaZero (original):** replace whenever the new model wins > 55% of 400 evaluation games
- **Leela Chess Zero (open-source AlphaZero):** always replace with the latest checkpoint and rely on the replay buffer's recency weighting
- **Our project:** replace when win rate > 55% over 40 games (balancing evaluation cost vs. reliability)

The 55% threshold (not 50%) is important. A 50% threshold with noisy win-rate estimates means the agent oscillates between old and new networks based on training variance. The higher threshold ensures a real improvement is required before replacement.

---

## Scaling to SSA: what changes

### 1. Imperfect information: unknown fuel reserves

In real SSA, an operator tracking an adversary satellite does not know the adversary's remaining delta-v budget. A satellite with full fuel can execute many more maneuvers than one nearly depleted — this fundamentally changes the game dynamics — but from the sensor perspective, the fuel state is unobservable.

AlphaZero assumes perfect information: both players see the full state. Adapting to imperfect information requires one of:

- **Determinized search**: sample a likely fuel state for the opponent, run MCTS on that determinized game, repeat for multiple samples, aggregate results. This is the "perfect information Monte Carlo" approach. It works reasonably well when the uncertainty is over discrete unknown parameters.
- **Belief state MCTS**: represent the state as a probability distribution over possible opponent fuel levels, and run MCTS over the belief state space. More principled but much harder to implement; the belief space is continuous.
- **Information-set MCTS (ISMCTS)**: a version of MCTS designed for imperfect-information games, where nodes in the tree represent information sets rather than game states. ISMCTS is the approach used in Module 5 for poker and Module 6 for the capstone.

### 2. Stochastic transitions: orbital debris

Near-Earth space contains thousands of tracked debris objects and millions of untracked ones. A satellite executing a maneuver has a small but nonzero probability of encountering a debris field that alters its orbit stochastically. This is not the controlled stochasticity of a board game (e.g., dice in backgammon) — it is a long-tailed, rare-event distribution.

Approaches:
- **Ignore rare events (simplification):** treat debris as background risk captured in the value function's training data. Works when debris encounters are rare enough that they rarely affect game outcomes in training.
- **Domain randomization:** during self-play training, randomly inject perturbation events at varying rates. The network learns a policy robust to a range of debris environments.
- **Monte Carlo integration in PUCT:** when evaluating a leaf node, run the value network evaluation multiple times with different sampled perturbation realizations, and use the average as the leaf value. Increases compute cost per node but produces more accurate value estimates.

### 3. Large branching factor: continuous maneuver space

A satellite can apply thrust continuously in 3D. Even after discretizing to a 2D maneuvering plane, the action space is a continuous disk. A 10x10 grid discretization gives 100 actions — a branching factor that makes PUCT's exploration bonus extremely small per action, requiring many more iterations to meaningfully evaluate each option.

Approaches:

**Progressive widening:** do not expand all children at once. Start with a small random subset of actions, and add new actions as the search budget grows. The number of expanded children grows sub-linearly with visit count:

```python
def should_add_child(node, alpha: float = 0.5, k: float = 1.0) -> bool:
    """
    Progressive widening: add a new child when N(node)^alpha > k * current_children.
    alpha=0.5 means we add children proportional to sqrt(visits).
    """
    return node.N ** alpha > k * len(node.children)
```

**Policy network as action sampler:** instead of discretizing the action space, train the policy network to output parameters of a distribution (e.g., mean and variance of a Gaussian over thrust direction and magnitude). Sample from this distribution to generate candidate actions for expansion. The policy network learns to concentrate samples around good actions.

**Action abstraction:** precompute a library of strategically meaningful maneuvers (phasing orbits, Hohmann transfers, debris avoidance burns) and define the action space over this library. Reduces branching factor from thousands to tens. The key insight: in SSA, not all thrust directions are equally interesting — orbits are constrained by physics, and the set of useful maneuvers is much smaller than the set of physically possible ones.

---

## Key Takeaways

- **The self-play data pipeline** consists of three interlocking components — game buffer, self-play workers, and training process — with the game buffer acting as the decoupling layer that allows each component to run at its own rate.
- **Temperature scheduling** (high \\(\tau\\) early in the game, low \\(\tau\\) late) is essential for self-play: early moves need diversity to fill the buffer with varied training examples, while late moves need precision to generate informative wins and losses.
- **Elo rating** provides a principled, interpretable measure of training progress: by pitting each new checkpoint against the previous best, you track not just loss curves but actual head-to-head improvement, preventing false positives from noisy training metrics.
- **The replacement threshold** (typically 55%) ensures the self-play loop replaces the best network only when there is genuine, statistically meaningful improvement, preventing training instability from noise-driven oscillation.
- **Imperfect information** (unknown fuel reserves in SSA) breaks AlphaZero's perfect-information assumption; extensions like determinized search or information-set MCTS are needed, and this is the direct bridge to Module 5's CFR-based approaches.
- **Continuous action spaces** (real satellite maneuvers) require progressive widening or distribution-parameterized policy networks to keep the effective branching factor tractable — the core engineering challenge when applying AlphaZero to realistic SSA scenarios.

## Quiz

{{#quiz 04-alphazero-self-play.toml}}
