# Module 4 Project: An AlphaZero-Lite Agent for Pursuit-Evasion


<!-- toc -->

## What you are building

You will train an AlphaZero-style agent on a simplified two-player pursuit-evasion game between spacecraft. One spacecraft (the evader) is trying to traverse a region of space without being detected. The other (the defender, equipped with a sensor) is trying to keep eyes on the evader. The evader can take evasive maneuvers; the defender can change sensor pointing direction. After a fixed number of moves, the defender wins if the evader was detected often enough; otherwise the evader wins.

This is a turn-based, two-player, zero-sum game. It is small enough to train on a laptop in 30-60 minutes. It is large enough that:
- Pure MCTS without a network is bad (the branching factor compounds quickly)
- A naive policy network without search is bad (no lookahead)
- AlphaZero (network + search trained by self-play) is much better than either

## The scenario

The state space:
- The evader's position on a 5x5 grid
- The defender's sensor pointing direction (one of 8 directions)
- The number of moves remaining

Each turn:
- The current player (alternating evader and defender) makes a move
- The evader chooses one of 9 actions: move to one of the 8 adjacent cells or stay still
- The defender chooses one of 8 actions: point the sensor in one of the 8 compass directions

Detection mechanic:
- The defender's sensor sees a 1-cell-wide cone in the chosen direction (3 cells total: directly forward and the two adjacent cells)
- If the evader is in the visible cone after the defender's move, the evader is detected this turn

The game ends after 20 moves. Winner: defender if the evader was detected on more than half the defender's turns; evader otherwise.

This is small enough (about 25 × 8 × 20 ≈ 4,000 distinct states) that AlphaZero can master it quickly.

## Step 1: define the game in OpenSpiel

The game lives at the scale where defining it via OpenSpiel's Python API is reasonable. Same scaffolding pattern as the Module 3 project.

```python
"""
pursuit_evasion.py: a small two-player pursuit-evasion game.
"""

import numpy as np
import pyspiel

GRID_SIZE = 5
NUM_SENSOR_DIRS = 8
MAX_TURNS = 20
DETECTION_THRESHOLD = 0.5  # defender wins if detection rate > this

EVADER, DEFENDER = 0, 1
EVADER_ACTIONS  = list(range(9))   # 8 directions + stay
DEFENDER_ACTIONS = list(range(NUM_SENSOR_DIRS))

# Direction offsets: 0=N, 1=NE, 2=E, ..., 7=NW
DX = [ 0, 1, 1, 1, 0, -1, -1, -1]
DY = [-1, -1, 0, 1, 1,  1,  0, -1]

class PursuitEvasionGame(pyspiel.Game):
    def __init__(self, params=None):
        game_type = pyspiel.GameType(
            short_name="pursuit_evasion",
            long_name="Pursuit Evasion 5x5",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=2,
            min_num_players=2,
            provides_information_state_string=False,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=True,
            parameter_specification={},
        )
        game_info = pyspiel.GameInfo(
            num_distinct_actions=max(len(EVADER_ACTIONS), len(DEFENDER_ACTIONS)),
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            max_game_length=MAX_TURNS * 2,
        )
        super().__init__(game_type, game_info, params or {})
    
    def new_initial_state(self):
        return PursuitEvasionState(self)


class PursuitEvasionState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        # Evader starts at center
        self.ex = GRID_SIZE // 2
        self.ey = GRID_SIZE // 2
        self.sensor_dir = 0  # initial direction (does not really matter)
        self.turns_remaining = MAX_TURNS
        self.player = EVADER  # evader moves first
        self.detections = 0
        self.defender_turns = 0  # for computing detection rate
        self._terminal = False
    
    def current_player(self):
        if self._terminal:
            return pyspiel.PlayerId.TERMINAL
        return self.player
    
    def legal_actions(self, player=None):
        if self._terminal:
            return []
        if self.player == EVADER:
            return list(EVADER_ACTIONS)
        return list(DEFENDER_ACTIONS)
    
    def _apply_action(self, action):
        if self.player == EVADER:
            # Action 0-7: move in direction; action 8: stay still
            if action < 8:
                new_x = max(0, min(GRID_SIZE - 1, self.ex + DX[action]))
                new_y = max(0, min(GRID_SIZE - 1, self.ey + DY[action]))
                self.ex, self.ey = new_x, new_y
            # action 8: stay still
            self.player = DEFENDER
        else:
            # Defender picks a sensor direction
            self.sensor_dir = action
            self.defender_turns += 1
            
            # Check detection: evader is in 3-cell cone from defender's "position" (center)
            # Simplification: defender is always at center; sensor points in chosen direction
            cx = GRID_SIZE // 2
            cy = GRID_SIZE // 2
            visible_cells = self._cone_cells(cx, cy, self.sensor_dir)
            if (self.ex, self.ey) in visible_cells:
                self.detections += 1
            
            self.turns_remaining -= 1
            self.player = EVADER
            
            if self.turns_remaining == 0:
                self._terminal = True
    
    def _cone_cells(self, cx, cy, direction):
        """Return the 3 cells the sensor can see when pointed in this direction."""
        cells = set()
        # Forward cell
        for r in range(1, 4):  # 3 cells out in the chosen direction
            x = cx + r * DX[direction]
            y = cy + r * DY[direction]
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                cells.add((x, y))
        return cells
    
    def returns(self):
        if not self._terminal:
            return [0.0, 0.0]
        detection_rate = self.detections / max(1, self.defender_turns)
        if detection_rate > DETECTION_THRESHOLD:
            # Defender wins
            return [-1.0, +1.0]  # [evader, defender]
        else:
            return [+1.0, -1.0]
    
    def is_terminal(self):
        return self._terminal
    
    def observation_tensor(self, player=0):
        # State features:
        #   - one-hot evader position (25 features)
        #   - one-hot sensor direction (8 features)
        #   - turns remaining normalized (1 feature)
        #   - whose turn it is (1 feature)
        evader_onehot = np.zeros(GRID_SIZE * GRID_SIZE)
        evader_onehot[self.ey * GRID_SIZE + self.ex] = 1
        sensor_onehot = np.zeros(NUM_SENSOR_DIRS)
        sensor_onehot[self.sensor_dir] = 1
        return np.concatenate([
            evader_onehot,
            sensor_onehot,
            [self.turns_remaining / MAX_TURNS],
            [float(self.player)],
        ]).astype(np.float32)
    
    def observation_string(self, player=0):
        return (f"evader=({self.ex},{self.ey}), sensor={self.sensor_dir}, "
                f"turns_left={self.turns_remaining}, player={self.player}")
```

The state vector has 25 + 8 + 1 + 1 = 35 features. The action space has 9 actions (max of the two players' action spaces).

## Step 2: build the AlphaZero network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 9

class AlphaZeroNetwork(nn.Module):
    def __init__(self, state_dim=35, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head  = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features)).squeeze(-1)
        return policy_logits, value
```

## Step 3: implement neural-guided MCTS

Use the implementation from lesson 3, adapted to use OpenSpiel's `pyspiel.State` interface. The key methods you need: `legal_actions()`, `apply_action(a)`, `is_terminal()`, `returns()`, `current_player()`, `observation_tensor()`.

A subtle detail: when backpropagating the value, you need to account for whose perspective the value is from. The value network outputs a value for the current player at the queried state. When backpropagating up the tree, flip the sign at every level (because the player alternates).

```python
import math
from copy import deepcopy

class MCTSNode:
    def __init__(self, state, prior=0.0, parent=None, action_taken=None):
        self.state = state  # reference, not deep copy yet
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.expanded = False
    
    def Q(self):
        return self.W / self.N if self.N > 0 else 0
    
    def puct_score(self, parent_N, c=1.5):
        return self.Q() + c * self.prior * math.sqrt(parent_N) / (1 + self.N)


def mcts_search(root_state, network, num_iterations=100, c=1.5):
    root = MCTSNode(deepcopy(root_state))
    
    # Initialize root
    legal = root.state.legal_actions()
    state_t = torch.tensor(root.state.observation_tensor(), dtype=torch.float32)
    with torch.no_grad():
        policy_logits, _ = network(state_t)
    legal_logits = policy_logits[legal]
    priors = F.softmax(legal_logits, dim=0).tolist()
    for action, prior in zip(legal, priors):
        next_state = deepcopy(root.state)
        next_state.apply_action(action)
        root.children[action] = MCTSNode(
            next_state, prior=prior, parent=root, action_taken=action
        )
    root.expanded = True
    
    for _ in range(num_iterations):
        # Selection
        node = root
        path = [node]
        while node.expanded and not node.state.is_terminal():
            best_action = max(
                node.children,
                key=lambda a: node.children[a].puct_score(node.N + 1, c)
            )
            node = node.children[best_action]
            path.append(node)
        
        # Expansion + evaluation
        if node.state.is_terminal():
            value = node.state.returns()[node.state.current_player() if node.state.current_player() >= 0 else 0]
            # Actually, when terminal, returns() gives both players' utilities.
            # We want the value from the perspective of the player at the parent.
            # Simpler: just evaluate from the parent's perspective.
            parent_player = path[-2].state.current_player() if len(path) >= 2 else 0
            value = node.state.returns()[parent_player]
        else:
            legal = node.state.legal_actions()
            state_t = torch.tensor(node.state.observation_tensor(), dtype=torch.float32)
            with torch.no_grad():
                policy_logits, value_pred = network(state_t)
            legal_logits = policy_logits[legal]
            priors = F.softmax(legal_logits, dim=0).tolist()
            for action, prior in zip(legal, priors):
                next_state = deepcopy(node.state)
                next_state.apply_action(action)
                node.children[action] = MCTSNode(
                    next_state, prior=prior, parent=node, action_taken=action
                )
            node.expanded = True
            value = value_pred.item()
        
        # Backpropagation
        # value is from the perspective of the player at `node`.
        # Walk back up; sign flips at each level.
        for n in reversed(path):
            n.N += 1
            n.W += value
            value = -value
    
    return root
```

This is a simplified implementation. Production-grade versions handle terminal-state value backup more carefully.

## Step 4: self-play training loop

```python
import random
from collections import deque

class AlphaZeroTrainer:
    def __init__(self, network, lr=1e-3):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=10_000)
    
    def play_game(self, mcts_iterations=80):
        game = PursuitEvasionGame()
        state = game.new_initial_state()
        examples = []
        move_count = 0
        
        while not state.is_terminal():
            root = mcts_search(state, self.network, num_iterations=mcts_iterations)
            
            # Compute search policy
            visits = np.zeros(NUM_ACTIONS)
            for action, child in root.children.items():
                visits[action] = child.N
            policy = visits / visits.sum()
            
            state_t = torch.tensor(state.observation_tensor(), dtype=torch.float32)
            examples.append((state_t, policy, state.current_player()))
            
            # Select action: temperature 1 for first few moves, then greedy
            if move_count < 10:
                action_probs = visits / visits.sum()
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                action = int(np.argmax(visits))
            
            state.apply_action(action)
            move_count += 1
        
        # Game over: assign values
        outcome = state.returns()
        for state_t, policy, player in examples:
            value = outcome[player]
            self.replay_buffer.append((state_t, policy, value))
        
        return outcome[0]  # evader's utility
    
    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None
        batch = random.sample(self.replay_buffer, batch_size)
        states, policies, values = zip(*batch)
        
        states = torch.stack(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        
        policy_logits, value_preds = self.network(states)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_preds, values)
        
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, num_iterations=50, games_per_iter=10, train_steps_per_iter=50):
        for it in range(num_iterations):
            outcomes = []
            for _ in range(games_per_iter):
                outcomes.append(self.play_game())
            avg_evader_outcome = np.mean(outcomes)
            
            losses = []
            for _ in range(train_steps_per_iter):
                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)
            avg_loss = np.mean(losses) if losses else 0.0
            
            print(f"Iteration {it+1:3d}: avg loss = {avg_loss:.4f}, "
                  f"evader avg utility = {avg_evader_outcome:+.2f}, "
                  f"buffer = {len(self.replay_buffer)}")
```

## Step 5: train and evaluate

```python
# Initialize
network = AlphaZeroNetwork(state_dim=35, hidden_dim=128)
trainer = AlphaZeroTrainer(network, lr=1e-3)

# Train (this takes 30-60 minutes)
trainer.train(num_iterations=50, games_per_iter=10)

# Evaluate against random play
def random_play_evaluation(network, num_games=50):
    """Play trained AlphaZero against random opponents."""
    wins_as_evader = 0
    wins_as_defender = 0
    
    for game_idx in range(num_games):
        game = PursuitEvasionGame()
        state = game.new_initial_state()
        # AlphaZero plays as one player, random as the other
        az_player = game_idx % 2
        
        while not state.is_terminal():
            if state.current_player() == az_player:
                root = mcts_search(state, network, num_iterations=80)
                action = max(root.children, key=lambda a: root.children[a].N)
            else:
                action = random.choice(state.legal_actions())
            state.apply_action(action)
        
        outcome = state.returns()[az_player]
        if outcome > 0:
            if az_player == EVADER:
                wins_as_evader += 1
            else:
                wins_as_defender += 1
    
    print(f"Wins as evader:   {wins_as_evader}/{num_games // 2}")
    print(f"Wins as defender: {wins_as_defender}/{num_games // 2}")

random_play_evaluation(network)
```

A well-trained AlphaZero agent should beat random play more than 80% of the time as either side.

## Step 6: reflect

1. Did the network learn? Plot the average loss over training iterations.
2. Did self-play produce balanced games (roughly 50-50 outcomes)? If one side always won, was it actually optimal play, or was the agent stuck in a local optimum?
3. How does increasing MCTS iterations per move affect both training time and final performance?
4. Can you visualize what the network learned? For example, given a fixed defender position, plot the value network's predictions for each evader position.
5. The game is very small. What changes if you scale to a 7x7 grid? An 11x11 grid? At what scale does the network start to underfit?

## What you have built

- A custom two-player game in OpenSpiel
- A neural-guided MCTS implementation
- A complete AlphaZero self-play training loop
- An agent that learned a non-trivial pursuit-evasion strategy from scratch

This is the foundation for the Module 8 capstone, where you will build something similar in Rust. The conceptual structure is the same; the implementation is what differs.

## What's next

Module 5 introduces game theory and CFR. Pursuit-evasion games are perfect-information; many real SSA scenarios are not. Two adversaries with hidden information (e.g., one cannot see what the other is doing) need a different framework. CFR is the algorithm that solves these.
