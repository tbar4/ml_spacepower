# Lesson 4: AlphaZero Self-Play

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

## Quiz

{{#quiz 04-alphazero-self-play.toml}}
