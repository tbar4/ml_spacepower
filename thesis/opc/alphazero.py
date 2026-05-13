"""AlphaZero self-play training loop for OPC.

The training cycle follows the original AlphaZero paper:
  1. Self-play: generate game episodes using MCTS + current network
  2. Train: update the network on (obs, policy_target, value_target) tuples
  3. Evaluate: track mean returns and policy entropy over time
  4. Repeat

Both players (attacker and defender) use the SAME network. Each player feeds
their own observation tensor to the network and gets their own perspective's
policy and value. This is a shared-network approach appropriate for a
zero-sum game where both players play from a common strategy pool.

Key design choice: we train on observations (not true states), so the learned
policy operates under the same imperfect information the player faces in play.
"""

import copy
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from opc.game import OPCGame, _N_JOINT, _OBS_SIZE, ATTACKER_ID, DEFENDER_ID, T_MAX
from opc.mcts import MCTS, TEMPERATURE_MOVES
from opc.networks import OPCNetwork, build_network


# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    n_iterations: int = 50          # outer AlphaZero iterations
    games_per_iteration: int = 20   # self-play games per iteration
    n_mcts_simulations: int = 100   # MCTS simulations per move
    replay_buffer_size: int = 50_000
    batch_size: int = 512
    n_train_steps: int = 200        # gradient steps per iteration
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0  # relative weighting of value vs policy loss
    temperature_threshold: int = TEMPERATURE_MOVES
    eval_games: int = 20            # games played for evaluation each iteration
    checkpoint_every: int = 5       # save checkpoint every N iterations
    checkpoint_path: str = "checkpoints/opc_net.pt"


# ── Experience tuple ──────────────────────────────────────────────────────────

@dataclass
class Experience:
    obs: np.ndarray          # observation tensor for the acting player
    policy_target: np.ndarray  # MCTS visit-count policy
    value_target: float      # outcome from the acting player's perspective


# ── Self-play episode ─────────────────────────────────────────────────────────

def play_episode(
    game: OPCGame,
    network: OPCNetwork,
    n_simulations: int,
    temperature_threshold: int,
) -> list[Experience]:
    """Play one self-play game and return a list of Experience objects."""
    mcts = MCTS(network, n_simulations=n_simulations)
    state = game.new_initial_state()
    move_num = 0

    trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []
    # trajectory entries: (obs_tensor, policy_target, acting_player)

    while not state.is_terminal():
        player = state.current_player()
        temperature = 1.0 if move_num < temperature_threshold else 0.0

        visit_counts = mcts.run(state, add_dirichlet_noise=True)
        policy_target = MCTS.visit_counts_to_policy(
            visit_counts, temperature=temperature
        )

        obs = np.array(state.observation_tensor(player), dtype=np.float32)
        trajectory.append((obs, policy_target, player))

        action = int(np.random.choice(len(policy_target), p=policy_target))
        state.apply_action(action)
        move_num += 1

    returns = state.returns()   # [attacker_return, defender_return]

    experiences = []
    for obs, policy_target, player in trajectory:
        experiences.append(Experience(
            obs=obs,
            policy_target=policy_target,
            value_target=float(returns[player]),
        ))

    return experiences


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int):
        self._buffer: deque[Experience] = deque(maxlen=max_size)

    def add(self, experiences: list[Experience]):
        self._buffer.extend(experiences)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        obs = torch.tensor(np.stack([e.obs for e in batch]), dtype=torch.float32)
        policy = torch.tensor(np.stack([e.policy_target for e in batch]), dtype=torch.float32)
        value = torch.tensor([[e.value_target] for e in batch], dtype=torch.float32)
        return obs, policy, value

    def __len__(self):
        return len(self._buffer)


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    network: OPCNetwork,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_loss_weight: float = 1.0,
) -> tuple[float, float]:
    """One gradient step. Returns (policy_loss, value_loss)."""
    network.train()
    optimizer.zero_grad()

    logits, value_pred = network(obs)

    # Policy loss: cross-entropy between MCTS policy and network policy
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()

    # Value loss: MSE between predicted and actual outcome
    value_loss = F.mse_loss(value_pred, value_target)

    loss = policy_loss + value_loss_weight * value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()

    return policy_loss.detach().item(), value_loss.detach().item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    game: OPCGame,
    network: OPCNetwork,
    n_games: int,
    n_simulations: int = 50,
) -> dict:
    """Play n_games with the network and return summary stats."""
    network.eval()
    atk_wins = 0
    total_turns = []
    policy_entropies = []

    mcts = MCTS(network, n_simulations=n_simulations)

    for _ in range(n_games):
        state = game.new_initial_state()
        turn = 0
        episode_entropies = []

        while not state.is_terminal():
            player = state.current_player()
            obs = torch.tensor(state.observation_tensor(player), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = network(obs)
                probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

            entropy = -np.sum(probs * np.log(probs + 1e-9))
            episode_entropies.append(entropy)

            action = int(np.argmax(probs))   # greedy at eval time
            state.apply_action(action)
            turn += 1

        returns = state.returns()
        if returns[ATTACKER_ID] > 0:
            atk_wins += 1
        total_turns.append(turn // 2)   # turns = moves / 2
        policy_entropies.extend(episode_entropies)

    return {
        "attacker_win_rate": atk_wins / n_games,
        "mean_episode_turns": float(np.mean(total_turns)),
        "mean_policy_entropy": float(np.mean(policy_entropies)),
    }


# ── Main training loop ────────────────────────────────────────────────────────

class AlphaZeroTrainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.game = OPCGame()
        self.network = build_network()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.n_iterations,
        )
        self.buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.history: list[dict] = []

    def run(self):
        """Run the full AlphaZero training loop."""
        cfg = self.config
        print(f"Starting AlphaZero training: {cfg.n_iterations} iterations, "
              f"{cfg.games_per_iteration} games/iter, "
              f"{cfg.n_mcts_simulations} MCTS sims/move")

        for iteration in range(1, cfg.n_iterations + 1):
            t_start = time.time()

            # ── Self-play ─────────────────────────────────────────────────────
            self.network.eval()
            new_experiences = 0
            for _ in range(cfg.games_per_iteration):
                episodes = play_episode(
                    self.game,
                    self.network,
                    cfg.n_mcts_simulations,
                    cfg.temperature_threshold,
                )
                self.buffer.add(episodes)
                new_experiences += len(episodes)

            # ── Training ──────────────────────────────────────────────────────
            if len(self.buffer) < cfg.batch_size:
                print(f"  Iter {iteration}: buffer too small ({len(self.buffer)}), skipping train")
                continue

            self.network.train()
            policy_losses, value_losses = [], []
            for _ in range(cfg.n_train_steps):
                obs, policy, value = self.buffer.sample(cfg.batch_size)
                pl, vl = train_step(
                    self.network, self.optimizer, obs, policy, value,
                    cfg.value_loss_weight,
                )
                policy_losses.append(pl)
                value_losses.append(vl)

            self.scheduler.step()

            # ── Evaluation ────────────────────────────────────────────────────
            self.network.eval()
            metrics = evaluate(self.game, self.network, cfg.eval_games, n_simulations=50)
            metrics["iteration"] = iteration
            metrics["policy_loss"] = float(np.mean(policy_losses))
            metrics["value_loss"] = float(np.mean(value_losses))
            metrics["buffer_size"] = len(self.buffer)
            metrics["elapsed_s"] = time.time() - t_start
            self.history.append(metrics)

            print(
                f"  Iter {iteration:3d} | "
                f"atk_win={metrics['attacker_win_rate']:.2f} | "
                f"entropy={metrics['mean_policy_entropy']:.2f} | "
                f"turns={metrics['mean_episode_turns']:.1f} | "
                f"p_loss={metrics['policy_loss']:.3f} | "
                f"v_loss={metrics['value_loss']:.3f} | "
                f"buf={metrics['buffer_size']} | "
                f"t={metrics['elapsed_s']:.1f}s"
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            if iteration % cfg.checkpoint_every == 0:
                self._save_checkpoint(iteration)

        print("Training complete.")
        return self.history

    def _save_checkpoint(self, iteration: int):
        import os
        os.makedirs(os.path.dirname(self.config.checkpoint_path), exist_ok=True)
        torch.save({
            "iteration": iteration,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, self.config.checkpoint_path)
        print(f"    Checkpoint saved at iteration {iteration}")
