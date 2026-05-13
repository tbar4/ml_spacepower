"""Entry point for OPC AlphaZero training.

Usage:
    venv/bin/python3.12 train.py                  # default config
    venv/bin/python3.12 train.py --quick          # smoke test (2 iters)
    venv/bin/python3.12 train.py --n-iterations 100 --games-per-iter 50
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opc.alphazero import AlphaZeroTrainer, TrainingConfig


def parse_args():
    p = argparse.ArgumentParser(description="OPC AlphaZero Training")
    p.add_argument("--quick", action="store_true", help="Smoke test: 2 iterations, 2 games")
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=20)
    p.add_argument("--mcts-sims", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--checkpoint", type=str, default="checkpoints/opc_net.pt")
    p.add_argument("--history-out", type=str, default="training_history.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        config = TrainingConfig(
            n_iterations=2,
            games_per_iteration=2,
            n_mcts_simulations=10,
            n_train_steps=5,
            eval_games=2,
            checkpoint_every=2,
            batch_size=32,
        )
    else:
        config = TrainingConfig(
            n_iterations=args.n_iterations,
            games_per_iteration=args.games_per_iter,
            n_mcts_simulations=args.mcts_sims,
            batch_size=args.batch_size,
            n_train_steps=args.train_steps,
            checkpoint_path=args.checkpoint,
        )

    trainer = AlphaZeroTrainer(config)
    history = trainer.run()

    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {args.history_out}")


if __name__ == "__main__":
    main()
