# Lesson 1: OpenSpiel Architecture

## Where this fits

You have used OpenSpiel piecemeal across several modules: defining a single-agent MDP in Module 3, running CFR on a built-in game in Module 5, and so on. Now we step back and look at the framework as a whole. The goal of this lesson is for you to come away with a clear mental model of OpenSpiel's abstractions, so that when you are reading the source or extending it, you know which file to look at and which class to subclass. This also informs how we design the Rust capstone: many of OpenSpiel's design choices are forced by the structure of the problem, so the Rust version will end up looking similar in places.

## The core abstractions

OpenSpiel's design rests on three central abstractions, plus some secondary ones. If you understand the central three, you can find your way around everything else.

### Game

A `Game` represents the rules of a game. It has no mutable state. You can think of it as the "type definition" or "schema": it describes what kinds of states exist, what actions are legal, what the players' utilities can be, and so on.

A `Game` exposes:
- `new_initial_state()`: produces a fresh starting state
- `num_distinct_actions()`: the size of the action space
- `num_players()`: how many players (chance is not counted as a player; it is handled separately)
- `min_utility()`, `max_utility()`: bounds on returns
- Type metadata: chance mode, information mode, dynamics, reward model

You define a game once. You can have many states derived from it during a session.

### State

A `State` represents a particular position in the game. It has mutable state internally and changes when actions are applied. This is the workhorse class.

A `State` exposes:
- `current_player()`: whose turn is it? Returns a player index, or one of the special sentinels: `CHANCE` (a stochastic event happens here), `TERMINAL` (the game is over), or `SIMULTANEOUS` (multiple players act at once).
- `legal_actions()`: which actions can the current player take?
- `apply_action(action)`: advance the state by one move
- `is_chance_node()`, `chance_outcomes()`: for stochastic events, what are the possible outcomes and their probabilities?
- `is_terminal()`, `returns()`: when the game ends, what utilities did each player receive?
- `clone()`: produce a copy. Important because algorithms like MCTS need to explore hypothetical futures without mutating the real state.

A typical algorithm interacts with the game by repeatedly checking `current_player()`, then either:
- calling `apply_action` with a chosen action (if a player or chance node)
- collecting `returns()` (if terminal)

### Observer / ObservationTensor / InformationStateTensor

This is where it gets interesting and where many people get confused.

In a perfect-information game (chess), the "state" is the same as what either player sees: both players see the full board. There is no distinction between the world's state and any player's view of it.

In an imperfect-information game (poker), the world's state contains things like the opponent's hidden cards, but each player only observes their own hand. The state describes the world; the **observation** describes what one player can see.

OpenSpiel handles this via two related concepts:

**Observation**: what a player sees at a given moment in time. This is a Markovian summary, in the sense that the observation reflects the current world state but not the player's history of what they have seen and done.

**Information state**: the full history of observations the player has made so far. This is what is conceptually relevant for game-theoretic algorithms like CFR, because two world states that produce the same history of observations for a player are indistinguishable to that player; they belong to the same "information set."

The framework provides both as strings (human-readable) and as tensors (machine-readable, suitable for neural network input). For an imperfect-information game you must implement at minimum the information state representations; the observation representations are optional but commonly provided.

The Observer abstraction, introduced more recently in OpenSpiel, is a more flexible way of producing observation tensors with configurable contents. It is used in newer game implementations and the Python algorithm modules. Older code uses the `observation_tensor` and `information_state_tensor` methods directly on `State`. Both work; new game implementations should use the Observer pattern.

## The directory structure

OpenSpiel has roughly this top-level layout (paths simplified):

```
open_spiel/
├── games/                   # C++ game implementations (chess, go, poker, ...)
├── algorithms/              # C++ algorithm implementations (MCTS, CFR, ...)
├── python/
│   ├── games/               # Pure-Python games
│   ├── algorithms/          # Python algorithm implementations
│   ├── examples/            # Example scripts using the framework
│   └── pybind11/            # The C++ to Python binding glue
├── integration_tests/       # Tests that exercise games against the API contract
└── docs/                    # Markdown documentation
```

The C++ and Python layers are kept in sync. Most algorithms are implemented in both. The Python algorithms are usually slower but easier to read and modify; they are what you should look at first when learning. The C++ versions are for production use.

If you want to see how an algorithm works, start in `python/algorithms/`. If you want to see how a particular game is implemented, look at `games/<game_name>.cc` for the C++ version or `python/games/<game_name>.py` for any Python version.

## How an algorithm uses the API

Here is the structure of a generic OpenSpiel algorithm:

```python
import pyspiel

def my_algorithm(game):
    state = game.new_initial_state()
    
    while not state.is_terminal():
        if state.is_chance_node():
            # Resolve chance: sample an outcome from the distribution
            outcomes = state.chance_outcomes()  # list of (action, prob)
            action, prob = sample_from(outcomes)
            state.apply_action(action)
        else:
            current_player = state.current_player()
            legal = state.legal_actions()
            
            # The algorithm's main logic: pick an action somehow
            action = my_action_selection(state, current_player, legal)
            
            state.apply_action(action)
    
    # Game over: collect utilities
    returns = state.returns()  # list of utilities per player
    return returns
```

This is the universal interaction pattern. Every algorithm in OpenSpiel, from random play up to AlphaZero and PSRO, follows this loop in some form (often with cloning the state to look ahead, batching across many parallel states, or interleaving with neural network calls).

## Two things that surprise newcomers

**Chance is not a player.** Chance nodes are a separate concept. `current_player()` returns the special value `pyspiel.PlayerId.CHANCE` (= -1) at chance nodes. You handle them by sampling from `chance_outcomes()` and applying the sampled action. This separation matters because algorithms like CFR treat chance differently from player decisions (CFR averages over chance outcomes; it does not search over them).

**Returns are utilities, not rewards.** In a typical RL framework, you observe a reward at every step. In OpenSpiel, the standard reward model returns utilities only at the end of the game (`reward_model = TERMINAL`). This is the natural model for games like chess (you win, lose, or draw at the end). Some games support per-step rewards (`reward_model = REWARDS`), in which case `state.rewards()` returns the per-player reward at the current step. The capstone game uses the terminal reward model.

## Bots and runtime tournaments

A `Bot` is an interface for an agent that plays a game. The interface is just one method:

```python
class Bot:
    def step(self, state) -> int:
        """Return the action this bot wants to take in the given state."""
```

OpenSpiel provides bots for many algorithms (random, MCTS, AlphaZero, etc.) and a `play_game.py` style tool that can run a tournament between bots. This is useful for evaluation: train a bot offline, then plug it into the bot interface and have it play against baselines.

The capstone will define a Bot wrapper around our trained CFR strategy, so we can play example games and watch the strategy in action.

## Observers, in slightly more detail

Modern OpenSpiel code uses the Observer abstraction, which is a way to specify what kind of observation you want and have the framework produce it consistently across games.

A typical use:

```python
from open_spiel.python.observation import make_observation

game = pyspiel.load_game("kuhn_poker")
obs = make_observation(game)

state = game.new_initial_state()
# ... advance state ...
obs.set_from(state, player=0)
print(obs.tensor)        # numpy array suitable for neural network input
print(obs.string_from(state, player=0))  # human-readable
```

The Observer is parameterized: you can ask for "perfect" information observation (everything visible), "private" (only your own information), and various combinations. Different algorithms need different observation types; the Observer abstraction lets you specify what you need without modifying the game class.

For the capstone, since we are writing the CFR solver from scratch, we will use a simple string-based information state representation directly on the state class, not the Observer abstraction. This is simpler and well-suited to small CFR examples.

## The pieces we will reuse in the capstone

The Rust capstone is going to mirror this architecture, scaled down:

- A `Game` trait: rules of the game, no state
- A `State` struct: mutable state, the workhorse
- An `InformationState` representation: a string per (player, history) that uniquely identifies what the player knows
- A solver that, like OpenSpiel's algorithms, iterates over states and accumulates statistics

The capstone does not need the bot abstraction (we are not running tournaments) or the Observer abstraction (we use simple information state strings). But the Game/State separation, the player/chance distinction, the terminal-utility model, and the information-state-as-string idiom are all coming from OpenSpiel and will reappear in our Rust code.

## The game interface in detail

Understanding every method a game must implement is necessary before you can write a custom game or evaluate an existing one. The table below covers the complete set. Methods marked "required" cause algorithm failures if absent; methods marked "optional" allow certain algorithms to degrade gracefully or skip certain features.

| Method | Required? | What it returns | Why algorithms need it |
|--------|-----------|-----------------|------------------------|
| `num_players()` | Required | int | Determines size of utility vectors everywhere |
| `num_distinct_actions()` | Required | int | Sizes the strategy table and action-value networks |
| `max_game_length()` | Required | int | Needed for fixed-length tensor representations; also bounds search depth |
| `min_utility()` / `max_utility()` | Required | float | Normalizes utilities for algorithms that work in [0,1] (AlphaZero) |
| `new_initial_state()` | Required | State | Entry point for every algorithm |
| `information_state_tensor_shape()` | Required for NN | list[int] | Neural network input layer size |
| `observation_tensor_shape()` | Optional | list[int] | Some actors only need Markovian observations |
| `get_type()` | Required | GameType | Algorithm routing: is this a chance game? Imperfect info? |
| `make_observer()` | Optional | Observer | For configurable observation generation |
| `deserialize_state()` | Optional | State | Distributed training, replay buffers |

### Why standardization matters for plug-and-play algorithms

Consider what CFR needs to run on any game: it must know how many players there are, how many information sets to expect, what actions are legal at each node, and what the utilities are. OpenSpiel's interface provides all of this through a handful of method calls with consistent semantics. An algorithm implemented against this interface works on Kuhn poker, Leduc Hold'em, a custom SSA game, or any future game you write — without modification.

This is the core value of the standardization. In practice it means:

```python
# This exact code runs CFR on ANY game that implements the interface:
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

def run_cfr_on_any_game(game_name: str, iterations: int = 1000):
    game = pyspiel.load_game(game_name)
    solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    policy = solver.average_policy()
    exp = exploitability.exploitability(game, policy)
    print(f"[{game_name}] exploitability after {iterations} iterations: {exp:.6f}")
    return policy

# All of these work with zero modification to run_cfr_on_any_game:
run_cfr_on_any_game("kuhn_poker")
run_cfr_on_any_game("leduc_poker")
run_cfr_on_any_game("liars_dice")
# And once you register your game:
run_cfr_on_any_game("mini_maneuver")
```

This plug-and-play property is not accidental; it is the point. Every method in the interface exists to serve at least one algorithm in the zoo.

### The `information_state_tensor_shape` method

This method deserves extra attention because it is easy to get wrong and hard to debug. The shape must be a flat list giving the dimensions of the tensor. For a game with two bits of private information and a three-step public history, you might return `[2 + 3] = [5]` for a flat vector, or `[2, 5]` for a 2D tensor.

The shape must be **consistent** across all states and all players. CFR variants that use neural networks (deep CFR) pre-allocate a fixed-size input array, so a shape mismatch causes a silent indexing error rather than an explicit exception. The integration tests catch this, which is why you should always run them after implementing a new game.

For the SSA context: when designing a game where a space operator's information state encodes orbital parameters (semi-major axis, eccentricity, inclination) as a feature vector, the tensor shape determines how expressive the network can be. Too small and the network cannot distinguish operationally important situations; too large and training data becomes sparse.

## Running algorithms on existing games

Before writing custom games it is worth seeing how algorithms plug into the existing game library. The following three examples illustrate the pattern for CFR, MCTS, and AlphaZero respectively.

### CFR on Kuhn poker

Kuhn poker is a simplified one-card poker game that is the canonical benchmark for CFR. Its Nash equilibrium has an analytical solution, so you can verify your solver converges to the right answer.

```python
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

game = pyspiel.load_game("kuhn_poker")
solver = cfr.CFRSolver(game)

print("Iteration | Exploitability")
print("-" * 30)
for i in range(0, 1001, 100):
    if i > 0:
        for _ in range(100):
            solver.evaluate_and_update_policy()
    policy = solver.average_policy()
    exp = exploitability.exploitability(game, policy)
    print(f"{i:9d} | {exp:.8f}")

# Kuhn poker Nash equilibrium has exploitability 0.0 at convergence.
# After 1000 iterations you should see something < 0.01.
```

**What you should see:** exploitability starts around 0.4-0.5 (uniform random play), drops quickly in the first few hundred iterations, and approaches zero. The exact rate depends on which CFR variant you use. Vanilla CFR converges as O(1/sqrt(T)).

### MCTS on Tic-Tac-Toe

MCTS is a Monte Carlo tree search algorithm appropriate for perfect-information games. Tic-Tac-Toe is a solved game (always draw with optimal play), so MCTS should find the draw if given enough simulations.

```python
import pyspiel
from open_spiel.python.algorithms import mcts

game = pyspiel.load_game("tic_tac_toe")
evaluator = mcts.RandomRolloutEvaluator(n_rollouts=4, random_state=42)

bot = mcts.MCTSBot(
    game,
    uct_c=1.5,               # exploration constant
    max_simulations=1000,    # simulations per move
    evaluator=evaluator,
)

state = game.new_initial_state()
while not state.is_terminal():
    current_player = state.current_player()
    action = bot.step(state)
    print(f"Player {current_player} plays action {action}")
    state.apply_action(action)

print(f"Game over. Returns: {state.returns()}")
# Expect [0.0, 0.0] with 1000 simulations — a draw.
```

Note how `bot.step(state)` takes a state and returns an action. This is the `Bot` interface in action: the MCTS algorithm is hidden inside the `MCTSBot` wrapper, which exposes a clean action-selection interface that the game loop can call without knowing anything about how MCTS works internally.

### AlphaZero training loop on Connect Four

AlphaZero in OpenSpiel trains a neural network policy and value function using self-play. Connect Four is large enough to be nontrivial but small enough to train a meaningful policy in a few hours on a laptop.

```python
import pyspiel
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as az_model

game = pyspiel.load_game("connect_four")

# Configure the AlphaZero training run
az_config = alpha_zero.Config(
    game="connect_four",
    path="/tmp/alphazero_connect_four",
    learning_rate=0.001,
    weight_decay=1e-4,
    train_batch_size=128,
    replay_buffer_size=2**14,
    replay_buffer_reuse=3,
    max_steps=50,              # training steps (keep small for illustration)
    checkpoint_freq=10,
    actors=2,                  # self-play actors
    evaluators=1,
    uct_c=1.0,
    max_simulations=100,
    policy_alpha=0.25,
    policy_epsilon=0.25,
    temperature=1.0,
    temperature_drop=30,
    nn_model="resnet",
    nn_width=64,
    nn_depth=4,
    observation_shape=None,    # inferred from game
    output_size=None,          # inferred from game
)

alpha_zero.alpha_zero(az_config)
# After 50 steps the policy is not strong but the mechanics are running.
# Real Connect Four training needs ~5000+ steps.
```

The key observation: the training loop calls `game.new_initial_state()`, advances states using `apply_action`, and reads `information_state_tensor` for neural network input — the same three methods every other algorithm uses.

## The algorithm zoo in OpenSpiel

OpenSpiel ships a wide range of algorithms. Understanding which algorithm to use for which game type is as important as understanding the game interface itself.

### Tabular algorithms (no neural networks)

**CFR (Counterfactual Regret Minimization)**: the foundational algorithm for imperfect-information games. Requires: imperfect-information sequential game, information state strings. Works when: game tree is small enough to enumerate. OpenSpiel implementations: `cfr.CFRSolver`, `cfr.CFRPlusSolver`, `cfr_br.CFRBRSolver`.

**CFR+**: a variant with a modified regret update that converges faster in practice. Same requirements as CFR. Use CFR+ as your default unless you have a reason to prefer vanilla CFR.

**External Sampling MCCFR (Monte Carlo CFR)**: samples external actions (opponent and chance) stochastically, computes exact regrets for the traversed player. Scales to larger games than vanilla CFR. Requirements: same as CFR but operates stochastically, so less memory per iteration.

**Fictitious Play**: each player best-responds to the opponent's historical average strategy. Converges for zero-sum games. Requirements: perfect or imperfect information. Slower than CFR in practice but theoretically clean. `fictitious_play.XFPSolver`.

### Search algorithms (for perfect-information games)

**MCTS (Monte Carlo Tree Search)**: builds a search tree via simulation. Does not require enumeration. Requirements: deterministic or stochastic game, but no hidden information per player (MCTS does not naturally handle information sets). OpenSpiel: `mcts.MCTSBot`.

**AlphaZero**: combines MCTS with a neural network value/policy prior. Requirements: perfect information, current player must be well-defined at each step. The neural network provides a value estimate that makes MCTS more sample-efficient. OpenSpiel: `alpha_zero.alpha_zero`.

**Minimax / Alpha-Beta**: classical adversarial search. Works for two-player zero-sum deterministic perfect-information games. Guarantees optimal play but does not scale without alpha-beta pruning. OpenSpiel: `minimax.minimax_search`.

### Reinforcement learning algorithms

**DQN (Deep Q-Network)**: trains a Q-value network via experience replay. Works on games that can be framed as single-agent (or treated as two-agent via self-play). Requirements: discrete actions, reward signal at each step or end. OpenSpiel: `dqn.DQN`.

**PPO (Proximal Policy Optimization)**: on-policy actor-critic algorithm. More sample-efficient than vanilla policy gradient. Requires: reward signal, differentiable policy. OpenSpiel: `policy_gradient.PolicyGradient` with PPO update.

**NFSP (Neural Fictitious Self-Play)**: combines RL with fictitious play. Trains two networks: a best-response network (via DQN) and an average-strategy network. Converges to approximate Nash in two-player zero-sum games. Requirements: two-player zero-sum, imperfect information supported. OpenSpiel: `nfsp.NFSP`.

### Compatibility summary

| Algorithm | Perfect info | Imperfect info | Simultaneous | Chance nodes |
|-----------|-------------|----------------|--------------|--------------|
| CFR / CFR+ | Yes | Yes (required) | No | Yes |
| MCCFR | Yes | Yes | No | Yes |
| Fictitious Play | Yes | Yes | No | Yes |
| MCTS | Yes | No | No | Yes (with rollout) |
| AlphaZero | Yes | No | No | No |
| Minimax | Yes | No | No | No |
| DQN | Yes | Yes | No | Yes |
| NFSP | Yes | Yes | No | Yes |

For SSA games with hidden information (which satellite has performed a maneuver, which sensor is allocated), you need algorithms from the imperfect-information column. CFR is the right starting point because it has convergence guarantees and its mechanics are transparent.

## Exploitability evaluation

Exploitability is the standard quantitative measure for how close a strategy profile is to Nash equilibrium. Understanding it precisely matters because the capstone uses it as the primary convergence criterion.

### Definition

For a two-player zero-sum game, the **exploitability** of a strategy profile $(\sigma_0, \sigma_1)$ is:

$$\text{exploitability}(\sigma_0, \sigma_1) = \frac{1}{2} \left[ \max_{\sigma_0'} u_0(\sigma_0', \sigma_1) - u_0(\sigma_0, \sigma_1) \right] + \frac{1}{2} \left[ \max_{\sigma_1'} u_1(\sigma_0, \sigma_1') - u_1(\sigma_0, \sigma_1) \right]$$

**Decoding:** Each term inside the brackets is the gain available to one player if they switch to their best response while the other player holds their strategy fixed. The first term is player 0's best-response gain; the second is player 1's best-response gain. Both are non-negative (you can only gain by switching to a best response). At Nash equilibrium, both terms are zero: no player gains by deviating. Exploitability measures how far below Nash equilibrium the current profile is. It is averaged over both players (divided by 2) so it is a symmetric measure. A value of 0.01 means each player is leaving at most 0.01 utility on the table by not playing a best response.

### How OpenSpiel computes exploitability

OpenSpiel's `exploitability.exploitability(game, policy)` works as follows:

1. **Compute the best response for each player**: For player $i$, hold the other player's strategy fixed (as given by `policy`) and solve for the strategy that maximizes player $i$'s expected utility. This is done via a depth-first traversal of the game tree, computing exact values at each node.

2. **Evaluate each best response against the opponent's strategy**: Compute $u_i(\text{BR}_i, \sigma_{-i})$ — the utility player $i$ gets by playing their best response against the opponent's average strategy.

3. **Compare to the current strategy's value**: The exploitability term for player $i$ is $u_i(\text{BR}_i, \sigma_{-i}) - u_i(\sigma_i, \sigma_{-i})$.

4. **Average**: return the mean of the two players' exploitability terms.

This computation is exact but only tractable for small games. For large games (like 7-intensity SSA with 5 sensor modes), approximate best response methods are needed.

### Code: exploitability decreasing over CFR iterations

```python
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability
import matplotlib.pyplot as plt

game = pyspiel.load_game("kuhn_poker")
solver = cfr.CFRPlusSolver(game)  # CFR+ converges faster than vanilla

iterations = []
exploitabilities = []

# Measure at several checkpoints
checkpoints = [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000]
prev = 0
for target in checkpoints:
    for _ in range(target - prev):
        solver.evaluate_and_update_policy()
    prev = target
    policy = solver.average_policy()
    exp = exploitability.exploitability(game, policy)
    iterations.append(target)
    exploitabilities.append(exp)
    print(f"Iter {target:5d}: exploitability = {exp:.8f}")

# At 5000 iterations, exploitability should be < 0.001.
# Analytical Nash for Kuhn poker has exploitability = 0.0.
```

Expected output (approximate):

```
Iter     1: exploitability = 0.45833333
Iter     5: exploitability = 0.24812030
Iter    10: exploitability = 0.15903614
Iter    50: exploitability = 0.05412809
Iter   100: exploitability = 0.03200000
Iter   200: exploitability = 0.01812030
Iter   500: exploitability = 0.00903614
Iter  1000: exploitability = 0.00512809
Iter  2000: exploitability = 0.00270000
Iter  5000: exploitability = 0.00112030
```

The pattern: rapid initial decrease, slower asymptotic convergence. CFR+ converges as roughly $O(1/T)$ rather than $O(1/\sqrt{T})$ for vanilla CFR, which is visible as the faster late-stage convergence.

### Why exploitability matters for SSA applications

In an SSA context, "exploitability" has a direct operational interpretation. If the Defender is running a strategy with exploitability 0.05 in a conjunction-masking game, it means an adversarial Adversary who knows the Defender's strategy could gain 0.05 expected utility by deviating to their best response. In a game where utilities represent detection probabilities and diplomatic penalties, this is a meaningful quantity. A well-converged CFR solution with exploitability near zero gives the Defender a strategy guarantee: regardless of what the Adversary does, the Defender cannot do better than a small epsilon by any unilateral switch.

This is a stronger property than simply "the Defender does well on average." The Nash equilibrium guarantee applies even when the Adversary is adversarially rational and knows the Defender's strategy distribution.

## Key Takeaways

- OpenSpiel's three core abstractions — Game (rules), State (current position), and Observer (what each player can see) — form a complete interface that lets any algorithm run on any game without modification.
- The `information_state_string` and `information_state_tensor` methods are the bridge between game mechanics and game-theoretic algorithms; getting them right is the hardest part of implementing a custom game.
- Chance is not a player: it is a separate node type returned by `current_player()`, treated by averaging over outcomes rather than optimizing over them.
- OpenSpiel's algorithm zoo spans tabular CFR, tree search (MCTS, AlphaZero), and deep RL (DQN, NFSP); the right choice depends on whether the game has hidden information, and how large the game tree is.
- Exploitability measures how far a strategy profile is from Nash equilibrium; for SSA applications it has a direct operational interpretation as the maximum gain an adversary can achieve by best-responding.
- The standardized interface is what makes plug-and-play possible: write a game once, run every algorithm in the zoo on it without changing either the game or the algorithm.

## Quiz

{{#quiz 01-openspiel-architecture.toml}}
