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

## Quiz

{{#quiz 01-openspiel-architecture.toml}}
