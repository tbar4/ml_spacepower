# Lesson 2: Implementing a Custom Game

## Where this fits

In Module 3 you implemented a single-agent MDP as an OpenSpiel game, mostly as a thin wrapper around an episodic environment. Now we do it properly for the case that matters for the capstone: a two-player imperfect-information sequential game. The mechanics here are also what you will translate into Rust traits in Module 8 lesson 3 and the capstone. Once you have done this in Python, the Rust version is largely a syntactic restatement.

## The game we will build: Mini Maneuver

To keep the focus on the OpenSpiel mechanics rather than the SSA semantics, we will use a deliberately simplified game. The capstone (lesson 4 and the project) will use a richer SSA-flavored variant.

**Mini Maneuver** is a two-player game:

1. Chance deals one of two private cards to the **Operator** (player 0): "Maneuver" (M) or "No-maneuver" (N), each with probability 0.5.
2. The Operator sees their card and decides to **Signal** (S) or **Stay quiet** (Q). The signal is public.
3. The **Observer** (player 1) sees the signal but not the card. They decide to **Watch** (W) or **Skip** (K).
4. The game ends. Payoffs:
   - If the card is M and the Observer chose Watch: Observer +2, Operator -2 (caught maneuvering)
   - If the card is M and the Observer chose Skip: Observer -1, Operator +1 (got away)
   - If the card is N and the Observer chose Watch: Observer -1, Operator +1 (wasted observation)
   - If the card is N and the Observer chose Skip: Observer 0, Operator 0 (nothing happens)

This is a 2-player, zero-sum (after centering), imperfect-information, sequential game. It has the same structural features as Kuhn poker or any small imperfect-information benchmark: hidden information per player, sequential moves, and a non-trivial equilibrium that requires randomization.

## The mathematical structure first

Before writing code, let us hand-trace what an "information set" is in this game.

**Operator information sets**: the Operator sees their card. So they have two information sets, one per card: `[M]` and `[N]`. In each, they choose Signal or Quiet.

**Observer information sets**: the Observer does not see the card. They only see the signal. So they have two information sets: `[S]` (Operator signaled) and `[Q]` (Operator stayed quiet). In each, they choose Watch or Skip.

The game tree has 8 terminal nodes (2 cards x 2 operator actions x 2 observer actions). Each information set may correspond to multiple terminal nodes that the player cannot tell apart; the player's strategy must be a function of the information set, not the underlying state.

For CFR (which we will run on this game in the next lesson) you need a unique string identifier per information set. A common encoding is "card or signal seen, history of public actions." For the Operator, the information state could be just the card: "M" or "N". For the Observer, it could be the public signal: "S" or "Q".

## The OpenSpiel implementation

```python
"""
mini_maneuver.py: a 2-player imperfect-information game for CFR practice.
"""

import enum
import numpy as np
import pyspiel

# Cards
class Card(enum.IntEnum):
    NO_MANEUVER = 0
    MANEUVER    = 1

# Operator actions
SIGNAL  = 0
QUIET   = 1

# Observer actions
WATCH  = 0
SKIP   = 1

# Players
OPERATOR = 0
OBSERVER = 1

_GAME_TYPE = pyspiel.GameType(
    short_name="mini_maneuver",
    long_name="Mini Maneuver: a 2-player imperfect-information game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={},
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=2,           # operator and observer each have 2 actions
    max_chance_outcomes=2,            # 2 possible cards
    num_players=2,
    min_utility=-2.0,
    max_utility=2.0,
    max_game_length=3,                # chance, operator, observer
)


class MiniManeuverGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
    
    def new_initial_state(self):
        return MiniManeuverState(self)


class MiniManeuverState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._card             = None    # set after chance
        self._operator_action  = None    # set after operator's move
        self._observer_action  = None    # set after observer's move
    
    def current_player(self):
        if self._observer_action is not None:
            return pyspiel.PlayerId.TERMINAL
        if self._card is None:
            return pyspiel.PlayerId.CHANCE
        if self._operator_action is None:
            return OPERATOR
        return OBSERVER
    
    def legal_actions(self, player=None):
        if self.is_terminal():
            return []
        if self.is_chance_node():
            return [Card.NO_MANEUVER.value, Card.MANEUVER.value]
        return [SIGNAL, QUIET] if self._operator_action is None else [WATCH, SKIP]
    
    def chance_outcomes(self):
        return [(Card.NO_MANEUVER.value, 0.5),
                (Card.MANEUVER.value,    0.5)]
    
    def _apply_action(self, action):
        if self._card is None:
            self._card = Card(action)
        elif self._operator_action is None:
            self._operator_action = action
        else:
            self._observer_action = action
    
    def is_terminal(self):
        return self._observer_action is not None
    
    def returns(self):
        """Return per-player utilities. Zero-sum."""
        if not self.is_terminal():
            return [0.0, 0.0]
        
        if self._card == Card.MANEUVER:
            if self._observer_action == WATCH:
                # Caught
                return [-2.0, 2.0]      # operator loses, observer wins
            else:
                # Got away
                return [1.0, -1.0]
        else:  # NO_MANEUVER
            if self._observer_action == WATCH:
                # Wasted observation
                return [1.0, -1.0]
            else:
                # Nothing happens
                return [0.0, 0.0]
    
    def information_state_string(self, player):
        """Unique string identifying this player's information set."""
        if player == OPERATOR:
            # Operator sees the card and remembers their own action history
            if self._card is None:
                return ""
            s = f"card={self._card.name}"
            if self._operator_action is not None:
                s += f",my_action={self._operator_action}"
            return s
        else:  # OBSERVER
            # Observer sees the operator's action (the public signal) but not the card
            if self._operator_action is None:
                return ""
            return f"signal={'S' if self._operator_action == SIGNAL else 'Q'}"
    
    def information_state_tensor(self, player):
        """Same info, as a fixed-length vector for neural networks."""
        # 4 features: card_M, card_N, signal_S, signal_Q
        # All 0 if not yet known to this player
        t = [0.0, 0.0, 0.0, 0.0]
        if player == OPERATOR and self._card is not None:
            t[Card.MANEUVER.value]    = 1.0 if self._card == Card.MANEUVER else 0.0
            t[Card.NO_MANEUVER.value] = 1.0 if self._card == Card.NO_MANEUVER else 0.0
        if self._operator_action is not None:
            if self._operator_action == SIGNAL:
                t[2] = 1.0
            else:
                t[3] = 1.0
        return np.array(t, dtype=np.float32)
    
    def observation_string(self, player):
        # For our purposes, observation = information state
        return self.information_state_string(player)
    
    def observation_tensor(self, player):
        return self.information_state_tensor(player)
    
    def __str__(self):
        s = f"card={self._card.name if self._card else '?'}"
        if self._operator_action is not None:
            s += f", op={'S' if self._operator_action == SIGNAL else 'Q'}"
        if self._observer_action is not None:
            s += f", obs={'W' if self._observer_action == WATCH else 'K'}"
        return s


# Register the game so OpenSpiel sees it
pyspiel.register_game(_GAME_TYPE, _GAME_INFO, MiniManeuverGame)
```

## Verifying the game with built-in checks

OpenSpiel provides a "game integration test" that checks your game implementation for consistency. It exercises many random game traces and verifies that every method returns sensible values, that information states are consistent, and so on.

```python
import pyspiel
from open_spiel.python.algorithms.get_all_states import get_all_states

# Sanity check: enumerate all states
game = MiniManeuverGame()
all_states = get_all_states(game, depth_limit=-1, include_terminals=True,
                            include_chance_states=True)
print(f"Total states: {len(all_states)}")
# Should be: 1 (initial chance) + 2 (after card dealt) + 4 (after operator) + 8 (terminal) = 15

# Check information sets
infosets_op  = set()
infosets_obs = set()
for state_str, state in all_states.items():
    if state.is_terminal() or state.is_chance_node():
        continue
    if state.current_player() == OPERATOR:
        infosets_op.add(state.information_state_string(OPERATOR))
    else:
        infosets_obs.add(state.information_state_string(OBSERVER))

print(f"Operator information sets: {sorted(infosets_op)}")
print(f"Observer information sets: {sorted(infosets_obs)}")
```

If everything is wired up correctly, the operator should have 2 information sets (one per card) and the observer should have 2 information sets (one per signal).

## Running CFR on this game

Now the payoff. With the game registered, you can run any of OpenSpiel's CFR implementations on it:

```python
from open_spiel.python.algorithms import cfr

solver = cfr.CFRSolver(game)
for i in range(1000):
    solver.evaluate_and_update_policy()

avg_policy = solver.average_policy()

print("\n=== Average strategy ===")
for state_str, state in all_states.items():
    if state.is_terminal() or state.is_chance_node():
        continue
    info_str = state.information_state_string(state.current_player())
    probs = avg_policy.action_probabilities(state)
    print(f"Player {state.current_player()}, infoset {info_str}: {probs}")
```

After 1000 iterations, the average strategy should be close to the Nash equilibrium of this game. You can compute the exact equilibrium by hand if you want (it is a small game), but the point is: the game class you wrote plugs straight into OpenSpiel's algorithm zoo.

You can also run exploitability:

```python
from open_spiel.python.algorithms import exploitability

exp = exploitability.exploitability(game, avg_policy)
print(f"Exploitability after 1000 iterations: {exp:.6f}")
```

This number should be near zero for a converged solver. Exploitability is the standard metric for measuring how close a strategy profile is to Nash equilibrium: it is the average gain available to a player who unilaterally switches to a best response against the opponent's strategy.

## What the rest of the OpenSpiel API needs

For algorithms beyond CFR you may need to implement additional methods:

- `clone()`: usually inherited correctly from the parent class via Python copy. For complex game state you might need to override.
- `apply_actions()` (note the s): for simultaneous-move games, where multiple players choose at once. Not relevant for our sequential game.
- `serialize()` and `deserialize_state()`: for saving state to disk. Not strictly needed for solver-time use; needed for things like distributed training or replay.

For Mini Maneuver, the implementation above is complete. Most simple games need only the methods we have shown.

## The Python game protocol in full

The code above works, but before you write your own game from scratch you need to understand the complete protocol: what methods are required, what methods are optional, and what the framework expects from each.

### Required methods and their contracts

**`current_player()`** must return one of:
- A non-negative integer (player index, 0-based)
- `pyspiel.PlayerId.CHANCE` (= -1) at stochastic nodes
- `pyspiel.PlayerId.TERMINAL` (= -4) when the game is over
- `pyspiel.PlayerId.SIMULTANEOUS` (= -2) for simultaneous-move games

The value determines which other methods the framework will call. If you return `CHANCE`, the framework expects `chance_outcomes()` to be valid. If you return `TERMINAL`, it expects `returns()` to be valid. Getting this wrong causes hard-to-diagnose errors downstream in algorithm code.

**`_apply_action(action)`** (note the underscore): this is the method you override, not `apply_action`. The base class `apply_action` does bookkeeping (history tracking, action count) before calling `_apply_action`. Always override `_apply_action`, never `apply_action`.

**`legal_actions(player=None)`**: for sequential games, `player` will be `None` or the current player. The returned list must be a sorted list of non-negative integers. Algorithms rely on the list being deterministic (same state, same legal actions, same order). Use `sorted()` if your action generation does not naturally produce a sorted list.

**`chance_outcomes()`**: returns a list of `(action, probability)` pairs summing to 1.0. Must only be called when `is_chance_node()` is True. Probabilities must be non-negative and sum to exactly 1.0 (floating-point tolerance is applied by the checker).

**`returns()`**: must return a list of floats of length `num_players()`. Must only be meaningful when `is_terminal()` is True. Before terminal, returning zeros is conventional but the framework never inspects this value for non-terminal states.

**`information_state_string(player)`**: the most important method for imperfect-information algorithms. The string must be:
- Identical for all states that belong to the same information set for `player`
- Different for states that belong to different information sets
- Efficiently computable (CFR calls this millions of times)

Do not include the world state in the string for a player who cannot observe it. A common mistake is encoding the opponent's private card in the information state string for both players, making them both perfectly informed.

### Common gotchas

**Gotcha 1: history vs. information state.** The information state string encodes what the player *knows*, not the full game history. In Mini Maneuver, the Operator knows their card. That is it for their information state when deciding. Do not encode the *order* in which events happened unless that order is observable to the player.

**Gotcha 2: the `legal_actions` method signature.** Some OpenSpiel calls pass `player` as a keyword argument; others call with no argument. Your signature should accept `player=None`. If your game is sequential (not simultaneous), the `player` argument is always the current player or `None`, and you can ignore it.

**Gotcha 3: action numbering is global.** `num_distinct_actions()` returns a single global count, not per-player counts. If the Operator has 2 actions (0, 1) and the Observer has 2 actions (0, 1), the game reports `num_distinct_actions = 2` because the action spaces overlap in index. If they had different sizes, you would report the maximum. This means action indices can be reused across players as long as each player only ever sees their own actions at their own nodes.

**Gotcha 4: the reward model must match your `returns()` behavior.** If you declare `reward_model = TERMINAL` but your `returns()` method returns non-zero values at non-terminal states, CFR will produce wrong results. Always keep these in sync.

**Gotcha 5: `clone()` must deep-copy mutable state.** The base class `clone()` uses Python's `copy.deepcopy`. If your state contains mutable objects that `deepcopy` handles incorrectly (e.g., custom C++ extension objects), you need to override `clone()` explicitly. For pure-Python states like Mini Maneuver, the default is fine.

## The pursuit-evasion SSA game

Mini Maneuver is a conceptual warmup. Now we build a richer SSA game that captures more of the operational texture: a satellite defender choosing which sensors to activate, and an adversary choosing how to approach an orbital zone. We call this the **Orbital Pursuit-Evasion** game.

### Scenario

A 5x5 grid represents a section of orbital phase space (think of it as a discretized altitude-vs-longitude map). The **Attacker** (player 0) starts outside the grid and wants to reach the center cell (2,2) without being detected. The **Defender** (player 1) has 5 sensors and, at each turn, must choose which sensor to activate. Each sensor covers a different region of the grid. The game lasts at most 3 turns; the Attacker moves one cell per turn.

This is a two-player zero-sum game with hidden information: the Attacker knows their own position and intended path, but the Defender only knows which sensor detected activity (if any). The Defender does not see the Attacker's position directly.

### Game structure

- **Chance (Stage 1)**: Nature picks the Attacker's entry point: one of 4 border cells (north, south, east, west entry). Attacker sees this; Defender does not.
- **Attacker (Stages 2-4)**: Attacker chooses a direction of movement each turn: `{N, S, E, W, Stay}`. Attacker wants to reach (2,2) undetected.
- **Defender (Stages 2-4)**: Simultaneously (or sequentially after observing detection events), Defender activates one of 5 sensors.
- **Detection (after each turn)**: If the Attacker is in a sensor's coverage area, detection occurs with probability depending on the sensor type.

For simplicity in this implementation, we make the game sequential: Defender acts first (choosing a sensor), then Attacker moves. Defender observes only whether their activated sensor triggered, not where the Attacker is.

### Full Python implementation

```python
"""
orbital_pursuit_evasion.py: a 5x5 grid SSA pursuit-evasion game.
Two players: Attacker (tries to reach center undetected),
             Defender (tries to detect Attacker by choosing sensors).
"""

import enum
import numpy as np
import pyspiel

GRID_SIZE = 5
CENTER = (2, 2)
NUM_SENSORS = 5
MAX_TURNS = 3

# Entry points: (row, col) for north/south/east/west edges
ENTRY_NORTH = (0, 2)
ENTRY_SOUTH = (4, 2)
ENTRY_EAST  = (2, 4)
ENTRY_WEST  = (2, 0)
ENTRY_POINTS = [ENTRY_NORTH, ENTRY_SOUTH, ENTRY_EAST, ENTRY_WEST]

ATTACKER = 0
DEFENDER = 1

# Attacker movement actions
MOVE_N   = 0
MOVE_S   = 1
MOVE_E   = 2
MOVE_W   = 3
MOVE_STAY = 4
ATTACKER_ACTIONS = 5

# Sensor coverage: list of (row, col) cells each sensor covers
SENSOR_COVERAGE = {
    0: [(0,0),(0,1),(1,0),(1,1)],         # NW quadrant
    1: [(0,3),(0,4),(1,3),(1,4)],         # NE quadrant
    2: [(3,0),(3,1),(4,0),(4,1)],         # SW quadrant
    3: [(3,3),(3,4),(4,3),(4,4)],         # SE quadrant
    4: [(1,2),(2,1),(2,2),(2,3),(3,2)],   # Center cross
}
SENSOR_DETECTION_PROB = {
    0: 0.7, 1: 0.7, 2: 0.7, 3: 0.7,      # corner sensors
    4: 0.9,                                 # center cross sensor is best
}

_GAME_TYPE = pyspiel.GameType(
    short_name="orbital_pursuit_evasion",
    long_name="Orbital Pursuit-Evasion: a 5x5 SSA grid game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={},
)

# Global action count: max of attacker (5) and defender (5 sensors).
# Convention: defender actions are 0-4 (sensor index), attacker actions 0-4 (moves).
_NUM_DISTINCT_ACTIONS = max(ATTACKER_ACTIONS, NUM_SENSORS)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=len(ENTRY_POINTS),
    num_players=2,
    min_utility=-1.0,    # Defender wins: Attacker gets -1
    max_utility=1.0,     # Attacker wins: Attacker gets +1
    max_game_length=1 + 2 * MAX_TURNS,  # chance + (defender, attacker) * turns
)


def _move(row, col, action):
    """Apply a movement action to (row, col), clamped to grid."""
    if action == MOVE_N:   row = max(0, row - 1)
    elif action == MOVE_S: row = min(GRID_SIZE - 1, row + 1)
    elif action == MOVE_E: col = min(GRID_SIZE - 1, col + 1)
    elif action == MOVE_W: col = max(0, col - 1)
    # MOVE_STAY: no change
    return row, col


class OrbitalPursuitEvasionGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self):
        return OrbitalPursuitEvasionState(self)


class OrbitalPursuitEvasionState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        # Entry point: set by chance node
        self._entry_idx = None
        # Attacker position: set after chance
        self._att_row = None
        self._att_col = None
        # Turn counter (0-indexed, increments after each attacker move)
        self._turn = 0
        # Phase within a turn: "defender" -> defender acts first, then "attacker"
        self._phase = "chance"
        # History of (sensor_chosen, detection_result) per turn for defender's info
        self._defender_history = []   # list of (sensor_idx, detected: bool)
        # Whether game is over
        self._terminal = False
        self._winner = None  # "attacker" or "defender"

    def current_player(self):
        if self._terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._phase == "chance":
            return pyspiel.PlayerId.CHANCE
        if self._phase == "defender":
            return DEFENDER
        if self._phase == "attacker":
            return ATTACKER
        return pyspiel.PlayerId.TERMINAL

    def legal_actions(self, player=None):
        if self._terminal:
            return []
        if self._phase == "chance":
            return list(range(len(ENTRY_POINTS)))
        if self._phase == "defender":
            return list(range(NUM_SENSORS))
        if self._phase == "attacker":
            return list(range(ATTACKER_ACTIONS))
        return []

    def chance_outcomes(self):
        n = len(ENTRY_POINTS)
        return [(i, 1.0 / n) for i in range(n)]

    def _apply_action(self, action):
        if self._phase == "chance":
            self._entry_idx = action
            self._att_row, self._att_col = ENTRY_POINTS[action]
            self._phase = "defender"

        elif self._phase == "defender":
            # Defender selects sensor; resolve detection stochastically.
            # We record the defender's sensor choice but detection is a chance event.
            # For simplicity, resolve detection here (deterministic w/ probability threshold).
            # In a more rigorous implementation, detection would be a second chance node.
            sensor = action
            att_pos = (self._att_row, self._att_col)
            in_coverage = att_pos in SENSOR_COVERAGE[sensor]
            # Deterministic threshold: use probability as detection indicator
            # (full stochastic version would add a chance node here)
            detected = in_coverage  # simplification: if in coverage, detected
            self._defender_history.append((sensor, detected))
            # Check if attacker is detected
            if detected:
                self._terminal = True
                self._winner = "defender"
            else:
                self._phase = "attacker"

        elif self._phase == "attacker":
            self._att_row, self._att_col = _move(self._att_row, self._att_col, action)
            # Check if attacker reached center
            if (self._att_row, self._att_col) == CENTER:
                self._terminal = True
                self._winner = "attacker"
            else:
                self._turn += 1
                if self._turn >= MAX_TURNS:
                    # Time's up: attacker failed to reach center, defender wins
                    self._terminal = True
                    self._winner = "defender"
                else:
                    self._phase = "defender"

    def is_terminal(self):
        return self._terminal

    def returns(self):
        if not self._terminal:
            return [0.0, 0.0]
        if self._winner == "attacker":
            return [1.0, -1.0]
        else:  # defender wins
            return [-1.0, 1.0]

    def information_state_string(self, player):
        if player == ATTACKER:
            # Attacker knows their entry point, position, and turn
            if self._entry_idx is None:
                return ""
            return (f"entry={self._entry_idx},"
                    f"pos=({self._att_row},{self._att_col}),"
                    f"turn={self._turn}")
        else:  # DEFENDER
            # Defender knows only the history of (sensor, detected) pairs
            if not self._defender_history:
                return "no_history"
            parts = [f"s{s}={'D' if d else 'N'}"
                     for (s, d) in self._defender_history]
            return ",".join(parts)

    def information_state_tensor(self, player):
        if player == ATTACKER:
            # 4 (entry one-hot) + 25 (position one-hot on 5x5) + 1 (turn/3)
            t = np.zeros(30, dtype=np.float32)
            if self._entry_idx is not None:
                t[self._entry_idx] = 1.0
                pos_idx = self._att_row * GRID_SIZE + self._att_col
                t[4 + pos_idx] = 1.0
                t[29] = self._turn / MAX_TURNS
        else:  # DEFENDER
            # 5 sensors x 2 outcomes x 3 turns = 30 features
            t = np.zeros(30, dtype=np.float32)
            for turn_idx, (sensor, detected) in enumerate(self._defender_history):
                base = turn_idx * 10  # 5 sensor bits + 5 detection bits
                t[base + sensor] = 1.0
                if detected:
                    t[base + 5 + sensor] = 1.0
        return t

    def observation_string(self, player):
        return self.information_state_string(player)

    def observation_tensor(self, player):
        return self.information_state_tensor(player)

    def __str__(self):
        if self._entry_idx is None:
            return "initial(chance)"
        return (f"turn={self._turn}, phase={self._phase}, "
                f"att=({self._att_row},{self._att_col}), "
                f"hist={self._defender_history}")


pyspiel.register_game(_GAME_TYPE, _GAME_INFO, OrbitalPursuitEvasionGame)
```

### What this game illustrates

The Orbital Pursuit-Evasion game demonstrates several features beyond Mini Maneuver:

- **Multiple turns with state evolution**: the attacker's position changes over time. The information state must capture the full relevant history, not just the current observation.
- **Asymmetric information state shapes**: the attacker's tensor is 30 features; the defender's is 30 features but with completely different semantics. Both must fit within `information_state_tensor_shape()`.
- **Mixed player ordering**: defender acts first within each turn (choosing a sensor), then attacker moves. This ordering choice affects the information structure: the attacker can react to knowledge of which sensor the defender chose (but in our formulation, the attacker does not observe the defender's sensor choice directly).
- **SSA realism**: the sensor coverage map mirrors how actual SSA sensor networks are allocated across orbital regimes. A center-cross sensor (sensor 4) covering the target regime has higher detection probability; corner sensors cover approach corridors.

## Testing your game

### The `check_game` utility

OpenSpiel provides `pyspiel.check_game` and the integration test infrastructure in `open_spiel/integration_tests/`. The Python-level checker is the simplest to run:

```python
from open_spiel.python.tests import games_test

# Run the standard game checkers on your game
game = OrbitalPursuitEvasionGame()

# The simplest check: play out random games and verify no exceptions
import random
def random_game(game, seed=42):
    rng = random.Random(seed)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = rng.choices(actions, weights=probs)[0]
        else:
            legal = state.legal_actions()
            action = rng.choice(legal)
        state.apply_action(action)
    return state.returns()

# Play 100 random games; if no exception, basic structure is correct
for seed in range(100):
    returns = random_game(game, seed)
    assert len(returns) == 2, f"Returns should have length 2, got {returns}"
    assert abs(sum(returns)) < 1e-6, f"Zero-sum violated: {returns}"

print("100 random games completed without errors.")
```

### What `check_game` verifies

The full integration test suite (invoked via `pyspiel.GameType` metadata checks and the integration test runner) verifies:

1. **Legal actions are consistent**: calling `legal_actions()` twice on the same state returns the same list.
2. **`apply_action` is deterministic**: applying the same action from the same state always produces the same successor state.
3. **Chance probabilities sum to 1.0**: for every chance node, `sum(p for a, p in state.chance_outcomes()) == 1.0`.
4. **Returns are in bounds**: every terminal state's returns satisfy `min_utility <= r <= max_utility` for each player.
5. **Information state strings are consistent**: all states in the same information set (reachable via different histories but producing the same information for a player) should have the same information state string.
6. **Tensor shapes are consistent**: `information_state_tensor(player)` always returns an array of the same shape, matching `information_state_tensor_shape()`.

### Debugging illegal action errors

The most common error when first running algorithms on a new game is `IllegalActionError`. The typical causes:

**Cause 1: `legal_actions()` returns a different set than what the algorithm tried to use.** If your `legal_actions()` changes based on mutable state that you accidentally modified, the algorithm may attempt an action that was legal at query time but is not legal after some intermediate operation. Make `legal_actions()` a pure function of the state.

**Cause 2: action index out of range.** If `num_distinct_actions()` returns 4 but your game sometimes returns action 4 (which is out of range), the algorithm's internal tables underflow. Always verify your action indices are in `[0, num_distinct_actions() - 1]`.

```python
# Debugging helper: trace every state and check action validity
def check_action_ranges(game, max_depth=5):
    from open_spiel.python.algorithms.get_all_states import get_all_states
    num_actions = game.num_distinct_actions()
    all_states = get_all_states(game, depth_limit=max_depth)
    for key, state in all_states.items():
        if state.is_terminal():
            continue
        legal = state.legal_actions()
        for a in legal:
            if a < 0 or a >= num_actions:
                print(f"ILLEGAL ACTION INDEX {a} in state: {state}")
                print(f"  num_distinct_actions = {num_actions}")
                print(f"  legal_actions = {legal}")
    print("Action range check complete.")

check_action_ranges(OrbitalPursuitEvasionGame())
```

**Cause 3: information state tensor shape mismatch.** Call `game.information_state_tensor_shape()` and compare it to what your `information_state_tensor()` method returns at a few states:

```python
game = OrbitalPursuitEvasionGame()
expected_shape = game.information_state_tensor_shape()
print(f"Declared shape: {expected_shape}")

state = game.new_initial_state()
# Advance past chance
state.apply_action(0)  # entry point 0
# Advance past defender
state.apply_action(0)  # sensor 0
# Now attacker acts
t0 = state.information_state_tensor(ATTACKER)
t1 = state.information_state_tensor(DEFENDER)
print(f"Attacker tensor shape: {t0.shape}")
print(f"Defender tensor shape: {t1.shape}")
# Both should match expected_shape
```

If these shapes do not match, any neural network that reads information state tensors will silently produce wrong results. Always run this check after changing the tensor encoding.

## Mapping this to the capstone

The capstone game (designed in lesson 4) extends Mini Maneuver in three ways:

1. The Operator chooses among 4 maneuver intensities, not just maneuver/no-maneuver. This makes the action space richer.
2. The Observer chooses among 5 sensor allocations, not just watch/skip. The richer action space lets the equilibrium have nontrivial mixed strategies.
3. The detection probability depends on both the maneuver intensity and the sensor allocation, not a hard yes/no. This requires multiple chance nodes (one for the card, one for the noisy detection).

Functionally, the structure is the same: chance generates hidden information, the operator acts on it, the observer responds. The information state strings, the legal action lists, the terminal returns, the chance outcomes are all there. Each capstone-specific feature is a small extension of what you wrote here.

This is also the design we will translate into Rust traits in lesson 3.

## Key Takeaways

- Implementing a custom game requires overriding `_apply_action` (not `apply_action`), returning the right `current_player()` sentinel at each stage, and ensuring information state strings encode only what each player can actually observe.
- The information state string must be identical for world states a player cannot distinguish and different for states they can — this is the contract that CFR depends on for correctness.
- Common gotchas include using global action numbering across all players, forgetting to deep-copy mutable state in `clone()`, and letting tensor shapes vary across states.
- The `check_game` utility and integration tests catch most structural errors early; always run them before attempting to solve a new game.
- The Orbital Pursuit-Evasion game shows how SSA sensor allocation naturally maps to the OpenSpiel game structure: sensor coverage regions, detection probabilities, and position tracking all fit within the standard interface.
- Once your game passes the integration tests and CFR produces non-degenerate mixed strategies, it is a genuine research artifact: you can swap in any algorithm from the OpenSpiel zoo without touching the game code.

## Quiz

{{#quiz 02-implementing-custom-games.toml}}
