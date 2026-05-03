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

## Mapping this to the capstone

The capstone game (designed in lesson 4) extends Mini Maneuver in three ways:

1. The Operator chooses among 4 maneuver intensities, not just maneuver/no-maneuver. This makes the action space richer.
2. The Observer chooses among 5 sensor allocations, not just watch/skip. The richer action space lets the equilibrium have nontrivial mixed strategies.
3. The detection probability depends on both the maneuver intensity and the sensor allocation, not a hard yes/no. This requires multiple chance nodes (one for the card, one for the noisy detection).

Functionally, the structure is the same: chance generates hidden information, the operator acts on it, the observer responds. The information state strings, the legal action lists, the terminal returns, the chance outcomes are all there. Each capstone-specific feature is a small extension of what you wrote here.

This is also the design we will translate into Rust traits in lesson 3.

## Quiz

{{#quiz 02-implementing-custom-games.toml}}
