"""Orbital Proximity Competition (OPC) — OpenSpiel custom game.

Attacker fleet (3 satellites) attempts to reach the defended asset at the origin.
Defender fleet (3 satellites) allocates sensor coverage and pursues attackers.
Turn structure: sequential — attacker first, then defender — for T_MAX rounds.

Information asymmetry: the defender observes attacker positions with Gaussian
noise whose variance is inversely proportional to monitoring allocation. The
attacker observes its own positions exactly.

Noise is sampled once per turn and stored in the state so that observation_tensor
is deterministic for a given state object, as OpenSpiel requires.
"""

import numpy as np
import pyspiel

# ── Game constants ────────────────────────────────────────────────────────────

NUM_SATS = 3
T_MAX = 50
PROXIMITY_THRESH = 10.0   # attacker wins if any satellite is within this range
INTERCEPT_RANGE = 6.0     # defender removes attacker within this range

N_MEAN_MOTION = 0.05      # HCW mean motion (rad/step); ~0.4 orbital periods in 50 steps
DT = 1.0
THRUST_DV = 2.0           # delta-v per attacker thrust action (units/step)
PURSUIT_DV = 2.5          # defender pursuit speed (units/step toward target)

# Gaussian std dev on defender's noisy attacker position observations
NOISE_BY_MONITORS = [8.0, 3.0, 0.5]   # index = min(num_monitors, 2)

ATK_INIT_RADIUS = 150.0
DEF_INIT_RADIUS = 40.0

# ── Action encoding ───────────────────────────────────────────────────────────
# Joint action = sum over i of (sat_i_action * 5^i), for i in {0,1,2}
# Attacker per-satellite: 0=Hold, 1=+x, 2=-x, 3=+y, 4=-y
# Defender per-satellite: 0=Hold, 1=Guard (move to PROXIMITY_THRESH intercept point on nearest
#                         attacker's approach line), 2=Monitor atk0, 3=Monitor atk1, 4=Monitor atk2

_N_SAT_ACTIONS = 5
_N_JOINT = _N_SAT_ACTIONS ** NUM_SATS   # 125

ATTACKER_ID = 0
DEFENDER_ID = 1

# Thrust vectors indexed by per-satellite attacker action
_ATK_DV = [
    np.array([0.0, 0.0]),
    np.array([THRUST_DV, 0.0]),
    np.array([-THRUST_DV, 0.0]),
    np.array([0.0, THRUST_DV]),
    np.array([0.0, -THRUST_DV]),
]

# ── Observation tensor layout (28 floats, same size for both players) ─────────
# [0:12]  3 attacker sats [x, y, xdot, ydot] — exact for attacker, noisy+zero-vel for defender
# [12:24] 3 defender sats [x, y, xdot, ydot] — exact for both players
# [24:27] monitoring count per attacker sat [m0, m1, m2]
# [27]    turn counter normalized to [0, 1]
_OBS_SIZE = 28

# ── OpenSpiel registration ────────────────────────────────────────────────────

_GAME_TYPE = pyspiel.GameType(
    short_name="python_opc",
    long_name="Orbital Proximity Competition",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification={},
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_N_JOINT,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=T_MAX * 2,
)


# ── Helper functions ──────────────────────────────────────────────────────────

def decode_joint(joint_action):
    """Decode joint action (0..124) into list of 3 per-satellite actions (0..4)."""
    acts = []
    for _ in range(NUM_SATS):
        acts.append(joint_action % _N_SAT_ACTIONS)
        joint_action //= _N_SAT_ACTIONS
    return acts


def _initial_positions():
    """Return (atk_pos, atk_vel, def_pos, def_vel) as (3,2) arrays."""
    angles_atk = np.array([45.0, 90.0, 135.0]) * np.pi / 180.0
    atk_pos = ATK_INIT_RADIUS * np.stack([np.cos(angles_atk), np.sin(angles_atk)], axis=1)
    atk_vel = np.zeros((NUM_SATS, 2))

    angles_def = np.array([0.0, 120.0, 240.0]) * np.pi / 180.0
    def_pos = DEF_INIT_RADIUS * np.stack([np.cos(angles_def), np.sin(angles_def)], axis=1)
    def_vel = np.zeros((NUM_SATS, 2))

    return atk_pos, atk_vel, def_pos, def_vel


def _hcw_propagate(pos, vel):
    """Propagate one time step under Hill-Clohessy-Wiltshire dynamics (no thrust).

    x = along-track, y = cross-track (2D HCW reduced to in-plane motion).
    """
    n = N_MEAN_MOTION
    theta = n * DT
    ct, st = np.cos(theta), np.sin(theta)

    x, y = pos
    xd, yd = vel

    x_new = (4 - 3 * ct) * x + (st / n) * xd + (2 * (1 - ct) / n) * yd
    y_new = 6 * (st - theta) * x + y - (2 * (1 - ct) / n) * xd + ((4 * st - 3 * theta) / n) * yd
    xd_new = 3 * n * st * x + ct * xd + 2 * st * yd
    yd_new = 6 * n * (ct - 1) * x - 2 * st * xd + (4 * ct - 3) * yd

    return np.array([x_new, y_new]), np.array([xd_new, yd_new])


def _apply_thrust_and_propagate(pos, vel, dv):
    """Apply impulsive delta-v then propagate one HCW step."""
    return _hcw_propagate(pos, vel + dv)


def _sample_noisy_obs(true_pos, monitor_counts, rng):
    """Return noisy observed positions for each attacker satellite."""
    noisy = np.empty_like(true_pos)
    for i in range(NUM_SATS):
        n_monitors = int(min(monitor_counts[i], 2))
        sigma = NOISE_BY_MONITORS[n_monitors]
        noisy[i] = true_pos[i] + rng.normal(0.0, sigma, size=2)
    return noisy


# ── Observer ──────────────────────────────────────────────────────────────────

class OPCObserver:
    """PyObserver implementation for OPC (see open_spiel/python/observation.py)."""

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"OPC observer does not accept params; got {params}")
        self.tensor = np.zeros(_OBS_SIZE, dtype=np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state, player):
        self.tensor.fill(0.0)

        if state.is_terminal():
            return

        atk_pos = state._atk_pos   # (3, 2) true positions
        atk_vel = state._atk_vel   # (3, 2) true velocities
        def_pos = state._def_pos
        def_vel = state._def_vel
        alive = state._atk_alive   # (3,) bool array
        monitor_counts = state._monitor_counts
        turn = state._turn

        if player == ATTACKER_ID:
            # Attacker: own positions exact, defender positions exact
            for i in range(NUM_SATS):
                base = i * 4
                if alive[i]:
                    self.tensor[base:base + 4] = [atk_pos[i, 0], atk_pos[i, 1],
                                                   atk_vel[i, 0], atk_vel[i, 1]]
        else:
            # Defender: attacker positions noisy (stored in state)
            for i in range(NUM_SATS):
                base = i * 4
                if alive[i]:
                    noisy = state._def_noisy_atk_pos[i]
                    self.tensor[base:base + 2] = noisy
                    # velocity unknown to defender; left as 0

        # Both players see defender positions exactly
        for i in range(NUM_SATS):
            base = 12 + i * 4
            self.tensor[base:base + 4] = [def_pos[i, 0], def_pos[i, 1],
                                           def_vel[i, 0], def_vel[i, 1]]

        # Monitoring counts (defender knows its own allocation; attacker gets zeros)
        if player == DEFENDER_ID:
            self.tensor[24:27] = monitor_counts

        self.tensor[27] = turn / T_MAX

    def string_from(self, state, player):
        self.set_from(state, player)
        return str(self.tensor)


# ── State ─────────────────────────────────────────────────────────────────────

class OPCState(pyspiel.State):
    """State for the Orbital Proximity Competition game."""

    def __init__(self, game, rng):
        super().__init__(game)
        self._rng = rng

        atk_pos, atk_vel, def_pos, def_vel = _initial_positions()
        self._atk_pos = atk_pos.copy()
        self._atk_vel = atk_vel.copy()
        self._def_pos = def_pos.copy()
        self._def_vel = def_vel.copy()
        self._atk_alive = np.ones(NUM_SATS, dtype=bool)

        self._turn = 0
        self._cur_player = ATTACKER_ID
        self._is_terminal = False
        self._returns = [0.0, 0.0]

        self._monitor_counts = np.zeros(NUM_SATS, dtype=float)
        # Noisy observations of attacker positions available to defender.
        # Initialized to perfect observations; updated after each attacker move.
        self._def_noisy_atk_pos = atk_pos.copy()

    # ── OpenSpiel interface ───────────────────────────────────────────────────

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        return self._cur_player

    def _legal_actions(self, player):
        return list(range(_N_JOINT))

    def _apply_action(self, action):
        if self._cur_player == ATTACKER_ID:
            self._apply_attacker_action(action)
        else:
            self._apply_defender_action(action)

    def _apply_attacker_action(self, joint_action):
        sat_actions = decode_joint(joint_action)
        for i in range(NUM_SATS):
            if not self._atk_alive[i]:
                continue
            dv = _ATK_DV[sat_actions[i]]
            new_pos, new_vel = _apply_thrust_and_propagate(
                self._atk_pos[i], self._atk_vel[i], dv
            )
            self._atk_pos[i] = new_pos
            self._atk_vel[i] = new_vel

        # Sample fresh noisy observations for the defender
        self._def_noisy_atk_pos = _sample_noisy_obs(
            self._atk_pos, self._monitor_counts, self._rng
        )

        self._cur_player = DEFENDER_ID

    def _apply_defender_action(self, joint_action):
        sat_actions = decode_joint(joint_action)

        # Compute new monitoring allocation from this action
        new_monitor_counts = np.zeros(NUM_SATS, dtype=float)
        for i, act in enumerate(sat_actions):
            if act >= 2:   # Monitor atk satellite (act - 2)
                target_atk = act - 2
                if self._atk_alive[target_atk]:
                    new_monitor_counts[target_atk] += 1.0

        # Propagate defender satellites (guard or hold)
        for i, act in enumerate(sat_actions):
            if act == 1:   # Guard: move to intercept point on nearest attacker's approach line
                nearest = self._nearest_surviving_attacker(i)
                if nearest is not None:
                    atk_pos = self._atk_pos[nearest]
                    atk_dist = np.linalg.norm(atk_pos)
                    if atk_dist > 1e-6:
                        # Guard point: stand at PROXIMITY_THRESH from origin in the
                        # attacker's approach direction. This blocks the approach path
                        # rather than chasing the attacker across the field.
                        guard_pt = (atk_pos / atk_dist) * PROXIMITY_THRESH
                        direction = guard_pt - self._def_pos[i]
                        dist = np.linalg.norm(direction)
                        if dist > 1e-6:
                            step = min(PURSUIT_DV, dist)
                            self._def_pos[i] += (direction / dist) * step
            # Hold and Monitor actions: defender satellite does not move

        # Check intercepts: defender satellites within INTERCEPT_RANGE of attackers
        for j in range(NUM_SATS):
            if not self._atk_alive[j]:
                continue
            for i in range(NUM_SATS):
                if np.linalg.norm(self._def_pos[i] - self._atk_pos[j]) <= INTERCEPT_RANGE:
                    self._atk_alive[j] = False
                    break

        self._monitor_counts = new_monitor_counts
        self._turn += 1
        self._cur_player = ATTACKER_ID

        self._check_terminal()

    def _check_terminal(self):
        # Attacker wins: any surviving attacker within proximity of origin
        for j in range(NUM_SATS):
            if self._atk_alive[j]:
                if np.linalg.norm(self._atk_pos[j]) <= PROXIMITY_THRESH:
                    self._is_terminal = True
                    self._returns = [1.0, -1.0]
                    return

        # Defender wins: all attackers intercepted
        if not np.any(self._atk_alive):
            self._is_terminal = True
            self._returns = [-1.0, 1.0]
            return

        # Defender wins: episode ends without attacker achieving proximity
        if self._turn >= T_MAX:
            self._is_terminal = True
            self._returns = [-1.0, 1.0]

    def _nearest_surviving_attacker(self, def_idx):
        min_dist = float("inf")
        nearest = None
        for j in range(NUM_SATS):
            if not self._atk_alive[j]:
                continue
            d = np.linalg.norm(self._atk_pos[j] - self._def_pos[def_idx])
            if d < min_dist:
                min_dist = d
                nearest = j
        return nearest

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        return list(self._returns)

    def _action_to_string(self, player, action):
        acts = decode_joint(action)
        if player == ATTACKER_ID:
            names = ["Hold", "+x", "-x", "+y", "-y"]
            return f"Atk[{','.join(names[a] for a in acts)}]"
        else:
            names = ["Hold", "Guard", "Mon0", "Mon1", "Mon2"]
            return f"Def[{','.join(names[a] for a in acts)}]"

    def information_state_string(self, player):
        obs = np.zeros(_OBS_SIZE, dtype=np.float32)
        observer = OPCObserver(None, None)
        observer.set_from(self, player)
        return str(observer.tensor)

    def information_state_tensor(self, player):
        observer = OPCObserver(None, None)
        observer.set_from(self, player)
        return observer.tensor.tolist()

    def observation_string(self, player):
        return self.information_state_string(player)

    def observation_tensor(self, player):
        return self.information_state_tensor(player)

    def __str__(self):
        alive_str = "".join("A" if a else "x" for a in self._atk_alive)
        atk_dists = [
            f"{np.linalg.norm(self._atk_pos[i]):.1f}" if self._atk_alive[i] else "dead"
            for i in range(NUM_SATS)
        ]
        return (
            f"Turn {self._turn} | Player {self._cur_player} | "
            f"Alive: {alive_str} | Dist-to-origin: {atk_dists} | "
            f"Monitors: {self._monitor_counts.tolist()}"
        )


# ── Game ──────────────────────────────────────────────────────────────────────

class OPCGame(pyspiel.Game):
    """Orbital Proximity Competition game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
        self._rng = np.random.default_rng()

    def new_initial_state(self):
        return OPCState(self, np.random.default_rng())

    def make_py_observer(self, iig_obs_type=None, params=None):
        return OPCObserver(iig_obs_type, params)


# ── Registration ──────────────────────────────────────────────────────────────

pyspiel.register_game(_GAME_TYPE, OPCGame)
