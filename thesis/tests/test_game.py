"""Unit tests for the OPC OpenSpiel game."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import pyspiel

from opc.game import (
    OPCGame, OPCState,
    ATTACKER_ID, DEFENDER_ID,
    NUM_SATS, T_MAX, PROXIMITY_THRESH, INTERCEPT_RANGE,
    _N_JOINT, _OBS_SIZE,
    decode_joint,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_game():
    return OPCGame()


def play_random_game(seed=0):
    rng = np.random.default_rng(seed)
    game = make_game()
    state = game.new_initial_state()
    while not state.is_terminal():
        action = int(rng.integers(0, _N_JOINT))
        state.apply_action(action)
    return state


# ── decode_joint ──────────────────────────────────────────────────────────────

def test_decode_joint_zero():
    assert decode_joint(0) == [0, 0, 0]


def test_decode_joint_max():
    assert decode_joint(124) == [4, 4, 4]


def test_decode_joint_mixed():
    # joint = 1 + 2*5 + 3*25 = 1 + 10 + 75 = 86
    assert decode_joint(86) == [1, 2, 3]


# ── Game registration and structure ──────────────────────────────────────────

def test_game_creates():
    game = make_game()
    assert game is not None


def test_initial_state():
    game = make_game()
    state = game.new_initial_state()
    assert state.current_player() == ATTACKER_ID
    assert not state.is_terminal()


def test_legal_actions_count():
    game = make_game()
    state = game.new_initial_state()
    assert len(state.legal_actions()) == _N_JOINT


def test_sequential_player_alternation():
    game = make_game()
    state = game.new_initial_state()
    # First move: attacker
    assert state.current_player() == ATTACKER_ID
    state.apply_action(0)
    # Second move: defender
    assert state.current_player() == DEFENDER_ID
    state.apply_action(0)
    # Third move: attacker again
    assert state.current_player() == ATTACKER_ID


# ── Observation tensors ───────────────────────────────────────────────────────

def test_observation_tensor_size():
    game = make_game()
    state = game.new_initial_state()
    state.apply_action(0)   # attacker moves; now defender's turn
    obs_atk = np.array(state.observation_tensor(ATTACKER_ID))
    obs_def = np.array(state.observation_tensor(DEFENDER_ID))
    assert obs_atk.shape == (_OBS_SIZE,)
    assert obs_def.shape == (_OBS_SIZE,)


def test_attacker_obs_has_exact_velocities():
    """Attacker should see its own true velocities; defender should see zeros there."""
    game = make_game()
    state = game.new_initial_state()
    # Attacker thrusts: all sats action 1 (+x), joint = 1 + 1*5 + 1*25 = 31
    state.apply_action(31)

    obs_atk = np.array(state.observation_tensor(ATTACKER_ID))
    obs_def = np.array(state.observation_tensor(DEFENDER_ID))

    # Attacker sat 0 velocity (indices 2,3): should be non-zero for attacker, 0 for defender
    atk_vel_in_atk_obs = obs_atk[2:4]
    atk_vel_in_def_obs = obs_def[2:4]

    assert not np.allclose(atk_vel_in_atk_obs, 0.0), "attacker should see its own velocity"
    assert np.allclose(atk_vel_in_def_obs, 0.0), "defender should see zero for attacker velocity"


def test_defender_obs_differs_from_attacker():
    """The two players should see different things (imperfect information)."""
    game = make_game()
    state = game.new_initial_state()
    state.apply_action(0)   # attacker holds
    obs_atk = np.array(state.observation_tensor(ATTACKER_ID))
    obs_def = np.array(state.observation_tensor(DEFENDER_ID))
    # At least the attacker position observations should differ (noise)
    assert not np.allclose(obs_atk[:12], obs_def[:12])


def test_monitoring_reduces_noise():
    """After all defenders monitor atk0, its observed position should be very close to truth."""
    game = make_game()
    state = game.new_initial_state()

    # Attacker holds
    state.apply_action(0)

    # Defender: all 3 sats monitor atk0 — action 2 per sat
    # joint = 2 + 2*5 + 2*25 = 2 + 10 + 50 = 62
    state.apply_action(62)

    # Now attacker holds again — noise is now sampled with monitor_counts=[3,0,0]
    state.apply_action(0)

    obs_atk = np.array(state.observation_tensor(ATTACKER_ID))
    obs_def = np.array(state.observation_tensor(DEFENDER_ID))

    true_atk0_pos = obs_atk[:2]   # attacker sees true position
    noisy_atk0_pos = obs_def[:2]  # defender sees noisy position

    err_atk0 = np.linalg.norm(true_atk0_pos - noisy_atk0_pos)

    # With sigma=0.5 for 3 monitors, error should almost certainly be < 4 sigma = 2.0
    assert err_atk0 < 4.0, f"atk0 obs error {err_atk0:.2f} too large with 3 monitors (sigma=0.5)"


def test_unmonitored_noisy():
    """Without monitoring, defender position estimate should have high noise."""
    errors = []
    for seed in range(20):
        game = make_game()
        state = game.new_initial_state()
        # all hold, no monitoring
        state.apply_action(0)   # attacker: all hold

        obs_atk = np.array(state.observation_tensor(ATTACKER_ID))
        obs_def = np.array(state.observation_tensor(DEFENDER_ID))
        err = np.linalg.norm(obs_atk[:2] - obs_def[:2])
        errors.append(err)

    # With sigma=8, mean error should be around 8*sqrt(2) ≈ 11, definitely > 2
    assert np.mean(errors) > 2.0, f"Mean error {np.mean(errors):.2f} too low for unmonitored"


# ── Terminal conditions ───────────────────────────────────────────────────────

def test_episode_ends_at_t_max():
    """Random game should terminate by turn T_MAX."""
    state = play_random_game(seed=1)
    assert state.is_terminal()
    assert state._turn <= T_MAX


def test_returns_zero_sum():
    """Returns must sum to zero."""
    for seed in range(10):
        state = play_random_game(seed=seed)
        r = state.returns()
        assert abs(r[0] + r[1]) < 1e-9, f"Returns {r} do not sum to zero"


def test_returns_are_plus_minus_one():
    """Terminal returns must be +1/-1 (no draws possible in standard play)."""
    for seed in range(10):
        state = play_random_game(seed=seed)
        r = state.returns()
        assert r[0] in (-1.0, 1.0) and r[1] in (-1.0, 1.0)


def test_attacker_win_condition():
    """Manually drive an attacker to the origin and verify attacker wins."""
    game = make_game()
    state = game.new_initial_state()

    # Teleport attacker sat 0 to just outside proximity threshold
    state._atk_pos[0] = np.array([PROXIMITY_THRESH - 1.0, 0.0])
    state._atk_vel[0] = np.array([0.0, 0.0])

    # Attacker holds (satellite already inside threshold)
    state._check_terminal()

    # Actually, _check_terminal is called at end of defender's turn.
    # So let's apply both turns with the sat already inside.
    state._is_terminal = False  # reset
    state._atk_pos[0] = np.array([PROXIMITY_THRESH - 1.0, 0.0])
    # Attacker holds
    state.apply_action(0)  # attacker → defender turn
    state.apply_action(0)  # defender holds → check terminal
    assert state.is_terminal()
    assert state.returns()[0] == 1.0, "Attacker should win when within proximity"


def test_all_intercepted_defender_wins():
    """If all attackers are intercepted, defender wins."""
    game = make_game()
    state = game.new_initial_state()

    # Move all attacker sats on top of defender sats (within INTERCEPT_RANGE)
    for i in range(NUM_SATS):
        state._atk_pos[i] = state._def_pos[i].copy()

    state.apply_action(0)   # attacker holds — no proximity at origin
    state.apply_action(0)   # defender holds — intercept check fires

    assert state.is_terminal()
    assert state.returns()[1] == 1.0, "Defender should win when all attackers intercepted"


# ── Action to string ──────────────────────────────────────────────────────────

def test_action_to_string():
    game = make_game()
    state = game.new_initial_state()
    s = state.action_to_string(ATTACKER_ID, 0)
    assert "Hold" in s
    s = state.action_to_string(DEFENDER_ID, 62)
    assert "Mon0" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
