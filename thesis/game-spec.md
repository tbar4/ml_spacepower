# Orbital Proximity Competition (OPC) — Game Specification

**Thesis**: From Self-Play to Strategy: Machine Learning-Derived Tactics in Multi-Satellite Orbital Competition

**Version**: 0.1 (initial spec)

---

## Overview

Two-player, zero-sum, sequential, imperfect-information game implemented in OpenSpiel. An attacker fleet attempts to achieve proximity to a defended asset; a defender fleet allocates sensor coverage and maneuvers to prevent it.

---

## Players

- **Attacker**: controls 3 satellites
- **Defender**: controls 3 satellites plus a defended asset fixed at the origin

---

## State Space

| Element | Representation |
|---|---|
| 3 attacker satellites | position (x, y), velocity (dx, dy) each |
| 3 defender satellites | position (x, y), velocity (dx, dy) each |
| Sensor allocation | which defender satellites are monitoring which attacker satellites |
| Turn counter | integer 0 to T |

All positions are expressed as (along-track, cross-track) offsets in the Hill-Clohessy-Wiltshire frame relative to the defended asset at the origin.

---

## Turn Structure

Sequential. Attacker moves first each turn, then Defender. Episode length T = 50 turns.

---

## Action Space

**Attacker** (per satellite, 5 options):
- Hold
- Thrust toward asset
- Thrust away from asset
- Thrust lateral left
- Thrust lateral right

Joint attacker action space: 5^3 = 125 actions per turn.

**Defender** (per satellite, 5 options):
- Hold
- Reposition (one step away from nearest attacker satellite)
- Monitor attacker satellite 1
- Monitor attacker satellite 2
- Monitor attacker satellite 3

Joint defender action space: 5^3 = 125 actions per turn.

---

## Information Structure

The defender observes all attacker positions but with Gaussian noise. Noise variance is inversely proportional to the number of defender satellites currently monitoring each attacker satellite.

- Attacker satellite monitored by 2 defender satellites: low positional uncertainty
- Attacker satellite monitored by 1 defender satellite: moderate positional uncertainty
- Unmonitored attacker satellite: maximum positional uncertainty

This forces the defender to choose between broad coverage (noisy estimates of all three attackers) and concentrated monitoring (precise track on one or two, blind to the rest). The attacker can exploit unmonitored gaps.

---

## Win Conditions

| Outcome | Condition |
|---|---|
| Attacker wins | Any attacker satellite reaches distance < 10 units from origin before turn 50 |
| Defender wins | No attacker satellite achieves proximity within 50 turns |

---

## Reward Structure

Zero-sum. No intermediate rewards.

| Player | Win | Otherwise |
|---|---|---|
| Attacker | +1 | 0 |
| Defender | +1 | 0 |

---

## Design Decisions and Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Orbital representation | 2D HCW frame | Tractable; captures approach geometry without physics simulator |
| Move structure | Sequential (attacker then defender) | Simpler OpenSpiel implementation; captures strategic dynamics |
| Episode length | 50 turns | Long enough for multi-step approach campaigns; short enough to train |
| Fleet size | 3v3 | Sufficient for fleet coordination dynamics; tractable action space |
| Intermediate rewards | None | Clean credit assignment for AlphaZero self-play |

---

## Extensions Reserved for Future Work

- 3D orbital plane (out-of-plane maneuvers, inclination changes)
- Simultaneous moves
- Asymmetric fleet sizes
- Variable maneuver fuel costs
- Stochastic sensor failures
- Multi-sided competition (3+ players, doctoral direction)

---

## Implementation Path

1. Module 2 (neural networks): foundations for AlphaZero value and policy networks
2. Module 4 (MCTS and AlphaZero): the training algorithm
3. Module 8 Lessons 1-2 (OpenSpiel architecture, custom game implementation): the game framework
4. Module 8 Lesson 5 (Ray RLlib): distributed self-play at scale
