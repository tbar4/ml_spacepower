# Lesson 2: Fictitious Play


<!-- toc -->

## Where this fits

The previous lesson established that independent Q-learning fails in multi-agent settings because agents treat each other as part of a stationary environment when in fact both are learning simultaneously. Fictitious play is the oldest and simplest algorithm that takes other agents' behavior into account explicitly. Instead of ignoring the other agent, each player tracks the empirical frequency of the opponent's past actions and best-responds to that historical average.

Fictitious play is not a deep learning algorithm. It is a simple tabular procedure, and it is best understood as a conceptual foundation. It precedes PSRO (the next lesson), which generalizes fictitious play to neural-network policies. It also shares a structural similarity to CFR (Module 5): both average past strategies rather than using the current strategy directly. Understanding fictitious play makes the intuition behind both PSRO and CFR more transparent.

## The algorithm

Fictitious play is defined for a normal-form game: two (or more) players repeatedly play the same game. At each round, each player:

1. Looks at the **empirical frequency** of the opponent's historical action choices — the fraction of past rounds the opponent played each action.
2. Computes the **best response** to that empirical frequency, treating it as a fixed mixed strategy.
3. Plays the best response (or any best response, if there are ties).

The empirical frequency after round t is a mixed strategy that reflects all observed play. As t grows, this empirical frequency converges (in well-behaved games) to a Nash equilibrium mixed strategy.

More formally: let \(\hat{\sigma}_{-i}^t\) be the empirical frequency of player \(-i\)'s actions through round t. Player \(i\)'s strategy at round \(t+1\) is:

\[ \sigma_i^{t+1} \in \arg\max_{a_i} \; u_i(a_i, \hat{\sigma}_{-i}^t) \]

**Decoding:**
- \(\hat{\sigma}_{-i}^t\): the empirical frequency (count vector normalized to sum to 1) of the opponent's past actions through round \(t\)
- \(\sigma_i^{t+1}\): the strategy player \(i\) will play in round \(t+1\)
- \(\arg\max_{a_i}\): the action (or set of actions) that maximizes the expression
- \(u_i(a_i, \hat{\sigma}_{-i}^t)\): the expected payoff to player \(i\) from action \(a_i\), when the opponent plays the historical average \(\hat{\sigma}_{-i}^t\)

The key quantity tracked by each player is the **action count**: how many times each of the opponent's actions has been played. Normalizing the count gives the empirical frequency.

```python
import numpy as np

def fictitious_play(payoff_matrix, n_rounds=1000, seed=0):
    """
    Fictitious play for a two-player normal-form game.

    payoff_matrix: shape (n_actions_p1, n_actions_p2, 2)
        payoff_matrix[a1, a2, 0] = Player 1's payoff when (a1, a2) is played
        payoff_matrix[a1, a2, 1] = Player 2's payoff when (a1, a2) is played

    Returns the empirical frequencies for both players over all rounds.
    """
    np.random.seed(seed)
    n1 = payoff_matrix.shape[0]  # number of actions for Player 1
    n2 = payoff_matrix.shape[1]  # number of actions for Player 2

    # Action counts: how many times each opponent action has been observed
    count1 = np.ones(n1)   # Player 2's counts for Player 1's actions (prior: 1 each)
    count2 = np.ones(n2)   # Player 1's counts for Player 2's actions

    history1 = []   # Player 1's action sequence
    history2 = []   # Player 2's action sequence

    for t in range(n_rounds):
        # Empirical frequencies (normalize counts)
        freq1 = count1 / count1.sum()   # Player 2's belief about Player 1
        freq2 = count2 / count2.sum()   # Player 1's belief about Player 2

        # Player 1 best-responds to freq2 (Player 2's empirical frequency)
        # Expected payoff for each action of Player 1:
        # E[u1(a1)] = sum_{a2} freq2[a2] * payoff_matrix[a1, a2, 0]
        eu1 = payoff_matrix[:, :, 0] @ freq2       # shape (n1,)
        a1_candidates = np.where(eu1 == eu1.max())[0]
        a1 = np.random.choice(a1_candidates)        # break ties randomly

        # Player 2 best-responds to freq1 (Player 1's empirical frequency)
        eu2 = payoff_matrix[:, :, 1].T @ freq1     # shape (n2,)
        a2_candidates = np.where(eu2 == eu2.max())[0]
        a2 = np.random.choice(a2_candidates)

        # Update action counts
        count2[a2] += 1   # Player 1 observed Player 2 play a2
        count1[a1] += 1   # Player 2 observed Player 1 play a1

        history1.append(a1)
        history2.append(a2)

    # Final empirical frequencies are the approximate Nash strategies
    final_freq1 = count1 / count1.sum()
    final_freq2 = count2 / count2.sum()
    return final_freq1, final_freq2, history1, history2
```

The count initialization to `np.ones(n)` rather than zeros implements a weak uniform prior. This prevents division by zero at the start and reflects the reasonable assumption that we have no strong prior belief about what the opponent will do before observing any play.

## Why it works

Intuitively: if the opponent's empirical frequency has converged to some fixed mixed strategy \(\hat{\sigma}\), then best-responding to \(\hat{\sigma}\) is also best-responding to the limit, which gives a fixed point. At a fixed point, neither player can improve their expected payoff by changing their action — that is a Nash equilibrium.

The formal result: **in two-player zero-sum games (and in two-player games with identical payoffs, i.e., coordination games), the time-averaged strategy profile produced by fictitious play converges to Nash equilibrium.**

The time-averaged strategy at round T is:

\[ \bar{\sigma}_i^T = \frac{1}{T} \sum_{t=1}^{T} \sigma_i^t \]

**Decoding:**
- \(\bar{\sigma}_i^T\): the time-averaged strategy for player \(i\) through round \(T\)
- \(\sigma_i^t\): the actual action played at round \(t\) (encoded as a one-hot vector, or as a probability distribution when ties are broken randomly)
- The sum averages out the variability in best responses from round to round

This time-averaged strategy is what converges, not the actual actions played at each round. The actions themselves may oscillate, but the running average settles.

Why does convergence require zero-sum (or coordination)? Because in general-sum games, best responses to empirical frequencies can cycle perpetually without converging. The next section shows this with an example.

## SSA example: radar tasking as a two-player game

Consider two satellite operators, Red and Blue, each with access to a high-powered space-surveillance radar for 4 hours per night. The radar can observe one of three orbital regimes during each session: LEO (regime 0), MEO (regime 1), or GEO (regime 2).

Red wants to know what Blue's satellites are doing; Blue wants to deny Red that information by keeping its activity in regimes Red is not watching. Blue also wants to observe Red's satellites. This has the structure of a zero-sum pursuit-evasion game over orbital regimes.

Simplified payoff matrix (Red's reward; Blue's reward is the negative):

```
            Blue: LEO    Blue: MEO    Blue: GEO
Red: LEO     +1           -1           -1
Red: MEO     -1           +1           -1
Red: GEO     -1           -1           +1
```

Red wants to match Blue's regime (intercept); Blue wants to mismatch (evade). This is a generalization of matching pennies to three actions.

The Nash equilibrium is for both players to randomize uniformly: play each regime with probability 1/3. Fictitious play should converge to this.

```python
import numpy as np

# Payoff matrix: [Red action, Blue action, player]
# Red gets +1 for matching, -1 for mismatching. Blue is the negative.
n = 3  # LEO, MEO, GEO

payoff = np.zeros((n, n, 2))
for i in range(n):
    for j in range(n):
        if i == j:
            payoff[i, j, 0] = +1.0   # Red intercepts
            payoff[i, j, 1] = -1.0   # Blue fails to evade
        else:
            payoff[i, j, 0] = -1.0   # Red misses
            payoff[i, j, 1] = +1.0   # Blue evades

freq1, freq2, hist1, hist2 = fictitious_play(payoff, n_rounds=2000, seed=42)

print("=== Radar Tasking Game (zero-sum) ===")
print(f"Red empirical frequency:  LEO={freq1[0]:.3f}  MEO={freq1[1]:.3f}  GEO={freq1[2]:.3f}")
print(f"Blue empirical frequency: LEO={freq2[0]:.3f}  MEO={freq2[1]:.3f}  GEO={freq2[2]:.3f}")
print("Nash equilibrium: all three regimes at probability 1/3 = 0.333")

# Check convergence over time
window_sizes = [100, 500, 1000, 2000]
print("\nConvergence of Red's LEO frequency over rounds:")
for w in window_sizes:
    freq = hist1[:w].count(0) / w
    print(f"  After {w:4d} rounds: LEO freq = {freq:.3f}  (target 0.333)")
```

After a few hundred rounds, both players' empirical frequencies should approach 1/3 for each regime, reflecting the Nash mixed strategy. The convergence is not monotone — players may oversample a regime for a while before correcting — but the time-average converges.

```rust
# extern crate rand;
// rand = "0.10"
use rand::{Rng, RngExt, SeedableRng};

// 3×3 radar tasking game: Red gets +1 for matching regime, -1 for mismatching.
// Blue wants to mismatch. Best response = regime with highest expected payoff.
fn best_response(rng: &mut impl Rng, freq_opp: &[f64; 3], want_match: bool) -> usize {
    let eu: [f64; 3] = [0, 1, 2].map(|i| {
        (0..3_usize).map(|j| {
            let raw = if i == j { 1.0_f64 } else { -1.0_f64 };
            (if want_match { raw } else { -raw }) * freq_opp[j]
        }).sum()
    });
    let max_eu = eu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let candidates: Vec<usize> = (0..3).filter(|&i| (eu[i] - max_eu).abs() < 1e-10).collect();
    // Break ties uniformly
    candidates[(rng.random::<f64>() * candidates.len() as f64) as usize]
}

fn main() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    // Uniform prior (1 observation of each regime before any play)
    let mut count_red  = [1.0_f64; 3];   // Blue's belief about Red
    let mut count_blue = [1.0_f64; 3];   // Red's belief about Blue
    let mut hist_red   = Vec::new();

    for _ in 0..2000 {
        let s_r: f64 = count_red.iter().sum();
        let s_b: f64 = count_blue.iter().sum();
        let freq_red  = [count_red[0]/s_r,  count_red[1]/s_r,  count_red[2]/s_r];
        let freq_blue = [count_blue[0]/s_b, count_blue[1]/s_b, count_blue[2]/s_b];

        let a_red  = best_response(&mut rng, &freq_blue, true);   // Red wants match
        let a_blue = best_response(&mut rng, &freq_red,  false);  // Blue wants mismatch

        count_blue[a_blue] += 1.0;
        count_red[a_red]   += 1.0;
        hist_red.push(a_red);
    }

    let s_r: f64 = count_red.iter().sum();
    let s_b: f64 = count_blue.iter().sum();
    println!("Red  empirical: LEO={:.3}  MEO={:.3}  GEO={:.3}",
             count_red[0]/s_r, count_red[1]/s_r, count_red[2]/s_r);
    println!("Blue empirical: LEO={:.3}  MEO={:.3}  GEO={:.3}",
             count_blue[0]/s_b, count_blue[1]/s_b, count_blue[2]/s_b);
    println!("Nash target:    LEO=0.333  MEO=0.333  GEO=0.333");

    for &w in &[100_usize, 500, 1000, 2000] {
        let leo = hist_red[..w].iter().filter(|&&a| a == 0).count() as f64 / w as f64;
        println!("  After {:4} rounds: Red LEO freq = {:.3}  (target 0.333)", w, leo);
    }
}
```

## When fictitious play fails: cyclic games

Fictitious play does not converge in all games. The canonical failure case is **Rock-Paper-Scissors** (or its SSA analog).

In zero-sum games where best responses cycle — A beats B, B beats C, C beats A — fictitious play also cycles: each player's best response keeps changing. The empirical frequency converges (because cycles are regular), but the actual actions played never settle.

More importantly: in **general-sum games** (not zero-sum), fictitious play can fail to converge entirely. The empirical frequency itself may cycle rather than converge. This is a fundamental limitation.

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rock-Paper-Scissors: payoff from Player 1's perspective
# Rows: P1 action (R=0, P=1, S=2)
# Cols: P2 action (R=0, P=1, S=2)
RPS_PAYOFF = np.array([
    [[0, 0], [-1, +1], [+1, -1]],   # P1 plays Rock
    [[+1, -1], [0, 0], [-1, +1]],   # P1 plays Paper
    [[-1, +1], [+1, -1], [0, 0]],   # P1 plays Scissors
], dtype=float)

def track_convergence(payoff_matrix, n_rounds=3000, seed=1):
    """
    Run fictitious play and track the trajectory of empirical frequencies.
    Returns arrays of shape (n_rounds, n_actions) for each player.
    """
    np.random.seed(seed)
    n1 = payoff_matrix.shape[0]
    n2 = payoff_matrix.shape[1]

    count1 = np.ones(n1)
    count2 = np.ones(n2)

    traj1 = []
    traj2 = []

    for _ in range(n_rounds):
        freq1 = count1 / count1.sum()
        freq2 = count2 / count2.sum()

        eu1 = payoff_matrix[:, :, 0] @ freq2
        a1 = np.random.choice(np.where(eu1 == eu1.max())[0])

        eu2 = payoff_matrix[:, :, 1].T @ freq1
        a2 = np.random.choice(np.where(eu2 == eu2.max())[0])

        count2[a2] += 1
        count1[a1] += 1

        traj1.append(count1 / count1.sum())
        traj2.append(count2 / count2.sum())

    return np.array(traj1), np.array(traj2)


traj1, traj2 = track_convergence(RPS_PAYOFF, n_rounds=3000, seed=1)

print("=== Rock-Paper-Scissors ===")
print("Nash equilibrium: (1/3, 1/3, 1/3) for both players")
print(f"After 3000 rounds, Player 1 empirical freq: {traj1[-1].round(3)}")
print(f"After 3000 rounds, Player 2 empirical freq: {traj2[-1].round(3)}")
print("These should be near (0.333, 0.333, 0.333) despite cycling behavior")

# Illustrate how individual action frequencies evolve over rounds
# The time-averaged frequencies converge even as actions cycle
rounds = np.arange(1, 3001)
labels = ["Rock", "Paper", "Scissors"]
print("\nPlayer 1 frequency trajectory snapshots:")
for checkpoint in [100, 500, 1000, 3000]:
    freqs = traj1[checkpoint - 1]
    print(f"  Round {checkpoint:4d}: {dict(zip(labels, freqs.round(3)))}")
```

The output shows that in RPS, the empirical frequencies do converge toward (1/3, 1/3, 1/3), but the path oscillates. At early rounds, players over-index on one action (because the opponent played it a lot) and then swing to over-index on another. The convergence is spiral rather than monotone.

The important distinction: **the time-averaged frequency converges, but the actual actions cycle.** Fictitious play gives you the Nash mixed strategy as the limit of the average, not as a stable point that the agents actually settle at. If you deployed the agents after a finite number of rounds and asked "what will they play next?", the answer would still be cycling.

## Simultaneous vs. sequential fictitious play

The standard fictitious play described above is **simultaneous**: both players update their strategies at the same time each round, based on the same historical record.

**Sequential fictitious play** updates one player at a time: Player 1 updates first using the current history, then Player 2 updates using the now-updated history (which includes Player 1's new action). This asymmetry makes the dynamics easier to analyze and tends to converge faster in practice.

In sequential fictitious play, there is a natural leader and follower at each round. The leader best-responds first; the follower best-responds to the leader's actual action (not just the historical average). This is related to Stackelberg equilibrium in game theory — the follower has an advantage in information.

For the radar tasking game, sequential fictitious play would have Red act first, then Blue observe Red's choice and respond. This speeds up convergence but changes the equilibrium concept slightly: Blue now has a best-response advantage.

The choice between simultaneous and sequential depends on the application. For symmetric games where neither player has a first-mover advantage, simultaneous is the natural model. For asymmetric games (one player commits before the other), sequential is more realistic.

## Connection to CFR

Fictitious play and CFR (Counterfactual Regret Minimization, Module 5) share a structural similarity that is worth making explicit.

In fictitious play, each player maintains the **average past strategy** (empirical frequency) and best-responds to it. The convergence guarantee comes from the fact that the average stabilizes even when individual best responses oscillate.

In CFR, each player maintains **cumulative regrets** and sets their current strategy proportional to positive regrets. The convergence guarantee comes from the fact that the time-averaged strategy has diminishing regret.

Both algorithms:
- Use the **time-average** of past strategies, not the current strategy, as the output
- Converge to Nash in zero-sum two-player games
- Can fail to converge in general-sum multi-player games
- Are tabular algorithms that work with action counts or regret counts

The key difference: CFR uses a more sophisticated update rule (regret matching instead of best response) that gives better convergence bounds (\(O(1/\sqrt{T})\) for both) and works for extensive-form games with imperfect information. Fictitious play applies only to normal-form games and uses pure best responses, which can cause oscillation in the actual actions even when the average converges.

PSRO (next lesson) can be seen as a generalization of fictitious play where the "actions" are entire neural-network policies rather than individual moves. The best response computation becomes a full RL training run rather than a simple argmax.

## Convergence verification

One practical way to check if fictitious play is converging is to compute the **exploitability** of the empirical frequency: how much could a best-responding opponent gain against the current empirical frequency? At a Nash equilibrium, exploitability is zero.

```python
def exploitability(payoff_matrix, freq1, freq2):
    """
    Compute the sum of exploitability for both players.
    A Nash equilibrium has exploitability = 0.
    """
    n1, n2 = payoff_matrix.shape[:2]

    # Best response value for Player 1 against freq2
    eu1 = payoff_matrix[:, :, 0] @ freq2
    br_value_1 = eu1.max()
    current_value_1 = freq1 @ eu1

    # Best response value for Player 2 against freq1
    eu2 = payoff_matrix[:, :, 1].T @ freq1
    br_value_2 = eu2.max()
    current_value_2 = freq2 @ eu2

    # Exploitability: how much each player could gain by deviating
    exploit_1 = max(0.0, br_value_1 - current_value_1)
    exploit_2 = max(0.0, br_value_2 - current_value_2)
    return exploit_1 + exploit_2


# Track exploitability over rounds for the radar tasking game
traj1_radar, traj2_radar = track_convergence(payoff, n_rounds=2000, seed=42)

print("=== Exploitability over rounds (radar tasking game) ===")
for checkpoint in [10, 50, 200, 500, 1000, 2000]:
    f1 = traj1_radar[checkpoint - 1]
    f2 = traj2_radar[checkpoint - 1]
    e = exploitability(payoff, f1, f2)
    print(f"  Round {checkpoint:4d}: exploitability = {e:.4f}")
```

The exploitability should decrease monotonically (in expectation) as rounds increase. By round 2000, it should be close to zero for the zero-sum radar tasking game.

## A worked comparison: convergence speed across game types

Not all zero-sum games converge at the same rate. Games where one strategy strongly dominates (high payoff differential) converge faster than games where payoffs are nearly equal (the empirical frequency needs many samples to distinguish). This section compares convergence for three game structures relevant to SSA.

```python
import numpy as np

def run_fp_and_measure(payoff_matrix, n_rounds, seed=0):
    """Run fictitious play and return exploitability trajectory."""
    np.random.seed(seed)
    n1 = payoff_matrix.shape[0]
    n2 = payoff_matrix.shape[1]
    count1 = np.ones(n1)
    count2 = np.ones(n2)
    exploit_history = []

    for t in range(n_rounds):
        freq1 = count1 / count1.sum()
        freq2 = count2 / count2.sum()

        eu1 = payoff_matrix[:, :, 0] @ freq2
        a1 = np.random.choice(np.where(eu1 == eu1.max())[0])

        eu2 = payoff_matrix[:, :, 1].T @ freq1
        a2 = np.random.choice(np.where(eu2 == eu2.max())[0])

        count2[a2] += 1
        count1[a1] += 1

        if (t + 1) % 100 == 0:
            f1 = count1 / count1.sum()
            f2 = count2 / count2.sum()
            e = exploitability(payoff_matrix, f1, f2)
            exploit_history.append((t + 1, e))

    return exploit_history


# Game 1: Radar tasking (zero-sum, 3 regimes, equal payoffs)
# Already defined as `payoff` above

# Game 2: Frequency deconfliction (zero-sum, 4 channels, unequal payoffs)
# One player wants to transmit; other wants to jam on the same channel.
# Asymmetric costs: jamming GEO downlink (channel 3) is worth more.
def make_freq_game():
    n = 4
    base_payoff = np.array([
        [[-1, +1], [+1, -1], [+1, -1], [+1, -1]],
        [[+1, -1], [-1, +1], [+1, -1], [+1, -1]],
        [[+1, -1], [+1, -1], [-1, +1], [+1, -1]],
        [[+2, -2], [+2, -2], [+2, -2], [-2, +2]],  # channel 3 is high-value
    ], dtype=float)
    return base_payoff

# Game 3: Coverage priority (general-sum, 3 regimes)
# Both operators want coverage of the same arc, but overlap is wasteful.
# This is NOT zero-sum and fictitious play may not converge cleanly.
def make_coverage_game():
    n = 3
    payoff = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                # Both cover same regime: overlap wastes resources
                payoff[i, j, 0] = -0.5
                payoff[i, j, 1] = -0.5
            else:
                # Different regimes: both get positive coverage
                payoff[i, j, 0] = +1.0
                payoff[i, j, 1] = +1.0
    return payoff


freq_game = make_freq_game()
coverage_game = make_coverage_game()

print("=== Exploitability convergence comparison ===\n")

print("Round    Radar(ZS)  Freq(ZS)  Coverage(GS)")
print("-" * 45)

hist_radar = run_fp_and_measure(payoff, 2000, seed=3)
hist_freq = run_fp_and_measure(freq_game, 2000, seed=3)
hist_cov = run_fp_and_measure(coverage_game, 2000, seed=3)

for idx in range(len(hist_radar)):
    t = hist_radar[idx][0]
    if t in [100, 500, 1000, 2000]:
        e_r = hist_radar[idx][1]
        e_f = hist_freq[idx][1]
        e_c = hist_cov[idx][1]
        print(f"  {t:4d}    {e_r:8.4f}   {e_f:8.4f}  {e_c:10.4f}")

print()
print("The zero-sum games (Radar, Freq) converge to near-zero exploitability.")
print("The general-sum coverage game may retain residual exploitability (cycling).")
```

The output illustrates three important points:

**Zero-sum games converge reliably.** Both the radar tasking and frequency deconfliction games (both zero-sum) reach near-zero exploitability within a few hundred rounds. The frequency game with unequal payoffs on channel 3 converges slightly faster because the payoff gradient is steeper and the best response is less ambiguous.

**General-sum games may not converge.** The coverage game has a general-sum structure (both players can win or both can lose depending on whether they overlap). Fictitious play oscillates: each operator alternates between regimes in response to the other. Exploitability stays elevated. This is the fundamental limitation that motivates PSRO (next lesson), which handles general-sum settings more robustly.

**Convergence speed reflects payoff informativeness.** In games where best responses are clear (large payoff differences between actions), fictitious play quickly settles. In games with nearly equal payoffs, many rounds are needed before the empirical frequency is statistically reliable enough to identify the Nash mixture accurately.

## Key Takeaways

- **Fictitious play is the simplest multi-agent learning algorithm that accounts for the opponent.** Each player tracks the empirical frequency of the opponent's historical actions and best-responds to it. This avoids the non-stationarity of treating the opponent as a fixed environment while remaining computationally trivial.
- **Convergence is guaranteed for zero-sum two-player games, not in general.** In zero-sum games, the time-averaged strategies converge to Nash equilibrium. In general-sum games, fictitious play can cycle. This is the primary limitation that motivates more sophisticated algorithms.
- **The time-averaged strategy converges, not the actual actions.** Individual actions may oscillate perpetually (as in RPS) while the running average converges to the Nash mixed strategy. The algorithm's output is the average, not the final action.
- **Fictitious play is a conceptual ancestor of PSRO.** PSRO generalizes fictitious play by replacing individual actions with entire neural-network policies and replacing argmax best response with a full RL training run. Understanding fictitious play makes PSRO's structure intuitive.
- **Fictitious play and CFR share the same averaging insight.** Both converge through the time-average of strategies rather than the instantaneous strategy. The difference is the update rule: fictitious play uses pure best response; CFR uses regret matching, giving better theoretical guarantees for extensive-form games.
- **Exploitability is the right convergence diagnostic.** Unlike monitoring action frequencies, exploitability directly measures how far the current empirical frequency is from Nash equilibrium. A converged fictitious play run has near-zero exploitability for both players.

## Quiz

{{#quiz 02-fictitious-play.toml}}
