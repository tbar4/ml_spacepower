# Lesson 4: Designing the SSA Game

## Where this fits

This is the last lesson before the capstone. The capstone implements one specific game; this lesson designs that game and explains the design choices. The aim is for you to be able to extend or replace the game with one that better suits your eventual thesis work, knowing what the design constraints are. Once you have read this lesson, the capstone is essentially execution.

The design problem is constrained: the game must be small enough to solve with vanilla CFR (so you can verify correctness against a tabular oracle), rich enough to require non-trivial mixed strategies (so the solution is interesting), and structured enough to have a clear SSA interpretation (so the work is connected to your research direction). Threading those needles is most of the work.

## The scenario: conjunction maneuver masking

Two operators share a region of orbital space. The **Adversary** (player 0) controls a satellite that may execute a covert maneuver to alter its conjunction geometry with a third-party object. The **Defender** (player 1) operates a sensor network that can be tuned to detect different kinds of orbital perturbations.

The Adversary wants to maneuver if doing so produces an advantageous conjunction geometry, but does not want the maneuver to be detected (detection triggers diplomatic and operational consequences). The Defender wants to detect any maneuver but has limited sensor capacity to distribute across detection modes.

### Single-shot structure

We are designing a single-shot game (one decision per player) rather than a multi-step game. This is a deliberate simplification:
- Vanilla CFR scales as the square of the number of information sets times game length. Multi-step games blow up quickly.
- The strategic structure (Adversary picks a maneuver intensity, Defender picks an allocation, both with hidden information about the situation) is preserved in the single-shot version.
- A multi-step extension would be a natural follow-up project but is out of scope here.

If you later want a multi-step variant, the right approach is MCCFR or deep CFR, both of which scale better. The capstone implements both, so the multi-step extension is incremental.

## The game tree

The game proceeds in 4 stages:

**Stage 1 (chance)**: Nature determines the **opportunity** for the Adversary. With probability 0.6, there is a routine situation (no compelling reason to maneuver). With probability 0.4, there is an opportunity (a maneuver would significantly improve the Adversary's conjunction geometry). The Adversary observes this; the Defender does not.

**Stage 2 (Adversary action)**: The Adversary chooses a **maneuver intensity** from {None (0), Light (1), Heavy (2)}. None is undetectable but provides no benefit. Light is harder to detect but provides only a small benefit. Heavy is most beneficial but easiest to detect.

**Stage 3 (Defender action)**: The Defender does not see the Adversary's choice or the opportunity. They choose a **sensor allocation** from {Wide (0), Narrow (1), Off (2)}. Wide allocation can detect any maneuver but with lower probability. Narrow can detect heavy maneuvers reliably but misses light ones. Off detects nothing.

**Stage 4 (chance)**: A noisy detection event resolves. The probability of detection depends on the (intensity, allocation) pair, given by:

| Intensity \ Allocation | Wide | Narrow | Off |
|----------------------|------|--------|-----|
| None                 | 0.05 | 0.05   | 0.0 |
| Light                | 0.50 | 0.30   | 0.0 |
| Heavy                | 0.65 | 0.85   | 0.0 |

The "None" probabilities are nonzero to model false positives. The Defender sees only the detection result, not the underlying truth.

## The payoff matrix

Payoffs depend on (opportunity, intensity, detection):

For the **Adversary**:
- If opportunity AND no detection: maneuver benefit = +intensity (so Heavy = +2, Light = +1, None = 0)
- If opportunity AND detection: maneuver benefit but exposed = -3 (any intensity)
- If no opportunity AND no detection: 0 (nothing happens, no harm done)
- If no opportunity AND detection: 0 unless they actually maneuvered, in which case -2 (caught maneuvering for no reason)

For the **Defender** (zero-sum, so just negate the Adversary's payoff).

A payoff table (Adversary's perspective):

| Opportunity | Intensity | Detected? | Adv reward |
|-------------|-----------|-----------|------------|
| Yes         | None      | -         | 0          |
| Yes         | Light     | No        | +1         |
| Yes         | Light     | Yes       | -3         |
| Yes         | Heavy     | No        | +2         |
| Yes         | Heavy     | Yes       | -3         |
| No          | None      | -         | 0          |
| No          | Light     | No        | 0          |
| No          | Light     | Yes       | -2         |
| No          | Heavy     | No        | 0          |
| No          | Heavy     | Yes       | -2         |

Defender rewards are the negation. Note: "no opportunity, light/heavy maneuver, no detection" gives 0 because nothing useful happened (no opportunity to exploit). The Adversary spent a maneuver budget for no reason.

## Strategic intuition

You should be able to reason qualitatively about what the Nash equilibrium looks like, before solving:

- The Adversary should never maneuver Heavy when there is no opportunity: the upside is 0 and the downside is -2.
- The Adversary should mix on Light vs. Heavy when there is an opportunity, because Heavy has higher upside but is more detectable. The exact mix depends on what allocation the Defender plays.
- The Defender should not always play Off: the Adversary would then always Heavy on opportunity and get +2 reliably. Some watchful allocation is needed.
- The Defender should not always play Wide or always Narrow: the Adversary would shift their intensity choice to the harder-to-detect option.

The result is a mixed-strategy equilibrium where both players randomize over multiple actions.

## Information sets

Now we enumerate information sets, because these are what CFR will operate on.

**Adversary information sets** (2 total): the Adversary sees only the opportunity.
- "opp=Yes": the opportunity exists. Action choice: {None, Light, Heavy}.
- "opp=No": no opportunity. Action choice: {None, Light, Heavy}.

**Defender information sets** (1 total before observation, plus terminal handling): the Defender sees nothing before acting.
- "": no information yet. Action choice: {Wide, Narrow, Off}.

So there are 3 information sets across both players. CFR will maintain a regret table and a strategy table for each. With 3 actions per information set, the strategy is a probability vector of length 3 per information set. Total: 9 strategy parameters across 3 info sets. Tiny.

This is small enough that vanilla CFR converges in a handful of iterations and you can verify correctness against analytical computation. It is also small enough that a tabular `HashMap<String, [f64; 3]>` representation works fine.

## Why this is a good capstone game

Several reasons:

1. **Solvable analytically**: With 9 strategy parameters, you can write down the equilibrium conditions and solve them as a small linear program. (You won't need to, but you could verify against this.)
2. **CFR-tractable**: Vanilla tabular CFR converges to negligible exploitability in well under 10,000 iterations.
3. **Mixed-strategy equilibrium**: The equilibrium genuinely requires randomization, so you see CFR producing non-degenerate strategies, not just identifying a pure strategy.
4. **SSA-meaningful payoff structure**: Each table entry has an intuitive justification grounded in the scenario. You are not optimizing an abstract reward function; you are computing equilibrium behavior in a recognizable adversarial space situation.
5. **Extension-ready**: The game can be made larger (more intensities, more allocations, multi-shot) without changing the algorithm structure. Deep CFR (the second part of the capstone) handles the larger versions.

## What deep CFR adds

Vanilla CFR maintains a HashMap. With 3 information sets, the table is trivially small. To exercise the deep CFR pathway, the capstone includes a "scaled" variant of the game with:

- 7 maneuver intensity levels (instead of 3)
- 5 sensor allocation modes (instead of 3)
- 4 chance opportunity types (instead of 2)

This produces a few dozen information sets, more action choices per set, and detection probability tables that are larger. Still small in absolute terms, but large enough that the neural network's interpolation behavior is observable: with a few thousand data points the network learns useful regret approximations.

The point is not that the scaled game is too large for tabular CFR (it is not). The point is that you can see the deep CFR mechanics working on a problem where you can also run tabular CFR and check that the answers match. This is the right pedagogical structure: build deep CFR where you can verify it.

## State representation in code

For the capstone, the State struct will contain:

```rust
pub struct GameState {
    /// Hidden state: the opportunity drawn at stage 1.
    /// None until chance node resolves.
    opportunity: Option<Opportunity>,
    
    /// Adversary action, if taken.
    adversary_action: Option<Intensity>,
    
    /// Defender action, if taken.
    defender_action: Option<Allocation>,
    
    /// Detection result, if resolved.
    detection: Option<bool>,
}
```

Information state strings:

- Adversary: `format!("opp={:?}", self.opportunity)` (always set when Adversary acts)
- Defender: `""` (Defender has no information at decision time)

Action enumeration is straightforward: each phase has a fixed action set. Chance outcomes have known probabilities.

The Game trait we will define:

```rust
pub trait Game {
    type State: GameState;
    fn new_initial_state(&self) -> Self::State;
    fn num_players(&self) -> usize;
    fn num_distinct_actions(&self) -> usize;
}

pub trait GameState {
    fn current_player(&self) -> Player;  // Chance, Player(usize), or Terminal
    fn legal_actions(&self) -> Vec<usize>;
    fn chance_outcomes(&self) -> Vec<(usize, f64)>;  // for chance nodes
    fn apply_action(&mut self, action: usize);
    fn information_state_string(&self, player: usize) -> String;
    fn information_state_tensor(&self, player: usize) -> Vec<f32>;
    fn is_terminal(&self) -> bool;
    fn is_chance_node(&self) -> bool;
    fn returns(&self) -> Vec<f64>;
    fn clone_state(&self) -> Self;
}
```

This mirrors the OpenSpiel pattern from lesson 1. The Rust generics let us specialize the State type per game while keeping the algorithm code generic.

## A note on cloning

OpenSpiel's `state.clone()` returns a `pyspiel.State` and the recursion just works in Python. In Rust, you have to be more deliberate about cloning. We use a `clone_state()` method on the trait (rather than the standard `Clone` trait) because the state contains owned data (HashMaps, Vecs in the more complex variants) and cloning needs to be intentional.

For CFR to work, you need to be able to clone the state at every traversal. For our small game, this is cheap. For large games, you might use a more efficient representation (e.g., immutable persistent data structures with structural sharing), but the small-game approach is simpler and sufficient.

## What the capstone will build

The capstone (the project file for this module) walks through, in order:

1. Setting up the Cargo workspace with three crates.
2. Implementing the Game and State traits for the basic SSA game.
3. Writing tabular CFR over the trait.
4. Computing exploitability via best-response calculation.
5. Verifying that exploitability drops to ~0 over training.
6. Defining the scaled game variant.
7. Implementing deep CFR using burn (network architecture + training loop + sampling).
8. Verifying that deep CFR's strategies match tabular CFR's on the small game (sanity check).
9. Building a CLI that runs everything and produces output you can inspect.

The pedagogy is: build it small and tabular first (you can verify every number by hand if needed), then add the deep CFR scaffolding (you can compare against the tabular ground truth), then scale up the game (the tabular version still works for verification at modest scale).

## Quiz

{{#quiz 04-designing-the-ssa-game.toml}}
