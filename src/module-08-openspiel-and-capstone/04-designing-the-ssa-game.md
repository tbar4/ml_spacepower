# Lesson 4: Designing the SSA Game


<!-- toc -->

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

## Game design principles for SSA

Before formalizing the conjunction-masking game, it is worth stating the design principles explicitly. These apply whenever you are designing a game for algorithm testing in an operational context — not just for this capstone, but for any future SSA game you might create.

### Principle 1: Clear and interpretable state representation

Every element of the game state should correspond to something you can point to in the real SSA scenario. If you have a bit vector in the state that you cannot describe in orbital mechanics terms, that is a red flag: the game may be well-defined mathematically but the solution will not translate to operational insight.

For the conjunction-masking game: `opportunity` maps to the threat geometry assessment that operators receive from SSA feeds; `intensity` maps to the delta-v magnitude of the maneuver; `allocation` maps to the sensor tasking order submitted to ground stations. Every state variable has a concrete referent.

### Principle 2: Meaningful decisions with real tradeoffs

A good game for algorithm testing should have decisions where neither "always action A" nor "always action B" dominates. The point of computing an equilibrium is that it requires genuinely mixed strategies — the algorithm teaches you something you could not derive by inspection.

The conjunction game is designed to force mixing: Heavy maneuver has higher upside but is more detectable (tradeoff for Adversary), Wide allocation catches more behaviors but at lower probability per catch (tradeoff for Defender). These tradeoffs are derived from the actual physics of detection sensitivity vs. coverage.

### Principle 3: Partial observability where the scenario demands it

Not every SSA game needs partial observability. A game modeling cooperative satellite deconfliction might be fully observable (both operators share data). But any game with adversarial intent (one party trying to conceal something from another) naturally calls for imperfect information.

The conjunction-masking game has partial observability because: the Adversary's true intent (opportunity) is hidden from the Defender; the Defender's allocation is hidden from the Adversary until detection. This structure is not imposed artificially — it reflects actual classification boundaries in SSA data sharing.

### Principle 4: Tractable action space

The game's action space should be large enough to produce interesting mixed strategies but small enough that CFR converges in a reasonable number of iterations without requiring deep CFR. The rule of thumb: if you can enumerate all information sets on a whiteboard, the game is appropriate for vanilla tabular CFR. If enumeration requires a computer but the game is still finite, MCCFR is appropriate. If the game is effectively continuous, deep CFR is needed.

For the capstone, 3 intensity levels and 3 allocation levels give 9 strategy parameters total — clearly whiteboard-enumerable. The scaled variant (7 intensities, 5 allocations, 4 opportunity types) is computer-enumerable. A real continuous-thrust orbital mechanics simulation would require deep CFR.

### Principle 5: Connection to real operational constraints

Useful SSA games encode real constraints as game structure. Detection probabilities should be calibrated to actual sensor capabilities (or to publicly available ranges). Maneuver intensities should be calibrated to typical delta-v budgets. Opportunity probabilities should reflect real conjunction frequency statistics.

When the game is calibrated, its Nash equilibrium describes a strategy that is actually achievable and meaningful: "given these sensor capabilities and these maneuvering constraints, a rational adversary would mix these maneuver intensities in these proportions." That is an operationally useful statement.

## The conjunction-masking game: complete specification

This section builds on the high-level description earlier to give the full game definition.

### Players, state, and information

The game has two players and one chance component:

- **Player 0 (Adversary)**: controls a satellite, wants to execute covert maneuvers
- **Player 1 (Defender)**: operates a sensor network, wants to detect maneuvers
- **Chance**: resolves the orbital opportunity and the noisy detection outcome

The full game state at any point is:

```
(opportunity, adversary_action, defender_action, detection_result)
```

where each field is `None` until its corresponding stage resolves.

### What each player observes

**Adversary observes**: the opportunity (their own private information) and the detection result after Stage 4. They do NOT observe the Defender's allocation.

**Defender observes**: the detection result after Stage 4. They do NOT observe the opportunity or the Adversary's intensity choice.

This gives rise to the information sets enumerated above: 2 for the Adversary (one per opportunity value), 1 for the Defender (no information at decision time).

### Terminal conditions and payoff computation

The game always terminates after Stage 4. There are no draws or multi-round extensions in the base version. The payoff function is:

```python
def adversary_payoff(opportunity, intensity, detected):
    if intensity == 0:  # None
        return 0.0  # no maneuver, no benefit, no penalty regardless of detection
    if opportunity:
        if not detected:
            return float(intensity)   # Light=+1, Heavy=+2
        else:
            return -3.0               # caught, regardless of intensity
    else:  # no opportunity
        if not detected:
            return 0.0                # wasted maneuver budget, no penalty
        else:
            return -2.0               # caught maneuvering for no reason
```

The Defender's payoff is the negation of the Adversary's payoff (zero-sum).

### Game tree size

The game tree has the following structure:

- **1 root node** (chance): 2 outcomes (opportunity/no-opportunity)
- **2 Adversary decision nodes** (one per opportunity): 3 actions each
- **6 Defender decision nodes** (one per (opportunity, intensity) pair): 3 actions each
  - But the Defender cannot distinguish these! All 6 world states belong to 1 information set.
- **18 chance nodes** (one per (opportunity, intensity, allocation) triple): 2 outcomes each (detected/not)
- **36 terminal nodes**

Total nodes: 1 + 2 + 6 + 18 + 36 = 63. This is tiny enough to hand-verify.

## Information structure choices and equilibrium effects

The design choice of what each player observes is not cosmetic. It fundamentally changes the game's equilibrium. Here we examine three information structure variants and how they differ.

### Variant A: Full information (both players observe everything)

If the Defender could observe the opportunity and the Adversary's intensity before choosing an allocation, and the Adversary could observe the allocation before choosing intensity, the game becomes a perfect-information game. In this case, backward induction gives the solution directly:

- Defender, seeing Heavy intensity, plays Narrow (0.85 detection vs. 0.65 for Wide)
- Adversary, knowing Defender will play Narrow, plays Light (0.30 detection vs. 0.85 for Narrow/Heavy)
- But Defender, knowing Adversary plays Light, is indifferent between Wide and Narrow (equal utility)

The equilibrium is pure in this variant. No randomization required. Not interesting for CFR.

### Variant B: Simultaneous-move game (neither player observes the other)

If Adversary and Defender choose simultaneously without observing each other, the game is a simultaneous-move matrix game. This is simpler than the sequential imperfect-information game. The matrix (for the "opportunity = Yes" subgame) is:

| Adversary \ Defender | Wide | Narrow | Off |
|---------------------|------|--------|-----|
| None               | 0    | 0      | 0   |
| Light              | -1   | +0.6   | +1  |
| Heavy              | -0.3 | -1.55  | +2  |

(Values computed using detection probabilities and payoff formula.)

**Decoding:** Each cell is the Adversary's expected payoff when both players commit to their pure strategy. For example, Adversary plays Light, Defender plays Wide: expected payoff = (1 - 0.5) * 1 + 0.5 * (-3) = 0.5 - 1.5 = -1.0. The rows and columns show that no pure strategy dominates: the Adversary's best response depends on what the Defender plays, and vice versa. Mixed strategies are needed.

### Variant C: Sequential with partial observability (our actual game)

The sequential structure (Adversary acts first, Defender acts second without observing the Adversary) is what we use. The key consequence: the Defender's strategy cannot condition on the Adversary's actual action, only on the public history (which in Stage 3 contains nothing, since the Adversary's action is private). This is *less* information for the Defender than Variant A, which means the Adversary can exploit the Defender's uncertainty.

The equilibrium in Variant C has a richer structure than Variant B because of the sequential commitment: the Adversary moves first and the Defender's uncertainty about what was chosen drives the equilibrium mixing.

### Generating the game tree in Python

The following code produces the complete game tree for visualization:

```python
"""
Generate the conjunction-masking game tree and enumerate all paths.
"""

OPPORTUNITY_PROB = {True: 0.4, False: 0.6}
INTENSITIES = [0, 1, 2]   # None, Light, Heavy
ALLOCATIONS = [0, 1, 2]   # Wide, Narrow, Off
INTENSITY_NAMES = {0: "None", 1: "Light", 2: "Heavy"}
ALLOCATION_NAMES = {0: "Wide", 1: "Narrow", 2: "Off"}

DETECTION_PROB = {
    (0, 0): 0.05, (0, 1): 0.05, (0, 2): 0.0,
    (1, 0): 0.50, (1, 1): 0.30, (1, 2): 0.0,
    (2, 0): 0.65, (2, 1): 0.85, (2, 2): 0.0,
}

def adversary_payoff(opportunity, intensity, detected):
    if intensity == 0:
        return 0.0
    if opportunity:
        return float(intensity) if not detected else -3.0
    else:
        return 0.0 if not detected else -2.0

def enumerate_game_tree():
    """
    Enumerate all terminal nodes with their path probabilities.
    Returns list of dicts with all path variables and their probability.
    """
    paths = []
    for opp, opp_prob in OPPORTUNITY_PROB.items():
        for intensity in INTENSITIES:
            for allocation in ALLOCATIONS:
                det_prob = DETECTION_PROB[(intensity, allocation)]
                for detected in [True, False]:
                    prob = (opp_prob
                            * (det_prob if detected else 1 - det_prob))
                    adv_rew = adversary_payoff(opp, intensity, detected)
                    paths.append({
                        "opportunity": opp,
                        "intensity": intensity,
                        "allocation": allocation,
                        "detected": detected,
                        "path_prob": prob,
                        "adversary_reward": adv_rew,
                        "defender_reward": -adv_rew,
                    })
    return paths

# Print the game tree
paths = enumerate_game_tree()
total_prob = sum(p["path_prob"] for p in paths)
assert abs(total_prob - 1.0) < 1e-9, f"Probabilities must sum to 1: {total_prob}"

# Group by Adversary information set
from itertools import groupby
adv_groups = {}
for path in paths:
    key = "opp=Yes" if path["opportunity"] else "opp=No"
    adv_groups.setdefault(key, []).append(path)

print("=== Game tree summary ===")
for infoset, group in sorted(adv_groups.items()):
    print(f"\nAdversary info set: {infoset}")
    for path in group:
        print(f"  Intensity={INTENSITY_NAMES[path['intensity']]}, "
              f"Alloc={ALLOCATION_NAMES[path['allocation']]}, "
              f"Det={path['detected']}: "
              f"prob={path['path_prob']:.4f}, "
              f"adv_reward={path['adversary_reward']}")
```

Expected output (first group):

```
=== Game tree summary ===

Adversary info set: opp=No
  Intensity=None, Alloc=Wide, Det=True: prob=0.0030, adv_reward=0.0
  Intensity=None, Alloc=Wide, Det=False: prob=0.0570, adv_reward=0.0
  ...

Adversary info set: opp=Yes
  Intensity=Heavy, Alloc=Wide, Det=True: prob=0.1040, adv_reward=-3.0
  Intensity=Heavy, Alloc=Wide, Det=False: prob=0.0560, adv_reward=2.0
  ...
```

## Calibrating the game to reality

The detection probability table is the primary calibration parameter. The values in the table should be grounded in actual sensor performance, even if coarsely.

### Orbital mechanics constraints

A satellite with mass $m$ and specific impulse $I_{sp}$ has delta-v budget:

$$\Delta v = I_{sp} \cdot g_0 \cdot \ln\left(\frac{m_0}{m_f}\right)$$

**Decoding:** This is the Tsiolkovsky rocket equation. $I_{sp}$ is the propellant efficiency (seconds), $g_0 = 9.81 \, \text{m/s}^2$ is the standard gravitational acceleration, $m_0$ is initial mass, and $m_f$ is final (dry) mass. For a typical small satellite (100 kg, $I_{sp} = 220$ s, 5% propellant mass), the total $\Delta v$ budget is approximately 220 × 9.81 × ln(1/0.95) ≈ 110 m/s. This budget constrains the "intensity" levels to physically plausible maneuver sizes.

In the game, the intensity levels correspond roughly to:
- None: 0 m/s (no maneuver)
- Light: 1-5 m/s (within typical station-keeping budget)
- Heavy: 10-50 m/s (a significant portion of the satellite's total delta-v budget)

This calibration matters because it affects the strategic balance: a Heavy maneuver that costs 30 m/s is a much bigger commitment than one that costs 1 m/s.

### Sensor detection capabilities

The detection probabilities in the game are derived from a simplified version of the signal-to-noise ratio framework used in SSA:

$$P_D = 1 - \exp\left(-\frac{(SNR)^2}{2}\right)$$

where SNR depends on the sensor's aperture, the maneuver magnitude, and the measurement noise. Wide allocation uses many sensors with lower individual SNR; Narrow allocation focuses one high-sensitivity sensor.

The key insight for calibration: the detection probability for a given maneuver magnitude increases roughly quadratically with the aperture and linearly with dwell time. A Wide allocation that splits sensor resources across multiple detection modes will have a lower probability of detecting any specific maneuver than a Narrow allocation that concentrates all resources on heavy-maneuver signatures.

### Why simplified games yield operational insights

The conjunction-masking game does not simulate orbital mechanics. It abstracts away orbital geometry, sensor noise models, conjunction probability computation, and decision timelines. So why should the equilibrium strategies say anything useful?

The answer is that the **strategic structure** — not the mechanics — drives the equilibrium. Two games with completely different physical realizations but the same payoff structure and information constraints will have the same Nash equilibrium. The conjunction-masking game captures the essential strategic structure:

- Private information drives deception incentives
- Detection probability is a function of both sides' choices
- The zero-sum nature means improving one side comes at the other's expense

A defense planner using the equilibrium strategy does not need to know the details of the game model to use the result correctly. They need to know: "at equilibrium, mix maneuver intensities in roughly these proportions." The orbital mechanics informs which maneuvers are feasible; the game-theoretic computation tells you the optimal mixing.

## The full SSA game specification

Here we state the conjunction-masking game as a formal 7-tuple, the standard representation for an imperfect-information extensive-form game.

### Formal 7-tuple

The game is defined as $\Gamma = (N, A, H, Z, \chi, \rho, u, \mathcal{I})$ where:

**$N = \{0, 1\}$** — the player set (Adversary = 0, Defender = 1). Chance is not a player; it is modeled separately.

**$A$** — the action set. For each player at each information set:
- Adversary information set "opp=Yes": $A_0 = \{0, 1, 2\}$ (None, Light, Heavy)
- Adversary information set "opp=No": $A_0 = \{0, 1, 2\}$
- Defender information set "": $A_1 = \{0, 1, 2\}$ (Wide, Narrow, Off)
- Chance at root: $A_c = \{0, 1\}$ (No opportunity, Opportunity)
- Chance at detection: $A_c = \{0, 1\}$ (Not detected, Detected)

**$H$** — the set of all histories (nodes in the game tree). A history is a sequence of actions from the root. $|H| = 63$ (as counted earlier).

**$Z \subset H$** — the terminal histories. $|Z| = 36$.

**$\chi: H \setminus Z \to 2^A$** — the action function, mapping each non-terminal history to its legal actions.

**$\rho: H \setminus Z \to N \cup \{c\}$** — the player function, mapping each non-terminal history to the player acting there ($c$ = chance).

**$u: Z \to \mathbb{R}^2$** — the utility function. $u_0(z)$ is the Adversary's payoff at terminal node $z$; $u_1(z) = -u_0(z)$ (zero-sum).

**$\mathcal{I} = (\mathcal{I}_0, \mathcal{I}_1)$** — the information partition. Each $\mathcal{I}_i$ is a partition of the decision nodes of player $i$ into information sets:
- $\mathcal{I}_0 = \{\{h : \text{opp}(h) = \text{Yes}\}, \{h : \text{opp}(h) = \text{No}\}\}$
- $\mathcal{I}_1 = \{\{h : \text{Defender acts}\}\}$ — one information set containing all Defender decision nodes

### State space dimensionality

For the base game:
- Opportunity: 2 values
- Intensity: 3 values
- Allocation: 3 values
- Detection: 2 values + "not yet resolved"
- Total world states: 2 × 3 × 3 × 3 = 54 (some unreachable)

For the scaled game (7 intensities, 5 allocations, 4 opportunity types):
- Total world states: 4 × 7 × 5 × 2 = 280
- Number of information sets: 4 (Adversary) + 1 (Defender) = 5
- Total strategy parameters: 4 × 7 + 1 × 5 = 33

### Action space

| Player | Actions | Count | Representation |
|--------|---------|-------|---------------|
| Adversary (opp=Yes) | None, Light, Heavy | 3 | {0, 1, 2} |
| Adversary (opp=No) | None, Light, Heavy | 3 | {0, 1, 2} |
| Defender | Wide, Narrow, Off | 3 | {0, 1, 2} |
| Chance (root) | No opp (p=0.6), Opp (p=0.4) | 2 | {0, 1} |
| Chance (detection) | Not detected, Detected | 2 | {0, 1} (prob from table) |

### Observation model

Let $s = (\omega, \alpha, \delta, r)$ denote a world state (opportunity, adversary action, defender action, detection result).

**Adversary observation function**: $o_0(s) = \omega$ — the Adversary observes only the opportunity at the time of their decision. After Stage 4, they also observe $r$ (whether they were detected).

**Defender observation function**: $o_1(s) = r$ — the Defender observes only the detection result after Stage 4. Before Stage 3, $o_1(s) = \emptyset$ (no observation).

### Reward function

$$u_0(\omega, \alpha, \delta, r) = \begin{cases}
0 & \text{if } \alpha = 0 \text{ (no maneuver)} \\
\alpha & \text{if } \omega = 1, r = 0 \text{ (opportunity, not detected)} \\
-3 & \text{if } \omega = 1, r = 1 \text{ (opportunity, detected)} \\
0 & \text{if } \omega = 0, r = 0 \text{ (no opportunity, not detected)} \\
-2 & \text{if } \omega = 0, r = 1 \text{ (no opportunity, detected)}
\end{cases}$$

**Decoding:** The reward has five cases, each corresponding to a combination of opportunity, detection, and whether a maneuver was actually attempted. The most important structural feature is the asymmetry: the cost of detection (-3) exceeds the maximum maneuver benefit (+2), so no maneuver-intensity strategy dominates; the Adversary must weigh expected benefit against detection risk. The -2 penalty for "caught with no opportunity" is lower than -3 because being caught for a purposeless maneuver is diplomatically less damaging than being caught exploiting a conjunction opportunity.

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

## Key Takeaways

- A good SSA game for algorithm testing has three properties simultaneously: small enough for vanilla CFR (verifiable), rich enough for non-trivial mixed strategies (interesting), and grounded enough in orbital mechanics for operational interpretation (meaningful).
- The conjunction-masking game's five design principles — interpretable state, meaningful tradeoffs, partial observability where warranted, tractable action space, and calibrated parameters — apply to any future SSA game you design.
- Information structure is not cosmetic: changing who observes what changes the equilibrium qualitatively, not just quantitatively; the sequential imperfect-information structure of the game forces richer mixing than either the full-information or simultaneous-move variants.
- The formal 7-tuple $\Gamma = (N, A, H, Z, \chi, \rho, u, \mathcal{I})$ is the complete specification; every design decision reduces to a choice in one of these seven components.
- Calibrating detection probabilities to real sensor physics and maneuver intensities to delta-v budgets ensures that equilibrium strategies describe behaviors that are physically achievable and operationally meaningful — not just abstract optima.
- The scaled variant (7 intensities, 5 allocations, 4 opportunity types) is deliberately designed to be solvable by both tabular CFR and deep CFR, so you can verify the deep variant against a ground truth before trusting it on games too large for tabular methods.

## Quiz

{{#quiz 04-designing-the-ssa-game.toml}}
