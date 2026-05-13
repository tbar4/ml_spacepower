# Lesson 5: Information Set Monte Carlo Tree Search

**Module:** Search and Planning — M04: Tree Search and Neural Guidance
**Source:** [cite: Cowling, Powley & Whitehouse "Information Set Monte Carlo Tree Search" IEEE Transactions on Computational Intelligence and AI in Games 2012; Silver et al. "A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play" (AlphaZero); Furtak & Buro "Recursive Monte Carlo Search for Imperfect Information Games"]

---


<!-- toc -->

## Where this fits

Lessons 2 and 3 built MCTS for perfect-information games: every node in the tree is a fully specified game state, and selection, expansion, and backpropagation all operate on that concrete state. This works because there is no ambiguity about "what state we are in" — both players see the full board.

In fog-of-war games — the defining feature of the SSA orbital dominance wargame in Module 8 — you know your own satellite positions but not the adversary's. Standard MCTS breaks down immediately. This lesson introduces **Information Set MCTS (IS-MCTS)**, the algorithm that extends neural-guided MCTS to imperfect information games by sampling concrete hypotheses about the hidden state. IS-MCTS is the recommended inference-time planner in the production architecture: after training the neural network via AlphaZero-style self-play (Lesson 4), IS-MCTS uses that network to guide search at decision time.

Forward links: Module 7 develops the partial observability framework (belief states, particle filters, POMDPs) that provides the probabilistic foundation for IS-MCTS determinization sampling. Module 8 assembles the complete SSA wargame and uses IS-MCTS as the online planner.

---

## Why standard MCTS breaks for imperfect information

Standard MCTS builds a search tree rooted at the current, fully-known game state. Every node stores a concrete state from which legal actions, transitions, and value estimates are computed. This requires a single definite answer to "what state are we in?"

In a fog-of-war game, there is no such answer. You observe your own assets — satellite positions, onboard sensor readings, fuel levels — but the adversary's orbital slots, sensor configurations, and operational intent are hidden. What you have instead is an **information set**: the collection of all game states consistent with your observations so far.

Formally, your information set at time t is:

\[ \mathcal{I}_t = \{ s \in \mathcal{S} \mid s \text{ is consistent with all observations } o_0, o_1, \ldots, o_t \} \]

**Decoding:** S is the full state space. A state s is in your information set if every observation you have received so far would have been possible if the world had been in state s.

The naive fix — root your MCTS tree at the information set and treat it like a single node — runs into a fundamental problem called **strategy fusion**.

### Strategy fusion

Strategy fusion occurs when an algorithm combines plans that are only individually optimal conditional on knowing which true state obtains, producing a plan that is optimal in neither case.

SSA example: your space fence has detected an adversary satellite somewhere in a band between 400 km and 600 km altitude, but your last precise track was 48 hours ago. You have two candidate tasking actions:

- **Task sensor X** (narrow-beam): optimal if the satellite is in orbital slot A (low-altitude band)
- **Task sensor Y** (wide-beam): optimal if the satellite is in orbital slot B (high-altitude band)

**Perfect Information Monte Carlo (PIMC)** — the simplest imperfect-information MCTS variant — picks one hypothetical world (e.g., "the satellite is in slot A"), runs MCTS to determine the best action (task sensor X), then picks another hypothetical world (slot B), runs MCTS (task sensor Y), then averages across hypotheticals. The result might be to task sensor X with probability 0.5 and sensor Y with probability 0.5.

The problem: this mixed strategy is exploitable. An adversary who knows you are using PIMC can sit exactly at the boundary between the two slots and guarantee that half your sensor taskings are wasted. A pure strategy that commits to, say, tasking the wide-beam sensor first to narrow down the region is unexploitable in a way that the fused strategy is not.

Strategy fusion arises because PIMC allows each determinization to recommend a different action, and the averaging step loses the correlation between "which state is true" and "which action is appropriate." IS-MCTS avoids strategy fusion by building a single consistent plan across determinizations — actions are selected not per-determinization but by aggregating the expected value of each action across all determinizations.

---

## Determinization

A **determinization** is one concrete game state drawn from the information set: a specific hypothesis about all hidden information, consistent with everything you have observed.

For an SSA scenario with 5 adversary satellites whose positions are unknown:
- **Information set**: all orbital configurations where the 5 satellites are at positions consistent with the last RA/Dec observations plus physically plausible maneuvers since then
- **One determinization**: a specific assignment of all 5 satellites to particular orbital slots — one complete, concrete, fully-specified game state

Sampling a determinization means drawing from the probability distribution over possible hidden states given your observations:

\[ d \sim P(s \mid \mathcal{I}_t) \]

**Decoding:** This is a sample from the posterior over game states given your information set. In Module 7, this posterior is maintained as a particle filter; each particle is effectively a determinization.

IS-MCTS samples many determinizations and runs MCTS on each, then aggregates the results. By committing to one concrete hidden state per simulation, IS-MCTS avoids strategy fusion: within each simulation, the plan is internally consistent with a single world hypothesis.

---

## The IS-MCTS algorithm

The outer loop of IS-MCTS is a simple iteration over sampled determinizations. Within each determinization, standard neural-guided MCTS runs one simulation on the concrete state. The key insight is that aggregating value estimates across many determinizations computes an approximation to the expected value of each action under uncertainty:

\[ \hat{V}(a) \approx \mathbb{E}_{s \sim P(s \mid \mathcal{I}_t)}[V(s, a)] \]

**Decoding:** For each action a, the average value across determinizations estimates what outcome you can expect from taking action a, averaged over all consistent hypotheses about the hidden state. The action with the highest such expected value is selected.

```python
from collections import defaultdict

def ismcts(root_information_set, n_simulations, neural_network):
    """
    IS-MCTS outer loop.

    root_information_set: object with .sample_determinization() method
    n_simulations: total number of MCTS simulations (one per determinization)
    neural_network: callable returning (policy_logits, value) for a concrete state

    Returns: best action at the root, averaged across all determinizations
    """
    action_visit_counts = defaultdict(int)
    action_total_values = defaultdict(float)

    for _ in range(n_simulations):
        # Step 1: Sample one concrete hypothesis from the information set
        det_state = root_information_set.sample_determinization()

        # Step 2: Run one MCTS simulation on this concrete state
        # neural_network guides selection (PUCT) and replaces rollouts (value head)
        root_node = ISMCTSNode(det_state, prior=1.0)
        mcts_simulation(root_node, neural_network)

        # Step 3: Record which action was selected at the root and its value
        for action, child in root_node.children.items():
            if child.N > 0:
                action_visit_counts[action] += child.N
                action_total_values[action] += child.W

    # Step 4: Select action with highest average value across all determinizations
    best_action = max(
        action_visit_counts.keys(),
        key=lambda a: action_total_values[a] / action_visit_counts[a]
    )
    return best_action
```

The `action_visit_counts` and `action_total_values` dictionaries aggregate statistics across all determinizations. An action that was consistently good across many different hypotheses about the hidden state accumulates high average value and is selected.

---

## UCB in IS-MCTS

Within each determinization's MCTS simulation, the standard PUCT formula from Lesson 3 applies. For a node representing state s, the score for child action a is:

\[ \text{PUCT}(s, a) = \frac{W(s, a)}{N(s, a)} + c \cdot p(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \]

**Decoding:**
- \(W(s, a) / N(s, a)\): empirical average value from simulations that took action a from state s (exploitation)
- \(p(a \mid s)\): prior probability from the neural network's policy head
- \(c \cdot p(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))\): exploration bonus, weighted by the prior and shrinking as action a accumulates visits

One subtlety in IS-MCTS: the visit count \(N(s, a)\) accumulated at an inner node spans **all determinizations that passed through a state equivalent to s and considered action a**. Since each determinization may produce a different concrete state at interior nodes (the hidden information resolves differently in each), the IS-MCTS implementation must identify "equivalent" states carefully — typically by the information available to the acting player, not the full state.

For the SSA wargame, this means: two determinizations with different adversary satellite positions but identical own-satellite positions and identical sensor readings so far are mapped to the same information-set node for the purposes of sharing visit counts.

```rust
// No external crates — pure f64 math demonstrating the PUCT formula.

fn puct_score(w: f64, n: f64, parent_n: f64, prior: f64, c: f64) -> f64 {
    if n == 0.0 { return f64::INFINITY; }
    w / n + c * prior * parent_n.sqrt() / (1.0 + n)
}

fn main() {
    let parent_n = 40.0_f64;
    let c = 1.5_f64;

    // (name, W, N, prior probability from policy network)
    let children = [("A", 14.0_f64, 20.0_f64, 0.50_f64),
                    ("B",  6.0,     15.0,      0.20),
                    ("C",  4.0,      5.0,      0.30)];

    println!("{:<6} {:>6} {:>5} {:>6} {:>11}", "Child", "W/N", "N", "Prior", "PUCT");
    let scores: Vec<f64> = children.iter()
        .map(|&(_, w, n, p)| puct_score(w, n, parent_n, p, c))
        .collect();
    let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for (&(name, w, n, prior), &score) in children.iter().zip(scores.iter()) {
        println!(
            "{:<6} {:>6.3} {:>5.0} {:>6.2} {:>11.3}{}",
            name, w / n, n, prior, score,
            if score == best { "  <-- select" } else { "" }
        );
    }
}
```

The PUCT exploration term \(c \cdot p(a) \cdot \sqrt{N} / (1 + N(a))\) differs from UCT's \(\sqrt{\ln N / N(a)}\): it is weighted by the policy prior, so a high-probability action retains a larger exploration bonus even after many visits. This allows the neural network's prior to guide early search without completely overriding the accumulated statistics.

```python
import math

class ISMCTSNode:
    """Node in an IS-MCTS tree. State is a concrete determinization at this point."""

    def __init__(self, state, prior=0.0, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = {}       # action -> ISMCTSNode
        self.N        = 0        # visit count across determinizations
        self.W        = 0.0      # total accumulated value
        self.P        = prior    # policy network prior
        self.expanded = False

    def puct_score(self, c=1.5):
        if self.N == 0:
            return float('inf')
        parent_n = self.parent.N if self.parent else 1
        exploit  = self.W / self.N
        explore  = c * self.P * math.sqrt(parent_n) / (1 + self.N)
        return exploit + explore

    def best_child(self, c=1.5):
        return max(self.children.values(), key=lambda ch: ch.puct_score(c))


def mcts_simulation(root_node, neural_network, c=1.5):
    """
    One MCTS simulation from root_node on a concrete determinization.
    Modifies root_node in place via backpropagation.
    """
    import torch
    import torch.nn.functional as F

    node = root_node

    # Phase 1: Selection — descend until leaf or terminal
    while node.expanded and not node.state.is_terminal():
        node = node.best_child(c)

    # Phase 2 & 3: Expansion and evaluation (neural network replaces rollout)
    if node.state.is_terminal():
        value = node.state.terminal_value()
    else:
        state_tensor = node.state.to_tensor()
        with torch.no_grad():
            policy_logits, value_tensor = neural_network(state_tensor)
        value = value_tensor.item()

        legal = node.state.legal_actions()
        priors = F.softmax(policy_logits[legal], dim=0).tolist()
        for action, prior in zip(legal, priors):
            next_state = node.state.apply(action)
            node.children[action] = ISMCTSNode(
                next_state, prior=prior, parent=node, action=action
            )
        node.expanded = True

    # Phase 4: Backpropagation — walk up, flipping sign at each level
    while node is not None:
        node.N += 1
        node.W += value
        value   = -value
        node    = node.parent
```

---

## Sampling determinizations

How to sample from the information set depends on what observations have been made. For the SSA wargame, the information set at turn t is characterized by:

- Your own satellites: known positions, velocities, and fuel levels (fully observed)
- Adversary satellites: last confirmed RA/Dec observation plus uncertainty from unobserved maneuvers since then
- Constraints from the rules of the game: maximum delta-v budgets, orbital mechanics, no-maneuver windows

A determinization is sampled by drawing adversary satellite states from a belief distribution — specifically, the particle filter maintained in Module 7:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class SSAInformationSet:
    """Information set for the SSA wargame. own_satellites are fully observed;
    adversary_particles is a particle filter over adversary configurations."""
    own_satellites:       list
    adversary_particles:  list   # one particle = one complete adversary hypothesis

    def sample_determinization(self):
        """Draw one concrete game state by sampling one particle uniformly."""
        particle = self.adversary_particles[
            np.random.randint(len(self.adversary_particles))
        ]
        return SSAConcreteState(
            own_satellites=self.own_satellites,
            adversary_satellites=particle.adversary_positions,
        )

    def update_after_observation(self, new_observation):
        """Standard SIR particle filter update — see Module 7."""
        weights = np.array([
            p.observation_likelihood(new_observation)
            for p in self.adversary_particles
        ])
        weights /= weights.sum()
        # Resample
        indices = np.random.choice(
            len(self.adversary_particles), size=len(self.adversary_particles),
            p=weights, replace=True
        )
        self.adversary_particles = [self.adversary_particles[i] for i in indices]
```

**SSA example:** Suppose the adversary has 3 satellites. After observing two optical passes at t=0 and t=6 hours, your particle filter contains 500 particles, each specifying a full 3-satellite orbital configuration consistent with both observations. Each call to `sample_determinization` returns one of those 500 particles as the adversary configuration in a concrete game state. IS-MCTS runs 200 simulations, each on a different concrete state, then aggregates.

The quality of IS-MCTS decisions depends directly on the quality of the particle filter. A well-calibrated belief distribution (from accurate sensor models) produces determinizations that cluster around the truth; a poorly calibrated one produces determinizations spread across improbable states, wasting simulation budget.

---

## IS-MCTS with the neural network prior

After AlphaZero-style training (Lesson 4), the neural network provides two things for each determinization:

- **Policy head** \(p(a \mid s)\): a prior over actions that IS-MCTS uses in PUCT to focus search on promising actions first
- **Value head** \(V(s)\): a direct value estimate that replaces random rollouts

This dramatically reduces the number of simulations needed. Without a neural network, IS-MCTS requires enough simulations for random rollouts to average out their noise. With the value head replacing rollouts, each simulation returns a low-variance estimate, and 50-200 simulations often suffice where 2,000 or more would be needed for random-rollout IS-MCTS.

The PUCT formula in the context of IS-MCTS:

\[ \text{PUCT}(s, a) = \underbrace{\frac{W(s,a)}{N(s,a)}}_{\text{exploitation: empirical value}} + c \cdot \underbrace{p(a \mid s)}_{\text{policy prior}} \cdot \underbrace{\frac{\sqrt{N(s)}}{1 + N(s,a)}}_{\text{exploration bonus}} \]

**Decoding:** When \(N(s, a) = 0\) (action never tried), the exploration term equals \(c \cdot p(a \mid s) \cdot \sqrt{N(s)}\) — pure prior. Actions the network considers likely are tried first. As \(N(s, a)\) grows, the empirical term dominates and the prior's influence fades. The network provides a smart starting point; the search overrides it when evidence accumulates.

**SSA example:** The policy network has learned from self-play that tasking the wide-beam sensor is nearly always the right first action when adversary satellite position uncertainty is high (particle spread > 50 km). It assigns \(p(\text{wide-beam}) \approx 0.72\) and \(p(\text{narrow-beam}) \approx 0.18\). IS-MCTS therefore spends roughly four times more simulations exploring wide-beam follow-on sequences than narrow-beam ones, even with only 50 total simulations. Without the prior, all actions would receive roughly equal initial exploration, spreading the budget too thin to produce reliable estimates.

---

## Implementation: IS-MCTS for a 2-player SSA reconnaissance game

A complete self-contained implementation for a simplified hidden-information game: one player controls a reconnaissance satellite (known position), the other controls an adversary satellite (position hidden from the first player).

```python
import math
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# ── Game state ──────────────────────────────────────────────────────────────

@dataclass
class ReconGameState:
    """Simplified SSA reconnaissance game. recon_pos is known to both players;
    adversary_pos is hidden from the recon player."""
    recon_pos:     int
    adversary_pos: int
    recon_fuel:    int
    turn:          int
    acting_player: int
    N_SLOTS:       int = 8
    MAX_TURNS:     int = 10

    def legal_actions(self):
        """Recon: move left/right/hold + observe. Adversary: move left/right/hold."""
        if self.acting_player == 0:
            actions = ['hold', 'observe']
            if self.recon_fuel > 0:
                actions += ['left', 'right']
            return actions
        else:
            return ['hold', 'left', 'right']

    def apply(self, action):
        rp, ap = self.recon_pos, self.adversary_pos
        fuel = self.recon_fuel
        if self.acting_player == 0:
            if action == 'left':
                rp   = (rp - 1) % self.N_SLOTS
                fuel -= 1
            elif action == 'right':
                rp   = (rp + 1) % self.N_SLOTS
                fuel -= 1
            next_player = 1
        else:
            if action == 'left':
                ap = (ap - 1) % self.N_SLOTS
            elif action == 'right':
                ap = (ap + 1) % self.N_SLOTS
            next_player = 0
        return ReconGameState(rp, ap, fuel, self.turn + 1, next_player,
                              self.N_SLOTS, self.MAX_TURNS)

    def is_terminal(self):
        return self.turn >= self.MAX_TURNS

    def terminal_value(self):
        """Recon wins (+1) if within 1 slot of adversary at game end."""
        dist = min(
            abs(self.recon_pos - self.adversary_pos),
            self.N_SLOTS - abs(self.recon_pos - self.adversary_pos)
        )
        return 1.0 if dist <= 1 else -1.0

    def to_tensor(self):
        import torch
        return torch.tensor([
            self.recon_pos / self.N_SLOTS,
            self.adversary_pos / self.N_SLOTS,
            self.recon_fuel / 5.0,
            self.turn / self.MAX_TURNS,
            float(self.acting_player),
        ], dtype=torch.float32)


# ── Information set ──────────────────────────────────────────────────────────

class ReconInformationSet:
    """Recon player's information set. adversary_belief is a distribution over slots."""
    def __init__(self, recon_pos, recon_fuel, turn, n_slots=8):
        self.recon_pos       = recon_pos
        self.recon_fuel      = recon_fuel
        self.turn            = turn
        self.n_slots         = n_slots
        # Uniform prior over adversary positions
        self.adversary_belief = np.ones(n_slots) / n_slots

    def observe(self, sensor_reading: Optional[int]):
        """
        Update belief after an 'observe' action.
        sensor_reading: the adversary slot if detected (adjacent slot),
                        or None if not detected.
        """
        likelihood = np.ones(self.n_slots)
        if sensor_reading is not None:
            likelihood[:] = 0.05
            likelihood[sensor_reading] = 0.95
        else:
            # Not detected: adversary unlikely in adjacent slots
            likelihood[self.recon_pos] = 0.1
            adjacent = [(self.recon_pos - 1) % self.n_slots,
                        (self.recon_pos + 1) % self.n_slots]
            for a in adjacent:
                likelihood[a] = 0.2
        self.adversary_belief *= likelihood
        self.adversary_belief /= self.adversary_belief.sum()

    def sample_determinization(self) -> ReconGameState:
        """Draw one concrete game state from the information set."""
        adversary_pos = int(np.random.choice(self.n_slots, p=self.adversary_belief))
        return ReconGameState(
            recon_pos     = self.recon_pos,
            adversary_pos = adversary_pos,
            recon_fuel    = self.recon_fuel,
            turn          = self.turn,
            acting_player = 0,
            N_SLOTS       = self.n_slots,
        )


# ── IS-MCTS ──────────────────────────────────────────────────────────────────

class ISMCTSNode:
    def __init__(self, state, prior=1.0, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = {}
        self.N        = 0
        self.W        = 0.0
        self.P        = prior
        self.expanded = False

    def puct_score(self, c=1.5):
        if self.N == 0:
            return float('inf')
        parent_n = self.parent.N if self.parent else 1
        return (self.W / self.N) + c * self.P * math.sqrt(parent_n) / (1 + self.N)

    def best_child(self, c=1.5):
        return max(self.children.values(), key=lambda ch: ch.puct_score(c))


def run_ismcts(info_set, n_simulations, neural_network=None, c=1.5):
    """
    IS-MCTS for the ReconGame.
    Returns the best action and per-action statistics.
    """
    action_visits = defaultdict(int)
    action_values = defaultdict(float)

    for _ in range(n_simulations):
        det_state = info_set.sample_determinization()
        root      = ISMCTSNode(det_state, prior=1.0)
        _simulate(root, neural_network, c)

        for action, child in root.children.items():
            action_visits[action] += child.N
            action_values[action] += child.W

    best = max(action_visits, key=lambda a: action_values[a] / action_visits[a])
    return best, dict(action_visits), dict(action_values)


def _simulate(node, neural_network, c=1.5):
    """One MCTS simulation from node on a concrete determinization."""
    # Selection
    while node.expanded and not node.state.is_terminal():
        node = node.best_child(c)

    # Expansion and evaluation
    if node.state.is_terminal():
        value = node.state.terminal_value()
    elif neural_network is not None:
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            policy_logits, val = neural_network(node.state.to_tensor().unsqueeze(0))
        value = val.item()
        legal = node.state.legal_actions()
        priors = F.softmax(policy_logits.squeeze(0)[:len(legal)], dim=0).tolist()
        for action, prior in zip(legal, priors):
            node.children[action] = ISMCTSNode(
                node.state.apply(action), prior=prior, parent=node, action=action
            )
        node.expanded = True
    else:
        # Fallback: uniform prior + random rollout (no network)
        legal = node.state.legal_actions()
        prior = 1.0 / len(legal)
        for action in legal:
            node.children[action] = ISMCTSNode(
                node.state.apply(action), prior=prior, parent=node, action=action
            )
        node.expanded = True
        value = _random_rollout(node.state)

    # Backpropagation
    while node is not None:
        node.N += 1
        node.W += value
        value   = -value
        node    = node.parent


def _random_rollout(state):
    while not state.is_terminal():
        action = random.choice(state.legal_actions())
        state  = state.apply(action)
    return state.terminal_value()
```

**Usage example:**

```python
# Create an information set: recon at slot 3, adversary position unknown
info_set = ReconInformationSet(recon_pos=3, recon_fuel=4, turn=0)

# Update belief after an observation (adversary not detected near slot 3)
info_set.observe(sensor_reading=None)

# Run IS-MCTS with 200 simulations (no neural network: random rollouts)
best_action, visits, values = run_ismcts(info_set, n_simulations=200)
print(f"Recommended action: {best_action}")
for action in sorted(visits, key=lambda a: -visits[a]):
    avg_val = values[action] / visits[action]
    print(f"  {action}: visits={visits[action]}, avg_value={avg_val:.3f}")
```

---

## Known weaknesses

IS-MCTS is a major improvement over PIMC but retains several limitations.

**Residual strategy fusion.** IS-MCTS reduces strategy fusion by averaging action values across determinizations rather than averaging action recommendations. But within a single simulation, the MCTS tree may still make decisions at interior nodes as if it had full knowledge of which determinization is true. For example, after branching left at the root (in determinization d_1), the tree may at depth 3 choose an action that is only optimal if the adversary's satellite is at the specific position encoded in d_1. An adversary who observes this depth-3 action can infer which determinization you were implicitly committed to — a subtle information leak.

**The cheating problem.** A MCTS simulation on a determinization can explore branches that reveal information the agent should not have. Consider: at depth 2, the simulation checks whether the adversary's satellite is in slot A and receives a definitive "yes" (because the determinization was constructed that way). The simulation then exploits this information by planning around slot A — even though the real agent cannot know this. The fix: inner nodes should be evaluated only from the perspective of what the acting player can observe, not from the full determinization. In practice, this means only the root determinization should be treated as observable; interior nodes must use the acting player's information-set projection.

**Scalability with belief complexity.** If the information set is very large (e.g., 10 adversary satellites with no recent tracks, each with 50 plausible orbital slots), the number of determinizations needed to adequately cover the space grows rapidly. With a well-trained neural network, 200-500 simulations often suffice because the value head produces accurate estimates without rollout variance. Without a network, thousands of simulations may be required.

**Action space explosion.** In the SSA wargame, the joint action space (sensor taskings, maneuvers, communication routing) can be large. IS-MCTS with PUCT manages this through the policy prior, but the branching factor still limits effective search depth in a fixed simulation budget.

---

## When to use IS-MCTS vs. CFR

Counterfactual Regret Minimization (CFR) and IS-MCTS are the two main algorithms for imperfect-information games. They have complementary strengths.

| Criterion | IS-MCTS | CFR |
|---|---|---|
| **Game size** | Scales to very large games; depth-first search | Requires full game tree traversal; impractical for large games |
| **Solution quality** | Approximate; no theoretical Nash guarantee | Converges to Nash equilibrium with sufficient iterations |
| **Neural network integration** | Natural; policy + value head directly guide search | Requires separate value function approximation (Deep CFR) |
| **Inference-time latency** | Fast with a trained network (50-200 sims) | CFR policy lookup is fast but training is offline |
| **Imperfect-recall handling** | Works naturally; no memory constraints | Standard CFR requires perfect recall; extensions exist |
| **Exploitability** | Residual strategy fusion; can be exploited | Nash convergence guarantees non-exploitability in 2-player zero-sum |
| **SSA wargame fit** | Strong: game is large, network is available, real-time required | Weak: game tree too large for full CFR traversal |

**Guidance:** Use IS-MCTS when the game is too large for full-tree traversal, a neural network is available from training, and decisions must be made in real time. Use CFR when the game tree is manageable, you need guaranteed Nash convergence, and you can afford offline computation. For the SSA orbital dominance wargame — large state space, trained AlphaZero network, real-time operational constraints — IS-MCTS is the recommended inference-time planner. Module 5 covers CFR in depth for games where its guarantees are practical.

---

## Key Takeaways

- **Standard MCTS requires a concrete state at every node**, but in fog-of-war games the current state is unknown; attempting to run MCTS directly on the information set produces strategy fusion — exploitable plans that merge optimal responses to mutually exclusive hypotheses about the hidden state.
- **A determinization is one concrete hypothesis about all hidden information**, sampled from the belief distribution over the information set; IS-MCTS runs MCTS independently on each determinization, then aggregates action values across all of them to compute the expected value of each action under uncertainty.
- **Strategy fusion is reduced but not eliminated**: IS-MCTS can still leak information at interior nodes where a simulation exploits the determinization's hidden state; the standard mitigation is to evaluate interior nodes only from the acting player's observable information.
- **The neural network prior dramatically reduces the simulation budget**: the policy head focuses IS-MCTS on plausible actions via PUCT, and the value head replaces high-variance random rollouts with direct value estimates, cutting required simulations from thousands to tens or hundreds.
- **IS-MCTS scales where CFR cannot**: for games too large for full-tree traversal, IS-MCTS combined with a trained neural network provides high-quality approximate play in real time with no offline game-tree enumeration required.
- **IS-MCTS is the recommended inference-time planner for the SSA wargame**: it bridges the AlphaZero-style training in Lesson 4 (which assumes perfect information during self-play) with the partial observability framework in Module 7 (which maintains the particle filter supplying determinizations) to produce a complete, deployable decision engine.

{{#quiz 05-information-set-mcts.toml}}
