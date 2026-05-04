# Lesson 1: The Multi-Agent Problem

## Where this fits

Module 3 trained a single RL agent to make decisions in an MDP. The world was simple: one agent, one environment, one reward signal. The optimal policy existed and gradient descent could find it. Module 5 introduced game theory — the language of strategic interaction among multiple rational agents. This lesson bridges those two modules: what happens when you try to run RL in a world that is itself a game?

The answer is that almost everything from single-agent RL breaks. The environment is no longer stationary because the other agents are learning too. Convergence proofs do not apply. The notion of "optimal" depends on what the other agents do, which depends on what you do, which is circular.

This lesson diagnoses the problem carefully. The next three lessons present the main solution families: fictitious play, PSRO, and Alpha-rank.

## Why satellite constellation management is inherently multi-agent

A single satellite controlled by a single operator is a single-agent RL problem. The state is the satellite's orbital parameters and sensor tasking queue; the actions are maneuver commands and observation assignments; the reward is coverage quality.

But real space operations involve multiple agents simultaneously:

- Multiple satellite operators share a finite orbital regime. A maneuver by one operator changes the conjunction geometry for everyone.
- Multiple ground stations compete for radar and optical telescope time to build a common space picture. Scheduling one ground station determines what others can and cannot observe.
- Adversarial actors may deliberately maneuver to deny coverage or degrade tracking quality for an opposing operator.
- Within a single constellation, individual satellites must coordinate — they share frequency bands, have overlapping fields of regard, and must deconflict sensor tasking to avoid redundant coverage and blind spots.

In each of these cases, what is optimal for one agent depends on what all other agents do. The environment is not fixed; it is itself a product of all agents' decisions. This is the defining feature of multi-agent problems.

## Types of multi-agent settings

Multi-agent problems divide into three broad categories based on how the agents' incentives relate to each other.

### Fully cooperative: joint reward

All agents share a single reward signal. Every agent's gain is every other agent's gain. The problem reduces to a distributed optimization: how can many agents, possibly without full communication, coordinate to maximize a shared objective?

**SSA example**: A network of ground stations tasked with maintaining a common operational picture of the GEO belt. Every time any station detects a maneuvering satellite, all stations benefit. Every time a gap in coverage allows an undetected maneuver, all stations lose. The shared objective is total surveillance coverage over a 24-hour window. Individual stations acting independently might point redundant sensors at the same easy targets; a cooperative policy would distribute them to maximize total coverage.

The challenge in cooperative settings is not incentive misalignment — everyone wants the same thing — it is coordination. How do agents share information efficiently? How do they avoid redundant actions? How do they divide up responsibilities when communication is limited or denied?

### Fully competitive: zero-sum

One agent's gain is exactly another agent's loss. The sum of all payoffs at every outcome is zero. No agreement can make both agents better off; any gain for one comes directly at the other's expense.

**SSA example**: An operator deploying an ISR satellite to observe a strategic facility, and an adversary operator attempting to block observation windows by maneuvering an interfering satellite into the ISR satellite's field of regard. The ISR operator wants maximum observation time; the adversary wants minimum observation time. Every minute of observation gained by the ISR operator is lost by the adversary. This is a pursuit-evasion game with a zero-sum payoff structure.

Zero-sum games have the strongest theoretical guarantees. A Nash equilibrium always exists (in mixed strategies), is unique in value (though not always in strategy), and is computable efficiently for two-player games. CFR (Module 5) is a zero-sum solver.

### Mixed cooperative-competitive: general-sum

Each agent has its own reward function. Incentives partially overlap (some outcomes are better for everyone) and partially conflict (agents disagree about which good outcomes to aim for). Most real-world multi-agent problems are general-sum.

**SSA example**: Three satellite operators sharing the Ka-band spectrum over a congested orbital arc. Each operator wants maximum downlink bandwidth for their own satellites. Congestion hurts everyone (interference degrades all operators' links), but how to share the spectrum is contested: each operator prefers a schedule that maximizes their own allocation even if it comes at the others' expense. The shared interest (avoid total congestion) creates partial cooperation; the competing priorities (each prefers more bandwidth for themselves) create partial competition.

General-sum games are the hardest. Nash equilibria may be multiple (which one gets played is indeterminate), Pareto-inefficient (there may be outcomes better for everyone that no Nash supports), and hard to compute (PPAD-complete in general). They are also the most realistic.

## Non-stationarity: the fundamental challenge

In single-agent RL, the environment is assumed stationary: the transition dynamics and reward function do not change. This is what makes convergence proofs work — the agent is learning a fixed target.

In multi-agent RL, the other agents are learning too. From any one agent's perspective, the environment is **non-stationary**: the effective transition function (which includes other agents' behaviors as part of the dynamics) is changing at every step of training. What was a good response to the other agents yesterday may be a bad response today, because they have updated their policies.

### Concrete example: two ground stations competing for telescope time

Consider two ground station operators, Alice and Bob, each controlling one optical telescope. There is a congested object population at a particular right ascension that is only accessible for 4 hours per night from both sites. Both telescopes can observe the same arc, but simultaneous observations of the same object are wasted — each should be observing a different object. There are 20 objects to cover.

Both Alice and Bob run independent Q-learning on a simplified tasking problem. Alice's observation of Bob's strategy: if Bob is tasking the first 10 objects heavily, Alice should task the last 10. If Bob is randomly distributing, Alice should too.

But Bob is also observing Alice and updating. When Alice shifts to the last 10, Bob shifts to the first 10, which makes Alice shift back, which makes Bob shift back. The two agents are chasing each other's responses in a cycle.

This is the non-stationarity problem made concrete. The fundamental issue: the training signal for Alice's Q-function includes Bob's policy implicitly, but Bob's policy is changing. Alice's Q-values are computed against a moving target.

Formally, consider the Bellman update for Q-learning:

\\[ Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a') \\]

**Decoding:**
- \\(Q(s, a)\\): the estimated value of taking action \\(a\\) in state \\(s\\)
- \\(r\\): the immediate reward received
- \\(\gamma\\): the discount factor
- \\(\max_{a'} Q(s', a')\\): the best estimated future value from the next state

This update assumes that the best future value \\(\max_{a'} Q(s', a')\\) does not depend on time — that the value function is converging toward a stationary target. When another agent is simultaneously updating their policy, \\(s'\\) itself is a function of both agents' policies, and the target keeps moving. The standard convergence theorem for Q-learning requires a stationary MDP; multi-agent RL violates this requirement.

## The solution concepts landscape

Different situations call for different solution concepts. Here is how the main ones apply to SSA.

### Nash equilibrium (review from Module 5)

A **Nash equilibrium** is a strategy profile (one policy per agent) where no agent can improve its own expected payoff by unilaterally changing its policy. Every other agent's strategy is already a best response to it.

Nash equilibrium is the right concept when agents are fully rational, self-interested, and have no coordination mechanism — each agent can only reason about what is best given the others' strategies. In SSA, this describes uncoordinated commercial operators sharing an orbital regime: no single operator has incentive to deviate from a Nash strategy given that all others are playing Nash.

The limitation: Nash equilibria may be Pareto-inefficient. All agents playing Nash may do worse than some alternative agreement that is not a Nash equilibrium (because any individual agent would want to defect from that agreement).

### Correlated equilibrium

A **correlated equilibrium** is a generalization of Nash equilibrium where a trusted mediator (or a shared communication protocol) sends each agent a recommended action, drawn from a joint distribution. Each agent's recommended action is a best response, given the others' recommended actions and given that all agents follow the mediator's recommendations.

Crucially, correlated equilibria always include Nash equilibria as a special case (with an independent recommendation distribution), but can also achieve outcomes that no Nash equilibrium can. They are computationally easier to find (polynomial-time via linear programming) and can be more efficient.

**SSA example**: An international coordination body (like ITU spectrum management) acts as the mediator for Ka-band frequency assignments. The body assigns specific frequency slots and pointing windows to each operator. Every operator's assigned slot is a best response given that all others follow their assignments. Operators have no individual incentive to deviate (doing so would cause interference to themselves or invite retaliation). This is a correlated equilibrium, not a Nash equilibrium — the correlation comes from the shared coordination protocol.

### Cooperative optimality (social welfare maximization)

In fully cooperative settings, the right concept is not equilibrium at all — it is **joint optimality**: find the strategy profile that maximizes the sum of all agents' rewards (or some other social welfare function). There is no game-theoretic tension; the agents are effectively one distributed optimizer.

**SSA example**: a constellation operator running multiple satellites as a coordinated team. Each satellite has its own local Q-function, but all are optimizing the same global coverage metric. The solution is the joint policy that maximizes total coverage, subject to communication constraints.

## Coordination failure: a simulation

The most vivid illustration of why multi-agent problems are hard: two agents each doing what seems locally optimal, producing a catastrophically bad joint outcome.

Suppose two satellites from different operators are both approaching the same piece of debris from different directions. Each satellite's ground controller runs an independent collision avoidance algorithm. Each algorithm independently concludes: "maneuver left." Both maneuver left. They collide with each other.

```python
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────────
# Two satellites, each at a different position, both approaching the same debris.
# Each independently decides to maneuver left (positive y-direction).
# "Left" is relative to their direction of travel, which differs.

np.random.seed(42)

class Satellite:
    """Simplified satellite with position, velocity, and an independent avoidance rule."""
    def __init__(self, name, position, velocity):
        self.name = name
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)

    def avoidance_maneuver(self, debris_pos):
        """
        Purely local rule: maneuver perpendicular to current velocity,
        in the direction that creates the most separation from debris.
        Each satellite computes this independently with no knowledge of the other.
        """
        to_debris = debris_pos - self.pos
        distance = np.linalg.norm(to_debris)

        # Unit vector perpendicular to current velocity (in 2D)
        perp = np.array([-self.vel[1], self.vel[0]])
        perp = perp / np.linalg.norm(perp)

        # Choose the perpendicular direction that moves away from debris
        if np.dot(perp, to_debris) > 0:
            perp = -perp  # flip to move away

        delta_v = 0.05 * perp  # small maneuver
        return delta_v

    def step(self, delta_v=None):
        if delta_v is not None:
            self.vel += delta_v
        self.pos += self.vel

    def distance_to(self, other):
        return np.linalg.norm(self.pos - other.pos)


def simulate_uncoordinated(n_steps=20):
    """Two satellites independently avoid debris and collide with each other."""
    debris = np.array([0.0, 0.0])

    # Satellite A approaches from the left, Satellite B from the right
    sat_a = Satellite("Alpha", position=[-2.0, 0.5], velocity=[0.2, -0.02])
    sat_b = Satellite("Bravo", position=[2.0, 0.5], velocity=[-0.2, -0.02])

    print("--- Uncoordinated avoidance ---")
    for t in range(n_steps):
        dv_a = sat_a.avoidance_maneuver(debris)
        dv_b = sat_b.avoidance_maneuver(debris)

        sat_a.step(dv_a)
        sat_b.step(dv_b)

        sep = sat_a.distance_to(sat_b)
        debris_dist_a = np.linalg.norm(sat_a.pos - debris)
        debris_dist_b = np.linalg.norm(sat_b.pos - debris)

        if t % 5 == 0 or sep < 0.3:
            print(
                f"  t={t:2d}: Alpha pos={sat_a.pos.round(2)}, "
                f"Bravo pos={sat_b.pos.round(2)}, "
                f"separation={sep:.3f}, "
                f"debris dist A={debris_dist_a:.3f} B={debris_dist_b:.3f}"
            )
        if sep < 0.1:
            print(f"  *** COLLISION between Alpha and Bravo at t={t} ***")
            break

    return sat_a, sat_b


def simulate_coordinated(n_steps=20):
    """
    Coordinated avoidance: before maneuvering, satellites exchange intended
    delta-v and check for mutual conflict. If conflict detected, one yields.
    """
    debris = np.array([0.0, 0.0])

    sat_a = Satellite("Alpha", position=[-2.0, 0.5], velocity=[0.2, -0.02])
    sat_b = Satellite("Bravo", position=[2.0, 0.5], velocity=[-0.2, -0.02])

    print("\n--- Coordinated avoidance (deconflicted) ---")
    for t in range(n_steps):
        dv_a = sat_a.avoidance_maneuver(debris)
        dv_b = sat_b.avoidance_maneuver(debris)

        # Coordination check: simulate where each would end up after maneuvering
        future_a = sat_a.pos + sat_a.vel + dv_a
        future_b = sat_b.pos + sat_b.vel + dv_b
        future_sep = np.linalg.norm(future_a - future_b)

        # If maneuvers would bring satellites too close, Bravo yields (holds)
        if future_sep < 0.5:
            dv_b = np.zeros(2)  # Bravo holds; Alpha maneuvers

        sat_a.step(dv_a)
        sat_b.step(dv_b)

        sep = sat_a.distance_to(sat_b)
        if t % 5 == 0:
            print(
                f"  t={t:2d}: Alpha pos={sat_a.pos.round(2)}, "
                f"Bravo pos={sat_b.pos.round(2)}, "
                f"separation={sep:.3f}"
            )

    return sat_a, sat_b


uncoord_a, uncoord_b = simulate_uncoordinated()
coord_a, coord_b = simulate_coordinated()
```

The code illustrates the core issue: each satellite's individually rational action — move away from the debris — produces an irrational collective outcome. The coordination fix is simple here (one satellite yields), but in general, establishing who yields requires a coordination protocol that both agents commit to in advance. That protocol is, at its core, a correlated equilibrium.

## The CTDE paradigm: centralized training, decentralized execution

The coordination collision example raises a practical question: if agents need to coordinate, do they need to communicate at all times? In many operational settings, the answer is no. Communication may be unavailable, latency-constrained, or security-sensitive.

The **centralized training, decentralized execution (CTDE)** paradigm resolves this tension. During training (offline, in simulation), all agents have access to global information: each agent's state, each agent's policy, the joint reward. This allows a coordinator to train policies that account for interactions. During execution (online, in operations), each agent uses only its local information — no communication required.

The key insight: if the policies are trained together, they can implicitly coordinate without needing to communicate at runtime. The coordination knowledge is baked into the policy weights during training.

**SSA application**: During a simulation exercise, a network of ground stations trains a joint sensor-tasking policy using full knowledge of what every station is observing and what every other station intends to task. The training algorithm optimizes the joint coverage objective. After training, each station runs its own policy independently using only local sensor readings and its own tasking queue. The stations do not communicate during operations, but their policies have been jointly optimized so that they implicitly divide up coverage responsibility.

CTDE is the organizing principle behind multi-agent algorithms like QMIX, MADDPG, and COMA — all of which appear in modern multi-agent RL research. The lesson is: centralize what you can during training to overcome non-stationarity, then push execution out to the decentralized agents who face real-world communication and latency constraints.

### CTDE in practice: what to centralize and what to leave decentralized

A common implementation pattern for CTDE in cooperative satellite networks:

- **Centralized**: the training environment has a global state (all satellite positions, all sensor readings, all tasking queues). A centralized critic estimates the joint Q-value or advantage function using this global state. Gradient updates for all agents' policies use the centralized critic signal.

- **Decentralized**: each agent's policy network takes only local observations as input (what this particular satellite or ground station can currently see). At execution time, only this local-observation policy is used — no global state is accessed.

The centralized critic is a bridge: it allows training signals to be computed with full information, but the learned policy only requires local information to act. The separation means that during operations, a satellite in a communication blackout can still act reasonably, because its policy was trained to perform well using only local observations.

The quality of the decentralized policy depends on how much relevant information is actually available locally. If the local observation captures most of what matters (e.g., this satellite's conjunction risk estimates, its current tasking queue, its sensor health), the decentralized policy can nearly match the quality of a fully centralized controller. If critical information is hidden (e.g., a conjunction event is visible only to a different satellite), the decentralized policy will necessarily make suboptimal decisions — and the CTDE training process will encode this degradation gracefully rather than catastrophically.

## Full example: independent Q-learning vs. coordination in a spectrum game

Consider two satellite operators sharing a frequency band. At each time step, each operator chooses to transmit on channel A or channel B. If both choose the same channel, interference degrades both signals (negative reward). If they choose different channels, both receive full bandwidth (positive reward). This is a coordination game — a game where there are multiple Nash equilibria and the challenge is landing on one.

```python
import numpy as np
from collections import defaultdict

# ── Game definition ────────────────────────────────────────────────────────────
# Two players simultaneously choose channel A (0) or channel B (1).
# Payoff matrix (each cell is [Op1 reward, Op2 reward]):
#
#            Op2: A      Op2: B
# Op1: A    (-1, -1)   (+2, +2)
# Op1: B    (+2, +2)   (-1, -1)
#
# Two pure Nash equilibria: (A, B) and (B, A). Both are coordination equilibria.
# Independent Q-learning may converge to neither — it may cycle.

PAYOFF = np.array([
    [[-1, -1], [+2, +2]],   # Op1 plays A: vs Op2 A, vs Op2 B
    [[+2, +2], [-1, -1]],   # Op1 plays B: vs Op2 A, vs Op2 B
])

N_ACTIONS = 2
ALPHA = 0.1       # Q-learning rate
GAMMA = 0.0       # no temporal discounting (one-shot game repeated)
EPSILON_START = 1.0
EPSILON_END = 0.05
N_EPISODES = 5000

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(q_values))


def run_independent_q_learning(seed=0):
    np.random.seed(seed)
    # Each operator has its own Q-table: Q[action] -> expected reward
    # (No state in this repeated one-shot game; single-state Q-table)
    q1 = np.zeros(N_ACTIONS)
    q2 = np.zeros(N_ACTIONS)

    results = []
    for ep in range(N_EPISODES):
        epsilon = max(EPSILON_END, EPSILON_START - ep * (EPSILON_START - EPSILON_END) / N_EPISODES)

        a1 = epsilon_greedy(q1, epsilon)
        a2 = epsilon_greedy(q2, epsilon)

        r1, r2 = PAYOFF[a1][a2]

        # Independent Q-updates — each agent treats the other as part of the environment
        q1[a1] += ALPHA * (r1 - q1[a1])
        q2[a2] += ALPHA * (r2 - q2[a2])

        results.append((a1, a2, r1, r2))

    return q1, q2, results


def run_coordinated(seed=0):
    """
    Coordinated approach: one agent (Op2) acts as the follower.
    Op1 commits to a channel; Op2 best-responds.
    Implements a simple leader-follower Stackelberg equilibrium.
    """
    np.random.seed(seed)
    q1 = np.zeros(N_ACTIONS)

    results = []
    for ep in range(N_EPISODES):
        epsilon = max(EPSILON_END, EPSILON_START - ep * (EPSILON_START - EPSILON_END) / N_EPISODES)

        a1 = epsilon_greedy(q1, epsilon)
        # Op2 always best-responds: pick the channel different from Op1's choice
        a2 = 1 - a1  # perfectly complement Op1

        r1, r2 = PAYOFF[a1][a2]
        q1[a1] += ALPHA * (r1 - q1[a1])

        results.append((a1, a2, r1, r2))

    return q1, results


# ── Run and report ─────────────────────────────────────────────────────────────
q1_indep, q2_indep, indep_results = run_independent_q_learning(seed=7)
q1_coord, coord_results = run_coordinated(seed=7)

# Compute average reward per episode over the last 500 episodes
window = 500
avg_indep = np.mean([r[2] + r[3] for r in indep_results[-window:]]) / 2  # per agent
avg_coord = np.mean([r[2] + r[3] for r in coord_results[-window:]]) / 2  # per agent

print("=== Spectrum Allocation Game ===")
print(f"\nIndependent Q-learning (last {window} episodes):")
print(f"  Q1 values: {q1_indep.round(3)}  (A={q1_indep[0]:.3f}, B={q1_indep[1]:.3f})")
print(f"  Q2 values: {q2_indep.round(3)}")
collision_rate = np.mean([r[0] == r[1] for r in indep_results[-window:]])
print(f"  Channel collision rate: {collision_rate:.1%}")
print(f"  Average reward per agent: {avg_indep:.3f}")

print(f"\nCoordinated (leader-follower, last {window} episodes):")
collision_rate_c = np.mean([r[0] == r[1] for r in coord_results[-window:]])
print(f"  Channel collision rate: {collision_rate_c:.1%}")
print(f"  Average reward per agent: {avg_coord:.3f}")

print("\n--- Interpretation ---")
print("Independent Q-learning may cycle or settle on a suboptimal equilibrium.")
print("Coordination (even simple leader-follower) achieves the efficient Nash outcome.")
```

The output will show that independent Q-learning often produces significant collision rates and lower average rewards than the coordinated approach. The collision rate under independent Q-learning can remain well above zero even after 5,000 episodes because neither agent has any mechanism to resolve the coordination ambiguity — when both Q-values are nearly equal, random tie-breaking leads to collisions roughly 50% of the time.

The lesson: even in the simplest possible coordination game, independent Q-learning fails to reliably find a coordinated equilibrium. The remaining lessons in this module are algorithms designed to overcome this failure.

## Key Takeaways

- **Multi-agent RL is not just RL with more agents.** The non-stationarity of the environment (other agents are learning simultaneously) invalidates convergence proofs from single-agent RL. Algorithms designed for single-agent settings will often cycle, fail to converge, or converge to poor equilibria in multi-agent settings.
- **The type of multi-agent setting determines the appropriate solution concept.** Cooperative games call for joint optimality. Zero-sum games admit unique Nash equilibria solvable by CFR. General-sum games require Nash or correlated equilibria, which may be multiple and hard to compute.
- **Coordination failure can be catastrophically worse than no coordination at all.** Two independently rational agents can together produce an outcome that is worse for both than if they had not moved at all, as in the satellite avoidance collision example. Game theory's value is diagnosing and preventing these failures.
- **Centralized training, decentralized execution (CTDE) is the practical organizing principle.** Train policies with global information and joint optimization. Deploy policies that use only local information. This is how coordination knowledge is baked into policies that must operate without communication.
- **Non-stationarity is not just a convergence technicality.** It is a practical operational problem: a strategy that was optimal against yesterday's adversary or partner may be exploitable or suboptimal against today's updated one. Multi-agent RL must account for co-evolution of all agents' policies.
- **The spectrum of equilibrium concepts covers different coordination mechanisms.** Nash equilibrium captures selfish rationality. Correlated equilibrium captures rule-based coordination through a mediator. Both are relevant to SSA: uncoordinated commercial operators may settle on Nash; regulated operators following ITU assignments are at a correlated equilibrium.

## Quiz

{{#quiz 01-multi-agent-problem.toml}}
