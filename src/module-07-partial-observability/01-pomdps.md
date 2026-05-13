# Lesson 1: Partially Observable Markov Decision Processes (POMDPs)


<!-- toc -->

## Where this fits

Module 3 built the MDP framework: an agent observes the full state, acts, and receives a reward. That assumption — full observability — is almost never true in practice. A ground telescope can only observe one satellite at a time. Radar cross-sections do not reveal the object's mass or fuel state. An adversarial satellite's maneuver intent is never transmitted openly.

POMDPs (Partially Observable Markov Decision Processes) extend MDPs to handle this gap. The underlying world still evolves as a Markov process, but the agent never sees the world directly. It sees a noisy, partial signal, and must infer what the world probably looks like.

This lesson connects two earlier threads: the MDP formalism from Module 3 and the Bayesian updating from Module 1 (lesson 2). The belief state — a probability distribution over what the true state might be — is how POMDPs bridge observations to decisions.

## The problem with treating observations as states

Before the formal definition, it is worth understanding the failure mode that POMDPs cure.

Suppose a telescope network tracks five resident space objects (RSOs). Each timestep, the operator selects one RSO to observe. The observation returns a noisy right ascension and declination measurement. A naive approach treats the most recent observation as the agent's "state" and trains an RL agent on it. This appears to work in a simulator.

The problem: the true orbital elements of *all five* RSOs determine the system risk, but the agent only ever observes *one* at a time. The "state" the agent sees is radically incomplete. It is as though a doctor tried to assess a patient's full health by looking at one vital sign, ignoring the others. The agent cannot tell whether unobserved RSOs are safe or approaching collision.

When this agent is deployed, it makes systematically overconfident decisions. It does not know what it does not know. The conjunction event it misses was on the satellite it had not observed in four days, and the agent had no way to represent that uncertainty.

POMDPs fix this by making uncertainty a first-class citizen. The agent maintains a belief distribution across all five RSOs at all times, even the ones not currently observed.

## The POMDP formulation

A POMDP is defined by a 7-tuple:

\[ \langle S, A, \Omega, T, R, O, \gamma \rangle \]

**Decoding** (extending the MDP 5-tuple with the two new pieces):

- \(S\): state space. The true world state, which the agent does *not* observe directly. For our telescope network, this is the true orbital elements of all five RSOs — a vector of position, velocity, and timestamp for each object.
- \(A\): action space. What the agent can do: which RSO to point the telescope at this timestep.
- \(\Omega\): observation space. What the agent actually receives: a noisy RA/Dec pair corresponding to the RSO currently being observed, or a null observation for unobserved RSOs.
- \(T(s' \mid s, a)\): transition function. The same as in an MDP. The true orbital mechanics propagate all five RSOs forward regardless of which one was observed. Observations do not affect the true state.
- \(R(s, a)\): reward function. Reward for catching high-risk conjunctions; penalties for missing them.
- \(O(o \mid s', a)\): observation function. The probability of receiving observation \(o\) given that the true state is \(s'\) and action \(a\) was taken. For the telescope, this is a Gaussian around the predicted RA/Dec of whichever RSO was pointed at, with measurement noise \(\sigma_{\text{obs}}\).
- \(\gamma\): discount factor, same as in an MDP.

The observation function \(O\) is the new piece. It explicitly models the noise and incompleteness of what the agent sees.

### The SSA telescope scenario in detail

State \(s\) at time \(t\): the true orbital elements of five RSOs, represented as five 6-vectors (position and velocity in the ECI frame, km and km/s). The state space is 30-dimensional and continuous.

Action \(a \in \{0, 1, 2, 3, 4\}\): which RSO the telescope is pointed at for this hour-long observation window.

Observation \(o \in \mathbb{R}^2 \cup \{\text{null}\}\): if you pointed at RSO \(i\), you receive a noisy RA/Dec pair for RSO \(i\). For all other RSOs, you receive null. This is the fundamental partial observability: at any timestep, four of the five RSOs are completely unobserved.

Transition: orbital propagation (Keplerian or J2-perturbed) moves each RSO's true state forward by one hour. Stochastic perturbations model atmospheric drag uncertainty and undetected micro-maneuvers.

Observation model: if RSO \(i\) is pointed at and its true state is \(s\), the observation is:

\[ o = \text{RA\_Dec}(s_i) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{obs}}^2 I) \]

where \(\text{RA\_Dec}(s_i)\) is the deterministic projection of the true orbital elements to sky coordinates, and \(\epsilon\) is Gaussian measurement noise.

## The belief state: sufficient statistic for the history

The agent cannot observe the true state \(s\). But it can maintain a **belief state**:

\[ b(s) = P(s \mid h) \]

where \(h = (a_0, o_0, a_1, o_1, \ldots, a_t, o_t)\) is the full history of actions and observations.

**Theorem (Astrom 1965):** The belief state \(b\) is a *sufficient statistic* for the history. That is, any additional information in \(h\) beyond what is encoded in \(b(s)\) is irrelevant for future decision-making.

**Decoding:** "Sufficient statistic" means: the optimal policy only needs to look at \(b\), not the raw history. Histories of different lengths and content that produce the same belief state should produce the same action. This is the analog of the Markov property for POMDPs. It says: maintain the belief distribution carefully and you lose nothing by discarding the raw history.

**The consequence:** a POMDP over states \(S\) reduces to an MDP over belief states \(\mathcal{B} = \Delta(S)\) (the simplex of probability distributions over \(S\)). The belief-space MDP has:
- States: beliefs \(b \in \mathcal{B}\)
- Actions: same as before
- Transition: the belief update (derived below)
- Reward: \(\rho(b, a) = \sum_s b(s) R(s, a)\) (expected reward under current belief)

The problem: \(\mathcal{B}\) is continuous and, for a 30-dimensional continuous state space, infinite-dimensional. Exact solution is computationally intractable. The rest of this lesson and the next are about making this tractable.

## The belief update formula

When the agent takes action \(a\) and receives observation \(o\), the belief must be updated. This is Bayes' rule applied to the POMDP structure.

The exact belief update is:

\[ b'(s') = \frac{O(o \mid s', a) \sum_s T(s' \mid s, a)\, b(s)}{P(o \mid b, a)} \]

where the normalizing constant is:

\[ P(o \mid b, a) = \sum_{s'} O(o \mid s', a) \sum_s T(s' \mid s, a)\, b(s) \]

**Decoding, term by term:**

- \(b'(s')\): the updated belief probability assigned to state \(s'\) after taking action \(a\) and receiving observation \(o\).
- \(O(o \mid s', a)\): the likelihood of observation \(o\) if the true state were \(s'\). High if \(o\) is consistent with \(s'\), near zero if it contradicts \(s'\). This is the "evidence" term, the same structure as the likelihood in Bayes' rule from Module 1.
- \(\sum_s T(s' \mid s, a)\, b(s)\): the **prediction step**. Before seeing \(o\), we predict the distribution of the next state \(s'\) by marginalizing over the current belief \(b(s)\) and the transition dynamics \(T\). This is the prior over \(s'\) before the observation arrives.
- \(P(o \mid b, a)\): normalization constant, ensuring \(b'\) sums to 1. This is the probability of seeing observation \(o\) from all possible true states, weighted by the current belief and transitions.

**Connection to Bayes' rule:** The structure is identical to what you saw in Module 1:

\[ \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}} \]

Here, the "prior" is \(\sum_s T(s' \mid s, a)\, b(s)\) (predicted next state distribution), the "likelihood" is \(O(o \mid s', a)\) (how surprising is this observation given each possible next state), and the "evidence" is the normalizing constant.

### Two-step interpretation

The belief update happens in two phases, which is helpful computationally:

**Step 1 — Predict:** propagate the current belief through the dynamics, ignoring the new observation:
\[ \bar{b}(s') = \sum_s T(s' \mid s, a)\, b(s) \]

This is the "prediction step" in a Kalman filter (or any Bayesian filter). Before seeing the new observation, we advance our uncertainty forward in time using the known physics.

**Step 2 — Update:** weight the predicted distribution by the observation likelihood and renormalize:
\[ b'(s') = \frac{O(o \mid s', a)\, \bar{b}(s')}{\sum_{s''} O(o \mid s'', a)\, \bar{b}(s'')} \]

For our telescope problem: Step 1 propagates all five RSOs forward one hour via orbital mechanics, increasing position uncertainty. Step 2 collapses the uncertainty on whichever RSO was observed (using the RA/Dec measurement), while leaving the unobserved RSOs' uncertainties untouched.

## POMDP solutions

The fully optimal solution to a POMDP is value iteration in belief space. It is rarely tractable, but understanding it motivates the approximations.

### Exact: belief-space value iteration

The value of a belief state \(b\) satisfies the Bellman equation:

\[ V^*(b) = \max_a \left[ \rho(b, a) + \gamma \sum_o P(o \mid b, a)\, V^*(\tau(b, a, o)) \right] \]

where \(\tau(b, a, o)\) is the belief update operator (the formula above). It can be shown that \(V^*\) is piecewise-linear and convex over the belief simplex, and is the upper envelope of finitely many hyperplanes (alpha vectors). This is the PWLC representation.

For small discrete state spaces (say, fewer than 50 states), this can be computed exactly. For continuous or large discrete spaces, it is intractable. The orbital mechanics problem has a continuous, 30-dimensional state space — exact methods do not apply.

### Approximate: PBVI and SARSOP

**Point-Based Value Iteration (PBVI)** samples a reachable set of belief points and performs the Bellman backup only at those beliefs. It maintains the alpha vector representation but only over the sampled points. This is practical for state spaces up to hundreds of states.

**SARSOP** (Successive Approximations of the Reachable Space under Optimal Policies) improves on PBVI by guiding the sampling toward beliefs that are actually reachable under good policies. For medium-sized problems (hundreds to thousands of states), SARSOP is the current state of the art in exact-approximate methods.

Neither is practical for the continuous 30-dimensional orbital state space.

### Deep: DRQN with LSTM

For large or continuous POMDPs, the standard modern approach is the **Deep Recurrent Q-Network (DRQN)**. Instead of maintaining an explicit belief state, a recurrent neural network (LSTM or GRU) processes the sequence of observations and implicitly maintains a compressed representation of belief in its hidden state.

Architecture:

```
observation_t --> [embedding layer] --> LSTM --> [Q-head] --> Q(a_1), ..., Q(a_k)
                                         |
                                  hidden state h_t
                                  (carries memory across steps)
```

The LSTM's hidden state \(h_t\) plays the role of the belief \(b_t\). It is not an explicit probability distribution — it is a learned, dense representation trained end-to-end to produce good Q-values. The memory it maintains captures exactly what is needed to make good decisions, no more and no less.

When to use each:

| Method | State space | When to use |
|--------|-------------|-------------|
| Exact PWLC | Small discrete (< 50 states) | Well-defined toy problems, proofs |
| PBVI / SARSOP | Medium discrete (100s-1000s) | Research, detailed SSA threat models |
| DRQN / LSTM | Large continuous | Production SSA, multi-RSO tracking |

## Full Python code: POMDP simulator with particle filter belief

The following implements the core POMDP simulator for the five-RSO observation problem, along with a particle filter for tractable belief updating (particle filters are covered in depth in lesson 2; here we show the full integration).

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Orbital propagation (simplified two-body Keplerian) ─────────────────────

@dataclass
class OrbitalState:
    """Position (km) and velocity (km/s) of one RSO in ECI frame."""
    pos: np.ndarray   # shape (3,)
    vel: np.ndarray   # shape (3,)

    def to_ra_dec(self) -> np.ndarray:
        """Project ECI position to right ascension and declination (degrees)."""
        x, y, z = self.pos
        r = np.linalg.norm(self.pos)
        dec = np.degrees(np.arcsin(z / r))
        ra  = np.degrees(np.arctan2(y, x)) % 360.0
        return np.array([ra, dec])

def propagate_keplerian(state: OrbitalState, dt_hours: float,
                         process_noise_km: float = 0.5) -> OrbitalState:
    """
    Propagate orbital state forward by dt_hours.
    Uses simplified linear propagation with stochastic perturbation.
    A full implementation would use numerical integration (RK4/SGP4).
    """
    mu_km3_s2 = 398600.4418   # Earth's gravitational parameter
    dt_sec = dt_hours * 3600.0
    r = np.linalg.norm(state.pos)
    # Simplified: tangential velocity adjustment for circular orbit
    omega = np.sqrt(mu_km3_s2 / r**3)          # rad/s
    angle = omega * dt_sec                       # angle swept
    c, s = np.cos(angle), np.sin(angle)
    # Rotation about z-axis (equatorial plane orbit approximation)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    new_pos = R @ state.pos + np.random.randn(3) * process_noise_km
    new_vel = R @ state.vel + np.random.randn(3) * (process_noise_km / dt_sec)
    return OrbitalState(pos=new_pos, vel=new_vel)

# ── Observation model ────────────────────────────────────────────────────────

OBS_NOISE_DEG = 0.01   # 0.01 degree ~ 36 arcsecond measurement noise

def observe(state: OrbitalState, noise_std: float = OBS_NOISE_DEG) -> np.ndarray:
    """Return noisy RA/Dec observation for a given RSO state."""
    true_ra_dec = state.to_ra_dec()
    return true_ra_dec + np.random.randn(2) * noise_std

def obs_likelihood(obs: np.ndarray, state: OrbitalState,
                   noise_std: float = OBS_NOISE_DEG) -> float:
    """P(obs | state): Gaussian likelihood of the observation."""
    predicted = state.to_ra_dec()
    diff = obs - predicted
    # Handle RA wraparound at 360 degrees
    diff[0] = (diff[0] + 180) % 360 - 180
    log_lik = -0.5 * np.sum((diff / noise_std) ** 2)
    log_lik -= np.log(2 * np.pi * noise_std**2)
    return np.exp(log_lik)

# ── POMDP environment ────────────────────────────────────────────────────────

class SatelliteObservationPOMDP:
    """
    5-RSO ground telescope POMDP.
    State: list of 5 OrbitalState objects.
    Action: integer in {0,1,2,3,4} (which RSO to point at).
    Observation: (action, noisy RA/Dec) tuple.
    """
    def __init__(self, n_rsos: int = 5, seed: int = 42):
        self.n_rsos = n_rsos
        rng = np.random.default_rng(seed)
        # Initialize RSOs at roughly circular LEO orbits
        self.true_states: List[OrbitalState] = []
        for i in range(n_rsos):
            radius = 6778.0 + rng.uniform(-200, 200)   # ~400 km altitude
            angle  = rng.uniform(0, 2 * np.pi)
            pos = radius * np.array([np.cos(angle), np.sin(angle), rng.uniform(-0.1, 0.1)])
            speed = np.sqrt(398600.4418 / radius)      # circular orbit speed
            vel = speed * np.array([-np.sin(angle), np.cos(angle), 0.0])
            self.true_states.append(OrbitalState(pos=pos, vel=vel))
        self.t = 0

    def step(self, action: int) -> Tuple[int, np.ndarray, float]:
        """
        Take one observation step.
        Returns (action, observation, reward).
        Reward: 1.0 for any observation (simplified; real reward would
        depend on conjunction risk reduction achieved).
        """
        # True state propagation (happens regardless of action)
        self.true_states = [propagate_keplerian(s, dt_hours=1.0)
                            for s in self.true_states]
        # Observation: only the pointed-at RSO
        obs = observe(self.true_states[action])
        reward = 1.0
        self.t += 1
        return action, obs, reward

# ── Particle filter belief ───────────────────────────────────────────────────

class ParticleBeliefState:
    """
    Particle filter approximation to the POMDP belief state.
    Each particle is a list of 5 OrbitalState objects (one per RSO).
    """
    def __init__(self, n_particles: int = 500,
                 initial_states: Optional[List[OrbitalState]] = None,
                 init_noise_km: float = 5.0):
        self.N = n_particles
        # Particles: list of (list of 5 OrbitalState)
        self.particles: List[List[OrbitalState]] = []
        self.weights = np.ones(n_particles) / n_particles

        if initial_states is not None:
            # Spread particles around the initial estimate
            for _ in range(n_particles):
                particle = []
                for s in initial_states:
                    noisy_pos = s.pos + np.random.randn(3) * init_noise_km
                    noisy_vel = s.vel + np.random.randn(3) * (init_noise_km / 3600)
                    particle.append(OrbitalState(pos=noisy_pos, vel=noisy_vel))
                self.particles.append(particle)
        else:
            raise ValueError("Must provide initial_states for particle initialization")

    def predict(self, dt_hours: float = 1.0) -> None:
        """Step 1: propagate all particles through orbital dynamics."""
        new_particles = []
        for particle in self.particles:
            new_particle = [propagate_keplerian(s, dt_hours) for s in particle]
            new_particles.append(new_particle)
        self.particles = new_particles

    def update(self, action: int, obs: np.ndarray) -> None:
        """Step 2: weight particles by observation likelihood and resample."""
        new_weights = np.zeros(self.N)
        for i, particle in enumerate(self.particles):
            # Only the observed RSO contributes likelihood
            lik = obs_likelihood(obs, particle[action])
            new_weights[i] = self.weights[i] * lik

        # Normalize
        total = new_weights.sum()
        if total < 1e-300:
            # Particle deprivation: reset weights to uniform
            print("Warning: particle deprivation detected. Resetting weights.")
            new_weights = np.ones(self.N) / self.N
        else:
            new_weights /= total

        self.weights = new_weights
        self._systematic_resample()

    def _systematic_resample(self) -> None:
        """Systematic resampling: low variance, O(N) time."""
        positions = (np.arange(self.N) + np.random.uniform()) / self.N
        cumsum = np.cumsum(self.weights)
        i, j = 0, 0
        new_particles = []
        while i < self.N:
            if positions[i] < cumsum[j]:
                new_particles.append(self.particles[j])
                i += 1
            else:
                j += 1
        self.particles = new_particles
        self.weights = np.ones(self.N) / self.N

    def mean_state(self) -> List[np.ndarray]:
        """Return the weighted mean position of each RSO."""
        means = []
        for rso_idx in range(len(self.particles[0])):
            pos_sum = np.zeros(3)
            for i, particle in enumerate(self.particles):
                pos_sum += self.weights[i] * particle[rso_idx].pos
            means.append(pos_sum)
        return means

    def position_uncertainty(self) -> List[float]:
        """Return position uncertainty (std dev, km) for each RSO."""
        uncertainties = []
        means = self.mean_state()
        for rso_idx in range(len(self.particles[0])):
            var = 0.0
            for i, particle in enumerate(self.particles):
                diff = particle[rso_idx].pos - means[rso_idx]
                var += self.weights[i] * np.dot(diff, diff)
            uncertainties.append(np.sqrt(var))
        return uncertainties

# ── Demonstration: belief divergence when ignoring partial observability ─────

def demonstrate_belief_vs_naive(n_steps: int = 20, seed: int = 0) -> None:
    """
    Compare:
    (A) Naive agent: treats last observation as ground truth for all RSOs.
    (B) Belief agent: maintains particle filter over all RSO states.
    Shows that (A) accumulates large position errors on unobserved RSOs.
    """
    np.random.seed(seed)
    env = SatelliteObservationPOMDP(n_rsos=5, seed=seed)

    # Rough initial estimate (slightly wrong, as is realistic)
    initial_estimate = [OrbitalState(
        pos=s.pos + np.random.randn(3) * 2.0,
        vel=s.vel + np.random.randn(3) * 0.001
    ) for s in env.true_states]

    belief = ParticleBeliefState(
        n_particles=300,
        initial_states=initial_estimate,
        init_noise_km=3.0
    )

    # Naive model: just stores the last known position of each RSO
    naive_pos = [s.pos.copy() for s in initial_estimate]

    print(f"{'Step':>4}  {'RSO':>3}  {'Belief err (km)':>15}  {'Naive err (km)':>14}")
    print("-" * 44)

    # Always observe RSO 0 to make the discrepancy clear for RSOs 1-4
    action = 0

    for step in range(n_steps):
        act, obs, _ = env.step(action)
        belief.predict(dt_hours=1.0)
        belief.update(act, obs)

        # Update naive: only RSO 0 gets updated
        naive_pos[0] = np.array([
            obs[0] / 180 * np.pi,  # rough inverse projection (illustrative)
            obs[1] / 180 * np.pi,
            0.0
        ]) * 6778.0  # very rough inversion

        # Report error for RSO 4 (never observed)
        rso = 4
        true_pos = env.true_states[rso].pos
        belief_mean_pos = belief.mean_state()[rso]
        belief_err = np.linalg.norm(true_pos - belief_mean_pos)
        naive_err  = np.linalg.norm(true_pos - naive_pos[rso])

        if step % 4 == 0:
            print(f"{step:>4}  {rso:>3}  {belief_err:>15.1f}  {naive_err:>14.1f}")

    print()
    print("The naive agent's error on unobserved RSOs grows unbounded.")
    print("The belief agent maintains a calibrated (though imperfect) estimate.")

if __name__ == "__main__":
    demonstrate_belief_vs_naive(n_steps=20)
```

## The common failure mode: observation-as-state

The code above demonstrates a fundamental failure mode: treating the most recent observation as the state.

For RSO 4, which was never observed in the 20-step run, the naive agent's position error grows linearly — it does not propagate orbital mechanics for unobserved objects, so its last known position drifts arbitrarily far from truth. After 20 hours, the error can be hundreds of kilometers.

The belief agent, by contrast, propagates all RSOs through orbital mechanics at every timestep (even unobserved ones). Its uncertainty grows with time (the particle cloud spreads as uncertainty accumulates), but its mean estimate remains physically sensible. When the agent eventually points at RSO 4, the belief update will snap the estimate toward the truth.

This is the key POMDP insight: **maintaining calibrated uncertainty over unseen parts of the state is as important as processing the observations you do receive.** The unobserved RSOs are not frozen in place. The world continues to evolve, and a well-designed agent knows it.

## Why POMDP solutions are hard

The belief-space MDP has a continuous, infinite-dimensional state space (all probability distributions over the original state space). Value iteration in this space:

- For \(|S|\) states and \(|A|\) actions over \(|\Omega|\) observations, each backup generates \(|A| \cdot |\Omega|\) new alpha vectors.
- After \(n\) iterations, the value function is represented by up to \(|A|^n \cdot |\Omega|^n\) alpha vectors — exponential growth.
- Pruning removes dominated alpha vectors, but the worst case is still exponential.

For continuous state spaces (like our orbital mechanics problem), even the first iteration of belief-space value iteration requires integrating over an infinite state space. Exact methods fail entirely.

This is why the practical path is either (a) approximate belief representations like particle filters (tractable, scalable, requires no closed-form observation model) or (b) implicit belief via recurrent neural networks (DRQN), which learns what to remember without ever explicitly representing the belief distribution.

## Key Takeaways

- A POMDP extends an MDP with an observation function \(O(o \mid s', a)\) that separates the true world state from what the agent actually sees. In SSA, the true orbital state of all RSOs always exists, but you only partially observe it.
- The belief state \(b(s) = P(s \mid \text{history})\) is the sufficient statistic for the observation history. Any policy that conditions on the raw history can be replaced by one that conditions on the belief, without loss.
- The belief update is two-step: predict (propagate through dynamics) then update (reweight by observation likelihood). This is Bayes' rule applied sequentially, identical in structure to the Bayesian updating from Module 1.
- Exact POMDP solutions are computationally intractable except for small discrete problems. Practical approaches use particle filters for moderate-scale problems or recurrent neural networks (DRQN) for large-scale continuous problems.
- Ignoring partial observability — treating the most recent observation as the full state — causes systematic errors. Unobserved parts of the state are not frozen; the world evolves while you are looking elsewhere, and a correct agent represents that uncertainty.
- The POMDP framework is the right foundation for multi-RSO SSA: the ground station has sensors that cover only a small fraction of the orbital environment at any moment, and a principled treatment of what is unknown is essential for correct risk assessment.

{{#quiz 01-pomdps.toml}}
