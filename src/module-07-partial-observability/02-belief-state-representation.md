# Lesson 2: Belief State Representation

## Where this fits

Lesson 1 defined what a belief state is: a probability distribution \\(b(s) = P(s \mid \text{history})\\) that summarizes everything the agent knows about the true world state. It showed that maintaining a belief is both necessary (ignoring partial observability causes systematic errors) and sufficient (the belief is all you need to make optimal decisions).

This lesson covers the practical question: how do you actually *represent* and *update* a belief distribution in code? The true answer — an arbitrary probability distribution over a continuous, high-dimensional state space — is almost never directly representable. The art of POMDP engineering is choosing a representation that is accurate enough to support good decisions and cheap enough to run in real time.

We examine four approaches: exact discrete updates, Gaussian filters, particle filters, and neural implicit belief. Each occupies a different point on the accuracy-compute tradeoff.

## Exact belief: the discrete case

For a POMDP with a small, finite state space, the belief is just a probability vector — one non-negative entry per state, summing to 1. The update from lesson 1 is a matrix-vector multiply followed by element-wise multiplication and normalization.

### A four-state SSA example

A simplified SSA scenario with a small discrete state space: a single RSO has four possible orbit types, labeled by their operational risk:

- State 0: "Safe orbit, slowly drifting" — routine observation
- State 1: "Safe orbit, approaching conjunction window" — moderate risk
- State 2: "In conjunction window, no maneuver detected" — high risk
- State 3: "Maneuver detected, new orbit uncertain" — high uncertainty

Each hour, the RSO transitions between states according to orbital dynamics and the probability of maneuver initiation. The telescope observes one of three observation classes: "no anomaly", "possible conjunction", or "maneuver signature detected."

```python
import numpy as np

# ── Exact discrete belief update ────────────────────────────────────────────

# Transition matrix T[s, a, s'] = P(s' | s, a)
# Here we have one action (observe), so T[s, s'] = P(s' | s)
T = np.array([
    [0.85, 0.13, 0.02, 0.00],   # from Safe/slow: mostly stays safe
    [0.10, 0.70, 0.18, 0.02],   # from Safe/approaching: may enter window
    [0.05, 0.15, 0.65, 0.15],   # from Conjunction: may trigger maneuver
    [0.20, 0.30, 0.30, 0.20],   # from Maneuver: highly uncertain
])
# T must have rows that sum to 1
assert np.allclose(T.sum(axis=1), 1.0)

# Observation matrix O[s', obs] = P(obs | s')
# Three observations: 0=no_anomaly, 1=possible_conjunction, 2=maneuver_signature
O = np.array([
    [0.90, 0.09, 0.01],   # Safe/slow: almost always no anomaly
    [0.50, 0.45, 0.05],   # Safe/approaching: often looks normal, sometimes flagged
    [0.10, 0.70, 0.20],   # Conjunction: usually flags possible conjunction
    [0.05, 0.25, 0.70],   # Maneuver: usually shows maneuver signature
])
# O must have rows that sum to 1
assert np.allclose(O.sum(axis=1), 1.0)

def exact_belief_update(b: np.ndarray, obs: int) -> np.ndarray:
    """
    Update belief b after receiving discrete observation obs.
    Returns new belief b'.
    
    Two steps:
      1. Predict: b_pred[s'] = sum_s T[s, s'] * b[s]
      2. Update:  b'[s'] = O[s', obs] * b_pred[s'] / Z
    """
    # Step 1: prediction (matrix-vector multiply)
    b_pred = T.T @ b        # shape (4,), b_pred[s'] = sum_s T[s,s'] * b[s]

    # Step 2: update (element-wise multiply by likelihood, renormalize)
    likelihood = O[:, obs]  # P(obs | s') for each s'
    b_new = likelihood * b_pred
    normalizer = b_new.sum()
    if normalizer < 1e-12:
        raise ValueError(f"Zero-probability observation {obs} given belief {b}")
    return b_new / normalizer

# Example: prior belief (uniform -- we know nothing)
b = np.array([0.25, 0.25, 0.25, 0.25])

observations = [0, 0, 1, 1, 2, 1, 2]   # a realistic observation sequence
labels = ["no_anomaly", "no_anomaly", "possible_conjunction",
          "possible_conjunction", "maneuver_signature",
          "possible_conjunction", "maneuver_signature"]

print("Exact belief update over observation sequence:")
print(f"{'Obs':<24}  S0:Safe/slow  S1:Approach  S2:Conjunct  S3:Maneuver")
for obs, label in zip(observations, labels):
    b = exact_belief_update(b, obs)
    print(f"{label:<24}  {b[0]:.3f}        {b[1]:.3f}       {b[2]:.3f}       {b[3]:.3f}")
```

Exact belief update is fast and exact. Its limitation is the state space: with a 30-dimensional continuous orbital state, the "belief vector" has infinitely many entries. The exact approach is reserved for small problems used for validation or simple threat models.

## Gaussian belief: the Kalman filter family

For linear Gaussian systems, the belief \\(b(s)\\) is always Gaussian. A Gaussian over \\(d\\) dimensions is parameterized by just \\(d + d^2/2\\) numbers: a mean vector \\(\mu\\) and a covariance matrix \\(\Sigma\\). This is far more compact than a full distribution, and the updates are analytically tractable.

The **Kalman filter** (KF) is the exact Bayesian filter for linear Gaussian systems. For nonlinear orbital mechanics, the **Extended Kalman Filter (EKF)** or **Unscented Kalman Filter (UKF)** approximates the nonlinear dynamics with a linearization.

### Kalman filter summary

For a linear system \\(s_{t+1} = A s_t + B u_t + w_t\\) and observation model \\(o_t = H s_t + v_t\\), where \\(w_t \sim \mathcal{N}(0, Q)\\) and \\(v_t \sim \mathcal{N}(0, R)\\):

**Predict step:**
\\[ \bar{\mu}_{t+1} = A \mu_t + B u_t \\]
\\[ \bar{\Sigma}_{t+1} = A \Sigma_t A^\top + Q \\]

**Update step:**
\\[ K = \bar{\Sigma}_{t+1} H^\top (H \bar{\Sigma}_{t+1} H^\top + R)^{-1} \\]
\\[ \mu_{t+1} = \bar{\mu}_{t+1} + K(o_{t+1} - H \bar{\mu}_{t+1}) \\]
\\[ \Sigma_{t+1} = (I - KH)\bar{\Sigma}_{t+1} \\]

**Decoding:**
- \\(\bar{\mu}\\), \\(\bar{\Sigma}\\): predicted mean and covariance (before observing \\(o_{t+1}\\))
- \\(K\\): Kalman gain. A matrix that says "how much should the new observation shift the mean?" Large \\(K\\) means we trust the observation heavily; small \\(K\\) means we trust our prediction.
- \\(o_{t+1} - H \bar{\mu}_{t+1}\\): the **innovation** — the difference between the actual observation and what we predicted we would see. If this is small, the world is evolving as expected. If it is large, something unexpected happened.
- \\((I - KH)\bar{\Sigma}_{t+1}\\): the covariance shrinks after an observation, because we learned something.

The Kalman filter is the foundation of operational space surveillance. Organizations track orbital objects using variants of the EKF or UKF applied to radar and optical measurements. The particle filter below is a generalization that handles non-Gaussian, multi-modal, and highly nonlinear cases that Kalman filters cannot.

## Particle filters: the general-purpose belief

A **particle filter** represents the belief as a set of \\(N\\) weighted samples (particles):

\\[ b(s) \approx \sum_{i=1}^{N} w^{(i)} \delta(s - s^{(i)}) \\]

where \\(s^{(i)}\\) is the \\(i\\)th particle's state, \\(w^{(i)}\\) is its weight (with \\(\sum_i w^{(i)} = 1\\)), and \\(\delta\\) is the Dirac delta function (a point mass at \\(s^{(i)}\\)).

**Decoding:** Each particle \\(s^{(i)}\\) is a hypothesis about the current true state. The weight \\(w^{(i)}\\) is how likely that hypothesis is, given all observations so far. The distribution they collectively represent approximates the true posterior belief.

The particle filter update follows the same two-step logic as the exact update, but applied to particles rather than a probability table:

### Sequential importance resampling (SIR)

The standard particle filter algorithm is called SIR:

1. **Predict**: propagate each particle through the dynamics model (with noise), giving \\(s^{(i)} \leftarrow p(s_{t+1} \mid s_t^{(i)}, a)\\).
2. **Update (importance weighting)**: multiply each particle's weight by the observation likelihood: \\(w^{(i)} \leftarrow w^{(i)} \cdot O(o \mid s^{(i)}, a)\\). Renormalize.
3. **Resample**: draw \\(N\\) new particles from the weighted distribution, replacing the old particles. Reset all weights to \\(1/N\\).

The resampling step focuses computational effort on high-probability regions. Without resampling, a few particles would accumulate nearly all the weight and the estimate would degrade.

## Full particle filter implementation for satellite tracking

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── Simplified state: 2D equatorial orbit (for clarity) ─────────────────────

@dataclass
class Particle:
    """Single particle: 2D orbital state (angle and angular rate)."""
    theta: float    # orbital angle (radians)
    omega: float    # angular rate (radians/hour)
    r:     float    # orbital radius (km)

    def to_ra_dec(self) -> np.ndarray:
        """Return (RA, Dec) assuming equatorial orbit and observer at Earth center."""
        ra  = np.degrees(self.theta) % 360.0
        dec = 0.0  # equatorial orbit: always Dec=0 (simplified)
        return np.array([ra, dec])

def propagate_particle(p: Particle, dt_hours: float = 1.0) -> Particle:
    """Propagate one particle forward by dt_hours with process noise."""
    mu = 398600.4418  # km^3/s^2
    n  = np.sqrt(mu / p.r**3) * 3600  # rad/hr, mean motion
    new_theta = p.theta + n * dt_hours + np.random.randn() * 0.0001
    new_omega = n + np.random.randn() * 1e-5    # small noise on angular rate
    new_r     = p.r + np.random.randn() * 0.01  # slight radius variation
    return Particle(theta=new_theta % (2 * np.pi), omega=new_omega, r=new_r)

def obs_likelihood_2d(obs_ra_deg: float, particle: Particle,
                      noise_std_deg: float = 0.01) -> float:
    """Gaussian likelihood P(obs | particle) for RA observation."""
    pred_ra = particle.to_ra_dec()[0]
    diff = (obs_ra_deg - pred_ra + 180) % 360 - 180  # handle wraparound
    return np.exp(-0.5 * (diff / noise_std_deg) ** 2)

# ── Resampling methods ───────────────────────────────────────────────────────

def multinomial_resample(particles: List[Particle],
                         weights: np.ndarray) -> List[Particle]:
    """Draw N samples from particles with replacement, weighted by weights."""
    N = len(particles)
    indices = np.random.choice(N, size=N, replace=True, p=weights)
    return [particles[i] for i in indices]

def systematic_resample(particles: List[Particle],
                        weights: np.ndarray) -> List[Particle]:
    """
    Systematic resampling: low-variance, O(N).
    Use one random number to generate N equally-spaced positions on [0,1].
    Produces more uniform coverage than multinomial.
    """
    N = len(particles)
    u = (np.arange(N) + np.random.uniform()) / N   # single draw, then uniform spacing
    cumsum = np.cumsum(weights)
    new_particles = []
    j = 0
    for i in range(N):
        while u[i] > cumsum[j]:
            j += 1
        new_particles.append(particles[j])
    return new_particles

def stratified_resample(particles: List[Particle],
                        weights: np.ndarray) -> List[Particle]:
    """
    Stratified resampling: N independent draws, one per stratum [k/N, (k+1)/N].
    Slightly higher variance than systematic but independent across strata.
    """
    N = len(particles)
    # One uniform draw per stratum
    u = (np.arange(N) + np.random.uniform(size=N)) / N
    cumsum = np.cumsum(weights)
    new_particles = []
    j = 0
    for i in range(N):
        while u[i] > cumsum[j]:
            j += 1
        new_particles.append(particles[j])
    return new_particles

def compare_resamplers(n_trials: int = 1000, N: int = 100) -> None:
    """
    Compare variance of unique-particle count across the three resamplers.
    Lower unique count means higher effective degeneracy (bad).
    Systematic typically achieves the lowest variance.
    """
    def run(resample_fn):
        counts = []
        for _ in range(n_trials):
            # Simulate a peaked weight distribution (one dominant particle)
            raw_weights = np.random.dirichlet(np.ones(N) * 0.5)
            new_p = resample_fn(list(range(N)), raw_weights)
            counts.append(len(set(new_p)))
        return np.mean(counts), np.std(counts)

    for name, fn in [("Multinomial", multinomial_resample),
                     ("Stratified ", stratified_resample),
                     ("Systematic ", systematic_resample)]:
        mean, std = run(fn)
        print(f"{name}: mean unique particles = {mean:.1f}  std = {std:.2f}")

# ── Particle filter class ────────────────────────────────────────────────────

class SatelliteParticleFilter:
    """
    Particle filter for tracking one equatorial RSO via RA observations.
    Demonstrates predict/update/resample cycle and particle deprivation handling.
    """
    def __init__(self, n_particles: int = 500, true_theta_0: float = 0.5,
                 true_r_km: float = 6778.0, init_noise_deg: float = 5.0):
        self.N = n_particles
        self.weights = np.ones(n_particles) / n_particles
        # Initialize particles around a rough initial estimate
        init_noise_rad = np.radians(init_noise_deg)
        mu = np.sqrt(398600.4418 / true_r_km**3) * 3600  # rad/hr
        self.particles = [
            Particle(
                theta=(true_theta_0 + np.random.randn() * init_noise_rad) % (2 * np.pi),
                omega=mu + np.random.randn() * 1e-4,
                r=true_r_km + np.random.randn() * 5.0
            )
            for _ in range(n_particles)
        ]

    def predict(self) -> None:
        """Propagate all particles through one hour of orbital dynamics."""
        self.particles = [propagate_particle(p) for p in self.particles]

    def update(self, obs_ra_deg: float, noise_std_deg: float = 0.01) -> None:
        """
        Reweight particles by observation likelihood.
        Handles particle deprivation by injecting roughening noise if needed.
        """
        new_weights = np.array([
            self.weights[i] * obs_likelihood_2d(obs_ra_deg, self.particles[i], noise_std_deg)
            for i in range(self.N)
        ])
        total = new_weights.sum()

        # Particle deprivation check
        if total < 1e-250:
            print(f"  [Warning] Particle deprivation at RA={obs_ra_deg:.2f}. "
                  "Applying roughening noise and reinitializing weights.")
            # Roughening: add noise to all particles to escape zero-weight regions
            self.particles = [
                Particle(
                    theta=(p.theta + np.random.randn() * 0.05) % (2 * np.pi),
                    omega=p.omega + np.random.randn() * 1e-4,
                    r=p.r + np.random.randn() * 2.0
                )
                for p in self.particles
            ]
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights = new_weights / total
            self.particles = systematic_resample(self.particles, self.weights)
            self.weights = np.ones(self.N) / self.N

    def effective_sample_size(self) -> float:
        """
        ESS = 1 / sum(w_i^2). 
        ESS close to N: good diversity. ESS << N: near-deprivation.
        """
        return 1.0 / np.sum(self.weights ** 2)

    def mean_theta(self) -> float:
        """Circular mean of particle angles (handles wraparound correctly)."""
        sin_mean = np.sum(self.weights * np.sin([p.theta for p in self.particles]))
        cos_mean = np.sum(self.weights * np.cos([p.theta for p in self.particles]))
        return np.arctan2(sin_mean, cos_mean) % (2 * np.pi)

    def std_theta_deg(self) -> float:
        """Standard deviation of particle angles in degrees."""
        thetas = np.array([p.theta for p in self.particles])
        mean   = self.mean_theta()
        diffs  = np.degrees(np.angle(np.exp(1j * (thetas - mean))))
        return np.std(diffs)

# ── Tracking demonstration ───────────────────────────────────────────────────

def run_tracking_demo(n_steps: int = 30) -> None:
    """
    Simulate a ground station observing an RSO every 3 steps (sparse observation).
    Shows how belief uncertainty grows between observations and collapses after.
    """
    np.random.seed(7)
    TRUE_THETA_0 = 0.5    # radians
    TRUE_R       = 6778.0 # km
    OBS_NOISE    = 0.01   # degrees

    pf = SatelliteParticleFilter(
        n_particles=300,
        true_theta_0=TRUE_THETA_0,
        true_r_km=TRUE_R,
        init_noise_deg=3.0
    )

    # Simulate true orbit
    true_theta = TRUE_THETA_0
    mu = np.sqrt(398600.4418 / TRUE_R**3) * 3600  # rad/hr

    print(f"{'Step':>4}  {'Observed':>8}  {'Error (deg)':>11}  {'Std (deg)':>9}  {'ESS':>6}")
    print("-" * 48)

    for step in range(n_steps):
        # Advance true state
        true_theta = (true_theta + mu) % (2 * np.pi)

        # Particle filter prediction
        pf.predict()

        # Observe only every 3 steps (sparse, realistic SSA cadence)
        observed = (step % 3 == 0)
        if observed:
            obs_ra = (np.degrees(true_theta) + np.random.randn() * OBS_NOISE) % 360.0
            pf.update(obs_ra, noise_std_deg=OBS_NOISE)

        est_theta = pf.mean_theta()
        err_deg   = abs((np.degrees(true_theta) - np.degrees(est_theta) + 180) % 360 - 180)
        std_deg   = pf.std_theta_deg()
        ess       = pf.effective_sample_size()

        if step % 3 == 0:
            obs_flag = "yes" if observed else "no"
            print(f"{step:>4}  {obs_flag:>8}  {err_deg:>11.4f}  {std_deg:>9.4f}  {ess:>6.0f}")

if __name__ == "__main__":
    print("=== Resampler variance comparison ===")
    compare_resamplers(n_trials=500, N=100)
    print()
    print("=== Sparse observation tracking ===")
    run_tracking_demo(n_steps=30)
```

## Particle deprivation: detection and prevention

Particle deprivation occurs when all particles have near-zero weight after an update — the true state has moved to a region of state space where no particles currently live. This causes the filter to fail silently: the belief becomes a bad approximation of the truth, and the agent does not know it.

**Detection:** monitor the effective sample size (ESS):

\\[ \text{ESS} = \frac{1}{\sum_{i=1}^N (w^{(i)})^2} \\]

**Decoding:** ESS measures how many "effective" particles are contributing to the estimate. If all weights are equal (\\(w^{(i)} = 1/N\\)), ESS = N (full diversity). If one particle has weight 1.0 and all others are 0, ESS = 1 (complete collapse).

An ESS below \\(N/10\\) is a warning sign. An ESS below \\(N/100\\) indicates imminent deprivation.

**Prevention strategies:**

1. **Roughening**: add small random noise to all particle states before resampling. This spreads particles slightly, preventing them from all clustering at the same point. Cost: slight blurring of the belief.

2. **Stratified and systematic resampling**: these resamplers have lower variance than multinomial resampling, meaning they spread the resampled particles more evenly across the weight distribution. They help by reducing variance in which particles are kept.

3. **Increased particle count**: more particles provide more coverage of state space. For the satellite tracking problem, 500-1000 particles is generally sufficient for single-RSO tracking; multi-RSO problems may need thousands.

4. **Particle injection**: maintain a small pool of "exploratory" particles sampled from the prior. When ESS drops below a threshold, inject a few of these into the filter. This ensures coverage of regions the weighted particles might have abandoned.

## Neural approaches: implicit belief representation

An alternative to explicit belief tracking is to use a recurrent neural network that reads in the observation history and outputs actions (or Q-values) directly, with no explicit probability distribution anywhere.

### DRQN: Deep Recurrent Q-Network

The standard architecture is DRQN (Hausknecht and Stone, 2015):

```
obs_t ── [Linear embedding] ──┐
                               ├──> LSTM cell ──> [Q-head] ──> Q(s_t, a_1), ..., Q(s_t, a_k)
hidden_{t-1} ──────────────────┘
     ↑
     └── hidden_t fed back next step
```

The LSTM maintains a hidden state \\(h_t\\) that compresses the observation history into a fixed-size vector. The Q-head maps \\(h_t\\) to Q-values for each action. Training uses standard DQN objectives (TD error minimization) with experience replay over sequences (not individual transitions, since the LSTM needs temporal context to build its hidden state).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network for POMDP-structured problems.
    The LSTM hidden state implicitly represents the agent's belief.
    """
    def __init__(self, obs_dim: int, n_actions: int,
                 embed_dim: int = 64, lstm_dim: int = 128):
        super().__init__()
        # Observation embedding
        self.embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        # LSTM: carries the implicit belief across timesteps
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True    # input shape: (batch, seq_len, embed_dim)
        )
        # Q-value head
        self.q_head = nn.Linear(lstm_dim, n_actions)

    def forward(self, obs_seq: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs_seq: (batch, seq_len, obs_dim)
        Returns Q-values at each timestep and updated hidden state.
        """
        # Embed each observation
        B, T, D = obs_seq.shape
        flat = obs_seq.view(B * T, D)
        embedded = self.embed(flat).view(B, T, -1)   # (batch, seq_len, embed_dim)

        # LSTM processes the sequence; hidden state carries belief
        lstm_out, new_hidden = self.lstm(embedded, hidden)  # (batch, seq_len, lstm_dim)

        # Q-values at each step
        q_values = self.q_head(lstm_out)   # (batch, seq_len, n_actions)
        return q_values, new_hidden

    def act_greedy(self, obs: torch.Tensor,
                   hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                  ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step greedy action selection.
        obs: (obs_dim,) — a single observation
        """
        obs_seq = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        with torch.no_grad():
            q_values, new_hidden = self.forward(obs_seq, hidden)
        action = q_values.squeeze().argmax().item()
        return action, new_hidden

# Example: DRQN for 5-RSO scheduling
# Observation: (RA, Dec) of the observed RSO + one-hot of which RSO was pointed at
# = 2 + 5 = 7 dimensional observation
model = DRQN(obs_dim=7, n_actions=5, embed_dim=32, lstm_dim=64)
print(f"DRQN parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass over a sequence of 20 observations
batch_obs = torch.randn(1, 20, 7)   # batch=1, seq_len=20, obs_dim=7
q_values, hidden = model(batch_obs)
print(f"Q-value output shape: {q_values.shape}")  # (1, 20, 5)
```

### Explicit vs. implicit belief: when to use each

| Criterion | Explicit (particle filter) | Implicit (DRQN) |
|-----------|---------------------------|-----------------|
| Interpretability | High: belief is a probability distribution you can inspect | Low: hidden state has no probabilistic interpretation |
| Accuracy | Principled: converges to true posterior with enough particles | Approximate: learns a heuristic compression |
| Handling new scenarios | Good: physics-based dynamics adapt without retraining | Poor: must retrain for new observation models |
| Scalability | Quadratic in RSO count for full joint tracking | Scales with model size, not state space |
| SSA recommendation | Single-RSO tracking, conjunction probability computation | Multi-RSO scheduling, policy optimization |

## SSA-specific challenges

**Sparse observations.** A ground station observes one RSO per timestep. With five RSOs, each RSO is on average observed only every five hours. During the unobserved intervals, belief uncertainty grows continuously. The particle filter handles this correctly by propagating all particles through orbital dynamics each step, even for unobserved RSOs. The DRQN must learn to maintain the relevant features in its LSTM hidden state across long un-observed stretches.

**Three-dimensional position uncertainty for conjunction probability.** The probability of collision between two RSOs depends on the joint uncertainty in their relative positions, not just the means. The standard covariance ellipsoid approach (used in operational conjunction analysis) is essentially a Gaussian belief representation for this joint state. The particle filter can represent non-Gaussian, multi-modal uncertainty (relevant when the orbit is poorly constrained after a maneuver); the Gaussian cannot.

The conjunction probability given a particle filter belief is computed via Monte Carlo:

```python
def conjunction_probability_monte_carlo(
    pf_rso1: SatelliteParticleFilter,
    pf_rso2: SatelliteParticleFilter,
    hard_body_radius_km: float = 0.01,   # 10 m combined hard-body radius
    n_samples: int = 1000
) -> float:
    """
    Estimate P(conjunction) by sampling pairs of particles from both filters
    and checking if the separation is less than the hard-body radius.
    This is Monte Carlo integration of the conjunction probability over the
    joint belief distribution.
    """
    count = 0
    for _ in range(n_samples):
        # Sample one particle from each filter
        i1 = np.random.choice(pf_rso1.N, p=pf_rso1.weights)
        i2 = np.random.choice(pf_rso2.N, p=pf_rso2.weights)
        p1 = pf_rso1.particles[i1]
        p2 = pf_rso2.particles[i2]
        # Compute separation using orbital radius and angle difference
        pos1 = p1.r * np.array([np.cos(p1.theta), np.sin(p1.theta), 0.0])
        pos2 = p2.r * np.array([np.cos(p2.theta), np.sin(p2.theta), 0.0])
        separation = np.linalg.norm(pos1 - pos2)
        if separation < hard_body_radius_km:
            count += 1
    return count / n_samples
```

This connects directly to the Monte Carlo methods introduced in Module 1, lesson 3. The particle filter *is* a running Monte Carlo estimate of the posterior belief; computing conjunction probability is one more layer of Monte Carlo on top.

## Key Takeaways

- Exact discrete belief update is a matrix-vector operation: predict with the transpose of the transition matrix, then reweight by the observation likelihood and normalize. Correct and fast for small discrete state spaces; intractable for continuous or large discrete spaces.
- Particle filters represent belief as a weighted point cloud. They are exact in the limit of infinite particles and handle arbitrary nonlinear dynamics and non-Gaussian distributions, which is critical for tracking satellites that may undergo unexpected maneuvers.
- The three-step particle filter loop — predict (propagate particles), update (reweight by observation likelihood), resample (focus particles on high-weight regions) — is a direct implementation of the predict-update Bayesian filter structure.
- Systematic resampling is preferred over multinomial resampling for lower variance in the particle diversity. The effective sample size (ESS) is the diagnostic to monitor; values below N/10 signal particle deprivation risk.
- DRQN uses an LSTM to implicitly represent belief in its hidden state, bypassing the need for explicit probability distributions. This scales to high-dimensional continuous observations but sacrifices interpretability and physics-based guarantees.
- For SSA, the particle filter is the principled choice for conjunction probability computation, since conjunction probability requires integrating over the joint position uncertainty — something the particle representation supports directly via Monte Carlo sampling.

{{#quiz 02-belief-state-representation.toml}}
