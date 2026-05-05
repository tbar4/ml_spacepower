# Module 7 Project: Particle-Filter Belief Tracker for RSO Tracking


<!-- toc -->

## What you are building

You will implement a bootstrap particle filter that tracks the orbital state of a resident space object (RSO) from noisy ground-based angular measurements. The filter maintains a set of weighted particles, each representing a hypothesis about the RSO's current position and velocity in Earth-centered inertial (ECI) coordinates. As new telescope observations arrive, particles are reweighted by their measurement likelihood and resampled. Between observations, particles propagate forward under simplified orbital dynamics plus a small stochastic perturbation modeling unmodeled forces.

The project connects Module 7's theory of belief states and particle deprivation to a concrete SSA tracking problem, and builds the belief-propagation infrastructure you will use in the Module 8 capstone.

## The scenario

A ground-based telescope tracks a single RSO in LEO at approximately 500 km altitude. The telescope takes an observation once per orbital period (~95 minutes) when the RSO passes overhead. Each observation is a right ascension (RA) and declination (Dec) measurement with 1 arcminute Gaussian noise. The RSO's true state is a six-dimensional vector [x, y, z, vx, vy, vz] in ECI coordinates.

## Step 1: orbital dynamics propagator

```python
import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH  = 6.371e6          # m

def two_body_dynamics(t, state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    a = -MU_EARTH / r**3
    return [vx, vy, vz, a*x, a*y, a*z]

def propagate(state: np.ndarray, dt: float, process_noise_std: float = 0.1) -> np.ndarray:
    """Propagate one orbital state forward by dt seconds with optional noise."""
    sol = solve_ivp(two_body_dynamics, [0, dt], state,
                    method="RK45", rtol=1e-8, atol=1e-10)
    propagated = sol.y[:, -1].copy()
    propagated[3:] += np.random.normal(0, process_noise_std, 3)
    return propagated

def circular_orbit_state(altitude_m: float, inclination_deg: float = 51.6) -> np.ndarray:
    r = R_EARTH + altitude_m
    v_circ = np.sqrt(MU_EARTH / r)
    inc = np.radians(inclination_deg)
    return np.array([r, 0.0, 0.0, 0.0, v_circ * np.cos(inc), v_circ * np.sin(inc)])
```

## Step 2: measurement model

```python
SITE_ECI = np.array([R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0])
OBS_NOISE_STD = np.radians(1.0 / 60.0)  # 1 arcminute in radians

def eci_to_radec(rso_eci: np.ndarray, site_eci: np.ndarray) -> np.ndarray:
    los = rso_eci[:3] - site_eci[:3]
    los_norm = los / np.linalg.norm(los)
    dec = np.arcsin(los_norm[2])
    ra  = np.arctan2(los_norm[1], los_norm[0])
    return np.array([ra, dec])

def simulate_observation(true_state: np.ndarray) -> np.ndarray:
    clean = eci_to_radec(true_state, SITE_ECI)
    return clean + np.random.normal(0, OBS_NOISE_STD, 2)

def observation_likelihood(particle_state: np.ndarray, observation: np.ndarray) -> float:
    predicted = eci_to_radec(particle_state, SITE_ECI)
    residual = observation - predicted
    residual[0] = (residual[0] + np.pi) % (2 * np.pi) - np.pi
    log_lik = -0.5 * np.sum((residual / OBS_NOISE_STD)**2)
    return np.exp(log_lik)
```

## Step 3: particle filter

```python
class OrbitalParticleFilter:
    def __init__(self, n_particles: int = 500, process_noise_std: float = 0.5):
        self.n = n_particles
        self.process_noise = process_noise_std
        self.particles = None
        self.weights   = None
        self._last_obs = None

    def initialize(self, prior_mean: np.ndarray, prior_std: np.ndarray):
        self.particles = prior_mean + np.random.randn(self.n, 6) * prior_std
        self.weights   = np.ones(self.n) / self.n

    def predict(self, dt: float):
        for i in range(self.n):
            self.particles[i] = propagate(self.particles[i], dt, self.process_noise)

    def update(self, observation: np.ndarray):
        self._last_obs = observation
        likelihoods = np.array([
            observation_likelihood(self.particles[i], observation)
            for i in range(self.n)
        ])
        self.weights *= likelihoods
        w_sum = self.weights.sum()
        if w_sum < 1e-300:
            print("WARNING: particle deprivation — reinitializing")
            self._handle_deprivation()
            return
        self.weights /= w_sum
        self._resample()

    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2). N = uniform, 1 = collapsed."""
        return 1.0 / np.sum(self.weights**2)

    def _resample(self):
        """Systematic resampling with roughening."""
        cumsum = np.cumsum(self.weights)
        positions = (np.arange(self.n) + np.random.uniform()) / self.n
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()
        self.weights   = np.ones(self.n) / self.n
        self.particles += np.random.randn(*self.particles.shape) * (self.process_noise * 0.1)

    def _handle_deprivation(self):
        """Inject fresh particles around the highest-likelihood region."""
        liks = np.array([
            observation_likelihood(self.particles[i], self._last_obs)
            for i in range(self.n)
        ])
        center = self.particles[np.argmax(liks)]
        self.particles = center + np.random.randn(self.n, 6) * np.array([1e4, 1e4, 1e4, 1, 1, 1])
        self.weights   = np.ones(self.n) / self.n

    def mean_estimate(self) -> np.ndarray:
        return (self.weights[:, None] * self.particles).sum(axis=0)

    def covariance_estimate(self) -> np.ndarray:
        mean = self.mean_estimate()
        diff = self.particles - mean
        return (self.weights[:, None] * diff).T @ diff
```

## Step 4: run the scenario

```python
def run_tracking_scenario(n_obs: int = 8, dt_orbit: float = 5700.0) -> dict:
    true_state = circular_orbit_state(altitude_m=500e3)
    prior_mean = true_state + np.array([5e3, 5e3, 5e3, 5, 5, 5])
    prior_std  = np.array([1e4, 1e4, 1e4, 10, 10, 10])

    pf = OrbitalParticleFilter(n_particles=500, process_noise_std=0.5)
    pf.initialize(prior_mean, prior_std)
    records = []

    for obs_idx in range(n_obs):
        true_state = propagate(true_state, dt_orbit, process_noise_std=0.0)
        pf.predict(dt_orbit)
        observation = simulate_observation(true_state)
        pf.update(observation)

        est = pf.mean_estimate()
        cov = pf.covariance_estimate()
        pos_error = np.linalg.norm(est[:3] - true_state[:3])
        ess = pf.effective_sample_size()

        records.append({
            "obs": obs_idx + 1,
            "position_error_m": pos_error,
            "ess": ess,
            "ess_fraction": ess / pf.n,
            "pos_std_km": np.sqrt(np.diag(cov)[:3]).mean() / 1e3,
        })
        print(f"Obs {obs_idx+1:2d} | pos_err={pos_error/1e3:8.2f} km | "
              f"ESS={ess:.0f}/{pf.n} | pos_std={records[-1]['pos_std_km']:.2f} km")

    return {"records": records, "filter": pf, "true_state": true_state}

result = run_tracking_scenario(n_obs=8)
```

## Step 5: visualize convergence

```python
import matplotlib.pyplot as plt

records   = result["records"]
obs_nums  = [r["obs"]              for r in records]
errors    = [r["position_error_m"] / 1e3 for r in records]
pos_stds  = [r["pos_std_km"]       for r in records]
ess_fracs = [r["ess_fraction"]     for r in records]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].semilogy(obs_nums, errors, "b-o")
axes[0].set_xlabel("Observation"); axes[0].set_ylabel("Position error (km)")
axes[0].set_title("Convergence: position error"); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(obs_nums, pos_stds, "r-o")
axes[1].set_xlabel("Observation"); axes[1].set_ylabel("Mean pos std (km)")
axes[1].set_title("Uncertainty reduction"); axes[1].grid(True, alpha=0.3)

axes[2].plot(obs_nums, ess_fracs, "g-o")
axes[2].set_ylim([0, 1])
axes[2].axhline(0.5, color="k", linestyle="--", alpha=0.5, label="50% ESS threshold")
axes[2].set_xlabel("Observation"); axes[2].set_ylabel("ESS / N")
axes[2].set_title("Effective sample size fraction")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("particle_filter_convergence.png", dpi=150)
plt.show()
```

## What to observe

1. **Error decreases with each observation**: after 3–4 observations the position error should drop below 50 km; after 6–8, below 5 km.

2. **ESS stays healthy**: ESS should remain above 30% of N. If it drops lower, your roughening is insufficient.

3. **Uncertainty ellipsoid shrinks asymmetrically**: along-track uncertainty (velocity direction) stays larger than cross-track because RA/Dec measurements are primarily sensitive to angular position. Check the diagonal of the covariance matrix to confirm.

4. **Stress test**: increase `process_noise_std` to 5.0 to inject large unmodeled accelerations. Observe how quickly the filter degrades and whether deprivation handling recovers the track.

5. **Observation gap experiment**: skip observation 4. How much does position error grow during the gap? How quickly does the filter re-converge?
