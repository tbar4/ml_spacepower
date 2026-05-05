# Lesson 3: Multi-Object Tracking and Fleet-Level Anomaly Scoring

**Module:** Applied SDA ML — M09: Building Commercial SDA Products
**Source:** Bar-Shalom et al. (2011) "Tracking and Data Fusion"; Blackman & Popoli (1999) "Design and Analysis of Modern Tracking Systems"; Mahler (2014) "Advances in Statistical Multisource-Multitarget Information Fusion"; Hall & Llinas (2001) "Handbook of Multisensor Data Fusion"; Vo & Ma (2006) "The Gaussian Mixture Probability Hypothesis Density Filter"

---

## Where this fits

Lessons 1 and 2 built single-object detectors: given one satellite's 30-day TLE window, output a maneuver probability. This is the right unit of analysis for building the model. It is the wrong unit of analysis for an operator who needs to watch a catalog.

An operator monitoring 200 objects does not care about individual window scores in isolation. They care about which objects in their catalog are behaving anomalously relative to their own history, relative to objects with similar orbits, and relative to each other. A single object executing a series of small maneuvers may not trigger any individual window's threshold — but the pattern across 10 consecutive windows, all with elevated scores, is highly anomalous. And two objects in the same orbital shell both executing correlated maneuvers on the same day is a different kind of anomaly entirely: it may indicate a coordinated approach campaign.

This lesson extends the single-object detector to a fleet-level anomaly scoring system. The new components are: a Bayesian state estimator for each tracked object (the connection to Module 7's particle filters), a data association step that connects TLE measurements to object identities, personalized anomaly baselines, and cross-catalog correlation detection.

---

## The multi-object tracking problem

Tracking a single object over time is a filtering problem: maintain a belief distribution over the object's state (orbital elements + uncertainty), update it with each new measurement (TLE epoch), and propagate it forward between measurements using orbital dynamics. Module 7 introduced particle filters as a general solution to this problem for non-Gaussian state distributions.

Tracking multiple objects simultaneously introduces a new problem: **data association**. When you receive a batch of TLE measurements, you need to decide which measurement corresponds to which object. For well-separated orbital objects this is usually unambiguous — the Space-Track catalog assigns each object a permanent NORAD catalog number that appears in every TLE. But data association is not trivial in three situations:

1. **Fragmentation events**: A debris cloud from a satellite breakup produces many new objects without established catalog entries. The initial measurements must be associated to newly created tracks.
2. **Close-approach maneuvers**: When two objects approach within a few kilometers, their TLE uncertainties may overlap, making the correct association ambiguous.
3. **Catalog errors**: Space-Track occasionally misidentifies TLEs, assigning a measurement to the wrong catalog entry. These errors inject outlier observations into tracking filters for the correct object.

For the commercial SDA use case, situation 3 is the most common and the most likely to corrupt anomaly scores. The tracking filter must be robust to occasional misassociated TLEs.

---

## Bayesian state estimation per object

For each tracked object, maintain a Gaussian belief over the orbital state vector:

```
state = [a, e, i, Ω, ω, M]  (semi-major axis, eccentricity, inclination,
                               RAAN, argument of perigee, mean anomaly)
```

The belief at time t is parameterized by a mean vector and covariance: `(μ_t, Σ_t)`.

**Predict step**: Propagate the mean forward using SGP4 dynamics for the time interval `Δt`:

```python
import numpy as np
from sgp4.api import Satrec

def predict_state(mu_t, sigma_t, delta_t_seconds, tle_epoch):
    """Propagate Gaussian belief forward using linearized SGP4 dynamics."""
    # Propagate mean
    mu_pred = sgp4_propagate(mu_t, delta_t_seconds)

    # Propagate covariance using a simple process noise model
    # Q encodes the accumulation of unmodeled forces (atmospheric drag variance,
    # solar radiation pressure, third-body perturbations) over delta_t
    Q = process_noise_matrix(delta_t_seconds, altitude_km=semi_major_to_altitude(mu_t[0]))
    F = linearize_sgp4(mu_t, delta_t_seconds)  # Jacobian of propagation
    sigma_pred = F @ sigma_t @ F.T + Q

    return mu_pred, sigma_pred
```

**Update step**: When a new TLE measurement `z_t` arrives, apply the Kalman update:

```python
def update_state(mu_pred, sigma_pred, z_t, R):
    """Bayesian update from new TLE measurement."""
    # Measurement model: TLE ≈ true orbital state + noise
    # R encodes TLE fit error covariance (varies by object quality)
    innovation = z_t - mu_pred
    S = sigma_pred + R
    K = sigma_pred @ np.linalg.inv(S)   # Kalman gain
    mu_updated = mu_pred + K @ innovation
    sigma_updated = (np.eye(len(mu_pred)) - K) @ sigma_pred
    return mu_updated, sigma_updated
```

The **innovation vector** `z_t - mu_pred` is the key anomaly signal: how much did the new TLE depart from the predicted orbital state? A quiet satellite in station-keeping should produce small innovations consistent with TLE fit error. A satellite that executed a maneuver since the last observation will produce a large innovation, particularly in mean motion (in-plane burn) or inclination/RAAN (out-of-plane burn).

The **innovation Mahalanobis distance** is the per-TLE anomaly score:

```python
def mahalanobis_score(innovation, S):
    return float(innovation @ np.linalg.inv(S) @ innovation)
```

An innovation consistent with the covariance `S` has a chi-squared distribution with 6 degrees of freedom under the null hypothesis of no maneuver. The 99.9th percentile threshold is approximately 22.5. An innovation with Mahalanobis distance above 22.5 rejects the no-maneuver null at 99.9% confidence.

---

## Personalized anomaly baselines

The Mahalanobis threshold above assumes you have an accurate model of the object's dynamics and TLE noise. In practice, TLE fit quality varies enormously: well-tracked GEO satellites may have TLE errors of tens of meters, while tumbling debris at 400 km altitude during solar maximum may have TLE errors of several kilometers. A single global threshold produces unacceptable false alarm rates for noisy objects and excessive miss rates for well-tracked objects.

The solution is a **personalized baseline**: for each object, maintain a rolling distribution of innovation magnitudes over the past 90 days. The threshold for each object is set at the 99.9th percentile of its own historical innovation distribution.

```python
class ObjectAnomaly:
    def __init__(self, norad_id: int, history_days: int = 90):
        self.norad_id = norad_id
        self.innovation_history = []  # rolling Mahalanobis scores
        self.history_days = history_days

    def update(self, score: float, epoch: float):
        self.innovation_history.append((epoch, score))
        # Trim to history window
        cutoff = epoch - self.history_days
        self.innovation_history = [(t, s) for t, s in self.innovation_history if t >= cutoff]

    def threshold(self, percentile: float = 99.9) -> float:
        if len(self.innovation_history) < 30:
            return 22.5  # fall back to chi-squared default
        scores = [s for _, s in self.innovation_history]
        return float(np.percentile(scores, percentile))

    def is_anomalous(self, score: float) -> bool:
        return score > self.threshold()
```

This approach automatically adapts to each object's noise characteristics. A debris object with noisy TLEs develops a high threshold; a well-tracked GEO satellite develops a low threshold. Both use the same parametric form; only the calibration differs.

---

## Fleet-level pattern detection

Single-object Mahalanobis scoring flags individual anomalous TLEs. The more operationally interesting cases are patterns:

**Sustained anomaly**: An object that has elevated (but sub-threshold) innovation scores for 7 consecutive days is exhibiting a different pattern than one with a single spike. Compute a **sustained anomaly score** as the sum of normalized innovation scores over a sliding window:

```python
def sustained_anomaly_score(innovation_scores, window_days=7, baseline_mean=1.0):
    """CUSUM-style sustained anomaly detector."""
    normalized = [s / baseline_mean for s in innovation_scores[-window_days:]]
    return sum(max(0, n - 1.5) for n in normalized)  # accumulate above-average days
```

This is a CUSUM (cumulative sum) control chart — a classic sequential anomaly detection method that is sensitive to sustained small deviations rather than single large spikes.

**Correlated maneuvers across catalog**: If two objects in close orbital proximity both execute maneuvers within the same 24-hour window, this is a qualitatively different event than two independent maneuvers. Compute a correlation matrix over the catalog at each daily step:

```python
def catalog_correlation_matrix(anomaly_scores_today, objects, proximity_km=100):
    """Flag pairs of objects with correlated elevated scores and close approach geometry."""
    n = len(objects)
    alerts = []
    for i in range(n):
        for j in range(i+1, n):
            if (anomaly_scores_today[i] > 5.0 and anomaly_scores_today[j] > 5.0
                    and orbital_proximity(objects[i], objects[j]) < proximity_km):
                alerts.append((objects[i].norad_id, objects[j].norad_id,
                                orbital_proximity(objects[i], objects[j])))
    return alerts
```

In a catalog of 200 watched objects, O(n^2) pairwise proximity checks are computationally trivial (~20,000 operations per daily update). For the full public catalog of 25,000 objects, an O(n^2) loop is too slow; approximate nearest-neighbor search in orbital element space (using KD-tree or similar) reduces this to O(n log n).

---

## Catalog segmentation

Not all objects should be treated as interchangeable. The watched catalog should be segmented before anomaly scoring:

| Segment | Criteria | Baseline method |
|---|---|---|
| Active LEO satellites | Altitude 200–2000 km, known active status | Personalized Mahalanobis |
| GEO satellites | Altitude ~35,786 km, near-circular | Personalized + class-conditional |
| Rocket bodies | High BSTAR, known source | Separate model (SRP-dominated) |
| Debris | No known active status | Personalized Mahalanobis, lower alerting priority |
| Unknown/suspect | Uncorrelated tracks, unusual orbital characteristics | Elevated monitoring priority |

The "unknown/suspect" category is the highest-value target for an SDA product. Objects in unusual orbital regimes that are not in any commercial satellite registry are exactly the objects worth watching most closely.

---

## Connecting to particle filters (Module 7)

The Gaussian tracking filter above assumes the state uncertainty is approximately Gaussian — true when no maneuver has occurred recently and the TLE measurement noise is roughly symmetric. After a maneuver, the state uncertainty is non-Gaussian: you know a burn occurred and you can bound the delta-V range, but the posterior over the new state may be multimodal.

Module 7's particle filter handles this correctly. Replace the Kalman update with a particle filter:

1. Represent the belief over orbital state as a set of N weighted particles: `{(x_i, w_i)}`.
2. At each propagation step, advance each particle forward using SGP4.
3. At each measurement step, weight each particle by its likelihood under the TLE measurement model.
4. Resample to prevent weight collapse.

After a maneuver detection event, you can inject additional particles spanning the plausible post-maneuver state space — encoding uncertainty about the magnitude and direction of the burn — and let subsequent TLE measurements progressively concentrate the belief distribution around the actual post-maneuver orbit. This is the correct Bayesian treatment of maneuver detection under uncertainty.

---

## Key Takeaways

- **Fleet-level monitoring is a different problem than single-object detection.** Single-object scores treat each window in isolation; fleet-level scoring adds personalized baselines, sustained anomaly patterns, and cross-catalog correlation — the three layers that turn a model into a product.
- **The Mahalanobis innovation score is the per-TLE anomaly signal.** The prediction error (new TLE minus predicted state) normalized by the prediction covariance follows a chi-squared distribution under the no-maneuver null. Deviations above the 99.9th percentile threshold (approximately 22.5 for 6-dimensional state) reject the null.
- **Personalized thresholds calibrate to each object's noise level.** A global threshold produces unacceptable false alarm rates for noisy debris objects and missed detections for well-tracked clean satellites. Maintain a 90-day rolling innovation distribution per object and threshold at the 99.9th percentile of each object's own history.
- **CUSUM scoring detects sustained sub-threshold anomalies.** A series of slightly-elevated innovation scores that individually never cross threshold may collectively indicate a long-duration maneuver campaign. Accumulate normalized scores over a 7-day sliding window to detect this pattern.
- **Correlated maneuvers across proximate objects are high-priority events.** Two objects in the same orbital shell both executing maneuvers on the same day warrants qualitatively different alerting than two independent anomalies — it may indicate a coordinated proximity operations campaign.
- **Particle filters are the correct posterior representation after maneuver detection.** When a maneuver has been detected, the orbital state belief is non-Gaussian: the post-maneuver orbit is uncertain but bounded. Use the Module 7 particle filter to represent this multimodal uncertainty and concentrate the belief as subsequent TLEs arrive.

---

## Quiz

{{#quiz 03-multi-object-tracking.toml}}
