# Lesson 3: SGP4 Propagation

**Module:** ML and Game Theory for Space Power — M00: Orbital Mechanics and the SDA Data Ecosystem
**Source:** Hoots & Roehrich (1980), "Models for Propagation of NORAD Element Sets"; Vallado et al. (2006), "Revisiting Spacetrack Report #3"; *Satellite Orbits* — Montenbruck & Gill, Chapter 3; python-sgp4 library documentation

---


<!-- toc -->

## Where this fits

Lessons 1 and 2 taught you what a TLE contains and what reference frame SGP4 uses. This lesson covers the propagator itself — what SGP4 actually computes, what physics it includes, and critically, what it gets wrong and why. Understanding SGP4's accuracy model is prerequisite to building any ML system that uses TLE-derived features: every ground truth label you compute, every feature residual you engineer, and every uncertainty estimate you use derives from propagation. If you do not know when SGP4 is reliable and when it is not, you cannot reason about your training data quality.

## A space scenario to motivate everything

Your maneuver detection model flags a Starlink satellite as having potentially maneuvered based on a divergence between two consecutive TLE propagations. Before you alert an operator, you need to answer three questions: Is the divergence within normal SGP4 uncertainty for this object? Could this be a geomagnetic storm artifact rather than a real maneuver? And how do you know whether the divergence is big enough to be operationally significant?

Those questions require knowing SGP4's accuracy envelope. That is what this lesson provides.

---

## What SGP4 is — and what it is not

SGP4 (Simplified General Perturbations model 4) is a **semi-analytical propagator** designed specifically to work with TLE mean elements. "Semi-analytical" means it is not a numerical integrator — it does not step forward in time using differential equations. Instead, it is a closed-form analytical solution that computes position and velocity directly at any specified epoch without stepping through intermediate states.

This distinction matters for your pipeline architecture:
- **Numerical integrators** (like Runge-Kutta methods used in high-fidelity propagators) compute state at time \\(t + \Delta t\\) from state at time \\(t\\). They are more accurate but slower — typically microseconds to milliseconds per state evaluation, and they require careful tuning of step sizes.
- **SGP4** computes the state at any time \\(t\\) from the TLE epoch in a single pass. The computation is extremely fast — microseconds per evaluation — making catalog-scale propagation of 50,000+ objects computationally trivial.

This speed advantage is why SGP4 dominates public SSA applications. The entire catalog can be propagated to any epoch in seconds on a single CPU core.

### Physical perturbations SGP4 includes

SGP4 models these orbital perturbations:

**Gravitational harmonics**: J2, J3, J4, and J6 zonal harmonics — corrections to Earth's gravitational field due to Earth's equatorial bulge (oblateness) and higher-order mass distribution asymmetries. J2 is by far the largest and dominates the secular RAAN drift and argument-of-perigee precession described in Lesson 1.

**Atmospheric drag**: modeled via the BSTAR drag coefficient combined with a simplified exponential atmospheric density model. This is a single averaged drag coefficient per TLE — not a time-varying drag model. Changes in atmospheric density due to solar activity between TLE updates are not captured.

**Resonance effects**: for objects near geosynchronous altitude, gravitational resonances between Earth's tesseral harmonics and the satellite's orbital period produce non-trivial perturbations. SGP4 includes special handling for deep-resonance orbits.

### What SGP4 does NOT include for standard LEO/MEO/GEO objects

**Lunar and solar third-body gravity**: for standard LEO and MEO orbits, the gravitational attraction of the Moon and Sun is small relative to Earth's oblateness effects. SGP4 does not include these perturbations. This is why SGP4 is adequate for LEO/MEO but would be inadequate for highly eccentric orbits with long periods.

For objects with orbital periods greater than approximately 225 minutes (roughly above a semi-major axis of ~40,000 km — high MEO, GEO, and HEO objects), the deep-space version **SDP4** is activated automatically by the sgp4 library. SDP4 adds lunar and solar gravity perturbations. When you call `satellite.sgp4()` in python-sgp4, the library automatically selects SGP4 or SDP4 based on the mean motion. You do not need to choose.

**Solar radiation pressure (SRP)**: not included in either SGP4 or SDP4. For high area-to-mass objects in GEO (like defunct spacecraft with large solar arrays), SRP is a significant perturbation. GEO debris with unknown attitude and area-to-mass ratios can have large along-track errors even with fresh TLEs, for this reason.

**High-fidelity atmospheric density**: the SGP4 density model is a simple exponential based on a historical average. Real atmospheric density varies significantly with solar activity (F10.7 flux and Kp index). This is the dominant error source for LEO objects during active solar periods.

---

## Accuracy characterization: be honest with yourself

SGP4 accuracy is often described with a hand-wavy "1 km per day" rule. That rule is wrong in both directions — too optimistic in some cases and too pessimistic in others. Here is an honest accuracy characterization:

### Quiet LEO debris with fresh TLEs

For passively decaying debris objects with TLEs less than 24 hours old, during periods of low solar activity:
- **Radial error**: ~100–500 m (1σ)
- **Cross-track error**: ~100–500 m (1σ)
- **Along-track error**: 1–3 km (1σ)

These are the best-case numbers. The along-track error is larger because along-track position errors accumulate from unmodeled perturbations and from the TLE fitting process.

### Error growth is not smooth or linear

SGP4 error does not grow at a constant rate. It is dominated by:

1. **Unmodeled maneuvers**: if a satellite executed a maneuver after the TLE epoch, the old TLE is fundamentally wrong. A 12-hour-old TLE for a satellite that maneuvered 6 hours ago may be 50–100+ km off. Maneuver age, not epoch age, determines accuracy.

2. **Atmospheric density variations**: during geomagnetic storms (Kp ≥ 5), the thermosphere swells significantly. Drag-sensitive LEO objects (high BSTAR, low altitude) can deviate tens of kilometers from their predicted positions within hours. Epoch age of the TLE is nearly irrelevant during a major geomagnetic storm if the BSTAR was fitted during quiet conditions.

3. **Batch update artifacts**: when the Space Surveillance Network re-fits a TLE from new radar observations, the new TLE may be slightly inconsistent with the previous one due to the OD fitting process. This creates apparent discontinuities in element histories that are OD artifacts, not real maneuvers.

### GEO objects

GEO objects have different accuracy limiters. J2 and drag are small at GEO altitude. The dominant unmodeled perturbation is solar radiation pressure, which is driven by the object's area-to-mass ratio and reflectivity — both unknown for debris. GEO debris can have large covariances even with fresh TLEs.

### The practical ML implication

**Epoch age is a weak signal for uncertainty.** Build your models with this mental model:

- High epoch age → elevated uncertainty, but only a loose bound
- CDM covariance size → much stronger signal, directly encoding the OD solution's uncertainty estimate
- Recent geomagnetic storm (high Kp) → dramatically elevated uncertainty for drag-sensitive LEO objects, regardless of epoch age
- Object in MANEUVERABLE category → any TLE older than the last known maneuver time is suspect

If you are building uncertainty-aware conjunction risk models, you want CDM covariances as primary inputs and epoch age as one of many context features, not the primary uncertainty signal.

---

## Using python-sgp4

The canonical SGP4 implementation for Python is the `sgp4` library by Brandon Rhodes. It is a direct port of the Vallado et al. (2006) reference implementation and is the de facto standard for production SDA pipelines.

**For production and catalog-scale work**: use python-sgp4 directly. It is fast (microseconds per call with the Fortran-accelerated backend), well-tested, and the output is well-understood.

**For single-object analysis and frame conversion**: Astropy wraps python-sgp4 through its `EarthSatellite` class. This is convenient for one-off analysis but adds substantial per-call overhead — Astropy's coordinate machinery does a lot of work per evaluation. For propagating 50,000 objects over multi-day windows at 5-minute intervals, use python-sgp4 directly. For converting one object's state vector from TEME to GCRS, use Astropy.

### Single-object propagation

```python
"""
Propagate the ISS using python-sgp4 and convert the TEME output to GCRS.

Demonstrates:
- python-sgp4 API for single-object propagation
- Astropy TEME-to-GCRS conversion
- Error code interpretation

Install: pip install sgp4 astropy
"""
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
import numpy as np

TLE_LINE1 = "1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991"
TLE_LINE2 = "2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697"

sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

# Propagate 6 hours after TLE epoch
t = Time("2024-10-01T18:00:00", format="isot", scale="utc")
jd, fr = jday(t.datetime.year, t.datetime.month, t.datetime.day,
               t.datetime.hour, t.datetime.minute, float(t.datetime.second))

error_code, r_km, v_kms = sat.sgp4(jd, fr)

# SGP4 error codes:
#   0 = success
#   1 = mean eccentricity < 0 or > 1 (bad TLE)
#   2 = mean motion < 0 (object below ground)
#   3 = pert eccentricity < 0 or > 1
#   4 = semi-latus rectum < 0
#   5 = epoch elements sub-orbital
#   6 = satellite has decayed (perigee below Earth surface)
if error_code != 0:
    raise RuntimeError(f"SGP4 propagation failed with error code {error_code}")

print(f"TEME position (km) : {r_km}")
print(f"TEME velocity (km/s): {v_kms}")

# Convert to GCRS
r_cart = CartesianRepresentation(np.array(r_km) * u.km)
v_cart = CartesianDifferential(np.array(v_kms) * u.km / u.s)
teme_coord = TEME(r_cart.with_differentials(v_cart), obstime=t)
gcrs_coord = teme_coord.transform_to(GCRS(obstime=t))

print(f"\nGCRS position (km) : "
      f"x={gcrs_coord.cartesian.x.to(u.km).value:.3f}, "
      f"y={gcrs_coord.cartesian.y.to(u.km).value:.3f}, "
      f"z={gcrs_coord.cartesian.z.to(u.km).value:.3f}")
```

---

## Batch propagation: ground track over one orbit

```python
"""
Propagate the ISS forward over one complete orbit (~92 minutes) at 10-minute steps.
Plot the ground track as geodetic latitude vs. longitude.

This demonstrates:
- Vectorized batch propagation with python-sgp4
- TEME -> GCRS -> ITRS chain for lat/lon conversion
- The ground track's relationship to inclination

Install: pip install sgp4 astropy numpy matplotlib
"""
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
import numpy as np

TLE_LINE1 = "1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991"
TLE_LINE2 = "2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697"

sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

# Generate time array: TLE epoch + 0 to 92 minutes in 10-minute steps
epoch_str = "2024-10-01T12:00:00"
t0        = Time(epoch_str, format="isot", scale="utc")
dt_min    = np.arange(0, 93, 10)          # 0, 10, 20, ..., 90 minutes
times     = t0 + dt_min * 60 * u.s        # Astropy Time array

# --- Batch propagation using python-sgp4 ---
# Build arrays of Julian dates for each time step
jd_arr = np.array([
    jday(t.datetime.year, t.datetime.month, t.datetime.day,
         t.datetime.hour, t.datetime.minute, float(t.datetime.second))
    for t in times
])
jd_ints  = jd_arr[:, 0]
jd_fracs = jd_arr[:, 1]

# Vectorized SGP4 call (returns arrays of error codes, positions, velocities)
e_arr, r_arr, v_arr = sat.sgp4_array(jd_ints, jd_fracs)

# Check all error codes are 0
if np.any(e_arr != 0):
    bad = np.where(e_arr != 0)[0]
    print(f"Warning: {len(bad)} propagation errors at indices {bad}")

r_arr = np.array(r_arr)   # shape (N, 3), TEME km
v_arr = np.array(v_arr)   # shape (N, 3), TEME km/s

# --- Convert to geodetic lat/lon via TEME -> GCRS -> ITRS ---
lats  = []
lons  = []
alts  = []

for i, t in enumerate(times):
    r_cart    = CartesianRepresentation(r_arr[i] * u.km)
    v_cart    = CartesianDifferential(v_arr[i] * u.km / u.s)
    teme_c    = TEME(r_cart.with_differentials(v_cart), obstime=t)
    gcrs_c    = teme_c.transform_to(GCRS(obstime=t))
    itrs_c    = gcrs_c.transform_to(ITRS(obstime=t))
    geodetic  = itrs_c.earth_location
    lats.append(geodetic.lat.deg)
    lons.append(geodetic.lon.deg)
    alts.append(geodetic.height.to(u.km).value)

lats = np.array(lats)
lons = np.array(lons)
alts = np.array(alts)

print("=== ISS Ground Track (one orbit at 10-minute intervals) ===")
print(f"{'Time (min)':>10}  {'Lat (°)':>8}  {'Lon (°)':>9}  {'Alt (km)':>9}")
print("-" * 45)
for i, dt in enumerate(dt_min):
    print(f"{dt:>10.0f}  {lats[i]:>8.2f}  {lons[i]:>9.2f}  {alts[i]:>9.1f}")

print(f"\nLatitude range: {lats.min():.1f}° to {lats.max():.1f}°")
print(f"(Should be within ±51.6° — the ISS inclination)")

# Optional: plot with matplotlib
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.plot(lons, lats, "b.-", markersize=8)
    for i, dt in enumerate(dt_min):
        plt.annotate(f"{int(dt)}m", (lons[i], lats[i]),
                     fontsize=7, textcoords="offset points", xytext=(3, 3))
    plt.axhline(51.6, color="r", linestyle="--", alpha=0.5, label="Max latitude (inclination)")
    plt.axhline(-51.6, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title("ISS Ground Track — One Orbit (~92 min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("iss_ground_track.png", dpi=100)
    plt.show()
    print("Ground track saved to iss_ground_track.png")
except ImportError:
    print("matplotlib not installed; skipping plot")
```

### Performance comparison: python-sgp4 vs. Astropy loop

```python
"""
Performance benchmark: python-sgp4 vectorized vs. Astropy EarthSatellite per-call.

This shows why you should use python-sgp4 directly for catalog-scale propagation.
"""
import time
import numpy as np
from sgp4.api import Satrec, jday

TLE_LINE1 = "1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991"
TLE_LINE2 = "2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697"

sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

# Generate 1000 time points (e.g., 5-minute intervals over ~3.5 days)
from astropy.time import Time
import astropy.units as u
t0    = Time("2024-10-01T12:00:00", format="isot", scale="utc")
times = t0 + np.arange(1000) * 5 * 60 * u.s

jd_arr = np.array([
    jday(t.datetime.year, t.datetime.month, t.datetime.day,
         t.datetime.hour, t.datetime.minute, float(t.datetime.second))
    for t in times
])
jd_ints  = jd_arr[:, 0]
jd_fracs = jd_arr[:, 1]

# --- Benchmark 1: python-sgp4 vectorized array call ---
n_trials = 20
start = time.perf_counter()
for _ in range(n_trials):
    e, r, v = sat.sgp4_array(jd_ints, jd_fracs)
elapsed_sgp4 = (time.perf_counter() - start) / n_trials

# --- Benchmark 2: Astropy EarthSatellite per-call ---
try:
    from astropy.coordinates import EarthSatellite
    tle_sat = EarthSatellite(TLE_LINE1, TLE_LINE2, "ISS")

    start = time.perf_counter()
    # Astropy supports batch mode via Time arrays — test that
    positions_astropy = tle_sat.at(times)
    elapsed_astropy = time.perf_counter() - start
    astropy_available = True
except Exception as ex:
    elapsed_astropy = None
    astropy_available = False
    print(f"Astropy batch mode failed: {ex}")

print("=== Performance Comparison: 1,000 time points ===")
print(f"python-sgp4 vectorized : {elapsed_sgp4 * 1000:.2f} ms per run")
print(f"  Per point             : {elapsed_sgp4 / 1000 * 1e6:.2f} μs")
if astropy_available:
    print(f"Astropy batch          : {elapsed_astropy * 1000:.2f} ms per run")
    print(f"  Per point             : {elapsed_astropy / 1000 * 1e6:.2f} μs")
    print(f"Speedup factor        : {elapsed_astropy / elapsed_sgp4:.0f}×")

print()
print("At 50,000 objects × 2016 points (7 days × 5 min intervals):")
total_points = 50_000 * 2016
t_sgp4_total = elapsed_sgp4 / 1000 * total_points
print(f"  python-sgp4 total time: {t_sgp4_total:.1f} seconds")
print(f"  (Astropy would be {elapsed_astropy / elapsed_sgp4:.0f}× slower if same ratio holds)")
```

---

## What a maneuver looks like in TLE history

When a satellite executes a maneuver, the Space Surveillance Network (SSN) re-acquires the object with radar, collects new observations, and fits updated TLEs to the new trajectory. In the resulting TLE history, a maneuver leaves characteristic signatures:

**Mean motion discontinuity**: the most reliable maneuver indicator. A burn that raises or lowers the orbit changes the orbital energy and thus changes mean motion. A 10 m/s radial burn on a 400 km LEO object produces a change in semi-major axis of roughly 20 km, which corresponds to a mean motion change of about 0.005 rev/day.

**Tracking gap**: after a maneuver, the SSN must re-acquire the object — it is not where predicted. There is often a gap in TLE history (no new TLE for 12–48 hours) followed by a new TLE with an updated element set number. The gap is itself a signal.

**Correlated changes in RAAN and argument of perigee residuals**: a maneuver with an out-of-plane component changes the inclination and RAAN. After subtracting the J2-predicted drift, an anomalous change in the corrected RAAN residual may indicate an out-of-plane maneuver.

### Why element jumps alone are not sufficient evidence

This is a critical domain concept for ML model development. Mean motion changes that look like maneuvers can also be caused by:

1. **Atmospheric density spikes from solar activity**: a geomagnetic storm (Kp ≥ 5, elevated F10.7 solar flux) increases thermospheric density substantially, increasing drag on all LEO objects simultaneously. Every passive debris object at similar altitudes shows correlated mean motion decreases during a major geomagnetic storm. If you flag these as maneuvers, your false positive rate will spike every solar event.

2. **OD batch update artifacts**: the SSN uses a batch OD process. When a new observation batch is processed, the fitted TLE may jump slightly due to the fitting algorithm, especially if the observation coverage was sparse or asymmetric. These artifacts can produce apparent element jumps that are not physically real.

3. **Area-to-mass ratio changes** (rare but real): for objects with flexible structures (like tangled debris), actual changes in the effective drag cross-section can change the observed mean motion without a propulsive maneuver.

**The correct ML approach**: a maneuver detection model should take as input not just element residuals but contextual features including geomagnetic indices (F10.7 solar flux, Kp index at the time of the change), whether similar changes appear in other objects at similar altitudes (correlated changes suggest environmental, not propulsive, cause), whether the object is in the MANEUVERABLE or PAYLOAD category, and the availability of a tracking gap before the new TLE.

This is precisely the feature engineering problem that sequence models (RNNs, Transformers) address well — and the motivation for that later curriculum lesson.

---

## Key Takeaways

- **SGP4 is a semi-analytical propagator, not a numerical integrator.** It computes position analytically from TLE mean elements, making it fast enough for catalog-scale propagation. The entire 50,000+ object catalog can be propagated in seconds with python-sgp4.

- **SGP4 includes J2–J6 harmonics and atmospheric drag (via BSTAR).** For objects with periods less than ~225 minutes (standard LEO/MEO/GEO), it does NOT include lunar and solar third-body gravity. SDP4, activated automatically for deep-space orbits (period > ~225 min), does include third-body effects.

- **SGP4 accuracy is NOT "1 km per day."** For fresh TLEs of quiet LEO debris, along-track error is 1–3 km. For maneuvering satellites or during geomagnetic storms, errors can be 10–100+ km regardless of TLE age. Epoch age is a weak uncertainty proxy.

- **Use python-sgp4 directly for catalog-scale propagation.** Astropy wraps python-sgp4 but adds overhead per call that makes it impractical for bulk propagation. Use Astropy for frame conversion after propagating with python-sgp4.

- **Maneuver detection from TLE history requires corroborating evidence.** A mean motion jump in one object is not sufficient. Check geomagnetic indices, look for correlated changes in nearby objects, check for tracking gaps, and consider the object's maneuverable status.

---

## Quiz

{{#quiz 03-sgp4-propagation.toml}}
