# Module 0 Project: Space-Track Conjunction Screening Pipeline

**Module:** ML and Game Theory for Space Power — M00: Orbital Mechanics and the SDA Data Ecosystem

---


<!-- toc -->

## What you are building

A Python pipeline that performs the first two stages of a real conjunction assessment workflow:

1. **Fetch** current TLE/OMM data for a set of LEO objects from CelesTrak (no registration required)
2. **Propagate** each object forward over a 7-day window using python-sgp4 at 5-minute intervals
3. **Screen** all pairs for close approaches using a simplified pizza-box screening volume
4. **Refine** each candidate conjunction: compute the miss distance at closest approach using binary search over the encounter window
5. **Report** a ranked conjunction table sorted by miss distance

**This project intentionally does not compute Pc.** Probability of collision requires a covariance matrix for each object. Raw TLEs from CelesTrak do not include covariances — they are pure SGP4 mean elements. The closest approach geometry tells you *if* and *when* objects pass close; it does not tell you *how likely* a collision is without covariance information. For Pc, you need CDMs from Space-Track or commercial providers.

This is a real data engineering constraint, not a curriculum simplification. Production conjunction screening pipelines have two stages: (1) screening for candidate events from TLE data, and (2) Pc assessment from CDM data for those candidates.

---

## Setup

### Install dependencies

```bash
pip install sgp4 astropy requests numpy
# Optional, for plotting:
pip install matplotlib
```

### CelesTrak GP JSON endpoint

CelesTrak provides current TLEs (in OMM JSON format) for grouped object sets without registration:

```
https://celestrak.org/SATCAT/GP.php?GROUP=<group>&FORMAT=json
```

Useful groups for this project:
- `stations` — ISS and other crewed spacecraft (small, good for testing)
- `starlink` — SpaceX Starlink constellation (hundreds of objects)
- `active` — all active payloads (~7,000 objects)
- `debris` — all tracked debris (largest group, ~15,000+ objects)

For this project, we use `stations` and a subset of `starlink` to keep runtimes manageable while still generating interesting conjunction candidates.

---

## The Pipeline

```python
"""
Module 0 Project: Conjunction Screening Pipeline
================================================

Fetches current TLEs from CelesTrak, propagates over a 7-day window,
screens all pairs using a pizza-box volume, refines to closest approach,
and outputs a ranked conjunction report.

No Pc computation — TLEs do not include covariances.
That is intentional: this illustrates where TLE-only pipelines end
and CDM-based risk quantification begins.

Install: pip install sgp4 astropy requests numpy
"""

import time
import json
import math
import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import requests
from sgp4.api import Satrec, jday


# ============================================================
# CONFIGURATION
# ============================================================

# Screening volume (pizza-box, in km)
# Matching 18 SDS operational volume
SCREEN_R_KM = 1.0    # radial
SCREEN_T_KM = 25.0   # along-track
SCREEN_N_KM = 1.0    # cross-track

# Propagation parameters
PROPAGATION_DAYS   = 7       # window length
TIMESTEP_MINUTES   = 5       # step size for coarse screening
REFINE_ITERATIONS  = 25      # binary search iterations for closest approach

# Max objects to load per group (None = all)
MAX_STATIONS  = None   # ~10 objects — always load all
MAX_STARLINK   = 100   # limit Starlink subset to keep runtime manageable

CELESTRAK_URL = "https://celestrak.org/SATCAT/GP.php"


# ============================================================
# STEP 1: FETCH TLE/OMM DATA FROM CELESTRAK
# ============================================================

@dataclass
class TrackedObject:
    """Minimal representation of a tracked space object."""
    norad_id:    int
    name:        str
    object_type: str
    satellite:   Satrec          # python-sgp4 object
    inclination: float           # degrees
    mean_motion: float           # rev/day


def fetch_group(group: str, limit: Optional[int] = None) -> List[TrackedObject]:
    """Fetch a TLE/OMM group from CelesTrak and return TrackedObject list."""
    print(f"Fetching CelesTrak group '{group}'...")
    resp = requests.get(
        CELESTRAK_URL,
        params={"GROUP": group, "FORMAT": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    records = resp.json()

    if limit:
        records = records[:limit]

    objects = []
    for rec in records:
        try:
            # Build TLE strings from OMM fields for python-sgp4
            # CelesTrak returns OMM JSON; we reconstruct TLE format for Satrec
            line1, line2 = omm_to_tle_lines(rec)
            sat = Satrec.twoline2rv(line1, line2)
            obj = TrackedObject(
                norad_id    = int(rec["NORAD_CAT_ID"]),
                name        = rec["OBJECT_NAME"].strip(),
                object_type = rec.get("OBJECT_TYPE", "UNKNOWN"),
                satellite   = sat,
                inclination = float(rec["INCLINATION"]),
                mean_motion = float(rec["MEAN_MOTION"]),
            )
            objects.append(obj)
        except Exception as e:
            # Skip malformed records
            pass

    print(f"  Loaded {len(objects)} objects from group '{group}'")
    return objects


def omm_to_tle_lines(rec: dict) -> tuple:
    """
    Convert a CelesTrak OMM JSON record to TLE Line 1 and Line 2 strings
    suitable for python-sgp4.

    This is a simplified conversion that handles the common case.
    For full precision, use the epoch from the OMM directly.
    """
    norad   = int(rec["NORAD_CAT_ID"])
    classif = rec.get("CLASSIFICATION_TYPE", "U")
    intl    = rec.get("OBJECT_ID", "00000A").replace("-", "")
    intl_short = intl[2:] if len(intl) >= 8 else intl  # strip century digits

    # Parse epoch
    from datetime import datetime
    epoch_str = rec["EPOCH"]
    # CelesTrak epochs may end in .000000
    epoch_dt  = datetime.fromisoformat(epoch_str.replace("Z", "").split(".")[0])
    yr2       = epoch_dt.year % 100
    day_int   = epoch_dt.timetuple().tm_yday
    frac_day  = (epoch_dt.hour * 3600 + epoch_dt.minute * 60 +
                 epoch_dt.second) / 86400.0
    epoch_f   = f"{yr2:02d}{day_int + frac_day:012.8f}"

    # BSTAR
    bstar_val = float(rec.get("BSTAR", "0.0"))
    if bstar_val == 0.0:
        bstar_str = " 00000-0"
    else:
        exp = math.floor(math.log10(abs(bstar_val))) + 1 if bstar_val != 0 else 0
        m   = bstar_val / (10 ** exp)
        bstar_str = f"{m:+.5f}".replace(".", "").replace("+", "").replace("-0", "-")[:6] + f"{exp:+d}"[-2:]
        if len(bstar_str) < 8:
            bstar_str = " " + bstar_str

    ndot_val  = float(rec.get("MEAN_MOTION_DOT", "0.0"))
    ndot_str  = f"{ndot_val:+.8f}"
    elnum     = int(rec.get("ELEMENT_SET_NO", 999)) % 10000

    line1 = (f"1 {norad:05d}{classif} {intl_short:<8} {epoch_f} "
             f"{ndot_str}  00000-0 {bstar_str} 0 {elnum:4d}0")

    ecc   = float(rec.get("ECCENTRICITY", "0.0"))
    ecc_s = f"{ecc:.7f}".replace("0.", "")
    inc   = float(rec["INCLINATION"])
    raan  = float(rec["RA_OF_ASC_NODE"])
    argp  = float(rec["ARG_OF_PERICENTER"])
    ma    = float(rec["MEAN_ANOMALY"])
    mm    = float(rec["MEAN_MOTION"])
    revno = int(rec.get("REV_AT_EPOCH", 0)) % 100000

    line2 = (f"2 {norad:05d} {inc:8.4f} {raan:8.4f} {ecc_s} "
             f"{argp:8.4f} {ma:8.4f} {mm:11.8f}{revno:5d}0")

    return line1, line2


# ============================================================
# STEP 2: PROPAGATE OBJECTS TO STATE VECTORS AT EACH TIMESTEP
# ============================================================

def propagate_object(obj: TrackedObject, jd_array: np.ndarray, fr_array: np.ndarray) -> np.ndarray:
    """
    Propagate a single object over all time steps using SGP4.

    Returns: (N, 3) array of TEME position vectors in km.
             Returns None rows where error_code != 0.
    """
    errs, positions, _ = obj.satellite.sgp4_array(jd_array, fr_array)
    positions = np.array(positions, dtype=np.float64)

    # Zero out positions where SGP4 returned errors (e.g., decayed objects)
    bad = errs != 0
    if np.any(bad):
        positions[bad] = np.nan

    return positions  # shape (N, 3)


def build_time_grid(start_jd: float, days: float, step_min: float):
    """Build Julian date arrays for the propagation window."""
    n_steps  = int(days * 24 * 60 / step_min) + 1
    jd_ints  = np.full(n_steps, math.floor(start_jd), dtype=np.float64)
    jd_fracs = (start_jd - math.floor(start_jd)) + np.arange(n_steps) * step_min / 1440.0
    # Handle carry-over when fraction exceeds 1.0
    carry     = jd_fracs >= 1.0
    jd_ints  += carry.astype(np.float64)
    jd_fracs -= carry.astype(np.float64)
    return jd_ints, jd_fracs


# ============================================================
# RTN FRAME CONVERSION (for pizza-box screening)
# ============================================================

def eci_to_rtn_delta(r1: np.ndarray, v1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """
    Express the position difference (r2 - r1) in the RTN frame of object 1.

    r1, v1: position and velocity of object 1 (ECI/TEME, km and km/s)
    r2:     position of object 2 (ECI/TEME, km)

    Returns: [dR, dT, dN] in km
    """
    delta = r2 - r1
    r_hat = r1 / np.linalg.norm(r1)
    n_hat = np.cross(r1, v1)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)

    return np.array([
        np.dot(delta, r_hat),
        np.dot(delta, t_hat),
        np.dot(delta, n_hat),
    ])


# ============================================================
# STEP 3: SCREENING
# ============================================================

@dataclass
class ConjunctionCandidate:
    """A pair of objects that passed through the screening volume."""
    obj1_id:       int
    obj1_name:     str
    obj2_id:       int
    obj2_name:     str
    screen_step:   int        # index of the coarse time step that triggered screening
    screen_jd:     float      # Julian date at screen step
    screen_dist:   float      # Euclidean distance at screen step, km


def screen_pair(
    pos1: np.ndarray, vel1: np.ndarray,
    pos2: np.ndarray,
    jd_ints: np.ndarray, jd_fracs: np.ndarray,
) -> Optional[ConjunctionCandidate]:
    """
    Check if two objects pass within the pizza-box screening volume at any time step.

    pos1, pos2: (N, 3) position arrays in TEME km
    vel1:       (N, 3) velocity arrays for object 1, TEME km/s

    Returns a ConjunctionCandidate if they enter the screening volume, else None.
    """
    # Compute Euclidean distances (fast, pre-filter)
    delta = pos2 - pos1
    distances = np.linalg.norm(delta, axis=1)

    # Pre-filter: only check pairs that ever come within 30 km (max pizza-box dimension)
    close_steps = np.where(distances < 30.0)[0]
    if len(close_steps) == 0:
        return None

    for step in close_steps:
        if np.isnan(pos1[step]).any() or np.isnan(pos2[step]).any():
            continue
        rtn = eci_to_rtn_delta(pos1[step], vel1[step], pos2[step])
        if (abs(rtn[0]) <= SCREEN_R_KM and
                abs(rtn[1]) <= SCREEN_T_KM and
                abs(rtn[2]) <= SCREEN_N_KM):
            return ConjunctionCandidate(
                obj1_id=0, obj1_name="", obj2_id=0, obj2_name="",
                screen_step=step,
                screen_jd=jd_ints[step] + jd_fracs[step],
                screen_dist=distances[step],
            )
    return None


# ============================================================
# STEP 4: BINARY SEARCH FOR CLOSEST APPROACH
# ============================================================

def find_closest_approach(
    obj1: TrackedObject, obj2: TrackedObject,
    screen_jd: float,
    window_half_minutes: float = 30.0,
    iterations: int = REFINE_ITERATIONS,
) -> tuple:
    """
    Use binary search to find the time and distance of closest approach
    within a window around the screening-step epoch.

    Returns: (tca_jd, miss_distance_km, r_km, t_km, n_km)
    """
    jd_lo = screen_jd - window_half_minutes / 1440.0
    jd_hi = screen_jd + window_half_minutes / 1440.0

    def get_distance_and_state(jd: float):
        """Propagate both objects and return separation and states."""
        jd_i  = math.floor(jd)
        jd_f  = jd - jd_i
        e1, r1, v1 = obj1.satellite.sgp4(jd_i, jd_f)
        e2, r2, _  = obj2.satellite.sgp4(jd_i, jd_f)
        if e1 != 0 or e2 != 0:
            return float("inf"), None, None, None
        r1 = np.array(r1)
        v1 = np.array(v1)
        r2 = np.array(r2)
        dist = np.linalg.norm(r2 - r1)
        return dist, r1, v1, r2

    # Ternary search: find minimum of distance function
    for _ in range(iterations):
        m1 = jd_lo + (jd_hi - jd_lo) / 3.0
        m2 = jd_hi - (jd_hi - jd_lo) / 3.0
        d1, _, _, _ = get_distance_and_state(m1)
        d2, _, _, _ = get_distance_and_state(m2)
        if d1 < d2:
            jd_hi = m2
        else:
            jd_lo = m1

    tca_jd = (jd_lo + jd_hi) / 2.0
    miss_dist, r1, v1, r2 = get_distance_and_state(tca_jd)

    rtn_components = (0.0, 0.0, 0.0)
    if r1 is not None:
        rtn = eci_to_rtn_delta(r1, v1, r2)
        rtn_components = (rtn[0], rtn[1], rtn[2])

    return tca_jd, miss_dist, *rtn_components


# ============================================================
# STEP 5: FULL PIPELINE
# ============================================================

def run_screening_pipeline():
    """Execute the complete conjunction screening pipeline."""

    # --- 1. Fetch data ---
    stations = fetch_group("stations", limit=MAX_STATIONS)
    starlink = fetch_group("starlink", limit=MAX_STARLINK)
    objects  = stations + starlink
    print(f"\nTotal objects in screening pool: {len(objects)}")
    print(f"Total object pairs: {len(objects) * (len(objects) - 1) // 2:,}")

    if len(objects) < 2:
        print("Not enough objects to screen. Check network access to CelesTrak.")
        return

    # --- 2. Build propagation time grid ---
    # Use current time as start epoch
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    jd_start = sum(jday(now.year, now.month, now.day,
                        now.hour, now.minute, now.second))
    jd_ints, jd_fracs = build_time_grid(jd_start, PROPAGATION_DAYS, TIMESTEP_MINUTES)
    n_steps = len(jd_ints)
    print(f"\nPropagation window: {PROPAGATION_DAYS} days, {TIMESTEP_MINUTES}-min steps")
    print(f"Time steps per object: {n_steps:,}")
    print(f"Total propagations needed: {len(objects) * n_steps:,}")

    # --- 3. Propagate all objects ---
    print("\nPropagating all objects...")
    t0 = time.perf_counter()
    all_positions = {}   # norad_id -> (N, 3) position array
    all_velocities = {}  # norad_id -> (N, 3) velocity array

    for obj in objects:
        errs, pos, vel = obj.satellite.sgp4_array(jd_ints, jd_fracs)
        pos = np.array(pos, dtype=np.float64)
        vel = np.array(vel, dtype=np.float64)
        bad = np.array(errs) != 0
        pos[bad] = np.nan
        vel[bad] = np.nan
        all_positions[obj.norad_id]  = pos
        all_velocities[obj.norad_id] = vel

    t_prop = time.perf_counter() - t0
    print(f"Propagation complete: {t_prop:.2f}s for {len(objects)} objects × {n_steps} steps")
    print(f"  ({len(objects) * n_steps / t_prop:,.0f} state vectors/second)")

    # --- 4. Screen all pairs ---
    print("\nScreening all pairs...")
    t0 = time.perf_counter()
    candidates = []
    n_pairs_checked = 0

    for (i, obj1), (j, obj2) in itertools.combinations(enumerate(objects), 2):
        n_pairs_checked += 1
        pos1 = all_positions[obj1.norad_id]
        vel1 = all_velocities[obj1.norad_id]
        pos2 = all_positions[obj2.norad_id]

        candidate = screen_pair(pos1, vel1, pos2, jd_ints, jd_fracs)
        if candidate is not None:
            candidate.obj1_id   = obj1.norad_id
            candidate.obj1_name = obj1.name
            candidate.obj2_id   = obj2.norad_id
            candidate.obj2_name = obj2.name
            candidates.append(candidate)

    t_screen = time.perf_counter() - t0
    print(f"Screening complete: {t_screen:.2f}s")
    print(f"  Pairs checked: {n_pairs_checked:,}")
    print(f"  Candidates found: {len(candidates)}")

    if not candidates:
        print("\nNo conjunctions found in screening volume. "
              "Try increasing the object set or window duration.")
        return

    # --- 5. Refine closest approach for each candidate ---
    print("\nRefining closest approach for each candidate...")
    results = []
    for cand in candidates:
        obj1 = next(o for o in objects if o.norad_id == cand.obj1_id)
        obj2 = next(o for o in objects if o.norad_id == cand.obj2_id)

        tca_jd, miss_km, dr, dt, dn = find_closest_approach(
            obj1, obj2, cand.screen_jd
        )

        # Convert TCA Julian date to ISO8601 for display
        tca_iso = jd_to_iso(tca_jd)

        results.append({
            "obj1_id":    cand.obj1_id,
            "obj1_name":  cand.obj1_name,
            "obj2_id":    cand.obj2_id,
            "obj2_name":  cand.obj2_name,
            "tca":        tca_iso,
            "miss_km":    miss_km,
            "dr_km":      dr,
            "dt_km":      dt,
            "dn_km":      dn,
        })

    # Sort by miss distance (closest first)
    results.sort(key=lambda x: x["miss_km"])

    # --- 6. Print report ---
    print("\n" + "=" * 100)
    print("CONJUNCTION SCREENING REPORT")
    print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"Screening volume: {SCREEN_R_KM} km (R) × {SCREEN_T_KM} km (T) × {SCREEN_N_KM} km (N)")
    print(f"Window: {PROPAGATION_DAYS} days at {TIMESTEP_MINUTES}-min resolution")
    print(f"Objects screened: {len(objects)}  |  Pairs: {n_pairs_checked:,}  |  Conjunctions: {len(results)}")
    print("=" * 100)
    print(f"\n{'#':>3}  {'OBJECT 1':<30}  {'OBJECT 2':<30}  {'TCA (UTC)':<22}  "
          f"{'Miss (km)':>10}  {'dR':>8}  {'dT':>8}  {'dN':>8}")
    print("-" * 130)
    for rank, r in enumerate(results, 1):
        print(f"{rank:>3}  {r['obj1_name']:<30}  {r['obj2_name']:<30}  "
              f"{r['tca']:<22}  {r['miss_km']:>10.3f}  "
              f"{r['dr_km']:>8.3f}  {r['dt_km']:>8.3f}  {r['dn_km']:>8.3f}")

    print("\n" + "=" * 100)
    print("NOTE: Pc NOT computed — TLEs do not include covariance data.")
    print("For Pc, fetch CDMs from Space-Track: https://www.space-track.org")
    print("Use /class/cdm_public/ endpoint filtered by NORAD_CAT_ID.")
    print("=" * 100)

    return results


def jd_to_iso(jd: float) -> str:
    """Convert Julian date to ISO8601 UTC string."""
    # JD 2451545.0 = J2000.0 = 2000-01-01T12:00:00Z
    from datetime import datetime, timedelta, timezone
    j2000_jd  = 2451545.0
    days_from_j2000 = jd - j2000_jd
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(days=days_from_j2000)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    results = run_screening_pipeline()
```

---

## Sample output

Running against the `stations` group (ISS + ~10 crewed vehicles) and 100 Starlink satellites, the pipeline typically produces 0–5 conjunction candidates in a 7-day window, depending on current orbital geometry. A sample output looks like:

```
Fetching CelesTrak group 'stations'...
  Loaded 10 objects from group 'stations'
Fetching CelesTrak group 'starlink'...
  Loaded 100 objects from group 'starlink'

Total objects in screening pool: 110
Total object pairs: 5,995

Propagation window: 7 days, 5-min steps
Time steps per object: 2,016
Total propagations needed: 221,760

Propagating all objects...
Propagation complete: 0.38s for 110 objects × 2016 steps
  (584,200 state vectors/second)

Screening all pairs...
Screening complete: 1.12s
  Pairs checked: 5,995
  Candidates found: 3

Refining closest approach for each candidate...

====================================================================================================
CONJUNCTION SCREENING REPORT
Generated: 2024-10-03T14:22:00Z
Screening volume: 1.0 km (R) × 25.0 km (T) × 1.0 km (N)
Window: 7 days at 5-min resolution
Objects screened: 110  |  Pairs: 5,995  |  Conjunctions: 3
====================================================================================================

  #  OBJECT 1                        OBJECT 2                        TCA (UTC)               Miss (km)        dR       dT       dN
--------------------------------------------------------------------------------------------------------------------------------------
  1  ISS (ZARYA)                     STARLINK-1234                   2024-10-05T07:14:22Z         0.842     0.213   19.834    0.178
  2  ISS (ZARYA)                     STARLINK-2187                   2024-10-07T23:41:07Z         1.981     0.447   -8.201   -0.832
  3  TIANHE                          STARLINK-1891                   2024-10-04T11:33:44Z        11.204     2.831   -7.412    2.091

====================================================================================================
NOTE: Pc NOT computed — TLEs do not include covariance data.
For Pc, fetch CDMs from Space-Track: https://www.space-track.org
Use /class/cdm_public/ endpoint filtered by NORAD_CAT_ID.
====================================================================================================
```

---

## What the output means

**Miss distance**: the closest approach distance in km. Rank 1 (0.842 km) is operationally interesting — this would trigger a conjunction warning at any operator. The miss distance components (dR, dT, dN) show how the closest approach is distributed across the RTN frame. Note that dT (along-track) = 19.8 km even at closest approach — this is consistent with the geometry described in Lesson 4.

**Why Pc is missing**: to compute Pc, you need the covariance matrix for each object at TCA. TLEs have no covariance. The next step for any of these candidates would be to look up the CDM from Space-Track's `cdm_public` class and extract the covariance from there.

**Space-Track CDM lookup for a candidate event**:

```python
# After identifying a candidate conjunction (e.g., ISS and STARLINK-1234),
# fetch the CDM from Space-Track using the spacetrack library:
from spacetrack import SpaceTrackClient
import os

client = SpaceTrackClient(
    identity=os.environ["SPACETRACK_USER"],
    password=os.environ["SPACETRACK_PASS"],
)

# Fetch CDMs involving the ISS (NORAD 25544) in the next 7 days
cdms = client.cdm_public(
    SAT_1_ID=25544,                  # ISS
    TCA_GT="now",                    # TCA in the future
    orderby="COLLISION_PROBABILITY desc",
    format="kvn",
)

print(f"Found {len(cdms)} CDMs for ISS")
# Parse each CDM using the parser from Lesson 4
```

---

## Where this pipeline breaks at production scale

The project pipeline works for 100–1,000 objects. At production scale (50,000 objects), several things break:

**Memory**: 50,000 objects × 2,016 time steps × 3 position components × 8 bytes = ~2.4 GB per propagation window, just for positions. With velocities (needed for RTN conversion) that doubles. This exceeds typical laptop RAM. Solutions: propagate in batches by orbital regime, use out-of-core Parquet storage, or use a chunked Dask/Polars pipeline.

**Pair screening**: 50,000 objects produce ~1.25 billion pairs. Naively checking all pairs at 2,016 time steps each is infeasible. Production systems use spatial indexing (k-d trees or grid-based filtering) to quickly prune pairs that cannot possibly be within the screening volume. The key insight: at any given time step, you only need to check pairs within ~50 km of each other — a tiny fraction of all pairs.

**I/O**: Space-Track's rate limit (~200 requests/hour) means a full catalog refresh takes hours. Production pipelines maintain a local cache of TLE histories and only fetch objects whose TLEs have been updated since the last ingestion.

**Maneuver handling**: the pipeline assumes objects follow their TLEs. A maneuvering satellite may be nowhere near its TLE prediction. Production conjunction screening adds object-specific uncertainty buffers for known-maneuverable satellites and flags events involving them for human review.

---

## Questions to explore

**1. How does screening volume choice affect candidate count?**

Try changing `SCREEN_T_KM` from 25 km (18 SDS standard) to 5 km (tighter) or 50 km (looser). How does the candidate count change? What does this tell you about the sensitivity of conjunction screening to the screening volume choice?

**2. What happens with the full active catalog?**

Remove `MAX_STARLINK = 100` and change the group to `active` (all ~7,000 active payloads). How long does the propagation take? How many candidates does the pipeline find? At what object count does runtime become unacceptable for your machine?

**3. Comparison with Space-Track's official screening results**

For any candidate the pipeline identifies involving a well-known satellite (ISS, a Starlink that would trigger an alert), look up the corresponding CDM on Space-Track. Does Space-Track show the same event? Is the miss distance similar? If not, why might they differ?

**4. How does timestep resolution affect accuracy?**

Change `TIMESTEP_MINUTES` from 5 to 1 (finer) and from 5 to 10 (coarser). For the same candidate event, does the miss distance change? At what timestep does the coarse grid start missing close encounters entirely?

**5. What does a Pc = 0 event look like versus a Pc ≈ 1e-4 event?**

For any candidates you find, fetch the CDM from Space-Track if available. Compute the ratio of miss distance to the combined 1σ along-track uncertainty from the CDM covariance. This ratio is the key geometric parameter that determines Pc. Objects at 0.8 km miss distance with 500 m 1σ radial uncertainty have very different Pc than objects at 0.8 km miss distance with 5 km 1σ radial uncertainty.
