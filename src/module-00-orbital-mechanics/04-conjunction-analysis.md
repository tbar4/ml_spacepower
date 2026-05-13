# Lesson 4: Conjunction Analysis

**Module:** ML and Game Theory for Space Power — M00: Orbital Mechanics and the SDA Data Ecosystem
**Source:** Foster & Estes (1992), "A Parametric Analysis of Orbital Debris Collision Probability"; Chan (1997), "Spacecraft Collision Probability"; CCSDS 508.0-B-1 Conjunction Data Message standard; Alfano (2005), "A Numerical Implementation of Spherical Object Collision Probability"; Letizia et al. (2018), "Application of a debris index for global evaluation of mitigation strategies"

---


<!-- toc -->

## Where this fits

The previous three lessons gave you the data artifact (TLE), the coordinate systems, and the propagation engine. This lesson puts them together for the most operationally important application in commercial SDA: conjunction analysis. When two cataloged objects will pass close to each other in the future, operators need to quantify the collision risk and decide whether to execute an avoidance maneuver. That decision is what conjunction analysis supports.

This lesson covers the entire chain from initial screening through probability of collision computation and the CDM format that encodes the result. Module 1 Lesson 3 extends this with Monte Carlo Pc computation. Module 7's particle filter application uses many of the same state estimation concepts for tracking maneuvering objects.

## A space scenario to motivate everything

Space-Track issues an automated email at 03:47 UTC: "CONJUNCTION WARNING — PROBABILITY OF COLLISION 1.1e-4 — TCA 2024-10-03 14:32:17 UTC — OBJECT1: 25544 (ISS (ZARYA)) — OBJECT2: 46876 (COSMOS 2499 DEB)." Your client's satellite operations team is asking three questions: Is 1.1e-4 a high Pc? Should we maneuver? And how do we know the CDM is accurate?

This lesson gives you the framework to answer all three.

---

## Conjunction screening

Before you can compute a probability of collision, you need to identify which pairs of objects are close enough to matter. The Space Surveillance Network tracks 50,000+ objects. That is roughly 1.25 billion possible pairs. You cannot compute full Pc for all of them — the computation would be far too slow. Screening narrows the field.

### The Space-Track pizza-box screening volume

The 18th Space Defense Squadron (18 SDS) uses an asymmetric screening volume around each satellite to identify candidate conjunction pairs. This is commonly called the "pizza-box" volume because of its shape:

- **Radial (R)**: ±1 km
- **Along-track (T)**: ±25 km  
- **Cross-track (N)**: ±1 km

This is NOT a sphere. The along-track dimension is 25× larger than the radial and cross-track dimensions. That asymmetry is deliberate.

**Why asymmetric?** Recall from Lesson 2 that position uncertainty in RTN space is dominated by the along-track direction — typical along-track uncertainty is 10–100× the radial uncertainty for LEO objects. If you used a spherical screening volume of radius 1 km, you would miss most true high-risk conjunctions because the objects' predicted close approach might be several kilometers apart in the along-track direction while actually being physically close due to TLE uncertainty. A spherical 25 km volume would work but would generate enormous numbers of false positives. The pizza-box matches the uncertainty geometry: tight radial and cross-track bounds (where uncertainty is small) and a generous along-track window (where uncertainty is large).

Commercial providers use different screening volumes. LeoLabs, for example, uses different dimensions based on their higher-quality OD solutions. When you compare event counts between providers, always check the screening volume.

### Miss distance and closest approach

For each pair that passes within the screening volume, the conjunction assessment pipeline computes the **Time of Closest Approach (TCA)** and **miss distance** — the separation vector and scalar range at TCA.

Miss distance is decomposed into R, T, N components. These components are important for understanding the encounter geometry:
- Large along-track miss distance + small radial miss distance = the objects are passing in nearly parallel orbits, separated mainly vertically — the along-track uncertainty dominates the Pc geometry
- Small miss distance in all three components = high geometric risk regardless of which direction uncertainty is largest

**Relative velocity** at TCA is also reported. For LEO-LEO head-on encounters, relative velocity is typically 10–15 km/s. For co-planar debris passes, it can be as low as 0.1–1 km/s. Higher relative velocity means shorter encounter duration, which simplifies the linear-motion approximation used in Pc calculation.

---

## The conjunction plane and Pc geometry

The probability of collision is computed by projecting the three-dimensional position uncertainty onto a two-dimensional plane.

### Linear relative motion approximation

For typical conjunction geometries, the encounter is brief — objects at 10 km/s relative velocity cross a 10 km screening volume in about 1 second. During this interval, both objects travel in nearly straight lines. The linear relative motion approximation replaces the true curved orbital trajectories with straight-line motion, transforming the 3D problem into a 2D problem.

### The conjunction plane

The **conjunction plane** is the plane perpendicular to the relative velocity vector at TCA. When the linear approximation holds, the probability of collision depends entirely on where the objects' positions, distributed according to the combined position uncertainty, land relative to each other in this 2D plane.

**Why project onto the conjunction plane?** The relative position uncertainty along the relative velocity direction does not affect whether collision occurs — if the objects are traveling toward each other at 10 km/s, a 5 km uncertainty along the velocity direction just means the closest approach might happen 0.5 milliseconds earlier or later. What matters is how spread out the positions are in the two directions perpendicular to the relative velocity — those are the directions in which a miss distance greater than the hard body radius actually prevents collision.

### The combined covariance

Each CDM provides a 3×3 position covariance (in RTN) for each object. The combined covariance for Pc computation is the sum:

\[ \mathbf{C}_\text{combined} = \mathbf{C}_1 + \mathbf{C}_2 \]

This assumes the two objects' position errors are independent — a reasonable approximation for objects in different orbits. After projecting onto the conjunction plane, the combined covariance becomes a 2×2 matrix describing the distribution of the relative position in the plane.

### The hard body radius (HBR)

A collision occurs when the centers of the two objects are within the sum of their physical radii. This combined radius is called the **hard body radius (HBR)**:

\[ R_\text{HBR} = r_1 + r_2 \]

Typical values:
- Active satellites: 2–10 m radius → 4–20 m HBR contribution
- Rocket bodies: 1–3 m radius
- Debris fragments: 0.05–0.5 m radius

If no physical dimensions are known (common for debris), conservative estimates are used. The ISS has an HBR of approximately 50 m due to its large structure and solar arrays.

### The Foster/Chan Pc calculation

The standard Pc method is due to Foster & Estes (1992) and Chan (1997). The calculation integrates the 2D Gaussian probability density function (representing the combined position uncertainty projected onto the conjunction plane) over the disk of radius \(R_\text{HBR}\) centered at the origin:

\[ P_c = \frac{1}{2\pi \sigma_1 \sigma_2} \int\!\!\!\int_{x^2 + y^2 \leq R_\text{HBR}^2} \exp\!\left(-\frac{x^2}{2\sigma_1^2} - \frac{y^2}{2\sigma_2^2}\right) dx\, dy \]

where \(\sigma_1\) and \(\sigma_2\) are the semi-axes of the combined uncertainty ellipse in the conjunction plane (assuming the principal axes are aligned — the general case requires a full 2×2 covariance matrix).

In practice, this integral is computed numerically, often via series expansion. The key insight: Pc depends on two things — how large the combined uncertainty ellipse is relative to the HBR disk, and how the uncertainty is shaped relative to the encounter geometry.

**What determines whether Pc is high or low?**

- Large combined uncertainty relative to HBR: the probability mass is spread over a large area, so the fraction that falls inside the small HBR disk is small → low Pc
- Small combined uncertainty, miss distance comparable to HBR: all the probability mass is concentrated near the HBR boundary → high Pc
- The miss distance in the conjunction plane: if the best-estimate closest approach is zero (head-on collision course), Pc is highest. As miss distance increases, Pc decreases rapidly.

### Covariance realism: why operational Pc values are often too small

TLE-derived covariances are systematically underestimated. The SSN fits TLEs using radar observations with known measurement noise, and the resulting covariance reflects that observation noise. It does not capture force model uncertainty — unmodeled atmospheric density variations, unmodeled maneuvers, OD batch update artifacts, and BSTAR fitting errors all contribute to actual position uncertainty but are not reflected in the fitted covariance.

Studies have found that actual position errors are typically 3–10× larger than TLE-derived covariances suggest. This means operational Pc values from Space-Track (which use these underestimated covariances) often understate the true collision risk.

**For ML applications**: this is a commercially significant problem. ML models that learn to predict "covariance inflation factors" — multipliers that scale the raw TLE-derived covariance to better match actual position errors — are one of the most valuable products in the commercial SDA market. Your ML model is not replacing Pc calculation; it is correcting the covariance input to Pc calculation.

The Pc method field in CDMs (`COLLISION_PROBABILITY_METHOD`) is important precisely because different methods make different covariance assumptions. Alfano (2005) provides a generalized short-encounter Pc method. Monte Carlo Pc (sampling from the combined covariance and counting simulated collisions) is a third approach. **Pc values computed by different methods for the same CDM geometry are not directly comparable.** If your ML pipeline ingests CDMs from multiple sources, you must either normalize to a single method or treat the method as a feature.

---

## The CCSDS CDM format

The Conjunction Data Message (CDM) is the standard format for exchanging conjunction assessment data. It is defined in CCSDS standard 508.0-B-1. Understanding every field is essential for any SDA ML pipeline.

CDMs come in two encoding formats: **KVN (Key-Value Notation)** — a simple text format, one field per line — and **XML**. Space-Track provides CDMs in KVN format via its REST API.

A CDM has a header section, then an OBJECT1 block and OBJECT2 block. OBJECT1 is typically the higher-priority or primary object (the satellite the operator cares about). OBJECT2 is the secondary object (the debris or other satellite).

### Critical CDM fields

**Header section:**
- `CREATION_DATE`: when this CDM was generated. Not the same as TCA. For freshness assessment, you want `CREATION_DATE` relative to `TCA`.
- `ORIGINATOR`: who generated the CDM (e.g., "18 SPACE DEFENSE SQUADRON")
- `MESSAGE_ID`: unique identifier for this specific CDM

**Conjunction geometry fields:**
- `TCA` (Time of Closest Approach): the predicted time of minimum miss distance, in UTC
- `MISS_DISTANCE`: scalar miss distance at TCA, in meters. This is the most human-readable risk indicator but does not account for uncertainty.
- `RELATIVE_SPEED`: relative velocity magnitude at TCA, in m/s
- `RELATIVE_POSITION_R/T/N`: miss distance components in RTN frame, meters
- `COLLISION_PROBABILITY`: the computed Pc value (e.g., `1.1e-4`)
- `COLLISION_PROBABILITY_METHOD`: which algorithm produced the Pc (e.g., `FOSTER-1992`, `CHAN-1997`, `ALFANO-2005`, `MONTE_CARLO`)

**Per-object fields (both OBJECT1 and OBJECT2 blocks):**
- `OBJECT` / `OBJECT_DESIGNATOR`: NORAD catalog ID
- `OBJECT_NAME`: common name
- `OBJECT_TYPE`: `PAYLOAD`, `ROCKET BODY`, `DEBRIS`, `UNKNOWN`, or `OTHER`. Note: the field is `OBJECT_TYPE`, not "object class."
- `MANEUVERABLE`: `YES`, `NO`, or `N/A`. A maneuverable satellite can execute a collision avoidance maneuver. This fundamentally changes the risk interpretation — a `YES` OBJECT1 has lower effective risk than the raw Pc suggests, because the operator can choose to maneuver.
- `X, Y, Z`: Cartesian position in GCRF (J2000 ECI), km, at TCA
- `X_DOT, Y_DOT, Z_DOT`: velocity in GCRF, km/s
- `CR_R, CT_T, CN_N, CRdot_R, CTdot_T, CNdot_N`: diagonal elements of the 6×6 RTN covariance matrix (m²). CDMs use the notation CR_R for the R-R element, CT_T for T-T, etc.
- Off-diagonal terms: `CT_R, CN_R, CN_T` for the lower triangle of the position block

---

## Code: parsing a CDM

```python
"""
Parse a synthetic but realistic CCSDS CDM in KVN format.
Extract key fields, reconstruct the 3x3 position covariance in RTN,
and compute basic statistics.

This is the data structure your ML pipeline will process for every conjunction event.
"""
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# A synthetic but realistic CDM KVN string
CDM_KVN = """CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2024-10-01T08:00:00.000
ORIGINATOR = 18 SPACE DEFENSE SQUADRON
MESSAGE_ID = 2024-274-00123
TCA = 2024-10-03T14:32:17.421
MISS_DISTANCE = 432.5
RELATIVE_SPEED = 14823.4
RELATIVE_POSITION_R = -38.2
RELATIVE_POSITION_T = 421.4
RELATIVE_POSITION_N = -62.1
COLLISION_PROBABILITY = 1.12E-04
COLLISION_PROBABILITY_METHOD = FOSTER-1992
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 25544
OBJECT_NAME = ISS (ZARYA)
INTERNATIONAL_DESIGNATOR = 1998-067A
OBJECT_TYPE = PAYLOAD
MANEUVERABLE = YES
ORBIT_CENTER = EARTH
REF_FRAME = EME2000
GRAVITY_MODEL = EGM-96: 36D 36O
ATMOSPHERIC_MODEL = JACCHIA 70
EPHEMERIS_NAME = NONE
COVARIANCE_METHOD = CALCULATED
MANEUVER_APPLICABLE = YES
X = -2338.512
Y = 5481.234
Z = 3834.721
X_DOT = -5.234812
Y_DOT = -1.823456
Z_DOT = 5.912345
CR_R = 40000.0
CT_R = -8500.0
CT_T = 12500000.0
CN_R = 1200.0
CN_T = -3400000.0
CN_N = 180000.0
CRDOT_R = -2.1
CTDOT_R = 420.0
CTDOT_T = -85000.0
CNDOT_R = 0.8
CNDOT_T = 17200.0
CNDOT_N = -340.0
CRRDOT_DOT = 0.00045
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 46876
OBJECT_NAME = COSMOS 2499 DEB
INTERNATIONAL_DESIGNATOR = 2014-028G
OBJECT_TYPE = DEBRIS
MANEUVERABLE = NO
ORBIT_CENTER = EARTH
REF_FRAME = EME2000
GRAVITY_MODEL = EGM-96: 36D 36O
ATMOSPHERIC_MODEL = JACCHIA 70
EPHEMERIS_NAME = NONE
COVARIANCE_METHOD = CALCULATED
MANEUVER_APPLICABLE = NO
X = -2338.082
Y = 5481.654
Z = 3834.143
X_DOT = -5.227341
Y_DOT = -1.815123
Z_DOT = 5.905678
CR_R = 62500.0
CT_R = -12000.0
CT_T = 18900000.0
CN_R = 1800.0
CN_T = -5100000.0
CN_N = 270000.0
CRDOT_R = -3.1
CTDOT_R = 630.0
CTDOT_T = -127500.0
CNDOT_R = 1.2
CNDOT_T = 25800.0
CNDOT_N = -510.0
CRRDOT_DOT = 0.00067"""


@dataclass
class ObjectBlock:
    """Holds per-object fields from a CDM."""
    designator:   Optional[str]   = None
    name:         Optional[str]   = None
    object_type:  Optional[str]   = None
    maneuverable: Optional[str]   = None
    x_km:         Optional[float] = None
    y_km:         Optional[float] = None
    z_km:         Optional[float] = None
    # Position covariance diagonal (m²)
    cr_r:  Optional[float] = None
    ct_t:  Optional[float] = None
    cn_n:  Optional[float] = None
    # Position covariance off-diagonal (m²)
    ct_r:  Optional[float] = None
    cn_r:  Optional[float] = None
    cn_t:  Optional[float] = None


def parse_cdm_kvn(cdm_text: str) -> dict:
    """
    Parse a CDM in KVN format into a structured dictionary.
    Returns header fields and two ObjectBlock instances.
    """
    header = {}
    objects = {}
    current_obj = None

    for raw_line in cdm_text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("COMMENT"):
            continue

        # Split on first ' = '
        if " = " not in line:
            continue
        key, _, value = line.partition(" = ")
        key   = key.strip()
        value = value.strip()

        # Track which object block we are in
        if key == "OBJECT" and value in ("OBJECT1", "OBJECT2"):
            current_obj = value
            objects[current_obj] = ObjectBlock()
            continue

        if current_obj is None:
            # Still in header section
            header[key] = value
        else:
            # In an object block
            obj = objects[current_obj]
            if   key == "OBJECT_DESIGNATOR": obj.designator   = value
            elif key == "OBJECT_NAME":       obj.name         = value
            elif key == "OBJECT_TYPE":       obj.object_type  = value
            elif key == "MANEUVERABLE":      obj.maneuverable = value
            elif key == "X":                 obj.x_km         = float(value)
            elif key == "Y":                 obj.y_km         = float(value)
            elif key == "Z":                 obj.z_km         = float(value)
            elif key == "CR_R":              obj.cr_r         = float(value)
            elif key == "CT_T":              obj.ct_t         = float(value)
            elif key == "CN_N":              obj.cn_n         = float(value)
            elif key == "CT_R":              obj.ct_r         = float(value)
            elif key == "CN_R":              obj.cn_r         = float(value)
            elif key == "CN_T":              obj.cn_t         = float(value)

    return {"header": header, "objects": objects}


def build_position_covariance_rtn(obj: ObjectBlock) -> np.ndarray:
    """
    Build the 3x3 RTN position covariance matrix from CDM fields.
    Order: [R, T, N] — covariance is in m².

    The CDM provides the lower triangle:
      [CR_R,  CT_R,  CN_R ]
      [CT_R,  CT_T,  CN_T ]
      [CN_R,  CN_T,  CN_N ]
    """
    C = np.array([
        [obj.cr_r,  obj.ct_r,  obj.cn_r],
        [obj.ct_r,  obj.ct_t,  obj.cn_t],
        [obj.cn_r,  obj.cn_t,  obj.cn_n],
    ])
    return C


# --- Parse the CDM ---
parsed = parse_cdm_kvn(CDM_KVN)
hdr    = parsed["header"]
obj1   = parsed["objects"]["OBJECT1"]
obj2   = parsed["objects"]["OBJECT2"]

print("=== CDM Header ===")
for k, v in hdr.items():
    print(f"  {k:<35} = {v}")

print("\n=== Object 1 ===")
print(f"  NORAD ID     : {obj1.designator}")
print(f"  Name         : {obj1.name}")
print(f"  Type         : {obj1.object_type}")
print(f"  Maneuverable : {obj1.maneuverable}")

print("\n=== Object 2 ===")
print(f"  NORAD ID     : {obj2.designator}")
print(f"  Name         : {obj2.name}")
print(f"  Type         : {obj2.object_type}")
print(f"  Maneuverable : {obj2.maneuverable}")

print("\n=== Conjunction Geometry ===")
miss_m  = float(hdr["MISS_DISTANCE"])
rel_r   = float(hdr["RELATIVE_POSITION_R"])
rel_t   = float(hdr["RELATIVE_POSITION_T"])
rel_n   = float(hdr["RELATIVE_POSITION_N"])
pc      = float(hdr["COLLISION_PROBABILITY"])
method  = hdr["COLLISION_PROBABILITY_METHOD"]
print(f"  TCA                : {hdr['TCA']}")
print(f"  Miss distance      : {miss_m:.1f} m")
print(f"  Miss distance R/T/N: {rel_r:.1f} / {rel_t:.1f} / {rel_n:.1f} m")
print(f"  Collision Prob (Pc): {pc:.2e}  [{method}]")
print(f"  Relative speed     : {float(hdr['RELATIVE_SPEED']):.1f} m/s = "
      f"{float(hdr['RELATIVE_SPEED'])/1000:.1f} km/s")

print("\n=== Covariance Analysis (RTN, position block only) ===")
C1 = build_position_covariance_rtn(obj1)
C2 = build_position_covariance_rtn(obj2)
C_combined = C1 + C2

sigma1_r = np.sqrt(C1[0, 0])
sigma1_t = np.sqrt(C1[1, 1])
sigma1_n = np.sqrt(C1[2, 2])

sigma2_r = np.sqrt(C2[0, 0])
sigma2_t = np.sqrt(C2[1, 1])
sigma2_n = np.sqrt(C2[2, 2])

sigma_comb_r = np.sqrt(C_combined[0, 0])
sigma_comb_t = np.sqrt(C_combined[1, 1])
sigma_comb_n = np.sqrt(C_combined[2, 2])

print(f"\n  Object 1 position 1σ (m) — R: {sigma1_r:.1f}  T: {sigma1_t:.1f}  N: {sigma1_n:.1f}")
print(f"  Object 2 position 1σ (m) — R: {sigma2_r:.1f}  T: {sigma2_t:.1f}  N: {sigma2_n:.1f}")
print(f"  Combined 1σ (m)          — R: {sigma_comb_r:.1f}  T: {sigma_comb_t:.1f}  N: {sigma_comb_n:.1f}")
print(f"\n  Along-track / Radial uncertainty ratio (Object 1): "
      f"{sigma1_t / sigma1_r:.0f}×")
print(f"  (Expected: 10–100× for well-tracked LEO objects — T >> R)")

print(f"\n=== Risk Interpretation ===")
print(f"  Pc = {pc:.2e}  (method: {method})")
print(f"  Rough operator thresholds:")
print(f"    Pc > 1e-4 : Red — maneuver likely warranted")
print(f"    1e-5 < Pc <= 1e-4: Yellow — monitor closely, maneuver possible")
print(f"    Pc <= 1e-5 : Green — routine monitoring")
print(f"  This event is at {pc:.1e} — {'RED' if pc > 1e-4 else 'YELLOW' if pc > 1e-5 else 'GREEN'}")
print(f"\n  Object 1 (ISS) maneuverable: {obj1.maneuverable}")
print(f"  → ISS can execute avoidance maneuver if decision is made.")
print(f"  Object 2 (debris) maneuverable: {obj2.maneuverable}")
print(f"  → Debris cannot maneuver — avoidance entirely on ISS.")

print(f"\n  NOTE: Pc computed using {method}.")
print(f"  A CDM from a different provider using MONTE_CARLO would give a")
print(f"  different Pc for the same geometry. Method must be held fixed")
print(f"  for consistent model training and evaluation.")
```

---

## Key Takeaways

- **Space-Track uses an asymmetric pizza-box screening volume (1×25×1 km in RTN), not a sphere.** The along-track dimension is 25× larger because along-track position uncertainty is much larger than radial or cross-track. Matching the screen volume to the uncertainty geometry prevents both false positives and missed events.

- **Pc is computed by projecting the combined RTN covariance onto the conjunction plane and integrating a 2D Gaussian over the hard body disk.** The Foster/Chan method is the standard. Along-track uncertainty often projects along the relative velocity direction, where it has less effect on Pc than the smaller radial and cross-track uncertainties.

- **TLE-derived covariances are systematically underestimated by 3–10×.** Operational Pc values from Space-Track are often too small because the covariance input understates actual position uncertainty. ML covariance inflation models are commercially valuable precisely for this reason.

- **`COLLISION_PROBABILITY_METHOD` determines which method produced the Pc.** Pc values from different methods for the same geometry are not directly comparable. Never compare Pc values across CDMs without checking that the same method was used.

- **The `MANEUVERABLE` flag changes the risk interpretation.** A maneuverable satellite has lower effective collision risk than its raw Pc suggests, because the operator can choose to maneuver if the risk is high enough. Pc alone overstates risk for active satellites with functioning propulsion.

- **`OBJECT_TYPE` is one of: PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, OTHER.** Not "object class." For ML models that use object type as a categorical feature, use these exact values.

---

## Quiz

{{#quiz 04-conjunction-analysis.toml}}
