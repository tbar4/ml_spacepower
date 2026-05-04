# Lesson 1: TLEs and Keplerian Elements

**Module:** ML and Game Theory for Space Power — M00: Orbital Mechanics and the SDA Data Ecosystem
**Source:** *Satellite Orbits* — Oliver Montenbruck & Eberhard Gill, Chapter 3; Space-Track.org API documentation; Celestrak GP documentation; Hoots & Roehrich (1980), "Models for Propagation of NORAD Element Sets"

---

## Where this fits

Before any ML model can process space domain data, a data engineer has to answer a fundamental question: what exactly is the input data, and where does it come from? In SDA, the answer starts here — with the Two-Line Element set. TLEs are the most widely used format for tracking the 50,000+ objects in Earth orbit. Every public conjunction screening pipeline, every orbital mechanics simulation, every satellite pass prediction tool starts by reading TLEs. This lesson makes you fluent in the format.

The data engineer's natural approach is to start with the artifact and work backward to the theory. That is exactly what we do here. By the end of this lesson you will be able to parse any TLE programmatically, understand every field's physical meaning, and know exactly where the format's limits are — including the critical warning about what you cannot do with TLE data that is a frequent source of bugs in production pipelines.

## A space scenario to motivate everything

It is early morning at your commercial SDA company. An automated alert fires: a client satellite has a conjunction event scheduled for 14:32 UTC — 11 hours from now. Your pipeline ingested the latest Space-Track CDM at 02:15 UTC. To validate the alert, you open the raw data and see this:

```
ISS (ZARYA)
1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991
2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697
```

This is a TLE. Three lines of ASCII text encode the complete orbital state of the International Space Station. Your ML feature engineering pipeline reads thousands of these per hour. This lesson explains every character.

---

## Here is a real TLE — let us parse it

A TLE has three lines. Line 0 is the satellite name. Lines 1 and 2 contain the orbital data. The format was designed in the 1970s for punched cards, which is why it looks the way it does — fixed-width ASCII fields with no delimiters.

### Line 0: the name

```
ISS (ZARYA)
```

This is simply the satellite's common name. It can be up to 24 characters. The NORAD catalog number on Line 1 is the authoritative identifier — the name is informational.

### Line 1: catalog and epoch data

```
1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991
```

Let us go field by field:

| Field | Value | Meaning |
|-------|-------|---------|
| Line number | `1` | Always 1 for Line 1 |
| NORAD catalog number | `25544` | The unique object ID — this is the key for Space-Track lookups |
| Classification | `U` | U = Unclassified, S = Secret, C = Classified |
| International designator | `98067A` | Launch year (98 = 1998), launch number (067), piece (A = primary payload) |
| Epoch | `24274.50000000` | Year (24 = 2024), day of year and fractional day (274.5 = noon on day 274) |
| First derivative of mean motion (ndot) | `.00015669` | Change in mean motion in rev/day² — often small, sometimes zeroed |
| Second derivative of mean motion (ndotdot) | `00000-0` | Change rate of ndot, in implied decimal notation — almost always zero |
| BSTAR drag term | `27837-3` | Atmospheric drag coefficient, implied decimal: 0.27837 × 10⁻³ |
| Ephemeris type | `0` | Always 0 for publicly distributed TLEs |
| Element set number | `999` | Sequential counter for this object's TLE updates |
| Checksum | `1` | Modulo-10 checksum for error detection |

**About ndot and ndotdot**: These fields are labeled "first and second time derivatives of mean motion" and were used in the legacy SDP4 analytical model. In SGP4/SDP4 as actually implemented, BSTAR and the mean motion directly drive the drag calculation. The ndot and ndotdot fields are present for format compatibility but are typically set to zero or near-zero in modern TLEs distributed by Space-Track. Do not use them as drag signals; use BSTAR.

**Reading BSTAR**: The implied decimal format `27837-3` means `0.27837 × 10⁻³ = 0.00027837`. Higher BSTAR means more atmospheric drag — the object has a large area-to-mass ratio (like a flat panel or solar array) or is at a lower altitude where the atmosphere is denser.

**Reading the epoch**: `24274.50000000` decodes as: year 2024, day 274.5. Day 274 of 2024 is September 30. Day 274.5 is noon on September 30, 2024 UTC. The epoch tells you when this TLE was fitted — it is the reference time from which SGP4 propagates forward or backward.

### Line 2: orbital elements

```
2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697
```

| Field | Value | Meaning |
|-------|-------|---------|
| Line number | `2` | Always 2 for Line 2 |
| NORAD catalog number | `25544` | Same as Line 1 — must match |
| Inclination | `51.6415` | Degrees, 0° to 180° |
| RAAN | `282.4781` | Right Ascension of Ascending Node, degrees |
| Eccentricity | `0001567` | Implied decimal: 0.0001567 (no leading "0.") |
| Argument of perigee | `231.1584` | Degrees |
| Mean anomaly | `128.9321` | Degrees, the linear angle proxy for position in orbit |
| Mean motion | `15.50095566` | Revolutions per day |
| Revolution number | `47269` | Total orbits completed since launch |
| Checksum | `7` | Modulo-10 checksum |

The six orbital elements on Line 2 (inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion) encode the satellite's orbit shape, orientation, and current position. The next section explains each one physically.

---

## The six classical Keplerian elements

Keplerian elements describe an orbit under idealized two-body gravity (Earth as a point mass, no perturbations). TLEs store a modified version of these elements — mean elements averaged over short-period perturbations — but the physical intuition comes from the classical Keplerian picture.

### Semi-major axis (a): orbit size

The semi-major axis is the "radius" of the ellipse — half the longest dimension. For a circular orbit, it equals the actual orbital radius. It is not stored directly in a TLE; instead TLEs store mean motion (revolutions per day), and you derive a from the relationship:

\\[ n = \sqrt{\frac{GM_\oplus}{a^3}} \implies a = \left(\frac{GM_\oplus}{n^2}\right)^{1/3} \\]

where \\(n\\) is mean motion in radians per second and \\(GM_\oplus = 3.986 \times 10^{14}\\) m³/s² is Earth's standard gravitational parameter. For the ISS with mean motion 15.5 rev/day, this gives a ≈ 6,785 km — about 407 km above Earth's surface (Earth radius ≈ 6,378 km).

**Orbit regime intuitions**:
- LEO (Low Earth Orbit): a ≈ 6,550 to 8,375 km (170 to 2,000 km altitude)
- MEO (Medium Earth Orbit): a ≈ 8,375 to 42,165 km (GPS at 20,200 km altitude)
- GEO (Geostationary Earth Orbit): a = 42,164 km exactly (35,786 km altitude, ~24h period)
- HEO (Highly Elliptical Orbit): large a with high eccentricity — Molniya orbits

### Eccentricity (e): orbit shape

Eccentricity describes how elliptical the orbit is. e = 0 is a perfect circle; e approaching 1 is a very elongated ellipse.

- ISS: e = 0.0001567 — nearly perfectly circular
- GPS satellites: e ≈ 0.001 — slightly elliptical but functionally circular
- Molniya orbit: e ≈ 0.74 — highly elliptical, designed to spend most time over high latitudes
- GTO (Geostationary Transfer Orbit): e ≈ 0.73 — elliptical, used to transfer from LEO to GEO

For circular orbits (e ≈ 0), the argument of perigee (ω) becomes poorly defined — there is no perigee to speak of because the orbit has no closest-approach point. This is a common source of numerical issues in algorithms that use ω as a feature for near-circular LEO objects.

### Inclination (i): orbital plane tilt

Inclination is the angle between the orbital plane and Earth's equatorial plane.

- i = 0°: equatorial orbit, satellite always over the equator (GEO)
- i = 51.6°: ISS — chosen to allow Russian cosmonaut launches from Baikonur
- i = 55.0°: GPS Block IIR constellation
- i = 90°: polar orbit, passes over both poles
- i = 97–98°: sun-synchronous orbit, the retrograde drift from J2 is designed to keep the orbital plane aligned with the Sun
- i > 90°: retrograde orbit, satellite moves opposite to Earth's rotation

Inclination directly determines ground coverage. A satellite at 51.6° can never pass over latitudes above 51.6° or below -51.6°. If you are asking "can this object observe [some ground target]?", inclination is the first filter.

### RAAN (Ω): which half of the sky

The Right Ascension of the Ascending Node describes where the orbital plane intersects the equatorial plane, measured from the vernal equinox (a fixed reference direction in inertial space). RAAN tells you which "slice" of the sky the orbit occupies.

For a sun-synchronous orbit at 500 km, 97.4° inclination, J2 causes RAAN to precess eastward at about +0.986°/day — exactly one degree per day, matching Earth's motion around the Sun. This keeps the orbital plane at a fixed angle relative to the Sun, which is why remote-sensing satellites use it (consistent illumination geometry).

For the ISS at 51.6° inclination and 400 km altitude, J2 causes RAAN to precess westward at approximately -6.75°/day. Over one week, RAAN shifts about 47°. This is a predictable, secular drift — it is not a maneuver. **Before using RAAN as a maneuver-detection feature, you must subtract the J2-predicted drift.** Otherwise you will flag every non-maneuvering LEO object as maneuvering.

The J2 RAAN precession rate formula:

\\[ \dot{\Omega} = -\frac{3}{2} n J_2 \left(\frac{R_\oplus}{a}\right)^2 \frac{\cos i}{(1 - e^2)^2} \\]

where \\(J_2 = 1.08263 \times 10^{-3}\\) is Earth's second zonal harmonic coefficient, \\(R_\oplus\\) is Earth's mean equatorial radius, and \\(n\\) is mean motion. For the ISS, this evaluates to approximately -6.75°/day.

### Argument of perigee (ω): which end is closest

For an elliptical orbit, one end is closer to Earth (perigee) and one is farther (apogee). The argument of perigee describes which direction perigee points, measured within the orbital plane from the ascending node.

For near-circular LEO orbits (e ≈ 0), ω is numerically unstable and physically meaningless — the "perigee" can be anywhere around the orbit because the orbit is nearly circular. SGP4 handles this gracefully internally, but be cautious using ω as a raw feature for LEO objects.

For Molniya or HEO orbits, ω is critical and well-defined. Molniya orbits are designed with ω = 270° so that apogee (the slow, high part) is over the northern hemisphere.

### Mean anomaly (M): where in the orbit right now

Mean anomaly is a linear angle proxy for the satellite's current position in its orbit. It increases uniformly from 0° to 360° over one orbital period, reaching 360° at the same time the satellite completes one revolution. At M = 0°, the satellite is at perigee (for an elliptical orbit).

**Important distinction**: TLEs store mean anomaly (M), not true anomaly (ν). These are different:
- **Mean anomaly (M)**: a mathematical construct that increases uniformly. It does not equal the actual angular position.
- **True anomaly (ν)**: the actual geometric angle of the satellite from perigee. For a circular orbit, M = ν. For an elliptical orbit, ν varies nonlinearly — the satellite moves faster near perigee and slower near apogee.

SGP4 internally converts mean anomaly to true anomaly (via the eccentric anomaly) when computing the position vector. You do not have to do this conversion yourself — just know that when you see M in a TLE, it is the uniform linear angle, not the geometric angle.

---

## Mean elements vs. osculating elements

This distinction is the source of a critical production bug that appears repeatedly in SDA pipelines.

**Osculating elements** are the instantaneous Keplerian elements that match the satellite's exact position and velocity at a specific moment, accounting for all perturbations at that instant. They change continuously as perturbations (J2, drag, third-body gravity) act on the orbit.

**Mean elements** are computed by averaging out short-period perturbations. TLEs store mean elements defined specifically for use with the SGP4 propagation algorithm. The mean element values have no meaning outside of SGP4 — they are SGP4 input parameters, not physical observables.

### The critical warning: never difference consecutive TLE Keplerian elements

Suppose you download a TLE history for a satellite and compute the difference in RAAN between consecutive TLEs, hoping to detect maneuvers as anomalous jumps. This approach has two fundamental problems:

1. **Secular J2 drift**: RAAN precesses at -6.75°/day for the ISS. If you difference two TLEs 3 days apart, you get a ~20° RAAN change from J2 alone. You have to subtract the predicted J2 drift — and that requires knowing the exact epoch difference and the J2 formula for this specific orbit.

2. **Mean elements are SGP4-internal**: Even after correcting for J2 drift, the mean element values from consecutive TLEs reflect SGP4's averaging conventions. Differencing them mixes secular drift, periodic perturbation modeling artifacts, and observation data batch update artifacts in ways that are not cleanly separable from maneuver signals.

**The correct approach**: propagate both TLEs to a common epoch using SGP4 and compare the resulting Cartesian position and velocity vectors. A maneuver appears as a sudden change in the propagated trajectory between two TLE epochs that exceeds what J2, drag, and other perturbations would predict. This is what the ML maneuver detection models in Module 9 do.

Additionally: TLE mean elements are not interchangeable with mean elements from other propagators. The \\(GM\\) value embedded in TLE mean motion (\\(GM = 398600.4418\\) km³/s²) differs slightly from the IAU standard. High-fidelity numerical propagators use a different \\(GM\\). Mixing them without conversion introduces systematic errors.

---

## TLE freshness and accuracy

TLE accuracy degrades with time from the epoch, but not smoothly or predictably.

**Epoch age** is the time since the TLE was fitted to radar observations. A 1-day-old TLE is generally more accurate than a 7-day-old TLE — but this is a weak proxy for uncertainty. The actual accuracy depends on:

1. **Unmodeled maneuvers**: if a satellite maneuvered after the TLE epoch, the old TLE's prediction can be wrong by tens to hundreds of kilometers, regardless of how fresh it is. A 12-hour-old TLE for a satellite that maneuvered 6 hours ago is essentially useless for collision avoidance.

2. **Atmospheric drag variability**: the BSTAR term encodes a mean drag coefficient. During geomagnetic storms (elevated Kp index) or periods of high solar activity (elevated F10.7 flux), the upper atmosphere expands and drag forces increase dramatically. LEO objects can deviate tens of kilometers from their predicted positions within hours of a major geomagnetic storm.

3. **Observation data quality**: TLEs are fitted to radar observations from the Space Surveillance Network. The accuracy of the fitted TLE depends on how many observations were available, their geometric diversity, and the fitting algorithm.

**A practical rule of thumb**: a TLE epoch older than 3 days for a maneuvering satellite should be treated with extreme skepticism. For passive debris with stable drag, a 7-day-old TLE might be good to a few kilometers. For a GEO object where solar radiation pressure is the dominant perturbation and is poorly modeled, even a 1-day-old TLE can be several kilometers off.

For ML purposes: **epoch age is a feature, but covariance from CDMs is a much stronger uncertainty signal.** Build your models to prefer covariance-based uncertainty quantification when available.

---

## The OMM format: TLE data in JSON

Space-Track's API now returns orbital data as Orbit Mean-elements Messages (OMM) in JSON or XML format. OMM is defined in the CCSDS standard (CCSDS 502.0-B-2). The data content is identical to a TLE — same SGP4 mean elements — but the format is structured, machine-readable, and extensible.

Key point: **OMM is not higher-fidelity than TLE**. It is the same data in a better format. You still use SGP4 to propagate it; the OMM fields map directly to TLE fields.

A sample Space-Track OMM JSON response:

```json
{
  "CCSDS_OMM_VERS": "2.0",
  "COMMENT": "GENERATED VIA SPACE-TRACK.ORG API",
  "CREATION_DATE": "2024-10-01T06:00:00",
  "ORIGINATOR": "18 SPACE DEFENSE SQUADRON",
  "OBJECT_NAME": "ISS (ZARYA)",
  "OBJECT_ID": "1998-067A",
  "CENTER_NAME": "EARTH",
  "REF_FRAME": "TEME",
  "TIME_SYSTEM": "UTC",
  "MEAN_ELEMENT_THEORY": "SGP4",
  "EPOCH": "2024-10-01T00:00:00.000000",
  "MEAN_MOTION": "15.50095566",
  "ECCENTRICITY": "0.0001567",
  "INCLINATION": "51.6415",
  "RA_OF_ASC_NODE": "282.4781",
  "ARG_OF_PERICENTER": "231.1584",
  "MEAN_ANOMALY": "128.9321",
  "EPHEMERIS_TYPE": "0",
  "CLASSIFICATION_TYPE": "U",
  "NORAD_CAT_ID": "25544",
  "ELEMENT_SET_NO": "999",
  "REV_AT_EPOCH": "47269",
  "BSTAR": "0.00027837",
  "MEAN_MOTION_DOT": "0.00015669",
  "MEAN_MOTION_DDOT": "0.0"
}
```

The field names are self-documenting. Note that `REF_FRAME` is listed as `TEME` — this is correct and important. SGP4 outputs are in TEME, not J2000 ECI. Lesson 2 covers reference frame conversion.

---

## Code

### Parsing a TLE with python-sgp4

```python
"""
Parse a real ISS TLE and inspect all fields using python-sgp4.

Install: pip install sgp4
"""
from sgp4.api import Satrec, jday
from datetime import datetime, timezone
import math

# Real ISS TLE (NORAD 25544) — Line 0, Line 1, Line 2
TLE_LINE0 = "ISS (ZARYA)"
TLE_LINE1 = "1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991"
TLE_LINE2 = "2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697"

# Parse the TLE into an sgp4 satellite record
satellite = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

# --- Inspect all SGP4-accessible fields ---
print("=== TLE Metadata ===")
print(f"NORAD catalog number : {satellite.satnum}")
print(f"International desig  : {satellite.intldesg}")
print(f"Classification       : {satellite.classification}")
print(f"Element set number   : {satellite.elnum}")
print(f"Revolution number    : {satellite.revnum}")

print("\n=== Epoch ===")
print(f"Epoch year (2-digit) : {satellite.epochyr}")
print(f"Epoch day of year    : {satellite.epochdays:.8f}")
# Reconstruct full epoch as a Python datetime for human readability
epoch_year = 2000 + satellite.epochyr if satellite.epochyr < 57 else 1900 + satellite.epochyr
epoch_day  = satellite.epochdays
epoch_int  = int(epoch_day)
epoch_frac = epoch_day - epoch_int
epoch_dt   = datetime(epoch_year, 1, 1, tzinfo=timezone.utc)
from datetime import timedelta
epoch_dt  += timedelta(days=epoch_int - 1, seconds=epoch_frac * 86400)
print(f"Epoch (UTC)          : {epoch_dt.isoformat()}")

print("\n=== Keplerian Mean Elements (SGP4 internal) ===")
# Mean motion is stored internally in radians per minute by sgp4
# Convert to revolutions per day for TLE convention
rev_per_day = satellite.no_kozai * (1440.0 / (2 * math.pi))
print(f"Mean motion (rev/day): {rev_per_day:.8f}")
print(f"Inclination (rad)    : {satellite.inclo:.6f}  ({math.degrees(satellite.inclo):.4f}°)")
print(f"RAAN (rad)           : {satellite.nodeo:.6f}  ({math.degrees(satellite.nodeo):.4f}°)")
print(f"Eccentricity         : {satellite.ecco:.7f}")
print(f"Arg of perigee (rad) : {satellite.argpo:.6f}  ({math.degrees(satellite.argpo):.4f}°)")
print(f"Mean anomaly (rad)   : {satellite.mo:.6f}  ({math.degrees(satellite.mo):.4f}°)")

print("\n=== Drag / Force Model ===")
print(f"BSTAR drag term      : {satellite.bstar:.8f}")
print(f"ndot (ndot/2 stored) : {satellite.ndot:.8f}  (rev/day²)")
print(f"ndotdot              : {satellite.nddot:.8f}  (rev/day³)")

print("\n=== Derived: Semi-major Axis ===")
# GM for SGP4 (XKMPER and XKE constants baked into sgp4 library)
# Use the TLE mean motion to derive semi-major axis
# n in rad/s, GM in km^3/s^2
GM_km3s2 = 398600.4418  # km³/s²
n_rad_s  = satellite.no_kozai / 60.0  # convert from rad/min to rad/s
a_km     = (GM_km3s2 / n_rad_s**2) ** (1.0 / 3.0)
alt_km   = a_km - 6378.137  # subtract Earth equatorial radius
print(f"Semi-major axis      : {a_km:.1f} km")
print(f"Approximate altitude : {alt_km:.1f} km")

# --- Propagate to epoch to get TEME position/velocity ---
print("\n=== Propagate to Epoch (TEME frame) ===")
# Get Julian date for the TLE epoch
jd_epoch, fr_epoch = satellite.jdsatepoch, satellite.jdsatepochF
e, r, v = satellite.sgp4(jd_epoch, fr_epoch)
print(f"Error code           : {e}  (0 = success)")
print(f"TEME position (km)   : x={r[0]:.3f}, y={r[1]:.3f}, z={r[2]:.3f}")
print(f"TEME velocity (km/s) : x={v[0]:.6f}, y={v[1]:.6f}, z={v[2]:.6f}")
orbital_speed = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
print(f"Orbital speed (km/s) : {orbital_speed:.3f}")
```

### Parsing an OMM JSON from Space-Track API

```python
"""
Ingest an OMM JSON response from Space-Track and convert to an sgp4 satellite record.

This demonstrates:
1. How to interpret OMM field names
2. How to build a TLE string from OMM fields for use with python-sgp4
3. Schema validation for a production ingestion pipeline

Install: pip install sgp4
"""
import json
import math
from sgp4.api import Satrec

# Sample OMM JSON — this is the format Space-Track's GP endpoint returns
# Endpoint: https://www.space-track.org/basicspacedata/query/class/gp/
#           NORAD_CAT_ID/25544/format/json
OMM_JSON = """
{
  "CCSDS_OMM_VERS": "2.0",
  "OBJECT_NAME": "ISS (ZARYA)",
  "OBJECT_ID": "1998-067A",
  "CENTER_NAME": "EARTH",
  "REF_FRAME": "TEME",
  "TIME_SYSTEM": "UTC",
  "MEAN_ELEMENT_THEORY": "SGP4",
  "EPOCH": "2024-10-01T12:00:00.000000",
  "MEAN_MOTION": "15.50095566",
  "ECCENTRICITY": "0.0001567",
  "INCLINATION": "51.6415",
  "RA_OF_ASC_NODE": "282.4781",
  "ARG_OF_PERICENTER": "231.1584",
  "MEAN_ANOMALY": "128.9321",
  "EPHEMERIS_TYPE": "0",
  "CLASSIFICATION_TYPE": "U",
  "NORAD_CAT_ID": "25544",
  "ELEMENT_SET_NO": "999",
  "REV_AT_EPOCH": "47269",
  "BSTAR": "0.00027837",
  "MEAN_MOTION_DOT": "0.00015669",
  "MEAN_MOTION_DDOT": "0.0"
}
"""

omm = json.loads(OMM_JSON)

# --- Print every field with explanation ---
FIELD_DESCRIPTIONS = {
    "CCSDS_OMM_VERS":      "CCSDS standard version",
    "OBJECT_NAME":         "Common name (informational)",
    "OBJECT_ID":           "International designator (YYYY-NNNP)",
    "CENTER_NAME":         "Central body for the orbit",
    "REF_FRAME":           "Reference frame for state vector output (TEME for SGP4)",
    "TIME_SYSTEM":         "Time system for epoch",
    "MEAN_ELEMENT_THEORY": "Propagator theory — must be SGP4 to use python-sgp4",
    "EPOCH":               "Reference epoch for the mean elements (UTC)",
    "MEAN_MOTION":         "Revolutions per day (TLE Line 2 column 53-63)",
    "ECCENTRICITY":        "Dimensionless, 0=circular, 1=parabolic escape",
    "INCLINATION":         "Orbital plane tilt from equator, degrees",
    "RA_OF_ASC_NODE":      "RAAN: right ascension of ascending node, degrees",
    "ARG_OF_PERICENTER":   "Argument of perigee, degrees (ill-defined if e≈0)",
    "MEAN_ANOMALY":        "Linear angle proxy for position, degrees (not true anomaly)",
    "EPHEMERIS_TYPE":      "Always 0 for public TLEs",
    "CLASSIFICATION_TYPE": "U=unclassified, S=secret, C=classified",
    "NORAD_CAT_ID":        "Primary key for Space-Track lookups",
    "ELEMENT_SET_NO":      "Sequential TLE revision counter for this object",
    "REV_AT_EPOCH":        "Total revolutions completed since launch at epoch",
    "BSTAR":               "SGP4 drag coefficient (incorporates B*CD*A/2m)",
    "MEAN_MOTION_DOT":     "First derivative of mean motion, rev/day² (usually small)",
    "MEAN_MOTION_DDOT":    "Second derivative of mean motion, rev/day³ (usually 0)",
}

print("=== OMM Field Inventory ===")
for field, value in omm.items():
    desc = FIELD_DESCRIPTIONS.get(field, "")
    print(f"  {field:<25} = {str(value):<20}  # {desc}")

# --- Build TLE strings from OMM for use with python-sgp4 ---
# python-sgp4 can also construct from OMM directly via Satrec.twoline2rv,
# but building explicit TLE strings is useful for logging and debugging.

def build_tle_from_omm(omm: dict) -> tuple[str, str]:
    """
    Reconstruct TLE Line 1 and Line 2 strings from an OMM JSON dict.
    Note: this produces a TLE-formatted string; the underlying data is identical.
    """
    norad    = int(omm["NORAD_CAT_ID"])
    intl     = omm["OBJECT_ID"].replace("-", "")[:8]  # e.g. 1998067A -> 98067A
    intl_fmt = intl[2:] if len(intl) > 6 else intl    # strip century from year

    # Parse epoch
    from datetime import datetime
    epoch_dt = datetime.fromisoformat(omm["EPOCH"].replace("Z", ""))
    year2    = epoch_dt.year % 100
    day_of_year = epoch_dt.timetuple().tm_yday
    frac_day    = (epoch_dt.hour * 3600 + epoch_dt.minute * 60 +
                   epoch_dt.second + epoch_dt.microsecond / 1e6) / 86400.0
    epoch_str   = f"{year2:02d}{day_of_year + frac_day:012.8f}"

    # Format drag fields (implied decimal notation for TLE)
    def fmt_implied(val: float, width: int = 8) -> str:
        """Format a float in TLE implied-decimal notation (e.g. 0.27837e-3 -> 27837-3)."""
        if val == 0.0:
            return " 00000-0"
        import math
        exp = math.floor(math.log10(abs(val))) + 1
        mantissa = val / (10 ** exp)
        return f"{mantissa:+.5f}".replace("0.", "").replace(".", "") + f"{exp:+02d}"[-2:]

    ndot_str   = f"{float(omm['MEAN_MOTION_DOT']):+.8f}"
    ndotdot_str = " 00000-0"  # almost always zero
    bstar_str   = fmt_implied(float(omm["BSTAR"]))
    classif     = omm.get("CLASSIFICATION_TYPE", "U")

    line1 = (f"1 {norad:05d}{classif} {intl_fmt:<8} {epoch_str} "
             f"{ndot_str} {ndotdot_str} {bstar_str} 0 {int(omm['ELEMENT_SET_NO']):4d}0")

    ecc_str  = f"{float(omm['ECCENTRICITY']):.7f}".replace("0.", "")
    line2 = (f"2 {norad:05d} "
             f"{float(omm['INCLINATION']):8.4f} "
             f"{float(omm['RA_OF_ASC_NODE']):8.4f} "
             f"{ecc_str} "
             f"{float(omm['ARG_OF_PERICENTER']):8.4f} "
             f"{float(omm['MEAN_ANOMALY']):8.4f} "
             f"{float(omm['MEAN_MOTION']):11.8f}"
             f"{int(omm['REV_AT_EPOCH']):5d}0")

    return line1, line2

line1, line2 = build_tle_from_omm(omm)
print(f"\n=== Reconstructed TLE ===")
print(f"Line 1: {line1}")
print(f"Line 2: {line2}")

# Verify by parsing with sgp4
sat = Satrec.twoline2rv(line1, line2)
print(f"\nParsed NORAD ID : {sat.satnum}")
print(f"Inclination     : {math.degrees(sat.inclo):.4f}°")
print(f"BSTAR           : {sat.bstar:.8f}")
```

---

## Key Takeaways

- **The TLE is a fixed-width ASCII format encoding six SGP4 mean orbital elements plus metadata.** Every field has a specific physical meaning. NORAD ID is the primary key for Space-Track lookups. Epoch encodes the reference time. BSTAR is the drag proxy. ndot and ndotdot are legacy fields, usually zeroed.

- **Mean anomaly (M) is a linear angle proxy, not the true geometric angle.** SGP4 converts M to a position vector internally. Never confuse M with true anomaly ν.

- **Mean elements are SGP4-internal quantities, not physical observables.** You cannot subtract consecutive TLE Keplerian elements to detect maneuvers. The mean element values are defined only in the context of SGP4. To detect maneuvers, propagate both TLEs to a common epoch and compare Cartesian positions.

- **RAAN precesses due to J2 at a predictable rate (~-6.75°/day for ISS orbit).** This is secular drift, not a maneuver. Subtract the J2-predicted drift before using RAAN as a feature.

- **Epoch age is a weak proxy for uncertainty.** Accuracy depends on unmodeled maneuvers and atmospheric density variability, not just how old the TLE is. A 6-hour-old TLE for a satellite that maneuvered 3 hours ago is useless for conjunction avoidance.

- **OMM is TLE in JSON format, not higher-fidelity data.** Space-Track's API returns OMM by default. Parse it with the same SGP4 library; the physics is identical.

---

## Quiz

{{#quiz 01-tles-and-keplerian-elements.toml}}
