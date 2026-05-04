# Lesson 2: Reference Frames

**Module:** ML and Game Theory for Space Power — M00: Orbital Mechanics and the SDA Data Ecosystem
**Source:** *Satellite Orbits* — Oliver Montenbruck & Eberhard Gill, Chapter 2; IERS Conventions (2010), SOFA Library documentation; Vallado, *Fundamentals of Astrodynamics and Applications*, Chapter 3

---

## Where this fits

In Lesson 1 you parsed a TLE and propagated the ISS to a position vector using SGP4. The output was three numbers — something like `(−2338.5, 5481.2, 3834.7)` km. But a position vector means nothing without knowing what those axes point toward. This lesson answers that question.

Reference frames matter in practical SDA work for two reasons. First, you cannot meaningfully subtract two position vectors unless they are expressed in the same frame. If your ML pipeline ingests telescope observations and TLE-derived positions without converting them to a common frame, the residuals are garbage — and the model will quietly learn to predict garbage. Second, the covariance matrices in CDMs are expressed in a specific non-inertial frame (RTN), and interpreting them correctly requires understanding what the axes represent.

## A space scenario to motivate everything

Your pipeline ingests two data streams for the same conjunction event:

1. A SGP4 propagation of each object's TLE, producing position vectors at the TCA
2. Telescope observations from a commercial optical network, providing astrometric positions at the same time

You want to compute the residual — how far off is the TLE-derived position from the telescope observation? You subtract one from the other and get a 400 km discrepancy. Is the TLE wrong? Did you make an error? Are the objects actually 400 km apart?

The answer is almost certainly that you forgot to convert from TEME to J2000. The SGP4 output is in TEME. The telescope observation is in J2000/GCRF. Subtracting them directly is physically meaningless. This specific bug is one of the most common errors in SSA software, and it takes less than a lesson to prevent it permanently.

---

## Why coordinate frames matter

A Cartesian position vector \\((x, y, z)\\) implicitly assumes three things:

1. An **origin** — the point (0, 0, 0) is "here"
2. An **orientation** — the x, y, z axes point in specific directions
3. A **scale** — what units are used (km, m, etc.)

All the orbital mechanics frames we care about share Earth's center as origin and use kilometers. The differences are entirely in axis orientation — which directions do x, y, and z point?

The practical consequence: if you have a position vector \\(\mathbf{r}_1\\) in frame A and a position vector \\(\mathbf{r}_2\\) in frame B, the difference \\(\mathbf{r}_1 - \mathbf{r}_2\\) tells you nothing about the actual spatial separation between the objects unless A and B are the same frame. Converting between frames requires a rotation matrix (and sometimes an additional correction for Earth's rotation).

---

## ECI: Earth-Centered Inertial (J2000 / GCRF)

**Earth-Centered Inertial (ECI)** frames have Earth's center as the origin and do not rotate with the Earth. The axes are fixed to the (approximately) inertial reference of distant stars, not to Earth's surface.

The specific ECI frame used in modern SDA work is the **GCRF (Geocentric Celestial Reference Frame)**, which is the practical realization of the ICRS (International Celestial Reference System) for geocentric calculations. For most SDA purposes, it is sufficient to call this "J2000 ECI" — the axes are defined at the J2000.0 epoch (January 1.5, 2000, i.e., noon on January 1, 2000 UTC).

**Axis definitions:**
- **X axis**: toward the mean vernal equinox of J2000.0 (the direction in the sky where the Sun crosses the equatorial plane moving north, averaged over short-period nutation, as defined at J2000.0)
- **Z axis**: toward the mean celestial north pole of J2000.0 — this is defined by Earth's mean rotation axis at J2000.0, not by geographic north. They are very close but not identical; the difference is the nutation and precession between J2000.0 and the current date.
- **Y axis**: completes the right-handed system (Y = Z × X)

**What ECI is used for**: orbital mechanics calculations. When you integrate equations of motion, compute orbital periods, or describe orbital geometry, you use ECI. Orbital elements (inclination, RAAN, argument of perigee) are defined with respect to ECI axes.

**What ECI is not used for**: ground station locations or geographic coordinates. The Earth rotates under the ECI frame, so a fixed ground station's position in ECI changes continuously — it traces out a circle. For anything geographic, use ECEF.

---

## ECEF: Earth-Centered Earth-Fixed

**Earth-Centered Earth-Fixed (ECEF)** has the same origin as ECI — Earth's center — but the axes rotate with Earth. A fixed point on Earth's surface has a constant position in ECEF.

**Axis definitions:**
- **X axis**: toward the prime meridian (0° longitude) at the equator
- **Z axis**: toward the geographic north pole (Earth's mean rotation axis, the same as the Z axis for the ITRF)
- **Y axis**: completes the right-handed system (90° east longitude at the equator)

**What ECEF is used for:**
- Ground station locations (ground station latitude/longitude/altitude converts directly to ECEF XYZ)
- Visibility calculations (which ground stations can see this satellite right now?)
- Geodetic coordinates (WGS84 latitude, longitude, altitude)

**Converting ECI to ECEF**: the transformation requires knowing the Earth's rotation angle at the specific epoch. The key quantity is the **Greenwich Mean Sidereal Time (GMST)** — the angle between the prime meridian and the vernal equinox direction. GMST rotates at approximately 360°/86164 seconds (one sidereal day). The conversion is a single rotation around the Z axis by the GMST angle:

\\[ \mathbf{r}_\text{ECEF} = R_z(\theta_\text{GMST}) \cdot \mathbf{r}_\text{ECI} \\]

where \\(R_z(\theta)\\) is a rotation matrix around the Z axis. For more precise conversions, polar motion corrections and Earth rotation irregularities are included (IERS corrections), but for SDA applications the GMST rotation is usually sufficient.

---

## TEME: the actual SGP4 output frame

This is the frame you need to know about before touching any SGP4 output.

**TEME (True Equator Mean Equinox)** is the reference frame in which SGP4 produces its position and velocity vectors. It is not the same as J2000 ECI, and using TEME output as if it were J2000 is the most common reference frame bug in SSA pipelines.

**What TEME is**: TEME uses:
- The **true equator of date**: the instantaneous equatorial plane, accounting for nutation (the periodic wobble of Earth's rotation axis due to lunar and solar torques). "Of date" means it is computed for the specific epoch of the propagation, not frozen at J2000.0.
- The **mean equinox of date**: the vernal equinox direction corrected for precession (the long-term drift of Earth's rotation axis through the sky) but not for nutation. This is a hybrid — it applies only part of the full nutation correction.

**Why TEME exists**: TEME is an artifact of the SGP4 algorithm's historical development. The original Hoots & Roehrich (1980) implementation was designed to work with TLE observations processed in a specific way by the Space Surveillance Network, and TEME was the frame in which those observations were reduced. SGP4 outputs TEME because that is how it was built, not because TEME is physically convenient.

**The magnitude of the TEME-to-J2000 error**: at LEO altitudes, the error from ignoring the TEME-to-J2000 conversion ranges from hundreds of meters to low kilometers, depending on the epoch. The nutation terms that differ between TEME and J2000 have amplitudes of up to ~17 arcseconds, which at 400 km altitude corresponds to position errors of about 30–60 meters per arcsecond contribution. During periods of high nutation amplitude, the total error can exceed 1 km.

**For ML feature engineering purposes**: if you are computing features entirely from TLE-derived SGP4 output (no external observations), staying consistently in TEME is acceptable. The frame matters when comparing with external observation sources — telescope astrometry (J2000), GPS-based precise ephemerides (GCRF), or external CDM state vectors.

**The correct tool for conversion**: Astropy's `astropy.coordinates` module has a `TEME` frame class that handles the conversion to GCRS (the Astropy equivalent of J2000/GCRF) correctly, including nutation and precession.

---

## RTN: the conjunction analysis frame

The RTN frame (also called RIC: Radial-In-track-Cross-track) is the frame in which CDM covariance matrices are expressed. It is not a global inertial frame — it is a local frame centered on one of the conjunction objects, and its axes move with that object's orbital position.

**Axis definitions** (centered on the primary object):
- **R (Radial)**: points radially outward from Earth's center, in the direction of the satellite's position vector. For a satellite at altitude h, R points "up" from Earth's surface directly below the satellite.
- **T (Transverse / In-track)**: for a circular orbit, T points along the velocity vector. For an elliptical orbit, T is defined along the instantaneous velocity vector projected perpendicular to R. T is approximately "forward" in the orbit.
- **N (Normal / Cross-track)**: perpendicular to the orbital plane, completing the right-handed system. N = R × T. This direction is approximately "up" out of the orbital plane.

**Why RTN is used for CDMs**: the position uncertainty of a tracked satellite is not isotropic. It has a characteristic shape aligned with the orbital geometry:

- **Along-track (T) uncertainty** is typically the **largest** — often 10 to 100 times the radial uncertainty. Along-track errors accumulate from J2 perturbations, drag modeling errors, and OD batch update timing. For a typical LEO debris object, along-track 1σ uncertainty might be 500m–5km, while radial uncertainty is 50–200m.
- **Radial (R) uncertainty** is typically the smallest for well-tracked objects — direct radar ranging constrains the radial distance well.
- **Cross-track (N) uncertainty** is typically intermediate.

This structure means the CDM covariance matrix in RTN is nearly diagonal, with \\(C_{TT} \gg C_{NN} > C_{RR}\\) in the position block. In ECI, the same covariance would be dense and rotation-dependent — harder to interpret and harder to validate.

**Reading a CDM covariance matrix**: CDMs provide a 6×6 covariance matrix in RTN for each object. The matrix is ordered [R, T, N, Ṙ, Ṫ, Ṅ] — position first, then velocity. The [0,0] element is radial position variance (\\(\sigma_R^2\\)), the [1,1] element is along-track position variance (\\(\sigma_T^2\\)), and the [2,2] element is cross-track position variance (\\(\sigma_N^2\\)).

**The conjunction plane**: for Pc calculation, the combined position uncertainty is projected onto the conjunction plane — the plane perpendicular to the relative velocity vector at TCA. In a typical LEO head-on encounter, the relative velocity is nearly along the T direction of one of the objects. The large along-track uncertainty (T) projects primarily along the relative velocity direction, which — because objects pass through this direction quickly — has relatively little effect on Pc. The cross-track and radial uncertainties, which are smaller in magnitude, determine how spread the position PDF is in the conjunction plane.

---

## Code: converting SGP4 TEME output to J2000/GCRS with Astropy

```python
"""
Demonstrate the TEME-to-GCRS (J2000) conversion using Astropy.

This example:
1. Propagates the ISS TLE to a specific epoch using python-sgp4 (fast)
2. Wraps the TEME output in an Astropy TEME frame
3. Converts to GCRS (the Astropy equivalent of J2000/GCRF)
4. Shows the difference between raw TEME and GCRS coordinates

Install: pip install sgp4 astropy
"""
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
import numpy as np

# ISS TLE
TLE_LINE1 = "1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991"
TLE_LINE2 = "2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697"

satellite = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

# Define the propagation epoch as an Astropy Time object
epoch_str = "2024-10-01T12:00:00"
t = Time(epoch_str, format="isot", scale="utc")

# Propagate using python-sgp4 at this epoch
# jday() converts to Julian date split into integer + fractional parts
jd, fr = jday(t.datetime.year, t.datetime.month, t.datetime.day,
               t.datetime.hour, t.datetime.minute,
               t.datetime.second + t.datetime.microsecond / 1e6)

error_code, r_teme_km, v_teme_kms = satellite.sgp4(jd, fr)

if error_code != 0:
    raise RuntimeError(f"SGP4 error code {error_code} — check TLE validity and epoch range")

print("=== SGP4 Raw Output (TEME frame) ===")
print(f"Position (km) : x={r_teme_km[0]:10.3f}, y={r_teme_km[1]:10.3f}, z={r_teme_km[2]:10.3f}")
print(f"Velocity (km/s): x={v_teme_kms[0]:10.6f}, y={v_teme_kms[1]:10.6f}, z={v_teme_kms[2]:10.6f}")

# Convert TEME to GCRS (J2000-equivalent) using Astropy
# Step 1: build an Astropy TEME coordinate object
r_teme = CartesianRepresentation(r_teme_km * u.km)
v_teme = CartesianDifferential(np.array(v_teme_kms) * u.km / u.s)

teme_coord = TEME(r_teme.with_differentials(v_teme), obstime=t)

# Step 2: convert to GCRS
gcrs_coord = teme_coord.transform_to(GCRS(obstime=t))

r_gcrs = gcrs_coord.cartesian.without_differentials()
v_gcrs = gcrs_coord.cartesian.differentials.get("s")

print("\n=== After Conversion to GCRS (J2000-equivalent) ===")
print(f"Position (km) : x={r_gcrs.x.to(u.km).value:10.3f}, "
      f"y={r_gcrs.y.to(u.km).value:10.3f}, "
      f"z={r_gcrs.z.to(u.km).value:10.3f}")

# Compute the difference (TEME raw vs GCRS-converted)
dx = r_gcrs.x.to(u.km).value - r_teme_km[0]
dy = r_gcrs.y.to(u.km).value - r_teme_km[1]
dz = r_gcrs.z.to(u.km).value - r_teme_km[2]
delta_km = (dx**2 + dy**2 + dz**2)**0.5

print(f"\n=== TEME vs GCRS Difference ===")
print(f"Component differences: dx={dx:.3f} km, dy={dy:.3f} km, dz={dz:.3f} km")
print(f"Total magnitude      : {delta_km:.3f} km  ({delta_km * 1000:.1f} m)")
print(f"\nIgnoring this conversion would introduce a {delta_km * 1000:.0f} m error")
print("when comparing SGP4 output with telescope observations in J2000.")
```

### Understanding frame relationships in context

```python
"""
Demonstrate ECI vs ECEF for ground station visibility.

Shows why you need ECEF for ground station geometry
and why you cannot use ECI positions for lat/lon lookups.
"""
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
import numpy as np

TLE_LINE1 = "1 25544U 98067A   24274.50000000  .00015669  00000-0  27837-3 0  9991"
TLE_LINE2 = "2 25544  51.6415 282.4781 0001567 231.1584 128.9321 15.50095566472697"
satellite  = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)

# Sample ground station: Schriever SFB, Colorado (approximate)
gs_lat_deg =  38.8    # degrees North
gs_lon_deg = -104.5   # degrees East (negative = West)
gs_alt_km  =  1.9     # km above sea level

t = Time("2024-10-01T12:00:00", format="isot", scale="utc")
jd, fr = jday(t.datetime.year, t.datetime.month, t.datetime.day,
               t.datetime.hour, t.datetime.minute, float(t.datetime.second))

_, r_teme_km, v_teme_kms = satellite.sgp4(jd, fr)

# Convert satellite TEME -> GCRS
r_cart = CartesianRepresentation(r_teme_km * u.km)
v_cart = CartesianDifferential(np.array(v_teme_kms) * u.km / u.s)
teme   = TEME(r_cart.with_differentials(v_cart), obstime=t)
gcrs   = teme.transform_to(GCRS(obstime=t))

# Convert GCRS -> ITRS (ECEF equivalent for Earth surface) for visibility check
itrs_sat = gcrs.transform_to(ITRS(obstime=t))

# Ground station in ECEF
gs_loc = EarthLocation(lat=gs_lat_deg * u.deg,
                        lon=gs_lon_deg * u.deg,
                        height=gs_alt_km * u.km)
gs_itrs = gs_loc.get_itrs(obstime=t)

# Compute range vector from ground station to satellite (in ECEF)
sat_xyz = itrs_sat.cartesian.xyz.to(u.km).value
gs_xyz  = gs_itrs.cartesian.xyz.to(u.km).value
range_vec_km = sat_xyz - gs_xyz
range_km     = np.linalg.norm(range_vec_km)

print("=== Ground Station Visibility Analysis ===")
print(f"Ground station       : Schriever SFB approx ({gs_lat_deg}°N, {gs_lon_deg}°E)")
print(f"Satellite ITRS pos   : {sat_xyz[0]:.1f}, {sat_xyz[1]:.1f}, {sat_xyz[2]:.1f} km")
print(f"Ground station ITRS  : {gs_xyz[0]:.1f}, {gs_xyz[1]:.1f}, {gs_xyz[2]:.1f} km")
print(f"Range to satellite   : {range_km:.1f} km")

# Elevation angle (above local horizon) — dot product of range vec with up vec
up_hat = gs_xyz / np.linalg.norm(gs_xyz)  # unit vector pointing radially up from ground station
range_hat = range_vec_km / range_km
sin_elev = np.dot(range_hat, up_hat)
elev_deg = np.degrees(np.arcsin(sin_elev))

print(f"Elevation angle      : {elev_deg:.1f}°")
print(f"Visible (>5° horizon): {'YES' if elev_deg > 5 else 'NO'}")
print()
print("Note: this calculation would be WRONG if we used ECI (GCRS) satellite")
print("position directly against a fixed ground station ECI position, because")
print("the ground station's ECI position changes continuously as Earth rotates.")
```

---

## Key Takeaways

- **A position vector is meaningless without specifying its reference frame.** ECI, ECEF, TEME, and RTN are four distinct coordinate systems used in SDA pipelines. Mixing them without conversion produces physically meaningless results.

- **SGP4 outputs TEME, not J2000 ECI.** TEME uses the true equator of date but the mean equinox of date — a historical artifact of how the SSN processed radar observations. The error from ignoring this conversion is hundreds of meters to low kilometers depending on epoch. Use Astropy's TEME frame class to convert correctly.

- **ECI is used for orbital mechanics; ECEF is used for ground station geometry.** Never use a satellite's ECI position to compute ground station visibility — the ground station's ECI position is constantly changing. Convert to ECEF (ITRS in Astropy) for any calculation involving geographic coordinates.

- **CDM covariance matrices are in RTN (Radial-Transverse-Normal).** The [1,1] element is the along-track (T) variance — typically the largest by a factor of 10–100 relative to radial. This along-track dominance is a structural feature of the uncertainty, not a data quality problem. Any ML model that ingests CDM covariances must understand this geometry.

- **For ML feature engineering from TLE-only data, staying consistently in TEME is acceptable.** The frame conversion matters when comparing SGP4 output with external observation sources (telescopes, GPS-based precise ephemerides). When all inputs come from SGP4 and all outputs stay in SGP4, frame consistency is maintained automatically.

---

## Quiz

{{#quiz 02-reference-frames.toml}}
