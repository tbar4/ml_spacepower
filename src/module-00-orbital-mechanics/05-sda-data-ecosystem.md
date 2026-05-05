# Lesson 5: The SDA Data Ecosystem

**Module:** ML and Game Theory for Space Power — M00: Orbital Mechanics and the SDA Data Ecosystem
**Source:** Space-Track.org API documentation; CelesTrak documentation (Dr. T.S. Kelso); LeoLabs API documentation; CCSDS 502.0-B-2 OMM standard; Space Force Organization public documentation

---


<!-- toc -->

## Where this fits

This is a reference lesson with no quiz. Its purpose is to give you a complete, honest map of every data source you will encounter when building commercial SDA ML products — what each source provides, how to access it, what its limitations are, and where it fits in a production pipeline architecture. You can return to this lesson anytime you encounter a new data source or API.

## The SSA/SDA distinction, explained fully

The terminology you use signals to customers whether you understand the domain.

**SSA (Space Situational Awareness)** was the dominant US government and commercial term through roughly 2018. SSA encompasses:
- Detection, tracking, and cataloging of resident space objects (RSOs)
- Conjunction assessment (screening for close approaches)
- Reentry prediction
- Characterization of object type (active payload, rocket body, debris, unknown)

SSA is fundamentally a *positional* and *catalog-maintenance* activity. The question SSA answers is: where are all the objects, what are they, and which ones are approaching each other?

**SDA (Space Domain Awareness)** was formalized in US Space Force doctrine in 2020, though the term appeared in planning documents before that. SDA extends SSA to include:
- Adversarial intent characterization: is a satellite conducting an intelligence-gathering close approach, or executing routine station-keeping?
- RF intelligence: what signals is a satellite emitting or receiving? Is its behavior consistent with its declared purpose?
- Pattern-of-life analysis: does a satellite's orbital behavior (maneuver history, proximity operations) suggest anomalous or threatening activity?
- Multi-source intelligence fusion: integrating positional data with RF signatures, optical observations, and human intelligence to build a complete operational picture of the space environment

SDA asks not just *where* but *what* and *why*.

**Practical implications for customer conversations:**

A satellite operator building conjunction avoidance automation needs SSA capabilities — TLE history, CDM feeds, covariance analysis. The word "SDA" is fine in the product name but they do not need the adversarial characterization layer.

A combatant command (Space Command, Indo-Pacific Command) buying services for mission assurance needs SDA — they want to know whether a Chinese or Russian satellite approaching a US asset is maneuvering for intelligence collection or just in an unlucky orbital slot. "SSA" undersells your product to this audience.

A spacecraft insurer pricing satellite hull coverage needs SSA risk quantification — historical conjunction rates, maneuver history, debris environment statistics. "SDA" may overstate the scope of what you are providing.

Know your audience. Use the right term.

---

## Free and accessible data sources

### Space-Track (18 SDS / CSpOC)

**What it is**: the authoritative public source for US Space Surveillance Network catalog data. Operated by the 18th Space Defense Squadron (18 SDS), Combined Space Operations Center (CSpOC) at Vandenberg SFB.

**Access**: free, registration required at space-track.org. Account approval typically takes 1–2 business days. US citizens can register immediately; foreign nationals may require additional review.

**Rate limits**: approximately 200 API requests per hour per account. For high-volume ingestion, contact 18 SDS for a data sharing agreement (DSA) that provides bulk access.

**Catalog size**: currently 50,000+ tracked objects. This has grown dramatically in recent years due to SpaceX Starlink deployments, the commissioning of Space Fence (Lockheed Martin / Space Force), and improved sensor capabilities. In 2020 the catalog was approximately 20,000 objects.

**Key API endpoints and format notes**:

The Space-Track API uses a REST interface with a query language. The base URL is `https://www.space-track.org/basicspacedata/query/class/`.

The most important classes for SDA ML work:

```
/class/gp/                  -- Current general perturbations (TLE/OMM), latest per object
/class/gp_history/          -- TLE history for specific objects (use this for feature engineering)
/class/cdm_public/          -- Publicly released conjunction data messages
/class/satcat/              -- Satellite catalog (object type, launch info, decay date)
/class/boxscore/            -- Object counts by country and type
```

**Critical note about OMM format**: since approximately 2020, Space-Track returns GP data in OMM (Orbit Mean-elements Message) JSON format by default, not in the legacy two-line text format. The data content is identical — same SGP4 mean elements — but the field names are different. The `spacetrack` Python library handles this transparently. If you are using raw `requests` calls, specify `format=tle` to get traditional TLE format or parse the JSON OMM as described in Lesson 1.

**The `spacetrack` Python library**:

```python
"""
Fetch TLE history for the ISS using the spacetrack Python library.

Install: pip install spacetrack
Account: register at space-track.org (free, 1-2 day approval)
"""
import os
from spacetrack import SpaceTrackClient

# Credentials from environment variables (never hardcode credentials)
ST_USER = os.environ.get("SPACETRACK_USER")
ST_PASS = os.environ.get("SPACETRACK_PASS")

if not ST_USER or not ST_PASS:
    raise ValueError("Set SPACETRACK_USER and SPACETRACK_PASS environment variables")

client = SpaceTrackClient(identity=ST_USER, password=ST_PASS)

# Fetch GP history for ISS (NORAD 25544) over a 30-day window
# The GP_HISTORY class returns all TLEs for an object in a date range
result = client.gp_history(
    NORAD_CAT_ID=25544,
    EPOCH=">2024-09-01",         # epoch after this date
    orderby="EPOCH asc",
    format="json",               # returns OMM JSON
    limit=200,                   # cap for safety; remove for full history
)

print(f"Fetched {len(result)} TLE records for ISS")
if result:
    print(f"\nFirst record fields:")
    for k, v in result[0].items():
        print(f"  {k}: {v}")

    print(f"\nEpoch range: {result[0]['EPOCH']} to {result[-1]['EPOCH']}")

# Fetch current TLE for a set of objects
current_tles = client.gp(
    NORAD_CAT_ID=[25544, 43226, 44235],  # ISS, Starlink-1, another object
    format="json",
)
print(f"\nCurrent TLEs for {len(current_tles)} objects")
for rec in current_tles:
    print(f"  {rec['NORAD_CAT_ID']:>7}  {rec['OBJECT_NAME']:<25}  epoch: {rec['EPOCH']}")
```

**The GP History endpoint** is the correct way to get TLE history for feature engineering. Do not scrape the human-readable website pages — use the API. The `gp_history` class returns the full historical sequence of TLEs for an object, which is the primary input for temporal ML models that detect maneuver anomalies.

### CelesTrak (Dr. T.S. Kelso)

**What it is**: a free, no-registration public data service run by Dr. T.S. Kelso, a leading expert in orbital mechanics and SGP4. CelesTrak redistributes Space-Track data and provides useful pre-grouped datasets.

**Access**: no registration required. Publicly accessible at celestrak.org.

**Important format change (2022)**: CelesTrak transitioned from the old TLE text file format to GP (General Perturbations) JSON format in 2022. The old URL patterns like `https://celestrak.org/SATCAT/TLE.txt` no longer work reliably. The current GP data endpoint is:

```
https://celestrak.org/SATCAT/GP.php?GROUP=<group>&FORMAT=json
```

Available groups include: `stations` (ISS and other crewed vehicles), `starlink`, `gps-ops`, `active`, `debris`, `geo`, and others.

```python
"""
Fetch current TLEs from CelesTrak (no registration required).

CelesTrak is suitable for:
- Development and testing (no account approval wait)
- Public demonstrations
- Academic work

NOT suitable for production SDA pipelines — Space-Track is the authoritative source.
"""
import requests
import json

# Current CelesTrak GP JSON endpoint for active satellites
CELESTRAK_GP_URL = "https://celestrak.org/SATCAT/GP.php"

# Fetch ISS and other crewed stations
response = requests.get(
    CELESTRAK_GP_URL,
    params={"GROUP": "stations", "FORMAT": "json"},
    timeout=30,
)
response.raise_for_status()
gp_data = response.json()

print(f"Fetched {len(gp_data)} objects from CelesTrak 'stations' group")
print(f"\nFields in each OMM record: {list(gp_data[0].keys())}")

# Show all objects with their epochs and mean motion
print(f"\n{'NORAD':>7}  {'Name':<25}  {'Epoch':<22}  {'Inc':>6}  {'MM (rev/day)':>12}")
print("-" * 80)
for rec in gp_data:
    print(f"{rec['NORAD_CAT_ID']:>7}  {rec['OBJECT_NAME']:<25}  "
          f"{rec['EPOCH']:<22}  {float(rec['INCLINATION']):>6.2f}  "
          f"{float(rec['MEAN_MOTION']):>12.5f}")
```

### OMM vs. TLE: same data, better format

As discussed in Lesson 1, OMM is the structured JSON/XML representation of the same SGP4 mean elements stored in a TLE. Key reminders:
- OMM is NOT higher fidelity than TLE
- You still use SGP4 to propagate OMM data
- OMM is now the default format from Space-Track's API
- The `spacetrack` library handles OMM transparently

---

## Commercial data with accessible tiers

### LeoLabs

**What it is**: a commercial phased-array radar network focused on LEO, operated from multiple sites worldwide (New Zealand, Alaska, Texas, Costa Rica, and others as of 2024). LeoLabs produces orbit determination solutions from their own radar observations, independent of the SSN.

**Key distinction**: LeoLabs does not produce TLEs in the traditional sense. Their OD solutions are expressed as OMM-format files or ephemeris data. LeoLabs OD precision for LEO objects often exceeds what the TLE format can fully represent — the format quantizes some element values. When LeoLabs data is converted to TLE format for interoperability, precision is lost.

**Why this matters for ML**: LeoLabs covariances are generally more realistic than TLE-derived covariances from Space-Track, because LeoLabs uses a higher-fidelity force model and tracks their own observation noise explicitly. If your ML model uses covariance features from LeoLabs data, the covariance realism problem is less severe.

**Access**: LeoLabs has a public API with trial access and commercial tiers. Registration at leolabs.space. Useful for prototyping higher-quality conjunction analysis pipelines.

### COMSPOC (formerly Ansys / Analytical Graphics)

**What it is**: a commercial space safety platform that competes directly with Space-Track for commercial customers who need higher-quality conjunction assessment. COMSPOC (Commercial Space Operations Center) provides high-fidelity ephemerides, CDMs, and event notifications.

**Key distinction**: COMSPOC uses more accurate force models than SGP4 for high-priority objects. Their CDM covariances are generally more realistic than Space-Track's TLE-derived covariances for well-tracked objects.

**Access**: contract required for production data. Contact Ansys/COMSPOC directly.

**Relevance**: if your ML product serves satellite operators who already use COMSPOC, your pipeline needs to ingest COMSPOC CDM format. The CDM standard (CCSDS 508.0-B-1) is the same; COMSPOC uses different ORIGINATOR values and may use different Pc methods.

### EU SST (European Union Space Surveillance and Tracking)

**What it is**: a consortium of European sensor networks and data centers providing space surveillance services to European operators. Funded and operated under EU regulation. The consortium includes sensors from France, Germany, Italy, Spain, and others.

**Relevance**: European spacecraft operators are increasingly required by regulation (EU Space Law) to use EU SST CDMs for collision avoidance decisions, rather than relying solely on US Space-Track data. If you are building products for European customers, you need to ingest EU SST CDM format and reconcile it with Space-Track data.

**Access**: EU SST provides a portal (eusst.eu) with limited public access. Operational data access for commercial operators requires registration and, in some cases, bilateral agreements.

---

## Enterprise and contract-required sources

### ExoAnalytic Solutions

**What it does**: operates a global network of commercial optical telescopes, primarily targeting GEO and high-altitude MEO objects. Optical sensors provide astrometric observations (right ascension and declination time series) rather than the range information that radar provides.

**Data product**: the product is angle-only observations (RA/Dec time series from each pass) and orbit determination solutions derived from these observations. ExoAnalytic does not provide raw video. Their OD solutions can detect small maneuvers and characterize object attitude changes for GEO objects.

**Relevance for SDA**: ExoAnalytic data is particularly valuable for GEO objects where radar returns are weak (large range). Optical data at GEO can detect maneuvers that radar-only solutions miss.

**Access**: contract required. ExoAnalytic is a defense contractor; expect a government contract vehicle or commercial contract depending on the use case.

### Kayhan Space

**What it does**: provides conjunction assessment, maneuver planning, and automated collision avoidance services specifically for satellite operators. Kayhan integrates multiple data sources (Space-Track, commercial providers) and provides decision support tools.

**Relevance**: if your ML product targets satellite operators, Kayhan is a direct competitor in the conjunction avoidance space. Understanding their product helps you position your value add — likely higher-quality risk prediction or behavioral analysis beyond what Kayhan provides.

**Access**: contract required. SaaS product with contract pricing.

### Slingshot Aerospace

**What it does**: space traffic management platform and analytics. Provides RSO characterization, conjunction analysis, and space domain awareness services. Also operates sensor infrastructure.

**Relevance**: competes with Space-Track commercial services and partially with DoD SDA analytics. A potential customer for your ML product or a competitor, depending on your target market.

**Access**: contract required.

### Unseenlabs

**What it does**: an RF emissions monitoring service. Unseenlabs operates a constellation of satellites that detect and geolocate RF emissions from ground objects, vessels, and other satellites. Their primary market is maritime surveillance (detecting ship AIS spoofing) but they have applications for satellite RF behavioral characterization.

**Why this is in a different category**: Unseenlabs data is not positional tracking data. It is behavioral intelligence — what signals is a satellite emitting? For SDA purposes, this is the "what is it doing?" layer on top of positional "where is it?" data. If a GEO satellite suddenly starts emitting in unusual frequency bands, Unseenlabs can detect that. This is a genuinely SDA (not SSA) data product.

**Access**: contract required.

---

## Missing providers worth knowing about

### SpaceFence (Lockheed Martin / US Space Force)

Space Fence is a ground-based S-band radar system on Kwajalein Atoll that came online in 2020. It is not a commercial API or data product — it is a classified US government sensor. However, it is directly responsible for the catalog growth from ~20,000 to 50,000+ objects since 2020. Space Fence can track objects as small as 10 cm in LEO, whereas the previous GEODSS/PAVE PAWS radar network had a detection floor around 10–30 cm depending on altitude.

Understanding Space Fence explains why the catalog grew so rapidly in 2020–2022 and why conjunction event rates increased accordingly: we are now tracking objects that were always present but previously undetected.

### Digantara

An Indian commercial SSA company building its own ground sensor network and developing LEO traffic management services. Relevant as the SDA market becomes more international. Not yet a major data provider for US commercial pipelines as of 2024, but worth monitoring as the Indian commercial space sector grows.

---

## Data pipeline architecture for ML feature engineering

Different ML use cases require different pipeline architectures. Here are the two primary patterns:

### Batch feature engineering (ML model training and evaluation)

For building and training ML models on historical data, you typically need:

1. **Ingest**: pull TLE histories from Space-Track's GP_HISTORY endpoint. Store raw OMM JSON.
2. **Storage**: Parquet files keyed by (NORAD_ID, epoch), partitioned by year/month. DuckDB or Polars for feature extraction queries — these are much more efficient than a time-series database for batch ML use cases because your access pattern is "give me all elements for NORAD_ID=X ordered by time," not a streaming write workload.
3. **Feature extraction**: propagate TLE sequences to Cartesian state in a common frame, compute element residuals after subtracting J2 predictions, extract BSTAR trends, compute epoch age statistics.
4. **Schema**: a well-structured TLE history table looks like this:

```python
"""
Schema for a TLE history Parquet table suitable for ML feature engineering.
This is the primary training data input for temporal models.
"""

TLE_HISTORY_SCHEMA = {
    "norad_id":           "int32",    # NORAD catalog number — primary key
    "epoch_jd":           "float64",  # Julian date of TLE epoch — partition/sort key
    "epoch_datetime":     "str",      # ISO8601 for human readability
    "inclination_deg":    "float64",  # degrees
    "eccentricity":       "float64",  # dimensionless, 0 to 1
    "raan_deg":           "float64",  # Right Ascension of Ascending Node, degrees
    "arg_perigee_deg":    "float64",  # Argument of perigee, degrees
    "mean_anomaly_deg":   "float64",  # degrees
    "mean_motion_revday": "float64",  # revolutions per day
    "bstar":              "float64",  # SGP4 drag coefficient
    "classification":     "str",      # 'U', 'S', 'C'
    "object_type":        "str",      # from satcat: PAYLOAD/ROCKET BODY/DEBRIS/UNKNOWN
    "maneuverable":       "bool",     # from satcat
    "epoch_age_days":     "float64",  # days since TLE epoch at query time (derived)
    "element_set_no":     "int32",    # sequential revision counter
    # Derived features (compute at ingestion time)
    "raan_j2_corrected":  "float64",  # raan minus predicted J2 drift from epoch
    "mm_sma_km":          "float64",  # semi-major axis derived from mean motion
    "altitude_km":        "float64",  # approximate altitude = sma - 6378.137
}

# Example DuckDB query for extracting features for objects of interest
FEATURE_QUERY = """
SELECT
    norad_id,
    epoch_datetime,
    mean_motion_revday,
    bstar,
    inclination_deg,
    raan_j2_corrected,
    altitude_km,
    -- Compute delta mean motion between consecutive TLEs for each object
    mean_motion_revday - LAG(mean_motion_revday) OVER (
        PARTITION BY norad_id ORDER BY epoch_jd
    ) AS delta_mean_motion,
    -- Time since previous TLE (useful for detecting tracking gaps)
    epoch_jd - LAG(epoch_jd) OVER (
        PARTITION BY norad_id ORDER BY epoch_jd
    ) AS days_since_prev_tle
FROM tle_history
WHERE
    norad_id IN (SELECT norad_id FROM active_payloads)
    AND epoch_datetime BETWEEN '2024-01-01' AND '2024-10-01'
ORDER BY norad_id, epoch_jd
"""

print("TLE history schema fields:")
for col, dtype in TLE_HISTORY_SCHEMA.items():
    print(f"  {col:<25} : {dtype}")

print(f"\nFeature extraction query (run with DuckDB or Polars):")
print(FEATURE_QUERY)
```

### Real-time alerting pipeline

For streaming conjunction alerts or maneuver detection alerts, a time-series database is more appropriate:

- **TimescaleDB** (PostgreSQL extension): good if you already use PostgreSQL and need SQL queries on time-series data
- **InfluxDB**: purpose-built time-series database, good for high-write-rate streaming ingestion
- **Architecture**: Space-Track CDM API polling (every 15–60 minutes) → parse CDMs → insert into time-series DB → trigger downstream alerting if Pc crosses threshold or shows adverse trend

For most commercial SDA ML products at early scale, a Parquet-based approach with DuckDB is simpler to operate and maintain than a dedicated time-series database. Reserve time-series databases for scenarios where you have truly streaming, high-rate writes that require sub-second latency.

---

## Product positioning and competitive landscape

Your ML model is built on top of the data sources above. So are your competitors' products. The competitive moat is not the data — it is the model quality and integration.

**What commercial customers buy today** (2024):

- **Satellite operators**: conjunction avoidance automation, maneuver planning, Pc risk trending. Primary buyers: commercial constellation operators (Starlink SpaceX, OneWeb, Planet Labs, etc.), GEO operators.
- **Spacecraft insurers**: risk scoring for launch and on-orbit coverage. A 50% improvement in Pc accuracy translates directly to better loss ratio modeling.
- **Government programs with unclassified data access**: SpaceWERX SBIR topics, AFWERX contracts, NOAA commercial data buys. These programs buy from commercial SDA companies using unclassified TLE/CDM data — the same data you can access.

**Competitors building analytics products on top of the same public data**:

- LeoLabs: builds analytics on their own radar data. Higher data quality at LeoLabs-covered altitudes.
- Slingshot Aerospace: analytics platform with operational focus.
- Kayhan Space: conjunction avoidance-specific tooling.
- ExoAnalytic: GEO-focused characterization.
- COMSPOC / Ansys: high-fidelity OD and conjunction products.

**Your differentiation**: in a market where the underlying data is largely the same, differentiation comes from model quality (better Pc prediction), integration quality (API reliability, latency, documentation), and domain depth (understanding edge cases that pure ML approaches miss — covariance realism, maneuver type discrimination, geomagnetic storm awareness). This curriculum is designed to give you that domain depth.

**A note on SAM.gov**: any entity receiving US federal contracts must be registered in the System for Award Management (SAM.gov). If your company targets government SDA contracts — SBIR, STTR, Other Transaction Authority (OTA) — SAM.gov registration is required before award. The process takes 2–4 weeks. Budget for this lead time if pursuing government work.

---

## Key Takeaways

- **SSA is positional catalog work; SDA extends this to adversarial intent and behavioral characterization.** Use the right term with the right customer: SSA for satellite operators, SDA for combatant command customers.

- **Space-Track is free, authoritative, and rate-limited.** Use the GP_HISTORY endpoint for TLE history. The API now returns OMM JSON by default. The `spacetrack` Python library handles authentication and format differences transparently.

- **CelesTrak is free with no registration**, but use the updated GP JSON endpoint (not old TLE text file URLs). It is suitable for development and testing; use Space-Track for production.

- **LeoLabs provides higher-quality LEO covariances** than TLE-derived estimates, with trial API access. Their data product is OMM-format OD solutions, not traditional TLEs.

- **For batch ML feature engineering**: Parquet files + DuckDB or Polars. Schema keyed by (NORAD_ID, epoch_jd). Compute J2-corrected RAAN and semi-major axis at ingestion time.

- **Your competitive moat is model quality and integration, not raw data access.** The same Space-Track data is available to all your competitors. Domain depth — understanding when TLE covariances are wrong, when SGP4 accuracy breaks down, what maneuver signatures really look like — is what differentiates a good ML product from a mediocre one. That is what this curriculum is building.
