# Module 0: Orbital Mechanics and the SDA Data Ecosystem

---


<!-- toc -->

## Why this module comes before everything else

Every lesson in this curriculum involves building, evaluating, or deploying a machine learning model that processes space domain data. Before you can reason about what those models are doing, you need to understand the data artifacts they consume.

This is not an orbital mechanics course. You will not derive Kepler's laws or integrate the equations of motion by hand. What you will do is develop working knowledge of the data structures, coordinate systems, propagation tools, and domain concepts that appear in every SDA data pipeline. When a conjunction assessment model ingests a CDM and produces a risk score, you need to know what a CDM is, what its fields mean, what assumptions went into generating it, and what the model is actually learning. Without that, you cannot debug the pipeline, assess model quality, or explain results to a customer.

The lessons here are deliberately practical. Each one starts with a real data artifact — a TLE, a CDM, an API response — and builds the theory needed to interpret it correctly.

---

## SSA vs. SDA: a distinction that matters for your customers

These terms are often used interchangeably in commercial settings, and carelessly conflating them will mark you as a newcomer to DoD customers.

**SSA (Space Situational Awareness)** is the legacy term, dominant through the 2010s. It refers primarily to catalog maintenance, object tracking, and conjunction screening — understanding where things are and predicting where they will be. SSA is positional: detecting, tracking, characterizing, and cataloging resident space objects.

**SDA (Space Domain Awareness)** is the current DoD term, adopted officially in 2020. SDA extends SSA to include adversarial intent characterization, RF intelligence, behavioral analysis, and the fusion of multi-source intelligence to understand not just where objects are but what they are doing and why. A maneuvering satellite that happens to reposition near a US asset is an SSA event (we detected a maneuver) and an SDA question (is this an intelligence-gathering approach or a routine station-keeping burn?).

The practical distinction for product positioning: commercial satellite operators primarily need SSA capabilities — conjunction avoidance, maneuver detection for collision risk, covariance realism for accurate Pc. Government customers, especially at the combatant command level, want SDA — behavioral pattern-of-life analysis, RF characterization, anomaly detection with adversarial context. Your ML product architecture is similar either way, but your customer conversations and contract structures differ significantly.

In this curriculum, we use SDA as the umbrella term because it encompasses everything we build. Where the distinction matters, we call it out explicitly.

---

## Lessons in this module

**Lesson 1: TLEs and Keplerian Elements**

The canonical data artifact for tracking space objects is the Two-Line Element set (TLE). This lesson starts with a real TLE, parses every field, then builds the six Keplerian elements needed to understand what those fields represent. Covers the critical warning about mean vs. osculating elements and why you cannot difference consecutive TLE elements to detect maneuvers. Introduces the OMM (Orbit Mean-elements Message) format that Space-Track now returns from its API.

**Lesson 2: Reference Frames**

A position vector means nothing without a reference frame. This lesson covers the four frames you will encounter in every SDA pipeline: ECI (J2000/GCRF) for orbital mechanics, ECEF for ground station geometry, TEME for SGP4 output, and RTN/RIC for conjunction analysis covariances. Includes the most common pipeline bug in SSA software: treating SGP4's TEME output as J2000 ECI and comparing it with telescope observations without converting.

**Lesson 3: SGP4 Propagation**

SGP4 is the propagation engine behind every public TLE. This lesson covers what SGP4 includes (J2 through J6 harmonics, atmospheric drag via BSTAR), what it excludes (third-body gravity for LEO/MEO objects — that is SDP4), and its honest accuracy characterization. Includes working Python code for single-object and batch propagation, and the signature of a maneuver in TLE history.

**Lesson 4: Conjunction Analysis**

Two objects pass close enough to trigger a screening event. This lesson covers the full conjunction analysis pipeline: asymmetric pizza-box screening volumes, the conjunction plane geometry, the Foster/Chan Pc computation method, and the CCSDS CDM format in detail. Covers covariance realism — why TLE-derived covariances are systematically too small, and why this makes ML covariance inflation models commercially valuable.

**Lesson 5: The SDA Data Ecosystem (no quiz)**

A reference lesson covering every major data source you will encounter: Space-Track/18 SDS, CelesTrak, LeoLabs, COMSPOC, ExoAnalytic, Kayhan, Slingshot, EU SST, and others. Covers API access patterns, data pipeline architecture for ML feature engineering, and the commercial product landscape your models will compete in.

---

## How this module connects to the rest of the curriculum

The connections are direct and specific:

**Module 1 (Foundations) — Monte Carlo Pc**: Module 1 Lesson 3 covers Monte Carlo estimation. The canonical SDA application is Monte Carlo Pc: sampling from the CDM's combined covariance and counting collision events. That lesson assumes you know what a covariance matrix in RTN space means, what a CDM is, and why the analytical Foster Pc differs from a Monte Carlo estimate. Lesson 4 here gives you that foundation.

**Module 7 (Partial Observability) — Particle Filters for Orbit Determination**: The particle filter lesson applies sequential Monte Carlo to tracking a maneuvering satellite. The state vector is position and velocity in ECI. The observation model involves reference frame conversions from ground-based radar. Every concept from Lessons 1–3 of this module feeds directly into that application.

**Module 8 (Capstone) — SSA Game**: The capstone strategic game involves RSO tracking, sensor scheduling, and maneuver attribution. Players work with TLE-derived features, CDM-derived risk scores, and the behavioral analysis concepts from the SDA ecosystem lesson. You need the full Module 0 vocabulary to engage with the capstone at the right level.

**Module 9 (Applied SDA ML)**: The applied module builds production ML features from TLE histories, CDM sequences, and behavioral indicators. Every feature engineering decision in that module traces back to domain concepts introduced here: J2-driven RAAN precession as a baseline to subtract, along-track uncertainty as the dominant CDM covariance dimension, epoch age as a weak uncertainty proxy.

---

## What you will be able to do after this module

After completing Module 0, you will be able to:

- Read a TLE or OMM from Space-Track and correctly interpret every field
- Propagate any tracked object forward using python-sgp4 and correctly identify the output reference frame as TEME
- Convert TEME positions to GCRS/J2000 using Astropy for comparison with external observation sources
- Parse a CCSDS CDM, extract the covariance matrix, understand which dimension is largest and why, and explain what the COLLISION_PROBABILITY field represents and what method produced it
- Access Space-Track and CelesTrak programmatically and build a simple TLE history ingestion pipeline
- Explain the SSA/SDA distinction to both commercial satellite operators and DoD customers
- Describe the competitive landscape of commercial SDA data and analytics providers

These capabilities are prerequisites for every applied lesson in the curriculum. The module project builds a complete conjunction screening pipeline from public data that demonstrates all of them together.
