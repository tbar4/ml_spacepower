# Module 9: Applied SDA ML

## Where this module fits

Modules 0 through 8 built the full stack: orbital mechanics, neural network training, reinforcement learning, search and planning, game theory, multi-agent RL, partial observability, OpenSpiel, and a Rust systems capstone. Every algorithm in those modules was motivated by SDA scenarios but trained on toy environments. This module drops the toy environments. You are now building a commercial product.

The target is the highest-value SDA ML product a solo uncleared founder can build from fully public data: a maneuver detection and trajectory anomaly detector running on TLE history from Space-Track. No sensor contract required. No clearance required. Real data, real engineering, real commercial upside.

This is the build-the-product module. Every concept in Modules 1–8 has a direct counterpart in what you build here.

## What we cover

**Sequence models for maneuver detection (Lesson 1).** A satellite's TLE history is a time series. A single TLE says almost nothing about intent; the history over days and weeks reveals whether an object is station-keeping, executing a campaign maneuver, or behaving anomalously. This lesson builds an LSTM trained on TLE history to classify windows as maneuver or no-maneuver. It covers the complete pipeline: feature engineering from orbital elements, synthetic training data generation (the honest solution to the label scarcity problem), object class stratification, handling irregular TLE cadence, and a full PyTorch implementation with operationally relevant evaluation metrics.

## Lessons

1. [Sequence models for maneuver detection](01-sequence-models-maneuver-detection.md)

## Module project: Production maneuver detection pipeline

You will build a complete end-to-end production pipeline:

1. Fetch 90-day TLE history from the Space-Track GP History API for a curated object set
2. Clean and preprocess: filter low-quality TLEs, remove rocket bodies, grid to daily resolution, handle observation gaps
3. Engineer time-normalized delta features with F10.7 solar flux correction
4. Generate synthetic training data via maneuver injection into quiet debris TLE histories
5. Train the LSTM classifier from Lesson 1
6. Evaluate on a real labeled test set of documented ISS reboost events
7. Run the trained model in a live simulation: process new TLEs and output maneuver alerts

The project is the capstone for the entire curriculum. It combines time-series modeling (Module 2), supervised classification training (Module 2), evaluation under class imbalance (Module 1 probability), and honest product framing from Modules 0 and 5.

## What makes this module different from the ones before it

Every prior module had a clean feedback loop: run the code, get a loss, watch it decrease, declare success. Commercial ML products have a different feedback loop. The loss decreasing does not mean the product works. The metrics that matter — detection latency, false alarm rate per object per month, miss rate by maneuver size — require domain knowledge to define, real data to measure, and honest acknowledgment of where the approach fails.

This module does not pretend the problem is easier than it is. Maneuver detection from public TLEs has been worked on by LeoLabs, Slingshot, ExoAnalytic, and academic labs for fifteen years. The commercial opportunity is not solving an unsolved problem; it is delivering a TLE-only product at a price point that radar-based solutions cannot match, and eventually integrating maneuver detection with the game-theoretic adversary modeling you built in Modules 5–8. That integration is the genuine differentiator. This module builds the first half.

## What's next

The natural extensions from this module are:
- Intent inference: given detected maneuver, classify intent (station-keeping, rendezvous, avoidance) using the game-theoretic framing from Module 5
- Operator-facing alerting: a lightweight service that wraps the trained model and delivers alerts via API
- Multi-object tracking: extend from single-object windows to fleet-level anomaly scoring across an entire watched catalog
