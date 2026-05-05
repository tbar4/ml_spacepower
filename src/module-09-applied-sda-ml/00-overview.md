# Module 9: Applied SDA ML


<!-- toc -->

## Where this module fits

Modules 0 through 8 built the full stack: orbital mechanics, neural network training, reinforcement learning, search and planning, game theory, multi-agent RL, partial observability, OpenSpiel, and a Rust systems capstone. Every algorithm in those modules was motivated by SDA scenarios but trained on toy environments. This module drops the toy environments. You are now building a commercial product.

The target is the highest-value SDA ML product a solo uncleared founder can build from fully public data: a maneuver detection and trajectory anomaly detector running on TLE history from Space-Track. No sensor contract required. No clearance required. Real data, real engineering, real commercial upside.

This is the build-the-product module. Every concept in Modules 1–8 has a direct counterpart in what you build here.

## What we cover

**Sequence models for maneuver detection (Lesson 1).** A satellite's TLE history is a time series. A single TLE says almost nothing about intent; the history over days and weeks reveals whether an object is station-keeping, executing a campaign maneuver, or behaving anomalously. This lesson builds an LSTM trained on TLE history to classify windows as maneuver or no-maneuver. It covers the complete pipeline: feature engineering from orbital elements, synthetic training data generation (the honest solution to the label scarcity problem), object class stratification, handling irregular TLE cadence, and a full PyTorch implementation with operationally relevant evaluation metrics.

**Transformers for orbital sequences (Lesson 2).** Attention-based architectures have displaced LSTMs across most sequence tasks, and orbital sequences are no exception once the training set grows large enough. This lesson replaces the LSTM with an encoder-only transformer using a CLS token, positional encoding adapted for daily-gridded TLE sequences, and observation masking for irregular cadence. Masked pretraining on the entire unlabeled TLE catalog — analogous to BERT — significantly improves fine-tuned maneuver detection by teaching the encoder what normal orbital evolution looks like before any maneuver labels are introduced. The lesson also extracts attention weights as an operationally relevant explainability mechanism: after flagging a maneuver, you can show an analyst exactly which TLE epochs drove the decision.

**Multi-object tracking and fleet-level anomaly scoring (Lesson 3).** An operator watching 200 objects does not care about individual window scores in isolation. This lesson builds the fleet-level infrastructure: a Bayesian state estimator per object (a Kalman filter with SGP4 prediction), Mahalanobis innovation scoring to detect per-TLE anomalies, and personalized thresholds calibrated to each object's historical noise level. On top of per-object scoring, a CUSUM accumulator detects sustained sub-threshold anomaly patterns — the signature of a slow, low-delta-V approach campaign — and a cross-catalog correlation check flags pairs of proximate objects that execute correlated maneuvers on the same day. The connection to Module 7's particle filters handles the non-Gaussian posterior after a confirmed maneuver event.

**Intent inference and game-theoretic adversary modeling (Lesson 4).** Detection without intent classification is a fire alarm without a fire marshal. This lesson builds an intent classifier that uses Hill-Clohessy-Wiltshire relative frame features — separation rate, along-track closure, approach geometry — to assign probability distributions over four intent categories: station-keeping, collision avoidance, repositioning, and rendezvous/proximity operations. The classifier is trained via PSRO (Module 6) against an adaptive adversary who actively tries to disguise RPO approaches as legitimate maneuvers, producing a Defender strategy robust to the best-known disguise tactics. The conjunction-masking signature from Module 8's game design maps directly to orbital geometry features detectable with this approach, and the Nash-equilibrium strategy profiles from `ssa_cfr` provide the hardest adversarial training examples. This lesson is the integration point: the LSTM/transformer detector, the fleet tracker, the game theory, and the deterrence thesis all converge here.

## Lessons

1. [Sequence models for maneuver detection](01-sequence-models-maneuver-detection.md)
2. [Transformers for orbital sequences](02-transformers-orbital-sequences.md)
3. [Multi-object tracking and fleet-level anomaly scoring](03-multi-object-tracking.md)
4. [Intent inference and game-theoretic adversary modeling](04-intent-inference.md)

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

The natural production extension from this module is an operator-facing alerting service: a lightweight API wrapper around the full pipeline (detector → tracker → intent classifier) that processes incoming Space-Track TLE batches on a scheduled cadence and delivers anomaly alerts with intent assessments to subscribed operators. The architecture is straightforward once the ML components are production-ready; the engineering challenge is latency, reliability, and the alert fatigue problem — calibrating thresholds so that every alert delivered is one an analyst will act on.
