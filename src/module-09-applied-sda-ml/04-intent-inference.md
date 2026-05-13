# Lesson 4: Intent Inference and Game-Theoretic Adversary Modeling

**Module:** Applied SDA ML — M09: Building Commercial SDA Products
**Source:** Harrison (2020) "Space Threat Assessment"; Langdon et al. (2019) "Modeling Intent in On-Orbit Proximity Operations"; Module 5 (CFR), Module 6 (PSRO, MAPPO), Module 7 (opponent modeling), Module SP (deterrence-by-detection thesis); Albright & Zhu (2022) "Rendezvous and Proximity Operations Maneuver Classification"

---


<!-- toc -->

## Where this fits

Lessons 1 and 2 detect that a maneuver occurred. Lesson 3 tracks it fleet-wide. This lesson asks the question that makes the product commercially and strategically valuable: *why did the satellite maneuver?*

Detection without intent inference is a fire alarm without a fire marshal. Knowing that an object's orbit changed is operationally useful only if you can characterize what the new orbit implies — is this station-keeping that should be ignored, a collision avoidance maneuver that is routine, or a rendezvous approach to a nearby asset that requires alerting?

This is also the lesson where the entire theoretical curriculum converges. Module 5's CFR and Module 6's PSRO were built specifically to handle this class of problem: an adversary with private information (their actual intent) acting against a defender with limited sensors and inference capability. The game-theoretic models are not academic exercises — they are the inference engine for intent classification.

---

## The detection-to-attribution gap

There are three distinct problems in orbital attribution:

1. **Detection**: Did this object maneuver? (Lessons 1–2)
2. **Intent inference**: What was the intent of the maneuver? (This lesson)
3. **Attribution**: Which actor authorized and executed this maneuver? (Requires additional intelligence beyond TLE data)

This lesson covers step 2. Step 3 — attributing the maneuver to a specific actor with enough confidence for a diplomatic or operational response — requires cross-domain intelligence fusion (satellite registry, launch records, operator behavior patterns, signals intelligence) that is outside the scope of public TLE data alone. Module SP's deterrence-by-detection thesis is specifically about step 2: ML-enabled intent inference at scale reduces orbital ambiguity, making gray zone operations harder to execute without detection.

A full attribution pipeline connects all three: the LSTM/transformer detector flags a maneuver, the intent classifier assigns a probability distribution over intent categories, and an analyst combines that inference with external information to assess whether an attributable actor executed an adversarial action. The ML contribution is steps 1 and 2; the analyst contribution is step 3.

---

## Intent taxonomy

Define four intent categories for LEO proximity operations:

**Station-keeping**: The operator is correcting for atmospheric drag and maintaining a planned orbit. Signature: small, periodic burns in the velocity direction, magnitude consistent with predicted drag at the object's altitude and solar flux level. The orbital change is predictable from the object's published mission parameters.

**Collision avoidance (CAM)**: The operator is executing a maneuver to increase separation from a predicted close approach. Signature: the maneuver is correlated in timing with a published conjunction warning (Space-Track CDM or equivalent), and the new orbit increases separation from the predicted conjunction object. The maneuver is reactive to external events, not internally motivated.

**Repositioning**: The operator is moving the satellite to a different operational orbit. Signature: a sustained maneuver campaign over multiple days, resulting in a significant semi-major axis or inclination change. No nearby objects involved. Purpose is operational reassignment, not proximity operations.

**Rendezvous/proximity operations (RPO)**: The satellite is maneuvering toward another specific object. Signature: the new orbit reduces separation from a specific nearby object, especially if the approach geometry is consistent with a Hohmann transfer or low-delta-V phasing orbit to that object. This is the high-priority category for adversarial intent inference.

The RPO category is further subdivided by approach geometry:

- **Inspection approach**: Slow, stable, maintaining separation > 1 km
- **Close approach**: Reducing separation to < 1 km on a trajectory consistent with further approach
- **Conjunction-masking approach** (Module 8 game): Maneuvering to a position where the approach is geometrically consistent with a natural conjunction rather than a deliberate approach — the adversary exploits orbital mechanics to disguise intent

---

## Features for intent classification

The LSTM/transformer features from Lesson 1 (orbital element rates) are necessary but not sufficient for intent classification. Intent inference requires orbital geometry features that characterize the relationship between the maneuvering object and nearby objects.

**Hill-Clohessy-Wiltshire (HCW) relative frame features**:

For each maneuvering object, identify all catalog objects within 100 km and compute relative state vectors in the HCW (Clohessy-Wiltshire) frame, also called the LVLH (Local Vertical Local Horizontal) frame:

```python
def relative_state_hcw(chief_tle, deputy_tle, epoch):
    """
    Compute relative position/velocity of deputy w.r.t. chief in HCW frame.
    Returns [Δx, Δy, Δz, Δvx, Δvy, Δvz] in km and km/s.
    """
    r_chief, v_chief = sgp4_propagate(chief_tle, epoch)
    r_deputy, v_deputy = sgp4_propagate(deputy_tle, epoch)

    # Rotating frame: x = radial, y = along-track, z = cross-track
    r_hat = r_chief / np.linalg.norm(r_chief)
    h = np.cross(r_chief, v_chief)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, r_hat)

    delta_r = r_deputy - r_chief
    delta_v = v_deputy - v_chief

    return np.array([
        np.dot(delta_r, r_hat),    # radial separation
        np.dot(delta_r, y_hat),    # along-track separation
        np.dot(delta_r, z_hat),    # cross-track separation
        np.dot(delta_v, r_hat),    # radial closing rate
        np.dot(delta_v, y_hat),    # along-track closing rate
        np.dot(delta_v, z_hat),    # cross-track closing rate
    ])
```

**Approach trajectory features**:

Compute the rate of change of relative position over consecutive TLE epochs to characterize whether the separation is decreasing, stable, or increasing:

```python
def approach_features(hcw_history):
    """Compute approach trajectory statistics from HCW sequence."""
    separations = [np.linalg.norm(h[:3]) for h in hcw_history]
    along_track = [h[1] for h in hcw_history]

    return {
        'separation_rate': np.polyfit(range(len(separations)), separations, 1)[0],  # km/day
        'min_separation': min(separations),
        'along_track_closure': along_track[-1] - along_track[0],  # total along-track change
        'approach_consistency': np.corrcoef(range(len(separations)), separations)[0,1],  # monotonicity
    }
```

An RPO approach will show: negative `separation_rate` (closing), decreasing `min_separation`, significant `along_track_closure`, and high `approach_consistency` (monotonically decreasing separation). A CAM will show the opposite: increasing separation, positive rate.

---

## Bayesian intent classifier

Given the orbital element features from Lessons 1–2 and the HCW geometry features above, train a classifier to assign probabilities over the four intent categories.

The output is not a hard label but a probability distribution: `P(intent = RPO | features) = 0.73`. This distribution is what you update through time as new TLE observations arrive — a Bayesian belief update over intent.

```python
class IntentClassifier(nn.Module):
    def __init__(self, n_sequence_features, n_geometry_features, n_intents=4):
        super().__init__()
        # Sequence encoder (reuse pretrained transformer or LSTM)
        self.sequence_encoder = OrbitalTransformer(n_features=n_sequence_features)
        # Geometry encoder
        self.geometry_encoder = nn.Sequential(
            nn.Linear(n_geometry_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # Combined classification head
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_intents),
        )

    def forward(self, sequence_x, geometry_x):
        seq_repr = self.sequence_encoder(sequence_x)  # (batch, 64)
        geo_repr = self.geometry_encoder(geometry_x)  # (batch, 32)
        combined = torch.cat([seq_repr, geo_repr], dim=-1)
        return torch.softmax(self.classifier(combined), dim=-1)  # (batch, 4)
```

The intent probabilities update at each new TLE observation. Use a running Bayesian update:

```python
def update_intent_belief(prior, likelihood, temperature=1.0):
    """Update intent belief given new observation likelihoods."""
    posterior_unnorm = prior * (likelihood ** temperature)
    return posterior_unnorm / posterior_unnorm.sum()
```

```rust
fn update_intent_belief(prior: &[f64; 4], likelihood: &[f64; 4], temperature: f64) -> [f64; 4] {
    let unnorm: [f64; 4] = std::array::from_fn(|i| prior[i] * likelihood[i].powf(temperature));
    let total: f64 = unnorm.iter().sum();
    std::array::from_fn(|i| unnorm[i] / total)
}

fn main() {
    // [station-keeping, CAM, repositioning, RPO]
    let mut belief = [0.20f64, 0.20, 0.20, 0.40];   // 40% prior RPO suspicion
    let labels = ["SK", "CAM", "Repos", "RPO"];

    // Three sequential TLE epochs with increasing RPO likelihood from HCW geometry
    let observations: &[[f64; 4]] = &[
        [0.10, 0.15, 0.20, 0.60],   // closing geometry visible
        [0.05, 0.10, 0.15, 0.80],   // further closure, less ambiguous
        [0.03, 0.05, 0.10, 0.90],   // approach trajectory nearly certain
    ];

    println!("Intent belief updates (temperature = 1.0):");
    for (i, obs) in observations.iter().enumerate() {
        belief = update_intent_belief(&belief, obs, 1.0);
        print!("  After obs {}: ", i + 1);
        for (label, &p) in labels.iter().zip(belief.iter()) {
            print!("{label}={:.0}%  ", p * 100.0);
        }
        println!();
    }
}
```

The temperature parameter allows calibration between how quickly the belief updates (high temperature = more weight on new observations, lower temperature = more inertia from prior). For slow-moving approach campaigns, a lower temperature is appropriate; for sudden burn events, higher.

---

## The game-theoretic framing

The Bayesian classifier above treats intent as a fixed latent variable. But an adversary with awareness of your detection capability will *adapt* — executing maneuvers in ways that look like legitimate station-keeping or CAM from the observational signature, while achieving a rendezvous objective. This is exactly the conjunction-masking game from Module 8.

The PSRO solution from Module 6 handles the adaptive adversary:

1. Initialize a policy for the Adversary (executes maneuvers) and a policy for the Defender (performs intent inference).
2. Train the Adversary to maximize approach success against the current Defender inference model.
3. Train the Defender to minimize mis-classification against the current Adversary policy.
4. Alternate, maintaining a population of both Adversary and Defender strategies.
5. At convergence, the Defender policy is robust to the best known Adversary strategies.

The practical implication: an intent classifier trained purely on historical maneuver data learns to recognize historical maneuver patterns. An adversary that has studied your classifier can route around it. A classifier trained via PSRO against an adaptive adversary learns to classify *adversarially disguised* RPO approaches — it is trained on the hardest examples the adversary can generate.

The output of PSRO training is a **mixed strategy** for the Defender: a probability distribution over intent classification rules. The Nash-approximating strategy does not commit to any single rule that an adversary could learn to exploit; it randomizes over a portfolio of inference approaches.

This is not an abstract game-theoretic property. It directly addresses the Harrison escalation problem from Module SP: if both sides have intent inference capability calibrated by adversarial training, neither side can easily execute gray zone operations that appear routine from the outside. The cost of disguising RPO as station-keeping increases when the defender's classifier was trained to detect exactly that disguise.

---

## Orbit-based intent reasoning: the conjunction-masking signature

Module 8 designed the conjunction-masking game around a specific adversary strategy: maneuvering to a position where your new orbit is geometrically consistent with a natural conjunction with a third object, making it ambiguous whether your close approach to a defended asset is deliberate or incidental.

The orbital signature of conjunction-masking is detectable with the features developed here:

1. **Two objects in proximity, one of which is a natural conjunction risk**: The approached object and the debris/defunct satellite nearby are at approximately the same altitude in a geometry that makes natural close approaches plausible.
2. **The approaching object's maneuver minimizes anomaly score**: The burn is executed to place the new orbit as close to the "expected station-keeping" distribution as possible, while still closing on the target.
3. **The approach is along-track rather than radial**: Natural conjunctions in LEO are typically along-track (relative velocity dominated by orbital mechanics). An RPO that exploits this will execute an along-track approach to minimize the radial HCW anomaly.

Training a classifier to specifically recognize this pattern requires examples of conjunction-masking maneuvers — which is precisely the synthetic data that the Module 8 Rust CFR solver generates. The Nash-equilibrium strategy profile from `ssa_cfr` characterizes the distribution of Adversary actions in a conjunction-masking game at equilibrium: these are the hardest-to-detect approach trajectories. Injecting these as training examples produces a classifier that specifically detects conjunction-masking attempts.

This is the integration point between the Rust capstone and the Python ML pipeline: the game solver generates adversarial training data; the classifier learns from it.

---

## Operator-facing output

The output of the full pipeline — detector + fleet tracker + intent classifier — should be an alert that an analyst can act on, not a raw probability vector:

```
ALERT — Object 58900 (COSMOS 2576, GEO, Russia)
Detected: Maneuver epoch 2026-04-28T14:32Z
Detection confidence: 94%

Intent assessment:
  Station-keeping:      3%
  Collision avoidance:  8%
  Repositioning:        12%
  Proximity operations: 77%  ← PRIMARY ASSESSMENT

Nearest object at maneuver epoch: AEHF-6 (USA-337, GEO, DOD)
Separation at maneuver: 847 km → 612 km (closing, rate: -47 km/day)
Projected closest approach: 2026-05-04T09:14Z at 183 km
Approach geometry: along-track, consistent with phasing maneuver

Action recommended: Elevated monitoring. Notify asset operator.
Basis: TLE history (Space-Track), conjunction geometry, transformer attention [epochs +3,+5,+7 highest]
```

This output format is designed for a DoD customer. The intent probability distribution is explicit, the primary assessment is labeled, the approach geometry is described in operationally meaningful terms, and the attribution is honest: this assessment is based on public TLE data and orbital geometry inference only, not on additional intelligence sources.

---

## Key Takeaways

- **Detection without intent inference is a fire alarm without a fire marshal.** Knowing a maneuver occurred is operationally useful only when paired with a characterization of what the new orbit implies.
- **Intent inference operates on HCW relative frame features, not just orbital element rates.** The relationship between the maneuvering object and nearby catalog objects — separation rate, along-track closure, approach consistency — is the signal that distinguishes RPO from station-keeping.
- **The Bayesian belief update tracks intent probability through time.** As new TLE observations arrive, update the posterior over intent categories. A single-point classifier answer at one epoch is less reliable than a posterior that has been conditioned on 10 days of post-maneuver observations.
- **PSRO against an adaptive adversary produces a classifier robust to disguised RPO.** A classifier trained on historical data can be exploited by an adversary who has studied it. PSRO trains the classifier against the hardest adversary strategies it will encounter, producing a mixed-strategy Defender that does not commit to any exploitable rule.
- **The conjunction-masking game (Module 8) generates the hardest adversarial training examples.** The Nash-equilibrium strategy profile from `ssa_cfr` describes exactly how an adversary optimally disguises an RPO approach as a natural conjunction. These equilibrium trajectories are the training data for the conjunction-masking classifier.
- **The full pipeline connects detection to attribution without conflating them.** Detection (maneuver occurred) + intent inference (likely RPO toward AEHF-6) is what ML can provide. Attribution (Russian operator authorized this action) requires cross-domain intelligence fusion that is not derivable from TLE data alone. The product is honest about this boundary.
- **Operator-facing output must be actionable, not probabilistic raw output.** The analyst alert format — primary assessment labeled, nearest object named, approach geometry described, action recommended, basis stated — is the interface between the ML pipeline and the operator who acts on it.

---

## Quiz

{{#quiz 04-intent-inference.toml}}
