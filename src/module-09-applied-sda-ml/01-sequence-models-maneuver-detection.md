# Lesson 1: Sequence Models for Maneuver Detection

**Module:** Applied SDA ML — M09: Building Commercial SDA Products
**Source:** Space-Track GP History API (public TLE catalog); NOAA F10.7 Solar Flux (public); Kelecy & Hall (2006) "Satellite Maneuver Detection Using Two-Line Element Sets"; ESA DISCOS database event catalog; UT Austin LASR laboratory maneuver characterization papers

---


<!-- toc -->

## Where this fits

Module 2 taught you how to train neural networks. Module 3 taught you sequential decision-making. This lesson combines them into the first commercially viable product in this curriculum: a classifier that reads a satellite's TLE history and flags maneuver events.

The input is a sequence of orbital element updates over 30 calendar days, engineered into time-normalized features. The model is an LSTM — the right tool for variable-length time series with irregular cadence. The output is a binary classification: maneuver detected in this window or not.

This lesson also addresses the engineering reality that classroom ML skips: the label problem. You cannot train a maneuver detector the way you train MNIST. The clean labeled dataset does not exist, and pretending otherwise produces a model that cannot generalize. The correct solution is synthetic data generation, and this lesson builds it from scratch.

---

## Why sequences? Why not tabular?

The naive approach to maneuver detection is to treat each TLE as a row in a table and train a classifier on single-TLE features. This fails for a fundamental reason: a single TLE contains almost no information about whether a maneuver occurred.

A TLE epoch gives you: inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion, and a drag term (BSTAR). Every one of those values is consistent with thousands of different physical histories. An inclination of 51.6° and mean motion of 15.5 rev/day could be the ISS station-keeping normally, the ISS mid-reboost, or an entirely different object. Without the context of what those values were last week and the week before, you cannot tell.

What reveals a maneuver is the *trajectory* of orbital elements over time. During quiet station-keeping, mean motion drifts slowly and predictably as atmospheric drag removes energy. During an active maneuver, mean motion changes abruptly — or inclination changes slightly from a plane-change burn — in a way that breaks the quiet-background trend. The signature of a maneuver is a change in the rate of change of orbital elements, which you can only see if you have the history.

This is the temporal structure problem. A tabular model on single TLEs is structurally blind to it. A sequence model learns from the history directly.

---

## The label problem — honest treatment required

Before building the model, you need to confront the training data problem. Maneuver detection is a supervised classification task: you need examples of (TLE history, label) pairs where the label indicates whether a maneuver occurred. The difficulty is that the positive class — confirmed maneuver events — is extremely sparse in any fully public dataset.

**What is actually public and labeled:**

ISS reboosts are the cleanest positive examples. NASA publishes reboost events with approximate dates through mission status reports. The ISS executes roughly 4–8 reboosts per year. Across 25 years of orbital operations, that is on the order of 100–200 events, many with uncertain exact timing and magnitude.

Academic sources extend this modestly. Kelecy & Hall (2006) analyzed historical maneuver events in the catalog. ESA's DISCOS database records some documented maneuver events for well-tracked objects. The UT Austin LASR laboratory has published maneuver characterization studies with specific events identified. Together, these sources provide at most a few hundred confirmed maneuver events across the entire public catalog history.

**What is not public despite appearances:**

Starlink collision avoidance maneuvers are commonly cited as a potential labeled dataset. This assumption is incorrect. SpaceX publishes annual statistics on CAM frequency across the fleet, but not per-event timestamps, magnitudes, or object identifiers. There is no public Starlink CAM log. Building a training set from this source is not feasible.

**Why this is a serious problem:**

A few hundred positive examples is not enough to train a generalizable LSTM. Even with heavy augmentation, the model will overfit to the specific orbital characteristics (altitude, inclination, maneuver magnitude) of the handful of real examples you have. Generalization to unknown objects with different characteristics will be poor.

**The solution: synthetic label generation**

The correct strategy for this problem class — where real positive labels are scarce but the data-generating process is well-understood — is synthetic data generation. The procedure:

1. Take a TLE history for a known non-maneuvering object (debris, dead satellite, or an active satellite in a quiet station-keeping period you can verify).
2. At a randomly chosen epoch within that history, inject a synthetic maneuver: perturb mean motion by a factor corresponding to a plausible delta-V.
3. Propagate the effect of that maneuver forward through the subsequent TLE history, updating mean motion (and indirectly semi-major axis) consistently.
4. Label the window containing the injection epoch as positive.
5. Label all windows from the unmodified history as negative.

This gives you unlimited labeled positive examples with known ground truth. The model learns to detect the *signature* of a maneuver — an anomalous change in orbital element rates — regardless of the exact magnitude or object identity. The real labeled ISS/DISCOS events become the test set, reserved exclusively for evaluating generalization.

---

## Feature engineering for orbital sequences

The quality of an LSTM maneuver detector depends heavily on feature choice. The wrong features add noise with no signal; worse, they add systematic biases that cause the model to fire on non-maneuver events.

### What to use

Extract the following from each TLE epoch, then compute time-normalized rates between consecutive TLEs:

**Mean motion rate: Δn/Δt (rev/min/day)**

Mean motion \(n\) is the orbital angular rate in revolutions per minute. It is inversely related to semi-major axis: higher orbit means lower mean motion. A prograde burn raises the orbit and decreases mean motion; a retrograde burn lowers the orbit and increases it. The rate Δn/Δt is the primary maneuver signal for in-plane burns.

```python
delta_n_per_day = (n_t2 - n_t1) / delta_t_days
```

**Eccentricity rate: Δe/Δt (per day)**

Most LEO station-keeping satellites maintain near-circular orbits. Eccentricity changes can indicate burns that are not perfectly tangential, or deliberate eccentricity management. The rate Δe/Δt is a secondary signal.

**Inclination rate: Δi/Δt (degrees/day)**

Inclination changes require enormous delta-V (the vector must rotate the orbital plane). Small inclination changes from TLE-to-TLE are mostly OD fitting noise for most objects. Large, sudden inclination changes are rare and indicate plane-change burns. The rate Δi/Δt is a weak signal for most objects but important for GEO synchronous satellites that execute north-south station-keeping.

**RAAN residual rate: ΔΩ_residual/Δt (degrees/day)**

RAAN (right ascension of the ascending node) drifts secularly due to J2 perturbations at a rate that depends on inclination and semi-major axis:

\[ \dot{\Omega}_{J2} = -\frac{3}{2} \frac{n J_2 R_E^2}{a^2 (1-e^2)^2} \cos i \]

At ISS altitude (approximately 400 km, 51.6° inclination), this drift is roughly −6.75°/day. This is not a maneuver signal — it is deterministic physics. If you use raw RAAN in your features, the secular drift dominates every other signal and your model learns nothing about maneuvers.

The correct feature is the RAAN residual: actual RAAN minus the predicted RAAN from J2 propagation. The residual captures anomalous RAAN changes that are not explained by J2, which can indicate maneuvers or solar radiation pressure effects.

```python
omega_j2_dot = -1.5 * n * J2 * RE**2 / (a**2 * (1-e**2)**2) * np.cos(i_rad)
omega_predicted = omega_t1 + omega_j2_dot * delta_t_days
omega_residual = omega_t2 - omega_predicted
delta_omega_residual_per_day = omega_residual / delta_t_days
```

**Observation gap: Δt (hours)**

The time since the previous TLE is informative. A gap of 72 hours means you missed two days of coverage — anything could have happened. A gap of 3 hours means continuous tracking. The observation gap should be included as an explicit feature so the model can discount uncertain transitions.

**Solar flux index: F10.7**

F10.7 is the 10.7 cm solar radio flux, a proxy for solar ultraviolet output, which drives thermospheric heating and atmospheric expansion. During high F10.7 periods, atmospheric drag increases, causing mean motion changes in LEO that are entirely physical and not maneuver signals. A model trained without F10.7 will have high false positive rates during solar maximum.

F10.7 data is freely available from NOAA. Fetch the daily index for each TLE epoch and include it as a feature alongside the orbital element rates.

**Object class embedding**

Include a learned embedding for object class (active satellite, debris, rocket body) as a categorical feature. Different object classes have systematically different quiet-background behaviors. Including class prevents the model from confusing debris drag decay with a satellite maneuver.

### What not to use

**Mean anomaly rate (ΔM/Δt)**

Do not include this. Mean anomaly changes by approximately 360° per orbit by definition — the rate is dominated by mean motion itself. For a 90-minute LEO orbit, mean anomaly completes one full revolution every 90 minutes. The rate contains no information about maneuvers beyond what mean motion already provides, and the numerical values are large and noisy.

**Raw argument of perigee (ω)**

Argument of perigee is geometrically ill-defined for near-circular orbits (eccentricity below approximately 0.01). The coordinate singularity causes wild numerical jumps in ω even for physically quiet objects. Most LEO satellites have eccentricities in the range 0.0001–0.001. For these objects, ω is meaningless. Do not include raw ω. If eccentricity is significant (GEO transfer orbits, Molniya orbits), you can include ω with a check on eccentricity magnitude.

**Raw RAAN (Ω)**

J2 drift dominates. The secular drift rate at ISS altitude is approximately 2,000× larger than a typical maneuver-driven RAAN change. If you include raw RAAN the model learns to respond to J2-driven drift, not maneuvers. Always subtract predicted J2 drift before computing RAAN features.

**Raw element deltas (Δa, Δe between consecutive TLEs)**

Raw deltas without time normalization are not physically meaningful. A semi-major axis change of 500 meters over 6 hours represents a significant maneuver (approximately 0.3 m/s delta-V). The same 500-meter change over 72 hours is consistent with normal atmospheric drag. The same number means different things depending on observation cadence. Always divide by Δt to compute a rate. TLE publication cadence is irregular enough that this matters constantly.

**BSTAR as a primary feature**

BSTAR deserves special treatment because it is widely misunderstood. BSTAR is not a direct measurement of atmospheric drag. It is an estimation artifact: the orbit determination fitting process uses BSTAR as a free parameter to absorb all unmodeled forces, including solar radiation pressure, atmospheric density uncertainty, and any other perturbation not in the propagation model. Consequences:

- BSTAR can be negative, which is physically impossible for atmospheric drag. Negative BSTAR values appear frequently for high-altitude objects where solar radiation pressure dominates.
- Many catalog objects have BSTAR set to a default value (0.21109e-4) because the OD process did not produce a meaningful fit. These objects' BSTAR values are identical and meaningless.
- BSTAR varies with solar activity in ways that look like signal but are not maneuver-related.

Include BSTAR as a feature but treat it cautiously: use it as a consistency check, normalize it, and do not expect it to be the primary discriminator.

### Feature vector summary

Each daily grid point contributes one feature vector of dimension 8:

```
[Δn/Δt, Δe/Δt, Δi/Δt, ΔΩ_residual/Δt, BSTAR, F10.7, Δt_hours, object_class_embed]
```

The object class embedding has a learned dimension (typically 4), bringing the total to 11 if the embedding is concatenated directly. The full input to the LSTM is a sequence of 30 such vectors, one per day in the window.

---

## Object class stratification

Never train a single model on all object types jointly without stratification. The quiet-background behavior differs systematically across object classes, and conflating them produces a model that is poorly calibrated for all of them.

**Rocket bodies**

Rocket bodies are the most problematic class for maneuver detection. They have high area-to-mass ratios relative to operational satellites — their large cylindrical cross-sections (typically 2–4 meters diameter, 5–10 meters long) combined with relatively low structural mass mean solar radiation pressure moves them significantly. This creates oscillations in orbital elements that are periodic with the orbital and seasonal illumination geometry. These oscillations look like maneuvers to a naive model: mean motion oscillates, eccentricity oscillates, RAAN residuals oscillate. None of these are maneuvers.

Additionally, rocket bodies do not maneuver. Training a maneuver detector that includes rocket bodies as a potential positive class adds false complexity. The right treatment is either to exclude rocket bodies from the training set entirely, or to include them only as negative examples (with the model learning that rocket bodies with large element oscillations are still not maneuvering).

In the Space-Track catalog, rocket bodies are identifiable from the international designator suffix (-R for rocket body) or the object type field in the GP records.

**Compact debris**

Small debris objects have predictable drag decay and no maneuver capability. Their element histories are quiet with slow secular trends. They provide excellent negative training examples: a debris TLE history with a synthetic maneuver injected is a clean positive example, because the injection stands out against the quiet background.

**Active satellites**

Active satellites have station-keeping patterns that produce small, periodic element changes. The model needs to distinguish station-keeping burns (small, regular) from anomalous maneuvers (larger, irregular, or outside the expected station-keeping band). Including object class as a feature helps here: a station-keeping event for an active satellite is a negative example (expected behavior); the same element change for an object with no station-keeping history is a positive example.

---

## Handling irregular TLE cadence

TLE publication cadence is not uniform across the catalog, and within a single object's history it varies over time. The ISS receives 4–8 TLEs per day during active periods and may have gaps during ground station outages. Active LEO commercial satellites typically receive 1–4 TLEs per day. Low-priority debris objects may receive one TLE every several days or less.

This means "30 TLEs" does not mean "30 days." A naive approach of taking the last 30 TLEs as the input sequence produces windows of wildly different calendar durations, making the LSTM's temporal structure meaningless.

The correct approach is calendar-time windowing with grid alignment:

1. Choose a window length in calendar time: 30 days.
2. Define a regular grid: one observation per day.
3. For each grid day, select the TLE whose epoch is closest to noon that day.
4. If no TLE exists within a tolerance (say, ±12 hours) of a grid point, mark that grid position as missing.
5. Include an observation mask alongside the feature vector: 1 for observed grid points, 0 for missing ones.

The LSTM receives both the feature sequence and the observation mask. On missing days, the feature vector is set to zeros (or the most recent observed value — either works, the mask tells the model which entries to discount).

```python
def align_to_grid(tle_records: list[dict], window_days: int = 30) -> tuple:
    """
    Given a list of TLE records with epoch timestamps, align to a daily grid.
    Returns (feature_matrix, obs_mask) each of shape (window_days, n_features).
    """
    grid_features = np.zeros((window_days, N_FEATURES))
    obs_mask      = np.zeros(window_days)

    # Find the window end date from the most recent TLE
    end_epoch = max(r['epoch'] for r in tle_records)
    start_epoch = end_epoch - timedelta(days=window_days)

    for day_idx in range(window_days):
        grid_time = start_epoch + timedelta(days=day_idx) + timedelta(hours=12)
        # Find closest TLE
        closest = min(tle_records,
                      key=lambda r: abs((r['epoch'] - grid_time).total_seconds()))
        gap_hours = abs((closest['epoch'] - grid_time).total_seconds()) / 3600
        if gap_hours < 12.0:
            grid_features[day_idx] = closest['features']
            obs_mask[day_idx] = 1.0
        # else: leave zeros, mask stays 0

    return grid_features, obs_mask
```

---

## LSTM architecture

With the feature engineering established, the model architecture is straightforward. The right choice here is LSTM, not Transformer. Module 2 explicitly excluded attention mechanisms, and for this application they are unnecessary overhead: the sequences are short (30 timesteps), the dataset is modest, and an LSTM with 64 hidden units is a tractable, debuggable baseline that you can train on a laptop.

```
Input:  sequence of (feature_vector || obs_mask_bit) pairs
        Shape: (batch, 30, 9)   # 8 features + 1 mask
LSTM:   hidden_size=64, num_layers=1
Output of LSTM final hidden state: (batch, 64)
Linear: (64, 2)
Softmax → (batch, 2)   # [P(no maneuver), P(maneuver)]
```

The observation mask is concatenated to the feature vector as an additional input dimension rather than handled separately. This lets the LSTM learn directly that timesteps with mask=0 should receive different weighting than timesteps with mask=1.

### Full PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TLEWindowDataset(Dataset):
    """
    Each sample is a 30-day window of daily-gridded TLE features.
    Positive samples have a synthetic maneuver injected at a random day
    in [10, 20] of the window.  Negative samples are clean backgrounds.

    Args:
        windows:    np.ndarray of shape (N, 30, 9)
                    Last column is the observation mask (0/1).
        labels:     np.ndarray of shape (N,) with values 0 or 1.
        maneuver_day: np.ndarray of shape (N,) giving the injection day
                    for positive samples (-1 for negatives).
    """
    def __init__(
        self,
        windows:      np.ndarray,
        labels:       np.ndarray,
        maneuver_day: Optional[np.ndarray] = None,
    ):
        self.windows      = torch.tensor(windows,      dtype=torch.float32)
        self.labels       = torch.tensor(labels,       dtype=torch.long)
        self.maneuver_day = maneuver_day

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

J2 = 1.08263e-3    # J2 zonal harmonic coefficient
RE = 6378.137      # km, Earth equatorial radius

def compute_j2_raan_rate(n_rev_per_min: float, a_km: float,
                          e: float, i_deg: float) -> float:
    """
    Returns J2-induced RAAN drift rate in degrees/day.
    n in rev/min, a in km, e dimensionless, i in degrees.
    """
    n_rad_per_sec = n_rev_per_min * 2 * np.pi / 60.0
    i_rad = np.radians(i_deg)
    # Secular J2 RAAN rate (rad/s)
    raan_dot = (
        -1.5 * n_rad_per_sec * J2 * (RE / a_km)**2
        / (1 - e**2)**2
        * np.cos(i_rad)
    )
    # Convert to degrees/day
    return np.degrees(raan_dot) * 86400

def mean_motion_to_sma(n_rev_per_day: float) -> float:
    """
    Convert mean motion (rev/day) to semi-major axis (km) via Kepler's third law.
    GM = 398600.4418 km^3/s^2.
    """
    GM = 398600.4418
    n_rad_per_sec = n_rev_per_day * 2 * np.pi / 86400.0
    return (GM / n_rad_per_sec**2) ** (1.0 / 3.0)

def build_feature_vector(rec_prev: dict, rec_curr: dict, f107: float) -> np.ndarray:
    """
    Compute time-normalized delta features between two consecutive TLE records.
    Each record has keys: epoch (datetime), n (rev/min), e, i_deg, raan_deg, bstar.
    Returns feature vector of length 8.
    """
    dt_days = (rec_curr['epoch'] - rec_prev['epoch']).total_seconds() / 86400.0
    if dt_days < 1e-6:
        return None  # duplicate epoch, skip

    # Mean motion rate (rev/min per day)
    dn_dt = (rec_curr['n'] - rec_prev['n']) / dt_days

    # Eccentricity rate (per day)
    de_dt = (rec_curr['e'] - rec_prev['e']) / dt_days

    # Inclination rate (deg/day)
    di_dt = (rec_curr['i_deg'] - rec_prev['i_deg']) / dt_days

    # RAAN residual rate (deg/day): remove predicted J2 drift
    n_rev_per_day = rec_prev['n'] * 60 * 24     # convert rev/min -> rev/day
    a_km = mean_motion_to_sma(n_rev_per_day)
    j2_rate = compute_j2_raan_rate(
        rec_prev['n'], a_km, rec_prev['e'], rec_prev['i_deg']
    )
    raan_predicted = rec_prev['raan_deg'] + j2_rate * dt_days
    raan_residual  = rec_curr['raan_deg'] - raan_predicted
    # Wrap to [-180, 180]
    raan_residual  = (raan_residual + 180) % 360 - 180
    draan_dt = raan_residual / dt_days

    dt_hours = dt_days * 24.0

    return np.array([
        dn_dt,
        de_dt,
        di_dt,
        draan_dt,
        rec_curr['bstar'],
        f107,
        dt_hours,
        0.0,   # placeholder for object_class (set at window level)
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic maneuver injection
# ---------------------------------------------------------------------------

def inject_maneuver(
    tle_records: list[dict],
    inject_day:  int,
    delta_n_fraction: float = 0.0005,  # fraction of mean motion to add
) -> list[dict]:
    """
    Inject a synthetic maneuver into a TLE history.
    At inject_day, multiply mean motion by (1 + delta_n_fraction).
    All subsequent TLEs are shifted by the same delta_n to preserve consistency.

    A delta_n_fraction of 0.0005 corresponds to roughly a 5 m/s delta-V at
    ISS altitude.  Vary this over [0.0001, 0.002] during training to get
    maneuvers of different sizes.

    Args:
        tle_records:     list of TLE dicts, sorted by epoch, on a daily grid
        inject_day:      index into tle_records where maneuver occurs
        delta_n_fraction: fractional change to apply to mean motion
    Returns:
        modified copy of tle_records
    """
    import copy
    records = copy.deepcopy(tle_records)
    n_at_injection = records[inject_day]['n']
    delta_n = n_at_injection * delta_n_fraction

    for idx in range(inject_day, len(records)):
        records[idx]['n'] += delta_n

    return records

def generate_synthetic_dataset(
    background_histories: list[list[dict]],  # list of clean TLE histories
    n_positive: int = 5000,
    n_negative: int = 5000,
    window_days: int = 30,
    f107_lookup: dict = None,  # date -> F10.7 value
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced dataset of positive (maneuver injected) and negative
    (clean background) windows.

    Returns:
        windows: (n_positive + n_negative, window_days, 9)
        labels:  (n_positive + n_negative,)
    """
    windows_list = []
    labels_list  = []

    rng = np.random.default_rng(seed=42)

    def history_to_features(records, obj_class_idx):
        """Convert a daily-gridded TLE history to a (window_days, 9) array."""
        feat_seq  = np.zeros((window_days, 8), dtype=np.float32)
        mask_seq  = np.zeros((window_days, 1), dtype=np.float32)
        for day in range(1, window_days):
            f107 = (f107_lookup.get(records[day]['epoch'].date(), 150.0)
                    if f107_lookup else 150.0)
            fvec = build_feature_vector(records[day-1], records[day], f107)
            if fvec is not None:
                fvec[7] = float(obj_class_idx)
                feat_seq[day] = fvec
                mask_seq[day] = 1.0
        return np.concatenate([feat_seq, mask_seq], axis=1)  # (30, 9)

    # Generate negative examples
    for _ in range(n_negative):
        hist = background_histories[rng.integers(len(background_histories))]
        # Random 30-day slice from a longer history
        if len(hist) < window_days:
            continue
        start = rng.integers(0, len(hist) - window_days)
        window_records = hist[start:start + window_days]
        obj_class = hist[0].get('obj_class', 1)  # 1 = debris
        windows_list.append(history_to_features(window_records, obj_class))
        labels_list.append(0)

    # Generate positive examples
    for _ in range(n_positive):
        hist = background_histories[rng.integers(len(background_histories))]
        if len(hist) < window_days:
            continue
        start = rng.integers(0, len(hist) - window_days)
        window_records = hist[start:start + window_days]

        # Inject maneuver at random day between day 10 and 20
        inject_day = int(rng.integers(10, 20))
        delta_frac = float(rng.uniform(0.0001, 0.002))
        window_records = inject_maneuver(window_records, inject_day, delta_frac)

        obj_class = hist[0].get('obj_class', 1)
        windows_list.append(history_to_features(window_records, obj_class))
        labels_list.append(1)

    windows = np.stack(windows_list)
    labels  = np.array(labels_list, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(labels))
    return windows[perm], labels[perm]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ManeuverLSTM(nn.Module):
    """
    LSTM-based maneuver detector.
    Input: (batch, seq_len, input_size)
    Output: (batch, 2) logits for [no_maneuver, maneuver]

    Architecture:
        LSTM(input_size, hidden_size) → take final hidden state →
        Linear(hidden_size, 2)

    The observation mask is included as the last feature channel.
    The LSTM sees the full sequence; the mask allows it to weight
    high-confidence timesteps appropriately.
    """
    def __init__(
        self,
        input_size:  int   = 9,   # 8 features + 1 obs mask
        hidden_size: int   = 64,
        num_layers:  int   = 1,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n:      (num_layers, batch, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use the final hidden state of the last layer
        last_hidden = h_n[-1]                 # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden) # (batch, 2)
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      ManeuverLSTM,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for windows, labels in loader:
        windows = windows.to(device)
        labels  = labels.to(device)
        optimizer.zero_grad()
        logits = model(windows)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:   ManeuverLSTM,
    loader:  DataLoader,
    device:  torch.device,
) -> dict:
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for windows, labels in loader:
        windows = windows.to(device)
        logits  = model(windows)
        probs   = F.softmax(logits, dim=1)[:, 1]  # P(maneuver)
        preds   = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)

    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / len(labels_arr)

    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'accuracy':  accuracy,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def train_maneuver_detector(
    train_windows: np.ndarray,
    train_labels:  np.ndarray,
    val_windows:   np.ndarray,
    val_labels:    np.ndarray,
    n_epochs:      int   = 30,
    batch_size:    int   = 64,
    lr:            float = 1e-3,
    pos_weight:    float = 100.0,
    device_str:    str   = 'cpu',
) -> ManeuverLSTM:
    """
    Train the maneuver detector LSTM.

    pos_weight=100 addresses the class imbalance in real deployment:
    for every real maneuver window, there are roughly 100 quiet windows
    in a typical catalog monitoring scenario.

    Args:
        train_windows: (N_train, 30, 9)
        train_labels:  (N_train,)
        val_windows:   (N_val, 30, 9)
        val_labels:    (N_val,)
    Returns:
        trained ManeuverLSTM
    """
    device = torch.device(device_str)
    model  = ManeuverLSTM().to(device)

    train_ds = TLEWindowDataset(train_windows, train_labels)
    val_ds   = TLEWindowDataset(val_windows,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Weighted cross-entropy: downweight the majority negative class
    weight = torch.tensor([1.0, pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_f1    = 0.0
    best_state = None

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics['f1'])

        if val_metrics['f1'] > best_f1:
            best_f1    = val_metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:>3} | loss={train_loss:.4f} | "
                f"val_f1={val_metrics['f1']:.3f} | "
                f"prec={val_metrics['precision']:.3f} | "
                f"rec={val_metrics['recall']:.3f}"
            )

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nTraining complete. Best val F1: {best_f1:.3f}")
    return model
```

---

## Operational evaluation metrics

Standard precision and recall on a balanced test set are necessary but not sufficient for a commercial maneuver detection product. A space operations customer does not want to know your F1 score on a held-out test set; they want to know whether the product will interrupt their analysts at 3 AM with false alerts, and whether it will catch the maneuvers that matter.

**Detection latency**

How many days after the maneuver epoch does the model first flag the window as positive? A detection latency of 1 day means the model catches the maneuver in the first window that covers the event. A latency of 5 days means the operator is notified 5 days late, which may be operationally irrelevant or disqualifying depending on the mission.

To measure this: for each real maneuver event in the test set, find the earliest window ending that produces a positive prediction. The detection latency is the number of days between the maneuver epoch and that window end. Target: latency less than 3 days for Δv greater than 5 m/s.

**False alarm rate per object per month**

Space operations analysts cannot tolerate a product that generates constant alerts. The acceptable false positive rate is approximately 1–2 false alerts per object per month for an actively monitored catalog. Higher than that and analysts will stop trusting the system.

To measure this: run the trained model on 90 days of clean TLE history for confirmed non-maneuvering objects. Count positive predictions. Divide by object-months of monitoring. This number should be below 2.0 for your product to be operationally credible.

**Miss rate by maneuver size**

Maneuvers below a certain Δv threshold produce TLE-visible changes below the TLE noise floor, and the model simply cannot detect them. This is a fundamental limitation of TLE-based detection, not a model deficiency — but it must be characterized and communicated honestly.

To measure this: in your synthetic test set, stratify positive examples by injected delta_n_fraction (which corresponds to Δv). Compute recall separately for small (delta_frac < 0.0002, approximately Δv < 2 m/s), medium (0.0002–0.001, 2–10 m/s), and large (> 0.001, > 10 m/s) maneuvers. You will find near-zero recall for small maneuvers and high recall for large ones. Report all three.

---

## Commercial framing — honest

TLE-based maneuver detection occupies a specific and honest tier in the commercial SDA market. Understanding where it fits and where it does not is as important as building the model.

**What it can do:**

Running entirely on free public data from Space-Track, the product monitors any object in the public catalog without any sensor contract. The unit economics are favorable: compute costs dominate, and a well-optimized pipeline can monitor thousands of objects on a single machine. Maneuvers of Δv approximately 5–10 m/s or larger in LEO produce TLE-visible changes that the model can detect with reasonable reliability. For operators who need to know whether a monitored asset has executed a significant maneuver, at a price point that radar-based services cannot approach, this is a viable product.

**What it cannot do:**

TLEs are published with latency of hours to days, so the product is not real-time. Small burns (Δv below approximately 5 m/s) may not produce detectable TLE changes, particularly in the presence of high atmospheric drag noise during solar maximum. The product has no inherent ability to infer intent from a detected maneuver: it can say "this object changed orbits" but not "this object is executing a rendezvous with your asset." Position accuracy is limited to TLE propagation fidelity (hundreds of meters to kilometers), not the centimeter-scale precision of modern radar or optical networks.

**The competitive landscape:**

LeoLabs operates a global phased-array radar network that provides radar-derived orbital solutions with uncertainty covariances far tighter than TLE-derived positions. Their maneuver detection is based on comparing consecutive high-precision orbital solutions, not TLE history. Slingshot Aerospace provides analyst tooling that includes maneuver assessment from multiple sensor inputs. ExoAnalytic Solutions specializes in GEO optical tracking with high temporal resolution. These are the dominant radar/optical-based services, and they compete on precision.

A TLE-only product does not compete head-to-head with these services. It competes at a different price point — one accessible to smaller operators, academic institutions, and early-stage commercial satellite operators who need reasonable maneuver awareness without a multi-hundred-thousand-dollar radar data contract.

The genuine differentiator available to a solo uncleared founder is integration: combining this maneuver detection module with the game-theoretic adversary modeling from Modules 5–8. No commercial product currently integrates "this object maneuvered" with "given the orbital geometry, this maneuver is consistent with a rendezvous approach profile." That inference requires game-theoretic reasoning about intent, not just anomaly detection. The maneuver detector built in this module is the sensor front-end for that larger product.

---

## Key Takeaways

- **Temporal structure is the reason to use an LSTM.** A single TLE gives almost no information about maneuver history. The sequence of TLE epochs over 30 calendar days reveals rates of change in orbital elements — the signature of a maneuver against the quiet background trend. Tabular ML on single TLEs is structurally blind to this.
- **The label problem requires synthetic data generation.** Real confirmed maneuver events number in the hundreds across the entire public catalog history — too few for training a generalizable model. Inject synthetic maneuvers into clean debris TLE histories to produce unlimited positive labels. Reserve real events (ISS reboosts, DISCOS-documented events) for the test set only.
- **Time-normalize all delta features.** A raw orbital element delta is not physically meaningful without dividing by the observation gap Δt. The same change over 6 hours and 72 hours have completely different interpretations. Always compute rates (Δn/Δt, Δe/Δt, Δi/Δt) not raw differences.
- **Remove secular J2 drift from RAAN before computing features.** The J2-driven RAAN drift (~−6.75°/day at ISS altitude) is 2000× larger than maneuver-driven changes. Including raw RAAN teaches the model to detect J2 perturbations, not maneuvers. Compute the RAAN residual after subtracting predicted J2 drift.
- **Exclude rocket bodies from maneuver training or treat separately.** High area-to-mass ratios cause solar radiation pressure oscillations in orbital elements that mimic maneuver signatures. A model trained on mixed-class data will have unacceptably high false positive rates for rocket bodies.
- **Use calendar-time windows, not TLE-count windows.** TLE publication cadence is irregular. "30 TLEs" ranges from 4 days to 90 days depending on the object. Grid to daily resolution and include an observation mask for missing days.
- **Operational metrics matter more than test-set F1.** Precision and recall on a balanced test set are necessary but insufficient for a commercial product. Measure detection latency (days after maneuver), false alarm rate per object per month, and miss rate by maneuver size. These are the metrics that determine whether an operator will pay for the product.
- **TLE-based detection is the low-cost tier.** It cannot compete with radar-based services on precision or latency. It competes on cost and accessibility. The genuine differentiator is integration with game-theoretic intent inference — the connection to Modules 5–8 that no current commercial product provides.

---

## Quiz

{{#quiz 01-sequence-models-maneuver-detection.toml}}
