# Module 2 Project: Approximating a Conjunction-Risk Value Function

## What you are building

In the Module 1 project, you wrote a Monte Carlo estimator for conjunction probability. It works: give it a scenario, run N samples, get Pc. The problem is it is slow. For N = 50,000 samples it takes a few seconds per evaluation. If you need to evaluate thousands of candidate maneuver decisions in real time, or use Pc as a reward signal inside an RL training loop, you cannot afford a Monte Carlo simulation for every single evaluation.

The solution is standard across all of RL and game theory: train a neural network to approximate the expensive computation. The network is fast at inference time (one forward pass, microseconds) and can be trained once offline. This is called a **value function approximator**, and it is the backbone of every deep RL algorithm in Modules 3 and 4.

In this project you will:

1. Generate a dataset of (orbital features, Pc) pairs using your Module 1 estimator
2. Train an MLP to approximate the Pc function from features
3. Evaluate how well the approximation generalizes
4. Explore what the network has learned

## The connection to later modules

Module 3 introduces DQN, where a network approximates the Q-value (expected return) for every state-action pair. The training loop you write here is identical in structure to the DQN training loop: generate data, compute targets, train a network to predict them. The only difference is where the targets come from (Monte Carlo simulation here, Bellman backups in DQN).

Module 5's deep CFR trains a network to approximate counterfactual regret values. Same structure again. Once you have the training loop working for this project, you have it for everything downstream.

## Setup: generating training data

Your Monte Carlo `estimate_pc` function from Module 1 is the data generator. You will call it many times with different orbital configurations to build a training set.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ── Scenario parameters ──────────────────────────────────────────────────────
# We will vary these across training examples:
#   - sigma: position uncertainty (km)
#   - cross_track_offset: y-component of r0_B (km), affects nominal miss distance
#   - approach_speed: relative closing speed (km/s)
# We will use these as the 3-feature input to the network.

# Nominal satellite configuration from Module 1
r0_A = torch.tensor([  0.0, 0.0, 0.0])
v_A  = torch.tensor([ 7.5, 0.0, 0.0])
DT   = 0.1
T_END = 20.0
t    = torch.arange(0.0, T_END + DT, DT)
THRESHOLD = 1.0

def estimate_pc_batch_free(cross_track_km, approach_speed_kms, sigma_km,
                            N=10_000):
    """
    Estimate Pc for a parameterized conjunction scenario.
    
    Satellite A starts at origin moving at +approach_speed/2 in x.
    Satellite B starts 100 km away with a cross_track_km y-offset,
    moving at -approach_speed/2 in x.
    """
    r0_B = torch.tensor([100.0, cross_track_km, 0.0])
    v_B  = torch.tensor([-approach_speed_kms / 2.0, 0.0, 0.0])
    v_A2 = torch.tensor([ approach_speed_kms / 2.0, 0.0, 0.0])
    
    deltas_A = sigma_km * torch.randn(N, 3)
    deltas_B = sigma_km * torch.randn(N, 3)
    r0A = r0_A + deltas_A
    r0B = r0_B + deltas_B
    
    motion_A = v_A2 * t.unsqueeze(1)
    motion_B = v_B  * t.unsqueeze(1)
    traj_A = r0A.unsqueeze(1) + motion_A
    traj_B = r0B.unsqueeze(1) + motion_B
    dists   = torch.linalg.norm(traj_A - traj_B, dim=2)
    min_d   = dists.min(dim=1).values
    return (min_d < THRESHOLD).float().mean().item()
```

## Step 1: generate the dataset

Sample many random scenarios and compute Pc for each. This is the slow step; run it once and save the results.

```python
print("Generating training data... (this takes a minute)")

N_SCENARIOS = 2000  # number of (features, Pc) pairs to generate

cross_tracks   = torch.FloatTensor(N_SCENARIOS).uniform_(0.1, 3.0)   # km
approach_speeds = torch.FloatTensor(N_SCENARIOS).uniform_(5.0, 15.0)  # km/s
sigmas         = torch.FloatTensor(N_SCENARIOS).uniform_(0.05, 0.5)   # km

features = torch.stack([cross_tracks, approach_speeds, sigmas], dim=1)
# shape: (N_SCENARIOS, 3)

labels = torch.zeros(N_SCENARIOS, 1)
for i in range(N_SCENARIOS):
    pc = estimate_pc_batch_free(
        cross_track_km    = cross_tracks[i].item(),
        approach_speed_kms= approach_speeds[i].item(),
        sigma_km          = sigmas[i].item(),
        N = 5_000  # smaller N for speed; noisier labels are fine
    )
    labels[i, 0] = pc
    if i % 200 == 0:
        print(f"  {i}/{N_SCENARIOS} scenarios computed, "
              f"last Pc = {pc:.4f}")

print(f"Done. Label range: [{labels.min():.3f}, {labels.max():.3f}]")

# Save so you do not have to regenerate
torch.save({'features': features, 'labels': labels}, 'conjunction_dataset.pt')
```

## Step 2: split and build the DataLoader

```python
# Load if already saved
data = torch.load('conjunction_dataset.pt')
features, labels = data['features'], data['labels']

# Train/validation split: 80/20
split = int(0.8 * len(features))
X_train, X_val = features[:split], features[split:]
y_train, y_val = labels[:split],   labels[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                          batch_size=256, shuffle=False)

print(f"Training: {len(X_train)} examples")
print(f"Validation: {len(X_val)} examples")
```

## Step 3: build and train the network

Your network maps 3 orbital features to a single Pc prediction. Choose an architecture; the suggestions below are starting points, not the only option.

```python
class PcPredictor(nn.Module):
    """
    Predicts conjunction probability Pc from three orbital features:
      - cross_track_km: nominal cross-track miss distance
      - approach_speed_kms: relative closing speed
      - sigma_km: position uncertainty
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # Pc is a probability: constrain output to [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)

model     = PcPredictor(hidden=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Network parameters: {sum(p.numel() for p in model.parameters())}")
```

**A note on the Sigmoid output**: Pc is a probability between 0 and 1. Adding `nn.Sigmoid()` as the final activation constrains the output to that range, which can help the network converge faster and prevents predicting negative probabilities. MSE loss still works with sigmoid outputs.

Training loop:

```python
best_val_loss = float('inf')
best_epoch    = 0

print(f"\n{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>10} | {'Val RMSE':>10}")
print("-" * 50)

for epoch in range(100):
    # Training
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        pred = model(X_b)
        loss = F.mse_loss(pred, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss  = F.mse_loss(val_preds, y_val).item()
        val_rmse  = val_loss ** 0.5
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save(model.state_dict(), 'best_pc_model.pt')
    
    if epoch % 10 == 0 or epoch == 99:
        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>10.6f} | "
              f"{val_rmse:>10.4f}")

print(f"\nBest validation MSE: {best_val_loss:.6f} at epoch {best_epoch}")
print(f"Best RMSE: {best_val_loss**0.5:.4f} (expected error in Pc units)")
```

## Step 4: evaluate the approximation

Load the best checkpoint and test it on some representative scenarios:

```python
# Load best weights
model.load_state_dict(torch.load('best_pc_model.pt'))
model.eval()

test_scenarios = [
    # (cross_track, approach_speed, sigma, description)
    (0.3, 12.0, 0.10, "High risk: small miss distance, fast, low uncertainty"),
    (0.3, 12.0, 0.50, "High risk scenario with larger uncertainty"),
    (2.5,  7.0, 0.10, "Low risk: large miss distance, slow, low uncertainty"),
    (1.0, 10.0, 0.20, "Medium risk scenario"),
]

print("\n=== Network predictions vs. Monte Carlo ground truth ===")
print(f"{'Scenario':>50} | {'Net pred':>10} | {'MC truth':>10} | {'Error':>8}")
print("-" * 90)

with torch.no_grad():
    for cross_track, speed, sigma, desc in test_scenarios:
        x = torch.tensor([[cross_track, speed, sigma]])
        net_pred = model(x).item()
        
        # Monte Carlo ground truth (expensive, high N for accuracy)
        mc_truth = estimate_pc_batch_free(cross_track, speed, sigma, N=50_000)
        
        error = abs(net_pred - mc_truth)
        print(f"{desc:>50} | {net_pred:>10.4f} | {mc_truth:>10.4f} | {error:>8.4f}")
```

A well-trained model should achieve RMSE below 0.05 (errors in Pc smaller than 5 percentage points). For very high or very low Pc values (near 0 or 1), it may need more training data in those regions.

## Step 5: speed comparison

This is the payoff. Compare inference time between Monte Carlo and the neural network:

```python
import time

x_batch = torch.tensor([[0.3, 12.0, 0.10]])  # high-risk scenario

# Monte Carlo timing
start = time.time()
for _ in range(10):
    pc_mc = estimate_pc_batch_free(0.3, 12.0, 0.10, N=10_000)
mc_time = (time.time() - start) / 10

# Neural network timing
model.eval()
start = time.time()
with torch.no_grad():
    for _ in range(10_000):
        pc_net = model(x_batch).item()
net_time = (time.time() - start) / 10_000

print(f"\nMonte Carlo (N=10,000):  {mc_time*1000:.1f} ms per evaluation")
print(f"Neural network:          {net_time*1000:.3f} ms per evaluation")
print(f"Speedup:                 {mc_time / net_time:.0f}x")
```

The neural network should be roughly 1,000 to 10,000 times faster. That speedup is what makes it practical to use as a value function inside a real-time decision loop.

## Step 6: reflect

Add a comment block to your script answering:

1. What RMSE did your best model achieve? Is that good enough for operational use?
2. Look at the error pattern: does your network do better in some regions of the input space (e.g., high Pc or low Pc) than others? Why might that be?
3. Your training labels (the Pc estimates from Monte Carlo) are noisy because they were computed with N = 5,000 samples. How does that label noise affect the network? Can the network ever be more accurate than the noise in its training labels?
4. If you wanted to improve accuracy in the low-Pc regime (Pc < 0.01), what would you change about the data generation strategy?
5. The network takes 3 features as input. What other features from the orbital mechanics would you add to make the approximation more realistic?

## What you should have at the end

A Python file or notebook containing:
- The data generation code (or a saved dataset)
- The `PcPredictor` network definition
- The training loop with validation monitoring
- The evaluation code comparing network to Monte Carlo
- The speed comparison
- Answers to the reflection questions as comments

Keep the whole thing under 300 lines. The point is not a production system; it is a clean demonstration that a neural network can approximate your Monte Carlo estimator fast enough to be useful in a decision loop.

## What comes next

Module 3 introduces Markov Decision Processes and reinforcement learning. The first algorithm (tabular Q-learning) does not use neural networks. The second (DQN) uses exactly the network you just trained: it approximates the expected cumulative reward for each action from the current state. Your `PcPredictor` architecture is structurally identical to a Q-network; the only differences are the number of outputs and what the targets represent.
