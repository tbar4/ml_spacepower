# Lesson 4: The Training Loop

## Where this fits

You have a network (lesson 2), a loss function (lesson 3), and gradient descent (Module 1, lesson 7). The training loop is how they combine into an actual learning algorithm. This lesson is mostly mechanical, but it is machinery you will run in every subsequent module. Module 3's DQN agent, Module 4's AlphaZero, and Module 5's deep CFR all execute a training loop at their core. The outer loop changes; the inner loop (forward, loss, backward, step) stays the same.

## The complete training loop, piece by piece

Here is a minimal complete training loop annotated in detail:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ── 1. Build the network ──────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

# ── 2. Choose an optimizer ────────────────────────────────────────────────────
# Adam is the standard choice. lr is the learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── 3. Wrap data in a DataLoader for automatic batching and shuffling ─────────
# X: features, shape (N, 4); y: targets, shape (N, 1)
# (Assume X_train and y_train already exist as tensors)
dataset    = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ── 4. Training loop ──────────────────────────────────────────────────────────
num_epochs = 50  # number of complete passes through the data

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    for X_batch, y_batch in dataloader:   # iterate over batches
        
        # ── 4a. Zero the gradients from the previous batch ──────────────────
        optimizer.zero_grad()
        # Without this, gradients accumulate across iterations.
        
        # ── 4b. Forward pass: compute predictions ──────────────────────────
        y_pred = model(X_batch)
        
        # ── 4c. Compute the loss ────────────────────────────────────────────
        loss = F.mse_loss(y_pred, y_batch)
        
        # ── 4d. Backward pass: compute gradients via chain rule ─────────────
        loss.backward()
        # After this, every parameter p has p.grad filled with
        # the gradient of the loss with respect to p.
        
        # ── 4e. Update parameters ────────────────────────────────────────────
        optimizer.step()
        # Adjusts each parameter using its gradient and the learning rate.
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3}: avg loss = {avg_loss:.6f}")
```

That is the complete loop. The rest of this lesson unpacks each piece.

## Piece 1: The optimizer

In Module 1, lesson 7, we manually updated parameters with `x -= learning_rate * x.grad`. The optimizer automates this and often does it more cleverly.

**SGD** (Stochastic Gradient Descent): the simplest optimizer. Each parameter is updated by `param -= lr * param.grad`. We did this manually in lesson 7.

**Adam** (Adaptive Moment Estimation): the practical default for most problems. Adam keeps a running average of recent gradients and a running average of recent squared gradients. It uses these to scale the learning rate adaptively for each parameter. Parameters that get consistent gradients in the same direction get larger effective steps. Parameters with noisy gradients get smaller steps.

For our purposes: use Adam with `lr=1e-3` as a starting point. If training is unstable (loss oscillates wildly), reduce the learning rate. If training is too slow, you can try increasing it.

```python
# Both are valid; Adam usually works better out of the box
optimizer_sgd  = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## Piece 2: DataLoader and batching

You rarely train on one example at a time (too slow) or the entire dataset at once (too memory-intensive and the gradient estimates are noisier). Batches of 32 to 256 examples are standard.

`DataLoader` handles:
- Splitting data into batches of size `batch_size`
- Shuffling the data before each epoch (so the network does not memorize the order)
- Iterating over batches in a for loop

```python
from torch.utils.data import TensorDataset, DataLoader

# TensorDataset pairs features with labels
dataset = TensorDataset(X_train, y_train)

# DataLoader creates an iterable that returns batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Each iteration of the for loop gives you one batch
for X_batch, y_batch in dataloader:
    print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
    break  # just to see the shapes
```

**One epoch** = one complete pass through the entire training dataset.
**One step** = one forward + backward + optimizer update on one batch.

If you have 1,000 examples and batch size 32, you have about 31 steps per epoch (1000 / 32 ≈ 31, with the last batch possibly smaller).

## Piece 3: Training versus evaluation mode

Some network components (like Dropout, which randomly zeroes activations during training to prevent overfitting) behave differently during training and evaluation. PyTorch uses `model.train()` and `model.eval()` to switch modes.

```python
# During training
model.train()
for X_batch, y_batch in train_loader:
    # ... training step ...

# When evaluating on validation data
model.eval()
with torch.no_grad():  # disable gradient tracking for efficiency
    y_val_pred = model(X_val)
    val_loss = F.mse_loss(y_val_pred, y_val)
```

`torch.no_grad()` tells PyTorch not to build the computational graph during evaluation. This saves memory and computation, since you are not going to call `.backward()` on evaluation predictions.

## Overfitting and validation

Here is the core tension in machine learning: you want the network to generalize to new examples it has never seen, not just memorize the training data.

**Overfitting** happens when the network's loss on the training data keeps decreasing but its loss on new examples (the validation set) stops decreasing or starts increasing. The network has learned the training examples too specifically.

The solution: hold out a portion of your data as a **validation set**. Monitor the validation loss alongside the training loss. Stop training (or reduce the learning rate) when the validation loss stops improving.

```python
# Split data: 80% training, 20% validation
N = len(X)
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
```

A typical learning curve looks like:

```
Epoch  1: train_loss=0.850, val_loss=0.870
Epoch 10: train_loss=0.120, val_loss=0.135
Epoch 20: train_loss=0.050, val_loss=0.080
Epoch 30: train_loss=0.030, val_loss=0.078  ← val loss plateaus
Epoch 40: train_loss=0.020, val_loss=0.085  ← val loss starts increasing: overfit
Epoch 50: train_loss=0.015, val_loss=0.098  ← definitely overfit
```

You would stop training around epoch 30, when the validation loss was lowest.

## A complete training example on SSA data

Let us put everything together. We will synthetically generate conjunction feature data with known risk scores, train a network to predict them, and evaluate on a held-out validation set.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ── Generate synthetic data ───────────────────────────────────────────────────
# Features: [approach_speed, miss_distance, alert_confidence, time_to_tca]
# True risk: a nonlinear function of the features
# (In Module 1's project, you would use your Monte Carlo Pc estimates here)

N = 2000
X = torch.rand(N, 4)
X[:, 0] *= 15.0   # approach speed: 0-15 km/s
X[:, 1] *= 5.0    # miss distance: 0-5 km
X[:, 2]           # alert confidence: 0-1
X[:, 3] *= 24.0   # time to TCA: 0-24 hours

# True risk: high speed + small miss distance = high risk
# (This is the ground truth our network will learn to approximate)
true_risk = torch.sigmoid(
    0.5 * X[:, 0]    # approach speed increases risk
    - 2.0 * X[:, 1]  # miss distance decreases risk
    + 0.3 * X[:, 2]  # confidence slightly increases risk
    - 0.1 * X[:, 3]  # more time to TCA slightly decreases risk
    - 2.0            # baseline shift
)
y = true_risk.unsqueeze(1)  # shape (N, 1)

# ── Split into train and validation ──────────────────────────────────────────
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# ── Build the network ─────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10}")
print("-" * 36)

for epoch in range(60):
    # --- Training phase ---
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = F.mse_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # --- Validation phase ---
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = F.mse_loss(val_pred, y_val).item()
    
    if epoch % 10 == 0 or epoch == 59:
        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>10.6f}")

# ── Test on a specific conjunction scenario ───────────────────────────────────
model.eval()
with torch.no_grad():
    # A high-risk conjunction: fast approach, small miss distance
    high_risk_example = torch.tensor([[12.0, 0.3, 0.9, 2.0]])
    high_risk_pred = model(high_risk_example).item()
    
    # A low-risk conjunction: slow approach, large miss distance
    low_risk_example = torch.tensor([[2.0, 4.5, 0.5, 20.0]])
    low_risk_pred = model(low_risk_example).item()
    
    print(f"\nHigh-risk scenario: predicted risk = {high_risk_pred:.4f}")
    print(f"Low-risk scenario:  predicted risk = {low_risk_pred:.4f}")
```

After 60 epochs, the high-risk prediction should be substantially higher than the low-risk prediction. The network has learned the relationship between features and risk.

## What training is actually doing

Underneath the loop, gradient descent is navigating a high-dimensional surface. For our network with ~4,000 parameters, the loss is a surface in a 4,000-dimensional space. Each parameter is one axis. The optimizer is trying to roll a ball downhill in this space.

A few things worth knowing:

**Why does it sometimes get stuck?** The loss surface has many local minima (places where the gradient is zero but the loss is not the global minimum) and saddle points (where the gradient is zero in some directions but not others). In practice, for neural networks of the sizes we use, local minima are usually good enough. Saddle points can slow training down.

**Why does the validation loss sometimes spike?** A particularly unlucky batch can push weights in the wrong direction temporarily. This is normal. Over many epochs, the trend should be downward.

**When should you stop?** When validation loss has not improved for several epochs. This is called "early stopping." For our purposes, training for a fixed number of epochs and picking the checkpoint with the best validation loss is a simple and reliable strategy.

## Quiz

{{#quiz 04-the-training-loop.toml}}
