# Lesson 4: The Training Loop


<!-- toc -->

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

## Learning rate schedules

The learning rate you set at the start of training is not necessarily the best learning rate throughout. Early in training, you want large steps to escape the random initialization region quickly. Late in training, you want small steps to converge precisely to a good minimum rather than oscillating around it.

### Fixed learning rate: the baseline

The simplest approach. Set `lr=1e-3` and leave it there. Works fine for many problems, but it is a single number chosen for the whole job, so it is likely too large at the end and possibly too small at the start.

### Learning rate decay

Reduce the learning rate by a factor after a fixed number of epochs or when the validation loss stops improving. A step decay schedule halves the LR every N epochs; an exponential decay multiplies it by a constant factor every step.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# StepLR: multiply LR by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)

for epoch in range(num_epochs):
    # ... training loop ...
    scheduler.step()   # called once per epoch, after the optimizer step
    print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
```

### Cosine annealing

Instead of a step function, the LR oscillates smoothly from `eta_max` down to `eta_min` following a cosine curve over `T_max` epochs. This is one of the most reliable schedules in practice: it explores broadly at the start, refines carefully at the end, and restarts are optional.

\[ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \cdot \pi}{T_{\max}}\right)\right) \]

**Decoding:**
- **\(\eta_{\max}\)**: the starting (maximum) learning rate
- **\(\eta_{\min}\)**: the floor (minimum) learning rate — commonly 0 or a small fraction of \(\eta_{\max}\)
- **\(T_{\max}\)**: the number of epochs for one full cosine cycle
- **\(\cos(\cdot)\)**: the cosine function decays from 1 to -1 over \([0, \pi]\), mapping to LR decaying from \(\eta_{\max}\) to \(\eta_{\min}\)

```python
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-5
)

for epoch in range(50):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    
    scheduler.step()   # advance the schedule after each epoch
```

### Warmup: start small, grow fast, then anneal

Warmup is especially useful when the model starts with random weights and early gradients are noisy. A very large LR at step 0 can destroy the initial parameters before training stabilizes. Warmup linearly increases the LR from near-zero to the target LR over the first N steps, then decays normally.

```python
from torch.optim.lr_scheduler import LambdaLR

warmup_steps = 100   # number of steps to ramp up

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps        # linear ramp from 0 to 1
    return 1.0                            # full LR thereafter (combine with another scheduler)

optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler  = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Call scheduler.step() after each optimizer step (not each epoch)
for step, (X_batch, y_batch) in enumerate(train_loader):
    optimizer.zero_grad()
    loss = F.mse_loss(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### Rule of thumb

| Symptom | Action |
|---------|--------|
| Loss diverges or oscillates wildly | Reduce LR by 10× (try `1e-4`) |
| Loss improves but very slowly | Increase LR by 3× (try `3e-3`) |
| Loss plateaus with training still to go | Add cosine annealing or step decay |
| Loss is fine but final accuracy is slightly off | Try warmup |

Start with `lr=1e-3`. If it diverges, try `1e-4`. If it is too slow, try `3e-3`. Adding cosine annealing on top of whatever LR you settle on is almost always a free improvement.

## Gradient clipping

### The problem: exploding gradients

During backpropagation, gradients are multiplied together as they flow backward through layers. When a network has many layers or processes long sequences, gradients can compound and grow exponentially — this is the **exploding gradient** problem. A single weight update that is orders of magnitude too large can completely destabilize training.

In RL contexts, this is especially dangerous. TD errors can be large (especially early in DQN training), rewards can be sparse or suddenly large, and the replay buffer may contain a mix of experiences from very different policy stages. Any of these can produce a gradient that is far larger than usual.

### The solution: clip by norm

Gradient clipping caps the total norm of the gradient vector before the optimizer step. If the gradient norm exceeds `max_norm`, all gradients are scaled down proportionally so that the norm equals exactly `max_norm`. Small gradients are unaffected.

```python
loss.backward()

# Clip gradients: compute the norm of all parameter gradients combined,
# and scale them down if the norm exceeds max_norm.
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

The order is critical:

```
optimizer.zero_grad()   # 1. clear old gradients
loss.backward()         # 2. compute new gradients
clip_grad_norm_(...)    # 3. clip before they are applied
optimizer.step()        # 4. apply clipped gradients
```

If you clip after `optimizer.step()`, you have already applied the exploding gradient. If you clip before `loss.backward()`, the gradients have not been computed yet.

### In DQN training

```python
# Inside the DQN training step:
optimizer.zero_grad()

q_values   = online_net(states).gather(1, actions)
with torch.no_grad():
    td_targets = rewards + gamma * target_net(next_states).max(1).values * (1 - dones)

loss = F.huber_loss(q_values.squeeze(), td_targets, delta=1.0)
loss.backward()

# Clip before applying: prevents a single large TD error from destabilizing Q-net
torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)

optimizer.step()
```

The DQN paper clipped gradients to ±1 per parameter; modern implementations typically use `max_norm` of 1–10 depending on the architecture. For the SSA sensor-tasking agent in Module 3, `max_norm=10.0` is a reasonable starting point.

### How to choose max_norm

Monitor the gradient norm during training:

```python
# After loss.backward(), before clipping:
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Grad norm: {total_norm:.4f}")
```

If the norm is consistently below 1.0, clipping at 1.0 has no effect (which is fine — it is a safety net). If it occasionally spikes to 50 or 100, clipping at 10 will prevent the worst updates while allowing normal training to proceed.

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

## Tracking training with metrics

A training loop that only prints final accuracy gives you almost no information about what went wrong. Detailed logging during training is how you diagnose problems before they waste compute.

### What to log

- **Training loss** (per epoch or per N steps): is the model learning at all?
- **Validation loss** (per epoch): is it generalizing, or just memorizing?
- **Learning rate** (if using a schedule): are you actually annealing?
- **Gradient norm** (optional): are gradients well-behaved?
- **Best validation loss and the epoch it occurred**: for model selection

### Detecting overfitting from learning curves

```
Epoch   1: train=0.850  val=0.870   ← both high, normal at start
Epoch  10: train=0.120  val=0.135   ← both falling together, healthy
Epoch  20: train=0.050  val=0.080   ← gap widening slightly, watch it
Epoch  30: train=0.030  val=0.078   ← val plateaus while train keeps falling
Epoch  40: train=0.020  val=0.085   ← val starts rising: OVERFITTING
Epoch  50: train=0.015  val=0.098   ← definitely overfit, stop here
```

The divergence between training and validation loss is the signature of overfitting. The best model is at epoch 30, when validation loss was lowest.

### Detecting underfitting

```
Epoch   1: train=0.850  val=0.870
Epoch  20: train=0.600  val=0.610   ← barely moved
Epoch  50: train=0.550  val=0.560   ← still barely moving
```

Both losses are high and barely improving. The model is too small, the learning rate is too low, or the features do not contain enough signal.

### A complete training loop with logging and best-model tracking

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def train_with_logging(model, train_loader, X_val, y_val,
                       num_epochs=60, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    best_val_loss   = float('inf')
    best_epoch      = 0
    history         = {'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'LR':>10} | {'Best?':>6}")
    print("-" * 55)

    for epoch in range(num_epochs):
        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, y_val).item()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # ── Track best model ──────────────────────────────────────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            # Save best weights in memory (see checkpoint section for file save)
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>10.6f} | "
                  f"{current_lr:>10.2e} | {'  *' if is_best else ''}")

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    # Restore best weights
    model.load_state_dict(best_state)
    return history
```

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

## Saving and loading models

### Why you need checkpointing

Training a neural network takes time. If it crashes at epoch 47 out of 50, you want to recover without starting over. More importantly, because validation loss can start rising before training ends (overfitting), you should save the best model during training and reload it at the end — not just use whatever weights happened to be in memory when the loop finished.

### Saving a model

```python
# Save only the weights (the state dict), not the full model object.
# This is preferred because the architecture definition lives in your code,
# not in the file — you can change the code and load old weights selectively.
torch.save(model.state_dict(), 'conjunction_risk_model.pt')

# Loading:
model = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
model.load_state_dict(torch.load('conjunction_risk_model.pt'))
model.eval()
```

### Why state_dict, not the full model

`torch.save(model, path)` pickles the entire model object, including the class definition. This breaks when you rename a class, move a file, or upgrade PyTorch. `state_dict()` is just an `OrderedDict` mapping parameter names to tensors — it has no dependency on the class definition. Always prefer saving the state dict.

### Complete checkpoint pattern used in DQN training

A DQN agent trains for millions of steps. The agent should checkpoint periodically (so a crash does not lose days of compute) and keep the best-performing checkpoint separately (so evaluation always uses the best policy, not the most recent):

```python
import os
import torch
import torch.nn.functional as F

def save_checkpoint(state, path):
    """Save a training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint. Returns the epoch and best_val_loss."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_loss', float('inf'))

# In your training loop:
best_val_loss  = float('inf')
checkpoint_dir = 'checkpoints/dqn_ssa_sensor_tasking'

for epoch in range(num_epochs):
    # ... training and validation ...

    # Always save the latest checkpoint (for crash recovery)
    save_checkpoint({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss':      train_loss,
        'val_loss':        val_loss,
        'best_val_loss':   best_val_loss,
    }, path=os.path.join(checkpoint_dir, 'latest.pt'))

    # Separately save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint({
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'val_loss':    val_loss,
        }, path=os.path.join(checkpoint_dir, 'best.pt'))
        print(f"  → New best model saved (val_loss={val_loss:.6f})")

# After training: load the best weights for deployment
_, _ = load_checkpoint(
    os.path.join(checkpoint_dir, 'best.pt'), model
)
model.eval()
```

The two-file pattern (`latest.pt` + `best.pt`) is standard in practice. `latest.pt` lets you resume after a crash. `best.pt` is what you deploy or evaluate on. They are usually different files by the end of training.

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

## Key Takeaways

- **The inner loop never changes:** zero gradients, forward pass, compute loss, backward pass, optimizer step. Every training loop in this course — DQN, AlphaZero, deep CFR — is built on this four-step core. Learn it until it is automatic.
- **Learning rate is the most consequential hyperparameter.** Start at `1e-3` with Adam. If training diverges, try `1e-4`. If it is too slow, try `3e-3`. Adding cosine annealing on top is almost always a free improvement and requires only two lines of code.
- **Gradient clipping is a safety net, not a crutch.** Set `max_norm=1.0` for supervised problems and `max_norm=10.0` for RL. It has no effect when gradients are well-behaved, and prevents catastrophic weight updates when they are not. Always place it between `loss.backward()` and `optimizer.step()`.
- **Log training loss, validation loss, and learning rate at every epoch.** You cannot debug what you cannot see. The divergence between training and validation loss is the earliest signal of overfitting; both high losses together indicate underfitting.
- **Save the best validation loss checkpoint separately from the latest checkpoint.** `latest.pt` is for crash recovery. `best.pt` is what you deploy. They are usually different files by end of training, and you want both.
- **Always save `state_dict()`, never `torch.save(model, path)`.** The state dict is architecture-independent and survives code refactors. The full model pickle breaks when you move or rename the class.
- **`model.eval()` and `torch.no_grad()` are not optional during validation.** `model.eval()` disables Dropout and other train-only layers. `torch.no_grad()` prevents PyTorch from storing the full computational graph for every forward pass, which would otherwise exhaust memory on large validation sets.

---

## Quiz

{{#quiz 04-the-training-loop.toml}}
