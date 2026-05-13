# Lesson 2: Building an MLP in PyTorch


<!-- toc -->

## Where this fits

Lesson 1 gave you the activation functions. Module 1 gave you linear layers. This lesson snaps those pieces together into a complete neural network and traces exactly what happens to a state vector as it flows through. Then it shows you how PyTorch packages all of this so you do not have to manage weights manually. The MLP you build here is the same architecture used as a value network in Module 4 and as a regret network in Module 5, just with different input/output dimensions and different training objectives.

## What is an MLP?

**MLP** stands for Multi-Layer Perceptron. It is the simplest complete neural network: a sequence of linear layers with nonlinear activations in between.

The structure is:

```
Input → [Linear → ReLU] → [Linear → ReLU] → ... → [Linear] → Output
```

The layers in brackets are "hidden layers." The final linear layer (without ReLU) produces the output. The whole network is a composition of functions applied in sequence.

"Multi-layer" means there is at least one hidden layer between input and output. "Perceptron" is a historical term for a single neuron. A multi-layer perceptron is many neurons organized into layers.

## Building one by hand first

Before using PyTorch's conveniences, let us trace a forward pass manually through a small MLP. This makes it impossible to treat the network as a black box.

**The scenario**: you want to estimate how much a satellite operator should trust a new conjunction alert. Your feature vector has 4 inputs:

```
x = [alert_confidence,   # 0 to 1: how confident the detection algorithm is
     approach_speed,     # km/s: how fast the objects are converging
     miss_distance,      # km: expected closest approach distance
     time_to_tca]        # hours: time until closest approach
```

Your network has:
- **Input size**: 4
- **Hidden layer**: 8 neurons
- **Output size**: 1 (a single trust score, higher means more urgent)

This is a 4 → 8 → 1 network.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(7)  # reproducible weights

# Layer 1 weights and biases (shape: 8x4 and 8)
W1 = torch.randn(8, 4) * 0.3
b1 = torch.zeros(8)

# Layer 2 weights and biases (shape: 1x8 and 1)
W2 = torch.randn(1, 8) * 0.3
b2 = torch.zeros(1)

# An example alert feature vector
x = torch.tensor([0.85, 7.2, 0.4, 1.5])
print(f"Input: {x.tolist()}")

# ----- Forward pass -----

# Step 1: Linear transformation (layer 1)
z1 = W1 @ x + b1
print(f"\nAfter linear layer 1 (z1): {z1.tolist()}")
# 8 raw values, can be positive or negative

# Step 2: ReLU activation
a1 = F.relu(z1)
print(f"After ReLU (a1):           {a1.tolist()}")
# Negative values become 0; positive values pass through

# Step 3: Linear transformation (layer 2)
z2 = W2 @ a1 + b2
print(f"\nAfter linear layer 2 (z2): {z2.tolist()}")
# A single raw score

# Step 4: No activation on the output (this is a regression network)
output = z2
print(f"Network output (trust score): {output.item():.4f}")
```

> **Dependencies** (Rust blocks in this lesson): `ndarray = "0.17"` and `rand = "0.10"` in `[dependencies]`.

```rust
# extern crate ndarray;
# extern crate rand;
use ndarray::{Array1, Array2};
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let w = |rng: &mut StdRng| (rng.random::<f64>() - 0.5) * 0.6; // scale ~0.3 std

    // Layer 1: 4 inputs -> 8 hidden (shape 8×4)
    let w1 = Array2::from_shape_vec((8, 4), (0..32).map(|_| w(&mut rng)).collect()).unwrap();
    let b1 = Array1::zeros(8);

    // Layer 2: 8 hidden -> 1 output (shape 1×8)
    let w2 = Array2::from_shape_vec((1, 8), (0..8).map(|_| w(&mut rng)).collect()).unwrap();
    let b2 = Array1::zeros(1);

    // Feature vector: [alert_confidence, approach_speed, miss_distance, time_to_tca]
    let x = Array1::from_vec(vec![0.85, 7.2, 0.4, 1.5]);
    println!("Input: {:?}", x.to_vec());

    // Step 1: linear layer 1
    let z1 = w1.dot(&x) + &b1;       // shape 8
    println!("After linear 1: {:?}", z1.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());

    // Step 2: ReLU — zero out negatives
    let a1 = z1.mapv(|v| v.max(0.0)); // shape 8
    println!("After ReLU:     {:?}", a1.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());

    // Step 3: linear layer 2
    let z2 = w2.dot(&a1) + &b2;      // shape 1
    println!("Network output (trust score): {:.4}", z2[0]);
    // Random weights — meaningless until trained; the shape flow is what matters.
}
```

Shape flow: `x` is length 4, `w1.dot(&x)` contracts the 8×4 matrix with the 4-vector to produce length 8, `relu` leaves the shape unchanged, `w2.dot(&a1)` contracts 1×8 with length 8 to produce length 1. Identical to the Python trace above.

Look at the shapes at each stage:
- x: length 4
- z1 = W1 @ x + b1: W1 is 8×4, x is length 4, result is length 8
- a1 = ReLU(z1): same shape as z1, length 8
- z2 = W2 @ a1 + b2: W2 is 1×8, a1 is length 8, result is length 1
- output: a single number

The dimensions flow through like water through pipes. Each layer's output becomes the next layer's input. The shapes must be compatible at every step.

## Building the same network with nn.Sequential

Doing this manually every time is tedious and error-prone. PyTorch's `nn` module packages layers into reusable objects.

```python
import torch
import torch.nn as nn

# Define the same 4 -> 8 -> 1 network
model = nn.Sequential(
    nn.Linear(4, 8),   # input layer: 4 inputs, 8 outputs
    nn.ReLU(),         # activation
    nn.Linear(8, 1),   # output layer: 8 inputs, 1 output
)

print(model)
# Sequential(
#   (0): Linear(in_features=4, out_features=8, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=8, out_features=1, bias=True)
# )

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Layer 1: 8*4 weights + 8 biases = 40
# Layer 2: 1*8 weights + 1 bias = 9
# Total: 49
```

A forward pass is now just:

```python
x = torch.tensor([0.85, 7.2, 0.4, 1.5])
output = model(x)
print(f"Output: {output.item():.4f}")
```

PyTorch runs the same sequence of operations: linear, relu, linear. The result is the same as the manual version, just packaged more cleanly.

## Adding more capacity: a deeper network

The 4 → 8 → 1 network has limited representational capacity. Real value functions and policies often need more. Here is a more capable network for the same problem:

```python
import torch.nn as nn

# A deeper network: 4 -> 64 -> 64 -> 1
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params}")
# Layer 1: 64*4 + 64 = 320
# Layer 2: 64*64 + 64 = 4160
# Layer 3: 1*64 + 1 = 65
# Total: 4545
```

This network has 4,545 parameters versus 49. It can represent much more complex functions of the input. The price is more computation and more training data needed to fit all those parameters without overfitting (more on this below).

In RL and game theory, 64-to-256 hidden units per layer is common for modest-scale problems. AlphaGo Zero used 20 residual blocks with 256 filters (a far more complex architecture), but the underlying structure is still "linear layers with nonlinearities."

## Overfitting and capacity: why more parameters can hurt

More parameters means more expressive power — but also more opportunity for the network to **overfit**. Overfitting happens when the network memorizes the training examples rather than learning the underlying pattern. The result is near-perfect performance on training data and poor performance on any data it has not seen before.

Here is the core tension: the 4,545-parameter network above could theoretically memorize 4,545 training examples perfectly just by storing them. If you only have 100 labeled conjunction alerts to train on, a 4,545-parameter network will almost certainly overfit.

The failure mode looks like this:

```
Training loss:    0.003 (near-perfect)
Validation loss:  0.48  (much worse)
```

The network learned the noise in your 100 training examples, not the signal.

For the conjunction risk network, realistic training datasets might be:
- 100–500 labeled alerts (a small dataset): keep the network small (4 → 32 → 1)
- 10,000+ labeled alerts: a larger network (4 → 64 → 64 → 1) is appropriate
- 100,000+: you have latitude to go deeper

### Dropout: regularization by randomness

**Dropout** is a technique for combating overfitting. During training, each call to a dropout layer randomly sets a fraction p of the activations to zero. The fraction p is called the **dropout rate** and is typically 0.1 to 0.5.

The intuition: by randomly disabling neurons during training, the network cannot rely on any single neuron always being present. It is forced to learn redundant representations and cannot simply memorize training examples through a fixed chain of activations.

```python
import torch
import torch.nn as nn

# Add dropout after each ReLU in the conjunction value network
model_with_dropout = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Dropout(p=0.3),     # 30% of neurons zeroed during training
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(64, 1),
)

x = torch.randn(4)

# Training mode: dropout is active (random zeros appear)
model_with_dropout.train()
out1 = model_with_dropout(x)
out2 = model_with_dropout(x)
print(f"Train mode (run 1): {out1.item():.4f}")
print(f"Train mode (run 2): {out2.item():.4f}")
# These will differ because different neurons are dropped each time.

# Eval mode: dropout is disabled (full network is used)
model_with_dropout.eval()
out3 = model_with_dropout(x)
out4 = model_with_dropout(x)
print(f"Eval mode (run 1): {out3.item():.4f}")
print(f"Eval mode (run 2): {out4.item():.4f}")
# These will be identical — no randomness in eval mode.
```

**Critical rule**: always call `model.train()` before a training loop and `model.eval()` before inference or evaluation. Forgetting to switch modes is a silent bug — evaluation under dropout underestimates the network's true performance because random neurons are disabled.

Dropout should not be applied to the final output layer. It is a training regularizer for hidden layers only.

## The forward pass as an SSA pipeline

Let us trace what happens conceptually when an orbital state flows through a value network.

Suppose you have a 6-element orbital state vector (position + velocity) and want to estimate the value of that state (roughly: how favorable is this orbital configuration for your satellite?).

```python
import torch
import torch.nn as nn

# State: [x_km, y_km, z_km, vx_kms, vy_kms, vz_kms]
state = torch.tensor([6371.0, 500.0, -200.0, 7.2, 0.3, -0.1])

# A value network: 6 inputs -> 128 hidden -> 64 hidden -> 1 value
value_net = nn.Sequential(
    nn.Linear(6, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

# Forward pass: what is the estimated value of this state?
value_estimate = value_net(state)
print(f"Value estimate: {value_estimate.item():.4f}")
# Random weights, so the number means nothing yet. Training will fix this.
```

And a policy network that outputs action probabilities:

```python
# Policy network: 6 inputs -> 64 hidden -> 4 actions
policy_net = nn.Sequential(
    nn.Linear(6, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
    nn.Softmax(dim=0),  # converts logits to probabilities
)

# Forward pass: what action probabilities does the current policy assign?
action_probs = policy_net(state)
print(f"Action probabilities: {[f'{p:.3f}' for p in action_probs.tolist()]}")
# Four probabilities summing to 1.0
print(f"Sum: {action_probs.sum().item():.4f}")  # 1.0000
```

Note: `nn.Softmax(dim=0)` applies softmax along dimension 0. For a single vector (not a batch), this is correct. When processing batches, you typically use `dim=1` because dimension 0 is the batch dimension.

## Defining networks as classes (the preferred pattern)

`nn.Sequential` is convenient for simple linear stacks. For anything more complex (networks with branches, skip connections, or custom behavior), you define the network as a Python class inheriting from `nn.Module`. This is the standard pattern in research code.

### Why `__init__` and `forward` are separate

`__init__` declares the architecture: which layers exist, how many parameters they have, what their shapes are. This runs once when you create the model.

`forward` declares the computation: how data flows through those layers. This runs every time you call the model on an input.

This separation matters because:
- Parameters defined in `__init__` are automatically tracked by PyTorch's optimizer
- The same `forward` method handles both single inputs and batches
- You can add arbitrary Python logic in `forward` (conditionals, loops, etc.) without affecting the parameter structure

### How `super().__init__()` works

`nn.Module` is PyTorch's base class for all neural networks. When you write `class ConjunctionValueNet(nn.Module)`, you are saying "this class IS an nn.Module." Calling `super().__init__()` runs `nn.Module`'s initialization code, which sets up the internal machinery for parameter tracking. If you forget it, assigning `self.fc1 = nn.Linear(...)` will raise an error because the parameter registry does not exist yet.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConjunctionValueNet(nn.Module):
    """Estimates the conjunction risk value from an orbital feature vector."""
    
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super().__init__()          # REQUIRED: sets up nn.Module internals
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))    # input -> hidden
        x = self.dropout(x)        # regularization (active in train mode only)
        x = F.relu(self.fc2(x))    # hidden -> hidden
        x = self.dropout(x)
        x = self.fc3(x)            # hidden -> output (no activation: regression)
        return x

# Instantiate
net = ConjunctionValueNet(input_dim=4, hidden_dim=64, dropout_rate=0.2)
print(net)
# ConjunctionValueNet(
#   (fc1): Linear(in_features=4, out_features=64, bias=True)
#   (fc2): Linear(in_features=64, out_features=64, bias=True)
#   (fc3): Linear(in_features=64, out_features=1, bias=True)
#   (dropout): Dropout(p=0.2, inplace=False)
# )

# Training: dropout is active
net.train()
x = torch.tensor([0.85, 7.2, 0.4, 1.5])
train_output = net(x)
print(f"Train pass output: {train_output.item():.4f}")

# Inference: dropout disabled, deterministic
net.eval()
with torch.no_grad():             # also disable gradient computation for speed
    eval_output = net(x)
print(f"Eval pass output:  {eval_output.item():.4f}")
# These may differ because dropout was active in train mode.
```

The pattern `net.eval()` + `torch.no_grad()` before inference is standard — `eval()` disables dropout and batch normalization's running stat updates; `no_grad()` disables gradient tracking, saving memory and computation.

## Inspecting what the network knows

After building a network (before training), its weights are randomly initialized. You can inspect them:

```python
# See all named parameters and their shapes
for name, param in net.named_parameters():
    print(f"{name}: shape={param.shape}, "
          f"mean={param.data.mean():.4f}, std={param.data.std():.4f}")
```

After training (lesson 4), the weights will have changed to reduce the loss on training data. The architecture (shapes) stays the same; the values inside change.

## Weight initialization: why random is not enough

When you create a network, PyTorch initializes the weights randomly. The scale of this initial randomness matters more than most beginners realize. Two failure modes:

**Too small (vanishing gradients)**: If weights are initialized very close to zero, the activations after each layer are tiny. The gradient signal shrinks as it propagates back through layers. Early layers learn almost nothing.

**Too large (exploding gradients)**: If weights are large, activations grow exponentially through the layers. Gradients also explode. Training becomes numerically unstable, often producing NaN losses.

The goal is initialization that keeps activations at a reasonable scale throughout the network — neither shrinking to zero nor blowing up.

### Xavier initialization (for tanh)

\\[ w \sim \mathcal{U}\left(-\frac{1}{\sqrt{\text{fan\_in}}}, \frac{1}{\sqrt{\text{fan\_in}}}\right) \\]

**Decoding:** fan\_in is the number of inputs to a layer (the "in" dimension of the weight matrix). Xavier initialization scales the initial weights by \\(1/\sqrt{\text{fan\_in}}\\), which keeps the variance of activations approximately constant across layers when using tanh.

### He initialization (for ReLU)

\\[ w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right) \\]

**Decoding:** He initialization uses a larger scale factor — \\(\sqrt{2/\text{fan\_in}}\\) instead of \\(1/\sqrt{\text{fan\_in}}\\) — because ReLU zeros out half its inputs (all negative values), which would otherwise cause activations to shrink. The factor of 2 compensates for this halving. He initialization is the PyTorch default for `nn.Linear`.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

def check_activation_scale(init_scale, n_layers=5, layer_size=64, input_size=64):
    """Show how activation std changes through layers under different initializations."""
    x = torch.randn(1, input_size)
    
    stds = [x.std().item()]
    for _ in range(n_layers):
        W = torch.randn(layer_size, x.shape[1]) * init_scale
        x = torch.relu(W @ x.T).T
        stds.append(x.std().item())
    return stds

# Naive small init
naive_stds = check_activation_scale(init_scale=0.01)
print("Naive (0.01 scale):", [f"{s:.4f}" for s in naive_stds])
# Vanishes quickly: ['1.0000', '0.0058', '0.0003', '0.0000', '0.0000', '0.0000']

# He initialization: sqrt(2 / fan_in) for fan_in=64 -> sqrt(2/64) ≈ 0.177
he_scale = (2 / 64) ** 0.5
he_stds = check_activation_scale(init_scale=he_scale)
print("He init:           ", [f"{s:.4f}" for s in he_stds])
# Stays relatively stable: ['1.0000', '0.5623', '0.5441', '0.5390', '0.5371', '0.5364']
```

With naive small initialization, the activation standard deviation shrinks to essentially zero after 4 layers — the network's early layers receive no meaningful gradient signal. He initialization keeps the scale stable, enabling reliable training.

PyTorch's `nn.Linear` uses Kaiming uniform initialization by default (a variant of He), so you usually do not need to do this manually. But understanding why it works helps when debugging training instability.

## Batched inputs: processing many examples at once

In practice, you never process one example at a time; you process batches of examples simultaneously. PyTorch handles this automatically through broadcasting.

A single input has shape `(4,)`. A batch of 32 inputs has shape `(32, 4)`. The linear layer `nn.Linear(4, 8)` handles both shapes correctly:

```python
# Single input
x_single = torch.randn(4)
out_single = net(x_single)
print(f"Single output shape: {out_single.shape}")  # (1,)

# Batch of 32 inputs
x_batch = torch.randn(32, 4)
out_batch = net(x_batch)
print(f"Batch output shape: {out_batch.shape}")   # (32, 1)
```

All 32 examples are processed simultaneously using matrix operations, which is much faster than a loop. Modern GPUs are optimized for exactly this kind of batch processing. Training typically works with batches of 32 to 512 examples at a time for efficiency.

Note: when using softmax on batched data, you want `F.softmax(x, dim=1)`, not `dim=0`, because dimension 0 is the batch dimension and dimension 1 is the feature/action dimension.

## Key Takeaways

- **An MLP is just linear layers alternating with activation functions.** The activation functions are what make it capable of learning nonlinear relationships — without them, the whole network collapses to a single linear transformation regardless of depth.
- **More parameters means more capacity, but also more risk of overfitting.** A network with 10,000 parameters trained on 100 examples will memorize the training data. Match network size to dataset size, or use regularization.
- **Dropout is a simple and effective regularizer.** It randomly zeros out activations during training, preventing the network from memorizing specific pathways. Always call `model.train()` before training and `model.eval()` before inference — forgetting this is a silent bug.
- **Weight initialization scale matters.** Too small causes vanishing gradients (early layers learn nothing). Too large causes exploding gradients (training becomes numerically unstable). He initialization (`sqrt(2 / fan_in)`) is the right default for ReLU networks and is what PyTorch uses by default.
- **The `nn.Module` class pattern (`__init__` + `forward`) is the standard for anything beyond simple sequential stacks.** Architecture is declared in `__init__`; computation is defined in `forward`. The separation allows arbitrary Python logic in the computation path without affecting parameter tracking.
- **Always pair `model.eval()` with `torch.no_grad()` during inference.** `eval()` disables dropout and running-stat updates; `no_grad()` disables gradient computation. Using either without the other is incomplete.

## Quiz

{{#quiz 02-building-an-mlp.toml}}
