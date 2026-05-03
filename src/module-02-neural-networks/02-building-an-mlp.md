# Lesson 2: Building an MLP in PyTorch

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

This network has 4,545 parameters versus 49. It can represent much more complex functions of the input. The price is more computation and more training data needed to fit all those parameters without overfitting (more on this in lesson 4).

In RL and game theory, 64-to-256 hidden units per layer is common for modest-scale problems. AlphaGo Zero used 20 residual blocks with 256 filters (a far more complex architecture), but the underlying structure is still "linear layers with nonlinearities."

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

`nn.Sequential` is convenient for simple linear stacks. For anything more complex (networks with branches, skip connections, or custom behavior), you define the network as a Python class inheriting from `nn.Module`. This is the standard pattern in research code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConjunctionValueNet(nn.Module):
    """Estimates the conjunction risk value from an orbital feature vector."""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # input -> hidden
        x = F.relu(self.fc2(x))  # hidden -> hidden
        x = self.fc3(x)          # hidden -> output (no activation: regression)
        return x

# Instantiate
net = ConjunctionValueNet(input_dim=4, hidden_dim=64)
print(net)

# Use it
x = torch.tensor([0.85, 7.2, 0.4, 1.5])
output = net(x)
print(f"Value estimate: {output.item():.4f}")
```

When you call `net(x)`, PyTorch internally calls `net.forward(x)`. The `__init__` method defines the layers; the `forward` method defines how data flows through them. All the layers you define in `__init__` as attributes (like `self.fc1`) are automatically tracked by PyTorch as parameters.

## Inspecting what the network knows

After building a network (before training), its weights are randomly initialized. You can inspect them:

```python
# See all named parameters and their shapes
for name, param in net.named_parameters():
    print(f"{name}: shape={param.shape}, "
          f"mean={param.data.mean():.4f}, std={param.data.std():.4f}")
```

After training (lesson 4), the weights will have changed to reduce the loss on training data. The architecture (shapes) stays the same; the values inside change.

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

## Quiz

{{#quiz 02-building-an-mlp.toml}}
