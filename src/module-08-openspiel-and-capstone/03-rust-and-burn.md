# Lesson 3: Rust and burn (The Production Gap)

## Where this fits

Up to this point you have used Python and OpenSpiel for everything. The capstone is going to be in Rust. Before we design the capstone, we need to be honest about what is available in the Rust ecosystem for ML and game-theoretic algorithms, what is not, and how the design of the capstone reflects these realities. The short version: there is no Rust-native equivalent to OpenSpiel, and that is a feature of the project (you fill the gap), not a bug.

This lesson is mostly informational. There is no implementation work; the goal is to give you accurate context for the design choices in lesson 4 and the project.

## What exists in the Rust ML ecosystem

### burn (the deep learning framework)

[`burn`](https://burn.dev) is the most viable deep learning framework in Rust as of this writing. It is an active project (Tracel AI), API-stable enough for serious use, and supports multiple backends: pure Rust (`ndarray`), CPU SIMD (`candle`), CUDA (`tch` and `cuda-jit`), Metal, and WGPU. It has an autograd engine, a layer abstraction (linear, convolutional, transformer, etc.), an optimizer module, and a training loop helper.

For our purposes, the important features are:
- Linear layers and ReLU activations (enough for the MLPs we use to approximate regret in deep CFR)
- Autograd (we can compute gradients of arbitrary computations on tensors)
- Loss functions (MSE, cross-entropy)
- Optimizers (Adam, SGD)

What it does not have natively (or has only in early form): pre-built RL algorithms, equilibrium solvers, or game-theoretic primitives. You have to build those yourself on top of the framework.

### candle

[`candle`](https://github.com/huggingface/candle) is HuggingFace's Rust-native deep learning framework. It is more focused on inference and on running large pretrained models. For training small models from scratch (which is what deep CFR needs), `burn` is the more idiomatic choice. We will use `burn` for the capstone.

### tch-rs

[`tch-rs`](https://github.com/LaurentMazare/tch-rs) is Rust bindings to libtorch (PyTorch's C++ API). This is the most feature-complete option if you need full PyTorch-compatible operators, but it links against a large C++ library and is more cumbersome to deploy. For a research artifact like the capstone, the pure-Rust `burn` is more aligned with the goals.

### linfa

[`linfa`](https://github.com/rust-ml/linfa) is a classical ML library (linear models, k-means, decision trees, etc.). Useful for non-deep-learning ML tasks. Not relevant to the capstone.

### dfdx

[`dfdx`](https://github.com/coreylowman/dfdx) is another deep learning crate. Type-system-heavy (compile-time tensor shapes). Powerful but the ergonomics differ enough from Python and PyTorch that we are choosing `burn` for closer conceptual mapping.

## What exists in the Rust game-theory ecosystem

### cfr / cfr-rs

There are a few crates implementing CFR. The most notable:

- [`cfr`](https://crates.io/crates/cfr): a small CFR implementation that supports tabular CFR for arbitrary game implementations exposed through a trait. Active in the past but small in scope.
- [`cfr-rs`](https://github.com/...): another CFR implementation, similarly narrow.

Both are helpful as references but neither is a comprehensive game-theory library. They do not have the breadth of CFR variants (MCCFR, deep CFR, ESCFR), the equilibrium solvers (Nash, alpha-rank), or the algorithm zoo that OpenSpiel provides.

For the capstone we will write our own CFR implementation rather than depending on these crates. The pedagogical value of writing CFR yourself is substantial, and you avoid pinning to a third-party library that may not match your game's API.

### rl

The [`rl`](https://crates.io/crates/rl) crate is an early-stage RL library. It provides some primitives (environments, agents) but is far from production-ready. Not used in the capstone.

### Other game-related crates

There are crates for specific games (chess, Go, etc.) that have their own engines. These are useful if you want to play those specific games but do not provide a general framework.

### What does not exist (yet)

- A Rust-native equivalent to OpenSpiel: no general-purpose framework with a Game/State/Information-state abstraction, multiple solver implementations, and a wide game catalog.
- A Rust equivalent to RLlib or Stable Baselines3: no comprehensive multi-algorithm RL training framework.
- Rust bindings to OpenSpiel itself: there are no actively maintained bindings as of this writing. (OpenSpiel does have C++ bindings to several languages, but Rust is not among them.)

This is the gap. The capstone exists to demonstrate filling part of it for one specific application (your SSA work).

## Why this gap exists

A practical observation: the Rust ML ecosystem is younger than Python's by about a decade, and it has different cultural priorities. Python ML grew up serving researchers who needed quick iteration; Rust ML is growing up serving deployment engineers who need reliable, fast inference. The result is that Rust's ML libraries are well-suited to running trained models efficiently but less mature for the iterative research workflow that a tool like OpenSpiel supports.

For game-theoretic algorithms specifically, the academic community using them is small and Python-heavy. There has been little pull on the Rust side to build comprehensive game-theory tooling, because the people doing the research are mostly happy with Python. The gap is real but it is also a niche; it is not where Rust ML investment has gone.

For your SSA work, this gap matters because: you want to embed game-theoretic reasoning into a larger Rust simulation system (your data engineering background matters here), you want the performance for large rollouts, and you do not want a Python interop layer in the production system. The capstone solves your specific problem rather than trying to build a general framework.

## What the capstone will use

Based on the above:

- **Language**: Rust 2021 edition.
- **Cargo workspace**: three crates (`game`, `solver`, `cli`) for clean separation.
- **`burn`**: for the neural network in the deep CFR variant. We will use the `ndarray` backend for portability (no GPU dependency for the capstone). If you later want to run on GPU, switching backends is mostly a Cargo feature change.
- **`rand`**: for the random number generation (sampling chance outcomes, MCCFR sampling).
- **Standard library only otherwise**: `HashMap` for the regret tables, `Vec<f64>` for strategy vectors. No third-party game-theory crate.

We do not use `cfr` or `cfr-rs` because:
1. Writing CFR yourself is the pedagogical point of the project.
2. The third-party crates' game representations may not match what you need for the SSA scenario.
3. Owning the code lets you extend it in any direction (your future thesis algorithms).

## A taste of `burn` syntax

Just so you have context for the capstone, here is what training a simple MLP looks like in `burn`. This is not the actual capstone code (lesson 4 designs the game; the project implements it); it is enough for you to see the API style.

```rust
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    layer3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Mlp<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(input_dim,  hidden_dim).init(device),
            layer2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            layer3: LinearConfig::new(hidden_dim, output_dim).init(device),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(input);
        let x = self.activation.forward(x);
        let x = self.layer2.forward(x);
        let x = self.activation.forward(x);
        self.layer3.forward(x)
    }
}
```

Conceptually similar to PyTorch: a struct with named layer fields, a forward method that runs the computation. The differences are syntactic (Rust's `<B: Backend>` generic parameter, the `init(device)` instead of just `init`, the `Relu::new()` instead of a free `relu` function). These differences matter for ergonomics but not for understanding what the code does.

The `Module` derive macro handles the autograd registration: `burn` knows that this struct contains trainable parameters (the linear layers) and that calls into them should produce gradient-tracked tensors. The training loop uses `burn::optim` to step parameters toward lower loss, just like PyTorch's optimizer.step().

## What you can expect to be hard

Some things are smoother in Rust; some are harder. Honestly:

- **Compile times**: noticeable, especially when adding `burn` (it pulls in many dependencies). Plan on 30-60 second incremental builds for the capstone after the first compile.
- **Error messages around generics**: `burn`'s `Backend` parameter shows up everywhere. The error messages can be long. The good news: once your code compiles, it tends to work.
- **Tensor shape debugging**: less ergonomic than PyTorch's `.shape` because Rust does not have a REPL. You will use `println!` more.

Things that will feel familiar:

- The forward-pass code looks like PyTorch with different syntax.
- The training loop structure (forward, loss, backward, step) is identical.
- Layer abstractions, optimizers, loss functions all map clearly.

Things that will be smoother:

- Performance: pure-Rust deep CFR will be substantially faster than Python equivalents for the small networks we use, simply because of language overhead reduction.
- Integration with other Rust code: when you later embed the CFR solver into a larger SSA simulation, no FFI boundary, no GIL.
- Memory management: no surprise allocations during inner loops; you can profile with standard Rust tools.

## What "deep CFR in Rust" actually means

Deep CFR (which we cover in Module 5 and reuse here) replaces the regret table with a neural network. Instead of `HashMap<InfoSet, RegretVec>`, you have a network that takes an information state tensor and outputs predicted regret values.

The Rust implementation has the same structure:
1. Sample game trajectories (using current strategy).
2. Compute counterfactual regrets at each information set encountered.
3. Train a neural network on (information state, regret) pairs.
4. The network's output defines the next iteration's strategy.

The data structures change (network instead of table) but the algorithm is the same. The capstone implements both: the tabular version first (for correctness verification on small games), then the deep version (for scalability).

## Recap: what to expect in the capstone

- A Rust crate that does what an OpenSpiel-based Python script would do, but for one specific game.
- Tabular CFR working correctly, with exploitability dropping to near zero.
- A `burn`-based deep CFR variant that approximates regret values with a neural network.
- A CLI for training, evaluating exploitability, and inspecting strategies.
- About 1500-2500 lines of Rust total, including tests.

This is small enough to actually finish. It is large enough to be a real artifact you can extend. And every line of code has a direct conceptual antecedent in the lessons you have already worked through.

## Why Rust for production SSA

The choice to use Rust for the capstone is not arbitrary. It reflects a genuine engineering reality in operational SSA systems and a deliberate training goal for working in that environment.

### Memory safety without garbage collection

Rust's ownership and borrowing system guarantees memory safety at compile time, with no runtime garbage collector. This matters for SSA systems for two reasons.

First, **GC pauses are unacceptable in real-time data pipelines**. A Java or Python-based conjunction screening pipeline that pauses for 50-200 milliseconds during a GC cycle cannot be used for high-frequency orbital data ingestion. The U.S. 18th Space Control Squadron processes hundreds of thousands of conjunction assessments per day; even small latency spikes compound across that volume.

Second, **memory safety errors are the most common class of CVE in critical infrastructure**. An SSA system that processes conjunction data from multiple sources and generates maneuvering recommendations has a large attack surface. Memory corruption bugs (buffer overflows, use-after-free, data races) are the class of vulnerabilities that adversaries exploit to inject false data or cause system failure. Rust's compile-time guarantees eliminate this entire bug class without the runtime cost of a managed language.

### Zero-cost abstractions

Rust's `trait` system lets you write generic code — like a CFR solver parameterized over `Game` implementations — without virtual dispatch overhead. In Python, every call through an abstract base class goes through Python's dynamic dispatch mechanism. In Rust, generic code over traits is monomorphized at compile time: the compiler generates a separate implementation for each concrete type, with all method calls inlined. The abstraction is free.

For a CFR solver that traverses the game tree millions of times, this matters. The inner loop is:

```
current_player() -> legal_actions() -> apply_action() -> information_state_string()
```

In Python, each of these is a virtual method call with Python overhead. In Rust with trait generics, each is a direct function call after monomorphization.

### No Python GIL, true parallelism

Python's Global Interpreter Lock (GIL) prevents multiple Python threads from executing Python bytecode simultaneously. External Sampling MCCFR and other Monte Carlo CFR variants are embarrassingly parallelizable: each sampling thread is independent. In Python, you achieve parallelism only via multiprocessing (which has significant memory overhead from process spawning) or by delegating to C extensions that release the GIL.

In Rust, parallelism is straightforward. The `rayon` crate provides data-parallel iterators that distribute work across all CPU cores without any data races, because Rust's ownership system enforces safe concurrent access at compile time. A parallelized sampling loop in Rust is approximately:

```rust
use rayon::prelude::*;

let regret_samples: Vec<_> = (0..NUM_SAMPLES)
    .into_par_iter()                      // parallel iterator
    .map(|_| sample_trajectory(&game, &strategy, &mut rng.clone()))
    .collect();
```

The `Send + Sync` trait bounds enforced by the compiler guarantee that this is safe. Compare to Python's multiprocessing approach, which requires serializing the game state across a process boundary.

### Relevance to AFSPC/USSF operational systems

The Air Force Space Command (now USSF Space Operations Command) and its supporting infrastructure have historically used C/C++ for their core SSA software. The Astrodynamics Standards (AstroStds) library, the Space Fence signal processing chain, and the SOCRATES conjunction assessment service are C/C++ at their core. The shift toward Rust in new government software (DARPA HARDEN program, NSA's guidance recommending memory-safe languages) is making Rust increasingly relevant for this domain.

If you eventually embed a game-theoretic reasoning module into an operational SSA data pipeline, you want that module to be compatible with the surrounding system without an FFI boundary. Writing the capstone in Rust is practice for that eventual integration.

## The burn neural network library in depth

The code snippet in the previous section showed the basic `burn` syntax. Here is a more complete picture, with an explicit comparison to the PyTorch equivalent.

### A simple MLP: PyTorch vs. burn

**PyTorch (Python):**

```python
import torch
import torch.nn as nn

class RegretNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu   = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

model = RegretNetwork(input_dim=10, hidden_dim=64, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training step
x = torch.randn(32, 10)     # batch of 32 info state tensors
y = torch.randn(32, 3)      # target regret values
pred = model(x)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**burn (Rust):**

```rust
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Tensor},
    train::RegressionOutput,
};

#[derive(Module, Debug)]
pub struct RegretNetwork<B: AutodiffBackend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
    relu:   Relu,
}

impl<B: AutodiffBackend> RegretNetwork<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize,
               device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(input_dim,  hidden_dim).init(device),
            layer2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            output: LinearConfig::new(hidden_dim, output_dim).init(device),
            relu:   Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.layer1.forward(x));
        let x = self.relu.forward(self.layer2.forward(x));
        self.output.forward(x)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let pred = self.forward(x);
        let loss = burn::tensor::loss::mse_loss(
            pred.clone(),
            targets,
            burn::nn::loss::Reduction::Mean,
        );
        RegressionOutput::new(loss, pred, targets)
    }
}

// Training step
fn train_step<B: AutodiffBackend>(
    model: RegretNetwork<B>,
    optim: &mut impl Optimizer<RegretNetwork<B>, B>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (RegretNetwork<B>, f32) {
    let output = model.forward_step(x, y);
    let loss_val = output.loss.clone().into_scalar();
    let grads = GradientsParams::from_grads(output.loss.backward(), &model);
    let model = optim.step(1e-3, model, grads);
    (model, loss_val)
}
```

The structural mapping is direct: `nn.Module` becomes `#[derive(Module)]`, `nn.Linear` becomes `Linear<B>`, `torch.optim.Adam` becomes `AdamConfig`, the forward pass is the same computation. The generic `<B: AutodiffBackend>` parameter is the main syntactic difference; it parameterizes the computation backend (CPU ndarray, GPU CUDA, etc.) without changing the logic.

### Tensor operations in burn

Burn's tensor API mirrors PyTorch's but uses method chaining on the `Tensor<B, D>` type (where `D` is the number of dimensions). Common operations:

```rust
use burn::tensor::{backend::Backend, Tensor};

fn tensor_ops_demo<B: Backend>(device: &B::Device) {
    // Create tensors
    let a: Tensor<B, 2> = Tensor::zeros([3, 4], device);
    let b: Tensor<B, 2> = Tensor::ones([3, 4], device);
    
    // Elementwise operations
    let c = a + b;
    let d = c * 2.0;
    
    // Matrix multiplication
    let e: Tensor<B, 2> = Tensor::zeros([4, 5], device);
    let f = d.matmul(e);  // [3, 5]
    
    // Reduction
    let mean = f.mean();       // scalar
    let row_means = f.mean_dim(1);  // [3, 1]
    
    // Shape operations
    let flat = f.reshape([1, -1]);  // [1, 15]
    
    // Softmax (used for converting regrets to strategies)
    let logits: Tensor<B, 2> = Tensor::zeros([3, 5], device);
    let probs = burn::tensor::activation::softmax(logits, 1);  // [3, 5], rows sum to 1
}
```

For the CFR deep variant, the key operation is: given a vector of cumulative regrets (one per action), apply the regret-matching formula to produce a mixed strategy. In burn:

```rust
fn regret_matching<B: Backend>(regrets: Tensor<B, 1>) -> Tensor<B, 1> {
    // Clamp negative regrets to 0 (only positive regrets drive the strategy)
    let positive_regrets = regrets.clamp_min(0.0);
    let sum = positive_regrets.clone().sum();
    // If all regrets are non-positive, play uniformly
    let n = positive_regrets.dims()[0];
    let uniform = Tensor::full([n], 1.0 / n as f32, &positive_regrets.device());
    // Select: if sum > 0, normalize; otherwise uniform
    let sum_scalar: f32 = sum.clone().into_scalar();
    if sum_scalar > 0.0 {
        positive_regrets / sum
    } else {
        uniform
    }
}
```

## Implementing CFR in Rust

The tabular CFR data structures in Rust map cleanly from the Python/OpenSpiel version. Here is the skeleton.

### Core data structures

```rust
use std::collections::HashMap;

/// An information set identifier: a string unique to each (player, visible history) pair.
/// Same semantics as OpenSpiel's `information_state_string`.
pub type InfoSetKey = String;

/// Cumulative regrets for all actions at one information set.
/// Regret for action a = sum over iterations of (counterfactual value of a - expected value).
#[derive(Debug, Clone)]
pub struct RegretTable {
    /// Cumulative regrets: one entry per action index
    pub regrets: Vec<f64>,
    /// Cumulative strategy weights: for computing the average strategy
    pub strategy_sum: Vec<f64>,
    /// Number of legal actions at this information set
    pub num_actions: usize,
}

impl RegretTable {
    pub fn new(num_actions: usize) -> Self {
        Self {
            regrets: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
            num_actions,
        }
    }

    /// Regret-matching: convert cumulative regrets to a current strategy.
    pub fn current_strategy(&self) -> Vec<f64> {
        let positive: Vec<f64> = self.regrets.iter().map(|r| r.max(0.0)).collect();
        let total: f64 = positive.iter().sum();
        if total > 0.0 {
            positive.iter().map(|r| r / total).collect()
        } else {
            // Uniform strategy when all regrets are non-positive
            vec![1.0 / self.num_actions as f64; self.num_actions]
        }
    }

    /// Average strategy: the time-average of current_strategy over all iterations.
    pub fn average_strategy(&self) -> Vec<f64> {
        let total: f64 = self.strategy_sum.iter().sum();
        if total > 0.0 {
            self.strategy_sum.iter().map(|s| s / total).collect()
        } else {
            vec![1.0 / self.num_actions as f64; self.num_actions]
        }
    }
}

/// The full strategy profile: one RegretTable per information set.
pub struct StrategyProfile {
    pub tables: HashMap<InfoSetKey, RegretTable>,
}

impl StrategyProfile {
    pub fn new() -> Self {
        Self { tables: HashMap::new() }
    }

    /// Get or create the table for a given info set.
    pub fn get_or_create(&mut self, key: &InfoSetKey, num_actions: usize) -> &mut RegretTable {
        self.tables
            .entry(key.clone())
            .or_insert_with(|| RegretTable::new(num_actions))
    }
}
```

### The CFR traversal function

The recursive CFR traversal mirrors the Python/OpenSpiel version exactly. The key difference is Rust's ownership rules: you cannot hold a mutable borrow into `profile.tables` while also passing `profile` to the recursive call. The solution is to collect the information state key and action set before the recursive calls.

```rust
/// Recursive CFR traversal. Returns the expected utility for `traversing_player`
/// under the given reach probabilities.
///
/// - `state`:             current game state (will be cloned for each child)
/// - `profile`:           mutable strategy profile (regret tables)
/// - `reach_probs`:       [p0_reach, p1_reach] — probability that each player
///                        "intends" to reach this node
/// - `traversing_player`: which player's regrets we are updating this pass
pub fn cfr_traverse<G: GameState>(
    state: &G,
    profile: &mut StrategyProfile,
    reach_probs: [f64; 2],
    traversing_player: usize,
) -> f64 {
    if state.is_terminal() {
        return state.returns()[traversing_player];
    }

    if state.is_chance_node() {
        let outcomes = state.chance_outcomes();
        let mut ev = 0.0;
        for (action, prob) in outcomes {
            let mut child = state.clone_state();
            child.apply_action(action);
            ev += prob * cfr_traverse(&child, profile, reach_probs, traversing_player);
        }
        return ev;
    }

    let current = state.current_player();
    let legal = state.legal_actions();
    let num_actions = legal.len();
    let info_key = state.information_state_string(current);

    // Get current strategy for this information set
    let strategy = {
        let table = profile.get_or_create(&info_key, num_actions);
        table.current_strategy()
    };

    // Recursively compute value for each action
    let action_values: Vec<f64> = legal.iter().enumerate().map(|(i, &action)| {
        let mut child = state.clone_state();
        child.apply_action(action);
        let mut new_reach = reach_probs;
        new_reach[current] *= strategy[i];
        cfr_traverse(&child, profile, new_reach, traversing_player)
    }).collect();

    // Expected value under current strategy
    let ev: f64 = action_values.iter().zip(strategy.iter())
        .map(|(v, p)| v * p)
        .sum();

    // Update regrets if this is the traversing player's node
    if current == traversing_player {
        let opponent_reach = reach_probs[1 - traversing_player];
        let table = profile.get_or_create(&info_key, num_actions);
        for (i, &action_val) in action_values.iter().enumerate() {
            // Counterfactual regret = opponent_reach * (action_value - expected_value)
            table.regrets[i] += opponent_reach * (action_val - ev);
            // Accumulate strategy sum weighted by traversing player's reach
            table.strategy_sum[i] += reach_probs[traversing_player] * strategy[i];
        }
    }

    ev
}
```

### Comparison to the Python version

The Python version in OpenSpiel's `cfr.py` has the same logical structure. The main differences:

| Aspect | Python (OpenSpiel) | Rust (capstone) |
|--------|--------------------|-----------------|
| State cloning | `state.clone()` via `copy.deepcopy` | `state.clone_state()` explicit method |
| Info set lookup | `dict[str, np.ndarray]` | `HashMap<String, RegretTable>` |
| Strategy vector | `np.ndarray` | `Vec<f64>` |
| Dispatch | Virtual method through Python ABC | Monomorphized via generic trait |
| Parallelism | GIL-limited | `rayon` parallel iteration |

The logic is identical. The Rust version is faster (no Python overhead, no GC), safer (no accidental aliasing of regret vectors), and more amenable to integration into a larger simulation system.

## Benchmarking Python vs. Rust

How much faster is the Rust CFR? Here is a principled comparison using Kuhn poker as the benchmark game.

### The benchmark

Both implementations run 1 million CFR iterations (alternating-player vanilla CFR) on Kuhn poker. Kuhn poker has 12 information sets and 2 actions per set, so the traversal is very shallow. This benchmark measures pure loop overhead and HashMap access, not algorithmic complexity.

**Python benchmark (OpenSpiel):**

```python
import time
import pyspiel
from open_spiel.python.algorithms import cfr

game = pyspiel.load_game("kuhn_poker")
solver = cfr.CFRSolver(game)

start = time.perf_counter()
for _ in range(1_000_000):
    solver.evaluate_and_update_policy()
elapsed = time.perf_counter() - start

print(f"1,000,000 CFR iterations: {elapsed:.2f}s")
print(f"Throughput: {1_000_000 / elapsed:.0f} iter/s")
```

Expected output on a modern laptop: approximately 8-15 seconds, or 65,000–125,000 iterations per second.

**Rust benchmark:**

```rust
use std::time::Instant;

fn main() {
    let game = KuhnPokerGame::new();
    let mut profile = StrategyProfile::new();

    let start = Instant::now();
    for t in 0..1_000_000 {
        let traversing = t % 2;  // alternate players
        let state = game.new_initial_state();
        cfr_traverse(&state, &mut profile, [1.0, 1.0], traversing);
    }
    let elapsed = start.elapsed();

    println!("1,000,000 CFR iterations: {:.2}s", elapsed.as_secs_f64());
    println!("Throughput: {:.0} iter/s", 1_000_000.0 / elapsed.as_secs_f64());
}
```

Expected output: approximately 0.3-0.8 seconds, or 1.25-3.5 million iterations per second. That is roughly a **10-30x speedup** over the Python version.

### When Python is fine and when Rust is needed

| Scenario | Python adequate? | Rust needed? |
|----------|-----------------|--------------|
| Prototyping a new game structure | Yes | No |
| Running CFR on a game with < 1000 info sets | Yes | No |
| Running CFR on a game with 100k+ info sets | No (too slow) | Yes |
| Deep CFR training loop (GPU backend) | Yes (PyTorch handles it) | No |
| Embedding solver in a production data pipeline | No (GIL, GC) | Yes |
| Real-time conjunction assessment (< 100ms budget) | No | Yes |
| Multi-threaded MCCFR sampling | No (GIL blocks) | Yes |
| Quick exploitability evaluation for research | Yes | No |

The practical rule: use Python/OpenSpiel for research and algorithm development; use Rust for production deployment and performance-critical loops. The capstone is designed to live at the boundary — it is a research artifact, but it is implemented in Rust to prepare you for eventual production use.

## Key Takeaways

- Rust provides memory safety without GC, zero-cost trait abstractions, true multi-threaded parallelism, and no GIL — all of which matter for embedding game-theoretic solvers into production SSA systems where latency, reliability, and integration with C/C++ codebases are constraints.
- The `burn` library is the closest Rust equivalent to PyTorch for training neural networks from scratch; its API differs mainly in the `<B: Backend>` generic parameter that makes computation-backend switching free at compile time.
- Tabular CFR in Rust centers on two data structures — `RegretTable` (cumulative regrets and strategy sums per information set) and `StrategyProfile` (a `HashMap` over information set keys) — that map directly to the Python/OpenSpiel equivalents.
- The Rust CFR traversal is logically identical to Python's but avoids virtual dispatch overhead, garbage collection pauses, and the GIL, yielding roughly 10-30x speedup on small games like Kuhn poker.
- Python with OpenSpiel remains the right tool for prototyping and algorithm development; Rust becomes necessary when the game exceeds ~1000 information sets, when the solver must run in a production pipeline, or when multi-threaded sampling is required.
- The Rust ecosystem currently lacks a general-purpose game-theory framework equivalent to OpenSpiel; this is a known gap, and the capstone is deliberate practice filling that gap for one specific SSA application.

## Quiz

{{#quiz 03-rust-and-burn.toml}}
