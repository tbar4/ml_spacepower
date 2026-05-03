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

## Quiz

{{#quiz 03-rust-and-burn.toml}}
