# Module 8 Capstone: A Rust CFR Solver for an SSA Conjunction-Masking Game


<!-- toc -->

## What you are building

A self-contained Rust crate, `ssa_cfr`, that implements:

1. The conjunction-masking game from lesson 4, as a Rust struct implementing a `Game` trait.
2. A vanilla CFR solver that reads from the trait and produces a Nash-approximating strategy.
3. A best-response calculator that computes the exploitability of any strategy profile (your CFR convergence metric).
4. A scaled variant of the game with more actions and chance outcomes.
5. A deep CFR variant using `burn` to approximate regret values with a neural network.
6. A command-line interface that runs the above and produces inspectable output.

This is the artifact that justifies the curriculum. By the end you will have working Rust code, in your strongest language, that solves a small but genuine adversarial SSA problem.

## Project structure

```
ssa_cfr/
├── Cargo.toml                  # workspace manifest
├── README.md                   # how to run
├── crates/
│   ├── game/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # Game and GameState traits
│   │       ├── basic.rs        # the basic conjunction-masking game
│   │       └── scaled.rs       # the scaled variant
│   ├── solver/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # public solver interface
│   │       ├── cfr.rs          # tabular vanilla CFR
│   │       ├── best_response.rs  # exploitability calculation
│   │       └── deep_cfr.rs     # neural network variant (feature-gated)
│   └── cli/
│       ├── Cargo.toml
│       └── src/
│           └── main.rs         # the CLI entry point
└── tests/
    └── integration.rs          # cross-crate tests
```

## Step 1: workspace setup

Create the workspace:

```bash
mkdir ssa_cfr && cd ssa_cfr
cargo init --vcs git
```

Replace the top-level `Cargo.toml` with a workspace manifest:

```toml
[workspace]
resolver = "2"
members = [
    "crates/game",
    "crates/solver",
    "crates/cli",
]

[workspace.package]
edition = "2021"
version = "0.1.0"
authors = ["Trevor Barnes"]

[workspace.dependencies]
rand = "0.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
burn = { version = "0.13", default-features = false, features = ["ndarray"] }
```

Create the three crates:

```bash
cargo new --lib crates/game
cargo new --lib crates/solver
cargo new --bin crates/cli
```

For each, update `Cargo.toml`'s `[package]` section to use the workspace inheritance:

```toml
[package]
name = "ssa_game"  # or ssa_solver, ssa_cli
version.workspace = true
edition.workspace = true
authors.workspace = true
```

## Step 2: the Game and GameState traits (game crate)

`crates/game/src/lib.rs`:

```rust
//! Core traits for two-player extensive-form games.

pub mod basic;
pub mod scaled;

use std::fmt::Debug;

/// Identifies who acts at a given decision point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Player(u8),
    Chance,
    Terminal,
}

/// A game's rules, separate from any particular state.
pub trait Game {
    type State: GameState;
    fn new_initial_state(&self) -> Self::State;
    fn num_players(&self) -> usize;
    fn num_distinct_actions(&self) -> usize;
}

/// A particular position in a game.
pub trait GameState: Debug {
    fn current_player(&self) -> Player;
    fn legal_actions(&self) -> Vec<usize>;
    fn chance_outcomes(&self) -> Vec<(usize, f64)>;
    fn apply_action(&mut self, action: usize);
    fn information_state_string(&self, player: u8) -> String;
    fn information_state_tensor(&self, player: u8) -> Vec<f32>;
    fn is_terminal(&self) -> bool;
    fn is_chance_node(&self) -> bool;
    fn returns(&self) -> Vec<f64>;
    fn clone_state(&self) -> Self;
}
```

## Step 3: the basic game (game crate)

`crates/game/src/basic.rs`:

```rust
//! The basic conjunction-masking game from Module 8 Lesson 4.

use crate::{Game, GameState, Player};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opportunity {
    Routine,
    Maneuver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intensity {
    None,
    Light,
    Heavy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allocation {
    Wide,
    Narrow,
    Off,
}

const ADVERSARY: u8 = 0;
const DEFENDER:  u8 = 1;

pub struct BasicGame;

impl Game for BasicGame {
    type State = BasicState;
    
    fn new_initial_state(&self) -> Self::State {
        BasicState::default()
    }
    
    fn num_players(&self) -> usize { 2 }
    fn num_distinct_actions(&self) -> usize { 3 }
}

#[derive(Debug, Clone)]
pub struct BasicState {
    opportunity:      Option<Opportunity>,
    adversary_action: Option<Intensity>,
    defender_action:  Option<Allocation>,
    detection:        Option<bool>,
}

impl Default for BasicState {
    fn default() -> Self {
        Self { opportunity: None, adversary_action: None,
               defender_action: None, detection: None }
    }
}

impl BasicState {
    fn detect_prob(&self, intensity: Intensity, allocation: Allocation) -> f64 {
        match (intensity, allocation) {
            (Intensity::None,  Allocation::Wide)   => 0.05,
            (Intensity::None,  Allocation::Narrow) => 0.05,
            (Intensity::None,  Allocation::Off)    => 0.00,
            (Intensity::Light, Allocation::Wide)   => 0.50,
            (Intensity::Light, Allocation::Narrow) => 0.30,
            (Intensity::Light, Allocation::Off)    => 0.00,
            (Intensity::Heavy, Allocation::Wide)   => 0.65,
            (Intensity::Heavy, Allocation::Narrow) => 0.85,
            (Intensity::Heavy, Allocation::Off)    => 0.00,
        }
    }
    
    fn adversary_payoff(&self) -> f64 {
        let opp = self.opportunity.unwrap();
        let int = self.adversary_action.unwrap();
        let det = self.detection.unwrap();
        
        match (opp, int, det) {
            (Opportunity::Maneuver, Intensity::None,  _)     => 0.0,
            (Opportunity::Maneuver, Intensity::Light, false) => 1.0,
            (Opportunity::Maneuver, Intensity::Light, true)  => -3.0,
            (Opportunity::Maneuver, Intensity::Heavy, false) => 2.0,
            (Opportunity::Maneuver, Intensity::Heavy, true)  => -3.0,
            (Opportunity::Routine,  Intensity::None,  _)     => 0.0,
            (Opportunity::Routine,  Intensity::Light, false) => 0.0,
            (Opportunity::Routine,  Intensity::Light, true)  => -2.0,
            (Opportunity::Routine,  Intensity::Heavy, false) => 0.0,
            (Opportunity::Routine,  Intensity::Heavy, true)  => -2.0,
        }
    }
}

impl GameState for BasicState {
    fn current_player(&self) -> Player {
        if self.detection.is_some() { return Player::Terminal; }
        if self.opportunity.is_none() { return Player::Chance; }
        if self.adversary_action.is_none() { return Player::Player(ADVERSARY); }
        if self.defender_action.is_none() { return Player::Player(DEFENDER); }
        Player::Chance  // detection chance event
    }
    
    fn legal_actions(&self) -> Vec<usize> {
        if self.is_terminal() { return vec![]; }
        if self.is_chance_node() {
            if self.opportunity.is_none() {
                return vec![0, 1];  // Routine, Maneuver
            } else {
                return vec![0, 1];  // detected, not detected
            }
        }
        vec![0, 1, 2]  // 3 actions per player decision
    }
    
    fn chance_outcomes(&self) -> Vec<(usize, f64)> {
        if self.opportunity.is_none() {
            return vec![(0, 0.6), (1, 0.4)];  // Routine more common
        }
        // Detection chance: probability depends on intensity and allocation
        let p = self.detect_prob(self.adversary_action.unwrap(),
                                 self.defender_action.unwrap());
        vec![(1, p), (0, 1.0 - p)]  // 1 = detected
    }
    
    fn apply_action(&mut self, action: usize) {
        if self.opportunity.is_none() {
            self.opportunity = Some(if action == 0 { Opportunity::Routine }
                                    else { Opportunity::Maneuver });
        } else if self.adversary_action.is_none() {
            self.adversary_action = Some(match action {
                0 => Intensity::None,
                1 => Intensity::Light,
                _ => Intensity::Heavy,
            });
        } else if self.defender_action.is_none() {
            self.defender_action = Some(match action {
                0 => Allocation::Wide,
                1 => Allocation::Narrow,
                _ => Allocation::Off,
            });
        } else {
            self.detection = Some(action == 1);
        }
    }
    
    fn information_state_string(&self, player: u8) -> String {
        match player {
            ADVERSARY => match self.opportunity {
                Some(Opportunity::Routine)  => "opp=R".to_string(),
                Some(Opportunity::Maneuver) => "opp=M".to_string(),
                None => String::new(),
            },
            DEFENDER => String::new(),
            _ => panic!("invalid player"),
        }
    }
    
    fn information_state_tensor(&self, player: u8) -> Vec<f32> {
        match player {
            ADVERSARY => match self.opportunity {
                Some(Opportunity::Routine)  => vec![1.0, 0.0],
                Some(Opportunity::Maneuver) => vec![0.0, 1.0],
                None => vec![0.0, 0.0],
            },
            DEFENDER => vec![],
            _ => panic!("invalid player"),
        }
    }
    
    fn is_terminal(&self) -> bool {
        self.detection.is_some()
    }
    
    fn is_chance_node(&self) -> bool {
        if self.is_terminal() { return false; }
        self.opportunity.is_none()
            || (self.adversary_action.is_some() && self.defender_action.is_some()
                && self.detection.is_none())
    }
    
    fn returns(&self) -> Vec<f64> {
        if !self.is_terminal() {
            return vec![0.0, 0.0];
        }
        let adv = self.adversary_payoff();
        vec![adv, -adv]  // zero-sum
    }
    
    fn clone_state(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_initial_state_is_chance() {
        let game = BasicGame;
        let state = game.new_initial_state();
        assert_eq!(state.current_player(), Player::Chance);
        assert_eq!(state.chance_outcomes(), vec![(0, 0.6), (1, 0.4)]);
    }
    
    #[test]
    fn test_full_trajectory() {
        let game = BasicGame;
        let mut state = game.new_initial_state();
        state.apply_action(1);  // Maneuver opportunity
        assert_eq!(state.current_player(), Player::Player(0));
        state.apply_action(2);  // Heavy intensity
        assert_eq!(state.current_player(), Player::Player(1));
        state.apply_action(1);  // Narrow allocation
        assert_eq!(state.current_player(), Player::Chance);
        state.apply_action(1);  // detected
        assert!(state.is_terminal());
        // Heavy maneuver detected, opportunity exists: payoff -3
        assert_eq!(state.returns(), vec![-3.0, 3.0]);
    }
    
    #[test]
    fn test_information_state_strings() {
        let game = BasicGame;
        let mut state = game.new_initial_state();
        state.apply_action(1);  // Maneuver
        assert_eq!(state.information_state_string(0), "opp=M");
        assert_eq!(state.information_state_string(1), "");
    }
}
```

## Step 4: tabular CFR (solver crate)

`crates/solver/src/cfr.rs`:

```rust
//! Tabular Counterfactual Regret Minimization.

use ssa_game::{Game, GameState, Player};
use std::collections::HashMap;

/// A regret table: information state string -> per-action cumulative regret.
pub type RegretTable = HashMap<String, Vec<f64>>;

/// A strategy table: information state string -> per-action cumulative strategy.
pub type StrategyTable = HashMap<String, Vec<f64>>;

pub struct CfrSolver {
    pub regrets:           RegretTable,
    pub strategy_sum:      StrategyTable,
    pub iterations:        usize,
    num_actions_per_info:  HashMap<String, usize>,
}

impl CfrSolver {
    pub fn new() -> Self {
        Self {
            regrets:              HashMap::new(),
            strategy_sum:         HashMap::new(),
            iterations:           0,
            num_actions_per_info: HashMap::new(),
        }
    }
    
    /// Run one CFR iteration over the entire game tree from the root.
    /// In zero-sum 2-player games, each iteration walks the tree once
    /// per player (training each player against the current strategy).
    pub fn run_iteration<G: Game>(&mut self, game: &G) {
        for traversing_player in 0..game.num_players() as u8 {
            let state = game.new_initial_state();
            self.cfr(state, traversing_player, 1.0, 1.0);
        }
        self.iterations += 1;
    }
    
    /// Recursive CFR.
    /// `traversing_player`: the player for whom we are computing regrets this pass.
    /// `pi_p`: the reach probability for the traversing player (product of their
    ///         action probabilities along the path so far).
    /// `pi_o`: the reach probability for the opponent and chance (everyone else).
    fn cfr<S: GameState>(&mut self, state: S, traversing_player: u8,
                         pi_p: f64, pi_o: f64) -> f64 {
        if state.is_terminal() {
            return state.returns()[traversing_player as usize];
        }
        
        if state.is_chance_node() {
            let mut value = 0.0;
            for (action, prob) in state.chance_outcomes() {
                let mut next = state.clone_state();
                next.apply_action(action);
                value += prob * self.cfr(next, traversing_player, pi_p, pi_o * prob);
            }
            return value;
        }
        
        let player = match state.current_player() {
            Player::Player(p) => p,
            _ => unreachable!(),
        };
        let info_str = state.information_state_string(player);
        let legal = state.legal_actions();
        let n = legal.len();
        
        self.num_actions_per_info.entry(info_str.clone()).or_insert(n);
        let regrets = self.regrets.entry(info_str.clone()).or_insert_with(|| vec![0.0; n]);
        let strategy = regret_matching(regrets);
        
        // Update cumulative strategy (weighted by reach probability)
        let strat_sum = self.strategy_sum.entry(info_str.clone())
                                        .or_insert_with(|| vec![0.0; n]);
        let weight = if player == traversing_player { pi_p } else { pi_o };
        for a in 0..n {
            strat_sum[a] += weight * strategy[a];
        }
        
        // Compute action utilities and node value
        let mut action_utils = vec![0.0; n];
        let mut node_value = 0.0;
        for (i, &action) in legal.iter().enumerate() {
            let mut next = state.clone_state();
            next.apply_action(action);
            let util = if player == traversing_player {
                self.cfr(next, traversing_player, pi_p * strategy[i], pi_o)
            } else {
                self.cfr(next, traversing_player, pi_p, pi_o * strategy[i])
            };
            action_utils[i] = util;
            node_value += strategy[i] * util;
        }
        
        // Update regrets only for the traversing player
        if player == traversing_player {
            let regrets_mut = self.regrets.get_mut(&info_str).unwrap();
            for a in 0..n {
                let regret = action_utils[a] - node_value;
                regrets_mut[a] += pi_o * regret;  // counterfactual: weight by opp/chance
            }
        }
        
        node_value
    }
    
    /// Extract the average strategy: cumulative strategy normalized to a distribution.
    /// This is the strategy that converges to Nash, NOT the current iteration's strategy.
    pub fn average_strategy(&self) -> HashMap<String, Vec<f64>> {
        let mut out = HashMap::new();
        for (info, sums) in &self.strategy_sum {
            let total: f64 = sums.iter().sum();
            let n = sums.len();
            let dist = if total > 0.0 {
                sums.iter().map(|&s| s / total).collect()
            } else {
                vec![1.0 / n as f64; n]
            };
            out.insert(info.clone(), dist);
        }
        out
    }
}

/// Regret matching: convert regrets to a strategy (distribution over actions).
/// Positive regret -> proportional probability. All-zero -> uniform.
fn regret_matching(regrets: &[f64]) -> Vec<f64> {
    let pos: Vec<f64> = regrets.iter().map(|&r| r.max(0.0)).collect();
    let total: f64 = pos.iter().sum();
    let n = regrets.len();
    if total > 0.0 {
        pos.iter().map(|&p| p / total).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}
```

## Step 5: best-response and exploitability

`crates/solver/src/best_response.rs`:

```rust
//! Best-response computation and exploitability calculation.

use ssa_game::{Game, GameState, Player};
use std::collections::HashMap;

pub type Strategy = HashMap<String, Vec<f64>>;

/// Compute the value of `responding_player` playing a best response
/// against `opponent_strategy`.
pub fn best_response_value<G: Game>(
    game: &G,
    responding_player: u8,
    opponent_strategy: &Strategy,
) -> f64 {
    let state = game.new_initial_state();
    br_value(state, responding_player, opponent_strategy)
}

fn br_value<S: GameState>(
    state: S, responding_player: u8, opp_strat: &Strategy,
) -> f64 {
    if state.is_terminal() {
        return state.returns()[responding_player as usize];
    }
    if state.is_chance_node() {
        let mut value = 0.0;
        for (action, prob) in state.chance_outcomes() {
            let mut next = state.clone_state();
            next.apply_action(action);
            value += prob * br_value(next, responding_player, opp_strat);
        }
        return value;
    }
    let player = match state.current_player() {
        Player::Player(p) => p,
        _ => unreachable!(),
    };
    let legal = state.legal_actions();
    if player == responding_player {
        // Best response: pick the action that maximizes value
        let mut best = f64::NEG_INFINITY;
        for &a in &legal {
            let mut next = state.clone_state();
            next.apply_action(a);
            let v = br_value(next, responding_player, opp_strat);
            if v > best { best = v; }
        }
        best
    } else {
        // Opponent: average over their (fixed) strategy
        let info = state.information_state_string(player);
        let strat = opp_strat.get(&info).cloned()
            .unwrap_or_else(|| vec![1.0 / legal.len() as f64; legal.len()]);
        let mut value = 0.0;
        for (i, &a) in legal.iter().enumerate() {
            let mut next = state.clone_state();
            next.apply_action(a);
            value += strat[i] * br_value(next, responding_player, opp_strat);
        }
        value
    }
}

/// Compute exploitability: the average gain per player when each switches
/// to a best response. Zero means Nash equilibrium.
pub fn exploitability<G: Game>(
    game: &G, strategy_per_player: &[Strategy],
) -> f64 {
    assert_eq!(strategy_per_player.len(), game.num_players());
    let mut total = 0.0;
    for p in 0..game.num_players() {
        // Compute best-response value for player p against opponents
        // (in a 2-player game, just the other player)
        let opp_strat = &strategy_per_player[1 - p];
        let br_val = best_response_value(game, p as u8, opp_strat);
        total += br_val;
    }
    total / game.num_players() as f64
}
```

A note on the strategy structure: for our 2-player game, both players' strategies could in principle be stored in one HashMap because the information state strings are unique across players (Adversary uses "opp=R" or "opp=M"; Defender uses ""). For general games, you would key by (player, info_string). The capstone keeps it simple by using the same combined HashMap and being careful about which info sets belong to which player.

## Step 6: solver crate plumbing

`crates/solver/src/lib.rs`:

```rust
pub mod cfr;
pub mod best_response;

#[cfg(feature = "nn")]
pub mod deep_cfr;

pub use cfr::CfrSolver;
pub use best_response::{exploitability, best_response_value, Strategy};
```

`crates/solver/Cargo.toml`:

```toml
[package]
name = "ssa_solver"
version.workspace = true
edition.workspace = true

[dependencies]
ssa_game = { path = "../game" }
rand = { workspace = true }
burn = { workspace = true, optional = true }

[features]
default = []
nn = ["burn"]
```

Notice the `nn` feature gating `burn`. Users who only want tabular CFR build without the feature; the network code is opt-in.

## Step 7: the CLI

`crates/cli/src/main.rs`:

```rust
use clap::{Parser, Subcommand};
use ssa_game::{basic::BasicGame, Game};
use ssa_solver::{exploitability, CfrSolver, Strategy};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "ssa_cfr")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train tabular CFR and report exploitability over time.
    Train {
        #[arg(long, default_value_t = 10000)]
        iterations: usize,
        #[arg(long, default_value_t = 500)]
        report_every: usize,
    },
    /// Print the average strategy after training.
    Strategy {
        #[arg(long, default_value_t = 10000)]
        iterations: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    let game = BasicGame;
    
    match cli.command {
        Command::Train { iterations, report_every } => {
            let mut solver = CfrSolver::new();
            for i in 1..=iterations {
                solver.run_iteration(&game);
                if i % report_every == 0 || i == 1 {
                    let strat = solver.average_strategy();
                    let p0 = filter_strategy(&strat, 0);
                    let p1 = filter_strategy(&strat, 1);
                    let exp = exploitability(&game, &[p0, p1]);
                    println!("iter {:6}: exploitability = {:.6}", i, exp);
                }
            }
        }
        Command::Strategy { iterations } => {
            let mut solver = CfrSolver::new();
            for _ in 0..iterations {
                solver.run_iteration(&game);
            }
            let strat = solver.average_strategy();
            println!("=== Average strategy ===");
            for (info, dist) in &strat {
                let label = if info.is_empty() { "(defender)" } else { info };
                println!("  {:>15}: {:?}", label, dist);
            }
        }
    }
}

/// Split a combined strategy table by player.
/// Adversary info sets start with "opp="; Defender's info set is the empty string.
fn filter_strategy(combined: &Strategy, player: u8) -> Strategy {
    let mut out = HashMap::new();
    for (info, dist) in combined {
        let belongs_to_player = if info.starts_with("opp=") { 0 } else { 1 };
        if belongs_to_player == player {
            out.insert(info.clone(), dist.clone());
        }
    }
    out
}
```

`crates/cli/Cargo.toml`:

```toml
[package]
name = "ssa_cli"
version.workspace = true
edition.workspace = true

[[bin]]
name = "ssa_cfr"
path = "src/main.rs"

[dependencies]
ssa_game = { path = "../game" }
ssa_solver = { path = "../solver" }
clap = { workspace = true }
```

## Step 8: run it

```bash
cargo build --release
./target/release/ssa_cfr train --iterations 5000 --report-every 500
```

Expected output (numbers are illustrative; small differences are normal):

```
iter      1: exploitability = 0.847123
iter    500: exploitability = 0.012345
iter   1000: exploitability = 0.005678
iter   1500: exploitability = 0.003421
iter   2000: exploitability = 0.002134
...
iter   5000: exploitability = 0.000412
```

Exploitability decreasing toward zero confirms convergence to Nash. Then:

```bash
./target/release/ssa_cfr strategy --iterations 5000
```

```
=== Average strategy ===
        opp=M: [0.083, 0.124, 0.793]   # Adversary on Maneuver: mostly Heavy, some Light
        opp=R: [0.972, 0.018, 0.010]   # Adversary on Routine: almost always None
   (defender): [0.418, 0.362, 0.220]   # Defender mixes Wide/Narrow, some Off
```

The exact numbers depend on convergence and your random initialization, but the qualitative pattern (Adversary mostly maneuvers Heavy when given the opportunity, Adversary almost never maneuvers without opportunity, Defender randomizes between Wide and Narrow with low Off rate) is what you should see.

## Step 9: deep CFR variant (with the `nn` feature)

This is the longer module of the project. The deep CFR file (`crates/solver/src/deep_cfr.rs`) implements:

1. A `RegretNetwork` struct using `burn` (an MLP that takes an information state tensor and outputs predicted regret per action).
2. A buffer of `(info_tensor, regret_vec)` samples accumulated over CFR iterations.
3. A training loop that trains the network on the buffer between CFR iterations.
4. Strategy extraction: at each information state, query the network for predicted regrets and apply regret matching.

The structure parallels the tabular version. The recursive tree walk is the same; the difference is in step "look up regrets for this info state" (HashMap lookup → network forward pass) and "store new regrets" (HashMap update → buffer append + periodic training).

We do not write the full deep CFR code in this document because it is mechanical translation of patterns you have already seen (Module 5 lesson 5 covered deep CFR conceptually; Module 8 lesson 3 showed the burn syntax). Implementing it is the natural extension exercise. A skeleton:

```rust
// crates/solver/src/deep_cfr.rs
#[cfg(feature = "nn")]
use burn::{module::Module, nn::{Linear, LinearConfig, Relu}, 
           tensor::{backend::Backend, Tensor}};

#[derive(Module, Debug)]
pub struct RegretNetwork<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

impl<B: Backend> RegretNetwork<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, num_actions: usize, device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(input_dim, hidden_dim).init(device),
            layer2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            output: LinearConfig::new(hidden_dim, num_actions).init(device),
            activation: Relu::new(),
        }
    }
    
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(x));
        let x = self.activation.forward(self.layer2.forward(x));
        self.output.forward(x)
    }
}

// Reservoir buffer:
// pub struct RegretBuffer { samples: Vec<(Vec<f32>, Vec<f64>, f64)> }
// Each tuple is (info_tensor, regret_vec, weight).
// Reservoir sampling keeps the buffer size bounded.

// Training loop (between CFR iterations):
// 1. Sample a batch from the buffer
// 2. Forward pass: predicted_regret = network(info_tensor)
// 3. Loss: MSE between predicted and target regrets, weighted
// 4. Backward + optimizer step

// Strategy extraction:
// For each info state, get info_tensor, run network, apply regret_matching
// to the output.
```

The key design decisions:
- **Reservoir buffer**: bounded-size sampling without bias.
- **Per-player networks**: one regret network per player, since they have different action spaces (in the scaled game, this matters more).
- **Verification**: train deep CFR on the basic game; check that exploitability still drops to near zero (a sanity check that the network learned the right regret function).

Because the basic game has only 3 information sets, the network is overkill there. The point of running deep CFR on the basic game is to verify the implementation is correct (the tabular oracle gives you the ground truth). Then you can scale up to the larger variant from lesson 4 with confidence.

## Step 10: scaled game and reflection

The scaled game (`crates/game/src/scaled.rs`) extends the basic game with:
- 7 maneuver intensity levels (instead of 3)
- 5 sensor allocation modes (instead of 3)  
- 4 chance opportunity types (instead of 2)
- Detection probability table extended accordingly

The implementation is mechanical: same trait, more enum variants, larger payoff table. With 4 opportunities, the Adversary has 4 information sets; with the added action richness, the strategy has many more parameters. Tabular CFR still works (the table is at most a few KB) but the deep CFR variant becomes the more natural choice as you scale further.

Run both on the scaled game and compare strategies. They should largely agree.

### Reflection

1. **Convergence rate**: How many CFR iterations are needed for the basic game's exploitability to drop below 0.001? What about the scaled game?

2. **Correctness check**: Compute the equilibrium of the basic game by hand (it's a 3x3-style game; you can solve it as a small LP) and compare to your CFR output.

3. **Implementation tradeoffs**: What was hardest to get right in Rust compared to Python? What was easier?

4. **Extending to multi-shot**: The current game is single-shot. Sketch (don't implement) what a 5-step variant would look like. How would you represent state? Would tabular CFR scale?

5. **Embedding in larger systems**: How would you integrate this crate into a larger SSA simulation? What APIs would the simulation need from your solver?

## What you have built

- A complete Rust crate implementing extensive-form game solving with both tabular and deep CFR variants
- A specific game capturing essential adversarial structure of an SSA scenario
- Working CFR convergence to Nash equilibrium with verifiable exploitability metric
- A starting point for your thesis-scale work

You can extend this in any direction your research requires: more complex games, different solver algorithms (Public CFR, Deep CFR variants, Online Outcome Sampling), different architectures, different scenarios. The infrastructure is yours.

## Where you could go next

- Read OpenSpiel's CFR Python implementation in detail and compare to your Rust version. You will see they are structurally near-identical.
- Implement MCCFR (the Module 5 outside-sampling variant) over your trait. It should be a few hundred lines.
- Build a multi-shot version of your game, where the adversary repeats over multiple opportunities and the defender accumulates evidence over time.
- Study more sophisticated equilibrium concepts: subgame-perfect equilibrium, sequential equilibrium, beyond Nash.
- For your thesis: pick a specific space domain awareness problem (sensor tasking, debris-conjunction decision making, signaling games for collision avoidance) and design a game for it. The capstone gives you the template.

This curriculum has taken you from "I do not know what a probability distribution is" to "I have implemented a CFR solver in Rust." That is a real distance. Most people who ostensibly know ML cannot implement CFR. You can.

The specific algorithms will evolve as research evolves. The frameworks will improve. The ideas (probability, value functions, policy gradients, search, equilibrium computation, belief tracking) will not. You now have the foundational toolkit to read papers in this area, implement what you read, and extend it for your own purposes.

Good luck with the thesis.
