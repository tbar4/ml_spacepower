# Module 8: OpenSpiel and the Rust Capstone

## Where this module fits

Modules 1 through 7 built a complete algorithmic toolkit: neural networks, RL, search and planning, game theory, multi-agent RL, and partial observability. Every lesson used toy environments or small Python prototypes. This module converts that toolkit into systems that could actually be deployed — and culminates in a Rust implementation of everything.

There are three distinct deliverables in this module, which is more than any other. They are genuinely separate in purpose:

**The Python pipeline**: OpenSpiel defines and solves your game in research-grade Python. PettingZoo and Ray RLlib train large-scale neural policies against it on a cluster. Lessons 1, 2, and 5 build this pipeline end to end. If you want to train a MAPPO or PSRO agent against your SSA game using 1,000 parallel workers, this is the path.

**The Rust capstone**: There is no Rust-native OpenSpiel equivalent. If you need a CFR solver running in a production system — millisecond latency, no Python interpreter, no garbage collector — you have to build it yourself. Lessons 3 and 4 design the game and the architecture. The project implements it: a `burn`-backed deep CFR solver for the conjunction-masking game, with a CLI and exploitability metrics.

**The non-technical foundation**: Lessons 6 and 7 cover what no algorithm lesson covers — how to get a government contract and how to build an LLM-adjudicated wargame that DoD customers will trust. These are not optional extras; they are the difference between a research prototype and a product.

## What we cover

**OpenSpiel architecture (Lesson 1)**: The three central abstractions — `Game`, `State`, and algorithm APIs — and the secondary abstractions (`Observer`, `Bot`, information state tensors) that you need for custom games. Goal: a clear mental model of which class to subclass, which file to read, and why the design looks the way it does. This mental model directly informs the Rust capstone architecture.

**Implementing a custom game (Lesson 2)**: Building a two-player imperfect-information sequential game in Python, from scratch, inside the OpenSpiel framework. The working example is Mini Maneuver — a deliberately small orbital game that forces you to implement every OpenSpiel hook correctly before moving to the full SSA game. The Python implementation is the specification the Rust capstone implements.

**Rust and burn: the production gap (Lesson 3)**: An honest audit of what the Rust ML ecosystem has, what it lacks, and what this means for the capstone design. Covers the `burn` deep learning framework (the only viable Rust option for neural network training), the absence of a Rust-native OpenSpiel, and the design choices forced by these realities. This is the context lesson before the build.

**Designing the SSA game (Lesson 4)**: The conjunction-masking game — the specific game the capstone implements — is designed here. Every design choice (two-player structure, single-shot for tractability, Adversary private information about maneuver intent, Defender limited sensor allocation) is explained in terms of the strategic assumption it encodes. The SSA strategic motivation from Module SP connects directly to the formal game structure here.

**PettingZoo, shimmy, and Ray RLlib (Lesson 5)**: The four-layer integration stack that connects OpenSpiel games to distributed RL training: OpenSpiel → shimmy compatibility wrapper → PettingZoo AEC environment → Ray RLlib `MultiAgentEnv`. Every adapter in the stack is explained explicitly, including the configuration for self-play, the parallelism math, and how MAPPO (Module 6) and APPO/IMPALA (Module 3) map to RLlib's training configuration.

**SBIR and government contracting (Lesson 6)**: The DoD innovation pipeline — SBIR/STTR, SpaceWERX, Other Transaction Authorities — mapped honestly for a small technical founder. Covers eligibility requirements, Phase I/II mechanics, the commercial-first vs. SBIR-first trade-off, ITAR basics, and the clearance path. The ML products in this curriculum have a direct route to government funding through these mechanisms.

**LLM-in-the-loop wargame adjudication (Lesson 7)**: Using a locally deployed language model as a wargame umpire — evaluating player actions against a rule set and producing consistent, auditable outcomes at the scale needed for RL training. Covers FedRAMP compliance constraints that force local deployment, the matrix game format that makes LLM adjudication auditable, prompt injection mitigations, and the combination of LLM adjudication with CFR or RL so agents can learn to play against the umpire model itself.

## Lessons

1. [OpenSpiel architecture](01-openspiel-architecture.md)
2. [Implementing a custom game](02-implementing-custom-games.md)
3. [Rust and burn: the production gap](03-rust-and-burn.md)
4. [Designing the SSA game](04-designing-the-ssa-game.md)
5. [PettingZoo, shimmy, and Ray RLlib](05-pettingzoo-rllib.md)
6. [From research to revenue: SBIR and government contracting](06-sbir-spacewerx.md)
7. [LLM-in-the-loop wargame adjudication](07-llm-wargame-adjudication.md)

## Module project: Rust CFR solver for a conjunction-masking game

The capstone is a self-contained Rust crate, `ssa_cfr`, that implements the full conjunction-masking game and a CFR solver over it. Specifically:

- A `Game` trait and a `State` trait mirroring the OpenSpiel abstractions, implemented in Rust
- The basic conjunction-masking game from Lesson 4, in Rust
- A vanilla CFR solver that computes Nash-approximating strategies and reports exploitability
- A scaled variant of the game with a larger action and chance space
- A deep CFR variant using `burn` neural networks to approximate regret values, replacing the tabular regret table for larger game instances
- A CLI that runs self-play, prints the equilibrium strategy profile, and outputs exploitability at each iteration

This is the artifact that connects the thesis claim to a deployable system. The Rust CFR solver is what you would embed in a production orbital intelligence pipeline — no Python runtime, no OpenSpiel dependency, exploitability metrics you can cite in a thesis chapter.

## How this module connects to everything before it

Every module contributed something to this capstone:

- **Module 0**: The SSA domain semantics behind the conjunction-masking game — what a maneuver is, what a sensor allocation constraint means, why detection latency matters
- **Module 1**: The probability framework for Bayesian belief updates in the Defender's information state
- **Module 2**: The `burn` neural network used in deep CFR — the MLP architecture, the training loop, the loss function
- **Module 3**: IMPALA and APPO, which Ray RLlib uses to train large-scale policies against OpenSpiel games
- **Module 4**: IS-MCTS as the inference-time planner alternative to CFR for fog-of-war games; the AlphaZero architecture that Lesson 5's RLlib pipeline trains
- **Module 5**: CFR — the algorithm the Rust capstone implements
- **Module 6**: PSRO and MAPPO — the multi-agent training methods wired to the RLlib pipeline in Lesson 5
- **Module 7**: Particle filters and opponent modeling — the belief-state machinery underlying the Defender's information set
