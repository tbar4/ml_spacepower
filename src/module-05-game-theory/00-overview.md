# Module 5: Game Theory and Equilibrium Computation

## Where this module fits

Until now, we have treated decision-making in two ways: single-agent (Modules 1-3) and two-player perfect-information (Module 4). Real SSA scenarios are messier. They involve multiple agents (operators, debris-flagging services, adversarial actors). They often involve imperfect information (you cannot see what the other operator is doing or planning). And in cooperative or adversarial multi-agent settings, "the optimal policy" is no longer a single thing: it depends on what the other agents are doing.

Game theory is the framework for this. Nash equilibria are the natural notion of "stable" multi-agent strategies: configurations where no agent can improve by unilaterally deviating. Counterfactual Regret Minimization (CFR) is the algorithm of choice for finding Nash equilibria in extensive-form games (games played out as sequences of decisions, possibly with hidden information). MCCFR is its sample-based variant. Deep CFR uses neural networks for function approximation.

This module is the conceptual heart of the curriculum and the most Rust-relevant. The capstone (Module 8) implements a custom CFR variant in Rust.

## What we cover

**Normal-form and extensive-form games (lesson 1)**: the formal language of game theory. Strategy profiles, Nash equilibrium, the difference between simultaneous-move (normal-form) and sequential (extensive-form) games. Information sets for hidden information.

**Extensive-form games in detail (lesson 2)**: game trees with chance nodes and information sets. Strategies vs. policies (yes, the distinction matters in game theory). Reach probabilities: how likely is a particular history given a strategy profile?

**Counterfactual Regret Minimization (lesson 3)**: the heart of the module. Counterfactual values, regret matching, why it converges to Nash. We work through CFR on a small game by hand to make the algorithm concrete.

**Monte Carlo CFR (lesson 4)**: vanilla CFR is impractical for large games (it sweeps the entire game tree every iteration). Outcome sampling and external sampling are the two main variants that make CFR tractable for large games. The variance-vs-speed tradeoff.

**Deep CFR (lesson 5)**: replace the per-information-set regret table with a neural network. Use sampled traversals as training data. This is the algorithm that produced superhuman poker play (Pluribus, Libratus).

## Lessons

1. [Normal-form and extensive-form games](01-normal-form-and-extensive-form.md)
2. [Extensive-form games in detail](02-extensive-form-games-in-detail.md)
3. [Counterfactual Regret Minimization (CFR)](03-counterfactual-regret-minimization.md)
4. [Monte Carlo CFR (MCCFR)](04-monte-carlo-cfr.md)
5. [Deep CFR](05-deep-cfr.md)

## Module project: a CFR solver for an SSA negotiation game

You will implement vanilla CFR (and optionally MCCFR) for a small extensive-form game: two satellite operators are facing a potential conjunction. Each must decide whether to maneuver. The catch: each operator pays a cost for maneuvering (fuel, mission disruption), but if neither maneuvers, both suffer a much larger cost (potential collision). This is a Stackelberg-flavored coordination game with imperfect information about the other operator's intent.

You will compute the Nash equilibrium and analyze what it tells you about strategic behavior in conjunction-avoidance situations. You will also do this with both vanilla CFR (to see the algorithm clearly) and MCCFR (to see how the sampling speeds things up).

The CFR data structures and algorithms are small enough to implement cleanly in either Python or Rust. We provide a Python reference; you are encouraged to also write a Rust translation, since this module's project is the most direct preview of the capstone.
