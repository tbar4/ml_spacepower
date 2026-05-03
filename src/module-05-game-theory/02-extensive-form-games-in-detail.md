# Lesson 2: Extensive-Form Games in Detail

## Where this fits

Lesson 1 introduced extensive-form games at a high level. CFR (lesson 3) operates on the detailed structure of these games: information sets, reach probabilities, and strategies defined as policies over information sets. This lesson develops that structure precisely. The vocabulary here is exactly the vocabulary used in OpenSpiel's API and in CFR research papers.

## The components of an extensive-form game

A formal extensive-form game has:

1. **A finite set of players** (we will mostly use two-player games)
2. **A game tree** with nodes representing decision points and edges representing actions
3. **A player function** that says whose turn it is at each non-terminal node (one of the players, or "chance")
4. **A chance function** that gives the probability distribution over actions at chance nodes
5. **Information sets**: for each player, a partition of their decision nodes into sets of nodes they cannot distinguish
6. **Utility functions**: for each player, a function from terminal nodes to real numbers (the payoff)

That is a lot. Let us walk through each piece with our running SSA-flavored example.

## A small extensive-form SSA game

Two satellite operators, Alice and Bob, face a potential conjunction. Each operator has a "mission state" that is private (hidden from the other): either "high-priority" or "low-priority." Maneuvering costs more for high-priority operators (they want to stay on station for their mission).

The game proceeds:
1. Chance assigns each operator a mission state (50/50 high or low, independently)
2. Alice decides whether to maneuver (M) or hold (H)
3. Bob, who can see whether Alice maneuvered (but not Alice's mission state), decides M or H
4. Payoffs are determined by the joint action and both operators' mission states

This is a 2-player game with chance nodes (the random mission assignments) and information sets (each operator only knows their own mission state).

The game tree:

```
                              [Chance: assign Alice's mission]
                             /                                \
                  [A=high, p=0.5]                    [A=low, p=0.5]
                          |                                  |
              [Chance: assign Bob's mission]    [Chance: assign Bob's mission]
                /                  \                /                  \
       [B=high, p=0.5]     [B=low, p=0.5]   [B=high, p=0.5]    [B=low, p=0.5]
                |                  |                 |                  |
        [Alice decides]    [Alice decides]   [Alice decides]    [Alice decides]
            /     \           /     \           /     \           /     \
          M         H        M         H        M         H        M         H
          |         |        |         |        |         |        |         |
       [Bob]    [Bob]    [Bob]    [Bob]    [Bob]    [Bob]    [Bob]    [Bob]
       /  \    /  \    /  \    /  \    /  \    /  \    /  \    /  \
      M  H  M  H  M  H  M  H  M  H  M  H  M  H  M  H
```

There are 16 terminal nodes (4 chance combinations × 2 Alice actions × 2 Bob actions). Each terminal has a payoff for Alice and a payoff for Bob.

## Information sets in this game

**Alice's information sets**: when Alice has to decide, she knows her own mission state but not Bob's. So:
- "Alice's information set 1": all nodes where Alice has high mission and is to move (regardless of Bob's hidden mission). 2 nodes here.
- "Alice's information set 2": all nodes where Alice has low mission and is to move. 2 nodes here.

Alice has 2 information sets total. Within each, she must use the same strategy because she cannot distinguish the underlying nodes.

**Bob's information sets**: when Bob has to decide, he knows his own mission state AND has observed Alice's action. So:
- "Bob's information set 1": Bob high, Alice maneuvered (across both possibilities for Alice's hidden mission). 2 nodes.
- "Bob's information set 2": Bob high, Alice held. 2 nodes.
- "Bob's information set 3": Bob low, Alice maneuvered. 2 nodes.
- "Bob's information set 4": Bob low, Alice held. 2 nodes.

Bob has 4 information sets total. He gets more information than Alice (he sees her move first), so he has finer information sets.

## Strategies as functions over information sets

In CFR and other extensive-form game algorithms, a player's **strategy** is a function from their information sets to probability distributions over actions:

\\[ \sigma_i(I) \in \Delta(A(I)) \\]

**Decoding:**
- \\(\sigma_i\\): the strategy of player i (sigma is conventional notation for a strategy)
- \\(I\\): an information set
- \\(\Delta(A(I))\\): the set of probability distributions over the actions available at information set I

For our game, Alice's strategy is two probability distributions:
- \\(\sigma_A(\text{Alice high}) = (p_M, p_H)\\): probability of maneuver and hold when Alice has high mission
- \\(\sigma_A(\text{Alice low}) = (q_M, q_H)\\): probability of maneuver and hold when Alice has low mission

Bob's strategy is four probability distributions, one per information set.

A complete strategy specifies all of these simultaneously. The size of the strategy space grows with the number of information sets, which is what makes large extensive-form games hard.

## The crucial subtlety: strategies vs. policies

In RL (Module 3), we used the word "policy" for a function from states to action distributions. In game theory, we use "strategy" for a function from **information sets** to action distributions. These are subtly different concepts.

A policy is conditioned on the observable state. In a perfect-information game, every node is its own information set, and policy = strategy. In an imperfect-information game, multiple nodes share the same information set, and the strategy must be the same at all of them.

This distinction matters for CFR: regret is computed per information set, not per node, because the player cannot distinguish the underlying nodes.

## Reach probabilities

Given a strategy profile (one strategy per player, plus the chance distributions), the **reach probability** of a node is the probability that the game actually arrives at that node when played according to the strategy profile.

For a node deep in the tree, the reach probability is the product of:
- The chance probabilities along the path
- The strategy probabilities of the actions along the path (each player's strategy applied to the relevant action)

Formally, the reach probability is decomposed:
- \\(\pi^c(h)\\): chance reach probability (product of chance probabilities on path to h)
- \\(\pi_i^\sigma(h)\\): player i's reach probability (product of player i's strategy probabilities on path to h)
- \\(\pi^\sigma(h) = \pi^c(h) \cdot \prod_i \pi_i^\sigma(h)\\): total reach probability

The reach probability tells you how much the game "weights" each node when computing expected payoffs. Nodes with high reach probability contribute more to expected outcomes than nodes with low reach probability.

## Counterfactual reach probabilities

Here is a concept that is crucial for CFR but takes some unpacking. The **counterfactual reach probability** of an information set, from player i's perspective, is the probability of reaching that information set if player i were trying to reach it.

Specifically, it is the product of chance probabilities and all OTHER players' strategy probabilities along the path:

\\[ \pi_{-i}^\sigma(I) = \pi^c(I) \cdot \prod_{j \neq i} \pi_j^\sigma(I) \\]

The "-i" subscript means "everyone except player i." We are computing the probability of reaching this information set assuming player i played to reach it, while all other players played their strategies normally.

This is the weight that CFR uses when updating regrets. It says: "how often would I face this decision if I were trying to face it?" If a particular information set is rarely reached anyway (because of opponent or chance), it gets less weight in the update.

The math gets thick here. The intuition: we want to update strategies based on how relevant each information set is given the current play of the other players. Counterfactual reach captures that relevance.

## Expected payoff and value

The **expected payoff** for player i under strategy profile σ is the average payoff over all terminal nodes, weighted by reach probability:

\\[ u_i(\sigma) = \sum_{z \in Z} \pi^\sigma(z) \cdot u_i(z) \\]

where Z is the set of terminal nodes and \\(u_i(z)\\) is player i's payoff at terminal node z.

Players try to maximize this expected payoff. CFR's job is to find a strategy profile σ where everyone is approximately maximizing simultaneously: a Nash equilibrium.

## What "approximate Nash" means

In the limit of infinite computation, CFR converges to an **exact Nash equilibrium**: a strategy profile where no player can improve by deviation.

In practice, CFR is run for some finite number of iterations, producing an **ε-Nash equilibrium**: a strategy profile where no player can improve by more than ε. As iterations increase, ε shrinks. For most practical purposes, ε in the range 0.01 to 0.001 is good enough.

The iteration counts needed depend on the size of the game. Vanilla CFR on a game with thousands of information sets might need millions of iterations. MCCFR (lesson 4) is the workhorse for larger games.

## Why the structure matters for CFR

CFR exploits the structure of extensive-form games to make Nash equilibrium computation tractable. Specifically:

1. It computes regret per information set (not per node), exploiting the fact that strategies are constant within information sets.
2. It uses reach probabilities to weight updates, exploiting the recursive structure of the game tree.
3. It only needs to traverse the game tree, not enumerate strategy profiles, exploiting the compact representation.

Without the formal structure of extensive-form games, none of these exploitations would be possible. The next lesson uses this structure to define and run CFR on a small game.

## Quiz

{{#quiz 02-extensive-form-games-in-detail.toml}}
