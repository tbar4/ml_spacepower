# Lesson 1: Normal-Form and Extensive-Form Games

## Where this fits

Game theory is the framework for reasoning about decisions when multiple agents are involved and what is best for one depends on what the others do. This lesson introduces the two main ways of representing such games and the central solution concept: Nash equilibrium. Without these foundations, CFR (lesson 3) would not make sense; with them, it is just a clever algorithm for computing something well-defined.

## Why a single-agent framework is not enough

In Module 3, an RL agent learned an optimal policy for an MDP. The optimum was well-defined: the policy that maximizes expected return.

When there are multiple agents, "optimal" becomes ambiguous. Each agent has its own objective. What is best for one might be terrible for another. And what is best for one depends on what the others are doing, which depends on what is best for them, which depends on what is best for the first one... circular.

Game theory is the framework that resolves this circularity. It defines what "stable" multi-agent strategies look like, even when no single agent's "best" is well-defined.

## Normal-form games

A **normal-form game** is the simplest setting: two (or more) players choose actions simultaneously, without knowing what the others will choose, and receive payoffs based on the joint action.

The classic example: two satellite operators are deciding whether to maneuver to avoid a conjunction. Each can either maneuver (M) or hold (H). The cost of each combination depends on what both do.

We represent this as a **payoff matrix**:

```
                    Operator 2
                  M           H
Op 1:   M    (-1, -1)    (-1, -3)
        H    (-3, -1)   (-10, -10)
```

Read each cell as (Operator 1's payoff, Operator 2's payoff). The numbers are negative because they represent costs. Smaller (more negative) means worse.

- (M, M): both maneuver, both pay the maneuver cost (-1, -1). Wasteful (only one needed to maneuver) but safe.
- (M, H): Op 1 maneuvers, Op 2 holds. Op 1 pays -1, Op 2 saves the maneuver cost but suffers reputational cost from making the other maneuver: -3.
- (H, M): symmetric. Op 2 maneuvers, Op 1 holds. (-3, -1).
- (H, H): neither maneuvers, both suffer the collision: (-10, -10).

If you were Op 1, what would you do? It depends on what you think Op 2 will do. If Op 2 is going to maneuver, you should hold (-1 vs. -3). If Op 2 is going to hold, you should maneuver (-1 vs. -10). There is no dominant strategy.

This is what game theory addresses.

## Strategies

A **pure strategy** for a player is one specific action (e.g., "always maneuver"). A **mixed strategy** is a probability distribution over actions (e.g., "maneuver with probability 0.6, hold with probability 0.4").

In games where pure strategies do not give a stable solution (which is most games), mixed strategies often do.

A **strategy profile** is a tuple of strategies, one per player. For the conjunction game, a strategy profile is one strategy for Op 1 and one for Op 2.

## Best response

A strategy is a **best response** to the other player's strategy if it maximizes the player's expected payoff given that the other player is using the specified strategy.

If Op 2 is going to maneuver with probability 0.5:
- Op 1 maneuvering: expected payoff = 0.5 × (-1) + 0.5 × (-1) = -1
- Op 1 holding: expected payoff = 0.5 × (-3) + 0.5 × (-10) = -6.5

Op 1's best response: maneuver. (-1 > -6.5.)

If Op 2 is going to maneuver with probability 0.95:
- Op 1 maneuvering: 0.95 × (-1) + 0.05 × (-1) = -1
- Op 1 holding: 0.95 × (-3) + 0.05 × (-10) = -3.35

Op 1's best response: still maneuver.

If Op 2 is going to maneuver with probability 0.1:
- Op 1 maneuvering: -1
- Op 1 holding: 0.1 × (-3) + 0.9 × (-10) = -9.3

Op 1's best response: maneuver. (-1 > -9.3.)

In this game, Op 1's best response is "maneuver" almost regardless of Op 2's strategy. Because the cost of (H, H) is so high, the safe play is to maneuver. By symmetry, the same is true for Op 2. So both should maneuver, and (M, M) is the equilibrium.

## Nash equilibrium

A **Nash equilibrium** is a strategy profile where every player is best-responding to the others. No player can improve their payoff by unilaterally changing their strategy.

In our game, (Maneuver, Maneuver) is a Nash equilibrium. If Op 1 is maneuvering, Op 2's best response is to maneuver too (you might think hold gives better payoff: -1 vs. -1, but actually it depends). Wait, let me re-examine.

Looking again at the payoffs: at (M, M), Op 1 gets -1. If Op 1 deviates to H (with Op 2 still at M), Op 1 gets -1. So Op 1 is indifferent between M and H when Op 2 is playing M. This means there are multiple Nash equilibria here: any strategy profile where at least one operator maneuvers with high probability.

This is reasonable! Real conjunction-avoidance protocols typically rely on coordination: one operator agrees to maneuver based on the conjunction warning, often the one with more delta-V budget remaining or the one operating in a "non-priority" satellite class. The Nash equilibrium framework reveals that the game has multiple stable solutions; coordination protocols are needed to pick among them.

For zero-sum two-player games (where one player's gain is exactly the other's loss), Nash equilibria are unique and computationally tractable. For general-sum games (like our conjunction game), there can be multiple equilibria, and which one gets played depends on factors outside the game model.

## Extensive-form games

A **normal-form** game assumes simultaneous moves with no information about what the other player is doing. Many real games are sequential: players move one at a time and see what others have done.

An **extensive-form game** represents this with a game tree. Each node is a decision point. Each edge is an action. Leaves are terminal states with payoffs. Some nodes belong to specific players (their decision); others are "chance" nodes (random events).

For sequential conjunction negotiations, the game tree might look like:

```
                [Conjunction warning issued]
                            |
                  [Op 1 decides first]
                /                       \
        [Op 1 maneuvers]          [Op 1 holds]
              |                          |
       [Op 2 decides]              [Op 2 decides]
        /        \                  /        \
   M(-1,-1)   H(-1,-3)        M(-3,-1)   H(-10,-10)
```

This sequential version is different from the simultaneous one. Now Op 2 can see what Op 1 did before deciding. Op 2's optimal strategy is: if Op 1 maneuvered, hold; if Op 1 held, maneuver.

Knowing this, what should Op 1 do? Reasoning by backward induction:
- If Op 1 maneuvers, Op 2 will hold; Op 1's payoff is -1.
- If Op 1 holds, Op 2 will maneuver; Op 1's payoff is -3.

So Op 1 should maneuver, and the unique equilibrium of this sequential game has Op 1 maneuvering and Op 2 holding. The first mover takes the cost.

Notice: making the game sequential (with Op 1 moving first) breaks the multiplicity of equilibria from the simultaneous version. The sequential structure carries information.

## Imperfect information: information sets

Some real games have hidden information. In poker, you do not know your opponent's hand. In SSA, you might not know what the other operator's mission profile or fuel budget is.

Extensive-form games handle this with **information sets**. An information set is a collection of game tree nodes that the current player cannot distinguish between. The player must use the same strategy at every node within an information set.

For our conjunction game with hidden information about the other operator's mission constraints, both operators might be in an information set covering "Op 2 is high-priority" and "Op 2 is low-priority": Op 1 cannot distinguish, so must use the same strategy.

Information sets are the reason extensive-form games are richer than normal-form games. They allow for partial observability, randomized signaling, and Bayesian belief updating during play.

## A worked example: matching pennies (a zero-sum game)

A simpler game to internalize Nash equilibrium: matching pennies.

Two players each have a penny. They simultaneously reveal heads or tails. If they match, Player 1 wins both pennies. If they differ, Player 2 wins both.

Payoff matrix (from Player 1's perspective; Player 2's are negatives):

```
                    Player 2
                  H        T
Player 1:  H    (+1, -1) (-1, +1)
           T    (-1, +1) (+1, -1)
```

Are there pure-strategy Nash equilibria? Try (H, H): Player 1 gets +1. Player 2 would deviate to T to get +1 instead of -1. Try (H, T): Player 2 gets +1. Player 1 would deviate to T. By symmetry, no pure-strategy Nash equilibrium exists.

The mixed-strategy Nash equilibrium: each player plays H with probability 0.5 and T with probability 0.5. At this equilibrium:
- Player 1's expected payoff is 0.5 × (+1) + 0.5 × (-1) = 0 regardless of what Player 2 does.
- Same for Player 2.

Neither can improve by unilateral deviation. This is the Nash equilibrium, and it requires randomization.

The fact that mixed strategies are the only Nash equilibria of matching pennies tells us something deep: **deterministic strategies are not always sufficient for game-theoretic optimality.** This is one major reason policy gradient methods (which produce stochastic policies) are useful for game theory.

## Why Nash equilibria are the right concept

Nash equilibria capture the idea of "stability under selfish optimization." If everyone is playing a Nash equilibrium strategy, no one has a private incentive to change. This is a minimum requirement for a strategy profile to be predictive of how rational agents would actually play.

It is not the only equilibrium concept (correlated equilibrium, evolutionary stable strategies, and others exist). But it is the most fundamental, and CFR is the algorithm we use to compute it.

## Quiz

{{#quiz 01-normal-form-and-extensive-form.toml}}
