# Lesson 1: Normal-Form and Extensive-Form Games

**Module/Source:** *An Introduction to Game Theory* (Osborne, 2004), Chapters 1–3 (normal-form games, Nash equilibrium, mixed strategies) and Chapter 7 (extensive-form games). The minimax theorem is from von Neumann (1928); the computational treatment follows *Algorithmic Game Theory* (Nisan et al., Chapter 1). CFR connections developed in Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information" and Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in Extensive Games."

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

## Nash equilibrium: the formal definition

The intuition is already clear: "no one wants to deviate." Here is the precise formulation.

Let \\(N = \{1, 2, \ldots, n\}\\) be the set of players. Each player \\(i\\) has a strategy space \\(\Sigma_i\\). A strategy profile is \\(\sigma = (\sigma_1, \sigma_2, \ldots, \sigma_n)\\). Let \\(\sigma_{-i}\\) denote the strategies of all players except player \\(i\\).

A strategy \\(\sigma_i^*\\) is a **best response** to \\(\sigma_{-i}\\) if:

\\[ u_i(\sigma_i^*, \sigma_{-i}) \geq u_i(\sigma_i, \sigma_{-i}) \quad \forall \sigma_i \in \Sigma_i \\]

**Decoding:**
- \\(u_i(\sigma_i, \sigma_{-i})\\): the expected utility for player \\(i\\) when playing \\(\sigma_i\\) against opponents \\(\sigma_{-i}\\)
- \\(\sigma_i^*\\): the strategy that achieves the highest expected utility for player \\(i\\) given what everyone else is doing
- The \\(\forall \sigma_i\\) says this must hold for every alternative strategy, not just the best among a few candidates

A **Nash equilibrium** is a strategy profile \\(\sigma^* = (\sigma_1^*, \ldots, \sigma_n^*)\\) where every player is simultaneously playing a best response:

\\[ \forall i \in N: \quad u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*) \quad \forall \sigma_i \in \Sigma_i \\]

**Decoding the "no one wants to deviate" property:** the defining feature is mutual consistency. At a Nash equilibrium, player \\(i\\) cannot gain by switching strategies *holding all other players' strategies fixed*. This is the stability condition. It does not say the outcome is globally optimal or socially efficient; only that no individual player has a private unilateral incentive to change.

In our satellite operator coordination game, (M, M) satisfies this: if Op 2 is maneuvering, Op 1's payoff from M is -1 and from H is also -1 (looking at the payoff matrix: Op 1 switches to H while Op 2 stays at M gives (-3, -1), wait — actually (-3, -1) gives Op 1 a payoff of -3, not -1). Let us recheck: at (M, M) = (-1, -1). If Op 1 deviates to H (with Op 2 still at M), payoff is (-3, -1), so Op 1's payoff drops from -1 to -3. So Op 1 does not want to deviate. By symmetry, Op 2 does not want to deviate. (M, M) is a Nash equilibrium.

### Code: computing best response given opponent strategy

```python
import numpy as np

def compute_best_response(payoff_matrix: np.ndarray, opponent_strategy: np.ndarray) -> np.ndarray:
    """
    Given a 2-player normal-form game payoff matrix and the opponent's mixed strategy,
    compute the best response for player 1.

    Args:
        payoff_matrix: shape (n_actions_p1, n_actions_p2), entries are player 1's payoffs
        opponent_strategy: shape (n_actions_p2,), probability distribution for player 2

    Returns:
        best_response: shape (n_actions_p1,), a pure strategy (one-hot) for player 1
    """
    # Expected payoff of each pure action for player 1 given opponent's mixed strategy
    # E[u1(a, sigma2)] = sum_j payoff_matrix[a, j] * sigma2[j]
    expected_payoffs = payoff_matrix @ opponent_strategy  # shape (n_actions_p1,)

    best_action = np.argmax(expected_payoffs)
    best_response = np.zeros(len(expected_payoffs))
    best_response[best_action] = 1.0
    return best_response, expected_payoffs


# Satellite operator conjunction game payoff matrix for Operator 1
# Actions: 0 = Maneuver, 1 = Hold
# payoff_matrix[i, j] = Op 1's payoff when Op 1 plays i and Op 2 plays j
conjunction_payoffs_op1 = np.array([
    [-1, -1],   # Op 1 maneuvers: payoff -1 regardless of Op 2
    [-3, -10],  # Op 1 holds: -3 if Op 2 maneuvers, -10 if Op 2 holds
])

# Suppose Op 2 is playing mixed strategy: maneuver with prob 0.7, hold with prob 0.3
op2_strategy = np.array([0.7, 0.3])

br, payoffs = compute_best_response(conjunction_payoffs_op1, op2_strategy)
action_names = ["Maneuver", "Hold"]
print("Expected payoffs:", dict(zip(action_names, payoffs)))
print("Best response:", action_names[np.argmax(br)])
# Expected payoffs: {'Maneuver': -1.0, 'Hold': -5.1}
# Best response: Maneuver
```

Notice that for almost any strategy Op 2 plays, Op 1's best response in this game is to maneuver: the asymmetric collision cost (-10) makes holding too risky unless Op 2 is nearly certain to maneuver.

## Mixed strategy Nash equilibrium

### Why pure Nash equilibria may not exist

Nash's 1950 theorem guarantees that every finite game has at least one Nash equilibrium — but it may require mixed strategies. The conjunction game has a pure Nash equilibrium (M, M), but many games of interest to SSA do not.

Consider an **ISR sensor allocation game**: a monitoring satellite must decide whether to observe Sector A or Sector B. An adversary is simultaneously deciding whether to operate covertly in Sector A or Sector B. The monitoring satellite wants to observe the adversary; the adversary wants to avoid observation.

Payoff matrix (monitoring satellite's payoff = 1 if observed, 0 if not; adversary's is the negative):

```
                     Adversary
                   Sector A    Sector B
Monitor:  Sector A  (+1, -1)   (0, 0)
          Sector B  (0, 0)     (+1, -1)
```

This is exactly matching pennies in structure. Check for pure Nash equilibria:
- (A, A): monitor gets +1, adversary wants to switch to B.
- (A, B): adversary gets 0, monitor wants to switch to B.
- (B, A): adversary gets 0, monitor wants to switch to A.
- (B, B): monitor gets +1, adversary wants to switch to A.

No pure Nash equilibrium exists. The mixed strategy Nash equilibrium: both monitor and adversary randomize 50/50 between the two sectors.

### Rock-paper-scissors: the canonical mixed NE

Rock-paper-scissors is the canonical three-action zero-sum game with no pure Nash equilibrium. The unique Nash equilibrium is each player randomizing uniformly: (1/3, 1/3, 1/3).

The logic: if you play rock with any probability greater than 1/3, your opponent can exploit you by playing paper more. At (1/3, 1/3, 1/3), your expected payoff is 0 no matter what the opponent does. You cannot be exploited, and you cannot exploit either.

This structure appears in every zero-sum game with no pure Nash equilibrium: the mixed NE is the strategy that makes the opponent indifferent among all their pure actions.

### Computing a 2×2 mixed Nash equilibrium

For a 2×2 zero-sum game, the mixed Nash equilibrium can be computed analytically by solving the indifference condition: player 1's strategy must make player 2 indifferent between their actions, and vice versa.

For the ISR sensor allocation game, let \\(p\\) be the probability the monitor chooses Sector A. The adversary is indifferent when:

\\[ \text{Adversary's payoff from Sector A} = \text{Adversary's payoff from Sector B} \\]
\\[ -1 \cdot p + 0 \cdot (1-p) = 0 \cdot p + (-1) \cdot (1-p) \\]
\\[ -p = -(1-p) \\]
\\[ p = 0.5 \\]

Similarly, the adversary must play each sector with probability 0.5 to make the monitor indifferent.

```python
import numpy as np
from scipy.optimize import linprog

def solve_2x2_mixed_ne(payoff_matrix: np.ndarray):
    """
    Compute the mixed Nash equilibrium for a 2x2 two-player zero-sum game.

    For zero-sum games, the NE is found by solving each player's indifference condition.
    Player 2's strategy makes Player 1 indifferent:
        sum_j payoff[0, j] * q[j] = sum_j payoff[1, j] * q[j]
        (for player 1's two actions to have equal expected payoff)

    Args:
        payoff_matrix: shape (2, 2), player 1's payoffs (player 2 gets negatives)

    Returns:
        (p_star, q_star): Nash equilibrium mixed strategies
    """
    A = payoff_matrix  # 2x2

    # Player 2's mixing probability q (prob of action 0)
    # A[0,0]*q + A[0,1]*(1-q) = A[1,0]*q + A[1,1]*(1-q)
    # q*(A[0,0] - A[0,1] - A[1,0] + A[1,1]) = A[1,1] - A[0,1]
    denom = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    if abs(denom) < 1e-10:
        q_star = np.array([0.5, 0.5])  # degenerate case
    else:
        q = (A[1, 1] - A[0, 1]) / denom
        q = np.clip(q, 0, 1)
        q_star = np.array([q, 1 - q])

    # Player 1's mixing probability p (prob of action 0)
    # A[0,0]*p + A[1,0]*(1-p) = A[0,1]*p + A[1,1]*(1-p)
    denom2 = A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1]
    if abs(denom2) < 1e-10:
        p_star = np.array([0.5, 0.5])
    else:
        p = (A[1, 1] - A[1, 0]) / denom2
        p = np.clip(p, 0, 1)
        p_star = np.array([p, 1 - p])

    return p_star, q_star


# ISR sensor allocation game (zero-sum)
# Monitor payoffs: +1 if they match sectors, 0 otherwise
isr_payoffs = np.array([
    [1, 0],   # Monitor chooses A: +1 if adversary in A, 0 if adversary in B
    [0, 1],   # Monitor chooses B: 0 if adversary in A, +1 if adversary in B
])

p_ne, q_ne = solve_2x2_mixed_ne(isr_payoffs)
print(f"Monitor NE strategy: P(Sector A) = {p_ne[0]:.3f}, P(Sector B) = {p_ne[1]:.3f}")
print(f"Adversary NE strategy: P(Sector A) = {q_ne[0]:.3f}, P(Sector B) = {q_ne[1]:.3f}")
# Monitor NE strategy: P(Sector A) = 0.500, P(Sector B) = 0.500
# Adversary NE strategy: P(Sector A) = 0.500, P(Sector B) = 0.500

# Verify: compute expected payoff under the NE
ne_payoff = p_ne @ isr_payoffs @ q_ne
print(f"Expected payoff for monitor at NE: {ne_payoff:.3f}")
# Expected payoff: 0.500 (monitor catches adversary half the time on average)
```

The key insight is that randomization is not weakness — it is the equilibrium strategy. A monitor that predictably focuses on one sector can be exploited. A monitor that randomizes uniformly cannot be.

## The minimax theorem for zero-sum games

### Von Neumann's theorem

For two-player zero-sum games, von Neumann's minimax theorem (1928) establishes a fundamental duality:

\\[ \max_{\sigma_1} \min_{\sigma_2} u_1(\sigma_1, \sigma_2) = \min_{\sigma_2} \max_{\sigma_1} u_1(\sigma_1, \sigma_2) \\]

**Decoding:**
- Left side: Player 1 chooses their strategy to maximize their worst-case payoff (maximize the minimum over Player 2's responses). This is the **maximin** value.
- Right side: Player 2 chooses their strategy to minimize Player 1's best-case payoff (minimize the maximum over Player 1's choices). This is the **minimax** value.
- The equality says these two quantities are the same. There is a unique **game value** \\(v\\), and both players' equilibrium strategies achieve it.

**Why minimax = maximin for zero-sum games:** in a zero-sum game, Player 2's payoff is \\(-u_1\\). Player 2 minimizing \\(u_1\\) is the same as Player 2 maximizing their own payoff. So the minimax formulation and Nash equilibrium formulation coincide. In non-zero-sum games this equality fails (hence the need for the more general Nash equilibrium concept).

### Connection to the minimax search tree from Module 4

In Module 4 (MCTS), you encountered minimax search: a game tree where each level alternates between maximizing and minimizing, and the optimal play is found by backward induction. That algorithm computes the **pure strategy** minimax value for perfect-information games.

Von Neumann's theorem extends this to **mixed strategies** and **imperfect information**. The minimax search tree gives the value of perfect-information games; von Neumann's theorem guarantees that the same value concept extends to the full class of finite two-player zero-sum games when players can randomize.

CFR is, at its core, an algorithm for computing the minimax value and associated strategies for imperfect-information zero-sum games — the setting where minimax search does not directly apply. The bridge from minimax trees to CFR is the minimax theorem.

```python
import numpy as np
from scipy.optimize import linprog

def solve_minimax(payoff_matrix: np.ndarray):
    """
    Solve a two-player zero-sum game using the minimax theorem via linear programming.

    Player 1 solves: max_{p, v} v s.t. p^T A e_j >= v for all j, sum(p) = 1, p >= 0
    which is equivalent to finding the maximin strategy.

    Args:
        payoff_matrix: shape (m, n), player 1's payoffs

    Returns:
        (p_star, q_star, game_value): minimax strategies and game value
    """
    m, n = payoff_matrix.shape

    # Player 1 maximizes minimum expected payoff (maximin):
    # max v s.t. A^T p >= v * 1, sum(p) = 1, p >= 0
    # As LP: min -v, variables = [p_1, ..., p_m, v]
    # Constraints: for each j: -sum_i A[i,j] * p[i] + v <= 0
    #              sum_i p[i] = 1, p >= 0
    c = np.zeros(m + 1)
    c[-1] = -1  # minimize -v (maximize v)

    # Inequality constraints: A_ub @ x <= b_ub
    # For each action j of player 2: -A[:,j]^T p + v <= 0
    A_ub = np.zeros((n, m + 1))
    for j in range(n):
        A_ub[j, :m] = -payoff_matrix[:, j]
        A_ub[j, m] = 1
    b_ub = np.zeros(n)

    # Equality: sum(p) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]  # p >= 0, v unbounded

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    p_star = res.x[:m]
    game_value = res.x[m]  # = -res.fun

    # Player 2 minimizes maximum expected payoff (minimax): symmetric LP
    # min v s.t. A q <= v * 1, sum(q) = 1, q >= 0
    c2 = np.zeros(n + 1)
    c2[-1] = 1  # minimize v

    A_ub2 = np.zeros((m, n + 1))
    for i in range(m):
        A_ub2[i, :n] = payoff_matrix[i, :]
        A_ub2[i, n] = -1
    b_ub2 = np.zeros(m)

    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1
    b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * n + [(None, None)]

    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2)
    q_star = res2.x[:n]

    return p_star, q_star, game_value


# Satellite-vs-jammer spectrum deconfliction game (zero-sum)
# Satellite chooses frequency band: [L-band, S-band, X-band]
# Jammer chooses which band to disrupt: [L, S, X]
# Satellite payoff: +1 if jammer picks wrong band, -1 if jammer disrupts satellite's band
spectrum_payoffs = np.array([
    [-1,  1,  1],  # Satellite on L-band
    [ 1, -1,  1],  # Satellite on S-band
    [ 1,  1, -1],  # Satellite on X-band
])

p_star, q_star, value = solve_minimax(spectrum_payoffs)
print("Satellite minimax strategy:", np.round(p_star, 3))
print("Jammer minimax strategy:", np.round(q_star, 3))
print(f"Game value: {value:.3f}")
# Satellite minimax strategy: [0.333, 0.333, 0.333]
# Jammer minimax strategy:    [0.333, 0.333, 0.333]
# Game value: 0.333
# (Satellite avoids jamming 2/3 of the time on average)
```

The spectrum deconfliction game is symmetric: the satellite should randomize uniformly across frequency bands, and the jammer should do the same. The game value of 1/3 means the satellite successfully avoids jamming 2/3 of the time — precisely the fraction of bands left unjammed.

## Key Takeaways

- A **normal-form game** represents simultaneous multi-agent decisions as a payoff matrix; an **extensive-form game** represents sequential decisions with information about what has been observed as a game tree.
- A **Nash equilibrium** is a strategy profile where every player is simultaneously best-responding: no individual has a unilateral incentive to deviate, which is the minimal stability requirement for predicting rational play.
- **Pure-strategy Nash equilibria may not exist**; mixed-strategy equilibria always exist (Nash's theorem) and are the solution concept in games like ISR sensor allocation where deterministic strategies are exploitable.
- **Von Neumann's minimax theorem** states that for two-player zero-sum games, the maximin and minimax values are equal, establishing a unique game value and connecting Nash equilibria to the minimax tree search from Module 4.
- **Information sets** are the key extension from normal-form to extensive-form games: they encode what a player can and cannot observe, and strategies are functions over information sets rather than over raw game states.
- CFR (Lesson 3) is best understood as an iterative algorithm for computing the minimax Nash equilibrium of an imperfect-information extensive-form game, building directly on regret minimization over information sets.

## Quiz

{{#quiz 01-normal-form-and-extensive-form.toml}}
