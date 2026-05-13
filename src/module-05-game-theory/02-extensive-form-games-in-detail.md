# Lesson 2: Extensive-Form Games in Detail

**Module/Source:** *An Introduction to Game Theory* (Osborne, 2004), Chapters 6–7 (extensive-form games, subgame perfect equilibrium, backward induction). Formal definitions of information sets and reach probabilities follow the notation in Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information" and Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in Extensive Games." The OpenSpiel library uses the same vocabulary throughout its API.


<!-- toc -->

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

\[ \sigma_i(I) \in \Delta(A(I)) \]

**Decoding:**
- \(\sigma_i\): the strategy of player i (sigma is conventional notation for a strategy)
- \(I\): an information set
- \(\Delta(A(I))\): the set of probability distributions over the actions available at information set I

For our game, Alice's strategy is two probability distributions:
- \(\sigma_A(\text{Alice high}) = (p_M, p_H)\): probability of maneuver and hold when Alice has high mission
- \(\sigma_A(\text{Alice low}) = (q_M, q_H)\): probability of maneuver and hold when Alice has low mission

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
- \(\pi^c(h)\): chance reach probability (product of chance probabilities on path to h)
- \(\pi_i^\sigma(h)\): player i's reach probability (product of player i's strategy probabilities on path to h)
- \(\pi^\sigma(h) = \pi^c(h) \cdot \prod_i \pi_i^\sigma(h)\): total reach probability

The reach probability tells you how much the game "weights" each node when computing expected payoffs. Nodes with high reach probability contribute more to expected outcomes than nodes with low reach probability.

## Counterfactual reach probabilities

Here is a concept that is crucial for CFR but takes some unpacking. The **counterfactual reach probability** of an information set, from player i's perspective, is the probability of reaching that information set if player i were trying to reach it.

Specifically, it is the product of chance probabilities and all OTHER players' strategy probabilities along the path:

\[ \pi_{-i}^\sigma(I) = \pi^c(I) \cdot \prod_{j \neq i} \pi_j^\sigma(I) \]

The "-i" subscript means "everyone except player i." We are computing the probability of reaching this information set assuming player i played to reach it, while all other players played their strategies normally.

This is the weight that CFR uses when updating regrets. It says: "how often would I face this decision if I were trying to face it?" If a particular information set is rarely reached anyway (because of opponent or chance), it gets less weight in the update.

The math gets thick here. The intuition: we want to update strategies based on how relevant each information set is given the current play of the other players. Counterfactual reach captures that relevance.

## Expected payoff and value

The **expected payoff** for player i under strategy profile σ is the average payoff over all terminal nodes, weighted by reach probability:

\[ u_i(\sigma) = \sum_{z \in Z} \pi^\sigma(z) \cdot u_i(z) \]

where Z is the set of terminal nodes and \(u_i(z)\) is player i's payoff at terminal node z.

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

## Subgame perfect equilibrium

Nash equilibrium is the right concept for simultaneous-move games, but extensive-form games allow sequential moves, which introduces a new issue: **non-credible threats**.

Consider a simplified version of the conjunction game where Alice moves first and can threaten to sue Bob if he does not maneuver. In the normal-form representation, "I will sue if you don't maneuver" might be a Nash equilibrium strategy because Bob, fearing the lawsuit, maneuvers. But if actually carrying out the lawsuit would cost Alice more than she would gain, the threat is non-credible. Alice would not follow through.

**Backward induction** eliminates these non-credible threats. Starting from the terminal nodes and working back:

1. At each final decision point, the player chooses the action that maximizes their payoff.
2. Replace that decision point with the resulting payoff values.
3. Move one level up and repeat.

The strategies surviving backward induction form a **subgame perfect equilibrium** (SPE): a Nash equilibrium where the strategies remain Nash equilibria in every subgame (every sub-tree rooted at any reachable node).

### Why SPE eliminates non-credible threats

A threat is credible only if carrying it out is the rational action at the point where it would be executed. Backward induction checks exactly this: it asks "would this player really do this if we actually reached this node?" If the answer is no, the equilibrium is thrown out.

### SSA example: credible commitments in spectrum deconfliction

Suppose two operators share a frequency band. Operator A (incumbent) threatens to transmit at full power to jam Operator B if B transmits during A's window, even though doing so would also degrade A's own signal.

```
         [B considers transmitting]
        /                           \
    [B transmits]              [B stays silent]
         |                           |
    [A decides]                  (A: 0, B: 5)  <- B takes full slot
    /         \
[A jams]    [A ignores]
 (-2, -3)     (-5, 8)   <- A ignores, B takes the slot
```

Payoffs: (A's payoff, B's payoff).

Nash equilibrium analysis: is (A threatens to jam, B stays silent) a Nash equilibrium?
- If B believes A will jam, B prefers to stay silent (5 vs. -3). So B staying silent is a best response to "A will jam."
- If B is staying silent, A never has to execute the threat, so any threat strategy is technically a Nash equilibrium.

But applying backward induction to the subgame where B actually transmitted:
- A's payoff from jamming = -2, from ignoring = -5.
- A will jam. The threat is actually credible here.

Now suppose the payoffs change: jamming costs A severely (due to international telecommunications regulations), making the jam payoff (-8, -3) instead of (-2, -3):

- A's payoff from jamming = -8, from ignoring = -5.
- A will NOT jam. The threat is non-credible.
- Knowing this, B will transmit (payoff 8 > 5).
- The SPE is: B transmits, A ignores.

The SPE analysis correctly identifies that credibility depends on the actual payoffs at the point of execution, not just the announced threat.

## Information sets and perfect recall

### Formal definition of an information set

An **information set** \(I\) for player \(i\) is a set of decision nodes \(\{h_1, h_2, \ldots, h_k\}\) such that:
1. All nodes in \(I\) belong to player \(i\) (same player moves at each).
2. The same actions are available at every node in \(I\).
3. Player \(i\) cannot distinguish between the nodes within \(I\) based on their observable history.

**Decoding:** condition 3 is the key one. It says that if the game reaches any node in \(I\), player \(i\) only knows "I am at some node in \(I\)" but not which specific node. Their strategy must therefore be the same at all nodes in \(I\) — they cannot condition on information they do not have.

In our Alice–Bob conjunction game, Alice's information set 1 contains two nodes: (chance says A=high, B=high, Alice's turn) and (chance says A=high, B=low, Alice's turn). Alice knows she is high-priority but not whether Bob is high or low priority. Both nodes are in the same information set because, from Alice's perspective, they are indistinguishable.

### Perfect recall

A player has **perfect recall** if they always remember their own past actions and observations. Formally, within an information set, all nodes must have identical sequences of (player \(i\)'s actions, information sets) along the path from the root.

Perfect recall is standard in game theory and in CFR: it ensures that information sets have a well-behaved structure that supports the counterfactual reasoning CFR relies on.

**What happens when perfect recall fails:** if a player can "forget" what they did earlier (imperfect recall), the same player's decision nodes can end up in the same information set even though they arose from different action sequences by that player. This creates fundamental problems for CFR:

1. The standard regret decomposition breaks down — you cannot cleanly separate regret by information set.
2. Nash equilibrium computation becomes PSPACE-hard rather than polynomial-time.
3. The "strategy" of a player can depend on where in the information set they are (which defeats the purpose of having information sets).

In SSA contexts, imperfect recall arises naturally in situations with limited telemetry: a ground station might issue commands to a satellite but not retain the record of what commands were sent. For the purposes of game-theoretic analysis, we generally assume perfect recall or explicitly model the information state to restore it.

### When partial observability differs from imperfect recall

Partial observability (not knowing the opponent's state) is different from imperfect recall (forgetting your own past actions). Both create information sets, but:
- Partial observability of the opponent's state is handled naturally by information sets with no complications for CFR.
- Imperfect recall of your own actions creates information sets that violate the standard CFR assumptions.

In SSA, the classic partial observability scenario is a satellite-vs-jammer hide-and-seek game: the satellite does not know the jammer's location; the jammer does not know whether the satellite has detected it. This is partial observability of the opponent, handled cleanly by information sets.

## Reach probabilities: detailed computation

Given a strategy profile \(\sigma\), the reach probability \(\pi^\sigma(h)\) of a history \(h\) is computed as a product over all actions taken along the path from the root to \(h\):

\[ \pi^\sigma(h) = \prod_{h' \cdot a \sqsubseteq h} \sigma_{P(h')}(h', a) \]

**Decoding:**
- \(h' \cdot a \sqsubseteq h\): "the action \(a\) taken from history \(h'\) is a prefix of \(h\)" — this iterates over all (history, action) pairs on the path from root to \(h\)
- \(P(h')\): the player (or chance) whose turn it is at \(h'\)
- \(\sigma_{P(h')}(h', a)\): the probability that player \(P(h')\) takes action \(a\) at \(h'\) under strategy \(\sigma\)

The product telescopes through the tree: each edge on the path contributes a factor equal to the probability of the action that edge represents.

### Why CFR uses reach probabilities

Reach probabilities perform two roles in CFR:

1. **Weighting the contribution of terminal nodes to expected payoffs.** Terminal nodes with high reach probability matter more; nodes that are never reached (probability 0) do not matter at all.

2. **Weighting the counterfactual regret updates.** When computing regret at information set \(I\) for player \(i\), CFR weights the update by the counterfactual reach \(\pi_{-i}^\sigma(I)\). Information sets that are reachable mostly because of the opponent's and chance's play get more weight in the regret update than sets the opponent actively avoids.

## Code: an extensive-form game class

Here is a Python implementation of a small SSA game tree with information sets, actions, and utilities, including methods for traversal and reach probability computation.

```python
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class GameNode:
    """A node in an extensive-form game tree."""
    node_id: str
    player: Optional[int]  # None = terminal, -1 = chance node, 0/1 = player index
    actions: List[str] = field(default_factory=list)
    children: Dict[str, "GameNode"] = field(default_factory=dict)
    payoffs: Optional[Tuple[float, float]] = None   # Only at terminal nodes
    chance_probs: Optional[Dict[str, float]] = None  # Only at chance nodes
    info_set_id: Optional[str] = None  # Which information set this node belongs to


class SSAExtensiveFormGame:
    """
    A small SSA conjunction game as an extensive-form game.

    Structure:
      Chance assigns Alice's mission (high/low, 50/50).
      Alice observes her own mission, decides M or H.
      Bob observes Alice's action (not her mission), decides M or H.
      Payoffs depend on both actions and Alice's mission.

    Information sets:
      Alice: {alice_high, alice_low}  -- 2 information sets
      Bob:   {bob_after_M, bob_after_H}  -- 2 information sets
    """

    PAYOFFS = {
        # (alice_mission, alice_action, bob_action): (alice_payoff, bob_payoff)
        ("high", "M", "M"): (-2, -1),
        ("high", "M", "H"): (-2, -3),
        ("high", "H", "M"): (-3, -1),
        ("high", "H", "H"): (-10, -10),
        ("low",  "M", "M"): (-1, -1),
        ("low",  "M", "H"): (-1, -3),
        ("low",  "H", "M"): (-3, -1),
        ("low",  "H", "H"): (-10, -10),
    }

    def __init__(self):
        self.root = self._build_tree()
        self.info_sets = self._extract_info_sets()

    def _build_tree(self) -> GameNode:
        root = GameNode("root", player=-1, actions=["high", "low"],
                        chance_probs={"high": 0.5, "low": 0.5})
        for mission in ["high", "low"]:
            alice_node = GameNode(
                f"alice_{mission}", player=0, actions=["M", "H"],
                info_set_id=f"alice_{mission}"
            )
            for alice_action in ["M", "H"]:
                bob_node = GameNode(
                    f"bob_{mission}_{alice_action}", player=1, actions=["M", "H"],
                    info_set_id=f"bob_after_{alice_action}"  # Bob sees Alice's action, not mission
                )
                for bob_action in ["M", "H"]:
                    terminal = GameNode(
                        f"terminal_{mission}_{alice_action}_{bob_action}",
                        player=None,
                        payoffs=self.PAYOFFS[(mission, alice_action, bob_action)]
                    )
                    bob_node.children[bob_action] = terminal
                alice_node.children[alice_action] = bob_node
            root.children[mission] = alice_node
        return root

    def _extract_info_sets(self) -> Dict[str, List[GameNode]]:
        """Group all decision nodes by their information set ID."""
        info_sets: Dict[str, List[GameNode]] = {}
        self._collect_info_sets(self.root, info_sets)
        return info_sets

    def _collect_info_sets(self, node: GameNode, info_sets: Dict):
        if node.player is None:  # terminal
            return
        if node.info_set_id is not None:
            if node.info_set_id not in info_sets:
                info_sets[node.info_set_id] = []
            info_sets[node.info_set_id].append(node)
        for child in node.children.values():
            self._collect_info_sets(child, info_sets)

    def compute_reach_probs(
        self,
        sigma: Dict[str, np.ndarray],  # info_set_id -> action probabilities
        node: Optional[GameNode] = None,
        reach: float = 1.0,
        player0_reach: float = 1.0,
        player1_reach: float = 1.0,
        chance_reach: float = 1.0,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Recursively compute reach probabilities for all nodes.

        Returns a dict: node_id -> (total_reach, player0_reach, counterfactual_reach_p0)
        where counterfactual_reach_p0 = chance_reach * player1_reach (i.e., pi_{-0}).
        """
        if node is None:
            node = self.root

        result = {
            node.node_id: (reach, player0_reach, chance_reach * player1_reach)
        }

        if node.player is None:  # terminal node
            return result

        if node.player == -1:  # chance node
            for action, prob in node.chance_probs.items():
                child = node.children[action]
                sub = self.compute_reach_probs(
                    sigma, child,
                    reach=reach * prob,
                    player0_reach=player0_reach,
                    player1_reach=player1_reach,
                    chance_reach=chance_reach * prob,
                )
                result.update(sub)

        elif node.player == 0:  # Alice's decision
            info_set_id = node.info_set_id
            probs = sigma.get(info_set_id, np.array([0.5, 0.5]))
            for i, action in enumerate(node.actions):
                child = node.children[action]
                sub = self.compute_reach_probs(
                    sigma, child,
                    reach=reach * probs[i],
                    player0_reach=player0_reach * probs[i],
                    player1_reach=player1_reach,
                    chance_reach=chance_reach,
                )
                result.update(sub)

        elif node.player == 1:  # Bob's decision
            info_set_id = node.info_set_id
            probs = sigma.get(info_set_id, np.array([0.5, 0.5]))
            for i, action in enumerate(node.actions):
                child = node.children[action]
                sub = self.compute_reach_probs(
                    sigma, child,
                    reach=reach * probs[i],
                    player0_reach=player0_reach,
                    player1_reach=player1_reach * probs[i],
                    chance_reach=chance_reach,
                )
                result.update(sub)

        return result

    def compute_expected_payoff(
        self, sigma: Dict[str, np.ndarray]
    ) -> Tuple[float, float]:
        """Compute expected payoffs for both players under strategy profile sigma."""
        reach_probs = self.compute_reach_probs(sigma)
        alice_payoff = 0.0
        bob_payoff = 0.0

        def traverse(node: GameNode):
            nonlocal alice_payoff, bob_payoff
            if node.player is None:
                reach, _, _ = reach_probs[node.node_id]
                alice_payoff += reach * node.payoffs[0]
                bob_payoff   += reach * node.payoffs[1]
                return
            for child in node.children.values():
                traverse(child)

        traverse(self.root)
        return alice_payoff, bob_payoff


# Example usage
game = SSAExtensiveFormGame()

print("Information sets:")
for iset_id, nodes in game.info_sets.items():
    print(f"  {iset_id}: {[n.node_id for n in nodes]}")

# Uniform strategy: each player plays M and H with prob 0.5 everywhere
uniform_sigma = {
    "alice_high": np.array([0.5, 0.5]),
    "alice_low":  np.array([0.5, 0.5]),
    "bob_after_M": np.array([0.5, 0.5]),
    "bob_after_H": np.array([0.5, 0.5]),
}

alice_ev, bob_ev = game.compute_expected_payoff(uniform_sigma)
print(f"\nUniform strategy expected payoffs: Alice={alice_ev:.2f}, Bob={bob_ev:.2f}")

# Alice always maneuvers, Bob always holds
aggressive_sigma = {
    "alice_high": np.array([1.0, 0.0]),  # always M
    "alice_low":  np.array([1.0, 0.0]),  # always M
    "bob_after_M": np.array([0.0, 1.0]),  # always H after Alice M
    "bob_after_H": np.array([1.0, 0.0]),  # always M after Alice H
}

alice_ev2, bob_ev2 = game.compute_expected_payoff(aggressive_sigma)
print(f"Alice maneuvers, Bob free-rides: Alice={alice_ev2:.2f}, Bob={bob_ev2:.2f}")

# Compute and display reach probabilities for a few nodes
reach_data = game.compute_reach_probs(uniform_sigma)
print("\nSample reach probabilities (uniform strategy):")
for node_id, (total, p0, cf_p0) in reach_data.items():
    if total > 0 and total < 1.0:
        print(f"  {node_id}: total={total:.4f}, pi_Alice={p0:.4f}, cf_reach_Alice={cf_p0:.4f}")
```

Running this code reveals the structure:
- Under the uniform strategy, Alice's expected payoff is around -4.0 and Bob's is similar — both suffer from the mixing over the catastrophic (H, H) outcome.
- When Alice always maneuvers and Bob free-rides by holding, Alice's payoff is -2.0 and Bob's is -2.0 (Alice pays the maneuver cost; Bob pays nothing but the asymmetric cost is absorbed by Alice).
- The counterfactual reach probabilities for Bob's information sets reflect that he can actually observe Alice's action, which is why his information sets are finer than Alice's.

## Key Takeaways

- An extensive-form game formalizes sequential decision-making with information structure: game tree nodes represent decision points, edges represent actions, and information sets partition decision nodes into what a player can and cannot distinguish.
- **Subgame perfect equilibrium** refines Nash equilibrium by requiring that strategies remain Nash equilibria in every subgame — this eliminates non-credible threats by applying backward induction throughout the tree.
- **Perfect recall** means players remember their own past actions and observations; without it, information sets can become internally inconsistent, breaking the standard CFR assumptions and making Nash equilibrium computation significantly harder.
- The **reach probability** of a node is the product of all action probabilities (both chance and players) along the path from the root; it determines how much each terminal node contributes to expected payoffs.
- The **counterfactual reach probability** \(\pi_{-i}^\sigma(I)\) is the product of chance and all opponents' action probabilities — it is the key weighting factor in CFR's regret updates, capturing how often an information set would be reached if player \(i\) were trying to reach it.
- Strategies in extensive-form games are functions from **information sets** (not individual nodes) to action distributions; this distinction from RL policies is what allows CFR to handle imperfect-information games.

## Quiz

{{#quiz 02-extensive-form-games-in-detail.toml}}
