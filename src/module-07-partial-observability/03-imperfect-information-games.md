# Lesson 3: Imperfect-Information Games

## Where this fits

Module 5 introduced game theory — normal-form games, extensive-form trees, information sets, and CFR. Module 7 so far has covered POMDPs and belief states: how an agent reasons under uncertainty about the world. This lesson fuses both threads.

In a POMDP, the world is partially observable but not adversarial. In an imperfect-information game, there are multiple agents, each making decisions, and each one cannot see the other's private information. The defender of a satellite constellation does not see the adversary's fuel reserves, maneuver intent, or sensor schedule. The adversary does not see the defender's confidence levels or coverage gaps. Both sides make decisions based on beliefs, not certainties.

This is the natural formalization for space ISR (intelligence, surveillance, and reconnaissance) operations: a competitive, partially-observable environment where each player's private information shapes strategy and the correct solution concept is equilibrium over beliefs, not states.

The central new ideas are: (1) information sets have a belief-state interpretation, not just a tree-structural one; (2) the value of private information can be quantified; and (3) reasoning about what the opponent knows about what you know requires either truncation (level-k reasoning) or equilibrium.

## Perfect vs. imperfect information

In a **perfect-information game**, both players always know the full game state. Chess is the classic example: the entire board is visible to both players at all times. Neither player has private information. The only uncertainty is about the future — what the opponent will do next — not about the present.

In an **imperfect-information game**, at least one player has information that the other lacks. This introduces a fundamentally different structure:

- In chess, the set of optimal actions is a deterministic function of the board state. Optimal chess is solved by backward induction (Minimax).
- In an imperfect-information game, optimal strategies are probabilistic even for rational, fully-capable players. This is not a computational limitation; it is a mathematical necessity, as we will see.

**Example contrast:**

In a perfect-information satellite conjunction game (both operators see each other's full mission profile, fuel state, and risk tolerance), the operator with the higher risk tolerance will maneuver, and both operators know this in advance. The equilibrium is deterministic: the low-risk-tolerance operator holds; the high-risk-tolerance operator maneuvers.

In the same game with private mission profiles, neither operator knows the other's risk tolerance. The equilibrium is a mixed strategy: each operator randomizes their decision, and the mixing probabilities depend on the distribution of types in the population. Deterministic strategies are exploitable (if you always hold, the opponent will always hold too, knowing you will hold).

## The SSA imperfect-information game

Consider a two-player SSA scenario: a **defender** operating a sensor network and a **challenging operator** (challenger) managing an adversarial satellite.

**Challenger's private information:**
- Remaining fuel budget (delta-V reserve)
- Target orbit (the conjunction orbit the challenging satellite is approaching)
- Maneuver timing window (when the maneuver must be executed to achieve the target orbit)

**Defender's private information:**
- Sensor schedule (which sectors are being observed, and when)
- Detection threshold (the minimum observable signal for anomaly detection)
- Confidence levels in current orbital element estimates for the challenging satellite

**Shared (common knowledge):**
- Both satellites' last publicly known orbital elements (from public catalogs)
- The physics of orbital mechanics (identical for both)
- The fact that this is a strategic interaction

**Why this is an imperfect-information game:**

The challenger's optimal maneuver timing depends on whether the defender's sensors are pointed away from them. If the challenger knew when the defender is looking, they would time the maneuver for a coverage gap. But the defender's sensor schedule is private.

The defender's optimal sensor schedule depends on when the challenger is likely to maneuver. If the defender knew the challenger's fuel reserve and timing window, they could concentrate observations on the conjunction approach. But those parameters are private.

Each player is forced to act on beliefs about the other's private state, not direct knowledge. This is the defining characteristic of an imperfect-information game.

## Information sets revisited: the belief-state interpretation

In Module 5, an **information set** was defined structurally: a set of game tree nodes that the current player cannot distinguish. The player's strategy must be identical at every node within an information set because, from that player's perspective, those nodes look identical.

Now we can give information sets a probabilistic interpretation: **an information set is the set of game tree nodes that are consistent with the current player's belief.**

Formally, if player \\(i\\) has information set \\(I\\) at time \\(t\\), there exists a probability distribution \\(b_I\\) over the nodes in \\(I\\) such that:

\\[ b_I(h) = P(\text{game is at node } h \mid \text{player } i\text{'s observations so far}) \\]

This is exactly the POMDP belief state, but embedded in a game tree rather than a single-agent MDP.

**Decoding the connection:**

In a POMDP, the belief \\(b(s) = P(s \mid \text{history})\\) is updated using Bayes' rule as new observations arrive.

In an imperfect-information game, each player maintains a belief over nodes within their current information set. As the game proceeds and new information arrives (observations of opponent actions, public chance outcomes), beliefs are updated using Bayes' rule in exactly the same way.

The formal solution concept for imperfect-information games, **Perfect Bayesian Equilibrium (PBE)**, makes this explicit: every player's strategy must be a best response to a consistent belief system, and the belief system must be updated by Bayes' rule wherever possible.

## Game tree formalization: the ISR game

Let us formalize the ISR game as an extensive-form tree with information sets.

**Players:** Defender (D), Challenger (C)

**Chance move (before play begins):** Nature assigns the challenger a type:
- \\(\theta_H\\) (high fuel, can reach conjunction orbit): probability \\(p\\)
- \\(\theta_L\\) (low fuel, cannot reach conjunction orbit): probability \\(1 - p\\)

The challenger knows their type. The defender does not.

**Challenger's action:** Choose to maneuver (M) or hold (H). This is observed by both players.

**Defender's action:** After observing the challenger's action, allocate sensors to the challenger (Focus) or distribute sensors evenly (Spread).

**Payoffs:** The defender wants to detect a conjunction-threatening maneuver. The challenger wants to reach the conjunction orbit undetected if they are type \\(\theta_H\\).

```
                    [Nature]
                  p /       \ 1-p
           [Type: H]         [Type: L]
              |                  |
         [Challenger]        [Challenger]
          M /   \ H            M /   \ H
           |     |              |     |
        {D info set 1}       {D info set 2}
          F / S              F / S
```

The defender's information sets are:
- \\(I_M\\) = {node after (H, M), node after (L, M)}: both look identical to the defender, because the defender observes a maneuver but not the type.
- \\(I_H\\) = {node after (H, H), node after (L, H)}: both look identical to the defender — no maneuver was seen.

At \\(I_M\\), the defender must use the same strategy (same probability of Focus vs. Spread) at both nodes, because those nodes are indistinguishable.

The defender's belief at \\(I_M\\) is:
\\[ b_{I_M}(H \mid M) = \frac{P(M \mid H) \cdot p}{P(M \mid H) \cdot p + P(M \mid L) \cdot (1-p)} \\]

**Decoding:** This is Bayes' rule. The defender has seen a maneuver (M) and is updating their belief about the challenger's type. \\(P(M \mid H)\\) is the probability the high-fuel challenger would maneuver (from the defender's perspective, this is the challenger's strategy). \\(P(M \mid L)\\) is the probability the low-fuel challenger would maneuver. The prior is \\(p\\) for type H.

This is the key moment where game theory and POMDPs merge: the defender's belief system is a POMDP belief, but the observation being processed is an opponent's *strategic action*, not a physical measurement.

## The value of information

**Definition:** The value of information is the expected improvement in a player's payoff from learning the value of a hidden variable, before making a decision.

\\[ \text{VOI}(X) = \mathbb{E}\left[\max_a u(a, X)\right] - \max_a \mathbb{E}[u(a, X)] \\]

**Decoding:**
- Left term: expected payoff when you *know* \\(X\\) before acting — you choose optimally for each realization of \\(X\\).
- Right term: expected payoff when you must choose \\(a\\) before knowing \\(X\\) — you choose the action that maximizes expected payoff under the prior over \\(X\\).
- VOI is the difference. It is always non-negative: knowing more information never hurts (in a single-agent setting).

**SSA example:** The defender must decide whether to Focus sensors on the challenger or Spread coverage. The hidden variable \\(X\\) is the challenger's type (H or L).

Suppose payoffs are (simplified):
- Focus + type H maneuver: +10 (detected a real threat)
- Focus + type L maneuver: -3 (wasted focus on non-threat, missed elsewhere)
- Spread + type H maneuver: -5 (threat happened but detection probability was lower)
- Spread + type L maneuver: +1 (balanced coverage, no major threat anyway)

With prior \\(p = P(\text{type H}) = 0.3\\):

```python
import numpy as np

# Payoff table: rows = defender actions, columns = challenger types
# [Focus, Spread] x [Type H, Type L]
payoffs = np.array([
    [10, -3],   # Focus
    [-5,  1],   # Spread
])

p_H = 0.3   # prior probability of high-fuel challenger
prior = np.array([p_H, 1 - p_H])

# Expected payoff without information (best action given prior)
expected_payoffs = payoffs @ prior
best_action_no_info = np.argmax(expected_payoffs)
value_no_info = expected_payoffs[best_action_no_info]

print(f"E[payoff | Focus]  = {expected_payoffs[0]:.2f}")
print(f"E[payoff | Spread] = {expected_payoffs[1]:.2f}")
print(f"Best action without info: {'Focus' if best_action_no_info == 0 else 'Spread'}, "
      f"value = {value_no_info:.2f}")

# Expected payoff WITH information (choose optimally per type realization)
best_per_type = payoffs.max(axis=0)   # best action for H, best action for L
value_with_info = best_per_type @ prior

print(f"\nBest payoff if type H revealed: {best_per_type[0]:.2f} (Focus)")
print(f"Best payoff if type L revealed: {best_per_type[1]:.2f} (Spread)")
print(f"Expected payoff with perfect info: {value_with_info:.2f}")

voi = value_with_info - value_no_info
print(f"\nValue of information: {voi:.2f}")
print(f"Interpretation: the defender would gain {voi:.2f} additional expected payoff units")
print(f"by learning the challenger's type before acting.")
```

The VOI gives a concrete bound: it tells the defender how much it is worth spending on intelligence gathering (additional observations, intelligence sources, etc.) to reduce uncertainty about the challenger's type.

## Level-k reasoning and why equilibrium matters

When a player builds a model of the opponent, they enter a recursive chain:

- **Level 0**: the opponent plays randomly.
- **Level 1**: I model the opponent as level 0 and best-respond.
- **Level 2**: I model the opponent as level 1 and best-respond.
- **Level k**: I model the opponent as level k-1 and best-respond.

This is level-k reasoning. In laboratory experiments, human subjects often behave at level 1 or level 2. The problem with level-k reasoning in adversarial SSA is twofold:

**First**, if the opponent is also reasoning at level k, they will be modeling you as level k-1 and responding accordingly. You will be surprised by their strategy.

**Second**, the chain does not converge to a stable strategy. Each level k generates a different response, and there is no terminal point.

Nash equilibrium is the principled resolution: an equilibrium strategy is a level-\\(\infty\\) best response that is also its own best response. Both players playing equilibrium means neither player can profit by deviating — there is no level at which reasoning breaks down.

The connection to CFR (Module 5, lesson 3): CFR iteratively updates both players' strategies until neither can improve by deviating. It is the computational method for finding the equilibrium that level-k reasoning approximates but never reaches.

## Perfect Bayesian Equilibrium: the solution concept

A **Perfect Bayesian Equilibrium (PBE)** of an imperfect-information game is a pair of:
1. A strategy profile \\(\sigma\\): one strategy per player at every information set.
2. A belief system \\(\mu\\): a probability distribution over game tree nodes, one distribution per information set.

Subject to two requirements:
- **Sequential rationality**: at every information set, each player's strategy is a best response to the other players' strategies, given their belief at that information set.
- **Belief consistency**: beliefs are derived from strategies via Bayes' rule wherever the prior probability of reaching that information set is nonzero.

**Decoding:** PBE combines equilibrium (Nash) with belief updating (Bayes). A strategy that is not a best response given the player's belief is eliminated. Beliefs that are inconsistent with Bayes' rule (given the strategy profile) are eliminated. What remains is a self-consistent system of strategies and beliefs.

In the ISR game, a PBE has the form: "challenger maneuveres with probability \\(\sigma_H\\) if type H and \\(\sigma_L\\) if type L; defender uses Focus with probability \\(\sigma_F \mid M\\) if a maneuver is observed and \\(\sigma_F \mid H\\) if no maneuver is observed; defender's beliefs at each information set are derived from the challenger's strategy via Bayes' rule."

## Full Python code: PBE computation via backward induction

```python
import numpy as np
from typing import Dict, Tuple

# ── ISR game specification ────────────────────────────────────────────────────

# Payoff structure:
# (challenger type, challenger action, defender action) -> (challenger payoff, defender payoff)
# Challenger types: H (high fuel), L (low fuel)
# Challenger actions: M (maneuver), H (hold)
# Defender actions: F (focus), S (spread)

PAYOFFS = {
    # (chal_type, chal_action, def_action): (chal_payoff, def_payoff)
    ('H', 'M', 'F'): (-2,  8),   # H type maneuvers, detected: bad for C, good for D
    ('H', 'M', 'S'): ( 5, -4),   # H type maneuvers, missed: good for C, bad for D
    ('H', 'H', 'F'): ( 0,  0),   # H type holds, focus: mutual low payoff
    ('H', 'H', 'S'): ( 1,  1),   # H type holds, spread: modest mutual benefit
    ('L', 'M', 'F'): (-3, -2),   # L type maneuvers (bluff), detected: both lose
    ('L', 'M', 'S'): ( 2, -1),   # L type maneuvers (bluff), missed: C gains some
    ('L', 'H', 'F'): (-1, -2),   # L type holds, focus (wasted): D overreacted
    ('L', 'H', 'S'): ( 0,  2),   # L type holds, spread: D covered correctly
}

def compute_pbe(p_H: float, tol: float = 1e-6, max_iter: int = 10000
               ) -> Dict[str, float]:
    """
    Compute the Perfect Bayesian Equilibrium of the 2-player ISR game
    via iterated best response (backward induction in the belief-extended tree).
    
    Returns a dictionary with equilibrium strategies and beliefs.
    
    p_H: prior probability that challenger is type H.
    """
    # Defender's optimal strategy given beliefs at each info set.
    # Let sigma_F_M = P(Focus | maneuver observed)
    # Let sigma_F_H = P(Focus | hold observed)
    # Challenger's strategy: sigma_H = P(Maneuver | type H), sigma_L = P(Maneuver | type L)

    # Initialize strategies
    sigma_H = 0.5   # P(maneuver | type H)
    sigma_L = 0.3   # P(maneuver | type L)

    def defender_belief_at_M(sH, sL):
        """P(type H | maneuver observed) via Bayes."""
        num = sH * p_H
        den = sH * p_H + sL * (1 - p_H)
        return num / den if den > 1e-12 else p_H

    def defender_belief_at_Hold(sH, sL):
        """P(type H | hold observed) via Bayes."""
        num = (1 - sH) * p_H
        den = (1 - sH) * p_H + (1 - sL) * (1 - p_H)
        return num / den if den > 1e-12 else p_H

    def defender_expected_payoff_focus(belief_H: float) -> float:
        """Expected payoff for Defender choosing Focus, given belief about type."""
        # Under Focus (F):
        # If type H (prob belief_H): M already happened, so payoff is PAYOFFS[H,M,F][1]
        # But here we are computing expected payoff at info set, given type belief.
        pay_H = PAYOFFS[('H', 'M', 'F')][1]   # defender payoff if H type maneuvered
        pay_L = PAYOFFS[('L', 'M', 'F')][1]   # defender payoff if L type maneuvered
        return belief_H * pay_H + (1 - belief_H) * pay_L

    def defender_expected_payoff_spread_at_M(belief_H: float) -> float:
        pay_H = PAYOFFS[('H', 'M', 'S')][1]
        pay_L = PAYOFFS[('L', 'M', 'S')][1]
        return belief_H * pay_H + (1 - belief_H) * pay_L

    def defender_best_response_at_M(belief_H: float) -> float:
        """Return P(Focus) for defender at info set after observing maneuver."""
        ev_F = defender_expected_payoff_focus(belief_H)
        ev_S = defender_expected_payoff_spread_at_M(belief_H)
        if ev_F > ev_S + tol:
            return 1.0   # pure Focus
        elif ev_S > ev_F + tol:
            return 0.0   # pure Spread
        else:
            return 0.5   # indifferent: any mixing works; equilibrium pins it down

    def defender_expected_payoff_focus_at_Hold(belief_H: float) -> float:
        pay_H = PAYOFFS[('H', 'H', 'F')][1]
        pay_L = PAYOFFS[('L', 'H', 'F')][1]
        return belief_H * pay_H + (1 - belief_H) * pay_L

    def defender_expected_payoff_spread_at_Hold(belief_H: float) -> float:
        pay_H = PAYOFFS[('H', 'H', 'S')][1]
        pay_L = PAYOFFS[('L', 'H', 'S')][1]
        return belief_H * pay_H + (1 - belief_H) * pay_L

    def defender_best_response_at_Hold(belief_H: float) -> float:
        ev_F = defender_expected_payoff_focus_at_Hold(belief_H)
        ev_S = defender_expected_payoff_spread_at_Hold(belief_H)
        if ev_F > ev_S + tol:
            return 1.0
        elif ev_S > ev_F + tol:
            return 0.0
        else:
            return 0.5

    def challenger_H_expected_payoff_maneuver(sigma_FM, sigma_FH):
        """Type-H challenger expected payoff from maneuvering."""
        ev = sigma_FM * PAYOFFS[('H', 'M', 'F')][0] + (1 - sigma_FM) * PAYOFFS[('H', 'M', 'S')][0]
        return ev

    def challenger_H_expected_payoff_hold(sigma_FM, sigma_FH):
        ev = sigma_FH * PAYOFFS[('H', 'H', 'F')][0] + (1 - sigma_FH) * PAYOFFS[('H', 'H', 'S')][0]
        return ev

    def challenger_L_expected_payoff_maneuver(sigma_FM, sigma_FH):
        ev = sigma_FM * PAYOFFS[('L', 'M', 'F')][0] + (1 - sigma_FM) * PAYOFFS[('L', 'M', 'S')][0]
        return ev

    def challenger_L_expected_payoff_hold(sigma_FM, sigma_FH):
        ev = sigma_FH * PAYOFFS[('L', 'H', 'F')][0] + (1 - sigma_FH) * PAYOFFS[('L', 'H', 'S')][0]
        return ev

    # Iterated best response loop
    for iteration in range(max_iter):
        old_sH, old_sL = sigma_H, sigma_L

        # Step 1: compute defender's beliefs given challenger's current strategy
        belief_at_M    = defender_belief_at_M(sigma_H, sigma_L)
        belief_at_Hold = defender_belief_at_Hold(sigma_H, sigma_L)

        # Step 2: defender best responds to those beliefs
        sigma_FM = defender_best_response_at_M(belief_at_M)
        sigma_FH = defender_best_response_at_Hold(belief_at_Hold)

        # Step 3: challenger best responds to defender's strategy
        # Type H
        ev_H_M = challenger_H_expected_payoff_maneuver(sigma_FM, sigma_FH)
        ev_H_H = challenger_H_expected_payoff_hold(sigma_FM, sigma_FH)
        if ev_H_M > ev_H_H + tol:
            new_sH = 1.0
        elif ev_H_H > ev_H_M + tol:
            new_sH = 0.0
        else:
            new_sH = sigma_H   # indifferent: stay at current mix

        # Type L
        ev_L_M = challenger_L_expected_payoff_maneuver(sigma_FM, sigma_FH)
        ev_L_H = challenger_L_expected_payoff_hold(sigma_FM, sigma_FH)
        if ev_L_M > ev_L_H + tol:
            new_sL = 1.0
        elif ev_L_H > ev_L_M + tol:
            new_sL = 0.0
        else:
            new_sL = sigma_L

        sigma_H = 0.9 * sigma_H + 0.1 * new_sH   # smooth update for convergence
        sigma_L = 0.9 * sigma_L + 0.1 * new_sL

        # Convergence check
        if abs(sigma_H - old_sH) < tol and abs(sigma_L - old_sL) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return {
        "sigma_H (P(maneuver | type H))": sigma_H,
        "sigma_L (P(maneuver | type L))": sigma_L,
        "sigma_FM (P(Focus | maneuver))": defender_best_response_at_M(
                                            defender_belief_at_M(sigma_H, sigma_L)),
        "sigma_FH (P(Focus | hold))":     defender_best_response_at_Hold(
                                            defender_belief_at_Hold(sigma_H, sigma_L)),
        "defender belief at M (P(H|M))":  defender_belief_at_M(sigma_H, sigma_L),
        "defender belief at Hold (P(H|Hold))": defender_belief_at_Hold(sigma_H, sigma_L),
    }

def run_pbe_analysis() -> None:
    """Run PBE computation for several prior probabilities of high-fuel type."""
    for p_H in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n--- Prior P(type H) = {p_H:.1f} ---")
        result = compute_pbe(p_H=p_H)
        for key, val in result.items():
            print(f"  {key}: {val:.4f}")

if __name__ == "__main__":
    run_pbe_analysis()
```

## Connecting to CFR

CFR (from Module 5, lesson 3) is the algorithm that efficiently finds the Nash equilibrium of imperfect-information extensive-form games. The structure we just described — information sets, belief updating, sequential rationality — is exactly what CFR operates on.

The connection is direct:

**Reach probabilities in CFR track beliefs.** At each information set \\(I\\), CFR maintains the counterfactual reach probability \\(\pi_{-i}(I)\\): the probability that play reaches \\(I\\) assuming all players *except* player \\(i\\) play according to the current strategy. This is the unnormalized belief over nodes in \\(I\\).

**Counterfactual values answer the POMDP question.** The counterfactual value \\(v_i^\sigma(I, a)\\) is the expected payoff if player \\(i\\) always took action \\(a\\) at information set \\(I\\) and all other players followed their current strategies. This is "what would my expected payoff be if I used this action, given the opponent's strategy" — exactly the question a rational agent with belief \\(b_I\\) asks when choosing an action.

**Regret accumulation drives belief-consistent strategies.** When CFR increases the probability of action \\(a\\) because its regret is positive (it would have done better), it is essentially saying: "given the opponent strategies I have encountered, this action performs well across the distribution of information sets I have been in." The distribution over information sets encountered is the induced belief distribution.

The advantage of CFR over the belief-state value iteration for POMDPs is computational: CFR operates on the *policy* space (strategies at information sets), not the *belief* space (distributions over states). The belief space is continuous; the strategy space is parameterized by a probability per action per information set, which is finite for finite games. CFR avoids the intractability of explicit belief-space planning.

## Key Takeaways

- Imperfect-information games arise whenever two strategic agents each have private information. The SSA defender does not see the challenger's fuel budget; the challenger does not see the defender's sensor schedule. Both sides must reason under uncertainty about hidden state.
- Information sets have both a structural interpretation (nodes the player cannot distinguish) and a probabilistic interpretation (a distribution over nodes consistent with the player's observations). The two are equivalent; the probabilistic view connects directly to POMDP belief states.
- The defender's belief about the challenger's type is updated by Bayes' rule when observing the challenger's actions. A maneuver is evidence about the challenger's type; the strength of the evidence depends on how likely each type would have maneuvered under the challenger's equilibrium strategy.
- The Value of Information quantifies how much the defender would benefit from learning the challenger's private state. It bounds the rational investment in intelligence collection and provides a principled way to prioritize observation resources.
- Level-k reasoning approximates equilibrium but never reaches it, and makes the agent exploitable by a higher-level reasoner. Perfect Bayesian Equilibrium is the principled solution concept: strategies and beliefs are jointly consistent, with beliefs derived from strategies via Bayes' rule.
- CFR (from Module 5) computes Nash equilibrium for imperfect-information games by operating on the strategy space at information sets, avoiding the intractability of explicit belief-space planning. Reach probabilities in CFR implicitly track the belief distribution the PBE framework makes explicit.

{{#quiz 03-imperfect-information-games.toml}}
