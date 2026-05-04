# Lesson 7: Hierarchical Reinforcement Learning

**Module:** Reinforcement Learning — M03: Sequential Decision-Making
**Source:** [cite: Sutton & Barto "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning" (options paper); Precup, Sutton & Singh "Between MDPs and Semi-MDPs"; Barto & Mahadevan "Recent Advances in Hierarchical Reinforcement Learning"; Nachum et al. "Data-Efficient Hierarchical Reinforcement Learning (HIRO)" NeurIPS 2018]

---

## Where this fits

Actor-critic (lesson 6) gave us a two-headed architecture — policy and value — that updates every primitive timestep. That works for tasks with compact action spaces and dense rewards. In the SSA wargame we are building toward, a single flat policy must simultaneously choose which orbital plane to contest, how to allocate sensors, and which satellite to maneuver on the current turn. That joint action space is enormous, and the reward signal for a strategic decision (contest GEO sector 3) may not materialize for dozens of turns. Gradients wash out before they can propagate backward to the strategic choice.

Hierarchical RL (HRL) solves this by decomposing the decision hierarchy — exactly as military doctrine decomposes decisions into strategic, operational, and tactical layers. A high-level policy makes slow, coarse decisions (sub-goals or options); a low-level policy executes fast, fine-grained actions to achieve them. Each level sees a reward signal at its own timescale, which makes credit assignment tractable at every layer.

Module 4's AlphaZero uses a two-level structure: the policy network (which option to expand) and the value network (what the current position is worth). The SSA wargame's full architecture, recommended by the Air University (2024) dissertation on wargame AI, extends this to three layers. This lesson teaches the theory and a concrete PyTorch implementation.

---

## Why flat policies fail for complex tasks

Consider what a single flat policy must output for an SSA orbital dominance wargame:

- **Strategic dimension**: which GEO belt sectors to contest, when to form coalitions with partner nations, how to posture the constellation for the next 30 days
- **Operational dimension**: how to allocate ground-based sensors across observation windows, which satellites to retask, when to execute a phasing maneuver campaign
- **Tactical dimension**: which satellite to maneuver right now, what delta-V to apply, which ground station to uplink through

If these decisions are all encoded in a single action vector, the effective action space is the Cartesian product of all three dimensions. With even modest cardinality at each level, the number of distinct actions reaches tens of thousands. A policy network trained by policy gradient must estimate \\(Q(s, a)\\) — or equivalently \\(A(s, a)\\) — for all of these simultaneously.

Two problems compound:

**1. Gradient signal sparsity at the strategic level.** When the agent chooses to "contest GEO sector 3," that choice has consequences over the next 30 simulated days. A reward obtained 100 timesteps in the future is discounted by \\(\gamma^{100} \approx 0.0\\) for \\(\gamma = 0.99\\). The policy gradient update that should reinforce the strategic choice is functionally zero. The agent cannot learn which strategic decisions are good because the signal disappears before it arrives.

**2. Effective action masking.** Most combinations of strategic, operational, and tactical actions are incoherent: the tactical action "apply 50 m/s east delta-V to satellite 4" is irrelevant to the strategic objective currently in force. A flat policy wastes representational capacity learning to avoid all the incoherent combinations.

The military analogy is apt. A combatant commander does not personally choose which frequency band each tactical radio uses; that decision is delegated. Each layer decides within a scope defined by the layer above. HRL formalizes exactly this: each level operates on its own timescale, with its own reward signal, within the scope assigned by the level above.

---

## The options framework

Sutton, Precup, and Singh formalized temporal abstraction with the **options framework**. An option is a temporally extended action — it may last for one primitive timestep or for hundreds.

Formally, an option \\(o\\) is a triple:

\\[ o = (I,\; \pi_o,\; \beta_o) \\]

**Decoding:**
- \\(I \subseteq \mathcal{S}\\): the **initiation set** — the set of states where option \\(o\\) can be started. Not all options are available from every state (you cannot execute "perform a Hohmann transfer to GEO" if your fuel budget is depleted).
- \\(\pi_o: \mathcal{S} \times \mathcal{A} \to [0,1]\\): the **intra-option policy** — the primitive action distribution to use while executing option \\(o\\). This is a full policy, focused on achieving the option's goal rather than the long-term global objective.
- \\(\beta_o: \mathcal{S} \to [0,1]\\): the **termination condition** — the probability that option \\(o\\) terminates at each state. When termination is sampled as true, execution returns to the higher level, which selects a new option.

### Semi-Markov Decision Processes

When options execute for variable durations, the resulting process is no longer Markov at the level of option transitions — the time between option selections is random. This is the **Semi-Markov Decision Process (SMDP)** formulation.

The SMDP Q-value for a policy-over-options \\(\mu\\) that selects option \\(o\\) in state \\(s\\) is:

\\[ Q_\mu(s, o) = \mathbb{E}_\mu\!\left[\sum_{k=0}^{\tau-1} \gamma^k R_{t+k+1} + \gamma^\tau \sum_{o'} \mu(o' \mid s_{t+\tau})\, Q_\mu(s_{t+\tau}, o') \;\middle|\; s_t = s,\, o_t = o\right] \\]

**Decoding:**
- \\(\tau\\): the random duration of option \\(o\\) in primitive timesteps, determined by sampling from \\(\beta_o\\) at each step during execution.
- \\(\sum_{k=0}^{\tau-1} \gamma^k R_{t+k+1}\\): the cumulative discounted reward collected during the option's execution — real primitive rewards, each discounted from the option's start time.
- \\(\gamma^\tau \sum_{o'} \mu(o' \mid s_{t+\tau}) Q_\mu(s_{t+\tau}, o')\\): the value of the state reached when the option terminates, discounted by the full option duration \\(\tau\\). The high-level policy \\(\mu\\) then selects the next option from that terminal state.

The discount \\(\gamma^\tau\\) is the key: options that complete quickly are discounted less than options that drag on, creating an incentive structure that prefers efficient goal achievement.

### SSA example: the "maneuver to eclipse" option

In an SSA context, consider the option "maneuver satellite 4 to eclipse geometry relative to the adversary's inspection satellite":

- **Initiation set** \\(I\\): states where satellite 4's delta-V budget is at least 15 m/s and its current orbital period is within 10% of the target's orbital period (Hohmann transfer is feasible).
- **Intra-option policy** \\(\pi_o\\): a low-level orbital mechanics controller that applies a sequence of delta-V burns following a Hohmann transfer trajectory. This can be handcrafted (orbital mechanics is analytic) or learned.
- **Termination condition** \\(\beta_o\\): terminates with probability 1.0 when satellite 4 enters the eclipse zone (defined by the Earth's shadow cone projected against the inspection satellite's line of sight), or with probability 1.0 if fuel falls below 2 m/s (option aborted for fuel conservation).

From the perspective of the high-level policy, "maneuver to eclipse" is a single action. The high level does not observe the 30–100 primitive timesteps of burns; it sees the state before the option starts and the state at termination. The intermediate rewards are accumulated and discounted into the single SMDP Q-value update.

---

## Two-level HRL architecture

The canonical two-level HRL structure separates timescales explicitly:

```
High-level policy  μ(o | s_t)         — selects every K primitive timesteps
        ↓ option o (sub-goal or abstract action)
Low-level policy   π_o(a | s_t)       — selects every primitive timestep
        ↓ primitive action a
Environment        s_{t+1}, R_{t+1}   — transitions at every timestep
```

**High-level** operates on the slow timescale: it observes the state every K steps and selects a new option or sub-goal. It may observe a coarsened version of the state — strategic variables like sector coverage percentages, coalition status, aggregate fuel budgets — rather than the full tactical state.

**Low-level** operates on the fast timescale: every primitive timestep it observes the full state and executes the primitive action that best achieves the option assigned by the high level.

### SMDP Bellman equation for the high level

The high level's value function satisfies a Bellman equation over option durations:

\\[ V_\mu(s) = \sum_o \mu(o \mid s)\; Q_\mu(s, o) \\]

\\[ Q_\mu(s, o) = \mathbb{E}\!\left[ \sum_{k=0}^{\tau-1} \gamma^k R_{t+k+1} + \gamma^\tau V_\mu(s_{t+\tau}) \;\middle|\; s, o \right] \\]

**Decoding:** The high-level Q-value is the expected sum of all primitive rewards during the option (each discounted from the option's start), plus the discounted value of the next state — where "next state" means the state when the option terminates, which may be many steps later. The high level reasons over longer time horizons; it does not need to track what happened at each primitive step.

This Bellman equation is structurally identical to the standard Bellman equation from lesson 2, but \\(\tau\\) replaces \\(1\\) in the discount. For deterministic fixed-duration options, it reduces exactly to the standard case with an effective discount of \\(\gamma^K\\).

### Why two timescales help credit assignment

Consider the strategic decision "contest GEO sector 3" which takes effect over 50 simulated turns. With a flat policy at \\(\gamma = 0.99\\), the gradient from the final outcome is multiplied by \\(\gamma^{50} \approx 0.61\\) before reaching the strategic decision. That is manageable, but the policy must correctly disentangle 50 steps of confounded tactical and operational actions to identify which strategic choice produced the outcome.

With HRL, the high-level update bootstraps from the option's terminal state after those 50 steps. The high-level policy gradient becomes:

\\[ \nabla_\phi J(\phi) = \mathbb{E}\!\left[ A_\mu(s, o)\, \nabla_\phi \log \mu(o \mid s;\, \phi) \right] \\]

where \\(A_\mu(s, o)\\) is the high-level advantage for that option. The high-level policy receives a **single clean update** per option execution rather than 50 noisy updates that must be integrated. The low-level policy, meanwhile, receives dense per-step feedback about whether it is achieving the option's goal efficiently. Each level gets the right kind of signal at the right timescale.

---

## Goal-conditioned policies and HIRO

Options define abstract discrete actions. A more flexible approach, used by HIRO (Nachum et al., 2018), replaces discrete options with a **continuous sub-goal vector** output by the high-level policy.

Instead of selecting option \\(o\\) from a finite set, the high level outputs a sub-goal \\(g \in \mathbb{R}^d\\) — a vector in state space (or an embedding space) specifying what the low level should achieve. The low level's policy becomes:

\\[ \pi_\text{lo}(a \mid s,\, g) \\]

a goal-conditioned policy that takes both the current state and the sub-goal as input. The low level's reward is the **sub-goal reward**, typically:

\\[ r_\text{lo}(s_t, g, s_{t+1}) = -\lVert s_{t+1} - g \rVert_2 \\]

The low-level policy is rewarded for moving the state closer to the sub-goal, regardless of the extrinsic environment reward. The high level is rewarded for choosing sub-goals whose pursuit yields high extrinsic reward.

### HIRO's off-policy sub-goal correction

Training HIRO off-policy from a replay buffer requires a correction. The sub-goal \\(g\\) stored in the buffer was generated by the high-level policy at collection time, but the current high-level policy might assign a different sub-goal to the same state. Using stale sub-goals biases the high-level Q-learning update.

HIRO's solution is **sub-goal relabeling**: when replaying a stored trajectory \\((s_t, g_t, a_t, \ldots, s_{t+K})\\), find the sub-goal \\(\hat{g}\\) that maximizes the probability that the current low-level policy would have generated the observed actions:

\\[ \hat{g} = \arg\max_{g \in \mathcal{G}} \sum_{i=0}^{K-1} \log \pi_\text{lo}(a_{t+i} \mid s_{t+i},\, g) \\]

**Decoding:** Among a finite set of candidate sub-goals (typically the original \\(g_t\\) plus several random perturbations), choose the sub-goal that the current low-level policy would be most likely to pursue given the observed actions. This corrects the mismatch between stale sub-goals and the current policy. The corrected \\(\hat{g}\\) is used in place of the stored \\(g_t\\) for the high-level Q-function update.

### SSA example: goal-conditioned constellation management

In an SSA wargame, the high-level policy outputs a sub-goal as a target state vector for the constellation:

\\[ g = [\text{coverage}_{GEO3} = 0.85,\; \text{coverage}_{LEO\_polar} = 0.60,\; \text{fuel\_reserve} = 0.40,\; \ldots] \\]

This specifies a desired aggregate constellation state: 85% coverage of GEO sector 3, 60% polar LEO coverage, 40% fuel reserves maintained. The low-level policy receives this sub-goal vector and commands individual satellite maneuvers — phasing burns, station-keeping corrections, sensor pointing adjustments — that move the actual constellation state toward the target.

The high-level policy does not specify which satellite performs which maneuver. That is the low level's job. The high level reasons about desired aggregate states; the low level reasons about how to achieve them from the current physical configuration.

---

## Option-critic architecture

Both the options framework and HIRO require either handcrafted option definitions or a two-stage training procedure. The **option-critic** architecture (Bacon, Harb & Precup, 2017) learns everything end-to-end from the external reward alone, including the termination functions \\(\beta_o\\).

Option-critic parameterizes:
- **Policy over options** \\(\mu(o \mid s;\, \phi)\\): a softmax over \\(K\\) options from state \\(s\\)
- **Intra-option policy** \\(\pi(a \mid s, o;\, \theta)\\): action distribution conditioned on current state and active option
- **Termination function** \\(\beta(s, o;\, \psi)\\): probability of terminating option \\(o\\) at state \\(s\\)

### Intra-option policy gradient

The gradient for the intra-option policy uses the **option advantage function**:

\\[ A_\Omega(s, o, a) = Q_U(s, o, a) - Q_\Omega(s, o) \\]

where \\(Q_U(s, o, a)\\) is the value of taking action \\(a\\) in state \\(s\\) while executing option \\(o\\), and \\(Q_\Omega(s, o)\\) is the value of option \\(o\\) in state \\(s\\) averaged over the intra-option policy. The intra-option policy gradient for \\(\theta\\) is:

\\[ \nabla_\theta \mathcal{L}(\theta) = \mathbb{E}\!\left[ A_\Omega(s_t, o_t, a_t)\; \nabla_\theta \log \pi(a_t \mid s_t, o_t;\, \theta) \right] \\]

This is exactly the actor-critic gradient from lesson 6, but conditioned on the currently active option \\(o_t\\).

### Termination gradient

The termination function \\(\beta_o\\) is trained to minimize the cost of continuing the current option versus switching. Define:

\\[ A_{\Omega,\text{term}}(s', o) = Q_\Omega(s', o) - V_\Omega(s') \\]

where \\(V_\Omega(s') = \sum_{o'} \mu(o' \mid s')\, Q_\Omega(s', o')\\) is the value of the policy-over-options at state \\(s'\\). The termination gradient is:

\\[ \nabla_\psi \mathcal{L}(\psi) = \mathbb{E}\!\left[ A_{\Omega,\text{term}}(s_{t+1}, o_t)\; \nabla_\psi \beta(s_{t+1}, o_t;\, \psi) \right] \\]

**Decoding:** If \\(A_{\Omega,\text{term}}(s', o) > 0\\), option \\(o\\) is more valuable than the average option at \\(s'\\) — decrease termination probability and keep executing. If \\(A_{\Omega,\text{term}}(s', o) < 0\\), the average option from \\(s'\\) is better — increase termination probability and return control to the high level. The termination function learns when to "give up" on the current option.

### SSA: how option-critic discovers useful options

In an SSA wargame, option-critic starts with \\(K\\) randomly initialized options and no human-specified semantics. Over training, gradient signals push options to differentiate based on what is useful. Satellites in LEO interact rapidly with adversary assets; satellites in GEO interact slowly but strategically. The termination gradient learns that GEO options should persist for many turns (low \\(\beta\\)) while LEO options should terminate and switch quickly (high \\(\beta\\)). Without explicit definition, two coherent behavioral modes tend to emerge: something resembling "consolidate sensors toward contested regions" and something resembling "disperse sensors for broad surveillance." Both are discovered by gradient descent on extrinsic reward; neither was specified by a human.

---

## Three-layer SSA wargame decomposition

The full SSA orbital dominance wargame calls for three decision layers with distinct timescales:

| Layer | Timescale | Decisions | State representation |
|-------|-----------|-----------|----------------------|
| **Strategic** | Every N turns (N = 10–20) | Which orbital planes to contest; coalition formation; campaign objectives; budget allocation | Sector coverage percentages, diplomatic variables, aggregate force structure |
| **Operational** | Every few turns (3–5) | Asset allocation between missions; sensor tasking plans; constellation management; maneuver campaign planning | Constellation state, sensor queue, fuel budgets, threat assessments |
| **Tactical** | Every turn | Individual satellite maneuver commands; intercept geometry; immediate sensor pointing | Full orbital state of each asset, current intercept geometries, real-time coverage |

Each layer has its own policy network, value network, and reward signal — a shaped version of the global reward that is meaningful at that layer's temporal resolution.

### Why this decomposition improves convergence

**Reduced effective action space at each level.** The strategic level chooses among roughly 10 high-level objectives. The operational level chooses among roughly 20 asset allocation configurations. The tactical level chooses among roughly 5 maneuver commands per satellite. Compare this to the flat alternative: \\(10 \times 20 \times 5 = 1000\\) joint actions that the policy must reason about simultaneously.

**Meaningful gradient signal at each level.** The strategic layer receives reward every N turns from outcomes attributable to strategic choices (sector contested or not). The tactical layer receives dense reward every turn from immediate maneuver outcomes. Neither level has to bridge the other's timescale.

**Hierarchical curriculum.** The operational and tactical layers can be pre-trained independently — or with handcrafted low-level controllers — before the strategic layer is introduced. Staged training prevents the strategic layer from receiving garbage signals from a randomly-acting tactical layer during early training.

---

## Implementation: a 2-level HRL agent in PyTorch

The following implements a simplified HIRO-style two-level agent applied to an SSA sub-task: the high level selects which orbital slot sector to prioritize (a coarse sub-goal), and the low level commands satellite maneuvers to achieve observation of that sector.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from collections import deque


# ---------------------------------------------------------------------------
# High-level policy: selects a sub-goal every K primitive timesteps
# ---------------------------------------------------------------------------

class HighLevelPolicy(nn.Module):
    """
    Takes a strategic state (aggregate constellation metrics) and outputs
    a sub-goal index representing which orbital slot sector to prioritize.
    Updates every K primitive timesteps.
    """
    def __init__(self, state_dim: int, n_subgoals: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_subgoals)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.net(state)
        logits   = self.policy_head(features)
        value    = self.value_head(features).squeeze(-1)
        return logits, value

    def select_subgoal(self, state: torch.Tensor):
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)
        subgoal  = dist.sample()
        log_prob = dist.log_prob(subgoal)
        return subgoal.item(), log_prob, value, dist.entropy()


# ---------------------------------------------------------------------------
# Low-level policy: goal-conditioned actor-critic
# ---------------------------------------------------------------------------

class LowLevelPolicy(nn.Module):
    """
    Takes (state, sub_goal_index) and outputs a primitive action.
    The sub-goal is embedded and concatenated with the state features
    so the policy can specialize its behavior per assigned sub-goal.
    """
    def __init__(self, state_dim: int, n_subgoals: int, n_actions: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.goal_embed = nn.Embedding(n_subgoals, hidden_dim // 2)
        self.state_net  = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.joint_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, subgoal: torch.Tensor):
        state_feat = self.state_net(state)
        goal_feat  = self.goal_embed(subgoal)
        combined   = torch.cat([state_feat, goal_feat], dim=-1)
        features   = self.joint_net(combined)
        logits = self.policy_head(features)
        value  = self.value_head(features).squeeze(-1)
        return logits, value

    def select_action(self, state: torch.Tensor, subgoal: int):
        subgoal_t = torch.tensor(subgoal)
        logits, value = self.forward(state, subgoal_t)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value, dist.entropy()


# ---------------------------------------------------------------------------
# Sub-goal relabeling: simplified HIRO off-policy correction
# ---------------------------------------------------------------------------

def relabel_subgoal(
    low_level:          LowLevelPolicy,
    state_seq:          list,
    action_seq:         list,
    candidate_subgoals: list,
) -> int:
    """
    Given a stored low-level trajectory and candidate sub-goals, return the
    sub-goal that the current low-level policy would most likely have produced.

    This is the HIRO relabeling: pick the sub-goal maximizing
        sum_i  log π_lo(a_i | s_i, g)
    over the stored (s_i, a_i) pairs.

    Args:
        low_level:           current LowLevelPolicy
        state_seq:           list of state tensors from the trajectory
        action_seq:          list of int primitive actions
        candidate_subgoals:  list of int sub-goal indices

    Returns:
        best_subgoal (int)
    """
    best_subgoal  = candidate_subgoals[0]
    best_log_prob = float('-inf')

    with torch.no_grad():
        for g in candidate_subgoals:
            total_lp = 0.0
            for s, a in zip(state_seq, action_seq):
                logits, _ = low_level(
                    s.unsqueeze(0),
                    torch.tensor(g).unsqueeze(0)
                )
                dist      = Categorical(logits=logits.squeeze(0))
                total_lp += dist.log_prob(torch.tensor(a)).item()
            if total_lp > best_log_prob:
                best_log_prob = total_lp
                best_subgoal  = g

    return best_subgoal


# ---------------------------------------------------------------------------
# HRL agent: separate optimizers, K-step high-level re-selection
# ---------------------------------------------------------------------------

class HRLAgent:
    """
    Two-level HRL agent.
    High level: selects sub-goal every K primitive steps; trained as SMDP actor-critic.
    Low level:  selects primitive action every step; trained with dense per-step reward.
    """
    def __init__(
        self,
        state_dim:    int,
        n_subgoals:   int,
        n_actions:    int,
        K:            int   = 5,
        gamma:        float = 0.99,
        lr_hi:        float = 3e-4,
        lr_lo:        float = 3e-4,
        entropy_coef: float = 0.01,
    ):
        self.K            = K
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.n_subgoals   = n_subgoals

        self.hi = HighLevelPolicy(state_dim, n_subgoals)
        self.lo = LowLevelPolicy(state_dim, n_subgoals, n_actions)

        # Separate optimizers: each level can be tuned independently
        self.opt_hi = torch.optim.Adam(self.hi.parameters(), lr=lr_hi)
        self.opt_lo = torch.optim.Adam(self.lo.parameters(), lr=lr_lo)

    def run_episode(self, env) -> dict:
        state = torch.tensor(env.reset(), dtype=torch.float32)

        # Low-level bookkeeping (every primitive step)
        lo_log_probs, lo_values, lo_rewards = [], [], []
        lo_entropies, lo_dones              = [], []

        # High-level bookkeeping (every K steps)
        hi_metrics = []

        # Initial sub-goal selection
        subgoal, hi_lp, hi_val, hi_ent = self.hi.select_subgoal(state)
        hi_cumulative_reward = 0.0
        hi_discount          = 1.0
        option_states        = []
        option_actions       = []

        done = False
        step = 0

        while not done:
            action, lo_lp, lo_val, lo_ent = self.lo.select_action(
                state.unsqueeze(0), subgoal
            )
            next_np, reward, done = env.step(action)
            next_state = torch.tensor(next_np, dtype=torch.float32)

            # Accumulate low-level trajectory
            lo_log_probs.append(lo_lp)
            lo_values.append(lo_val)
            lo_rewards.append(reward)
            lo_entropies.append(lo_ent)
            lo_dones.append(done)

            # Accumulate within-option trajectory for relabeling
            option_states.append(state)
            option_actions.append(action)
            hi_cumulative_reward += hi_discount * reward
            hi_discount          *= self.gamma
            step += 1

            # High-level re-selection every K steps or at episode end
            if step % self.K == 0 or done:
                # Sub-goal relabeling: correct for off-policy mismatch
                relabeled = relabel_subgoal(
                    self.lo,
                    option_states,
                    option_actions,
                    list(range(self.n_subgoals)),
                )
                hi_metrics.append({
                    'log_prob': hi_lp,
                    'value':    hi_val,
                    'entropy':  hi_ent,
                    'reward':   hi_cumulative_reward,
                    'done':     done,
                })

                if not done:
                    subgoal, hi_lp, hi_val, hi_ent = \
                        self.hi.select_subgoal(next_state)
                    hi_cumulative_reward = 0.0
                    hi_discount          = 1.0
                    option_states        = []
                    option_actions       = []

            state = next_state

        # --- Low-level update (Monte Carlo actor-critic) ---
        lo_lp_t  = torch.stack(lo_log_probs)
        lo_val_t = torch.stack(lo_values)
        lo_ent_t = torch.stack(lo_entropies)

        lo_returns = []
        G = 0.0
        for r, d in zip(reversed(lo_rewards), reversed(lo_dones)):
            G = r + self.gamma * G * (1.0 - float(d))
            lo_returns.insert(0, G)
        lo_ret_t  = torch.tensor(lo_returns, dtype=torch.float32)
        lo_adv_t  = lo_ret_t - lo_val_t.detach()

        lo_loss = (
            -(lo_lp_t * lo_adv_t).mean()
            + 0.5 * F.mse_loss(lo_val_t, lo_ret_t)
            - self.entropy_coef * lo_ent_t.mean()
        )
        self.opt_lo.zero_grad()
        lo_loss.backward()
        nn.utils.clip_grad_norm_(self.lo.parameters(), 0.5)
        self.opt_lo.step()

        # --- High-level update (SMDP actor-critic) ---
        if hi_metrics:
            hi_lp_t  = torch.stack([m['log_prob'] for m in hi_metrics])
            hi_val_t = torch.stack([m['value']    for m in hi_metrics])
            hi_ent_t = torch.stack([m['entropy']  for m in hi_metrics])
            hi_r_list = [m['reward'] for m in hi_metrics]
            hi_d_list = [m['done']   for m in hi_metrics]

            # High-level returns use γ^K as the effective discount per option
            hi_returns = []
            G = 0.0
            for r, d in zip(reversed(hi_r_list), reversed(hi_d_list)):
                G = r + (self.gamma ** self.K) * G * (1.0 - float(d))
                hi_returns.insert(0, G)
            hi_ret_t = torch.tensor(hi_returns, dtype=torch.float32)
            hi_adv_t = hi_ret_t - hi_val_t.detach()

            hi_loss = (
                -(hi_lp_t * hi_adv_t).mean()
                + 0.5 * F.mse_loss(hi_val_t, hi_ret_t)
                - self.entropy_coef * hi_ent_t.mean()
            )
            self.opt_hi.zero_grad()
            hi_loss.backward()
            nn.utils.clip_grad_norm_(self.hi.parameters(), 0.5)
            self.opt_hi.step()

        return {
            'ep_return': sum(lo_rewards),
            'lo_loss':   lo_loss.item(),
            'n_options': len(hi_metrics),
        }


def train_hrl(env, agent: HRLAgent, n_episodes: int = 300,
              print_every: int = 50) -> list:
    returns = []
    for ep in range(n_episodes):
        m = agent.run_episode(env)
        returns.append(m['ep_return'])
        if (ep + 1) % print_every == 0:
            recent = returns[-print_every:]
            avg    = sum(recent) / len(recent)
            print(f"Episode {ep+1:>4}: avg_return={avg:.2f}, "
                  f"n_options={m['n_options']}, lo_loss={m['lo_loss']:.4f}")
    return returns
```

### How this maps to the SSA wargame

In the SSA context, the environment above represents a simplified wargame where the state is a 16-dimensional vector encoding orbital slot coverages, fuel budgets, and threat proximity for four GEO sectors. The sub-goals are four discrete objectives corresponding to "prioritize sector 0 through 3." The primitive actions are five maneuver commands per satellite turn (station-keep, east phasing, west phasing, raise orbit, lower orbit).

The high level re-evaluates every K = 5 turns. During those five turns, the low level executes maneuvers consistent with achieving good coverage of the selected sector. At the end of five turns, the high level observes the resulting state and may switch its sector priority if the tactical situation has changed — for example, an adversary asset entered a previously low-priority sector.

The sub-goal relabeling ensures that older buffer data, collected under a different high-level policy, is corrected before being used to update the current high-level policy. Without this correction, the high level would receive biased gradient signals from transitions where the low level was pursuing a sub-goal different from the one stored.

---

## Failure modes

### Sub-goal assignment problem

The high-level policy may assign a sub-goal that is impossible to achieve from the current state. In the SSA context: if the high level assigns "observe GEO sector 3" but all satellites capable of reaching that sector have insufficient fuel, the low level can never satisfy the sub-goal. The low-level reward is persistently negative (never close to the sub-goal state), and the high-level policy receives no useful gradient about whether sector 3 is worth prioritizing.

**Mitigations:** Mask infeasible sub-goals at the high level (requires an explicit feasibility function); add a penalty to the high level when the low level fails to make progress toward the sub-goal; use curriculum training that introduces demanding sub-goals only after the low level has mastered simpler ones.

### Reward hacking at the sub-goal level

If the low-level reward is a simple proximity measure \\(-\lVert s - g \rVert\\), the low level may find policies that reduce distance along dimensions that are easy to change but not meaningful. For example, moving a satellite's velocity vector closer to the sub-goal vector without actually achieving the intended orbital coverage. The low-level policy satisfies the shaped reward while producing no useful behavior for the high level.

**Mitigation:** Design the sub-goal reward to capture only the dimensions that matter for the high-level objective (coverage achieved, not velocity components), and evaluate the high level's extrinsic reward separately from the low level's intrinsic sub-goal reward.

### Multi-timescale credit assignment problem

Even with HRL, credit assignment remains imperfect across very long horizons. If a strategic decision sets up a situation that pays off 50 high-level option executions later, the high-level discount \\(\gamma^{50 K}\\) still attenuates the signal substantially. HRL reduces the problem relative to flat RL but does not eliminate it for extremely long-horizon planning. Very deep hierarchies may require additional mechanisms: hindsight experience replay, explicit task decomposition, or learning a world model to plan forward.

### When flat policies beat HRL

HRL introduces meaningful implementation complexity: two training loops, separate optimizers, buffer management, and sub-goal relabeling. In simple environments with dense rewards and small action spaces, this overhead exceeds the benefit. A standard DQN or actor-critic agent will typically outperform HRL when:

- Episodes are short (fewer than 50 steps) and rewards are dense throughout
- The action space has fewer than 20 actions, all meaningful at every step
- The task has a single natural timescale with no useful sub-task decomposition

For the SSA wargame — with hundreds of turns, multi-level command decisions, and sparse strategic reward — HRL's additional complexity pays for itself. For the simpler satellite scheduling environments in this module's project, a flat DQN or A2C agent is appropriate and easier to debug.

---

## Key Takeaways

- **An option** \\(o = (I, \pi_o, \beta_o)\\) is a temporally extended action defined by an initiation set (where it can start), an intra-option policy (what to do during execution), and a termination condition (when to return control to the high level). Options execute for variable numbers of primitive timesteps, producing a Semi-Markov Decision Process at the high level whose Q-values accumulate discounted rewards over the full option duration before bootstrapping from the terminal state.
- **Temporal abstraction solves the credit assignment problem** for hierarchical tasks: the high level receives a single update per option execution rather than attempting to propagate gradients across dozens to hundreds of confounded primitive steps. Each level receives dense reward appropriate to its own timescale, making learning tractable at every layer simultaneously.
- **Goal-conditioned HRL (HIRO)** replaces discrete options with continuous sub-goal vectors; the low-level policy is conditioned on the sub-goal and rewarded for state-space proximity to it. HIRO's off-policy sub-goal relabeling corrects stale buffer data by finding the sub-goal that the current low-level policy would most likely have been pursuing given the observed action sequence.
- **Option-critic** learns termination functions \\(\beta_o\\) end-to-end from extrinsic reward, discovering useful option boundaries without manual specification. The termination gradient increases termination probability when the average option at the current state is better than the current option, and decreases it when the current option is the best available — options persist when they are working and switch when something better is available.
- **The three-layer SSA decomposition** — strategic (N-turn), operational (few-turn), tactical (every turn) — reduces the effective action space at each level, provides meaningful gradient signal at each timescale, and enables curriculum training where lower levels stabilize before upper levels are introduced. This architecture mirrors military command doctrine and is validated by recent Air University research on AI for wargame agents.
- **HRL adds complexity that must be justified**: the sub-goal assignment problem, low-level reward hacking, and residual multi-timescale credit assignment are failure modes that do not exist in flat RL. In environments with short episodes, dense rewards, and compact action spaces, a flat actor-critic outperforms HRL with far less implementation overhead. Reserve HRL for tasks where the decision hierarchy is genuinely multi-scale.

{{#quiz 07-hierarchical-rl.toml}}
