# Lesson 5: Centralized Training, Decentralized Execution

**Module:** Multi-Agent Reinforcement Learning — M06: MARL
**Source:** [cite: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" NeurIPS 2022; Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" ICML 2018; Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" NeurIPS 2017 (MADDPG); Oliehoek & Amato "A Concise Introduction to Decentralized POMDPs"]

---


<!-- toc -->

## Where this fits

Lesson 1 diagnosed the core difficulty of multi-agent RL: non-stationarity. Because every agent is learning simultaneously, no agent faces a fixed environment, and the convergence guarantees of single-agent RL break down. Lesson 3 addressed one half of the problem — adversarial settings — with PSRO, which builds a population of policies and converges toward Nash equilibrium through iterative best-response training. PSRO is the right tool when agents are opponents.

But many operationally important settings are cooperative: a coalition of allied satellites must share ISR coverage, deconflict communications windows, and coordinate orbit change maneuvers. These agents share a common reward and want to coordinate, not compete. PSRO's adversarial orientation does not apply.

This lesson introduces the cooperative counterpart: **Centralized Training, Decentralized Execution (CTDE)**. CTDE is the organizing principle behind MAPPO and QMIX — the two algorithms most widely used in cooperative MARL research. The lesson covers CTDE's conceptual foundation, MAPPO's architecture and implementation, QMIX's value decomposition approach, and how both fit into the full SSA wargame architecture built on Ray RLlib and MARLlib (Module 8).

Module 3's actor-critic framework is directly relevant here: MAPPO is actor-critic multi-agent learning with a centralized critic. Readers who have not yet read Lesson 1 of this module should start there for the non-stationarity framing.

---

## The cooperative MARL problem

In adversarial MARL — the domain of PSRO and self-play — every agent is trying to outplay the others. What is good for one agent is bad for another. The equilibrium concept is Nash: no agent wants to unilaterally deviate.

Cooperative MARL is different: all agents share a common reward. Every satellite in the allied coalition receives the same coverage score. There are no competing incentives; the challenge is purely coordination. How do five satellites collectively cover the orbital regime without redundancy and without gaps?

Naive independent RL fails here in a specific way. Each agent runs its own policy gradient loop — its own PPO or Q-learning — treating the other agents as part of the environment. The gradient update for agent i assumes that agent j's policy is fixed during the current update step. But agent j is also updating. The assumption is false.

The result is that independent RL agents often converge to **miscoordinated** policies. Consider a canonical SSA coordination problem: five satellites must collectively cover 20 observation slots in a GEO belt sector. No slot should be assigned to two satellites (waste) and no slot should be unassigned (gap). If each satellite runs independent PPO with only its local coverage reward, a common failure mode is that all five satellites converge toward the same high-value slots — because those slots have the highest immediate reward signal — and the remaining slots are never covered. Each satellite's gradient pointed toward the individually rewarding slots, with no mechanism to account for what the others were doing.

The mathematical reason: independent PPO for agent i computes policy gradients of the form:

\[ \nabla_{\theta_i} J_i \approx \mathbb{E} \left[ \nabla_{\theta_i} \log \pi_i(a_i | o_i) \cdot \hat{A}_i \right] \]

**Decoding:**
- \(\theta_i\): the parameters of agent i's policy
- \(\pi_i(a_i | o_i)\): agent i's policy — the probability of taking action \(a_i\) given local observation \(o_i\)
- \(\hat{A}_i\): the advantage estimate for agent i — how much better was the action than expected
- The expectation is over trajectories sampled from the joint policy of all agents

The problem is that \(\hat{A}_i\) is computed from agent i's value function \(V_i(o_i)\), which only depends on agent i's local observation. If agent i's advantage is estimated from \(r + \gamma V_i(o_i') - V_i(o_i)\), and the reward r partly results from what all other agents did, then the advantage function conflates the contributions of all agents. Agent i's gradient update receives credit (or blame) for outcomes that were driven by agent j's actions, not its own.

This is not merely a theoretical concern. Empirical results in the MARL literature consistently show that independent PPO with per-agent value functions fails on cooperative tasks that require careful division of labor.

---

## CTDE: the core idea

Centralized Training, Decentralized Execution resolves the cooperative MARL problem by separating the training-time information structure from the execution-time information structure.

**During training**: The critic (value function) observes the full **joint state** — the concatenation of all agents' observations, actions, and positions. The joint state is available in simulation because the training environment has access to all information. Crucially, the joint state transition is Markov even when individual agents' observations are not. A single satellite cannot predict where debris will move without knowing what the other satellites are observing; but the collective state of all five satellites and all tracked debris objects evolves according to known orbital mechanics. The centralized critic, seeing the full joint state, eliminates the non-stationarity problem from the value function's perspective: the value target is no longer a moving target contaminated by other agents' unseen updates.

**During execution**: Each agent's actor policy uses only its own **local observation** \(o_i\). The centralized critic is discarded at test time; it was a training crutch, not a deployment requirement. Each satellite's policy network takes as input only what that satellite's sensors can measure and outputs actions for that satellite alone. No communication between satellites is required at runtime.

This separation is critical for real SSA deployment. In operations, each satellite in a coalition may be in a communication blackout, may have high-latency ground links, or may be operating in an electronically contested environment where inter-satellite communication is denied. The CTDE-trained policy works correctly under all of these conditions because it was designed to function with only local information at execution time.

The insight is elegant: use the training period (offline, in simulation, with full information available) to solve the coordination problem, and bake the coordination into the policy weights. At runtime, the coordination knowledge is implicit in how the policy responds to its local observations — no explicit communication or centralized controller is needed.

---

## MAPPO: Multi-Agent PPO

MAPPO is the simplest and most effective CTDE algorithm. Yu et al. (2022) showed, somewhat surprisingly, that MAPPO with straightforward hyperparameters matches or outperforms much more complex cooperative MARL algorithms on standard benchmarks including StarCraft Multi-Agent Challenge (SMAC) and Multi-Agent MuJoCo. The centralized critic is what makes it work — it almost entirely solves the non-stationarity problem that defeats independent PPO.

### Architecture

MAPPO has two components:

**One centralized critic** \(V_\phi(s_\text{global})\): takes the concatenation of all agents' observations as input and outputs a single scalar value estimate. This is the joint state value — the expected discounted return from the current joint state onward when all agents follow their current policies.

**N decentralized actor policies** \(\pi_{\theta_i}(a_i | o_i)\): each takes only agent i's local observation as input and outputs a distribution over agent i's actions. Each actor has its own separate parameters \(\theta_i\).

### Advantage estimation

The advantage for agent i is computed using the centralized critic:

\[ A_i(s_\text{global}, a_i) = r + \gamma V_\phi(s'_\text{global}) - V_\phi(s_\text{global}) \]

**Decoding:**
- \(A_i\): the advantage for agent i — how much better was agent i's action than the value predicted by the joint state value function
- \(s_\text{global}\): the full joint state at the current step (all agents' observations concatenated)
- \(s'_\text{global}\): the full joint state at the next step
- \(r\): the shared team reward
- \(\gamma\): the discount factor
- The advantage uses the centralized critic for both the target and the baseline, so it accounts for all agents' contributions to the joint outcome

In practice, Generalized Advantage Estimation (GAE) is used instead of the single-step advantage above. GAE reduces variance by blending multi-step returns:

\[ \hat{A}_i^{\text{GAE}} = \sum_{t=0}^{T} (\gamma \lambda)^t \delta_{t+k} \]

\[ \delta_t = r_t + \gamma V_\phi(s'_{\text{global},t}) - V_\phi(s_{\text{global},t}) \]

**Decoding:**
- \(\lambda \in [0, 1]\): the GAE parameter controlling the bias-variance tradeoff (\(\lambda = 0\) is pure TD; \(\lambda = 1\) is pure Monte Carlo)
- \(\delta_t\): the TD residual at step t, computed with the centralized critic
- The sum discounts future TD residuals by \((\gamma \lambda)^t\), giving a smooth interpolation between low-variance/high-bias and high-variance/low-bias advantage estimates

Because \(\delta_t\) uses the centralized critic — which sees the full joint state — the advantage estimate for agent i correctly accounts for the joint outcome, not just agent i's local observation. This is what prevents the credit assignment confusion that plagues independent PPO.

### Actor update

Each actor is updated with the PPO clipped surrogate objective:

\[ L_i^{\text{clip}}(\theta_i) = \mathbb{E}_t \left[ \min\left( r_t(\theta_i) \hat{A}_i, \; \text{clip}(r_t(\theta_i), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right] \]

\[ r_t(\theta_i) = \frac{\pi_{\theta_i}(a_i | o_i)}{\pi_{\theta_i^{\text{old}}}(a_i | o_i)} \]

**Decoding:**
- \(r_t(\theta_i)\): the probability ratio — how much more or less likely is action \(a_i\) under the new policy versus the old policy that collected the data
- \(\epsilon\): the clipping threshold (typically 0.1 or 0.2) — the policy cannot move more than this fraction away from the old policy in a single update step
- \(\hat{A}_i\): the advantage from the centralized critic via GAE
- The \(\min\) with the clipped ratio prevents destructively large policy updates while still taking steps in the direction of positive advantage

Each actor is updated independently with this objective, using the same advantage estimates from the shared centralized critic. The N actors do not share parameters unless domain knowledge suggests a symmetric role structure.

### Training loop structure

```
Collect rollouts:
  For each step t in rollout of length T:
    For each agent i:
      Sample action a_i ~ pi_i(o_i)   # decentralized actors
    Execute joint action (a_1, ..., a_N) in environment
    Observe next joint state s'_global, shared reward r, done flag
    Store (s_global, o_1,...,o_N, a_1,...,a_N, r, s'_global, done)

Compute advantages:
  For each step t:
    delta_t = r_t + gamma * V(s'_global_t) - V(s_global_t)
  Compute GAE: A_hat_t = sum_k (gamma*lambda)^k * delta_{t+k}

Update critic:
  Minimize MSE loss: L_critic = (V(s_global) - (A_hat + V_old(s_global)))^2

Update actors:
  For each agent i:
    Maximize clipped PPO objective using A_hat as the advantage
```

The SSA parallel: in a 5-satellite allied coalition, the rollout collector runs the simulation for T steps. At each step, each satellite's actor policy reads its local coverage footprint and task queue and selects a slot assignment. The shared reward is the total coalition coverage score for that step. The centralized critic sees all five satellites' positions, coverage maps, and task queues and estimates the joint expected return. GAE uses this centralized value estimate to produce per-agent advantages. Each satellite's actor then updates with its own PPO gradient using the centralized advantage. After training, only the five actor networks are deployed; the centralized critic stays in the training environment.

---

## MAPPO implementation in PyTorch

The following is a complete, functional MAPPO implementation. The design keeps the centralized critic and decentralized actors cleanly separated.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class CentralizedCritic(nn.Module):
    """
    Takes the concatenated observations of all agents as input.
    Outputs a single scalar joint state value estimate.

    During deployment this module is discarded; it is only used
    during training to compute advantages for the actor updates.
    """
    def __init__(self, joint_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """
        joint_obs: shape (batch, joint_obs_dim)
                   joint_obs_dim = n_agents * local_obs_dim (concatenated)
        Returns:   shape (batch, 1) — value estimate for the joint state
        """
        return self.net(joint_obs)


class DecentralizedActor(nn.Module):
    """
    Takes only this agent's local observation as input.
    Outputs a categorical distribution over discrete actions.

    This is the only network deployed on the satellite at execution time.
    """
    def __init__(self, local_obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        """
        local_obs: shape (batch, local_obs_dim)
        Returns:   shape (batch, action_dim) — unnormalized logits
        """
        return self.net(local_obs)

    def get_action(self, local_obs: torch.Tensor):
        """
        Sample an action and return (action, log_prob).
        Used during rollout collection.
        """
        logits = self.forward(local_obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probs and entropy for given actions.
        Used during the PPO update.

        Returns: (log_probs, entropy) both shape (batch,)
        """
        logits = self.forward(local_obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class MAPPOAgent:
    """
    MAPPO for N cooperative agents sharing a common reward.

    The centralized critic takes the joint observation (concatenation of
    all agents' observations). Each decentralized actor takes only its
    own agent's local observation.

    SSA application: n_agents satellites in an allied ISR coalition.
      - local_obs_dim: size of a single satellite's observation vector
          (e.g., coverage footprint bitmap, current task queue, fuel level)
      - joint_obs_dim: local_obs_dim * n_agents (all agents' obs concatenated)
      - action_dim: number of discrete slot assignments available per satellite
    """
    def __init__(
        self,
        n_agents: int,
        local_obs_dim: int,
        joint_obs_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        n_epochs: int = 10,
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs

        # One shared centralized critic for the joint state
        self.critic = CentralizedCritic(joint_obs_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # One decentralized actor per agent
        self.actors: List[DecentralizedActor] = [
            DecentralizedActor(local_obs_dim, action_dim)
            for _ in range(n_agents)
        ]
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]

    def compute_advantages(
        self,
        joint_obs_batch: np.ndarray,   # (T, joint_obs_dim)
        next_joint_obs: np.ndarray,    # (T, joint_obs_dim)
        rewards: np.ndarray,           # (T,)
        dones: np.ndarray,             # (T,) boolean
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and value targets using the centralized critic.

        Returns:
          advantages:    shape (T,) — used for all N actor updates
          value_targets: shape (T,) — used for critic MSE update
        """
        with torch.no_grad():
            joint_obs_t = torch.FloatTensor(joint_obs_batch)
            next_joint_obs_t = torch.FloatTensor(next_joint_obs)

            values = self.critic(joint_obs_t).squeeze(-1).numpy()           # (T,)
            next_values = self.critic(next_joint_obs_t).squeeze(-1).numpy() # (T,)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            # TD residual: delta_t = r_t + gamma * V(s'_t) * (1 - done) - V(s_t)
            next_val = next_values[t] * (1.0 - float(dones[t]))
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages[t] = gae

        value_targets = advantages + values  # A_hat + V_old = TD-lambda target

        # Normalize advantages across the rollout batch (stabilizes training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return torch.FloatTensor(advantages), torch.FloatTensor(value_targets)

    def update(self, rollouts: dict) -> dict:
        """
        PPO update for all actors and the centralized critic.

        rollouts keys:
          'joint_obs':      (T, joint_obs_dim)
          'next_joint_obs': (T, joint_obs_dim)
          'local_obs':      (N, T, local_obs_dim) — per-agent local observations
          'actions':        (N, T) — per-agent discrete actions taken
          'log_probs_old':  (N, T) — log-probs under the collecting policy
          'rewards':        (T,)
          'dones':          (T,)

        Returns: dict of training metrics for logging
        """
        advantages, value_targets = self.compute_advantages(
            rollouts['joint_obs'],
            rollouts['next_joint_obs'],
            rollouts['rewards'],
            rollouts['dones'],
        )

        joint_obs_t = torch.FloatTensor(rollouts['joint_obs'])
        metrics = {'actor_losses': [], 'critic_loss': 0.0, 'entropies': []}

        for _ in range(self.n_epochs):
            # ── Critic update ──────────────────────────────────────────────────
            values_pred = self.critic(joint_obs_t).squeeze(-1)
            critic_loss = nn.functional.mse_loss(values_pred, value_targets)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic_optimizer.step()
            metrics['critic_loss'] += critic_loss.item() / self.n_epochs

            # ── Actor updates (one per agent) ──────────────────────────────────
            for i, (actor, optimizer) in enumerate(
                zip(self.actors, self.actor_optimizers)
            ):
                local_obs_i = torch.FloatTensor(rollouts['local_obs'][i])  # (T, d)
                actions_i = torch.LongTensor(rollouts['actions'][i])       # (T,)
                log_probs_old_i = torch.FloatTensor(
                    rollouts['log_probs_old'][i]
                )  # (T,)

                log_probs_new, entropy = actor.evaluate_actions(
                    local_obs_i, actions_i
                )

                # Probability ratio for PPO clipping
                ratio = torch.exp(log_probs_new - log_probs_old_i)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()
                total_loss = actor_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                optimizer.step()

                metrics['actor_losses'].append(actor_loss.item())
                metrics['entropies'].append(entropy.mean().item())

        return metrics


# ── Example: 5-satellite ISR coalition ────────────────────────────────────────
# Each satellite observes its coverage footprint (20 slots, binary),
# its current task queue depth (10 values), and fuel level (1 scalar).
# local_obs_dim = 20 + 10 + 1 = 31
# joint_obs_dim = 31 * 5 = 155
# action_dim = 20 (assign this satellite to one of 20 observation slots)

N_AGENTS = 5
LOCAL_OBS_DIM = 31
JOINT_OBS_DIM = N_AGENTS * LOCAL_OBS_DIM
ACTION_DIM = 20

mappo = MAPPOAgent(
    n_agents=N_AGENTS,
    local_obs_dim=LOCAL_OBS_DIM,
    joint_obs_dim=JOINT_OBS_DIM,
    action_dim=ACTION_DIM,
)

# At deployment: each satellite uses only its local actor network.
# The centralized critic is not needed for inference.
print("Actor (deployed on each satellite):")
print(mappo.actors[0])
print(f"\nCritic (training only, joint_obs_dim={JOINT_OBS_DIM}):")
print(mappo.critic)
```

The output shows that the actor is a lightweight two-layer network taking a 31-dimensional local observation — suitable for deployment on a satellite flight computer. The critic is a deeper network consuming all 155 dimensions and is used only during training.

---

## QMIX: value decomposition

MAPPO is an actor-critic method: it maintains both a policy (actor) and a value function (critic). QMIX takes a different approach — it is a value-based method that learns Q-functions and uses value decomposition to enable decentralized execution.

### The core challenge: decentralized argmax

In single-agent Q-learning, the optimal action is \(\text{argmax}_a Q(s, a)\). In multi-agent settings, finding the optimal joint action requires \(\text{argmax}_{a_1, \ldots, a_N} Q_\text{joint}(s, a_1, \ldots, a_N)\). This is intractable for large N because the joint action space grows exponentially: with N=5 agents and 20 actions each, there are \(20^5 = 3.2 \times 10^6\) joint actions to evaluate.

QMIX solves this by learning factored Q-functions that respect a **monotonicity constraint**:

\[ \frac{\partial Q_\text{total}}{\partial Q_i} \geq 0 \quad \forall i \]

**Decoding:**
- \(Q_i(o_i, a_i)\): the individual Q-function for agent i, depending only on agent i's local observation and action
- \(Q_\text{total}(s_\text{global}, \mathbf{a})\): the joint Q-function that combines all individual Q-values
- \(\frac{\partial Q_\text{total}}{\partial Q_i} \geq 0\): the monotonicity constraint — \(Q_\text{total}\) is a non-decreasing function of each \(Q_i\)

The monotonicity constraint has a critical consequence: the argmax over the joint action decomposes into N independent argmaxes:

\[ \text{argmax}_\mathbf{a} \, Q_\text{total}(s_\text{global}, \mathbf{a}) = \bigl( \text{argmax}_{a_1} Q_1(o_1, a_1), \; \ldots, \; \text{argmax}_{a_N} Q_N(o_N, a_N) \bigr) \]

**Decoding:**
- Because \(Q_\text{total}\) is non-decreasing in each \(Q_i\), increasing \(Q_i\) by choosing a better action \(a_i\) can only increase or leave unchanged \(Q_\text{total}\)
- Therefore each agent can independently maximize its own Q-function without needing to coordinate with the others at execution time
- This is the Individual-Global-Max (IGM) principle: the joint argmax equals the element-wise argmax when monotonicity holds

The result: each satellite independently picks the slot that maximizes its own local Q-function, and the joint behavior is guaranteed to maximize the global Q-function — as long as the monotonicity constraint holds.

### The mixing network architecture

QMIX enforces monotonicity through a **mixing network** — a small neural network that takes the individual Q-values \(Q_1, \ldots, Q_N\) as inputs and outputs \(Q_\text{total}\). Monotonicity is enforced by constraining all weights in the mixing network to be non-negative.

The key insight: the weights are not fixed. QMIX uses **hypernetworks** — separate networks that take the global state \(s_\text{global}\) as input and generate the mixing network's weights. The hypernetwork outputs are passed through absolute value to guarantee non-negativity.

```python
import torch
import torch.nn as nn


class QMIXMixingNetwork(nn.Module):
    """
    QMIX mixing network: takes individual Q-values Q_1,...,Q_N and the
    global state s_global, and outputs Q_total.

    Monotonicity is enforced by generating non-negative mixing weights
    via hypernetworks conditioned on s_global. Non-negativity is achieved
    by taking the absolute value of hypernetwork outputs.

    SSA application: 5 satellites each provide a coverage slot Q-value;
    the mixing network combines them into a joint coverage Q_total
    conditioned on the full constellation state.
    """
    def __init__(
        self,
        n_agents: int,
        global_state_dim: int,
        mixing_hidden: int = 32,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden

        # Hypernetwork 1: generates first-layer weights (n_agents -> mixing_hidden)
        # abs() applied to outputs ensures non-negative mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, n_agents * mixing_hidden),
        )
        # Bias for first layer: unconstrained (bias does not break monotonicity)
        self.hyper_b1 = nn.Linear(global_state_dim, mixing_hidden)

        # Hypernetwork 2: generates output-layer weights (mixing_hidden -> 1)
        # abs() applied to outputs ensures non-negative mixing weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, mixing_hidden),
        )
        # Final state-conditioned bias (scalar)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, 1),
        )

    def forward(
        self,
        q_values: torch.Tensor,      # (batch, n_agents)
        global_state: torch.Tensor,  # (batch, global_state_dim)
    ) -> torch.Tensor:
        """
        Returns Q_total of shape (batch, 1).

        The forward pass:
          1. Generate mixing weights from hypernetworks conditioned on global_state
          2. Apply abs() to weight tensors (monotonicity guarantee)
          3. Pass individual Q-values through the two-layer mixing network
        """
        batch = q_values.size(0)
        q_values = q_values.view(batch, 1, self.n_agents)  # (B, 1, N)

        # First mixing layer: (B, 1, N) x (B, N, H) -> (B, 1, H)
        w1 = torch.abs(self.hyper_w1(global_state))           # (B, N*H)
        w1 = w1.view(batch, self.n_agents, self.mixing_hidden)  # (B, N, H)
        b1 = self.hyper_b1(global_state).view(batch, 1, self.mixing_hidden)
        hidden = torch.nn.functional.elu(torch.bmm(q_values, w1) + b1)

        # Second mixing layer (output): (B, 1, H) x (B, H, 1) -> (B, 1, 1)
        w2 = torch.abs(self.hyper_w2(global_state))        # (B, H)
        w2 = w2.view(batch, self.mixing_hidden, 1)
        b2 = self.hyper_b2(global_state).view(batch, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.view(batch, 1)


class IndividualQNetwork(nn.Module):
    """
    Per-agent Q-network taking only agent i's local observation.
    Outputs Q(o_i, a_i) for each discrete action a_i.

    Deployed on each satellite at execution time.
    The greedy action is argmax over this network's outputs.
    """
    def __init__(self, local_obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        """
        local_obs: (batch, local_obs_dim)
        Returns:   (batch, action_dim) Q-values for each action
        """
        return self.net(local_obs)


# ── QMIX training loss (sketch) ────────────────────────────────────────────────
# For a batch of transitions (o_i, a_i, r, o'_i, s_global, s'_global):
#
#   Q_i = individual_q_net_i(o_i)[a_i]      # scalar Q-value for taken action
#   Q_total = mixing_network([Q_1,...,Q_N], s_global)
#
#   # Bellman target (with frozen target networks):
#   Q_i_next_max = max_{a'} individual_q_net_i_target(o'_i)
#   Q_total_target = mixing_network_target([Q_1_next,...,Q_N_next], s'_global)
#   y = r + gamma * Q_total_target   (if not done)
#
#   loss = MSE(Q_total, y.detach())
#
# Gradients flow from Q_total back through the mixing network into each
# individual Q-network. All networks are trained jointly to minimize the
# Bellman loss, with target networks updated via Polyak averaging.
```

The SSA application: each of the 5 satellites runs its own Q-network on its local coverage state and outputs a Q-value for each of the 20 possible slot assignments. The mixing network takes all five Q-values and the full constellation state (all five satellites' positions and fuel levels) and produces a joint coverage Q-value. The Bellman target trains the whole system end-to-end. At deployment, each satellite independently takes the argmax of its own Q-network — the monotonicity guarantee ensures this produces the jointly optimal slot assignment.

---

## MAPPO vs. QMIX: when to use each

Both MAPPO and QMIX are CTDE algorithms. Their differences matter for practical application.

**Action space**: MAPPO works for continuous or discrete actions. QMIX requires discrete actions — the monotonicity decomposition requires evaluating Q over all possible actions to find the argmax. For continuous satellite thrust commands or pointing angles, MAPPO is the only option. For discrete slot assignments, both apply.

**Sample efficiency**: QMIX is generally more sample-efficient than MAPPO on cooperative discrete-action tasks. The value decomposition structure encodes the cooperative reward structure directly — the mixing network is an inductive bias toward cooperative behavior. MAPPO must learn cooperation purely from the policy gradient signal, which requires more interactions.

**Role heterogeneity**: MAPPO handles heterogeneous agents naturally — each agent has its own separate actor network, and there is no constraint on how different the agents' policies are. QMIX's mixing network combines all individual Q-values under a single architecture, which can be strained when agents have very different observation spaces or action semantics.

**Monotonicity failures**: The QMIX monotonicity assumption can fail in competitive settings. If an agent's action sometimes benefits the team and sometimes hurts it depending on the global state, the strict monotonicity constraint prevents the mixing network from representing this. MAPPO has no such structural constraint and handles mixed cooperative-competitive dynamics naturally.

**Tuning burden**: MAPPO inherits PPO's generally stable training behavior. QMIX requires careful tuning of target network update rates, replay buffer size, and the hypernetwork architecture. For applied research with limited compute budget, MAPPO is usually the faster path to a working result.

**Recommendation for the SSA wargame ally coalition**: MAPPO. The ally satellite coalition has diverse roles — ISR satellites, communication relays, and space control assets with structurally different observation spaces and action sets. MAPPO's per-agent actor architecture handles this naturally. The Yu et al. (2022) empirical evidence also supports MAPPO as a strong baseline that rarely needs to be replaced by a more complex algorithm.

---

## CTDE in the full SSA wargame architecture

The recommended implementation stack for the SSA orbital dominance wargame is Ray RLlib with the MARLlib extension (covered fully in Module 8). Understanding how CTDE maps onto this stack is useful for implementation planning.

**Training environment**: The Ray environment wrapper exposes two observation modes. The critic-mode observation for each agent concatenates all ally satellites' state vectors — positions in ECI coordinates, fuel reserves, current task queue, sensor health, and last observed coverage map. This joint observation feeds the centralized critic. The actor-mode observation for each agent contains only that satellite's own sensor readings and state. The wrapper handles this split automatically: the environment returns both, and MARLlib's MAPPO implementation routes them to the correct network.

**Centralized critic via global state augmentation**: MARLlib implements CTDE through a pattern called global state augmentation. During training, when the critic's forward pass is called, MARLlib passes the full joint state to it. When the actor's forward pass is called, MARLlib passes only the agent's local observation. The actor and critic are separate network classes (matching the `CentralizedCritic` and `DecentralizedActor` shown earlier), and MARLlib manages which inputs each receives.

**Deployment**: When training is complete, only the actor networks are exported. Each actor is a lightweight PyTorch module that takes a local observation tensor and returns a probability distribution over actions. In a real deployment, each actor would run on a satellite's onboard processor or a dedicated ground segment compute node for that satellite. No inter-satellite communication is required during normal operations.

**Adversarial and cooperative training in the same wargame**: The full wargame has both components. The Red faction's agents use PSRO with self-play (adversarial, Lesson 3). The Blue ally coalition uses MAPPO (cooperative, this lesson). Both run in the same Ray simulation environment. The interface is clean: Red agents and Blue agents share the same orbital mechanics simulation but have separate training loops, separate population management, and separate value functions. The adversarial and cooperative paradigms coexist because they operate on different factions with different objective structures.

**Scaling**: A constellation of 5 to 12 allied satellites is a practical size for MAPPO in the MARLlib stack. The centralized critic's input size scales linearly with the number of agents, and its computational cost during training is modest relative to the simulation cost. For very large constellations (50+ satellites), parameter-sharing — all agents share a single actor conditioned on an agent ID embedding — is a common approximation that drastically reduces parameter count while retaining most of the cooperative coordination benefit.

---

## Key Takeaways

- **Independent RL fails at cooperative tasks because each agent's advantage estimate conflates its own contribution with its teammates'.** The centralized critic in CTDE fixes this: by seeing the full joint state, the critic provides an advantage signal that correctly accounts for the joint outcome, and each actor's gradient update reflects its true marginal contribution to the team reward.
- **The centralized critic is a training crutch, not a deployment requirement.** It is discarded after training. The deployed agents use only their local-observation actor networks, which means CTDE-trained policies work correctly under communication denial, bandwidth constraints, and sensor blackouts — critical properties for operational SSA systems.
- **MAPPO's core insight (Yu et al. 2022) is that a centralized critic with standard PPO is sufficient.** Per-agent PPO actors, GAE advantages computed from the joint state value function, and standard clipping match or outperform far more complex MARL algorithms on cooperative benchmarks. The centralized critic does most of the work.
- **QMIX's monotonicity constraint enables decentralized greedy execution by making the individual argmax equal to the joint argmax.** Each agent independently maximizes its own Q-function, and the mixing network's non-negative weights guarantee this produces the globally optimal joint action — no coordination communication required at runtime.
- **MAPPO is preferred over QMIX for the SSA ally coalition** because the satellites have heterogeneous roles and observation spaces, MAPPO handles continuous action variants, and its training is more stable with less tuning. QMIX's sample efficiency advantage matters less than MAPPO's architectural flexibility for diverse agent types.
- **CTDE is the bridge between centralized planning and decentralized execution in the SSA wargame architecture.** Train the coalition with full information; deploy each satellite with only local information. The coordination knowledge is encoded in the policy weights during training and expressed through each satellite's autonomous behavior at runtime — no real-time coordination infrastructure required.

{{#quiz 05-ctde-mappo.toml}}
