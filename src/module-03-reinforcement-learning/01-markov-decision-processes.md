# Lesson 1: Markov Decision Processes

## Where this fits

Reinforcement learning needs a precise way to describe "an agent acts in an environment over time." The Markov Decision Process (MDP) is that description. Every algorithm in this curriculum from now on assumes the world is structured as an MDP (or a generalization of one). When you read about "states," "actions," "rewards," "transitions," and "discount factors" in any RL paper or codebase, those terms are coming from the MDP framework. Get comfortable with the vocabulary in this lesson and the rest of the module unfolds naturally.

## A space scenario to motivate everything

Imagine you are operating a single ground-based optical telescope. There are 5 satellites you could track at any given time. Once an hour, you decide which one to point at. Some of these satellites are doing routine things (boring to track), some are at risk of conjunction with debris (high value to track), and some are doing maneuvers that warrant close attention (also high value).

When you observe a satellite, you learn something about its current state and may detect interesting events. Tracking a quiet satellite gives you a small reward (basic mission accomplishment). Tracking a satellite during a conjunction event gives you a large reward (you caught it). Missing an event because you were pointed elsewhere is a missed opportunity.

You want a strategy: a rule for which satellite to point at, given everything you know about the current situation. This is a sequential decision problem. The MDP framework is how we formalize it.

## The five pieces of an MDP

An MDP is defined by five things, which together describe the entire decision problem.

### 1. State (S)

A **state** is everything the agent knows about the world at a particular moment.

For our telescope problem, the state might be:

```
state = (
  time_to_next_conjunction_satellite_1,  # hours
  time_to_next_conjunction_satellite_2,
  time_to_next_conjunction_satellite_3,
  time_to_next_conjunction_satellite_4,
  time_to_next_conjunction_satellite_5,
  hours_since_last_observation_1,
  hours_since_last_observation_2,
  hours_since_last_observation_3,
  hours_since_last_observation_4,
  hours_since_last_observation_5,
)
```

A 10-dimensional vector. The agent uses this to decide what to do next.

The set of all possible states is called the **state space**. For our problem, this is the space of all possible 10-tuples of non-negative real numbers. That is a continuous, infinite space; in practice we either discretize it or use function approximation (which is what neural networks do).

In simpler problems, the state space might be small and discrete. A tic-tac-toe game has fewer than 39 = 19,683 possible board states (with many illegal). A chess game has roughly 10^47 states (mostly illegal). The state space size determines what algorithms are practical.

### 2. Action (A)

An **action** is something the agent can do that affects the state.

For our telescope, there are 5 possible actions:

```
A = {Point at sat 1, Point at sat 2, Point at sat 3, Point at sat 4, Point at sat 5}
```

The set of possible actions is the **action space**. Action spaces can be:
- **Discrete and finite** (our telescope: 5 choices)
- **Discrete and infinite** (rare in practice)
- **Continuous** (e.g., a thrust vector with continuous magnitude and direction)

Different algorithms suit different action spaces. Q-learning and DQN work for discrete actions. Policy gradient methods work for both discrete and continuous. Pure tabular methods work only for small discrete action spaces.

## Continuous vs. discrete action spaces in practice

The telescope example has a clean discrete action space: 5 satellites, 5 choices. Real satellite operations are rarely this clean. Maneuvering a satellite involves commanding continuous thrust — a magnitude in Newtons and a direction in 3D space. There is no natural discretization.

Consider a debris avoidance maneuver. The action space might be:
- Thrust magnitude: any value in [0, 5] N
- Thrust direction: any unit vector in R³ (parameterized as azimuth and elevation)

This is a **continuous action space** with 3 degrees of freedom. No finite list of discrete actions captures it.

### The discretization approach

One option is to **bin** the continuous action space into a finite set of discrete choices:
- Thrust magnitudes: {0, 1, 2, 3, 4, 5} N — 6 levels
- Thrust azimuth: {0°, 45°, 90°, ..., 315°} — 8 directions
- Thrust elevation: {-45°, 0°, 45°} — 3 levels

This gives 6 × 8 × 3 = 144 discrete actions. DQN or Q-learning can now work on this problem. The cost is **loss of resolution**: the agent can only command one of 144 thrust vectors, not any vector in the continuous space. Fine maneuvers may be impossible.

### The direct continuous approach

Policy gradient methods (REINFORCE, PPO, SAC) work directly on continuous action spaces. Instead of outputting a probability distribution over a discrete set, the policy outputs parameters of a continuous distribution — typically a Gaussian mean and variance for each action dimension. The agent samples from this Gaussian to get an actual thrust command.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# ── Discrete telescope pointing ──────────────────────────────────────────────
# 5 satellites, 5 discrete actions. DQN-style output.

class TelescopePolicy(nn.Module):
    """Discrete policy: output one probability per satellite."""
    def __init__(self, state_dim, n_satellites=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_satellites),
        )

    def forward(self, state):
        logits = self.net(state)
        return Categorical(logits=logits)   # discrete distribution over 5 actions

# ── Continuous satellite thrust ───────────────────────────────────────────────
# Thrust: magnitude [0,5] N and direction in 3D. Policy gradient output.

class ThrustPolicy(nn.Module):
    """Continuous policy: output mean and log-std for each thrust dimension."""
    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, action_dim)
        # log-std is a learned parameter (not input-dependent here for simplicity)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.net(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).clamp(min=1e-4)
        return Normal(mean, std)   # continuous distribution over thrust vectors

# Example: sample an action from each policy
state = torch.randn(10)   # 10-dimensional state vector

telescope_policy = TelescopePolicy(state_dim=10)
dist_discrete = telescope_policy(state)
action_discrete = dist_discrete.sample()
print(f"Discrete action (which satellite): {action_discrete.item()}")

thrust_policy = ThrustPolicy(state_dim=10, action_dim=3)
dist_continuous = thrust_policy(state)
action_continuous = dist_continuous.sample()
print(f"Continuous action (thrust vector, N): {action_continuous.tolist()}")
```

The key insight: discrete action spaces require networks that output a categorical distribution (one logit per action); continuous action spaces require networks that output distribution parameters (mean and variance). This changes both the network architecture and the training algorithm.

For SSA, most telescope scheduling problems are naturally discrete (which satellite to observe). Maneuver planning problems are naturally continuous (what thrust to apply). Recognizing this early shapes your entire algorithm choice.

## Designing state representations

The state you give the agent is one of the most consequential design decisions in an RL problem. Get it wrong and no amount of algorithm sophistication will compensate. The state must satisfy the **Markov property**: the next state must be predictable (statistically) from the current state and action alone, without needing to know the history.

**The Markov property stated precisely**: the state \\(s_t\\) contains enough information about the history that \\(P(s_{t+1} \mid s_0, a_0, s_1, a_1, \ldots, s_t, a_t) = P(s_{t+1} \mid s_t, a_t)\\). The full history adds no predictive value beyond the current state.

Violating the Markov property does not cause the algorithm to crash — it causes the agent to learn a suboptimal policy because it cannot distinguish situations that look the same but have different futures.

### Three state representations for the telescope problem

Here are three candidate state representations for our 5-satellite telescope problem, with analysis of each.

**Representation A: Which satellite I observed last**

```python
# Bad Markovian design
state_A = {
    "last_observed_satellite": 3,  # just a single integer, 0-4
}
```

This is **not Markovian**. Knowing I observed satellite 3 last step tells me almost nothing about the current risk levels of all five satellites. The agent cannot tell if satellite 2 is about to have a conjunction event because that information is not in the state. The agent would need to remember the last 10 observations to have any useful context.

**Representation B: Observation timestamps**

```python
# Better: last observation timestamp for each satellite
state_B = {
    "last_obs_time_sat1": 2.0,   # hours ago
    "last_obs_time_sat2": 0.5,
    "last_obs_time_sat3": 6.0,
    "last_obs_time_sat4": 3.5,
    "last_obs_time_sat5": 1.0,
}
# 5-dimensional vector
```

This is **Markovian** — given this state and any action, the next state's observation timestamps are deterministic. But it omits something important: the current conjunction risk estimates. An agent using this state can try to keep all satellites observed recently, but cannot prioritize satellites with active conjunction events because that information is missing.

**Representation C: Full risk-aware state**

```python
# Best: observation recency + current conjunction risk estimates
state_C = {
    "hours_since_last_obs": [2.0, 0.5, 6.0, 3.5, 1.0],      # 5 values
    "conjunction_risk_score": [0.1, 0.05, 0.8, 0.2, 0.15],   # 5 values
    "hours_to_peak_risk":     [48.0, 100.0, 3.0, 24.0, 72.0],# 5 values
}
# 15-dimensional vector
```

This is **Markovian and informative**. The agent knows both how stale each observation is and how dangerous each satellite currently is. It can now develop a sensible policy: prioritize satellites with high conjunction risk AND stale observations.

```python
import numpy as np

# Illustrating the information difference
def can_agent_prioritize_risky_satellite(state, repr_type):
    """Can the agent tell that satellite 3 is in immediate danger?"""
    if repr_type == "A":
        # Only knows the last satellite observed — no risk information
        return False
    elif repr_type == "B":
        # Knows satellite 3 was observed 6 hours ago — helpful for staleness
        # but no direct risk score
        hours_since = state["last_obs_time_sat3"]
        # A long time since observation is correlated with risk, but indirect
        return hours_since > 5.0
    elif repr_type == "C":
        # Direct risk score available
        return state["conjunction_risk_score"][2] > 0.5

state_B_example = {"last_obs_time_sat3": 6.0}
state_C_example = {
    "hours_since_last_obs": [2.0, 0.5, 6.0, 3.5, 1.0],
    "conjunction_risk_score": [0.1, 0.05, 0.8, 0.2, 0.15],
    "hours_to_peak_risk": [48.0, 100.0, 3.0, 24.0, 72.0],
}

print(f"Repr A can prioritize: {can_agent_prioritize_risky_satellite(None, 'A')}")
print(f"Repr B can prioritize: {can_agent_prioritize_risky_satellite(state_B_example, 'B')}")
print(f"Repr C can prioritize: {can_agent_prioritize_risky_satellite(state_C_example, 'C')}")
# False, True (by proxy), True (directly)
```

The tradeoff between B and C is also real: C requires computing conjunction risk estimates as an input to the RL policy. This adds complexity and a dependency on an upstream estimator. If that estimator is wrong, the RL agent's decisions will be wrong too. Representation B is simpler and more robust to upstream errors but limits the agent's reasoning. The right choice depends on how reliable your upstream conjunction risk estimates are.

### 3. Transition function (P)

The **transition function** describes how the state changes when an action is taken. Specifically:

\\[ P(s' \mid s, a) \\]

is the probability of ending up in state \\(s'\\) after taking action \\(a\\) in state \\(s\\).

**Decoding:**
- \\(s\\) is the current state
- \\(a\\) is the action you took
- \\(s'\\) is the next state (the prime mark indicates "next")
- \\(P(s' \mid s, a)\\) reads as "probability of \\(s'\\) given \\(s\\) and \\(a\\)"

For our telescope problem, the transition is partly stochastic (random) and partly deterministic:
- Time advances by 1 hour deterministically (so all `time_to_next_conjunction` values decrease by 1)
- The satellite you observed has its `hours_since_last_observation` reset to 0; others increment by 1
- Random new conjunction events may occur (this is the stochastic part)

**The Markov property** is what gives MDPs their name. It says: the next state depends only on the current state and action, not on the history of how you got to the current state. If you know the current state, you know everything relevant for predicting the future.

In practice, the Markov property is satisfied or approximately satisfied by careful state design. If your "state" does not contain enough information to predict the future, you have not really captured the state, and you should add more features.

### 4. Reward function (R)

The **reward function** describes the immediate feedback the agent receives:

\\[ R(s, a, s') \\]

is the reward received when taking action \\(a\\) in state \\(s\\) and transitioning to state \\(s'\\).

For our telescope problem:
- +10 reward if the satellite you observed turned out to have a conjunction event
- +1 reward for a routine observation
- 0 reward for any other satellite (no penalty, just no reward for not observing them)

The reward function encodes what we want the agent to do. Tweaking the reward function changes the agent's incentives. This is called **reward shaping**, and it is both very useful and very dangerous: getting the rewards subtly wrong can lead the agent to find unexpected and undesired behaviors.

For example, if you penalized "not observing satellite 1 for more than 5 hours" too heavily, the agent might develop a rigid rotation pattern that ignored real-time conjunction priority. Reward design matters.

## Reward shaping pitfalls: when incentives backfire

Reward shaping is the most common source of policy failures in real RL applications. The agent does not care about your intent — it cares about the number it receives. If your reward function is slightly misaligned with your actual goal, the agent will find ways to maximize the number that you did not anticipate.

**Reward hacking** is the term for this failure mode. The agent finds an unexpected behavior that maximizes the reward signal but does not match the intended goal.

### The observation-count trap

Suppose you reward your telescope agent for the number of satellites successfully observed per day:

```python
def bad_reward(satellite_observed, had_conjunction_event):
    """Rewards based on observation count — a recipe for reward hacking."""
    return 1.0  # +1 for every observation, regardless of risk
```

The agent will quickly learn to spend all its time observing the satellites that are easiest to confirm — calm, low-risk satellites where observations are quick and certain. A satellite on the verge of a high-risk conjunction event might have uncertain, ambiguous observations that the agent has learned to avoid. The agent is maximizing observations, but you wanted it to maximize detection of dangerous events.

### Sparse vs. dense rewards

Another common failure: rewards that are too sparse.

A **sparse reward** only gives feedback at key moments:
```python
def sparse_reward(observed_conjunction):
    """Only +10 if a conjunction is actually caught; 0 otherwise."""
    return 10.0 if observed_conjunction else 0.0
```

This is hard to learn from because the agent takes many steps between positive rewards. If the agent is taking random actions and conjunctions are rare, it might go 1,000 steps with zero reward. Gradient descent has no signal about what to improve.

A **dense reward** provides feedback at every step:
```python
def dense_reward(satellite_risk, hours_since_last_obs):
    """Reward is highest for observing high-risk satellites that haven't been seen recently."""
    recency_bonus = min(hours_since_last_obs / 4.0, 1.0)  # bonus for stale observations
    risk_bonus = satellite_risk                             # bonus for high-risk satellites
    return 0.5 * recency_bonus + 0.5 * risk_bonus
```

Dense rewards guide learning faster, but they introduce the shaping problem: the agent may optimize the proxy signal rather than the true goal.

### A safer design: shaped rewards with a true goal backup

```python
import numpy as np

def shaped_telescope_reward(
    satellite_idx,
    satellite_risk_scores,  # [0,1] for each of 5 satellites
    hours_since_last_obs,   # hours for each satellite
    caught_conjunction,     # bool: did we catch a real event?
):
    """
    Dense shaping reward that guides learning, anchored by a true goal signal.
    
    The key safety property: the shaping terms can only add bounded signal.
    The large bonus for catching a real event keeps the agent focused on
    the true objective even if the shaping terms pull slightly wrong.
    """
    # True goal: big reward for catching conjunction events
    true_goal_reward = 20.0 if caught_conjunction else 0.0
    
    # Shaping 1: prefer observing high-risk satellites
    risk = satellite_risk_scores[satellite_idx]
    risk_bonus = 2.0 * risk  # max +2 for risk=1.0
    
    # Shaping 2: prefer observing satellites we haven't seen recently
    hours = hours_since_last_obs[satellite_idx]
    staleness_bonus = min(hours / 6.0, 1.0)  # caps at +1 after 6 hours
    
    total = true_goal_reward + risk_bonus + staleness_bonus
    return total

# Example: observe satellite 2, which is high-risk and hasn't been seen in 8 hours
risk_scores = [0.05, 0.1, 0.85, 0.2, 0.3]
hours_stale  = [1.0, 0.5, 8.0, 3.0, 2.0]

reward_good = shaped_telescope_reward(
    satellite_idx=2,
    satellite_risk_scores=risk_scores,
    hours_since_last_obs=hours_stale,
    caught_conjunction=True,
)
print(f"Observing high-risk sat 2 and catching conjunction: {reward_good:.2f}")
# 20.0 (true goal) + 1.7 (risk) + 1.0 (staleness) = 22.70

reward_safe = shaped_telescope_reward(
    satellite_idx=0,
    satellite_risk_scores=risk_scores,
    hours_since_last_obs=hours_stale,
    caught_conjunction=False,
)
print(f"Observing low-risk sat 0 (routine): {reward_safe:.2f}")
# 0.0 (no conjunction) + 0.1 (low risk) + 0.17 (not stale) = 0.27
```

The design principle: the shaping terms (risk bonus, staleness bonus) provide dense guidance, but their magnitude is small compared to the true goal signal (20.0 for a caught conjunction). The agent has strong incentive to pursue the real objective, and the shaping terms steer it toward productive exploration without dominating its behavior.

Reward design is as much art as science. The telescope reward above still has failure modes — for example, an agent that learns to declare every observation a "conjunction event" via some upstream classification manipulation. Good reward design requires thinking adversarially: assume the agent will find every loophole in your specification, and close them before they are found in production.

### 5. Discount factor (γ)

The **discount factor** \\(\gamma\\) (Greek lowercase gamma) is a number between 0 and 1 that says how much to value future rewards versus immediate rewards.

A reward received \\(t\\) steps in the future is worth \\(\gamma^t\\) times as much as an immediate reward.

| t (steps in future) | γ = 0.9 | γ = 0.99 | γ = 1.0 |
|---------------------|---------|----------|---------|
| 0 (immediate)       | 1.000   | 1.000    | 1.000   |
| 1                   | 0.900   | 0.990    | 1.000   |
| 5                   | 0.590   | 0.951    | 1.000   |
| 10                  | 0.349   | 0.904    | 1.000   |
| 100                 | 0.000027 | 0.366   | 1.000   |

With γ = 0.9, a reward 100 steps away is worth essentially nothing. With γ = 0.99, it is worth about 37% of an immediate reward. With γ = 1.0, future rewards are worth as much as immediate ones.

**Why discount?** Three reasons:
1. **Mathematical convenience**: γ < 1 ensures that the total reward over an infinite horizon is finite, even if the agent runs forever.
2. **Modeling uncertainty about the future**: rewards far in the future may not actually happen (the system might end, the environment might change). Discounting reflects this uncertainty.
3. **Encouraging timely action**: the agent prefers to get rewards sooner rather than later.

For our telescope problem, γ = 0.95 or so would be reasonable. A conjunction event 10 hours from now is still important, but slightly less urgent than one happening immediately.

## The agent-environment loop

Putting it all together, an MDP describes the following loop:

```
1. The agent observes the current state s_t
2. The agent selects an action a_t (using its policy)
3. The environment:
   a. Computes the next state s_{t+1} according to P(s_{t+1} | s_t, a_t)
   b. Gives the agent a reward r_t = R(s_t, a_t, s_{t+1})
4. Time advances: t -> t+1
5. Repeat until the episode ends (or forever, if there is no end)
```

The subscript t (often called a "timestep") indexes time. \\(s_t\\) is the state at time t. \\(a_t\\) is the action chosen at time t. \\(r_t\\) is the reward received at time t.

## What the agent is trying to do

The agent's goal is to maximize the expected sum of discounted future rewards:

\\[ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\]

This sum, called the **return** (or the **discounted return** or **cumulative reward**), is what the agent ultimately cares about.

**Decoding:**
- \\(G_t\\): the return starting from time t (G is conventional notation for "gain")
- \\(R_{t+1}\\): the reward received at time t+1 (immediate reward from the action taken at time t)
- \\(\gamma R_{t+2}\\): the reward at time t+2, discounted by one step
- \\(\gamma^2 R_{t+3}\\): the reward at time t+3, discounted by two steps
- And so on

In compact form:

\\[ G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \\]

The agent wants to choose actions that maximize the expected value of \\(G_t\\). Note the word "expected": the future is uncertain (because of the stochastic transitions and possibly stochastic rewards), so we are talking about the expectation (lesson 1 of Module 1) over all possible futures.

## Policies

A **policy** is the agent's strategy: a rule for selecting actions given states. It is denoted \\(\pi\\) (Greek lowercase pi).

A policy can be:
- **Deterministic**: \\(\pi(s) = a\\) means "in state s, always take action a"
- **Stochastic**: \\(\pi(a \mid s)\\) is the probability of taking action a in state s

Almost everything in this curriculum uses stochastic policies, because they:
- Naturally support exploration (trying actions to learn about them)
- Are needed for game theory (in mixed-strategy equilibria)
- Are needed for partial observability (sometimes randomization is genuinely the best strategy)

The notation \\(\pi(a \mid s)\\) is conditional probability notation from Module 1, lesson 2: probability of \\(a\\) given \\(s\\). A policy is just a probability distribution over actions, conditional on the state.

## A simple worked example

Let us hand-trace one episode of a tiny MDP to make all this concrete.

**The MDP**: A 2-state, 2-action MDP.

States: \\(s_0\\), \\(s_1\\)
Actions: \\(a_0\\), \\(a_1\\)
Transitions:
- From \\(s_0\\) taking \\(a_0\\): go to \\(s_0\\) with prob 0.8, \\(s_1\\) with prob 0.2
- From \\(s_0\\) taking \\(a_1\\): go to \\(s_1\\) with prob 1.0
- From \\(s_1\\) taking \\(a_0\\): go to \\(s_0\\) with prob 0.5, \\(s_1\\) with prob 0.5
- From \\(s_1\\) taking \\(a_1\\): go to \\(s_1\\) with prob 1.0

Rewards:
- Reward in \\(s_0\\): 1.0
- Reward in \\(s_1\\): 5.0

Discount: γ = 0.9

**Policy**: Always take action \\(a_1\\) (a deterministic policy).

**Sample episode**:
- Start in \\(s_0\\). Reward at time 0: 1.0. (Sometimes we do not collect reward at the start; this varies by convention. Let us say we get the reward for being in the state.)
- Take \\(a_1\\). Transition to \\(s_1\\) with probability 1. New state: \\(s_1\\). Reward at time 1: 5.0.
- Take \\(a_1\\). Transition to \\(s_1\\). Reward at time 2: 5.0.
- Take \\(a_1\\). Transition to \\(s_1\\). Reward at time 3: 5.0.
- ... and so on. The agent stays in \\(s_1\\) forever.

**Return from time 0**:
\\[ G_0 = 1.0 + 0.9 \cdot 5.0 + 0.9^2 \cdot 5.0 + 0.9^3 \cdot 5.0 + \ldots \\]
\\[ = 1.0 + 5.0 \cdot (0.9 + 0.81 + 0.729 + \ldots) \\]
\\[ = 1.0 + 5.0 \cdot \frac{0.9}{1 - 0.9} \\]
\\[ = 1.0 + 5.0 \cdot 9 = 1.0 + 45.0 = 46.0 \\]

(The infinite sum \\(0.9 + 0.81 + 0.729 + \ldots\\) is a geometric series with sum \\(\frac{0.9}{1 - 0.9} = 9\\). You do not need to derive this; just take it on faith.)

So under this policy, starting from \\(s_0\\), the agent expects total discounted rewards of 46.0.

## Code: representing this MDP in Python

```python
import numpy as np

# State and action indices
S0, S1 = 0, 1
A0, A1 = 0, 1

# Transition probabilities: P[s, a, s'] = P(s' | s, a)
# Shape: (num_states, num_actions, num_states)
P = np.zeros((2, 2, 2))
P[S0, A0, S0] = 0.8
P[S0, A0, S1] = 0.2
P[S0, A1, S1] = 1.0
P[S1, A0, S0] = 0.5
P[S1, A0, S1] = 0.5
P[S1, A1, S1] = 1.0

# Rewards (here, just by state)
R = np.array([1.0, 5.0])  # R[S0] = 1.0, R[S1] = 5.0

# Discount factor
gamma = 0.9

# Simulate one episode under the policy "always take A1"
def simulate_episode(start_state, policy_action, num_steps=100):
    state = start_state
    total_return = 0.0
    discount = 1.0
    
    for t in range(num_steps):
        reward = R[state]
        total_return += discount * reward
        
        action = policy_action  # always take this action
        # Sample next state from the transition probabilities
        next_state = np.random.choice(2, p=P[state, action])
        
        state = next_state
        discount *= gamma
    
    return total_return

np.random.seed(0)
returns = [simulate_episode(S0, A1) for _ in range(1000)]
print(f"Average return starting from S0: {np.mean(returns):.2f}")
# Should be close to 46.0
```

## Why all this matters

Every RL algorithm we will see asks one of two questions:
1. **Value-based**: "for each state (or state-action pair), what is the expected return under some policy?" This gives us V(s) or Q(s, a).
2. **Policy-based**: "what policy maximizes the expected return?"

Both questions are framed in terms of the MDP we just defined. The states, actions, rewards, transitions, and discount factor all show up in the algorithms. Without the MDP framework, we could not even state precisely what the agent is trying to do.

The next lesson introduces value functions, the central mathematical object for the value-based approach.

## Key Takeaways

- **An MDP is a formal description of a sequential decision problem.** Its five components — state, action, transition, reward, discount — must all be specified before any RL algorithm can be applied. Informal problem descriptions do not suffice; the formalization forces you to be precise about what the agent observes, what it can do, and what it is trying to maximize.
- **The Markov property is a design constraint, not a free assumption.** If your state representation does not capture enough information to predict the next state, you have violated the Markov property and your algorithms will underperform. Good state design means asking: "does this state tell the agent everything relevant about the past?"
- **Discrete and continuous action spaces require different algorithms.** Q-learning and DQN are for discrete actions. Policy gradient methods handle both. Discretizing a continuous action space loses resolution; use continuous methods when resolution matters (maneuver planning, pointing precision).
- **Reward hacking is the primary failure mode in practice.** Agents do not pursue your intent — they maximize the number. Every reward function has loopholes; think adversarially about what an agent optimizing that signal might do that you would not want.
- **Dense rewards guide learning but risk shaping failure; sparse rewards are honest but slow.** The practical compromise: use a dense shaping signal with small magnitude, anchored by a large true-goal reward that prevents the agent from ignoring the actual objective.
- **State representation engineering is often more valuable than algorithm choice.** A good state representation with a simple algorithm often outperforms a poor state representation with a sophisticated one. Invest time in deciding what information to include before tuning hyperparameters.

## Quiz

{{#quiz 01-markov-decision-processes.toml}}
