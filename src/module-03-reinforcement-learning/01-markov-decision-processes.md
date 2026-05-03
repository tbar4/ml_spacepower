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

## Quiz

{{#quiz 01-markov-decision-processes.toml}}
