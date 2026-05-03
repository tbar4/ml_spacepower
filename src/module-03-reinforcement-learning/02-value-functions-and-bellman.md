# Lesson 2: Value Functions and Bellman Equations

## Where this fits

The MDP framework from lesson 1 lets us describe a sequential decision problem. But describing it does not tell us how to solve it. Value functions are the central mathematical object that makes solving possible: they assign a number to each state (or each state-action pair) that captures "how good is it to be here?" The Bellman equations express value functions recursively (a state's value relates to the values of its successor states), and that recursion is the engine that drives Q-learning, DQN, MCTS, AlphaZero, and essentially every other algorithm in this curriculum. This lesson builds the central machinery.

## What is a value function?

A **value function** answers the question: "starting from this state and following some policy, what is the expected total discounted return?"

Two flavors:

**State-value function V(s)**: how good is state s, assuming I follow my current policy from now on?

**Action-value function Q(s, a)**: how good is taking action a in state s, assuming I follow my current policy from then on?

Both are expectations over all the randomness in the environment and the policy: random transitions, random reward outcomes, and random action selections (if the policy is stochastic). The expectation collapses all that uncertainty into a single number.

## Why two value functions?

V(s) tells you how good a state is, on average, under your current policy. That is useful for evaluating the policy.

Q(s, a) tells you how good each action is in state s, on average, under your current policy from then on. That is more useful for **choosing** actions: just pick the action with the highest Q value.

In a sense, Q is a finer-grained version of V. If you know all the Q values in a state, you can compute V by averaging over the actions according to the policy:

\\[ V^\pi(s) = \sum_a \pi(a \mid s) \cdot Q^\pi(s, a) \\]

**Decoding:**
- \\(V^\pi(s)\\): the value of state s under policy π. The superscript π says "this value depends on which policy you are following."
- \\(\pi(a \mid s)\\): the probability of taking action a in state s under policy π
- \\(Q^\pi(s, a)\\): the Q value of taking action a in state s, then following π

This is just the expectation formula from Module 1, lesson 1. V is the expected Q over the policy's action distribution.

## Building intuition with a tiny example

Let us use a small concrete MDP. Two satellites, S1 and S2. Each turn, you observe one. The state describes the current "alert level" of each satellite: 0 (calm) or 1 (alert).

Possible states: (calm, calm), (calm, alert), (alert, calm), (alert, alert). Four states total.

Actions: observe S1 or observe S2.

Rewards: +5 for observing an alert satellite, +1 for observing a calm one. Then both satellites' alert levels evolve: observed satellite resets to calm; unobserved satellite has 30% chance of becoming alert (or staying alert).

For now, let us use a simple uniform random policy: 50% chance of either action in any state.

**Computing V((alert, calm)) by Monte Carlo simulation**

Start in state (alert, calm). Run 10,000 simulated episodes of, say, 50 steps each, following the random policy. Average the discounted returns.

```python
import numpy as np

np.random.seed(0)

def step(state, action):
    """Apply one transition. state is (s1, s2) tuple, action is 0 (obs S1) or 1 (obs S2)."""
    s1, s2 = state
    
    if action == 0:  # observe S1
        reward = 5 if s1 == 1 else 1
        new_s1 = 0  # observed: reset to calm
        # S2 evolves randomly
        if s2 == 0:
            new_s2 = 1 if np.random.rand() < 0.3 else 0
        else:
            new_s2 = 1 if np.random.rand() < 0.7 else 0  # tends to stay alert
    else:  # observe S2
        reward = 5 if s2 == 1 else 1
        new_s2 = 0
        if s1 == 0:
            new_s1 = 1 if np.random.rand() < 0.3 else 0
        else:
            new_s1 = 1 if np.random.rand() < 0.7 else 0
    
    return (new_s1, new_s2), reward

def estimate_V(start_state, num_episodes=10_000, num_steps=50, gamma=0.9):
    returns = []
    for _ in range(num_episodes):
        state = start_state
        total_return = 0
        discount = 1.0
        for _ in range(num_steps):
            action = np.random.choice([0, 1])  # uniform random policy
            state, reward = step(state, action)
            total_return += discount * reward
            discount *= gamma
        returns.append(total_return)
    return np.mean(returns)

print(f"V((alert, calm)) under random policy: {estimate_V((1, 0)):.2f}")
print(f"V((calm, calm))  under random policy: {estimate_V((0, 0)):.2f}")
print(f"V((alert, alert)) under random policy: {estimate_V((1, 1)):.2f}")
```

You will get values around 25-30, depending on the starting state. (alert, alert) should have the highest value because there are more opportunities for the +5 reward.

This is the brute-force way to compute V: simulate many episodes and average. It works but is wasteful: every state requires its own batch of simulations. Bellman's insight is that we can do much better by exploiting recursion.

## The Bellman equation: value functions are self-referential

Here is the key observation. The value of a state can be decomposed into two parts:
1. The immediate reward from acting in this state
2. The discounted value of wherever you end up next

Formally, for the state-value function:

\\[ V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right] \\]

This is dense. Let us decode it carefully.

**\\(V^\pi(s)\\)**: the value of state s under policy π.

**\\(\sum_a \pi(a \mid s)\\)**: average over actions, weighted by the policy's probability of taking each one. This is the "expected over actions" piece.

**\\(\sum_{s'} P(s' \mid s, a)\\)**: average over possible next states, weighted by the transition probability. This is the "expected over next states" piece.

**\\(R(s, a, s') + \gamma V^\pi(s')\\)**: the contribution from each (action, next-state) combination. The immediate reward, plus the discounted value of the state we end up in.

**Reading in plain English**:

> "The value of state s is the average (over actions you might take and states you might transition to) of (the immediate reward, plus the discounted value of the next state)."

The recursion is: V appears on both sides. The value of s depends on the values of states reachable from s. Those values in turn depend on the values of states reachable from them. And so on.

For the Q function, the Bellman equation is similar but slightly different:

\\[ Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') \cdot Q^\pi(s', a') \right] \\]

Reading: "the Q value of (s, a) is the expected reward plus the discounted expected Q value of (s', a'), where a' is sampled from the policy."

## Solving the Bellman equation: iterative computation

The Bellman equation is a self-consistency condition: a true value function must satisfy it. We can use this to compute V iteratively:

1. Initialize V(s) = 0 for all states.
2. For each state, update V(s) using the Bellman equation, treating the current V values as inputs.
3. Repeat until V stops changing.

This is called **value iteration** (technically, "iterative policy evaluation" when the policy is fixed). It converges to the true V values.

For our 4-state MDP, let us solve it analytically. We have four states; we get four equations (one per state); we solve the system.

This is too tedious to do by hand. Let us do it iteratively in code:

```python
import numpy as np

# State enumeration: 0=(calm,calm), 1=(calm,alert), 2=(alert,calm), 3=(alert,alert)
states = [(0,0), (0,1), (1,0), (1,1)]
gamma = 0.9

def transitions(state, action):
    """Return list of (next_state_index, probability, reward)."""
    s1, s2 = state
    out = []
    if action == 0:  # observe S1
        reward = 5 if s1 == 1 else 1
        new_s1 = 0
        # Next S2 distribution
        if s2 == 0:
            for new_s2, p in [(0, 0.7), (1, 0.3)]:
                out.append((states.index((new_s1, new_s2)), p, reward))
        else:
            for new_s2, p in [(0, 0.3), (1, 0.7)]:
                out.append((states.index((new_s1, new_s2)), p, reward))
    else:  # observe S2
        reward = 5 if s2 == 1 else 1
        new_s2 = 0
        if s1 == 0:
            for new_s1, p in [(0, 0.7), (1, 0.3)]:
                out.append((states.index((new_s1, new_s2)), p, reward))
        else:
            for new_s1, p in [(0, 0.3), (1, 0.7)]:
                out.append((states.index((new_s1, new_s2)), p, reward))
    return out

# Iterative policy evaluation: random policy (50/50)
V = np.zeros(4)
for iteration in range(200):
    V_new = np.zeros(4)
    for s_idx in range(4):
        # For each action, compute expected (reward + γ * V[s'])
        for action in [0, 1]:
            policy_prob = 0.5  # uniform random
            for next_s_idx, prob, reward in transitions(states[s_idx], action):
                V_new[s_idx] += policy_prob * prob * (reward + gamma * V[next_s_idx])
    if np.max(np.abs(V_new - V)) < 1e-6:
        print(f"Converged after {iteration} iterations.")
        break
    V = V_new

for i, s in enumerate(states):
    print(f"V({s}) = {V[i]:.2f}")
```

This converges quickly (within ~100 iterations) to values that match what the Monte Carlo simulation produced (within sampling noise).

The advantage of value iteration over Monte Carlo: it is exact (in the limit of convergence), uses no random samples, and is computationally cheap when the state space is small. The disadvantage: it requires you to know the transition probabilities P(s' | s, a) and reward function R(s, a, s'). In real problems (Atari, chess, real SSA), you usually do not have explicit access to these, so you cannot do straight value iteration. That is what Q-learning fixes (next lesson).

## The optimal value function

So far we have talked about the value function for a specific policy: V^π and Q^π. But often we want the value of the **best possible** policy. That is the **optimal value function**, denoted V* and Q*:

\\[ V^*(s) = \max_\pi V^\pi(s) \\]

In words: the maximum value achievable in state s by any policy.

For Q*:

\\[ Q^*(s, a) = \max_\pi Q^\pi(s, a) \\]

The corresponding Bellman equations look slightly different. The optimal policy in any state takes the action that maximizes Q*, so the "average over actions" gets replaced by "max over actions":

\\[ V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma V^*(s')] \\]

\\[ Q^*(s, a) = \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma \max_{a'} Q^*(s', a')] \\]

These are called the **Bellman optimality equations**. They define the optimal value functions self-referentially. Solving them gives you the optimal policy automatically: in each state, take the action that maximizes Q*.

## Why this all matters

The Bellman equation for Q* is the foundation of Q-learning. Q-learning is essentially: "use experience to estimate Q* by enforcing the Bellman optimality equation for the samples we have seen." We will see this concretely in the next lesson.

The Bellman equation for V is the foundation of policy evaluation methods, which appear inside actor-critic algorithms (lesson 6) and elsewhere.

The recursion idea (a state's value depends on the values of its successors) shows up in MCTS (Module 4), where we estimate values by recursively averaging over rollouts. It shows up in CFR (Module 5), where regret values are propagated through the game tree.

If you take one thing from this lesson: **value functions describe what the agent thinks the future is worth from each state, and the Bellman equation lets us compute these values using only local information about transitions and rewards.**

## Quiz

{{#quiz 02-value-functions-and-bellman.toml}}
