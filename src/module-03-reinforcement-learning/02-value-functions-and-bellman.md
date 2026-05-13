# Lesson 2: Value Functions and Bellman Equations


<!-- toc -->

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

\[ V^\pi(s) = \sum_a \pi(a \mid s) \cdot Q^\pi(s, a) \]

**Decoding:**
- \(V^\pi(s)\): the value of state s under policy π. The superscript π says "this value depends on which policy you are following."
- \(\pi(a \mid s)\): the probability of taking action a in state s under policy π
- \(Q^\pi(s, a)\): the Q value of taking action a in state s, then following π

This is just the expectation formula from Module 1, lesson 1. V is the expected Q over the policy's action distribution.

## The value hierarchy

Before going further, here is a reference table for all the value-related concepts you will encounter in this curriculum. They are easy to confuse; keep this table in mind.

| Concept | Symbol | What it means | Used in |
|---------|--------|---------------|---------|
| Policy value | \(V^\pi(s)\) | Expected return following π from state s | Policy evaluation, Actor-Critic |
| Optimal value | \(V^*(s)\) | Best possible expected return from state s, over all policies | Value iteration |
| Action-value | \(Q^\pi(s, a)\) | Expected return taking action a then following π | Q-learning, DQN |
| Optimal Q | \(Q^*(s, a)\) | Best possible Q value for (s, a) | Q-learning target |
| Advantage | \(A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)\) | How much better is action a compared to the average action in state s | A3C, PPO |

The advantage function \(A^\pi(s, a)\) deserves a note. It answers: "if I take action a instead of whatever my policy normally does, how much better or worse will I do?" A positive advantage means action a is better than average; a negative advantage means it is worse. PPO and A3C use the advantage rather than Q directly because it has lower variance — subtracting the baseline V(s) cancels out the part of the return that has nothing to do with the specific action choice.

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

\[ V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right] \]

This is dense. Let us decode it carefully.

**\(V^\pi(s)\)**: the value of state s under policy π.

**\(\sum_a \pi(a \mid s)\)**: average over actions, weighted by the policy's probability of taking each one. This is the "expected over actions" piece.

**\(\sum_{s'} P(s' \mid s, a)\)**: average over possible next states, weighted by the transition probability. This is the "expected over next states" piece.

**\(R(s, a, s') + \gamma V^\pi(s')\)**: the contribution from each (action, next-state) combination. The immediate reward, plus the discounted value of the state we end up in.

**Reading in plain English**:

> "The value of state s is the average (over actions you might take and states you might transition to) of (the immediate reward, plus the discounted value of the next state)."

The recursion is: V appears on both sides. The value of s depends on the values of states reachable from s. Those values in turn depend on the values of states reachable from them. And so on.

For the Q function, the Bellman equation is similar but slightly different:

\[ Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') \cdot Q^\pi(s', a') \right] \]

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

## Bellman error as training signal

The iterative value computation above is conceptually clean, but how does it connect to machine learning with neural networks? The connection runs through **Bellman error** (also called **TD error** — temporal difference error).

The Bellman equation tells us exactly what V(s) should equal:

\[ V^\pi(s) = \mathbb{E}\left[ R(s, a, s') + \gamma V^\pi(s') \right] \]

If our current estimate of V is not correct, the two sides will not match. The **TD error** \(\delta\) measures this mismatch for a single observed transition \((s, a, r, s')\):

\[ \delta = r + \gamma V(s') - V(s) \]

**Decoding each piece:**
- \(r + \gamma V(s')\): the **TD target** — what V(s) should be, based on the actual reward we received and our current estimate of the next state's value
- \(V(s)\): our current estimate of V(s)
- \(\delta = \text{target} - \text{estimate}\): how wrong we are

The sign of δ tells us which direction to update:

- **Positive δ** (δ > 0): the TD target exceeds our estimate. We underestimated V(s) — the state turned out better than we thought. Increase V(s) toward the target.
- **Negative δ** (δ < 0): the TD target is below our estimate. We overestimated V(s) — the state was worse than we expected. Decrease V(s) toward the target.
- **δ = 0**: our estimate is consistent with the observed transition. No update needed.

```python
import numpy as np

# Use the 4-state satellite MDP from above.
# Suppose V is our current (imperfect) estimate.
V_estimate = np.array([20.0, 24.0, 23.0, 28.0])  # initial guess
gamma = 0.9

# Simulate one step and compute TD error
state = (1, 0)   # (calm, alert) -- index 2
s_idx = states.index(state)

action = 0  # observe S1
np.random.seed(42)
possible_transitions = transitions(state, action)
# Sample a next state according to probabilities
probs = [t[1] for t in possible_transitions]
chosen_idx = np.random.choice(len(possible_transitions), p=probs)
next_s_idx, _, reward = possible_transitions[chosen_idx]

# Compute TD error
td_target = reward + gamma * V_estimate[next_s_idx]
td_error = td_target - V_estimate[s_idx]

print(f"State: {state} (idx {s_idx}), V estimate: {V_estimate[s_idx]:.2f}")
print(f"Action: observe S1, Reward: {reward}")
print(f"Next state idx: {next_s_idx}, V(s') estimate: {V_estimate[next_s_idx]:.2f}")
print(f"TD target: {td_target:.2f}")
print(f"TD error δ: {td_error:.2f}")

if td_error > 0:
    print("We underestimated this state — update V upward.")
elif td_error < 0:
    print("We overestimated this state — update V downward.")
else:
    print("Our estimate is consistent with this transition.")

# A simple TD update: move V(s) toward the target
learning_rate = 0.1
V_estimate[s_idx] += learning_rate * td_error
print(f"Updated V({state}): {V_estimate[s_idx]:.2f}")
```

```rust
fn main() {
    // Current V estimates: (calm,calm)=0, (calm,alert)=1, (alert,calm)=2, (alert,alert)=3
    let mut v = [20.0, 24.0, 23.0, 28.0_f64];
    let gamma = 0.9_f64;

    // Example transition: state 2 (alert,calm), observe S1 → reward=1, next state=0
    let s_idx      = 2_usize;
    let reward     = 1.0_f64;
    let next_s_idx = 0_usize;

    let td_target = reward + gamma * v[next_s_idx];
    let td_error  = td_target - v[s_idx];

    println!("V estimate state 2: {:.2}", v[s_idx]);
    println!("TD target:          {:.2}", td_target);
    println!("TD error δ:         {:.2}", td_error);

    if td_error > 0.0 {
        println!("Underestimated — update V upward.");
    } else {
        println!("Overestimated — update V downward.");
    }

    let lr = 0.1_f64;
    v[s_idx] += lr * td_error;
    println!("Updated V(state 2): {:.2}", v[s_idx]);
    // td_error < 0: td_target (1 + 0.9*20 = 19.0) < V(2) (23.0) → move downward
}
```

This is the core of Q-learning and DQN: use the Bellman equation to generate training targets, compute the error between our current estimate and those targets, and update our estimate in the direction that reduces the error. In DQN, the "estimate" is a neural network, and the update is a gradient descent step. The machinery is more complex, but the idea is exactly this TD error computation.

## Bootstrapping: using our own estimates to update our estimates

The TD update above uses \(V(s')\) — our current estimate of the next state's value — to update \(V(s)\). This is called **bootstrapping**: we use our own possibly-incorrect estimates to generate new estimates.

This is philosophically strange. If our estimates are wrong, won't updating with wrong estimates just produce more wrong estimates? Yes — but the key insight is that bootstrapped estimates converge because the Bellman equation is a contraction. Each iteration brings us closer to the true values. The process is self-correcting over many updates.

### The alternative: Monte Carlo

The alternative to bootstrapping is **Monte Carlo estimation**: run an episode to completion, observe the actual total return \(G_t = R_{t+1} + \gamma R_{t+2} + \ldots\), and use that as the update target.

\[ V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t - V(s_t) \right] \]

Monte Carlo does not use \(V(s')\) at all — the target is the actual return, which involves no estimates.

### The bias-variance tradeoff

| Method | Bias | Variance | Data efficiency |
|--------|------|----------|-----------------|
| TD (1-step bootstrapping) | Biased (uses V(s'), which may be wrong) | Low (one step of randomness) | High (updates at every step) |
| Monte Carlo | Unbiased (uses actual return) | High (entire episode of randomness) | Low (waits for episode to end) |

**Bias** here means: is the training target systematically wrong? TD targets can be — if V(s') is wrong, the TD target is wrong. Monte Carlo targets are unbiased because actual returns are the ground truth.

**Variance** means: how much does the training target fluctuate between runs? Monte Carlo returns accumulate randomness over the entire episode (each random transition adds noise). TD targets only accumulate one step of randomness.

In SSA problems, episodes can be long (a telescope tracking problem might run for days with hundreds of timesteps). Monte Carlo would require waiting for a day-long episode to end before making any updates. TD methods update after every observation and are therefore far more practical.

### n-step returns: the middle ground

The \(n\)-step return is a principled interpolation between 1-step TD and Monte Carlo:

\[ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n V(s_{t+n}) \]

**Decoding:** Take the actual rewards for the next \(n\) steps, then bootstrap with V at step \(n\). Setting \(n=1\) gives standard TD. Setting \(n=T\) (the episode length) gives Monte Carlo.

Higher n reduces bias (we rely on fewer bootstrapped estimates) but increases variance (more actual random returns are included). The optimal n depends on the problem and is often treated as a hyperparameter. PPO and A3C typically use n between 5 and 20.

```python
import numpy as np

def n_step_return(rewards, V_final, gamma, n):
    """
    Compute the n-step return from a sequence of rewards.
    rewards: list of actual rewards [r_1, r_2, ..., r_n]
    V_final: V(s_{t+n}), the bootstrapped value at the end
    """
    G = V_final
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# Example: 3-step return for the satellite MDP
# Rewards observed: 1, 5, 1 (3 steps), then bootstrap with V(s_{t+3})
rewards_observed = [1.0, 5.0, 1.0]
V_bootstrap = 25.0  # our estimate of V at the state 3 steps out
gamma = 0.9

G_3step = n_step_return(rewards_observed, V_bootstrap, gamma, n=3)
print(f"3-step return: {G_3step:.2f}")
# = 1 + 0.9*(5 + 0.9*(1 + 0.9*25)) = 1 + 0.9*(5 + 0.9*23.5)
#                                    = 1 + 0.9*26.15 = 1 + 23.54 = 24.54

# Compare: 1-step TD target (immediate + bootstrap)
G_1step = rewards_observed[0] + gamma * V_bootstrap
print(f"1-step TD target: {G_1step:.2f}")
# = 1 + 0.9 * 25 = 23.5

# The 3-step return uses more actual data and less of our (potentially wrong) estimate.
```

```rust
fn n_step_return(rewards: &[f64], v_final: f64, gamma: f64) -> f64 {
    // Work backwards: G = r_n + γ*(r_{n-1} + γ*(...))
    rewards.iter().rev().fold(v_final, |g, &r| r + gamma * g)
}

fn main() {
    let rewards    = [1.0, 5.0, 1.0_f64];
    let v_bootstrap = 25.0_f64;
    let gamma       = 0.9_f64;

    let g3 = n_step_return(&rewards, v_bootstrap, gamma);
    println!("3-step return: {:.2}", g3); // 24.54

    let g1 = rewards[0] + gamma * v_bootstrap;
    println!("1-step TD target: {:.2}", g1); // 23.5

    // Verify 3-step by expanding: 1 + 0.9*(5 + 0.9*(1 + 0.9*25))
    let manual = 1.0 + 0.9 * (5.0 + 0.9 * (1.0 + 0.9 * 25.0));
    println!("Manual expansion: {:.2}", manual); // 24.54
}
```

`.fold(v_final, |g, &r| r + gamma * g)` iterates in reverse: start with \(G = V_\text{final}\), then for each reward (from last to first) apply \(G \leftarrow r + \gamma G\). This is the same backward accumulation as the Python `reversed(rewards)` loop, in one expression.

## The optimal value function

So far we have talked about the value function for a specific policy: V^π and Q^π. But often we want the value of the **best possible** policy. That is the **optimal value function**, denoted V* and Q*:

\[ V^*(s) = \max_\pi V^\pi(s) \]

In words: the maximum value achievable in state s by any policy.

For Q*:

\[ Q^*(s, a) = \max_\pi Q^\pi(s, a) \]

The corresponding Bellman equations look slightly different. The optimal policy in any state takes the action that maximizes Q*, so the "average over actions" gets replaced by "max over actions":

\[ V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma V^*(s')] \]

\[ Q^*(s, a) = \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma \max_{a'} Q^*(s', a')] \]

These are called the **Bellman optimality equations**. They define the optimal value functions self-referentially. Solving them gives you the optimal policy automatically: in each state, take the action that maximizes Q*.

## Why this all matters

The Bellman equation for Q* is the foundation of Q-learning. Q-learning is essentially: "use experience to estimate Q* by enforcing the Bellman optimality equation for the samples we have seen." We will see this concretely in the next lesson.

The Bellman equation for V is the foundation of policy evaluation methods, which appear inside actor-critic algorithms (lesson 6) and elsewhere.

The recursion idea (a state's value depends on the values of its successors) shows up in MCTS (Module 4), where we estimate values by recursively averaging over rollouts. It shows up in CFR (Module 5), where regret values are propagated through the game tree.

If you take one thing from this lesson: **value functions describe what the agent thinks the future is worth from each state, and the Bellman equation lets us compute these values using only local information about transitions and rewards.**

## Key Takeaways

- **V(s) measures the long-term value of being in a state; Q(s, a) measures the long-term value of taking an action in a state.** Q is strictly more informative — you can recover V from Q by averaging over the policy, but not vice versa. When you want to improve a policy, Q values give you direct action comparisons.
- **The Bellman equation is a self-consistency constraint.** A correct value function satisfies it exactly. An incorrect one violates it; the violation (the TD error δ) is the training signal that drives all TD-based RL algorithms. Positive δ means you underestimated the state; negative δ means you overestimated it.
- **Bootstrapping is biased but low-variance; Monte Carlo is unbiased but high-variance.** For long-horizon problems like satellite scheduling (where episodes span hundreds of steps), TD methods are almost always preferred because they update continuously rather than waiting for episode completion. The bias shrinks as estimates improve.
- **n-step returns interpolate between 1-step TD and Monte Carlo.** Using more actual steps before bootstrapping reduces bias at the cost of higher variance. PPO and A3C use n between 5 and 20 in practice; the exact value is a hyperparameter tuned per problem.
- **The advantage function A(s, a) = Q(s, a) - V(s) measures action quality relative to the baseline.** Subtracting V(s) from Q(s, a) reduces variance in policy gradient estimates without introducing bias, which is why modern algorithms like PPO use advantage rather than raw Q values for their policy update.
- **Optimal value functions satisfy the Bellman optimality equations with a max instead of an average.** This single change — from averaging over the policy to taking the maximum — converts policy evaluation into policy optimization and is the key step that makes Q-learning work.

## Quiz

{{#quiz 02-value-functions-and-bellman.toml}}
