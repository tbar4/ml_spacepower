# Lesson 3: Tabular Q-Learning

## Where this fits

This is the first lesson where we actually build a learning agent. Q-learning is the simplest reinforcement learning algorithm that solves the central RL problem: finding a good policy without being told the dynamics of the environment in advance. The agent learns purely from experience, by trial and error. The "tabular" version (this lesson) stores Q values in a literal table, one entry per state-action pair. The next lesson replaces the table with a neural network, and you get DQN. Everything else in deep RL builds on the ideas you will see here.

## The problem Q-learning solves

In lesson 2, value iteration computed Q* by sweeping over all states and actions and using the Bellman optimality equation. It needed two things:
1. The transition probabilities P(s' | s, a)
2. The reward function R(s, a, s')

In real problems, you usually have neither. You have an environment you can interact with: take an action, get a reward and a next state. That is it. No closed-form access to the underlying dynamics.

Q-learning's job is to estimate Q* from this interaction alone. It does this by maintaining a table of Q value estimates and updating them every time the agent takes an action and observes what happens.

## The core idea: temporal difference learning

The Bellman optimality equation says:

\\[ Q^*(s, a) = \mathbb{E}[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')] \\]

In words: the Q value of (s, a) equals the expected immediate reward plus the discounted Q value of the best next action.

Suppose we have a current estimate Q(s, a). We take action a in state s, observe reward r and next state s'. Now we have one sample of the right-hand side of the Bellman equation:

\\[ \text{sample} = r + \gamma \max_{a'} Q(s', a') \\]

This is what the Q value "should be" according to this one piece of experience. Compare it to our current estimate Q(s, a). The difference is called the **TD error** (temporal difference error):

\\[ \delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a) \\]

**Decoding:**
- \\(\delta\\) (Greek delta): standard notation for the TD error
- \\(r + \gamma \max_{a'} Q(s', a')\\): the "target" (what Q should be, according to this sample)
- \\(Q(s, a)\\): our current estimate
- The difference is positive if our current estimate is too low, negative if too high

Q-learning updates the estimate by moving it a small step toward the target:

\\[ Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \delta \\]

**Decoding:**
- \\(\leftarrow\\): assignment (overwrites the old value)
- \\(\alpha\\) (Greek alpha): the **learning rate**, a small positive number (like 0.1)
- \\(\alpha \cdot \delta\\): how much to adjust toward the target

If we use a learning rate of 0.1, we move 10% of the way toward the new sample's target each time. Over many updates, the estimates converge to Q*.

## Walking through one update by hand

Suppose for a tiny MDP with 3 states and 2 actions, our current Q table is:

|       | Action 0 | Action 1 |
|-------|----------|----------|
| State 0 | 1.0    | 0.5      |
| State 1 | 2.0    | 3.0      |
| State 2 | 0.0    | 0.0      |

We are in state 0, take action 0, observe reward 2.0 and next state 1.

**Step 1**: Compute max over next-state Q values:
- Q(state 1, action 0) = 2.0
- Q(state 1, action 1) = 3.0
- max = 3.0

**Step 2**: Compute the target (with γ = 0.9):
- Target = 2.0 + 0.9 × 3.0 = 2.0 + 2.7 = 4.7

**Step 3**: Compute the TD error:
- δ = 4.7 − Q(state 0, action 0) = 4.7 − 1.0 = 3.7

**Step 4**: Update Q (with α = 0.1):
- Q(state 0, action 0) ← 1.0 + 0.1 × 3.7 = 1.37

The Q value for (state 0, action 0) moved from 1.0 to 1.37. The other Q values are unchanged. After many such updates from many trajectories, the table converges to good estimates of Q*.

## The exploration-exploitation tradeoff

Here is a problem we have not addressed: if the agent always takes the action with the highest current Q value (greedy action selection), it will keep doing whatever looks best with its current (noisy, possibly wrong) estimates. It will never try other actions and never learn whether those actions might actually be better.

This is the **exploration-exploitation tradeoff**:
- **Exploit**: take the action that looks best given current knowledge
- **Explore**: try other actions to learn more

The simplest solution: **ε-greedy** (epsilon-greedy). With probability ε, take a random action (exploration). Otherwise, take the action with the highest Q value (exploitation).

```python
import numpy as np

def epsilon_greedy(Q_state, epsilon):
    """Return an action using epsilon-greedy selection."""
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q_state))  # random action
    else:
        return int(np.argmax(Q_state))  # greedy action
```

Common values of ε:
- 0.1 (10% exploration) is a common starting point
- Often ε is annealed: start at 1.0 (pure exploration), decrease to 0.05 (mostly exploit) over training

The right schedule depends on the problem. We will use ε = 0.1 for the example below.

## A complete tabular Q-learning algorithm

Putting it all together:

```python
import numpy as np

class TabularQLearner:
    def __init__(self, num_states, num_actions, learning_rate=0.1, 
                 discount=0.9, epsilon=0.1):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.num_actions = num_actions
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state, action, reward, next_state, done):
        # The TD target: if the episode ended, no future value
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD error
        td_error = target - self.Q[state, action]
        
        # Update toward the target
        self.Q[state, action] += self.alpha * td_error
```

That is the complete algorithm. Train by repeatedly:
1. Reset the environment to a starting state
2. Loop: select action, take it, observe reward and next state, call `update`, repeat until episode ends
3. Continue for many episodes

After enough episodes, Q converges (in tabular settings, under mild conditions) to Q*, and the greedy policy with respect to Q is the optimal policy.

## A worked example: a tiny gridworld

Let us train a Q-learning agent on a simple gridworld. The agent is on a 4x4 grid, starting at (0,0), trying to reach (3,3). Each step gives reward -1 (encouraging short paths). Reaching the goal gives reward 0 and ends the episode. There is a "dangerous" cell at (1,1) that gives reward -10.

```python
import numpy as np

np.random.seed(42)

# Environment
GRID_SIZE = 4
START   = (0, 0)
GOAL    = (3, 3)
DANGER  = (1, 1)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

def step(state, action_idx):
    dy, dx = ACTIONS[action_idx]
    new_y = max(0, min(GRID_SIZE - 1, state[0] + dy))
    new_x = max(0, min(GRID_SIZE - 1, state[1] + dx))
    new_state = (new_y, new_x)
    
    if new_state == GOAL:
        return new_state, 0, True
    elif new_state == DANGER:
        return new_state, -10, False
    else:
        return new_state, -1, False

def state_to_idx(state):
    return state[0] * GRID_SIZE + state[1]

# Initialize Q learner
agent = TabularQLearner(num_states=GRID_SIZE * GRID_SIZE, 
                       num_actions=4,
                       learning_rate=0.1,
                       discount=0.99,
                       epsilon=0.1)

# Train for 1000 episodes
returns = []
for episode in range(1000):
    state = START
    total_reward = 0
    for step_count in range(50):  # max 50 steps
        action = agent.select_action(state_to_idx(state))
        next_state, reward, done = step(state, action)
        agent.update(state_to_idx(state), action, reward, 
                    state_to_idx(next_state), done)
        state = next_state
        total_reward += reward
        if done:
            break
    returns.append(total_reward)

# Look at average performance over time
print("Average return over training:")
print(f"  Episodes 1-100:    {np.mean(returns[:100]):.2f}")
print(f"  Episodes 500-600:  {np.mean(returns[500:600]):.2f}")
print(f"  Episodes 900-1000: {np.mean(returns[900:1000]):.2f}")

# Visualize the learned policy (greedy actions)
arrows = ['→', '←', '↓', '↑']
print("\nLearned policy:")
for y in range(GRID_SIZE):
    row = ""
    for x in range(GRID_SIZE):
        if (y, x) == GOAL:
            row += " G "
        elif (y, x) == DANGER:
            row += " X "
        else:
            best_action = np.argmax(agent.Q[state_to_idx((y, x))])
            row += f" {arrows[best_action]} "
    print(row)
```

After training, you will see the agent learns to:
- Move toward the goal (you will see arrows generally pointing right and down)
- Avoid the danger cell at (1, 1)

The first 100 episodes produce poor returns (~ -25 on average) because the agent is mostly random. By episode 1000, returns improve to around -6 (close to the optimal path of 6 steps × -1 reward each).

## Key properties of tabular Q-learning

**Convergence**: in tabular settings, Q-learning is proven to converge to Q* under these conditions:
- All state-action pairs are visited infinitely often (or at least often enough)
- The learning rate α decays appropriately over time

In practice, with a fixed reasonable α (like 0.1) and reasonable exploration (like ε = 0.1), Q-learning converges to a near-optimal policy on small problems.

**Off-policy**: Q-learning is "off-policy" because the update uses max over next-state Q values, regardless of what action the policy actually takes. This means you can learn the optimal policy even while following an exploratory or otherwise suboptimal policy. (Compare to SARSA, an on-policy variant that uses the actual next action; we will not cover SARSA in detail, but it appears in some literature.)

**Limitation**: tabular Q-learning needs one entry per state-action pair. For our 16-state gridworld, that is 64 entries. Trivial. For chess, with 10^47 states, the table is impossible. For continuous state spaces (like the orbital state vector), the table is infinitely large.

This is what motivates DQN (next lesson): replace the table with a neural network that approximates the Q function. The conceptual algorithm is the same; the storage and lookup change.

## What changes with neural network function approximation

In tabular Q-learning, updating Q(s, a) is a single table-cell update. The values for all other state-action pairs are unaffected.

With a neural network Q(s, a; θ) parameterized by weights θ, updating one (s, a) pair adjusts the weights, which subtly affects all other Q estimates simultaneously. This is what gives neural networks their generalization power: a state we have never seen before can get a reasonable Q estimate based on its similarity to states we have seen. It is also what causes new failure modes (instability, overestimation) that the engineering tricks in DQN address.

## Quiz

{{#quiz 03-tabular-q-learning.toml}}
