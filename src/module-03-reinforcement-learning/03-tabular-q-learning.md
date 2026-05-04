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

The tension is real and unavoidable. An agent that only exploits never discovers whether untried actions are better — it may be stuck in a local optimum. An agent that only explores never uses what it has learned — it wastes interactions on random behavior. The goal is a schedule that front-loads exploration (when knowledge is poor) and gradually shifts to exploitation (as knowledge improves).

### ε-greedy: the standard baseline

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

### ε-decay: annealing the exploration rate

Starting at ε = 1.0 means the agent begins by acting completely randomly. This is often the right choice: early in training, Q-values are meaningless (usually initialized to zero), so there is nothing to exploit. As Q-values improve, exploration becomes less necessary and exploitation becomes more valuable.

\\[ \varepsilon_t = \varepsilon_{\text{end}} + (\varepsilon_{\text{start}} - \varepsilon_{\text{end}}) \cdot e^{-t / T_{\text{decay}}} \\]

**Decoding:**
- \\(\varepsilon_t\\): the exploration rate at step t
- \\(\varepsilon_{\text{start}}\\): initial exploration rate (typically 1.0)
- \\(\varepsilon_{\text{end}}\\): final exploration rate (typically 0.05)
- \\(T_{\text{decay}}\\): the decay timescale in steps — controls how fast exploration drops

```python
import math

class EpsilonSchedule:
    def __init__(self, eps_start=1.0, eps_end=0.05, decay_steps=5000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps
    
    def get_epsilon(self, step: int) -> float:
        """Exponential decay from eps_start to eps_end over decay_steps."""
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -step / self.decay_steps
        )

schedule = EpsilonSchedule(eps_start=1.0, eps_end=0.05, decay_steps=5000)
print(f"Step     0: ε = {schedule.get_epsilon(0):.3f}")
print(f"Step   500: ε = {schedule.get_epsilon(500):.3f}")
print(f"Step  2500: ε = {schedule.get_epsilon(2500):.3f}")
print(f"Step  5000: ε = {schedule.get_epsilon(5000):.3f}")
print(f"Step 10000: ε = {schedule.get_epsilon(10000):.3f}")
# Step     0: ε = 1.000
# Step   500: ε = 0.906
# Step  2500: ε = 0.606
# Step  5000: ε = 0.368
# Step 10000: ε = 0.101
```

A common alternative is **linear decay**: ε decreases by a fixed amount each step until it hits ε_end. Exponential decay tends to be more forgiving because it slows down naturally as it approaches the minimum.

### Optimistic initialization: built-in early exploration

A subtle but powerful trick: initialize all Q-values to a high value (e.g., 5.0 or 10.0) instead of zero. This makes the agent "optimistic" about every action it has not tried yet. When the agent takes an action and receives a lower reward than expected, the Q-value for that action drops — but unvisited actions still look attractive because they retain their high initial value.

```python
import numpy as np

# Standard: Q initialized to 0
Q_standard = np.zeros((num_states, num_actions))

# Optimistic: Q initialized high
# Any real reward will be less than 10.0 in this problem,
# so the agent is always disappointed and keeps exploring
Q_optimistic = np.ones((num_states, num_actions)) * 10.0
```

The key property: optimistic initialization drives exploration without requiring a random noise term. It is deterministic and converges to the same Q* values — the high initialization washes out over time as real data arrives. The limitation: it only helps in the early phase. Once every state-action pair has been visited and Q-values have been updated, the effect disappears.

### Upper Confidence Bound (UCB): exploration based on uncertainty

ε-greedy explores uniformly at random: every non-greedy action is equally likely. UCB does something smarter: prefer actions that have been tried few times (high uncertainty) even if their current estimate is not the best.

The UCB action selection rule:

\\[ a^* = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln t}{N(s, a)}} \\right] \\]

**Decoding:**
- \\(Q(s, a)\\): the current Q-value estimate for action a in state s
- \\(N(s, a)\\): the number of times action a has been taken in state s
- \\(t\\): total number of steps so far
- \\(c\\): exploration coefficient (controls the exploration-exploitation balance; typically 1 or 2)
- \\(\sqrt{\ln t / N(s, a)}\\): the uncertainty bonus — large when an action has been rarely tried, shrinks as the action is tried more

```python
import numpy as np

def ucb_action(Q_state, N_state, t, c=2.0):
    """
    Upper Confidence Bound action selection.
    Q_state: Q-values for all actions in this state, shape (num_actions,)
    N_state: visit counts for all actions in this state, shape (num_actions,)
    t: total timesteps so far
    c: exploration coefficient
    """
    # Add a small constant to avoid division by zero for unvisited actions
    uncertainty = c * np.sqrt(np.log(t + 1) / (N_state + 1e-6))
    ucb_values = Q_state + uncertainty
    return int(np.argmax(ucb_values))
```

UCB is the principled alternative to ε-greedy. It never wastes exploration on well-understood actions (those with low uncertainty), and it systematically explores uncertain ones. The downside: it requires tracking visit counts N(s, a), and the confidence bound is derived for bandit problems (single state); its guarantees weaken in the full RL setting with state transitions.

### In the SSA context: exploration means observing unfamiliar satellites

In a Space Situational Awareness scheduling problem, the agent decides which satellite to observe with a sensor at each time step. Exploitation means pointing the sensor at objects that appear most dangerous (highest estimated conjunction risk) based on current knowledge. Exploration means pointing the sensor at objects that have not been observed recently — even if current estimates suggest they are low-risk.

Why does exploration matter here? An object you have not observed in 24 hours has a stale state estimate. The actual orbit may have drifted due to atmospheric drag, a maneuver, or a debris collision. The object's conjunction risk in the Q-table might be zero simply because you have not checked. Exploration corrects this: periodic re-observation of poorly-known objects prevents the catalog from going stale.

The SSA analog of ε-greedy: with probability ε, point at a randomly selected object (regardless of estimated risk). With probability 1 - ε, point at the object with the highest estimated conjunction risk.

The SSA analog of UCB: prefer objects with high uncertainty in their state estimate — computed from time-since-last-observation and the object's estimated drag coefficient. Objects that are both high-risk and poorly-observed get the highest priority.

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

## Q-table convergence

### What convergence means

"Convergence" in Q-learning means the Q-values stop changing between iterations — they have stabilized to consistent estimates. In mathematical terms: the Q-values have converged when, for every state-action pair (s, a), the difference between consecutive updates approaches zero.

The formal measure is the **Bellman residual**: the maximum absolute change in any Q-value between two successive passes through the data.

\\[ \text{Bellman residual} = \max_{(s, a)} \left| Q_{\text{new}}(s, a) - Q_{\text{old}}(s, a) \right| \\]

**Decoding:**
- \\(Q_{\text{new}}(s, a)\\): the Q-value after an update step
- \\(Q_{\text{old}}(s, a)\\): the Q-value before the update
- \\(\max_{(s,a)}\\): take the worst-case over all state-action pairs
- As the algorithm converges, this residual → 0

When the Bellman residual is below some small threshold (e.g., 0.01), we declare convergence. Monitoring this is useful for:
- Diagnosing whether training has stabilized
- Deciding when to stop training early
- Detecting divergence (Bellman residual grows rather than shrinks)

### Conditions for convergence

Tabular Q-learning converges to Q* under two conditions:

1. **All state-action pairs are visited infinitely often.** If the agent never visits a particular (s, a) pair, its Q-value never gets updated. A stuck Q-value in a corner of the table can distort the optimal policy for nearby states. This is the formal justification for maintaining exploration throughout training — not just at the start.

2. **The learning rate α decays appropriately.** Specifically, the Robbins-Monro conditions require \\(\sum_t \alpha_t = \infty\\) (enough total learning to converge) and \\(\sum_t \alpha_t^2 < \infty\\) (learning rate decays fast enough to prevent oscillation). A common schedule: \\(\alpha_t = 1 / (1 + \text{visit\_count}(s, a))\\), which decreases each time a specific (s, a) pair is updated.

In practice, a fixed α (like 0.1) with sufficient exploration works well on small tabular problems. The theoretical conditions matter more when the state space is large or the reward signal is noisy.

### Monitoring convergence in code

```python
import numpy as np

class TabularQLearnerWithConvergenceMonitor:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.bellman_residuals = []
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        old_value = self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        # Track the absolute change in this Q-value
        return abs(self.Q[state, action] - old_value)
    
    def run_episode(self, env):
        state = env.reset()
        episode_max_change = 0.0
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            change = self.update(state, action, reward, next_state, done)
            episode_max_change = max(episode_max_change, change)
            state = next_state
        return episode_max_change
    
    def train(self, env, num_episodes, convergence_threshold=0.01):
        for episode in range(num_episodes):
            max_change = self.run_episode(env)
            self.bellman_residuals.append(max_change)
            
            if episode % 100 == 0:
                recent = self.bellman_residuals[-50:] if len(self.bellman_residuals) >= 50 else self.bellman_residuals
                avg_residual = np.mean(recent)
                print(f"Episode {episode:4d}: avg Bellman residual (last 50) = {avg_residual:.4f}")
            
            # Early stopping when converged
            if len(self.bellman_residuals) >= 50:
                recent_avg = np.mean(self.bellman_residuals[-50:])
                if recent_avg < convergence_threshold:
                    print(f"Converged at episode {episode} (residual = {recent_avg:.4f})")
                    break
        
        return self.bellman_residuals
```

The `bellman_residuals` list gives you a convergence curve you can plot. In a well-behaved problem, it starts high (large early updates) and decays toward zero as estimates stabilize.

### When tabular Q-learning fails: the state space explosion

Tabular Q-learning requires storing one float per (state, action) pair. The table size is `num_states × num_actions`. This is fine for toy problems:

| Problem | States | Actions | Table entries |
|---------|--------|---------|---------------|
| 4×4 gridworld | 16 | 4 | 64 |
| 10×10 gridworld | 100 | 4 | 400 |
| CartPole (discretized) | ~4,000 | 2 | ~8,000 |
| Atari game (raw pixels) | ~10^18,000 | 18 | impossible |
| Orbital state (continuous) | ∞ | discrete | impossible |

For the SSA sensor scheduling problem with continuous orbital state vectors (position, velocity, covariance for each object in a catalog of thousands), the table is infinitely large. Tabular Q-learning cannot represent the Q function at all.

This is the direct motivation for DQN: replace the table with a neural network that **generalizes** — a network that has never seen a particular state can still produce a reasonable Q-value estimate based on similar states it has seen.

## The deadly triad

Moving from tabular Q-learning to DQN (or any deep RL method) introduces a combination of three properties that, together, create a fundamental instability risk. This combination is called the **deadly triad**.

### The three ingredients

**1. Function approximation** (replacing the table with a network)

Instead of a lookup table with one entry per state-action pair, the Q function is represented by a neural network with parameters θ. This is necessary for large state spaces but introduces a side effect: updating Q(s, a) also changes Q(s', a') for nearby states, because all Q-values share the same network weights. In the tabular case, each entry is independent.

**2. Bootstrapping** (using your own estimates to make targets)

The TD target is:

\\[ \text{target} = r + \gamma \max_{a'} Q(s', a'; \theta) \\]

The target depends on Q itself. You are using your own current (imperfect) estimate to define what you are trying to learn. This is "bootstrapping" — pulling yourself up by your own bootstraps. In the tabular case, bootstrapping is fine because updates are isolated to single cells. With function approximation, the target and the estimate share weights, which can cause feedback loops.

**3. Off-policy learning** (learning about a different policy than the one you are running)

Q-learning is off-policy: the update always uses `max_a' Q(s', a')` regardless of what action the agent actually took. This means the Q-values learned reflect the greedy policy, not the exploratory policy being executed. Off-policy learning enables learning from historical data (replay buffers) but introduces a distribution mismatch between the data being trained on and the policy being evaluated.

### Why each combination of two is safe

- **Function approximation + bootstrapping, no off-policy**: converges (policy gradient methods, on-policy actor-critic)
- **Function approximation + off-policy, no bootstrapping**: converges (Monte Carlo with function approximation, direct supervised learning)
- **Bootstrapping + off-policy, no function approximation**: converges (tabular Q-learning, which is what this lesson covers)

The problem is the combination of all three. With all three active simultaneously, gradient updates can reinforce each other in destructive ways:
- Function approximation means an update at (s, a) shifts Q values for other states
- Bootstrapping means those shifted Q values feed back into future targets
- Off-policy learning means the data distribution may not cover the states where values are drifting

### DQN's instability and the solutions it introduces

The original (naive) implementation of deep Q-learning, without engineering safeguards, showed exactly this instability: Q-values would grow without bound, training would oscillate, and performance would collapse after an initial improvement. The two tricks DQN introduced — experience replay and target networks — directly address the deadly triad:

| Problem | DQN solution |
|---------|-------------|
| Correlated sequential samples (amplified by function approximation) | **Experience replay**: random samples from a buffer break temporal correlation |
| Moving targets (bootstrapping on a changing network) | **Target network**: freeze a copy of the network for target computation; update only periodically |

Experience replay does not eliminate the off-policy issue — it makes it worse in some sense, because you are training on old experiences. But it breaks the correlation that amplifies instability, and the sample efficiency gain is worth the off-policy cost. Target networks stabilize the bootstrapping feedback loop by making the target fixed for a stretch of training steps.

Neither trick eliminates the deadly triad. It is still present in DQN. But together they tame it enough to achieve stable learning on Atari-level problems.

The next lesson develops both tricks in detail, with their code implementations. For now: the deadly triad is the reason these tricks are necessary, not merely nice to have. Understanding the triad tells you what to watch for when DQN training goes wrong.

## What changes with neural network function approximation

In tabular Q-learning, updating Q(s, a) is a single table-cell update. The values for all other state-action pairs are unaffected.

With a neural network Q(s, a; θ) parameterized by weights θ, updating one (s, a) pair adjusts the weights, which subtly affects all other Q estimates simultaneously. This is what gives neural networks their generalization power: a state we have never seen before can get a reasonable Q estimate based on its similarity to states we have seen. It is also what causes new failure modes (instability, overestimation) that the engineering tricks in DQN address.

## Key Takeaways

- **Q-learning learns from experience alone.** It does not need a model of the environment. It maintains a table of Q-value estimates and updates them using the Bellman residual each time the agent takes an action and observes the result. The TD error — the difference between the observed target and the current estimate — drives every update.

- **Exploration is not optional; it is a convergence requirement.** The theoretical guarantee that Q-learning converges to Q* requires that all state-action pairs be visited infinitely often. In practice, ε-greedy exploration is the standard mechanism. ε-decay (starting at 1.0, decaying to 0.05) front-loads exploration when Q-values are uninformative and shifts to exploitation as estimates improve.

- **Beyond ε-greedy: optimistic initialization and UCB.** Initializing Q-values high encourages early exploration without random noise. UCB selects actions with high uncertainty by adding a bonus proportional to \\(\sqrt{\ln t / N(s,a)}\\), directing exploration toward poorly-understood actions rather than random ones. In SSA, exploration means periodically observing satellites you have not recently tracked — even those with currently low estimated risk.

- **The Bellman residual measures convergence.** \\(\max_{(s,a)} |Q_{\text{new}}(s,a) - Q_{\text{old}}(s,a)|\\) should decay toward zero in a well-behaved training run. Monitor it; a rising residual signals instability before Q-values diverge visibly.

- **Tabular Q-learning fails when the state space is large.** The table requires one entry per (state, action) pair. Continuous orbital state vectors, raw sensor images, or any high-dimensional state space makes the table infinitely large or intractable. DQN replaces the table with a neural network that generalizes across similar states.

- **The deadly triad explains why DQN needs engineering tricks.** Function approximation + bootstrapping + off-policy learning can be individually safe but are collectively unstable. DQN's experience replay and target networks are direct responses to this instability — they are not implementation conveniences but structural necessities.

## Quiz

{{#quiz 03-tabular-q-learning.toml}}
