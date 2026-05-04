# Lesson 4: Deep Q-Networks (DQN)

## Where this fits

Tabular Q-learning works beautifully for small problems. It cannot work for the kinds of problems we care about (chess, satellite scheduling, anything with a continuous state space) because the table would be impossibly large or infinite. DQN replaces the table with a neural network: instead of looking up Q(s, a) in a table indexed by states, you pass the state through a network that outputs Q values for all actions. The conceptual algorithm is unchanged. The implementation requires two engineering tricks (experience replay and target networks) that solve specific instability problems caused by the function approximator. This is the algorithm that achieved superhuman Atari play in 2013-2015 and is the foundation of every modern value-based deep RL method.

## The basic idea

In tabular Q-learning, your Q values lived in a table:

```
Q[state_index, action_index] = value
```

In DQN, your Q values come from a neural network:

```python
Q_values = network(state)  # returns a vector of Q values, one per action
```

The network's input is the state vector (any features that describe the state). Its output is a vector of length `num_actions`, where each entry is the Q value for one action. To get Q(s, a), you forward-pass s through the network and take the entry at index a.

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),  # one output per action
        )
    
    def forward(self, state):
        return self.net(state)

# Example: 6-dimensional state, 4 possible actions
q_net = QNetwork(state_dim=6, num_actions=4)
state = torch.randn(6)
q_values = q_net(state)
print(f"Q values: {q_values.tolist()}")
# Returns 4 values, one per action

# Greedy action selection
best_action = q_values.argmax().item()
print(f"Best action: {best_action}")
```

This is structurally identical to the conjunction-risk regressor from Module 2, except the output dimension is `num_actions` instead of 1.

## Adapting the Q-learning update for a neural network

In tabular Q-learning, an update was:

```
Q[s, a] += α × (r + γ × max(Q[s']) − Q[s, a])
```

A direct table-cell change. With a neural network, we cannot directly assign a value to Q(s, a). The Q value comes from a function with thousands of parameters. We have to **train** the network so that its output matches the target.

The TD target is the same as before:

\\[ \text{target} = r + \gamma \max_{a'} Q(s', a'; \theta) \\]

(The notation \\(Q(s, a; \theta)\\) emphasizes that Q is computed by a network with parameters \\(\theta\\).)

We define a loss function: how far is the network's current prediction from the target?

\\[ L(\theta) = \left( Q(s, a; \theta) - \text{target} \right)^2 \\]

This is just MSE loss between the prediction and the target. Then we use gradient descent on \\(\theta\\) to reduce the loss. The chain rule (Module 1, lesson 7) and PyTorch's autograd handle the rest.

```python
def compute_loss(q_net, state, action, reward, next_state, done, gamma):
    # Current Q estimate for the (s, a) pair
    q_values = q_net(state)
    q_estimate = q_values[action]
    
    # TD target
    if done:
        target = reward
    else:
        with torch.no_grad():
            next_q_values = q_net(next_state)
            target = reward + gamma * next_q_values.max()
    
    # MSE loss
    loss = (q_estimate - target) ** 2
    return loss
```

Notice the `with torch.no_grad():` around the target computation. We do not want gradients to flow through the target. The target is a stale estimate that we are trying to make Q(s, a) match; we are not trying to update the target itself.

## Why naive DQN does not work

If you train this naive version, you will probably see:
- Loss oscillating or even diverging
- Q values exploding to large positive or negative numbers
- Performance getting worse over time

There are two main reasons.

**Problem 1: Sequential samples are highly correlated.**

In RL, the agent generates samples by interacting with the environment. Consecutive samples come from consecutive timesteps and are highly correlated: state at t+1 is determined by state at t and the action taken. Standard supervised learning assumes samples are independent and identically distributed (i.i.d.). Correlated samples violate this assumption and can cause training to oscillate.

**Problem 2: The target moves with the network.**

In our naive update, the target is:

\\[ r + \gamma \max_{a'} Q(s', a'; \theta) \\]

Notice θ on both sides. The target depends on the current network parameters. When we update θ to reduce the loss, the target also changes, because it is computed using the same network. This is like trying to hit a moving target: every time you move toward where the target was, the target moves to a new location.

Both problems are real, and DQN solves them with two engineering tricks.

## Trick 1: Experience replay

Instead of training on the current transition immediately, store transitions in a **replay buffer** (a queue of past experiences). At each training step, sample a random batch from the buffer and train on those.

### Why replay works: breaking temporal correlation

The replay buffer converts the sequential stream of agent experience into something resembling i.i.d. samples. When you sample a random batch from a buffer containing 100,000 transitions, consecutive items in the batch may come from completely different episodes, environments, and time periods. The temporal correlation that plagues sequential training is broken.

A secondary benefit: data efficiency. Each transition stored in the buffer can be sampled and trained on multiple times. In a buffer of size 100,000 with batch size 64 drawn each step, a given transition will on average be part of roughly 64 training batches before being evicted. This multiplies the effective number of gradient updates per unit of agent-environment interaction.

### Ring buffer implementation

The replay buffer is typically implemented as a ring buffer (circular queue): when the buffer is full, the oldest experience is overwritten by the newest. This ensures the buffer always contains the most recent `capacity` transitions, and memory usage is bounded.

```python
import torch
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Ring-buffer replay buffer for DQN.
    
    When capacity is reached, the oldest transition is overwritten.
    The buffer holds (state, action, reward, next_state, done) tuples.
    """
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        
        # Pre-allocate arrays for efficiency (avoids Python object overhead)
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0         # points to next write position
        self.size = 0        # current number of stored transitions
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition. Overwrites the oldest entry when full."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        
        # Advance pointer with wrap-around (ring buffer semantics)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """
        Sample a random batch of transitions.
        Returns tensors ready for PyTorch training.
        """
        assert self.size >= batch_size, (
            f"Buffer has {self.size} transitions but batch_size={batch_size}. "
            f"Wait until the buffer has at least {batch_size} transitions before training."
        )
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.states[indices]),
            torch.tensor(self.actions[indices]),
            torch.tensor(self.rewards[indices]),
            torch.tensor(self.next_states[indices]),
            torch.tensor(self.dones[indices]),
        )
    
    def ready(self, min_size: int) -> bool:
        """Return True when the buffer has enough data to start training."""
        return self.size >= min_size
    
    def __len__(self):
        return self.size
```

The pre-allocated numpy arrays (rather than a Python `deque`) are important for performance: at 100,000 transitions with 6-dimensional states, the buffer holds ~4.8 million floats. A deque of Python tuples would use significantly more memory due to object overhead.

### Priority replay: learning more from surprising experiences

Standard replay samples uniformly at random. Priority replay samples with probability proportional to the magnitude of the TD error: experiences where the current Q-network prediction was very wrong are sampled more often, because they have more to teach.

The sampling probability for transition i:

\\[ P(i) = \frac{|\delta_i|^\alpha}{\sum_j |\delta_j|^\alpha} \\]

**Decoding:**
- \\(|\delta_i|\\): the absolute TD error for transition i (how wrong our prediction was)
- \\(\alpha\\): controls the degree of prioritization (\\(\alpha = 0\\) is uniform sampling; \\(\alpha = 1\\) is fully proportional)
- New transitions are assigned maximum priority until their TD error is computed at first training

Priority replay is used in Prioritized Experience Replay (PER), a common DQN extension. It requires an importance sampling correction to account for the non-uniform sampling distribution, and a data structure (sum-tree) to make priority sampling efficient. For the scope of this lesson, uniform replay is the baseline — priority replay is a well-known enhancement worth knowing about.

### Minimum replay buffer size before training starts

A critical implementation detail: do not start training immediately. Wait until the buffer has accumulated a minimum number of transitions — typically a few thousand, or at least 10× the batch size.

Why: with only a handful of transitions in the buffer, random sampling produces highly correlated batches (the same transitions appear repeatedly). This defeats the purpose of the buffer. More importantly, with few transitions all from early exploration, Q-values will be updated based on an extremely narrow slice of the state space, often causing early divergence.

```python
MIN_REPLAY_SIZE = 1000  # collect this many transitions before any training
BATCH_SIZE = 64

# In the training loop:
if not replay_buffer.ready(MIN_REPLAY_SIZE):
    # Just collect experience, do not train yet
    continue
```

In SSA scheduling, this warmup period is especially important: early transitions are all from random sensor pointing (pure exploration), and the Q-network should not lock in estimates based only on randomly-sampled observations before it has seen the full range of satellite states.

## Trick 2: Target networks

Maintain a second network with the same architecture, called the **target network**, with parameters \\(\theta^-\\) (theta-minus, the convention). The target network's parameters are kept frozen, updated only periodically (every N training steps) to match the main network.

The TD target is computed using the target network:

\\[ \text{target} = r + \gamma \max_{a'} Q(s', a'; \theta^-) \\]

The loss is still computed using the main network:

\\[ L(\theta) = \left( Q(s, a; \theta) - \text{target} \right)^2 \\]

This solves Problem 2: the target is now stable for a stretch of training updates (it changes only every N steps when we sync the target network to match the main network). The main network can train against this stable target without chasing a moving goal.

### Hard update: copying weights every C steps

The original DQN paper used a **hard update**: every C training steps, copy the online network's weights directly into the target network. Between updates, the target network is completely frozen.

\\[ \theta^- \leftarrow \theta \quad \text{every C steps} \\]

```python
import torch.nn as nn

def hard_update(online_net: nn.Module, target_net: nn.Module):
    """Copy online network weights to target network."""
    target_net.load_state_dict(online_net.state_dict())

# In the training loop:
if steps % target_update_freq == 0:
    hard_update(q_net, target_net)
```

The hard update creates a step function for the target: it is constant for C steps, then jumps to match the current online network. The jump can be large if C is small (causing target instability) or leave targets very stale if C is large (slowing learning).

### Soft update (Polyak averaging): smoother target drift

An alternative used in many modern algorithms: **soft update** (also called Polyak averaging). Rather than copying the weights periodically, blend the target network slightly toward the online network at every training step.

\\[ \theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^- \\]

**Decoding:**
- \\(\tau\\) (tau): a small blending coefficient, typically 0.005 or smaller
- \\(\theta\\): the online network's current weights
- \\(\theta^-\\): the target network's current weights
- With \\(\tau = 0.005\\), each step moves the target 0.5% of the way toward the online network

```python
def soft_update(online_net: nn.Module, target_net: nn.Module, tau: float = 0.005):
    """
    Polyak averaging: blend target network toward online network.
    tau=0.005 is typical for DQN variants; tau=1.0 is a hard copy.
    """
    for online_param, target_param in zip(
        online_net.parameters(), target_net.parameters()
    ):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )

# Side by side: hard vs soft
def hard_update(online_net: nn.Module, target_net: nn.Module):
    target_net.load_state_dict(online_net.state_dict())

# Hard update: call every C steps (e.g., every 500 training steps)
# if steps % 500 == 0:
#     hard_update(q_net, target_net)

# Soft update: call every training step
# soft_update(q_net, target_net, tau=0.005)
```

Why soft update is often more stable: the target network drifts smoothly rather than jumping. The gradient signal the online network trains against changes gradually, which prevents oscillation. Hard updates are simpler and worked well in the original DQN, but soft updates are the default in many modern algorithms (DDPG, TD3, SAC all use soft updates). The tradeoff: soft updates with very small τ can make the target so slow-moving that it does not keep up with rapid policy improvement.

Typical values:
- Hard update: every 100–1000 training steps (smaller problems → more frequent; larger → less frequent)
- Soft update: τ = 0.005 (used in many continuous control papers)

## A complete DQN implementation

Putting it all together:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, num_actions, 
                 lr=1e-3, gamma=0.99, epsilon=0.1,
                 buffer_capacity=10_000, batch_size=64,
                 target_update_freq=500):
        # Two networks
        self.q_net      = QNetwork(state_dim, num_actions)
        self.target_net = QNetwork(state_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())  # initialize identical
        
        self.optimizer  = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.gamma   = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        
        self.buffer     = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size
        
        self.target_update_freq = target_update_freq
        self.steps = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None  # not enough data yet
        
        # Sample a batch from the replay buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states      = torch.tensor(states,      dtype=torch.float32)
        actions     = torch.tensor(actions,     dtype=torch.int64)
        rewards     = torch.tensor(rewards,     dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones,       dtype=torch.float32)
        
        # Current Q estimates: Q(s, a) for the actions actually taken
        q_values     = self.q_net(states)              # shape (batch, num_actions)
        q_estimates  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)
        
        # Targets: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values   = self.target_net(next_states)  # (batch, num_actions)
            max_next_q      = next_q_values.max(dim=1).values  # (batch,)
            targets         = rewards + self.gamma * max_next_q * (1 - dones)
        
        # MSE loss
        loss = F.mse_loss(q_estimates, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodically sync target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
```

The key new operations:

**`q_values.gather(1, actions.unsqueeze(1))`**: this picks the Q value for the action that was actually taken in each sample of the batch. `q_values` is shape (batch, num_actions); `actions` is shape (batch,). The gather operation indexes into dimension 1 using the action indices.

**`(1 - dones)`**: when an episode ends (done = 1), there is no future to bootstrap from. Multiplying by `(1 - dones)` zeroes out the future-value term in those cases, leaving just the immediate reward as the target.

## Training loop

The full agent-environment loop now interleaves environment interaction with training:

```python
def train_dqn(env, agent, num_episodes=500):
    episode_returns = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        
        for step in range(200):  # max steps per episode
            # 1. Select an action
            action = agent.select_action(state)
            
            # 2. Take the action and observe the result
            next_state, reward, done = env.step(action)
            
            # 3. Store the transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # 4. Train on a batch from the replay buffer
            agent.train_step()
            
            state = next_state
            episode_return += reward
            
            if done:
                break
        
        episode_returns.append(episode_return)
        
        if episode % 50 == 0:
            recent = episode_returns[-50:]
            print(f"Episode {episode}: avg return over last 50 = {sum(recent)/len(recent):.2f}")
    
    return episode_returns
```

Steps 1-3 are environment interaction. Step 4 is the supervised-style training update on a batch from the buffer. The two are interleaved: each timestep generates one new transition and triggers one training update.

## DQN failure modes

Even with experience replay and target networks, DQN can fail in predictable ways. Understanding these failure modes is what separates practitioners who can debug DQN from those who can only run it.

### Overestimation bias: Q-values systematically too high

The `max` in the TD target introduces a systematic upward bias:

\\[ \text{target} = r + \gamma \max_{a'} Q(s', a'; \theta^-) \\]

The `max` over noisy Q-value estimates tends to pick the most overestimated value. If Q-values have random noise (they always do, especially early in training), the max over noisy estimates is higher in expectation than the true max over the true values. This bias accumulates over many bootstrapping steps and causes Q-values to grow without bound in the worst case.

Symptoms of overestimation bias:
- Q-values keep growing throughout training (check by logging `q_estimates.mean()` each step)
- Loss is large and not decreasing
- The agent's policy appears confident (always high Q values) but performance is poor

### Double DQN: decoupling action selection from action evaluation

Double DQN is a targeted fix for overestimation bias. The key insight: the bias comes from using the same network to both **select** the best action and **evaluate** its Q-value. If the estimates are noisy, the selector picks the most overestimated action, and the evaluator confirms the overestimate.

Double DQN decouples these two operations:
- Use the **online network** to select which action is best in the next state
- Use the **target network** to evaluate the Q-value of that action

\\[ \text{target}_{\text{Double DQN}} = r + \gamma \cdot Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \, \theta^-\right) \\]

**Decoding:**
- \\(\arg\max_{a'} Q(s', a'; \theta)\\): the online network selects the best action
- \\(Q(\ldots; \theta^-)\\): the target network evaluates that specific action's Q-value
- Since the selector (online) and evaluator (target) have different parameters, the same overestimation cannot dominate both

```python
import torch
import torch.nn.functional as F

def compute_double_dqn_loss(
    online_net,
    target_net,
    states,
    actions,
    rewards,
    next_states,
    dones,
    gamma: float = 0.99,
):
    """
    Double DQN loss: online network selects action, target network evaluates it.
    All inputs are PyTorch tensors with batch dimension first.
    """
    # --- Current Q-value estimates (online network) ---
    q_values    = online_net(states)                                 # (batch, num_actions)
    q_estimates = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

    with torch.no_grad():
        # --- Double DQN target ---
        # Step 1: online network selects the best next action
        next_q_online  = online_net(next_states)                     # (batch, num_actions)
        best_actions   = next_q_online.argmax(dim=1, keepdim=True)   # (batch, 1)

        # Step 2: target network evaluates that action's Q-value
        next_q_target  = target_net(next_states)                     # (batch, num_actions)
        next_q_values  = next_q_target.gather(1, best_actions).squeeze(1)  # (batch,)

        # Bellman target
        targets = rewards + gamma * next_q_values * (1.0 - dones)

    loss = F.mse_loss(q_estimates, targets)
    return loss
```

Double DQN is a drop-in replacement for standard DQN — same network architecture, same replay buffer, same target network — with only the target computation changed. It consistently reduces overestimation bias across a wide range of environments and is now considered the default DQN variant.

### Reward scaling: keeping Q-values in a sane range

Q-values are sums of discounted future rewards. If rewards are large in magnitude, Q-values can be enormous (or tiny), which causes numerical problems:
- Very large Q-values → large gradients → unstable training
- Very small Q-values → vanishing gradients → no learning

The standard fix is to scale rewards to a small range before storing them in the replay buffer. The two most common approaches:

**Clip rewards to [-1, 1]:** the original Atari DQN paper used this. It discards reward magnitude information but works well when the sign of the reward (positive/negative) is what matters.

```python
reward = max(-1.0, min(1.0, reward))
```

**Normalize rewards:** scale by a running estimate of the standard deviation. This preserves relative magnitude information.

```python
class RunningStats:
    """Welford online algorithm for running mean and variance."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2  += delta * (x - self.mean)
    
    @property
    def std(self):
        return (self.M2 / max(self.n - 1, 1)) ** 0.5 + 1e-8

reward_stats = RunningStats()

def normalize_reward(r):
    reward_stats.update(r)
    return r / reward_stats.std
```

In the SSA scheduling context, rewards might represent detection probability improvements measured in fractions (e.g., 0.05 to 0.3) or orbital uncertainty reductions measured in meters (e.g., 1.0 to 1000.0). The raw values from your simulation domain need to be normalized before feeding them to the DQN. A reward of 500 meters of uncertainty reduction is fine for the human operator; it is a disaster for the Q-network's gradient unless scaled.

### NaN and inf in Q-values: diagnosis and prevention

NaN (Not a Number) or inf in Q-values means training has catastrophically failed. Common causes and fixes:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Q-values → inf after a few thousand steps | Rewards too large, unstable bootstrapping | Clip or normalize rewards |
| Loss → NaN immediately | Learning rate too high | Reduce lr by 10x |
| Q-values → NaN, loss was fine | Numerical instability in the network | Add gradient clipping |
| Q-values oscillate then NaN | Target update frequency too low (targets move too fast) | Increase `target_update_freq` |

Gradient clipping is a defensive measure that prevents single large gradient updates from destabilizing the network:

```python
# Add after loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
```

This caps the global gradient norm at 10.0. If the gradient is smaller than 10.0, it passes through unchanged. If larger, all gradient components are scaled down proportionally. The original DQN paper used gradient clipping with max_norm=10.

## Key hyperparameters and what they do

- **Learning rate (lr)**: standard neural network learning rate. Typical: 1e-3 to 1e-4.
- **Discount factor (gamma)**: as in MDP. Typical: 0.99 for problems with long horizons.
- **Epsilon**: exploration rate. Often annealed from 1.0 down to 0.05 over training.
- **Buffer capacity**: how many transitions to remember. Typical: 10,000 to 1,000,000.
- **Batch size**: how many transitions to sample per training step. Typical: 32 to 256.
- **Target update frequency**: how often to sync the target network. Typical: every 500 to 10,000 steps.
- **Update frequency**: how often to train. Often every step, but can be every N steps to make the agent collect more data per update.

DQN is sensitive to these hyperparameters. The defaults work for many problems but tuning is sometimes necessary.

## DQN hyperparameter reference table

| Hyperparameter | Typical value | Effect if too large | Effect if too small |
|---|---|---|---|
| `learning_rate` | 1e-4 | Unstable training, loss oscillates or diverges | Very slow convergence, may not converge at all |
| `batch_size` | 32–128 | Slow per-step wall time, but more stable gradient estimates | Fast updates but noisy gradients; high variance |
| `replay_buffer_size` | 100k–1M | More memory usage; older experiences may be irrelevant | Less diverse samples; transitions reused too often |
| `target_update_freq` (hard) | 100–1000 steps | Very stale targets; slow to incorporate policy improvement | Targets move too fast; unstable training (approaches naive DQN) |
| `tau` (soft update) | 0.005 | Target moves too fast toward online net | Target barely moves; learning stalls |
| `epsilon_start` | 1.0 | (Not applicable — higher only means more random behavior at start) | Insufficient early exploration; Q-values lock in too quickly |
| `epsilon_end` | 0.01–0.1 | Too much random action at test time; policy appears suboptimal | Essentially no exploration at end of training |
| `gamma` | 0.95–0.99 | Overvalues distant future rewards; Q-values grow large | Short-sighted policy; ignores delayed consequences |
| `min_replay_size` | 1k–10k | Slower to start training | Training starts on too-narrow data; early divergence |

A useful heuristic for SSA scheduling: start with `gamma = 0.99` (orbital planning problems have long-horizon consequences), `epsilon` annealed from 1.0 to 0.05 over 50,000 steps (the catalog needs to be surveyed before the agent can exploit knowledge), and `target_update_freq = 500` with hard update. These are reasonable defaults; monitor Q-value magnitude and adjust.

## Where DQN succeeds and fails

DQN is well-suited to:
- Discrete action spaces (it computes a Q value per action)
- Problems where the value function is reasonably smooth
- Single-agent settings (multi-agent introduces non-stationarity that DQN does not handle natively)

DQN struggles with:
- Continuous action spaces (max over continuous actions is hard; use DDPG or SAC instead)
- Sparse rewards (without good initial exploration, the agent may never see any reward)
- Highly stochastic environments (the max in the target overestimates Q values, a known issue called overestimation bias; Double DQN partially fixes this)

For our SSA-flavored sensor scheduling problem, DQN is a reasonable choice: discrete actions (which sensor to point), reasonably structured rewards (detection events), and a single agent.

## Key Takeaways

- **DQN replaces the Q-table with a neural network, keeping the same conceptual algorithm.** The forward pass produces Q values for all actions; training minimizes MSE between the current Q estimate and the TD target. The `gather` operation selects the Q value for the action actually taken; the `(1 - done)` mask zeroes out bootstrapping on terminal states.

- **Experience replay breaks temporal correlation and enables data reuse.** A ring buffer stores the last N transitions; training samples randomly from this buffer rather than training on the current transition sequentially. Do not start training until the buffer has accumulated a meaningful number of diverse transitions (at least 10× the batch size, ideally thousands).

- **Target networks stabilize bootstrapping by fixing the target for a stretch of training steps.** Hard update (copy every C steps) is simple and effective. Soft update (Polyak averaging with small τ) is smoother and often more stable. Both are better than the naive approach where target and online network are the same.

- **Overestimation bias is systematic, not random.** The `max` over noisy Q-value estimates skews high in expectation. Double DQN fixes this by decoupling action selection (online network) from action evaluation (target network) — a two-line change to the target computation that consistently improves training stability.

- **Reward scaling is not optional.** Q-values are cumulative discounted sums: a reward of 500 becomes a Q-value in the thousands under bootstrapping. Clip rewards to [-1, 1] or normalize by running standard deviation before storing in the replay buffer. In SSA, orbital uncertainty values must be scaled before feeding to the DQN.

- **NaN/inf Q-values have predictable causes.** Too-large rewards, too-high learning rate, and too-low target update frequency are the usual culprits. Gradient clipping (max_norm=10) is a low-cost defensive measure that prevents single catastrophic updates.

- **The deadly triad (function approximation + bootstrapping + off-policy) is not solved — it is managed.** Experience replay and target networks tame the instability enough to learn successfully on practical problems. Understanding the triad tells you what to monitor: Q-value magnitude, loss trend, and whether the policy is actually improving.

## Quiz

{{#quiz 04-deep-q-networks.toml}}
