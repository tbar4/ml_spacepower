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

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)
```

This solves Problem 1: the random sample from the buffer is much closer to i.i.d. than the sequence of recent transitions. It also has a side benefit: each experience can be reused for many training updates, making sample-efficient use of the agent's interactions.

Capacity is typically 10,000 to 1,000,000 transitions. Batch size for training is typically 32 to 256.

## Trick 2: Target networks

Maintain a second network with the same architecture, called the **target network**, with parameters \\(\theta^-\\) (theta-minus, the convention). The target network's parameters are kept frozen, updated only periodically (every N training steps) to match the main network.

The TD target is computed using the target network:

\\[ \text{target} = r + \gamma \max_{a'} Q(s', a'; \theta^-) \\]

The loss is still computed using the main network:

\\[ L(\theta) = \left( Q(s, a; \theta) - \text{target} \right)^2 \\]

This solves Problem 2: the target is now stable for a stretch of training updates (it changes only every N steps when we sync the target network to match the main network). The main network can train against this stable target without chasing a moving goal.

A typical sync frequency is every 1,000 to 10,000 training steps. Some variants use a "soft update": gradually blend the target network toward the main network at each step (\\(\theta^- \leftarrow (1 - \tau) \theta^- + \tau \theta\\) with small \\(\tau\\) like 0.005).

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

## Key hyperparameters and what they do

- **Learning rate (lr)**: standard neural network learning rate. Typical: 1e-3 to 1e-4.
- **Discount factor (gamma)**: as in MDP. Typical: 0.99 for problems with long horizons.
- **Epsilon**: exploration rate. Often annealed from 1.0 down to 0.05 over training.
- **Buffer capacity**: how many transitions to remember. Typical: 10,000 to 1,000,000.
- **Batch size**: how many transitions to sample per training step. Typical: 32 to 256.
- **Target update frequency**: how often to sync the target network. Typical: every 500 to 10,000 steps.
- **Update frequency**: how often to train. Often every step, but can be every N steps to make the agent collect more data per update.

DQN is sensitive to these hyperparameters. The defaults work for many problems but tuning is sometimes necessary.

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

## Quiz

{{#quiz 04-deep-q-networks.toml}}
