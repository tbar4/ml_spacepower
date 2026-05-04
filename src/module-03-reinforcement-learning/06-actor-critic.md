# Lesson 6: Actor-Critic Methods

**Module:** Reinforcement Learning — M03: Sequential Decision-Making
**Source:** *Reinforcement Learning: An Introduction* — Sutton & Barto, Chapters 9 & 13 (Function Approximation and Actor-Critic); *Deep Reinforcement Learning Hands-On* — Lapan, Chapter 10 (Actor-Critic); *Algorithms for Reinforcement Learning* — Szepesvári, Section 4.3 (Actor-Critic Algorithms)

---

## Where this fits

Actor-critic methods combine the strengths of value-based learning (low variance, sample efficiency) and policy gradient methods (direct policy parameterization, support for continuous actions and stochastic policies). They are the architecture used by AlphaZero (Module 4), most modern deep RL (PPO, A3C, SAC), and the intuition matters for deep CFR (Module 5). If you understand REINFORCE with a value baseline (lesson 5), you already have most of actor-critic. This lesson adds the engineering and the standard naming.

## The structure

An actor-critic agent has **two networks** that learn together:

1. **The actor**: a policy network \\(\pi(a \mid s; \theta)\\) that outputs action probabilities. Trained using policy gradient.

2. **The critic**: a value network \\(V(s; \phi)\\) that estimates the value of states. Trained using TD learning (similar to DQN, but for V instead of Q).

The names come from theater: the actor performs (chooses actions); the critic evaluates the performance (estimates value). They learn together, with the critic's evaluations guiding the actor's improvement.

The two networks are usually trained simultaneously, in the same loop.

## The advantage function

Recall from lesson 5 that a baseline can reduce policy gradient variance. The natural baseline is the value function \\(V(s_t)\\), and the resulting quantity \\(G_t - V(s_t)\\) is called the **advantage**:

\\[ A(s_t, a_t) = G_t - V(s_t) \\]

Reading: "how much better was the actual return from this trajectory than the average expected return from this state?"

If A is positive, the trajectory was better than expected: increase the probability of the action that started it. If A is negative, the trajectory was worse than expected: decrease the probability.

The policy gradient with the value baseline becomes:

\\[ \nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t A(s_t, a_t) \nabla_\theta \log \pi(a_t \mid s_t; \theta) \right] \\]

Same structure as REINFORCE, but \\(G_t\\) is replaced by \\(A(s_t, a_t)\\). This dramatically reduces variance because the advantage typically has much smaller magnitude than the raw return.

## Estimating the advantage with the critic

The critic provides \\(V(s_t)\\). For \\(G_t\\), we have a few options:

**Monte Carlo estimate** (full return):
\\[ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\]

This requires waiting for the episode to end. High variance, no bias.

**One-step TD estimate**:
\\[ G_t \approx R_{t+1} + \gamma V(s_{t+1}) \\]

This bootstraps off the critic's estimate of \\(V(s_{t+1})\\). Available immediately after each step. Lower variance, but biased (if V is wrong, this estimate is wrong).

The one-step TD advantage is:

\\[ A(s_t, a_t) \approx R_{t+1} + \gamma V(s_{t+1}) - V(s_t) \\]

This is the same as the TD error from Q-learning (lesson 3), just for V instead of Q. It is sometimes called the "TD error" or "δ" in actor-critic literature.

In between are n-step returns and the Generalized Advantage Estimator (GAE), which trade off bias and variance. We will use the one-step version for simplicity.

## Training the critic

The critic is trained like any value function: minimize the mean squared TD error.

\\[ L(\phi) = \left( R_{t+1} + \gamma V(s_{t+1}; \phi) - V(s_t; \phi) \right)^2 \\]

We want \\(V(s_t)\\) to match the bootstrapped estimate \\(R_{t+1} + \gamma V(s_{t+1})\\). Same MSE loss as DQN, just for V instead of Q. As with DQN, you should use `torch.no_grad()` around the target so gradients only flow through \\(V(s_t)\\), not through the target.

In practice, both the policy update and the critic update happen at every step (or every batch of steps), using the same recently observed transitions.

## A complete actor-critic implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Combined actor-critic network with shared trunk and separate heads.
    Many implementations share the lower layers between actor and critic
    for efficiency; here we do the same.
    """
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden_dim, num_actions)  # outputs logits
        self.critic_head = nn.Linear(hidden_dim, 1)             # outputs V(s)
    
    def forward(self, state):
        features = self.shared(state)
        logits = self.actor_head(features)
        value  = self.critic_head(features).squeeze(-1)
        return logits, value


class ActorCriticAgent:
    def __init__(self, state_dim, num_actions, lr=3e-4, gamma=0.99, 
                 entropy_coef=0.01):
        self.net = ActorCritic(state_dim, num_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
    
    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        logits, value = self.net(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value, entropy
    
    def update(self, log_probs, values, rewards, entropies, dones):
        """
        Process a single episode (or batch). Computes returns,
        advantages, and updates both actor and critic.
        """
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        values    = torch.stack(values)
        entropies = torch.stack(entropies)
        rewards   = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute returns G_t (Monte Carlo, full discounted return)
        returns = []
        G = 0
        for r, done in zip(reversed(rewards.tolist()), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Advantages: A_t = G_t - V(s_t)
        # Detach values when computing advantages so gradients flow only through 
        # the policy gradient term, not through the critic indirectly.
        advantages = returns - values.detach()
        
        # Actor loss: maximize advantage-weighted log probabilities
        # (negative sign for minimization)
        actor_loss = -(log_probs * advantages).sum()
        
        # Critic loss: MSE between V(s_t) and G_t
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy bonus: encourage exploration by rewarding policies
        # with high entropy (more uncertain action distributions)
        entropy_bonus = entropies.sum()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_bonus
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def train_actor_critic(env, agent, num_episodes=500):
    episode_returns = []
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs, values, rewards, entropies, dones = [], [], [], [], []
        
        for step in range(200):
            action, log_prob, value, entropy = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            dones.append(done)
            
            state = next_state
            if done:
                break
        
        agent.update(log_probs, values, rewards, entropies, dones)
        
        total_return = sum(rewards)
        episode_returns.append(total_return)
        
        if episode % 50 == 0:
            recent = episode_returns[-50:]
            avg = sum(recent) / len(recent)
            print(f"Episode {episode}: avg return over last 50 = {avg:.2f}")
    
    return episode_returns
```

## Three things in this loss

The total loss combines three terms:

**1. Actor loss**: drives the policy to take better actions.

**2. Critic loss**: drives V(s) to predict the actual return.

**3. Entropy bonus**: rewards the policy for being more random (higher entropy).

The entropy bonus is the trick from Module 1, lesson 4. By subtracting \\(c \cdot H(\pi)\\) from the loss (which is the same as adding to the reward objective), we encourage the policy to remain stochastic. Without it, the policy quickly concentrates on a single action and stops exploring. The coefficient (0.01 here) is tuned per problem.

## Why combine actor and critic in one network?

In the implementation above, both the actor and critic share the same lower layers (the `shared` MLP) and have separate output heads. This is common practice and has two benefits:

1. **Computational efficiency**: one forward pass produces both the action distribution and the value estimate.

2. **Representation learning**: the shared layers learn features useful for both tasks. Useful state representations should be relevant both for predicting value and for selecting actions.

Some implementations use completely separate networks. Both work; shared trunks are slightly more parameter-efficient.

## Synchronous vs. asynchronous variants

The basic actor-critic above is a synchronous, on-policy algorithm: collect a trajectory, update, repeat. This is sometimes called **A2C** (Advantage Actor-Critic).

**A3C** (Asynchronous Advantage Actor-Critic) was an early influential variant that used multiple parallel agents to collect experience asynchronously, decoupling data collection from training. A3C was largely superseded by A2C running on multiple GPUs.

**PPO** (Proximal Policy Optimization) is the current dominant policy gradient algorithm. It is essentially actor-critic with one additional engineering trick: it constrains how far the policy can change in a single update (using a clipped objective related to the KL divergence from Module 1, lesson 4). PPO is very robust and is what you should reach for in practice. We are not implementing PPO from scratch in this curriculum because the additional bookkeeping does not teach new concepts; we will use PPO via OpenSpiel's built-in implementation in later modules.

## Where actor-critic appears in the rest of the curriculum

**Module 4 (AlphaZero)**: AlphaZero uses an actor-critic-like architecture: a single neural network outputs both a policy (action probabilities) and a value (expected outcome). The policy guides MCTS; the value replaces rollouts. The training objective combines a policy loss (cross-entropy against MCTS-improved policy) and a value loss (MSE against game outcomes).

**Module 5 (deep CFR)**: Deep CFR uses a network to approximate regret values, which serve a similar role to advantage values. The structural similarity to actor-critic (network-driven policy updates with a value-based component) is real.

**Module 6 (PSRO)**: At each iteration of PSRO, you compute best responses using some inner-loop RL algorithm, often actor-critic.

In all cases, the basic structure is: parameterize a policy, parameterize a value function, train them jointly using gradient descent.

## What we cover in the project

The Module 3 project focuses on DQN rather than actor-critic, because DQN is more sample-efficient for the discrete-action SSA scheduling problem and the buffer-based training loop is good practice for the off-policy methods we will use later. Actor-critic shows up properly in Module 4, where it powers AlphaZero. The mental model from this lesson is what you will need.

---

## The advantage function as the critic's output

The previous sections introduced the advantage \\(A(s_t, a_t) = G_t - V(s_t)\\) informally. Let us be precise about what the advantage function measures, why using it rather than raw returns is so important, and how to implement a critic that produces advantage estimates in PyTorch.

### Q, V, and A

There are three related value functions in RL:

\\[ V(s) = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s \right] \\]

\\[ Q(s, a) = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right] \\]

\\[ A(s, a) = Q(s, a) - V(s) \\]

**Decoding:**
- \\(V(s)\\): the state value — expected cumulative reward starting from state \\(s\\), following policy \\(\pi\\). This is a **baseline**: what the agent expects to get from here on average.
- \\(Q(s, a)\\): the action value — expected cumulative reward starting from state \\(s\\), taking action \\(a\\), then following \\(\pi\\). This tells you the value of a specific action from a specific state.
- \\(A(s, a) = Q(s, a) - V(s)\\): the advantage — how much better (or worse) is taking action \\(a\\) compared to the average action the policy would take from state \\(s\\)?

The advantage has two useful properties that raw returns do not:

1. **Zero-centered in expectation**: \\(\mathbb{E}_{a \sim \pi}[A(s, a)] = 0\\) for all \\(s\\). Actions better than average get positive advantage; actions worse than average get negative advantage. This centering reduces gradient variance.

2. **Action-relative**: the advantage isolates *which action was taken* from *where we are*. A return of 500 from a state where the expected return is 490 indicates a slightly-above-average action. A return of 500 from a state where the expected return is 100 indicates a great action. Raw returns confuse these two things; advantages separate them.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ValueNetwork(nn.Module):
    """
    Critic: estimates V(s), the expected return from state s.
    Separate from the actor for clarity.
    """
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # scalar output: V(s)
        )
    
    def forward(self, state):
        return self.net(state).squeeze(-1)


class PolicyNetwork(nn.Module):
    """
    Actor: outputs a distribution over actions.
    """
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, state):
        return self.net(state)  # logits


def compute_advantages(rewards, values, next_values, dones, gamma=0.99):
    """
    Compute advantages using Monte Carlo returns and critic baseline.
    
    Args:
        rewards:      list of floats, one per step
        values:       tensor of V(s_t) estimates from the critic
        next_values:  tensor of V(s_{t+1}) estimates (or 0 at episode end)
        dones:        list of bool, True if this step ends the episode
        gamma:        discount factor
    
    Returns:
        advantages:   A_t = G_t - V(s_t), tensor of shape (T,)
        returns:      G_t, tensor of shape (T,)
    """
    T = len(rewards)
    returns = torch.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            G = 0.0
        G = rewards[t] + gamma * G
        returns[t] = G
    
    # Advantage = return - value baseline
    advantages = returns - values.detach()
    return advantages, returns


# --- Demonstration: advantages vs. raw returns ---
torch.manual_seed(3)

state_dim = 8   # e.g., orbital elements of 5 satellites + time
n_actions = 5   # choose which satellite to observe

critic = ValueNetwork(state_dim=state_dim)
actor  = PolicyNetwork(state_dim=state_dim, num_actions=n_actions)

# Simulate a small episode to see advantage vs. return magnitudes
n_steps = 10
states  = torch.randn(n_steps, state_dim)
rewards = torch.tensor([50., 10., 80., 5., 90., 20., 60., 15., 40., 100.])
dones   = [False] * 9 + [True]

with torch.no_grad():
    values = critic(states)

# Fake next_values: V(s_{t+1}) = V(s_t shifted by one)
next_values = torch.cat([values[1:], torch.zeros(1)])

advantages, returns = compute_advantages(
    rewards.tolist(), values, next_values, dones
)

print("Step-by-step: returns vs. values vs. advantages")
print(f"{'t':>3}  {'reward':>8}  {'V(s_t)':>10}  {'G_t':>10}  {'A_t':>10}")
for t in range(n_steps):
    print(f"{t:>3}  {rewards[t].item():>8.1f}  {values[t].item():>10.3f}  "
          f"{returns[t].item():>10.3f}  {advantages[t].item():>10.3f}")

print(f"\nReturn statistics:    mean={returns.mean().item():.1f}, "
      f"std={returns.std().item():.1f}")
print(f"Advantage statistics: mean={advantages.mean().item():.3f}, "
      f"std={advantages.std().item():.1f}")
print("Advantages have smaller variance relative to returns,")
print("and are zero-mean (approximately) once the critic is trained.")
```

The critic does not directly output \\(A(s, a)\\). It outputs \\(V(s)\\), and the advantage is computed as \\(G_t - V(s_t)\\) (or \\(\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)\\) for the TD version). This is an important distinction: the critic estimates a state-level quantity (independent of which action was taken), while the advantage is computed by comparing the actual trajectory to that baseline.

---

## TD(0) critic vs. Monte Carlo critic

The REINFORCE baseline and simple actor-critic implementations in the previous sections used Monte Carlo returns: wait for the episode to end, compute \\(G_t = \sum_k \gamma^k R_{t+k}\\), use it as the return estimate. This is high variance but unbiased. TD(0) bootstraps after a single step and is low variance but biased. The choice between them is the fundamental bias-variance tradeoff in RL.

### Monte Carlo critic

The MC critic uses the full episodic return to train the value network:

\\[ \text{Target}: y_t^{\text{MC}} = G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\]

\\[ L_{\text{MC}}(\phi) = \frac{1}{T} \sum_t \left( V(s_t; \phi) - G_t \right)^2 \\]

**When to use MC:**
- Episodes are short (the full return is cheap to wait for)
- The value function is very wrong initially (bootstrapping off a bad V creates bias that compounds)
- Rewards are dense and informative throughout the episode
- SSA example: a 10-step satellite observation schedule where each step gives immediate reward

### TD(0) critic

The TD(0) critic bootstraps: use the current value estimate of the next state as part of the target:

\\[ \text{Target}: y_t^{\text{TD}} = R_{t+1} + \gamma V(s_{t+1}; \phi) \\]

\\[ L_{\text{TD}}(\phi) = \frac{1}{T} \sum_t \left( V(s_t; \phi) - \left(R_{t+1} + \gamma V(s_{t+1}; \phi_{\text{target}}) \right) \right)^2 \\]

**When to use TD(0):**
- Episodes are long (waiting for the full return is expensive)
- The value function is reasonably initialized (bootstrapping introduces little bias)
- Online learning (update after every step, not every episode)
- SSA example: a continuous orbital maneuvering task that runs for hundreds of steps

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        return self.net(state).squeeze(-1)


def update_critic_mc(critic, optimizer, states, returns, gamma=0.99):
    """
    Monte Carlo critic update: use full episode returns as targets.
    
    states:   tensor (T, state_dim)
    returns:  tensor (T,) — G_t for each step
    """
    predicted_values = critic(states)
    # MC target: the actual discounted return from each state
    loss = F.mse_loss(predicted_values, returns)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def update_critic_td0(critic, optimizer, states, rewards, next_states,
                      dones, gamma=0.99):
    """
    TD(0) critic update: bootstrap from next state value.
    
    states:       tensor (T, state_dim)
    rewards:      tensor (T,)
    next_states:  tensor (T, state_dim)
    dones:        tensor (T,) — 1.0 if episode ends at step t
    """
    predicted_values = critic(states)
    
    # Compute TD target: r + γ * V(s') (stop gradient on target)
    with torch.no_grad():
        next_values = critic(next_states)
        td_targets = rewards + gamma * next_values * (1.0 - dones)
    
    loss = F.mse_loss(predicted_values, td_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# --- Compare bias-variance tradeoff on a toy value estimation problem ---
torch.manual_seed(11)

state_dim = 4  # simplified orbital state
T = 20         # episode length

# "True" value function: V*(s) = some linear function of state
# We will measure how fast each method converges to it.
true_weights = torch.tensor([1.0, -0.5, 0.3, 0.2])

def true_value(state):
    return (state * true_weights).sum(dim=-1)

# Generate an episode
states = torch.randn(T, state_dim)
true_vals = true_value(states)
# Rewards: correlated with state value changes (simplified)
rewards = true_vals[1:] - true_vals[:-1] * 0.99
rewards = torch.cat([rewards, torch.tensor([0.0])])  # terminal step
dones = torch.zeros(T)
dones[-1] = 1.0

# Compute MC returns
returns = torch.zeros(T)
G = 0.0
for t in reversed(range(T)):
    G = rewards[t].item() + 0.99 * G * (1.0 - dones[t].item())
    returns[t] = G

# Train both critics from scratch for 100 gradient steps
critic_mc = CriticNetwork(state_dim)
critic_td = CriticNetwork(state_dim)
opt_mc = torch.optim.Adam(critic_mc.parameters(), lr=1e-3)
opt_td = torch.optim.Adam(critic_td.parameters(), lr=1e-3)

n_steps = 100
mc_losses  = []
td_losses  = []

for step in range(n_steps):
    mc_loss = update_critic_mc(critic_mc, opt_mc, states, returns)
    td_loss = update_critic_td0(critic_td, opt_td, states, rewards,
                                 torch.cat([states[1:], states[-1:]], dim=0),
                                 dones)
    mc_losses.append(mc_loss)
    td_losses.append(td_loss)

print(f"After {n_steps} updates:")
print(f"  MC critic final loss:   {mc_losses[-1]:.6f}")
print(f"  TD(0) critic final loss: {td_losses[-1]:.6f}")
print(f"\nBias-variance tradeoff summary:")
print(f"  MC:   unbiased (uses real returns), but high variance per episode")
print(f"  TD(0): lower variance per step, but biased if V is initially wrong")
```

**Decoding the `torch.no_grad()` in TD(0):** The TD target \\(R_{t+1} + \gamma V(s_{t+1})\\) involves the critic's own output on the next state. If we allow gradients to flow through \\(V(s_{t+1})\\), the loss becomes a function of both \\(V(s_t)\\) and \\(V(s_{t+1})\\), creating a "chasing your own tail" phenomenon where updates to \\(V(s_t)\\) also shift the target. Using `no_grad()` freezes the target, making it a stable supervised learning problem: fit \\(V(s_t)\\) toward a fixed target, then recompute the target in the next batch.

---

## The n-step return

Between the extremes of TD(0) (1-step bootstrap) and Monte Carlo (full episode), there is a spectrum parameterized by \\(n\\): the n-step return.

\\[ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n V(s_{t+n}) \\]

**Decoding:**
- \\(R_{t+1}, \ldots, R_{t+n}\\): the actual observed rewards for the next \\(n\\) steps. These are not estimated — they are sampled from the real environment.
- \\(\gamma^n V(s_{t+n})\\): the bootstrapped value from the state after \\(n\\) real steps. This is the only estimated component.
- \\(n = 1\\): recovers TD(0). \\(n = T\\) (full episode): recovers Monte Carlo.

The n-step advantage is:

\\[ A_t^{(n)} = G_t^{(n)} - V(s_t) = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n V(s_{t+n}) - V(s_t) \\]

Larger \\(n\\) reduces bias (more real reward signal, less reliance on potentially wrong \\(V\\)) but increases variance (more stochastic reward steps). The sweet spot is typically \\(n \in [3, 10]\\) for most tasks; PPO uses a related approach called Generalized Advantage Estimation (GAE) which is essentially an exponentially-weighted average over all \\(n\\).

```python
import torch

def compute_nstep_returns(rewards, values, dones, n, gamma=0.99):
    """
    Compute n-step returns for all timesteps in an episode.
    
    Args:
        rewards:  list of floats, length T
        values:   tensor of V(s_t) estimates, shape (T,)
        dones:    list of bool, length T
        n:        number of steps to unroll before bootstrapping
        gamma:    discount factor
    
    Returns:
        nstep_returns: tensor of shape (T,)
        nstep_advantages: tensor of shape (T,)
    """
    T = len(rewards)
    nstep_returns = torch.zeros(T)
    
    for t in range(T):
        G = 0.0
        # Accumulate n real steps of reward
        for k in range(n):
            if t + k >= T:
                break
            G += (gamma ** k) * rewards[t + k]
            if dones[t + k]:
                # Episode ended before n steps; no bootstrap needed
                break
        else:
            # We completed all n steps without a terminal state
            # Bootstrap from V(s_{t+n}) if available
            if t + n < T:
                G += (gamma ** n) * values[t + n].item()
            # If t+n >= T, the episode ended; no bootstrapping needed
        
        nstep_returns[t] = G
    
    nstep_advantages = nstep_returns - values.detach()
    return nstep_returns, nstep_advantages


# Demonstrate: how n affects the return estimates
torch.manual_seed(17)

T = 15
rewards = [10., 5., 20., 0., 15., 30., 5., 10., 8., 25., 12., 6., 18., 9., 50.]
dones   = [False] * 14 + [True]
values  = torch.rand(T) * 100  # random "critic" estimates

print(f"Episode rewards: {rewards}")
print(f"\nn-step return comparison (first 5 timesteps):")
print(f"{'n':>4}  {'G_0':>10}  {'G_1':>10}  {'G_2':>10}  {'G_3':>10}  {'G_4':>10}")
for n in [1, 2, 4, 8, 15]:
    rets, _ = compute_nstep_returns(rewards, values, dones, n=n)
    row = "  ".join([f"{rets[t].item():>10.2f}" for t in range(5)])
    print(f"{n:>4}  {row}")

print("\nObservation:")
print("  n=1:  returns are close to V(s) bootstraps — low variance, potentially biased")
print("  n=15: returns are full MC — high variance, no bias from V")
print("  Intermediate n: interpolates between these extremes")

# Variance of advantage estimates across different n
print(f"\nAdvantage std as a function of n:")
for n in [1, 2, 4, 8, 15]:
    _, adv = compute_nstep_returns(rewards, values, dones, n=n)
    print(f"  n={n:>2}: advantage std = {adv.std().item():.3f}")
```

In practice, the n-step return gives you a direct knob on the bias-variance tradeoff. For SSA tasks where the satellite makes observations over a fixed horizon (say, a 24-hour scheduling window), \\(n \approx 5\text{–}10\\) often works well: enough real reward signal to reduce bias, not so many steps that variance explodes.

---

## A2C: Advantage Actor-Critic for satellite sensor scheduling

Now let us put everything together in a complete A2C implementation applied to a realistic SSA scheduling problem: 5 satellites with different observation priorities, and an agent that must decide which satellite to observe at each timestep to maximize total information gathered.

### The SSA scheduling environment

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SatelliteSchedulingEnv:
    """
    SSA Sensor Scheduling Environment.
    
    State:  [time_remaining (normalized), 
             last_obs_time_1, ..., last_obs_time_5,  (time since last observation)
             priority_1, ..., priority_5]             (observation priority, changes slowly)
    Action: choose one of 5 satellites to observe (0–4)
    Reward: priority_i * freshness_i * success_probability
    
    Freshness: decreases the longer since last observation.
    The agent must decide which satellite to observe each timestep,
    balancing high-priority targets with stale data.
    """
    def __init__(self, n_satellites=5, episode_len=20, seed=None):
        self.n_satellites = n_satellites
        self.episode_len  = episode_len
        self.state_dim    = 1 + n_satellites + n_satellites  # time + staleness + priority
        if seed is not None:
            torch.manual_seed(seed)
        self.reset()
    
    def reset(self):
        self.t = 0
        # Observation priorities: fixed per episode, vary across episodes
        self.priorities = torch.rand(self.n_satellites) * 0.9 + 0.1  # [0.1, 1.0]
        # Staleness: time since last observation (starts at 0 = just observed)
        self.staleness = torch.zeros(self.n_satellites)
        return self._get_state()
    
    def _get_state(self):
        time_remaining = torch.tensor([(self.episode_len - self.t) / self.episode_len])
        return torch.cat([time_remaining, self.staleness / self.episode_len,
                          self.priorities])
    
    def step(self, action):
        # Observation success probability: decreases with "cloud cover" randomness
        success = torch.rand(1).item() > 0.2  # 80% success rate
        
        # Freshness reward: higher for fresher observations (less staleness)
        freshness = 1.0 / (1.0 + self.staleness[action].item())
        
        if success:
            reward = self.priorities[action].item() * freshness * 10.0
            self.staleness[action] = 0.0  # reset: just observed
        else:
            reward = -0.5  # small penalty for failed observation (wasted slot)
        
        # All non-observed satellites get more stale
        self.staleness += 1.0
        self.staleness[action] = self.staleness[action] * 0  # reset observed
        
        self.t += 1
        done = (self.t >= self.episode_len)
        next_state = self._get_state()
        return next_state, reward, done


class A2CNetwork(nn.Module):
    """
    Shared backbone with separate actor and critic heads.
    """
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor_head(features)
        value  = self.critic_head(features).squeeze(-1)
        return logits, value
    
    def get_action(self, state):
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action.item(), log_prob, value, entropy


class A2CAgent:
    """
    Advantage Actor-Critic agent.
    Collects full episodes, computes n-step advantages, updates networks.
    """
    def __init__(self, state_dim, n_actions, lr=3e-4, gamma=0.99,
                 n_steps=5, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5):
        self.net           = A2CNetwork(state_dim, n_actions)
        self.optimizer     = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma         = gamma
        self.n_steps       = n_steps
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm
    
    def collect_episode(self, env):
        """Run one full episode and return all transitions."""
        state = env.reset()
        transitions = []
        
        done = False
        while not done:
            state_t  = state if isinstance(state, torch.Tensor) else torch.tensor(
                state, dtype=torch.float32)
            action, log_prob, value, entropy = self.net.get_action(state_t)
            next_state, reward, done = env.step(action)
            
            transitions.append({
                'state':    state_t,
                'action':   action,
                'log_prob': log_prob,
                'value':    value,
                'reward':   reward,
                'done':     done,
                'entropy':  entropy,
            })
            state = next_state
        
        return transitions
    
    def compute_returns_and_advantages(self, transitions):
        """Compute n-step returns and advantages from episode transitions."""
        T = len(transitions)
        rewards = [t['reward'] for t in transitions]
        values  = torch.stack([t['value'] for t in transitions])
        dones   = [t['done'] for t in transitions]
        
        # Compute n-step returns
        returns_list, _ = compute_nstep_returns(
            rewards, values, dones, n=self.n_steps, gamma=self.gamma
        )
        returns_t    = returns_list
        advantages_t = returns_t - values.detach()
        
        # Normalize advantages for training stability
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        return returns_t, advantages_t
    
    def update(self, transitions, returns, advantages):
        """Compute and apply actor + critic + entropy loss."""
        log_probs = torch.stack([t['log_prob'] for t in transitions])
        values    = torch.stack([t['value'] for t in transitions])
        entropies = torch.stack([t['entropy'] for t in transitions])
        
        # Actor loss: policy gradient with advantage weighting
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss: MSE between predicted value and n-step return
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy bonus: prevent premature convergence to deterministic policy
        entropy_loss = -entropies.mean()
        
        total_loss = (actor_loss
                      + self.value_coef  * critic_loss
                      + self.entropy_coef * entropy_loss)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping: prevents explosive updates when advantages are large
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'total_loss':   total_loss.item(),
            'actor_loss':   actor_loss.item(),
            'critic_loss':  critic_loss.item(),
            'entropy':      entropies.mean().item(),
        }
    
    def train(self, env, n_episodes=300, print_every=50):
        episode_returns = []
        
        for ep in range(n_episodes):
            transitions = self.collect_episode(env)
            returns, advantages = self.compute_returns_and_advantages(transitions)
            metrics = self.update(transitions, returns, advantages)
            
            ep_return = sum(t['reward'] for t in transitions)
            episode_returns.append(ep_return)
            
            if (ep + 1) % print_every == 0:
                recent = episode_returns[-print_every:]
                avg = sum(recent) / len(recent)
                print(f"Episode {ep+1:>4}: avg_return={avg:.2f}, "
                      f"entropy={metrics['entropy']:.3f}, "
                      f"critic_loss={metrics['critic_loss']:.4f}")
        
        return episode_returns


# Run the A2C agent on the satellite scheduling task
env   = SatelliteSchedulingEnv(n_satellites=5, episode_len=20, seed=42)
agent = A2CAgent(
    state_dim    = env.state_dim,
    n_actions    = env.n_satellites,
    lr           = 3e-4,
    gamma        = 0.99,
    n_steps      = 5,
    entropy_coef = 0.01,
    value_coef   = 0.5,
)

print("Training A2C on satellite sensor scheduling (5 satellites, 20 steps/episode)")
print("="*65)
returns_history = agent.train(env, n_episodes=200, print_every=50)

# Evaluate final policy
print("\nFinal policy evaluation (10 episodes):")
eval_returns = []
for _ in range(10):
    transitions = agent.collect_episode(env)
    ep_return = sum(t['reward'] for t in transitions)
    eval_returns.append(ep_return)
avg_eval = sum(eval_returns) / len(eval_returns)
print(f"  Average return: {avg_eval:.2f}")
print(f"  Min/Max: {min(eval_returns):.2f} / {max(eval_returns):.2f}")
```

### SSA reward design discussion

The reward function in the environment above encodes several real SSA considerations:

**Priority weighting** (`priorities[action]`): Different RSOs have different importance. A high-inclination, large LEO object passing over many populated areas deserves more observation time than a defunct satellite in a quiet GEO slot. The agent should learn to preferentially observe high-priority targets.

**Freshness decay** (`1.0 / (1.0 + staleness)`): Data freshness matters. An orbit determination that was last updated three days ago has large uncertainty; one updated an hour ago has small uncertainty. Observing a satellite that was just observed is wasteful; observing one with stale data is valuable. This term pushes the agent toward round-robin strategies with priority weighting.

**Observation success probability**: Ground-based optical sensors have cloud cover, atmospheric seeing, and solar illumination constraints. The 80% success rate is a simplification; real systems model these factors per site, per pass, per time of day.

**Wasted slot penalty**: A failed observation is not neutral — it consumes a resource (telescope time, uplink window) that could have been given to another target. The −0.5 penalty for failure teaches the agent to account for sensor reliability when scheduling.

---

## Common failure modes

Actor-critic training has several failure modes that do not appear in simpler value-based methods. Understanding them is essential for debugging.

### Failure mode 1: The actor learns too fast relative to the critic

The most common failure mode. If the actor updates too aggressively, it changes the policy faster than the critic can track. The critic's value estimates \\(V(s_t)\\) then reflect the old policy, not the current one. The advantage estimates become wrong — sometimes wildly so — and the actor receives garbage gradient signals.

**Symptoms**: policy entropy drops sharply and early, then performance plateaus or collapses. Loss curves show critic loss spiking repeatedly.

**Fixes:**
- Use a lower actor learning rate (or separate learning rates for actor and critic)
- Increase the `value_coef` to prioritize critic convergence
- Use larger batches to reduce gradient noise in the actor
- Add a trust-region constraint (the approach PPO takes)

```python
import torch
import torch.nn as nn

# Example: separate learning rates for actor and critic
class SeparateLRActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)
    
    def forward(self, state):
        f = self.backbone(state)
        return self.actor_head(f), self.critic_head(f).squeeze(-1)

def make_optimizer_with_separate_lrs(model, actor_lr=1e-4, critic_lr=3e-4):
    """
    Give the critic a higher learning rate so it tracks the policy better.
    When actor and critic share a backbone, backbone uses actor LR (conservative).
    """
    return torch.optim.Adam([
        {'params': model.backbone.parameters(),    'lr': actor_lr},
        {'params': model.actor_head.parameters(),  'lr': actor_lr},
        {'params': model.critic_head.parameters(), 'lr': critic_lr},
    ])
```

### Failure mode 2: Hyperparameter sensitivity

Actor-critic is more sensitive to hyperparameters than DQN because the policy and value function interact during learning. The following parameters interact:

| Hyperparameter | Effect if too high | Effect if too low |
|----------------|-------------------|-------------------|
| `lr` (learning rate) | Unstable, oscillating loss | Slow convergence |
| `entropy_coef` | Policy stays too random (low reward) | Policy collapses to deterministic too early |
| `value_coef` | Critic dominates, slow actor improvement | Actor receives noisy, inaccurate advantages |
| `n_steps` (n-step return) | High variance advantages | Biased advantages if critic is wrong |
| `gamma` | Myopic (ignores long-term return) | Exploding/vanishing values for long episodes |

The most reliable starting configuration for a new SSA task:
- `lr = 3e-4` (Adam)
- `entropy_coef = 0.01` (small but nonzero)
- `value_coef = 0.5` (standard)
- `n_steps = 5` (balance bias/variance)
- `max_grad_norm = 0.5` (gradient clipping)

### Failure mode 3: Entropy collapse and premature convergence

Without an entropy bonus, actor-critic policies almost always converge to a near-deterministic policy within a few hundred episodes. This is not because the agent has found the optimal policy — it is because the policy gradient updates continuously increase the probability of whatever actions have positive advantage, eventually pushing all probability mass onto a small subset of actions.

Once the policy collapses to near-deterministic, exploration stops. Any suboptimal deterministic policy will stay there indefinitely because the agent never tries the actions it has abandoned.

```python
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

def monitor_entropy_collapse(logits_history):
    """
    Given a list of logit tensors (one per training step),
    compute and display entropy to detect premature convergence.
    """
    print("Monitoring entropy over training (should stay nonzero):")
    print(f"{'Step':>6}  {'Entropy':>10}  {'Max prob':>10}  {'Status'}")
    for step, logits in enumerate(logits_history):
        dist = Categorical(logits=logits)
        H = dist.entropy().item()
        max_p = F.softmax(logits, dim=-1).max().item()
        status = "OK" if H > 0.3 else ("WARN: low" if H > 0.05 else "FAIL: collapsed")
        if step % (len(logits_history) // 5) == 0 or step == len(logits_history) - 1:
            print(f"{step:>6}  {H:>10.4f}  {max_p:>10.4f}  {status}")

# Simulate entropy collapse (no entropy bonus)
torch.manual_seed(5)
n_actions = 5
logits = torch.zeros(n_actions)
logits_history = [logits.clone()]
# Simulate a policy gradient update that keeps increasing one action's probability
for step in range(50):
    grad = torch.zeros(n_actions)
    grad[2] = 0.2  # keep reinforcing action 2 (simulating positive advantage)
    logits = logits + grad
    logits_history.append(logits.clone())

print("Without entropy bonus (collapses to greedy):")
monitor_entropy_collapse(logits_history[::10])

# With entropy bonus: gradient is modified by entropy term
logits = torch.zeros(n_actions)
logits_history_ent = [logits.clone()]
entropy_coef = 0.1
for step in range(50):
    grad = torch.zeros(n_actions)
    grad[2] = 0.2
    # Entropy gradient pushes back toward uniform distribution
    probs = F.softmax(logits, dim=-1)
    entropy_grad = -(torch.log(probs + 1e-8) + 1.0)
    logits = logits + grad + entropy_coef * entropy_grad
    logits_history_ent.append(logits.clone())

print("\nWith entropy bonus (maintains exploration):")
monitor_entropy_collapse(logits_history_ent[::10])
```

The SSA scheduling implication: a collapsed policy might learn to always observe satellite 0 (the highest priority) and never explore other satellites. It misses the compound benefit of occasionally observing lower-priority but highly-stale satellites, which prevents conjunction surprises from objects the agent has not checked recently.

---

## Key Takeaways

- **The advantage function** \\(A(s, a) = Q(s, a) - V(s)\\) measures how much better a specific action is compared to the average action from that state. It has lower variance than raw returns (smaller magnitude, zero-mean in expectation once the critic converges) and naturally separates the quality of the action from the quality of the state.
- **TD(0) bootstraps** from the next state value (low variance, biased by critic quality), while **Monte Carlo** uses the full episode return (high variance, unbiased). The n-step return generalizes both, with \\(n\\) as a hyperparameter controlling the bias-variance tradeoff — values \\(n \in [3, 10]\\) are typically a good middle ground for SSA scheduling tasks.
- **A2C** (Advantage Actor-Critic) is the synchronous actor-critic baseline: collect an episode, compute n-step advantages using the critic, update both actor (policy gradient) and critic (MSE loss) jointly with gradient clipping and an entropy bonus. It is the conceptual foundation for PPO, A3C, and SAC.
- **The actor and critic interact during learning**: if the actor changes too fast, the critic's value estimates become stale and advantages become wrong. Use a higher learning rate for the critic, gradient clipping (typically `max_norm=0.5`), and conservative actor updates to keep them in sync.
- **Entropy collapse is a silent failure mode**: without an entropy bonus, actor-critic policies converge to near-deterministic within hundreds of episodes. In SSA scheduling, this produces an agent that obsessively tasks high-priority satellites while letting lower-priority objects go unobserved — missing important conjunction events. Keep `entropy_coef` nonzero and monitor policy entropy during training.
- **SSA reward design encodes domain knowledge**: freshness decay pushes toward round-robin coverage, priority weighting concentrates resources on high-value targets, and failure penalties account for sensor reliability. The agent learns the right balance automatically, but the reward function must encode the right tradeoffs — garbage reward, garbage policy.

---

## Quiz

{{#quiz 06-actor-critic.toml}}
