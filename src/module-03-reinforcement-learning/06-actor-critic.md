# Lesson 6: Actor-Critic Methods

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

## Quiz

{{#quiz 06-actor-critic.toml}}
