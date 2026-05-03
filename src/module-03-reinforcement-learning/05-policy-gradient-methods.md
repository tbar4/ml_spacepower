# Lesson 5: Policy Gradient Methods

## Where this fits

Q-learning and DQN are **value-based**: they learn a value function and derive a policy from it (greedy with respect to Q). This works well, but has limitations:
- The greedy policy is deterministic; getting stochastic policies requires hacks like ε-greedy
- It needs a max over actions, which is awkward for continuous action spaces
- It cannot directly optimize the policy's parameters; you have to optimize Q and hope the implied policy is good

**Policy gradient methods** take a fundamentally different approach: parameterize the policy directly with a neural network, and use gradient descent to make it better. The agent learns the policy itself, not a value function from which a policy is derived.

This approach has its own tradeoffs but is essential for the algorithms we will see later. AlphaZero (Module 4) uses a policy network. Most modern RL (PPO, SAC) uses policy gradient methods. CFR (Module 5) updates strategies in a way that has the same flavor as policy gradients. Understanding the gradient of expected return with respect to policy parameters is the foundation.

## The core idea

Suppose your policy is a neural network with parameters θ. The output is a probability distribution over actions: \\(\pi(a \mid s; \theta)\\) is the probability of taking action a in state s, computed by passing s through the network.

The agent's objective is to maximize the expected return:

\\[ J(\theta) = \mathbb{E}\left[ \sum_t \gamma^t R_t \right] \\]

This expectation is over all the sources of randomness: the policy's action selection, the environment's stochastic transitions, and the random rewards. \\(J(\theta)\\) is a function of the policy parameters: different policies produce different expected returns.

We want to make \\(J(\theta)\\) larger. So we use gradient ascent: compute \\(\nabla_\theta J(\theta)\\) (the gradient of expected return with respect to the policy parameters) and step the parameters in the positive direction.

\\[ \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta) \\]

This is gradient ascent (note the +, not - as in gradient descent). The algorithm is the same; the sign just flips because we are maximizing instead of minimizing.

The hard part is computing \\(\nabla_\theta J(\theta)\\). The expectation is over all possible trajectories the agent might take. Direct computation is intractable. We need an estimator we can compute from samples.

## The score function estimator (REINFORCE)

Here is the magic trick that makes policy gradient methods work. The gradient of expected return turns out to have a particularly clean form:

\\[ \nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t G_t \nabla_\theta \log \pi(a_t \mid s_t; \theta) \right] \\]

Where \\(G_t\\) is the return from time t onward (the cumulative discounted reward from t to the end of the episode).

**Decoding:**
- \\(G_t\\): the total discounted return from time t onward, summed over the rest of the episode
- \\(\log \pi(a_t \mid s_t; \theta)\\): the log of the probability the policy assigned to the action it actually took
- \\(\nabla_\theta\\): gradient with respect to the policy parameters

This formula has a beautiful interpretation. To increase expected return:
- For actions that led to high return (large \\(G_t\\)), increase their log-probability
- For actions that led to low or negative return, decrease their log-probability

Each transition contributes a "policy gradient direction" that is the gradient of its log-probability, scaled by how much return that action contributed to.

The proof of this formula relies on a calculus trick called the "log-derivative trick" (\\(\nabla \log f = \nabla f / f\\)). You do not need to derive it. What you need to know is that this is the formula and it gives an unbiased estimator of \\(\nabla J\\).

### The Monte Carlo estimator

Since the formula is an expectation, we can estimate it by sampling: run an episode, compute \\(G_t\\) for each step, and form the empirical average:

\\[ \hat{\nabla} J(\theta) = \sum_t G_t \nabla_\theta \log \pi(a_t \mid s_t; \theta) \\]

For one episode, drop the average and use the sum directly. For multiple episodes, average across episodes.

This is the **REINFORCE algorithm** (also called the score function estimator, the likelihood ratio method, or vanilla policy gradient). It is the simplest policy gradient method.

## A complete REINFORCE implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, state):
        # Output logits; convert to probabilities with softmax
        logits = self.net(state)
        return logits

class REINFORCEAgent:
    def __init__(self, state_dim, num_actions, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.num_actions = num_actions
    
    def select_action(self, state):
        """Sample an action from the policy and return both the action and its log-probability."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits = self.policy(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, log_probs, returns):
        """
        log_probs: list of log-probabilities of actions taken (one per timestep)
        returns: list of G_t values (one per timestep)
        """
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        returns   = torch.tensor(returns, dtype=torch.float32)
        
        # The "loss" we minimize is -G_t * log π(a_t | s_t).
        # Minimizing this is equivalent to maximizing G_t * log π.
        # PyTorch does gradient descent on the loss, which (with the negative sign)
        # is equivalent to gradient ascent on the policy gradient objective.
        loss = -(log_probs * returns).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def train_reinforce(env, agent, num_episodes=500):
    episode_returns = []
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs   = []
        rewards     = []
        
        # Run one full episode
        for step in range(200):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # Compute returns G_t for each step (working backwards)
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + agent.gamma * G
            returns.insert(0, G)
        
        # Update policy
        agent.update(log_probs, returns)
        
        total_return = sum(rewards)
        episode_returns.append(total_return)
        
        if episode % 50 == 0:
            recent = episode_returns[-50:]
            avg = sum(recent) / len(recent)
            print(f"Episode {episode}: avg return over last 50 = {avg:.2f}")
    
    return episode_returns
```

The key parts:

**`Categorical(logits=logits)`**: PyTorch's distribution class. Pass logits and it handles the softmax internally. `dist.sample()` samples an action; `dist.log_prob(action)` returns the log probability of that action under the current policy parameters.

**Computing returns backwards**: \\(G_t = R_{t+1} + \gamma G_{t+1}\\). Starting from the end (where \\(G_T = 0\\)), work backwards through the episode. This is computationally efficient.

**The loss**: `-(log_probs * returns).sum()`. The negative sign converts gradient ascent (on the objective) into gradient descent (on the negated objective), which is what PyTorch optimizers do.

## Why is this called the "score function" estimator?

The term comes from statistics. The "score" of a probability distribution is the gradient of its log-likelihood:

\\[ \text{score}(\theta) = \nabla_\theta \log p(x; \theta) \\]

In our case, the score is \\(\nabla_\theta \log \pi(a \mid s; \theta)\\): how much does changing the policy parameters change the log-probability of the action we took? The estimator weights each score by the return achieved.

You will sometimes see this called the "REINFORCE trick" or the "log-derivative trick" or the "likelihood ratio estimator." All the same thing.

## High variance: the central problem

REINFORCE has a serious problem: the gradient estimates have very high variance.

Why? Because the return \\(G_t\\) can be very different across episodes, depending on which actions were taken (chance) and which environments were sampled. One episode might give G = 100; the next might give G = -50. The policy gradient updates are scaled by these G values, so they swing wildly.

High variance means slow learning: many of your gradient steps point in directions that are mostly noise. You need many samples to average out the noise enough to make consistent progress.

There are several variance reduction techniques. The most important one is **baseline subtraction**.

## Baseline subtraction

Subtract a baseline \\(b(s_t)\\) from \\(G_t\\) before using it in the policy gradient:

\\[ \nabla_\theta J = \mathbb{E}\left[ \sum_t (G_t - b(s_t)) \nabla_\theta \log \pi(a_t \mid s_t; \theta) \right] \\]

This is mathematically valid: subtracting any function of the state (one that does not depend on the action) does not change the expected gradient. (The proof uses the fact that the expectation of \\(\nabla \log \pi(a | s)\\) over the policy distribution is zero, so subtracting a state-only constant does not bias the estimator.) But it can drastically reduce variance.

A natural choice for the baseline is the value function \\(V(s_t)\\): the expected return from state \\(s_t\\). The quantity \\(G_t - V(s_t)\\) is called the **advantage**: how much better was this trajectory than what we would expect on average from this state?

This leads naturally to actor-critic methods (next lesson), where we maintain both a policy network (the "actor") and a value network (the "critic" providing the baseline).

## Comparison: value-based vs. policy-based

| Aspect | Value-based (Q-learning, DQN) | Policy-based (REINFORCE) |
|--------|-------------------------------|--------------------------|
| What it learns | Q(s, a) | π(a \| s) |
| Action space | Discrete (max needed) | Discrete or continuous |
| Stochastic policy | No (greedy is deterministic) | Yes (samples from π) |
| Sample efficiency | Higher (uses replay buffer, off-policy) | Lower (on-policy: each sample used once) |
| Variance | Generally lower | Generally higher (without baselines) |
| Stability | Can diverge with function approximation | More stable but slower |

Both have their place. Modern algorithms (PPO, SAC, A3C) often combine ideas from both approaches.

## When policy gradient methods are preferred

- **Continuous action spaces**: parameterize the policy as outputting parameters of a continuous distribution (mean and variance of a Gaussian, for example), then sample from it.
- **You need a stochastic policy**: in game theory, mixed strategies are often optimal. Value-based methods cannot represent these directly.
- **Direct policy improvement**: if you know what makes a policy good (some performance metric), it is conceptually cleaner to optimize the policy parameters directly.
- **Combining with planning**: AlphaZero uses a policy network to guide tree search. The network outputs action probabilities directly.

For our SSA-flavored problems, REINFORCE alone would be too noisy and sample-inefficient to compete with DQN. But REINFORCE introduces concepts (policy networks, log-probability gradients, returns) that are foundational for actor-critic and AlphaZero.

## Quiz

{{#quiz 05-policy-gradient-methods.toml}}
