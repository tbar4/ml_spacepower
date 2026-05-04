# Lesson 5: Policy Gradient Methods

**Module:** Reinforcement Learning — M03: Sequential Decision-Making
**Source:** *Reinforcement Learning: An Introduction* — Sutton & Barto, Chapter 13 (Policy Gradient Methods); *Deep Learning* — Goodfellow, Bengio & Courville, Chapter 20 (Deep Generative Models, score function estimator); *Algorithms for Reinforcement Learning* — Szepesvári, Chapter 4 (Policy Search)

---

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

---

## The advantage of continuous action spaces

One of the most compelling reasons to use policy gradients over DQN is their natural handling of **continuous action spaces**. DQN requires computing \\(\arg\max_a Q(s, a)\\) — a discrete search over all possible actions. With 5 discrete actions that is trivial; with an infinite continuous action space it is intractable.

Policy gradients sidestep this entirely. Instead of learning Q-values and deriving a policy, we directly parameterize the policy as a probability distribution. For continuous actions, that distribution is typically a **multivariate Gaussian**: the network outputs a mean vector and a standard deviation vector, and actions are sampled from that Gaussian.

### SSA example: satellite delta-v maneuver

Consider a satellite orbit-raising maneuver. The satellite must decide on a delta-v vector at each thrust opportunity — a continuous 3D vector \\((\Delta v_x, \Delta v_y, \Delta v_z)\\) in the RTN (radial-tangential-normal) frame. DQN would require discretizing this space — say, 10 values per axis — giving 1,000 discrete actions, each requiring a separate Q-value output from the network. Policy gradients make this a single forward pass producing six scalars: three means and three standard deviations.

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class ContinuousThrustPolicy(nn.Module):
    """
    Policy network for satellite delta-v maneuver decisions.
    Input: orbital state (position + velocity in some representation)
    Output: distribution over delta-v vector (RTN frame, km/s)
    """
    def __init__(self, state_dim=6, action_dim=3, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Mean head: unbounded, represents the center of the thrust distribution
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Log-std head: learn log(std) instead of std directly for numerical stability
        # Initialize to produce small, conservative maneuvers at the start
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_head(features)
        # Clamp log_std to avoid collapse (too small) or explosion (too large)
        log_std = self.log_std_head(features).clamp(-4.0, 0.5)
        std = torch.exp(log_std)
        return mean, std
    
    def get_action(self, state):
        """Sample a delta-v action and return log-probability."""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()  # rsample allows gradients to flow through the sample
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dimensions
        return action, log_prob, mean, std


class ContinuousREINFORCE:
    def __init__(self, state_dim=6, action_dim=3, lr=3e-4, gamma=0.99):
        self.policy = ContinuousThrustPolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        action, log_prob, mean, std = self.policy.get_action(state_t)
        return action.detach().numpy(), log_prob, mean.detach(), std.detach()
    
    def update(self, log_probs, returns):
        log_probs = torch.stack(log_probs)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stability (explained in detail below)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = -(log_probs * returns_t).sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()


# Demonstrate: why Normal is the right distribution for thrust
torch.manual_seed(42)
policy = ContinuousThrustPolicy(state_dim=6, action_dim=3)

# Simulate orbital state (position components + velocity components, normalized)
state = torch.randn(6)
mean, std = policy(state)

print("Policy output for a random orbital state:")
print(f"  Mean delta-v (RTN, km/s): {mean.detach().numpy()}")
print(f"  Std  delta-v (RTN, km/s): {std.detach().numpy()}")

dist = Normal(mean, std)
action_sample = dist.rsample()
log_prob = dist.log_prob(action_sample).sum()
print(f"  Sampled delta-v:          {action_sample.detach().numpy()}")
print(f"  Log-probability of sample: {log_prob.item():.4f}")

# Contrast with DQN discrete approximation:
# If we discretize each axis into 10 values → 10^3 = 1000 discrete actions
# Each needs a Q-value output head entry. And we lose precision between grid points.
print(f"\nDQN discrete approximation:")
print(f"  With 10 bins per axis: 1000 discrete actions")
print(f"  With 20 bins per axis: 8000 discrete actions")
print(f"  Continuous policy: 1 forward pass, exact sampling, no discretization error")
```

**Decoding the key differences from the discrete case:**

- \\(\text{Normal}(\mu, \sigma)\\): a Gaussian distribution parameterized by mean \\(\mu\\) and standard deviation \\(\sigma\\). We use `Normal` from `torch.distributions` which handles log-probability computation automatically.
- `dist.rsample()`: the "reparameterization sample." Unlike `dist.sample()`, this version lets gradients flow through the sampling operation by writing the sample as \\(a = \mu + \sigma \cdot \epsilon\\) where \\(\epsilon \sim \mathcal{N}(0, 1)\\). Essential for certain policy gradient variants.
- `log_prob(action).sum(dim=-1)`: for a multivariate action, the log-probability of the full action vector is the sum of log-probabilities along each dimension (since dimensions are independent in a diagonal Gaussian).
- `log_std` instead of `std`: learning the log of the standard deviation prevents the network from producing negative values and stabilizes training. The clamp keeps exploration alive but bounded.

The `clamp(-4, 0.5)` on log_std is a practical engineering detail: `exp(-4) ≈ 0.018` (very precise, small maneuvers) and `exp(0.5) ≈ 1.65` (aggressive, exploratory maneuvers). This range covers the sensible operating regime for a satellite that needs to both explore and refine its strategy.

---

## REINFORCE variance analysis

The core weakness of REINFORCE is **high variance** in the gradient estimates. Understanding why — and quantifying how much — is important for knowing when REINFORCE is sufficient and when you need actor-critic or PPO.

### Why variance is high

The return \\(G_t\\) on which each gradient update is scaled varies enormously across episodes. Consider a satellite sensor scheduling agent: in one episode it happens to observe the most important RSO early (large positive reward), while in another episode it misses all priority targets (near-zero reward). The gradient for the same action might be scaled by G = +800 in one episode and G = +5 in another — a ratio of 160:1. When the policy updates by \\(\alpha G_t \nabla \log \pi\\), the update magnitude swings wildly.

Formally, the variance of the REINFORCE gradient estimator scales as \\(\text{Var}(G_t)\\). The standard error of the gradient estimate from K episodes is:

\\[ \text{SE}(\hat{\nabla} J) \approx \frac{\text{Std}(G)}{\sqrt{K}} \\]

**Decoding:**
- \\(\text{Std}(G)\\): the standard deviation of episode returns. If returns range from 0 to 1000, this is on the order of hundreds.
- \\(\sqrt{K}\\): the number of episodes helps, but only as a square root. To cut the error in half, you need four times the episodes.
- The ratio \\(\text{Std}(G) / \sqrt{K}\\): tells you how noisy your gradient estimate is. When this is large relative to the true gradient signal, most update steps point in unhelpful directions.

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

torch.manual_seed(0)

# Simulate the distribution of REINFORCE gradient estimates
# for a simple SSA scheduling problem.
# We will approximate the variance by running 50 "episodes" and observing
# how much the episode return varies.

def simulate_ssa_episode(policy_logits, n_satellites=5, n_timesteps=10):
    """
    Toy SSA scheduling simulation.
    At each step, agent chooses which of 5 satellites to task.
    Reward: random (satellite priority * observation quality).
    This is simplified to show variance, not a real environment.
    """
    satellite_priorities = torch.tensor([0.9, 0.3, 0.7, 0.5, 0.1])
    total_reward = 0.0
    log_probs = []
    
    dist = Categorical(logits=policy_logits)
    for t in range(n_timesteps):
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        # Stochastic reward: priority * random observation quality
        obs_quality = torch.rand(1).item()
        reward = satellite_priorities[action].item() * obs_quality * 100
        total_reward += reward
    
    return total_reward, log_probs

# Fixed policy logits (uniform-ish: slight preference for satellite 0)
policy_logits = torch.tensor([0.5, 0.0, 0.2, 0.1, -0.2])

n_episodes = 50
episode_returns = []
for _ in range(n_episodes):
    ret, _ = simulate_ssa_episode(policy_logits)
    episode_returns.append(ret)

returns_t = torch.tensor(episode_returns)
mean_return = returns_t.mean().item()
std_return  = returns_t.std().item()
min_return  = returns_t.min().item()
max_return  = returns_t.max().item()

print("REINFORCE return distribution over 50 episodes:")
print(f"  Mean:  {mean_return:.1f}")
print(f"  Std:   {std_return:.1f}")
print(f"  Min:   {min_return:.1f}")
print(f"  Max:   {max_return:.1f}")
print(f"  Coefficient of variation (Std/Mean): {std_return/mean_return:.2%}")

# Standard error of gradient estimate decreases as 1/sqrt(K)
print(f"\nGradient SE for K episodes (proportional to Std/sqrt(K)):")
for K in [1, 5, 10, 50, 100, 500]:
    se = std_return / (K ** 0.5)
    print(f"  K={K:>4}: SE ≈ {se:.1f}  (need {K} episodes per update)")

# Show that averaging over more episodes reduces gradient noise
print(f"\nPractical implication:")
print(f"  To reduce gradient SE below 10% of mean return ({0.1*mean_return:.1f}),")
n_needed = int((std_return / (0.1 * mean_return)) ** 2) + 1
print(f"  need approximately {n_needed} episodes per gradient update.")
print(f"  That is {n_needed} full environment rollouts before each parameter update.")
```

The coefficient of variation (Std/Mean) tells you what fraction of the mean return the typical episode deviates by. Values above 50% indicate severe variance — the gradient estimates are mostly noise. This is the regime where REINFORCE struggles and baselines or actor-critic methods are necessary.

### The 1/sqrt(N) averaging argument

When you average gradient estimates over \\(K\\) episodes, the standard error of the average shrinks as \\(1/\sqrt{K}\\). This is the same Central Limit Theorem convergence from Module 1. Intuitively: some episodes have returns above the mean (pushing the gradient estimate positive) and some are below (pushing negative), and they partially cancel.

The problem is that \\(\text{Std}(G)\\) is typically large — potentially hundreds of reward units — while the true gradient signal might be small. The signal-to-noise ratio is low, so even after averaging over many episodes, the noisy component dominates. Baseline subtraction reduces \\(\text{Std}(G_t - b)\\) directly, which is a more effective lever than increasing K.

---

## Why the baseline does not bias the gradient

The claim in the baseline subtraction section is strong: "subtracting any function of the state does not change the expected gradient." Let us see why, and then verify empirically.

### The mathematical proof sketch

We want to show that for any state-dependent function \\(b(s_t)\\):

\\[ \mathbb{E}\left[ b(s_t) \nabla_\theta \log \pi(a_t \mid s_t; \theta) \right] = 0 \\]

If this is zero, subtracting \\(b(s_t)\\) from \\(G_t\\) does not change the expected gradient.

**Proof:**

Fix state \\(s_t\\). Since \\(b(s_t)\\) does not depend on \\(a_t\\), we can factor it out of the expectation over actions:

\\[ \mathbb{E}_{a_t \sim \pi(\cdot | s_t)}\left[ b(s_t) \nabla_\theta \log \pi(a_t \mid s_t; \theta) \right] = b(s_t) \cdot \mathbb{E}_{a_t \sim \pi(\cdot | s_t)}\left[ \nabla_\theta \log \pi(a_t \mid s_t; \theta) \right] \\]

Now, the inner expectation:

\\[ \mathbb{E}_{a \sim \pi}\left[ \nabla_\theta \log \pi(a \mid s; \theta) \right] = \sum_a \pi(a \mid s) \cdot \frac{\nabla_\theta \pi(a \mid s)}{\pi(a \mid s)} = \sum_a \nabla_\theta \pi(a \mid s) = \nabla_\theta \underbrace{\sum_a \pi(a \mid s)}_{= 1} = \nabla_\theta 1 = 0 \\]

**Decoding:**
- \\(\nabla_\theta \log \pi = \nabla_\theta \pi / \pi\\): the log-derivative trick, applied in reverse
- \\(\sum_a \pi(a | s) = 1\\): probabilities sum to one, always
- \\(\nabla_\theta 1 = 0\\): gradient of a constant is zero

The entire expression collapses to \\(b(s_t) \cdot 0 = 0\\). This holds for any baseline \\(b(s_t)\\) that does not depend on the action — a constant, the mean return, or the value function \\(V(s_t)\\).

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

torch.manual_seed(7)

# Demonstrate empirically: adding a constant baseline changes no gradient direction
# but reduces variance significantly.

class TinyPolicy(nn.Module):
    def __init__(self, n_actions=5):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_actions))
    
    def forward(self):
        return self.logits

def estimate_policy_gradient(policy, n_episodes=200, baseline=0.0):
    """
    Estimate the policy gradient using REINFORCE.
    Returns gradient estimates (one per episode) for the first logit.
    """
    satellite_priorities = torch.tensor([0.9, 0.3, 0.7, 0.5, 0.1])
    grad_estimates = []
    
    for _ in range(n_episodes):
        logits = policy()
        dist = Categorical(logits=logits)
        # Single-step "episode" for clarity
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Stochastic return
        ret = satellite_priorities[action].item() * (50 + 50 * torch.rand(1).item())
        
        # Gradient estimate scaled by (return - baseline)
        advantage = ret - baseline
        # This is the gradient contribution from this sample
        # We approximate it by the value of (advantage * log_prob)
        grad_estimate = advantage * log_prob.item()
        grad_estimates.append(grad_estimate)
    
    return torch.tensor(grad_estimates)

policy = TinyPolicy()

# No baseline
grads_no_baseline = estimate_policy_gradient(policy, n_episodes=500, baseline=0.0)

# Constant baseline = mean return (a reasonable estimate of E[G])
mean_return_estimate = 50 * 0.9 * 0.5  # rough expected return for best satellite
grads_with_baseline = estimate_policy_gradient(policy, n_episodes=500,
                                                baseline=mean_return_estimate)

print("Gradient estimates — no baseline:")
print(f"  Mean:  {grads_no_baseline.mean().item():.4f}")
print(f"  Std:   {grads_no_baseline.std().item():.4f}")

print("\nGradient estimates — constant baseline subtracted:")
print(f"  Mean:  {grads_with_baseline.mean().item():.4f}")
print(f"  Std:   {grads_with_baseline.std().item():.4f}")

print(f"\nVariance reduction factor: {grads_no_baseline.std().item() / grads_with_baseline.std().item():.2f}x")
print("The means are similar (same expected gradient, unbiased),")
print("but the baseline version has lower variance (less noise per estimate).")
```

The key observation: the **means** of the two gradient estimates should be approximately equal — both are unbiased estimators of the true gradient. But the **standard deviations** differ substantially. The baseline version concentrates gradient estimates around their mean, so each update step contains more signal and less noise. This is the entire point of variance reduction in policy gradients.

---

## Normalized returns

A practical trick that improves training stability is **return normalization**: before using returns \\(G_t\\) to scale gradient updates, subtract their mean and divide by their standard deviation.

\\[ \tilde{G}_t = \frac{G_t - \bar{G}}{\sigma_G + \epsilon} \\]

**Decoding:**
- \\(\bar{G}\\): the mean of returns in this episode (or batch of episodes)
- \\(\sigma_G\\): the standard deviation of returns in this episode
- \\(\epsilon\\): a small constant (typically \\(10^{-8}\\)) that prevents division by zero when all returns are identical
- \\(\tilde{G}_t\\): the normalized return, which has mean ≈ 0 and std ≈ 1

This is not the same as a value-function baseline — it is a simpler, episode-local normalization. It does **not** guarantee unbiasedness in the same rigorous way (the normalization itself introduces a small bias), but it provides two practical benefits:

1. **Keeps gradient scale consistent across episodes**: episodes with large absolute returns do not produce enormous gradient updates that blow up the learning rate's implicit effect.
2. **Automatic advantage interpretation**: normalized returns above zero become "better than average this episode" and below zero become "worse than average," which is semantically similar to an advantage function without requiring a separate critic network.

```python
import torch

torch.manual_seed(42)

# Simulate a batch of episode returns with high absolute scale
# (e.g., conjunction-avoidance reward in some large unit system)
raw_returns = torch.tensor([
    # Episode 1: many timesteps, large rewards
    [850.0, 730.0, 620.0, 540.0, 300.0, 100.0, 50.0],
    # Episode 2: different scale — very poor performance
    [10.0,   5.0,  12.0,   8.0,   6.0,   3.0,  2.0],
    # Episode 3: moderate performance
    [400.0, 350.0, 280.0, 200.0, 150.0, 90.0, 30.0],
])

def normalize_returns(returns_2d):
    """Normalize returns across all timesteps in a batch of episodes."""
    flat = returns_2d.flatten()
    mean = flat.mean()
    std  = flat.std()
    return (returns_2d - mean) / (std + 1e-8), mean.item(), std.item()

normalized, mean_ret, std_ret = normalize_returns(raw_returns)

print("Raw returns (3 episodes, 7 timesteps each):")
for i, ep in enumerate(raw_returns):
    print(f"  Episode {i}: {ep.tolist()}")

print(f"\nBatch statistics: mean={mean_ret:.1f}, std={std_ret:.1f}")

print("\nNormalized returns:")
for i, ep in enumerate(normalized):
    print(f"  Episode {i}: {[f'{v:.2f}' for v in ep.tolist()]}")

print(f"\nNormalized batch: mean≈{normalized.mean().item():.4f}, std≈{normalized.std().item():.4f}")

# Compare gradient update magnitudes
example_log_probs = torch.ones(3, 7) * (-1.5)  # constant for illustration
loss_raw  = -(example_log_probs * raw_returns).sum()
loss_norm = -(example_log_probs * normalized).sum()

print(f"\nGradient magnitude (loss.item()):")
print(f"  Without normalization: {loss_raw.item():.1f}")
print(f"  With normalization:    {loss_norm.item():.4f}")
print("Normalization keeps the loss in a predictable range,")
print("preventing learning rate sensitivity to return scale.")
```

Without normalization, a policy that has learned a high-scoring strategy (large absolute returns) will produce large gradient updates, which can destabilize training. With normalization, the gradient magnitude stays bounded regardless of the return scale — the same learning rate works across different reward scales.

The tradeoff: normalization introduces a batch-level dependency (the normalization uses the statistics of the current batch). This is fine for on-policy REINFORCE but requires care in off-policy settings.

---

## When to use policy gradients vs. Q-learning

Both policy gradient methods and Q-learning are valid RL approaches, and the choice depends on the specific problem structure. Here is a concrete decision guide:

| Factor | Use Q-learning / DQN | Use Policy Gradients (REINFORCE, PPO, SAC) |
|--------|---------------------|---------------------------------------------|
| **Action space** | Discrete, small-to-medium (up to ~1000 actions) | Continuous, or discrete but very large |
| **Policy type needed** | Deterministic OK (greedy policy is fine) | Stochastic required (mixed strategies, exploration) |
| **Sample efficiency** | High priority (limited environment interactions) | Sample efficiency is secondary |
| **Reward shaping** | Shaped, dense rewards | Sparse or terminal rewards also OK |
| **Exploration strategy** | ε-greedy is sufficient | Need principled stochastic exploration |
| **Stability** | Sensitive to hyperparams with function approx | More robust, especially with shared trunk |
| **Multi-agent** | Works for small games | Preferred: stochastic policies = mixed strategies |
| **SSA example** | Discrete sensor tasking (5 satellites, pick one per step) | Continuous thrust vector optimization |

### Concrete SSA application decisions

**Satellite sensor scheduling (discrete):** At each timestep, choose which of N satellites to task. The action space is \\(\{1, \ldots, N\}\\). DQN is appropriate. The argmax operation over Q-values is cheap and the policy can be deterministic (always task the highest-priority satellite given current state).

**Orbital maneuver planning (continuous):** Choose a delta-v vector \\((\Delta v_x, \Delta v_y, \Delta v_z)\\) for an orbit correction. Action space is \\(\mathbb{R}^3\\). Policy gradients with `Normal` output are appropriate. DQN cannot handle this without severe discretization loss.

**Conjunction avoidance (stochastic preferred):** Multiple operators observe the same RSO and must decide simultaneously whether to maneuver. Game-theoretic reasoning suggests a mixed strategy (sometimes maneuver, sometimes hold) to avoid symmetric deadlocks. Policy gradients naturally represent stochastic policies; DQN's greedy policy is pure strategy.

**Telescope allocation scheduling (large discrete):** Allocate ground-based telescope time across hundreds of RSOs. With 500 potential targets, DQN requires 500 Q-value outputs — tractable. But if the scheduler must commit to a probabilistic allocation (observe each RSO with some probability), policy gradients are cleaner.

---

## Key Takeaways

- **Policy gradients parameterize the policy directly** as a neural network \\(\pi(a \mid s; \theta)\\) and use the REINFORCE gradient \\(\nabla J = \mathbb{E}[G_t \nabla \log \pi]\\) to improve it. The agent optimizes the policy itself, not a value function from which a policy is derived.
- **Continuous action spaces** are handled naturally by outputting distribution parameters (mean and std of a Gaussian) from the policy network and sampling via `torch.distributions.Normal`. DQN cannot extend to continuous actions without expensive discretization; REINFORCE handles satellite delta-v maneuvers with a single forward pass.
- **REINFORCE has high variance** because returns \\(G_t\\) vary enormously across episodes. Standard error shrinks as \\(\text{Std}(G)/\sqrt{K}\\) where \\(K\\) is the number of episodes — the 1/√K convergence rate means you often need hundreds of episodes before gradient estimates are reliable.
- **Baseline subtraction is zero-bias variance reduction**: subtracting any state-dependent function \\(b(s_t)\\) from returns does not change the expected gradient because \\(\mathbb{E}[\nabla \log \pi] = 0\\). Using the value function as baseline gives the advantage \\(A_t = G_t - V(s_t)\\), the foundation for actor-critic.
- **Return normalization** (subtract mean, divide by std within each episode) is a practical stabilization trick that keeps gradient updates at a consistent scale regardless of reward magnitude, preventing the learning rate from becoming effectively too large or too small across different reward regimes.
- **Policy gradients vs. Q-learning** is a design choice: use Q-learning when the action space is discrete and a deterministic policy suffices; use policy gradients when the action space is continuous, a stochastic policy is needed (multi-agent mixed strategies, exploration), or the problem naturally frames as direct policy optimization.

---

## Quiz

{{#quiz 05-policy-gradient-methods.toml}}
