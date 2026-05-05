# Module 3 Project: A DQN Sensor Allocation Agent


<!-- toc -->

## What you are building

You will build a DQN agent that learns to allocate ground-based sensor time across multiple satellites under uncertainty about which ones are in conjunction risk situations. The environment is a custom OpenSpiel game (your first OpenSpiel touchpoint), and the Q-network is structurally similar to the conjunction-risk approximator from Module 2.

By the end, you will have:
1. A working OpenSpiel environment (a custom game definition you wrote)
2. A DQN agent that solves it
3. An evaluation showing the trained agent significantly outperforms a random baseline

This project unifies everything from Modules 1 to 3: probability and uncertainty (Module 1), neural network function approximation (Module 2), and reinforcement learning (this module).

## The scenario

A space operations center has one ground-based optical telescope and 5 satellites it is responsible for. At each timestep:

1. The agent picks one of the 5 satellites to observe (5 discrete actions)
2. The observation reveals whether that satellite is currently in a conjunction risk window
3. Each satellite has its own probability of entering a conjunction window in the next timestep, which depends on its current alert status

States that the agent needs to consider:
- For each satellite, an "alert level" from 0 (no recent activity) to 4 (high alert, conjunction likely)
- For each satellite, the number of timesteps since it was last observed

Rewards:
- +10 for observing a satellite that is currently in a conjunction window
- +1 for observing any satellite (basic mission credit)
- 0 for time spent not observing a satellite that turns out to be in conjunction (missed opportunity, but no negative reward)

The optimal strategy involves balancing recent observations (to know alert levels) with attention to high-alert satellites. A pure greedy strategy ("always observe the satellite I am most uncertain about") is suboptimal; so is "always observe the highest-alert satellite." The agent has to learn the right tradeoff.

## Step 1: install OpenSpiel

```bash
pip install open_spiel
```

This installs OpenSpiel's Python interface. We do not need to compile from source.

Verify with:

```python
import pyspiel
print(pyspiel.registered_names()[:10])  # show 10 built-in games
```

## Step 2: define the custom environment

OpenSpiel's API is designed for general games (including multi-agent and imperfect-information ones). For now, we are using it for a single-agent MDP, which OpenSpiel handles as a one-player "game" where chance plays the role of the environment.

```python
"""
sensor_allocation_env.py: a simplified SSA sensor scheduling environment.
"""

import numpy as np
import pyspiel
from open_spiel.python.observation import IIGObserverForPublicInfoGame, make_observation

NUM_SATELLITES = 5
MAX_ALERT = 4
MAX_STEPS = 50

# Per-satellite probabilities of alert level transitions
# A satellite at alert level k has probability p[k] of being in a conjunction
# window if observed THIS step, and probability q[k] of escalating to k+1 next step
ALERT_TO_CONJUNCTION_PROB = [0.05, 0.15, 0.30, 0.55, 0.85]  # by alert level
ALERT_ESCALATION_PROB     = [0.10, 0.15, 0.20, 0.25, 0.00]   # at level 4 it can't go higher

class SensorAllocationGame(pyspiel.Game):
    def __init__(self, params=None):
        game_type = pyspiel.GameType(
            short_name="sensor_allocation",
            long_name="Sensor Allocation Single-Agent",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.REWARDS,
            max_num_players=1,
            min_num_players=1,
            provides_information_state_string=False,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=True,
            parameter_specification={},
        )
        game_info = pyspiel.GameInfo(
            num_distinct_actions=NUM_SATELLITES,
            max_chance_outcomes=2 ** NUM_SATELLITES,  # each sat: alert escalates or not
            num_players=1,
            min_utility=-1000.0,
            max_utility=1000.0,
            max_game_length=MAX_STEPS,
        )
        super().__init__(game_type, game_info, params or {})
    
    def new_initial_state(self):
        return SensorAllocationState(self)


class SensorAllocationState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._alert_levels       = [0] * NUM_SATELLITES
        self._steps_since_obs    = [0] * NUM_SATELLITES
        self._cumulative_reward  = 0.0
        self._step_count         = 0
        self._is_terminal        = False
        self._pending_outcomes   = None  # used during chance node resolution
    
    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._pending_outcomes is not None:
            return pyspiel.PlayerId.CHANCE
        return 0  # the single agent
    
    def legal_actions(self, player=None):
        if self._is_terminal:
            return []
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return list(range(2 ** NUM_SATELLITES))
        return list(range(NUM_SATELLITES))
    
    def chance_outcomes(self):
        # Each satellite escalates independently. Build a joint distribution.
        # For 5 sats, that's 32 joint outcomes. (Smaller would be faster but
        # this matches OpenSpiel's chance-node interface cleanly.)
        outcomes = []
        for outcome_idx in range(2 ** NUM_SATELLITES):
            prob = 1.0
            for i in range(NUM_SATELLITES):
                escalates = bool(outcome_idx & (1 << i))
                p = ALERT_ESCALATION_PROB[self._alert_levels[i]]
                prob *= (p if escalates else (1.0 - p))
            outcomes.append((outcome_idx, prob))
        return outcomes
    
    def _apply_action(self, action):
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            # Process the joint chance outcome
            for i in range(NUM_SATELLITES):
                escalates = bool(action & (1 << i))
                if escalates and self._alert_levels[i] < MAX_ALERT:
                    self._alert_levels[i] += 1
            self._pending_outcomes = None
            self._step_count += 1
            if self._step_count >= MAX_STEPS:
                self._is_terminal = True
            return
        
        # Player action: observe satellite `action`
        sat_idx = action
        prob_in_conjunction = ALERT_TO_CONJUNCTION_PROB[self._alert_levels[sat_idx]]
        in_conjunction = np.random.rand() < prob_in_conjunction
        
        # Reward: +10 if conjunction caught, +1 for any observation
        reward = 1.0
        if in_conjunction:
            reward += 10.0
        
        # Reset alert level for observed satellite (we just dealt with it)
        self._alert_levels[sat_idx] = 0
        self._steps_since_obs[sat_idx] = 0
        
        # Increment "steps since observed" for unobserved satellites
        for i in range(NUM_SATELLITES):
            if i != sat_idx:
                self._steps_since_obs[i] += 1
        
        self._cumulative_reward += reward
        
        # Schedule the chance node next
        self._pending_outcomes = "alert_evolution"
    
    def rewards(self):
        if self._is_terminal:
            return [0.0]
        # Returns the most recent step's reward; for our purposes we track
        # cumulative and let the agent compute per-step rewards externally
        return [self._cumulative_reward]
    
    def returns(self):
        return [self._cumulative_reward]
    
    def is_terminal(self):
        return self._is_terminal
    
    def observation_tensor(self, player=0):
        # The observation tensor concatenates:
        #   - alert levels (NUM_SATELLITES values, normalized to [0, 1])
        #   - steps since observation (NUM_SATELLITES values, normalized)
        return np.array(
            [a / MAX_ALERT for a in self._alert_levels]
            + [min(s, 10) / 10.0 for s in self._steps_since_obs],
            dtype=np.float32
        )
    
    def observation_string(self, player=0):
        return f"alerts={self._alert_levels}, since_obs={self._steps_since_obs}"
    
    def __str__(self):
        return self.observation_string()


# Register the game
pyspiel.register_game(
    pyspiel.GameType(
        short_name="sensor_allocation",
        long_name="Sensor Allocation Single-Agent",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        reward_model=pyspiel.GameType.RewardModel.REWARDS,
        max_num_players=1,
        min_num_players=1,
        provides_information_state_string=False,
        provides_information_state_tensor=False,
        provides_observation_string=True,
        provides_observation_tensor=True,
        parameter_specification={},
    ),
    pyspiel.GameInfo(
        num_distinct_actions=NUM_SATELLITES,
        max_chance_outcomes=2 ** NUM_SATELLITES,
        num_players=1,
        min_utility=-1000.0,
        max_utility=1000.0,
        max_game_length=MAX_STEPS,
    ),
    SensorAllocationGame
)
```

A note on this OpenSpiel scaffolding: it is more verbose than a vanilla Gym environment because OpenSpiel is built for general games (multi-agent, imperfect information, formal extensive-form structure). For our single-agent MDP we are using a tiny subset of its features. The investment pays off in Module 5 when we use the same OpenSpiel infrastructure to define multi-player game-theoretic problems.

## Step 3: a Gym-like wrapper for the DQN agent

The DQN code from lesson 4 expects a simpler interface than OpenSpiel's `pyspiel.State` API. Let us wrap the OpenSpiel game in a Gym-like interface:

```python
class SensorAllocationEnv:
    """Gym-like wrapper around the OpenSpiel sensor allocation game."""
    
    def __init__(self):
        self.game  = SensorAllocationGame()
        self.state = None
    
    def reset(self):
        self.state = self.game.new_initial_state()
        return self.state.observation_tensor()
    
    def step(self, action):
        # Apply the agent's action
        prev_cum_reward = self.state.returns()[0]
        self.state.apply_action(action)
        
        # Resolve chance nodes
        while self.state.is_chance_node():
            outcomes = self.state.chance_outcomes()
            actions, probs = zip(*outcomes)
            chosen = np.random.choice(actions, p=probs)
            self.state.apply_action(chosen)
        
        new_cum_reward = self.state.returns()[0]
        step_reward    = new_cum_reward - prev_cum_reward
        
        next_obs = self.state.observation_tensor() if not self.state.is_terminal() \
                   else np.zeros(2 * NUM_SATELLITES, dtype=np.float32)
        done = self.state.is_terminal()
        
        return next_obs, step_reward, done
    
    @property
    def state_dim(self):
        return 2 * NUM_SATELLITES
    
    @property
    def num_actions(self):
        return NUM_SATELLITES
```

## Step 4: drop in the DQN agent from lesson 4

Use the `DQNAgent` class from lesson 4 directly. The state and action dimensions are now `env.state_dim` and `env.num_actions`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, state):
        return self.net(state)

# (The DQNAgent class from lesson 4 goes here, unchanged)
```

## Step 5: train the agent

```python
env = SensorAllocationEnv()
agent = DQNAgent(state_dim=env.state_dim, 
                num_actions=env.num_actions,
                lr=1e-3, gamma=0.95, epsilon=0.2,
                buffer_capacity=10_000, batch_size=64,
                target_update_freq=200)

NUM_EPISODES = 1000
episode_returns = []
loss_history    = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_return = 0.0
    
    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()
        if loss is not None:
            loss_history.append(loss)
        
        state = next_state
        episode_return += reward
        if done:
            break
    
    episode_returns.append(episode_return)
    
    # Anneal exploration: decrease epsilon over training
    agent.epsilon = max(0.05, agent.epsilon * 0.995)
    
    if (episode + 1) % 50 == 0:
        recent = episode_returns[-50:]
        print(f"Episode {episode + 1:4d}: "
              f"avg return = {sum(recent)/len(recent):6.2f}, "
              f"epsilon = {agent.epsilon:.3f}")
```

After about 1000 episodes, the agent should achieve average returns substantially higher than a random baseline.

## Step 6: evaluate against baselines

```python
def evaluate(env, action_fn, num_episodes=200):
    """Run the action_fn for num_episodes and return mean return."""
    returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total = 0
        for _ in range(MAX_STEPS):
            action = action_fn(state)
            state, reward, done = env.step(action)
            total += reward
            if done:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns)

# Random baseline: pick a satellite uniformly at random
random_action = lambda s: random.randrange(env.num_actions)

# Highest-alert baseline: pick the satellite with the highest alert level
def highest_alert_action(s):
    alerts = s[:NUM_SATELLITES]  # first 5 features are alert levels (normalized)
    return int(np.argmax(alerts))

# Trained DQN: use the policy network with epsilon = 0
def dqn_action(s):
    with torch.no_grad():
        q = agent.q_net(torch.tensor(s, dtype=torch.float32))
        return int(q.argmax().item())

agent.epsilon = 0  # disable exploration for evaluation

print("\n=== Evaluation (200 episodes each) ===")
for name, action_fn in [
    ("Random", random_action),
    ("Highest alert", highest_alert_action),
    ("DQN (trained)", dqn_action),
]:
    mean_ret, std_ret = evaluate(env, action_fn)
    print(f"{name:20}: mean return = {mean_ret:6.2f} ± {std_ret:5.2f}")
```

A trained DQN should beat both baselines. The "highest alert" baseline is competitive because alert level genuinely correlates with conjunction probability, but it ignores the value of getting fresh information about other satellites. DQN learns to balance these.

## Step 7: reflect

1. How quickly did your agent's performance improve? At what episode did it pass the random baseline? At what episode did it pass the highest-alert baseline?
2. The agent's input state has 10 features (5 alert levels + 5 steps-since-observation). What other features might help? What features might be unnecessary?
3. What happens if you remove the `target_update_freq` (i.e., use the main network for the target too)? Does training become unstable?
4. What happens if you remove the replay buffer (train on the most recent transition only)? Does training become unstable?
5. The reward function gives +10 for conjunction detections and +1 for any observation. What if you only gave +10 for conjunctions (no participation reward)? Would the agent still learn?

## What you have built

- An OpenSpiel game definition for a single-agent MDP
- A DQN agent integrated with OpenSpiel
- A working RL training loop on a non-trivial problem
- An evaluation comparing learned to scripted policies

Module 4 takes the next step: search and planning. Instead of a model-free agent that learns a Q function, MCTS does explicit search over the game tree and uses the search results to make decisions. AlphaZero combines MCTS with neural networks (a value network and a policy network), and we will use the same OpenSpiel infrastructure to define and play larger games. The actor-critic structure from lesson 6 reappears as the AlphaZero training objective.

The DQN you built here will serve as a baseline comparison for the AlphaZero-style agents in Module 4.
