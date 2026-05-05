# Lesson 5: PettingZoo, Shimmy, and Ray RLlib

**Module:** OpenSpiel and the Rust Capstone — M08: Production Engineering
**Source:** [cite: Terry et al. "PettingZoo: Gym for Multi-Agent Reinforcement Learning" NeurIPS 2021; Liang et al. "RLlib: Abstractions for Distributed Reinforcement Learning" ICML 2018; Farama Foundation shimmy documentation; Hu et al. "MARLlib: A Scalable and Efficient Library for Multi-Agent Reinforcement Learning" JMLR 2023]

---


<!-- toc -->

## Where this fits

Lessons 1 and 2 of this module built and registered a custom OpenSpiel game (the conjunction-masking scenario). Lesson 3 added the Rust solver. Lesson 4 designed the SSA game for CFR. Those lessons focused on game logic and tabular equilibrium computation. This lesson asks a different question: once you have a working game, how do you train large-scale neural policies against it using a cluster of CPUs and a GPU?

The answer requires four distinct software layers sitting between your OpenSpiel game and Ray RLlib's distributed training engine. None of those layers is obvious, and wiring them together is where most practitioners lose days. This lesson walks through every connection explicitly.

After this lesson you can wire your OpenSpiel SSA game to train at scale with Ray RLlib. The lesson also connects back to Module 6 (CTDE and MAPPO, implemented here via MARLlib on top of RLlib) and Module 3 (IMPALA and APPO, which are the distributed training algorithms you will configure).

---

## 1. The Integration Problem

OpenSpiel implements game logic. Ray RLlib trains policies at scale. They cannot talk to each other directly. The gap between them has three parts:

1. OpenSpiel exposes a C++-oriented `pyspiel.State` API that advances game states one action at a time. RLlib expects a Gymnasium-compatible environment that returns batched transitions.
2. OpenSpiel is inherently multi-agent and sequential. RLlib's multi-agent interface expects a specific dictionary-of-observations format that OpenSpiel does not produce.
3. Distributed training with RLlib requires that environments can be serialized and cloned across worker processes. Raw OpenSpiel games satisfy this, but the glue code must be structured to allow it.

The recommended production pipeline is:

```
OpenSpiel (C++ game logic)
    |
    | pyspiel Python bindings
    v
shimmy.OpenSpielCompatibilityV0
    |
    | translates OpenSpiel AEC semantics to PettingZoo AEC API
    v
PettingZoo AEC environment
    |
    | custom wrapper (or supersuit for preprocessing)
    v
RLlib MultiAgentEnv wrapper
    |
    | registered via register_env()
    v
Ray RLlib / MARLlib
    (distributed rollout workers, learner GPU)
```

Each layer has a precisely defined responsibility:

- **OpenSpiel**: game rules, legal actions, terminal conditions, payoffs, information state tensors. No training logic.
- **shimmy**: translates OpenSpiel's game-state-advance loop into the PettingZoo Agent Environment Cycle (AEC) API. Handles player-ID-to-agent-name mapping, terminal state signaling, observation array conversion.
- **PettingZoo**: the multi-agent equivalent of Gymnasium. A standard interface that many algorithm libraries understand. Preprocessing wrappers (frame stacking, observation normalization, action masking) can be inserted here.
- **RLlib MultiAgentEnv wrapper**: bridges PettingZoo's turn-by-turn AEC cycle to RLlib's simultaneous-step interface, which expects one `step(action_dict)` call per episode step rather than one call per agent per step.
- **RLlib / MARLlib**: handles distributed rollout collection, policy updates, replay buffers, and GPU-accelerated training. Knows nothing about game rules.

Getting the layers wrong is the most common source of bugs in this stack. A mismatch between PettingZoo's observation space declaration and what the game actually returns causes silent shape errors in the neural network's input layer. An incorrect `__all__` termination signal in the RLlib wrapper causes episodes to never end. The sections below show each layer working correctly.

---

## 2. PettingZoo and the AEC Model

PettingZoo is the multi-agent equivalent of Gymnasium. Where Gymnasium defines a single-agent `env.step(action) -> (obs, reward, terminated, truncated, info)` interface, PettingZoo defines a multi-agent interface where agents take turns in a cycle.

### The Agent Environment Cycle

The Agent Environment Cycle (AEC) model is the core abstraction. In AEC, at any moment exactly one agent is designated as the "current actor" via `env.agent_selection`. The caller observes the current actor's state, selects an action, and calls `env.step(action)`. Control then passes to the next agent in the cycle.

The key API methods:

| Method / Attribute | Purpose |
|---|---|
| `env.reset()` | Start a new episode; returns `{agent_id: obs}` and `{agent_id: info}` |
| `env.agent_selection` | The agent whose turn it is right now |
| `env.step(action)` | Advance by one agent-turn; updates internal state |
| `env.observe(agent)` | Return the current observation for a specific agent |
| `env.last()` | Return `(obs, reward, terminated, truncated, info)` for `agent_selection` |
| `env.rewards` | Dict mapping each agent to its most recent reward |
| `env.terminations` | Dict mapping each agent to its termination flag |
| `env.truncations` | Dict mapping each agent to its truncation flag |
| `env.agent_iter()` | Iterator that cycles through agents, stopping when all are done |

**Note on `env.last()` vs `env.observe()`:** `env.last()` returns the reward and termination flags for the current agent since the last time that agent acted, which is what you want in the AEC loop. `env.observe()` returns only the observation tensor, without reward or termination status. Use `env.last()` inside the `agent_iter()` loop.

The AEC model is correct for turn-based games for an important reason: it avoids state-aliasing. In older parallel-step models, all agents submitted actions simultaneously even in sequential games; this required the environment to buffer actions for agents that had not yet acted, which created subtle bugs when actions arrived out of order or when an agent was already terminated. AEC makes the ordering explicit in the API.

For simultaneous-move games (where all agents act at the same time), PettingZoo also provides a parallel AEC interface where `env.step(actions_dict)` accepts a dictionary of actions from all live agents at once. The SSA coverage game below uses the sequential AEC interface because the conjunction-masking game is turn-based.

### A Minimal PettingZoo Environment: 2-Agent SSA Coverage

The following implements a simplified two-agent SSA coverage game as a PettingZoo AEC environment. Two operators (Blue and Red) alternate turns claiming radar coverage windows over a contested orbital arc. The agent that accumulates more unique coverage windows at episode end wins.

```python
"""
ssa_coverage_env.py
Minimal PettingZoo AEC environment for a 2-agent SSA coverage game.
Two agents alternate claiming coverage windows (0-3) over 8 turns total.
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


NUM_WINDOWS = 4        # coverage windows per turn (actions 0..3)
EPISODE_TURNS = 8      # total turns (4 per agent)


class SSACoverageEnv(AECEnv):
    """
    Two-agent sequential coverage game.
    Each agent, on their turn, claims one of 4 orbital coverage windows.
    Reward: +1 for each unique window claimed. -1 for a duplicate claim.
    Episode ends after EPISODE_TURNS total turns.
    """

    metadata = {"render_modes": [], "name": "ssa_coverage_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["blue_operator", "red_operator"]
        self._agent_selector = agent_selector(self.possible_agents)

        # Observation: [my_claimed_windows (4 bits), opponent_claimed_windows (4 bits),
        #                turn_number (normalized)]
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(NUM_WINDOWS)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._claimed = {agent: set() for agent in self.agents}
        self._turn = 0

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        observations = {agent: self._observe(agent) for agent in self.agents}
        return observations, self.infos

    def _observe(self, agent):
        other = [a for a in self.agents if a != agent][0]
        my_bits = np.array(
            [1.0 if w in self._claimed[agent] else 0.0 for w in range(NUM_WINDOWS)],
            dtype=np.float32,
        )
        other_bits = np.array(
            [1.0 if w in self._claimed[other] else 0.0 for w in range(NUM_WINDOWS)],
            dtype=np.float32,
        )
        turn_norm = np.array([self._turn / EPISODE_TURNS], dtype=np.float32)
        return np.concatenate([my_bits, other_bits, turn_norm])

    def observe(self, agent):
        return self._observe(agent)

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # Agent already done: absorb the dead step
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        reward = 1.0 if action not in self._claimed[agent] else -1.0
        self._claimed[agent].add(action)
        self._turn += 1

        # Zero out rewards for agents not currently acting
        self.rewards = {a: 0.0 for a in self.agents}
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        if self._turn >= EPISODE_TURNS:
            for a in self.agents:
                self.terminations[a] = True

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def last(self):
        agent = self.agent_selection
        obs = self._observe(agent)
        reward = self._cumulative_rewards[agent]
        terminated = self.terminations[agent]
        truncated = self.truncations[agent]
        info = self.infos[agent]
        return obs, reward, terminated, truncated, info
```

**Note on `_was_dead_step`:** This is a PettingZoo utility that handles the case where `step()` is called for an agent that is already terminated. When using the `agent_iter()` loop correctly, you must call `step(None)` for terminated agents to advance the cycle past them. The `_was_dead_step` helper absorbs this call without corrupting state.

To run the environment in the canonical AEC loop:

```python
env = SSACoverageEnv()
env.reset()
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
```

---

## 3. Shimmy: OpenSpiel to PettingZoo

The shimmy library provides a single class that wraps any OpenSpiel game as a standards-compliant PettingZoo AEC environment: `shimmy.OpenSpielCompatibilityV0`.

### What shimmy does

When you call `shimmy.OpenSpielCompatibilityV0(game=spiel_game)`, shimmy:

1. **Reads `game.information_state_tensor_shape()`** and declares the corresponding `observation_space` as a `Box` for each player. This is what tells PettingZoo (and downstream wrappers) the shape and dtype of observations.
2. **Maps player indices to agent names**: player 0 becomes `"player_0"`, player 1 becomes `"player_1"`, etc. This is consistent across all wrapped games.
3. **Handles chance nodes internally**: OpenSpiel chance nodes are not exposed to the caller. When the underlying OpenSpiel state hits a chance node, shimmy samples the outcome automatically and advances the state. From the PettingZoo caller's perspective, chance nodes are invisible.
4. **Signals termination correctly**: when `state.is_terminal()`, shimmy sets all agents' `terminations` to `True` and returns the terminal utility from `state.returns()` as the final reward.
5. **Translates information state tensors**: `env.observe(agent)` calls `state.information_state_tensor(player_idx)` and returns the resulting numpy array.

### What shimmy does not do

Shimmy is a compatibility layer, not a preprocessing pipeline. It does not:

- Normalize observations to a useful range (OpenSpiel information state tensors can have very different scales across games)
- Shape rewards (OpenSpiel terminal utilities might be in [-10, 10] while RLlib training is more stable with rewards in [-1, 1])
- Apply action masking (illegal actions are legal from PettingZoo's perspective; you must handle this in the RLlib wrapper)

These are your responsibility, and the debugging checklist in section 9 returns to all three.

### Usage

```python
import pyspiel
import shimmy

# Load your registered custom game
spiel_game = pyspiel.load_game("conjunction_ssa")

# Wrap it as a PettingZoo AEC environment
env = shimmy.OpenSpielCompatibilityV0(game=spiel_game, render_mode=None)

# Now env is a standard PettingZoo AEC environment
env.reset()
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        # Replace with your trained policy
        action = env.action_space(agent).sample()
    env.step(action)
```

**Note on `render_mode=None`:** shimmy accepts `render_mode` to match the Gymnasium 0.26+ API signature. For training, always pass `None`. Rendering adds overhead and is not needed when rollout workers are collecting transitions at scale.

**Note on registering a custom OpenSpiel game:** `pyspiel.load_game("conjunction_ssa")` works only if the game has been registered via `pyspiel.register_game(game_type, game_class)` before the call. The registration pattern is covered in lesson 2. For a shimmy-wrapped game, registration must happen in every worker process, not just the main process — which is why the environment factory function (section 5) re-loads the game from scratch rather than passing a pre-built object.

### Inspecting what shimmy produces

```python
import pyspiel
import shimmy

game = pyspiel.load_game("kuhn_poker")
env = shimmy.OpenSpielCompatibilityV0(game=game, render_mode=None)

print("Agents:", env.possible_agents)
# Agents: ['player_0', 'player_1']

print("Observation space (player_0):", env.observation_space("player_0"))
# Box(low=0.0, high=1.0, shape=(11,), dtype=float32)
# Shape is determined by game.information_state_tensor_shape()

print("Action space (player_0):", env.action_space("player_0"))
# Discrete(3)  -- Kuhn poker has 3 actions: fold, call, raise

env.reset()
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    print(f"  {agent}: obs.shape={obs.shape}, reward={reward:.2f}, done={terminated}")
    env.step(None if terminated else env.action_space(agent).sample())
```

---

## 4. Ray RLlib MultiAgentEnv

RLlib's multi-agent interface is `ray.rllib.env.multi_agent_env.MultiAgentEnv`. Custom environments must subclass it. The interface uses a simultaneous-step model: one call to `step(action_dict)` advances the entire environment by one "round," and the returned dictionaries contain entries for every agent that is active in that round.

This is the fundamental impedance mismatch with PettingZoo AEC. AEC is sequential (one agent per `step` call). RLlib is simultaneous (all active agents per `step` call). The wrapper below resolves this by translating the PettingZoo AEC turn cycle into the RLlib format.

### The MultiAgentEnv interface

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import numpy as np


class SSAWargameEnv(MultiAgentEnv):
    """
    RLlib MultiAgentEnv wrapping a PettingZoo AEC environment.

    Steps through the AEC cycle internally and returns combined
    observations and rewards as RLlib-style dicts.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        # Build the underlying PettingZoo AEC environment.
        # In production, replace SSACoverageEnv() with your shimmy-wrapped
        # OpenSpiel game.
        self._env = SSACoverageEnv()
        self._env.reset()

        self.possible_agents = self._env.possible_agents
        self.agents = self.possible_agents[:]

        # RLlib requires these to be set on the instance
        self.observation_space = self._env.observation_space(self.possible_agents[0])
        self.action_space = self._env.action_space(self.possible_agents[0])

        # For heterogeneous obs/action spaces, use a spaces.Dict instead:
        # self.observation_space = spaces.Dict({
        #     agent: self._env.observation_space(agent)
        #     for agent in self.possible_agents
        # })

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        self.agents = self._env.agents[:]
        return obs_dict, info_dict

    def step(self, action_dict):
        """
        RLlib calls this with action_dict = {agent_id: action} for all
        currently active agents. We step through each agent's AEC turn
        and accumulate the transitions.
        """
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {"__all__": False}
        truncated_dict = {"__all__": False}
        info_dict = {}

        for agent_id, action in action_dict.items():
            self._env.step(action)
            obs, reward, terminated, truncated, info = self._env.last()
            obs_dict[agent_id] = obs
            reward_dict[agent_id] = reward
            terminated_dict[agent_id] = terminated
            truncated_dict[agent_id] = truncated
            info_dict[agent_id] = info

        # "__all__" signals episode end to RLlib
        terminated_dict["__all__"] = all(
            terminated_dict.get(a, False) for a in self.possible_agents
        )
        truncated_dict["__all__"] = all(
            truncated_dict.get(a, False) for a in self.possible_agents
        )

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
```

**Note on `"__all__"`:** RLlib uses the special key `"__all__"` in the `terminated` and `truncated` dictionaries to decide when to reset the environment. If `terminated["__all__"]` is `True`, RLlib ends the episode and calls `reset()`. If you omit this key or set it incorrectly, episodes either never end (training hangs) or terminate too early (rewards are cut short). Always compute `"__all__"` explicitly.

**Note on action ordering in turn-based games:** In a strictly sequential AEC game, only one agent acts per RLlib step. The cleaner design is to include only the currently active agent's action in `action_dict`. You can enforce this by checking `self._env.agent_selection` before submitting actions, or by configuring the rollout to produce single-agent steps. The wrapper above is a general starting point; production code should tighten the ordering logic.

### The PettingZoo-to-MultiAgentEnv wrapper for the OpenSpiel SSA game

For a shimmy-wrapped OpenSpiel game, the complete wrapper looks like this:

```python
import pyspiel
import shimmy
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class OpenSpielRLlibEnv(MultiAgentEnv):
    """
    Production wrapper: OpenSpiel game -> shimmy -> PettingZoo -> RLlib.

    The game is re-loaded inside __init__ so this class is safe to
    instantiate in parallel Ray worker processes.

    Usage:
        register_env("ssa_wargame", lambda cfg: OpenSpielRLlibEnv(cfg))
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        game_name = config.get("game_name", "kuhn_poker")

        # Re-load the game in this process (safe for multi-process workers)
        spiel_game = pyspiel.load_game(game_name)
        self._pz_env = shimmy.OpenSpielCompatibilityV0(
            game=spiel_game, render_mode=None
        )
        self._pz_env.reset()

        self.possible_agents = self._pz_env.possible_agents
        self.agents = self.possible_agents[:]

        # Homogeneous spaces: OpenSpiel games share obs/action space across players
        self.observation_space = self._pz_env.observation_space(self.possible_agents[0])
        self.action_space = self._pz_env.action_space(self.possible_agents[0])

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._pz_env.reset(seed=seed)
        self.agents = self._pz_env.agents[:]
        return obs_dict, info_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {}
        info_dict = {}

        for agent_id, action in action_dict.items():
            self._pz_env.step(action)
            obs, reward, terminated, truncated, info = self._pz_env.last()
            obs_dict[agent_id] = obs
            reward_dict[agent_id] = float(reward)
            terminated_dict[agent_id] = terminated
            truncated_dict[agent_id] = truncated
            info_dict[agent_id] = info

        all_done = all(
            self._pz_env.terminations.get(a, False)
            or self._pz_env.truncations.get(a, False)
            for a in self.possible_agents
        )
        terminated_dict["__all__"] = all_done
        truncated_dict["__all__"] = False

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
```

---

## 5. Configuring APPO for the SSA Wargame

With the `OpenSpielRLlibEnv` wrapper registered, you can configure APPO (Asynchronous Proximal Policy Optimization, covered in Module 3) for the SSA wargame. APPO is the right choice here: it is the distributed variant of PPO that uses asynchronous rollout workers, making it efficient when environments have variable episode lengths (typical for wargames where some episodes terminate quickly).

### Full working configuration

```python
import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
from gymnasium import spaces
import numpy as np

# Register the environment factory before ray.init()
register_env(
    "ssa_wargame",
    lambda config: OpenSpielRLlibEnv(config),
)

ray.init()

# Observation and action spaces (must match what the env declares)
obs_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
action_space = spaces.Discrete(4)

config = (
    APPOConfig()
    .environment("ssa_wargame", env_config={"game_name": "conjunction_ssa"})
    .multi_agent(
        policies={
            "blue_policy": (None, obs_space, action_space, {}),
            "red_policy":  (None, obs_space, action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, **kwargs:
            "blue_policy" if agent_id.startswith("blue") else "red_policy",
    )
    .rollouts(num_rollout_workers=16, num_envs_per_worker=8)
    .training(train_batch_size=2048, lr=5e-4)
    .resources(num_gpus=1)
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iter {i}: reward={result['episode_reward_mean']:.2f}")
```

**Decoding: `.environment("ssa_wargame", env_config=...)`**
The string `"ssa_wargame"` must match what you passed to `register_env`. The `env_config` dict is forwarded as the `config` argument to `OpenSpielRLlibEnv.__init__`. Use it to pass game-specific parameters (game name, reward scale, episode length cap) without hardcoding them in the wrapper class.

**Decoding: `.multi_agent(policies=...)`**
Each entry in the `policies` dict is a tuple `(policy_class, obs_space, action_space, policy_config)`. Setting `policy_class=None` tells RLlib to use its default policy for the configured algorithm (an APPO neural network policy in this case). The `obs_space` and `action_space` here must exactly match what `env.observation_space` and `env.action_space` return; a mismatch causes a silent shape error during the first neural network forward pass.

**Decoding: `policy_mapping_fn`**
This function tells RLlib which policy to use for each agent ID at runtime. It is called every time an agent needs an action. The mapping in the example routes any agent whose ID starts with `"blue"` to `"blue_policy"` and everything else to `"red_policy"`. This enables three training modes without changing any other configuration:

- **Asymmetric play**: blue and red use different architectures or training objectives.
- **Self-play**: route both agents to the same policy by returning `"blue_policy"` unconditionally. Both sides train the same weights, which prevents exploiting a fixed opponent.
- **Population-based training**: change the mapping dynamically to match against different historical snapshots of the policy (see section 7).

**Decoding: `.rollouts(num_rollout_workers=16, num_envs_per_worker=8)`**
`num_rollout_workers` is the number of Ray actor processes collecting transitions. Each worker runs `num_envs_per_worker` independent environment instances in parallel. Total parallel game instances: 16 × 8 = 128. More workers means more throughput but also more GPU-to-worker communication overhead; the tradeoff is hardware-dependent (see section 8 for sizing guidance).

**Decoding: `.training(train_batch_size=2048, lr=5e-4)`**
`train_batch_size` is the number of environment transitions collected before each policy update. With 128 parallel instances, each training iteration collects roughly 16 transitions per instance before updating. A larger batch reduces variance in the policy gradient estimate but requires more memory. `lr=5e-4` is a reasonable starting point for APPO on wargame environments; tune downward if training is unstable.

**Decoding: `.resources(num_gpus=1)`**
Allocates one GPU to the learner process. Rollout workers use CPU only. This is the standard configuration: GPU for the backward pass and policy update, CPUs for environment simulation.

---

## 6. MARLlib: Adding MAPPO

Vanilla RLlib APPO uses independent learners: each agent's policy is trained against its own rewards without access to other agents' observations during training. This is fine for competitive games but suboptimal for cooperative tasks where agents can share information during training to produce a better joint policy.

Module 6 introduced Centralized Training with Decentralized Execution (CTDE): during training, the critic can observe all agents' states; during execution, each agent uses only its own local observation. MAPPO (Multi-Agent PPO) is the standard CTDE algorithm. Implementing CTDE manually in vanilla RLlib requires writing a custom model that concatenates all agents' observations for the value function while keeping the policy head local. This is tedious and error-prone.

MARLlib provides a cleaner path. It is a library built on top of RLlib that implements CTDE algorithms (MAPPO, QMIX, MADDPG, and others) with the centralized critic construction handled automatically.

### Using MAPPO from MARLlib

```python
from marllib import marl

# Register your environment with MARLlib's wrapper
env = marl.make_env(environment_name="ssa_wargame", map_name="standard")

# Select MAPPO and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")

# Build the model (MLP with two 128-unit layers)
model = marl.build_model(
    env,
    mappo,
    {"core_arch": "mlp", "encode_layer": "128-128"},
)

# Train for 5 million timesteps
mappo.fit(
    env,
    model,
    stop={"timesteps_total": 5_000_000},
    checkpoint_freq=50,
    local_dir="~/ssa_results",
)
```

**Note on what MARLlib does automatically:** When you call `mappo.fit()`, MARLlib builds a PPO policy where the value function (critic) receives the concatenation of all agents' observation tensors as input, while the policy (actor) receives only the local agent's observation. The concatenation happens inside the model at training time; the actor head is unchanged. During evaluation (`algo.compute_actions()`), only the actor is used, so each agent acts on its own local observation — the CTDE property is preserved end-to-end.

**Contrast with vanilla RLlib APPO:** In vanilla RLlib, implementing the centralized critic requires a custom `ModelV2` subclass that receives the full joint observation via the `other_agent_batches` callback in `postprocess_trajectory`. This is documented but requires substantial boilerplate, and getting the batch shapes right across multiple agents and variable episode lengths is a common source of bugs. MARLlib abstracts this away entirely.

**Note on `hyperparam_source="common"`:** MARLlib ships with several hyperparameter presets. `"common"` uses broadly applicable defaults (clip ratio 0.2, value function loss coefficient 0.5, entropy coefficient 0.01). For SSA wargame training, you will likely need to tune the entropy coefficient upward in early training (higher entropy encourages exploration before the policy has learned the game structure) and the clip ratio downward for long-horizon orbital planning scenarios where large policy updates destabilize the value function.

**Note on `map_name="standard"`:** MARLlib uses map names to distinguish between scenario variants of the same environment. Define `"standard"` as your baseline SSA scenario. If you later add a `"contested_arc"` scenario with different reward structures or a `"multi_sensor"` scenario with more agents, you register them as additional map names and train on each independently or with transfer.

---

## 7. Self-Play with RLlib

Training against a fixed random opponent produces an overfitted policy that performs poorly against any non-random adversary. Self-play is the standard solution: train the policy against copies of itself, so the opponent is always at the frontier of the policy's own capability.

The simplest self-play configuration in RLlib uses the `policy_mapping_fn` to randomly assign the "opponent" role to a historical snapshot of the current policy. RLlib's `callbacks` API lets you snapshot the policy periodically.

### Opponent history self-play

```python
import random
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class SelfPlayCallback(DefaultCallbacks):
    """
    After each training iteration, snapshot the current policy
    and add it to the opponent pool with a fixed probability.
    """

    def __init__(self):
        super().__init__()
        self._opponent_snapshots = []  # list of policy weight dicts
        self._snapshot_interval = 20   # snapshot every 20 training iterations
        self._iter = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        self._iter += 1
        if self._iter % self._snapshot_interval == 0:
            weights = algorithm.get_policy("blue_policy").get_weights()
            self._opponent_snapshots.append(weights)

    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # At episode start, randomly pick a historical snapshot for red_policy
        if self._opponent_snapshots:
            snapshot = random.choice(self._opponent_snapshots)
            policies["red_policy"].set_weights(snapshot)


# Add to your config:
config = (
    APPOConfig()
    .environment("ssa_wargame")
    .multi_agent(
        policies={
            "blue_policy": (None, obs_space, action_space, {}),
            "red_policy":  (None, obs_space, action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, **kwargs:
            "blue_policy" if agent_id.startswith("blue") else "red_policy",
        policies_to_train=["blue_policy"],  # Only train blue; red is frozen
    )
    .callbacks(SelfPlayCallback)
    .rollouts(num_rollout_workers=16, num_envs_per_worker=8)
    .training(train_batch_size=2048, lr=5e-4)
    .resources(num_gpus=1)
)
```

**Note on `policies_to_train=["blue_policy"]`:** This tells RLlib to only compute gradients for `blue_policy`. `red_policy` weights are set externally by the callback and remain frozen during each gradient update. Without this flag, both policies would train simultaneously against each other, which creates a non-stationary training objective (both targets are moving) and often produces cycling or instability.

**Why self-play produces more robust strategies:** A policy trained against a fixed opponent converges to "beat this specific opponent," which may exploit strategies that only work against that opponent's particular weaknesses. Self-play periodically updates the opponent to match the current policy's strength, forcing the learner to find strategies that work across a range of capability levels. For an SSA wargame, this matters: a blue policy trained only against random red play will perform poorly against an adversarially optimal red.

The `_opponent_snapshots` list grows throughout training. Sampling uniformly from the full history (rather than always using the most recent snapshot) prevents "recency bias": the policy cannot simply learn to beat the most current version of itself; it must maintain strategies that are robust across historical skill levels. This is a simplified version of Prioritized Fictitious Self-Play (PFSP) used in AlphaStar, where snapshot selection is weighted toward snapshots that the current policy loses against most often.

---

## 8. Parallelism Math and Hardware Sizing

Getting the hardware configuration right is the difference between a 3-hour run and a 16-minute run. The math is straightforward.

### Throughput calculation

With `num_rollout_workers=32` and `num_envs_per_worker=16`:

```
Parallel game instances = 32 workers x 16 envs/worker = 512 instances
```

Environment step time depends heavily on whether you are using OpenSpiel via Python or via C++ bindings:

| Configuration | Step time | Throughput |
|---|---|---|
| Python OpenSpiel (pure Python game) | ~50 ms | 512 / 0.050 = 10,240 steps/sec |
| C++ OpenSpiel via pyspiel bindings | ~5 ms | 512 / 0.005 = 102,400 steps/sec |
| Custom Rust game (via PyO3 bindings) | ~2 ms | 512 / 0.002 = 256,000 steps/sec |

For a 100-million-step training run:

```
Python OpenSpiel:  100,000,000 / 10,240  ~= 9,766 seconds ~= 2.7 hours
C++ OpenSpiel:     100,000,000 / 102,400 ~=   977 seconds ~= 16 minutes
Rust (PyO3):       100,000,000 / 256,000 ~=   391 seconds ~= 6.5 minutes
```

This is why the curriculum teaches both OpenSpiel (established, well-tested game logic) and Rust (fast custom implementations): for a 100M-step SSA training run, implementation language is a first-order concern, not a stylistic preference.

**Note:** These estimates assume that environment stepping is the bottleneck, which is true when the neural network forward pass is fast (small MLP policies) and game episodes are short. For deeper neural policies or very long episodes, the learner GPU becomes the bottleneck and adding more rollout workers provides diminishing returns. Profile before scaling.

### Threadripper PRO 7995WX configuration

The AMD Threadripper PRO 7995WX (96 cores) with an NVIDIA RTX 6000 Ada is the recommended workstation for this curriculum's scale of experiments. Suggested allocation:

| Resource | Allocation | Purpose |
|---|---|---|
| 64 CPU cores | 32 rollout workers x 2 cores/worker | Environment simulation |
| 8 CPU cores | 1 learner process | Policy update, replay buffer management |
| 24 CPU cores | Spare | OS, logging, Ray Tune overhead, interactive use |
| RTX 6000 Ada (48 GB VRAM) | 1 GPU | Learner backward pass |

The 2 cores per worker allocation accounts for the environment's Python process and the Ray worker overhead. If your game is very fast (C++ or Rust), you can lower this to 1 core per worker and run more workers. If your game uses external physics simulation (a high-fidelity orbital propagator), you may need 4 cores per worker.

```python
config = (
    APPOConfig()
    .rollouts(num_rollout_workers=32, num_envs_per_worker=16)
    .resources(
        num_gpus=1,
        num_cpus_per_worker=2,
        num_cpus_for_local_worker=8,
    )
)
```

**Note on Ray memory budgeting:** Each rollout worker holds a copy of the environment and a copy of the policy weights for local inference. With 32 workers, a 100 KB policy (small MLP) uses about 3.2 MB across workers — negligible. A larger transformer policy at 100 MB would use 3.2 GB across workers, which fits comfortably in DDR5 but matters if you are also running multiple training jobs. Budget worker memory as `num_workers x (env_memory + policy_size)` when planning multi-experiment sessions.

---

## 9. Debugging Checklist

Wiring OpenSpiel to RLlib produces a specific set of failure modes. This checklist covers the five most common.

### 1. Observation space mismatch

**Symptom:** `ValueError: obs shape (11,) does not match declared obs space Box(shape=(9,))` at the start of training, or silent incorrect training where the policy receives wrong-shaped observations that get silently broadcast or truncated by numpy.

**Cause:** The `observation_space` declared in `MultiAgentEnv.__init__` does not match the shape of arrays returned by `reset()` and `step()`. This happens when you change the game's information state tensor shape without updating the wrapper's space declaration, or when shimmy reports a different shape than expected.

**Fix:** Print `env.observation_space("player_0").shape` and compare it to `env.observe("player_0").shape` after `env.reset()`. They must be identical. Then verify that the `obs_space` you pass to `APPOConfig().multi_agent(policies=...)` matches both.

### 2. Reward scaling

**Symptom:** Training loss spikes and destabilizes, or the value function converges to a nearly constant estimate regardless of the game state.

**Cause:** RLlib's PPO/APPO training is most stable when per-step rewards are roughly in the range [-1, 1]. OpenSpiel terminal utilities (for example, -3 for caught maneuvering, +2 for a successful covert maneuver) are in a different range, and they arrive only at the terminal step (all intermediate rewards are 0). This creates high variance in the value function estimates during the early phase of training.

**Fix:** Scale rewards in the wrapper's `step()` method:

```python
reward_dict[agent_id] = float(reward) / 3.0   # scale to [-1, 1] for conjunction game
```

Or use RLlib's built-in reward clipping:

```python
config = APPOConfig().training(clip_rewards=1.0)
```

For the conjunction-masking game, dividing by 3 (the maximum absolute utility) is exact. For other games, divide by `max(abs(min_utility), abs(max_utility))`, which you can read from `game.min_utility()` and `game.max_utility()`.

### 3. Episode length limits

**Symptom:** Rollout workers time out or training iteration wall time is much longer than expected.

**Cause:** A bug in the environment's termination logic causes some episodes never to terminate (the `"__all__"` key is never set to `True`). The worker blocks waiting for the episode to end.

**Fix:** Always add a `max_episode_steps` cap. RLlib will force-terminate episodes that exceed this limit by setting `truncated["__all__"] = True`:

```python
config = APPOConfig().environment(
    "ssa_wargame",
    max_episode_steps=200,   # force truncation after 200 steps
)
```

For the conjunction-masking game, the correct episode length is exactly 4 steps (chance node, adversary, defender, chance node). Any episode running longer than 10 steps indicates a termination bug and should be flagged immediately during development by asserting on episode length in the wrapper.

### 4. Action masking for illegal actions

OpenSpiel tracks legal actions via `state.legal_actions()`. In a resource-constrained SSA game, certain actions (for example, a maneuver that exceeds the remaining delta-v budget) become illegal mid-episode. RLlib does not know about these constraints unless you tell it explicitly.

The clean solution is to pass a legal action mask as part of the observation and modify the policy to zero out illegal action logits before the softmax:

```python
def _observe_with_mask(self, agent):
    """Return (base_obs, action_mask) concatenated as a single vector."""
    base_obs = self._pz_env.observe(agent)

    # Build action mask: 1.0 = legal, 0.0 = illegal
    legal = self._spiel_state.legal_actions()
    mask = np.zeros(self.action_space.n, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0

    return np.concatenate([base_obs, mask])
```

In RLlib, use a custom `ActionMaskModel` that reads the mask from the observation and applies it:

```python
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch


class ActionMaskModel(TorchModelV2):
    """Splits obs into [base_obs | mask], applies mask to action logits."""

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        base_obs = obs[:, : -self.num_outputs]
        mask = obs[:, -self.num_outputs :]

        logits, _ = self._base_model({"obs": base_obs}, state, seq_lens)

        # Replace logits for illegal actions with a large negative value
        masked_logits = logits + (mask - 1.0) * 1e9
        return masked_logits, state
```

**Why this matters for SSA:** In a multi-step orbital wargame, the set of legal maneuvers changes as fuel is consumed. An agent that proposes an illegal action and has it silently clipped to the nearest legal action learns incorrect value estimates for fuel-constrained states. Action masking forces the policy to learn the constraint correctly rather than relying on the environment to absorb illegal actions.

### 5. Agent name mismatch between shimmy and policy_mapping_fn

**Symptom:** `KeyError: 'player_0'` during training, or the policy mapping silently routes all agents to the wrong policy.

**Cause:** shimmy names agents `"player_0"`, `"player_1"`, etc. Your `policy_mapping_fn` may use different names (`"blue_1"`, `"red_1"`). If the mapping does not cover every agent name that the environment emits, RLlib raises a `KeyError` when it tries to look up the policy for an unknown agent ID.

**Fix:** Either use shimmy's naming convention in the `policy_mapping_fn`:

```python
policy_mapping_fn=lambda agent_id, **kwargs:
    "blue_policy" if agent_id == "player_0" else "red_policy",
```

Or rename agents in the wrapper's `reset()` by aliasing shimmy's default names to your preferred names before returning the observation dict. Using your own names throughout the wrapper and keeping shimmy's names only in the lowest layer is cleaner for large multi-agent games where agent names carry semantic meaning.

---

## Key Takeaways

- The production pipeline from OpenSpiel to distributed training has four distinct layers — OpenSpiel, shimmy, PettingZoo, and RLlib MultiAgentEnv — each with a specific and non-overlapping responsibility; getting any one layer wrong silently corrupts training rather than raising an obvious error.
- PettingZoo's AEC model makes agent turn order explicit in the API, which is the correct representation for sequential imperfect-information games; shimmy's `OpenSpielCompatibilityV0` handles chance nodes internally and translates information state tensors to PettingZoo observation arrays, but does not handle reward scaling or action masking.
- The `policy_mapping_fn` in RLlib's multi-agent config is the routing function that assigns each agent ID to a policy at runtime; setting it correctly enables asymmetric play, self-play, and population-based training without changing any other configuration.
- MARLlib's MAPPO constructs the centralized critic automatically by concatenating all agents' observations at training time, preserving the CTDE property (local observations at execution time); vanilla RLlib APPO requires a custom `ModelV2` subclass and `postprocess_trajectory` callback to achieve the same effect.
- Implementation language is a first-order throughput concern: a 100M-step training run takes approximately 2.7 hours with Python OpenSpiel and approximately 16 minutes with C++ bindings; for the Threadripper PRO 7995WX configuration, allocate 64 cores to 32 rollout workers, 8 cores to the learner, and assign the RTX 6000 Ada to the learner backward pass.
- The five most common wiring bugs are observation space mismatch, reward scaling (RLlib is most stable with rewards in [-1, 1]), missing `"__all__"` termination signals, unmasked illegal actions, and agent name mismatches between shimmy's default naming convention and the policy mapping function.

{{#quiz 05-pettingzoo-rllib.toml}}
