# Lesson 4: Opponent Modeling


<!-- toc -->

## Where this fits

Lessons 1 through 3 of this module developed the tools for reasoning under partial observability: belief states, particle filters, and imperfect-information equilibria. Those tools assume either a single agent (POMDPs) or rational opponents (Nash equilibrium). This lesson addresses a different regime: real adversaries in space operations are not perfectly rational, and their irrationality is exploitable if modeled correctly.

Opponent modeling is the practice of building and using a predictive model of the adversary's strategy. It draws on everything built so far: Bayesian updating (Module 1) to maintain a distribution over opponent types; RL best response (Module 3) to compute what to do given the model; game-tree structures (Module 5) to reason about what the opponent might do next; and POMDP belief states (Module 7, lessons 1-2) to track hidden opponent parameters.

The central tension in opponent modeling is the exploit-generalize tradeoff: a model that perfectly captures the current opponent lets you beat them decisively, but may be completely wrong about the next opponent. Managing this tradeoff requires understanding when to trust a model and when to abandon it.

## The exploit-generalize tradeoff

Suppose you have observed an adversarial satellite operator over 20 maneuvers and built a model that predicts their next maneuver with 85% accuracy. You can compute the best response to this model — the sensor allocation strategy that maximizes coverage given the predicted maneuver timing and target orbit.

The problem: you are not playing against the model. You are playing against the actual operator, who may change strategy if they realize they are being predicted. An adversary who detects that their maneuver timing is being anticipated will change their timing. Your model, perfectly calibrated to their past behavior, becomes wrong the moment they adapt.

The tradeoff is:

- **Best response to the current model** (exploit): maximally effective against the current opponent if the model is correct. Completely wrong if the opponent adapts.
- **Nash equilibrium strategy** (generalize): safe against any opponent strategy, including adversarial adaptation. Cannot exploit predictable opponents — leaves value on the table.
- **Mixture**: use the model when confidence is high, hedge toward equilibrium when confidence is low.

In SSA, this tradeoff has operational consequences. Over-committing to a model of a "routine" operator and then encountering an adversary who behaves differently can mean missing a critical conjunction event or misallocating sensors at precisely the wrong time.

## Frequency-based opponent models

The simplest model: track the opponent's historical action frequencies.

For an adversary whose action space is {maneuver-small, maneuver-large, hold}, count how often each action has been taken:

```python
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

class FrequencyModel:
    """
    Frequency-based opponent model.
    Tracks how often the opponent has taken each action,
    optionally conditioned on the observable game state.
    """
    def __init__(self, actions: List[str], smoothing: float = 1.0):
        """
        actions: list of possible opponent action strings.
        smoothing: Laplace smoothing count (prevents zero probabilities).
        """
        self.actions = actions
        self.smoothing = smoothing
        # Counts unconditional and conditional on observable context
        self.counts: Dict[Optional[str], np.ndarray] = defaultdict(
            lambda: np.full(len(actions), smoothing)
        )

    def observe(self, action: str, context: Optional[str] = None) -> None:
        """Record one observed opponent action, optionally with context."""
        action_idx = self.actions.index(action)
        self.counts[context][action_idx] += 1.0

    def predict(self, context: Optional[str] = None) -> np.ndarray:
        """Return probability distribution over opponent's next action."""
        counts = self.counts[context]
        return counts / counts.sum()

    def best_response_action(self, defender_payoffs: np.ndarray,
                              context: Optional[str] = None) -> int:
        """
        Return the defender action index that maximizes expected payoff,
        given the predicted opponent action distribution.
        
        defender_payoffs: (n_defender_actions, n_opponent_actions) matrix.
        """
        opp_probs = self.predict(context)
        ev = defender_payoffs @ opp_probs   # expected value per defender action
        return int(np.argmax(ev))

# Example: tracking a challenger operator's maneuver decisions
MANEUVER_ACTIONS = ["hold", "small_maneuver", "large_maneuver"]

model = FrequencyModel(actions=MANEUVER_ACTIONS, smoothing=0.5)

# Simulated history: this operator mostly holds, occasionally small maneuvers
observed_history = (["hold"] * 12 + ["small_maneuver"] * 5 +
                    ["large_maneuver"] * 2 + ["hold"] * 3)
for action in observed_history:
    model.observe(action)

probs = model.predict()
print("Frequency model prediction:")
for a, p in zip(MANEUVER_ACTIONS, probs):
    print(f"  {a}: {p:.3f}")
```

Frequency models are transparent and require minimal data. Their limitation is stationarity: they assume the opponent's strategy does not change over time. A moving-average variant partially addresses this by weighting recent observations more heavily:

```python
class ExponentialMovingFrequencyModel:
    """
    Exponentially-weighted frequency model.
    Recent observations count more than old ones.
    Adapts to strategy shifts.
    """
    def __init__(self, actions: List[str], decay: float = 0.95,
                 smoothing: float = 0.5):
        self.actions = actions
        self.decay = decay
        self.weights = np.full(len(actions), smoothing)

    def observe(self, action: str) -> None:
        # Decay all existing weights toward zero
        self.weights *= self.decay
        # Add one to the observed action (no decay for the new observation)
        self.weights[self.actions.index(action)] += 1.0

    def predict(self) -> np.ndarray:
        return self.weights / self.weights.sum()
```

## Bayesian opponent modeling

A richer approach: maintain a prior over *types* of opponents, where each type is associated with a different behavioral strategy. Update the type posterior as actions are observed.

**Setup:**
- Define \\(K\\) types \\(\Theta = \{\theta_1, \ldots, \theta_K\}\\), each with a known action distribution \\(\pi_{\theta_k}(a)\\).
- Maintain a prior \\(P(\theta)\\) over types.
- After observing action \\(a_t\\), update:

\\[ P(\theta_k \mid a_{1:t}) \propto P(a_t \mid \theta_k) \cdot P(\theta_k \mid a_{1:t-1}) \\]

**Decoding:** This is Bayes' rule applied recursively. The likelihood \\(P(a_t \mid \theta_k)\\) is the probability that type \\(\theta_k\\) would have taken action \\(a_t\\). The prior \\(P(\theta_k \mid a_{1:t-1})\\) is the current type belief after all previous observations. Multiplying and normalizing gives the updated type belief.

The best response is computed against the *mixture* of type strategies, weighted by the type posterior:

\\[ \pi_{\text{predicted}}(a) = \sum_k P(\theta_k \mid a_{1:t}) \cdot \pi_{\theta_k}(a) \\]

**Decoding:** The predicted action distribution is a mixture of the type-specific distributions, where the mixing weights are the type posteriors. As more actions are observed, the posterior concentrates on the most consistent type, and the predicted distribution approaches the true opponent strategy.

## SSA scenario: adversarial operator type tracking

An adversarial satellite operator is known to adopt one of three behavioral strategies:

- **Type A: Minimum-fuel** — always takes the smallest maneuver that achieves the objective. Predictable: small maneuvers, gradual orbit change.
- **Type B: Maximum-coverage** — maximizes the area of sky visible during approach. Takes larger, faster maneuvers.
- **Type C: Random-perturbation** — makes randomly-sized maneuvers to avoid predictability. Delta-V drawn uniformly from the full budget.

```python
import numpy as np
from typing import List, Tuple

class BayesianOpponentModel:
    """
    Maintains a posterior over operator types and predicts the next action.
    """
    def __init__(self, types: List[str],
                 type_priors: np.ndarray,
                 action_labels: List[str],
                 type_likelihoods: np.ndarray):
        """
        types: list of type names.
        type_priors: prior probability of each type (sums to 1).
        action_labels: list of observable action categories.
        type_likelihoods: (n_types, n_actions) array.
            type_likelihoods[k, j] = P(action j | type k).
        """
        assert len(types) == len(type_priors) == type_likelihoods.shape[0]
        assert len(action_labels) == type_likelihoods.shape[1]
        self.types = types
        self.type_priors = type_priors.astype(float)
        self.type_posterior = type_priors.astype(float).copy()
        self.action_labels = action_labels
        self.type_likelihoods = type_likelihoods
        self.history: List[str] = []

    def observe(self, action: str) -> None:
        """
        Update type posterior after observing the opponent's action.
        Applies Bayes' rule: posterior ∝ likelihood × prior.
        """
        action_idx = self.action_labels.index(action)
        likelihoods = self.type_likelihoods[:, action_idx]
        unnorm = likelihoods * self.type_posterior
        total = unnorm.sum()
        if total < 1e-12:
            print(f"Warning: all types assign near-zero probability to action '{action}'. "
                  "Resetting to uniform.")
            self.type_posterior = np.ones(len(self.types)) / len(self.types)
        else:
            self.type_posterior = unnorm / total
        self.history.append(action)

    def predict_next_action(self) -> np.ndarray:
        """
        Return the predicted action distribution under the current type posterior.
        = sum over types of P(type) * P(action | type).
        """
        return self.type_posterior @ self.type_likelihoods

    def entropy(self) -> float:
        """Entropy of the type posterior (bits). Zero = certain about type."""
        p = self.type_posterior
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    def report(self) -> None:
        """Print current type posterior."""
        print(f"After {len(self.history)} observations:  "
              + "  ".join(f"{t}: {p:.3f}" for t, p in
                          zip(self.types, self.type_posterior)))

def run_operator_tracking_demo() -> None:
    """
    Simulate 20 decisions from a type-B (max-coverage) operator.
    Show that the Bayesian model converges on type B.
    """
    np.random.seed(3)

    # Action categories: small, medium, large maneuver
    ACTIONS = ["small", "medium", "large"]

    # Likelihoods: (3 types, 3 actions)
    # Type A (min-fuel): mostly small
    # Type B (max-coverage): mostly large or medium
    # Type C (random): uniform
    LIKELIHOODS = np.array([
        [0.70, 0.25, 0.05],   # Type A: min-fuel
        [0.10, 0.35, 0.55],   # Type B: max-coverage
        [0.33, 0.34, 0.33],   # Type C: random
    ])

    model = BayesianOpponentModel(
        types=["min_fuel", "max_coverage", "random"],
        type_priors=np.array([1/3, 1/3, 1/3]),
        action_labels=ACTIONS,
        type_likelihoods=LIKELIHOODS,
    )

    # True opponent is type B (max-coverage)
    true_type_likelihoods = LIKELIHOODS[1]

    print("Tracking an adversarial satellite operator over 20 decisions")
    print(f"{'Decision':>8}  {'Action':>8}  "
          f"{'P(min_fuel)':>12}  {'P(max_cov)':>10}  "
          f"{'P(random)':>10}  {'Entropy':>8}")
    print("-" * 65)

    for decision in range(1, 21):
        # Simulate true operator's action from type B distribution
        action = np.random.choice(ACTIONS, p=true_type_likelihoods)
        model.observe(action)

        if decision % 2 == 0 or decision == 1:
            tp = model.type_posterior
            ent = model.entropy()
            print(f"{decision:>8}  {action:>8}  "
                  f"{tp[0]:>12.3f}  {tp[1]:>10.3f}  "
                  f"{tp[2]:>10.3f}  {ent:>8.3f}")

    print()
    print("Predicted next action distribution:")
    pred = model.predict_next_action()
    for a, p in zip(ACTIONS, pred):
        print(f"  {a}: {p:.3f}")

if __name__ == "__main__":
    run_operator_tracking_demo()
```

## The response function

Given an opponent model, the **response function** maps the model's predicted action distribution to the best defender action.

For a pure best response against a fixed opponent:

\\[ a^* = \arg\max_{a \in A_D} \sum_{a' \in A_C} \hat{P}(a') \cdot R(a, a') \\]

**Decoding:**
- \\(A_D\\): defender's action space.
- \\(A_C\\): challenger's action space.
- \\(\hat{P}(a')\\): the model's predicted probability that the challenger takes action \\(a'\\).
- \\(R(a, a')\\): defender's reward for action \\(a\\) when challenger takes \\(a'\\).

The pure best response is optimal if the opponent model is correct and the opponent is not adapting. Against an adaptive opponent, the pure best response is exploitable.

The **safe hedge**: mix between the best response and the Nash equilibrium strategy. The mixing weight is the confidence in the model:

\\[ \sigma_D = \lambda \cdot \sigma^*_{\text{BR}} + (1 - \lambda) \cdot \sigma^*_{\text{Nash}} \\]

where \\(\lambda \in [0, 1]\\) is the confidence in the opponent model. High confidence: act mostly on the model. Low confidence: fall back toward Nash.

```python
def hedged_defender_strategy(
    best_response_action: int,
    nash_strategy: np.ndarray,
    model_confidence: float,
    n_actions: int
) -> np.ndarray:
    """
    Mix between pure best response and Nash equilibrium strategy,
    weighted by model confidence.
    
    model_confidence: float in [0, 1]. 1.0 = full trust in model.
    """
    # One-hot best response
    br_strategy = np.zeros(n_actions)
    br_strategy[best_response_action] = 1.0

    # Convex combination
    return model_confidence * br_strategy + (1 - model_confidence) * nash_strategy
```

## Neural opponent modeling with an LSTM

For richer history-dependent prediction, we can train a recurrent neural network to predict the opponent's next action given the sequence of past actions and observable game state.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

class LSTMOpponentModel(nn.Module):
    """
    LSTM that takes a sequence of (defender_action, challenger_action) pairs
    and predicts the challenger's next action probability distribution.
    
    This captures longer-range patterns than the frequency model:
    e.g., "challenger tends to use large maneuvers two steps after holding".
    """
    def __init__(self, n_defender_actions: int, n_challenger_actions: int,
                 embed_dim: int = 16, lstm_dim: int = 32):
        super().__init__()
        # Input: one-hot concatenation of both players' last actions
        input_dim = n_defender_actions + n_challenger_actions
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_head = nn.Sequential(
            nn.Linear(lstm_dim, n_challenger_actions),
            # No softmax here — use CrossEntropyLoss which includes log-softmax
        )
        self.n_def = n_defender_actions
        self.n_chal = n_challenger_actions

    def forward(self, action_seq: torch.Tensor,
                hidden=None) -> Tuple[torch.Tensor, Tuple]:
        """
        action_seq: (batch, seq_len, n_def + n_chal) — one-hot concatenated actions.
        Returns logits for the next challenger action and the updated hidden state.
        """
        lstm_out, new_hidden = self.lstm(action_seq, hidden)
        logits = self.output_head(lstm_out)   # (batch, seq_len, n_challenger_actions)
        return logits, new_hidden

    def predict_next(self, action_seq: torch.Tensor,
                     hidden=None) -> Tuple[np.ndarray, Tuple]:
        """
        Return probability distribution over challenger's next action.
        action_seq: (seq_len, n_def + n_chal) — single sequence (no batch dim).
        """
        seq = action_seq.unsqueeze(0)   # add batch dimension
        with torch.no_grad():
            logits, new_hidden = self.forward(seq, hidden)
        probs = torch.softmax(logits[0, -1, :], dim=-1)  # last timestep
        return probs.numpy(), new_hidden

def train_lstm_opponent_model(
    episodes: List[List[Tuple[int, int]]],   # list of (def_action, chal_action) sequences
    n_def: int = 3,
    n_chal: int = 3,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> LSTMOpponentModel:
    """
    Train the LSTM opponent model on observed (defender_action, challenger_action) episodes.
    
    Target: predict challenger's action at each step from the history.
    Loss: cross-entropy on challenger action prediction.
    """
    model = LSTMOpponentModel(n_def, n_chal, embed_dim=16, lstm_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def encode_sequence(episode: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of (def_action, chal_action) pairs to input tensor and target tensor.
        Input: one-hot of (def_action || chal_action) at t.
        Target: challenger action at t+1.
        """
        inputs, targets = [], []
        for t in range(len(episode) - 1):
            def_a, chal_a = episode[t]
            # One-hot encode both actions
            one_hot = torch.zeros(n_def + n_chal)
            one_hot[def_a] = 1.0
            one_hot[n_def + chal_a] = 1.0
            inputs.append(one_hot)
            targets.append(episode[t + 1][1])   # next challenger action
        return torch.stack(inputs), torch.tensor(targets)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for episode in episodes:
            if len(episode) < 2:
                continue
            inputs, targets = encode_sequence(episode)
            inputs = inputs.unsqueeze(0)   # (1, seq_len, input_dim)
            logits, _ = model(inputs)
            # logits: (1, seq_len, n_chal); targets: (seq_len,)
            loss = loss_fn(logits.squeeze(0), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}: avg loss = {total_loss / len(episodes):.4f}")

    return model

# Example: simulate episodes from a type-B opponent and train the LSTM
def generate_synthetic_episodes(n_episodes: int = 200, episode_len: int = 15,
                                  seed: int = 42) -> List[List[Tuple[int, int]]]:
    """
    Simulate interaction episodes for the LSTM training.
    Challenger follows type B (max-coverage): prefers large maneuvers.
    Defender uses a random policy.
    """
    np.random.seed(seed)
    TYPE_B_PROBS = [0.10, 0.35, 0.55]   # small, medium, large
    episodes = []
    for _ in range(n_episodes):
        episode = []
        for _ in range(episode_len):
            def_action  = np.random.randint(0, 3)              # random defender
            chal_action = np.random.choice(3, p=TYPE_B_PROBS)  # type B challenger
            episode.append((def_action, chal_action))
        episodes.append(episode)
    return episodes
```

## Epistemic humility: when to abandon a model

Every opponent model is an approximation. The operator may be following a different strategy than any modeled type. They may adapt. The environment may change in a way that shifts which strategy is optimal for them.

**Detecting model failure:** Monitor the KL divergence between what the model predicts and what is actually observed. If the model's predictions are consistently wrong, the KL divergence will be large:

\\[ D_{KL}(P_{\text{observed}} \| P_{\text{predicted}}) = \sum_a P_{\text{observed}}(a) \log \frac{P_{\text{observed}}(a)}{P_{\text{predicted}}(a)} \\]

**Decoding:**
- \\(P_{\text{observed}}\\): empirical action distribution over a recent window.
- \\(P_{\text{predicted}}\\): model's predicted distribution over the same window.
- If this is small (near zero), the model is a good fit. If large, the model is systematically wrong.

```python
def kl_divergence(p_observed: np.ndarray, p_predicted: np.ndarray,
                   epsilon: float = 1e-10) -> float:
    """
    KL divergence of observed from predicted.
    Large value indicates model misfit: model's predictions are wrong.
    """
    p = p_observed + epsilon
    q = p_predicted + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

class ModelHealthMonitor:
    """
    Tracks KL divergence between model predictions and actual observations.
    Triggers a model reset when divergence exceeds a threshold.
    """
    def __init__(self, model: BayesianOpponentModel, window: int = 10,
                 kl_threshold: float = 0.5):
        self.model = model
        self.window = window
        self.kl_threshold = kl_threshold
        self.recent_actions: List[str] = []
        self.recent_predictions: List[np.ndarray] = []
        self.kl_history: List[float] = []

    def step(self, predicted_dist: np.ndarray, actual_action: str) -> bool:
        """
        Record a prediction and the action that actually occurred.
        Returns True if model failure is detected (KL too high).
        """
        self.recent_predictions.append(predicted_dist.copy())
        self.recent_actions.append(actual_action)

        # Only evaluate once we have a full window
        if len(self.recent_actions) < self.window:
            return False

        # Empirical action distribution over the window
        action_labels = self.model.action_labels
        n = len(action_labels)
        p_observed = np.zeros(n)
        for a in self.recent_actions[-self.window:]:
            p_observed[action_labels.index(a)] += 1.0
        p_observed /= p_observed.sum()

        # Average predicted distribution over the window
        p_predicted = np.mean(self.recent_predictions[-self.window:], axis=0)

        kl = kl_divergence(p_observed, p_predicted)
        self.kl_history.append(kl)

        if kl > self.kl_threshold:
            print(f"Model failure detected (KL={kl:.3f} > threshold={self.kl_threshold}). "
                  "Recommend model reset or type prior reset.")
            return True
        return False

def demonstrate_model_failure_detection() -> None:
    """
    Simulate a regime change: operator starts as type A, switches to type B at step 15.
    Show that KL divergence detects the switch.
    """
    np.random.seed(11)
    ACTIONS = ["small", "medium", "large"]
    LIKELIHOODS = np.array([
        [0.70, 0.25, 0.05],   # Type A: min-fuel
        [0.10, 0.35, 0.55],   # Type B: max-coverage
        [0.33, 0.34, 0.33],   # Type C: random
    ])
    model = BayesianOpponentModel(
        types=["min_fuel", "max_coverage", "random"],
        type_priors=np.array([1/3, 1/3, 1/3]),
        action_labels=ACTIONS,
        type_likelihoods=LIKELIHOODS,
    )
    monitor = ModelHealthMonitor(model, window=8, kl_threshold=0.4)

    print("Phase 1 (steps 1-14): true operator is type A (min-fuel)")
    print("Phase 2 (steps 15-30): true operator switches to type B (max-coverage)")
    print()

    for step in range(1, 31):
        # Predict before observing
        predicted = model.predict_next_action()

        # Simulate true action
        if step <= 14:
            true_dist = LIKELIHOODS[0]    # type A
        else:
            true_dist = LIKELIHOODS[1]    # type B

        action = np.random.choice(ACTIONS, p=true_dist)

        # Update model and monitor
        model.observe(action)
        failure = monitor.step(predicted, action)

        if step % 5 == 0 or step in (14, 15, 16):
            kl_str = (f"KL={monitor.kl_history[-1]:.3f}"
                      if monitor.kl_history else "KL=N/A")
            tp = model.type_posterior
            print(f"Step {step:>2}: action={action:<7}  {kl_str:<12}  "
                  f"P(A)={tp[0]:.2f}  P(B)={tp[1]:.2f}  P(C)={tp[2]:.2f}  "
                  f"{'ALERT' if failure else ''}")

if __name__ == "__main__":
    demonstrate_model_failure_detection()
```

The KL divergence trigger connects directly to anomaly detection in SSA more broadly: an operator whose behavior is inconsistent with their historical pattern is either a different operator, using a new strategy, or responding to something in the environment. All three are operationally significant signals.

## Connection to anomaly detection in SSA

The model health monitor is a generalization of the anomaly detection methods common in operational SSA. Traditional conjunction analysis flags an RSO when its observed position diverges from its predicted trajectory beyond a threshold (position innovation divided by position uncertainty, i.e., a Mahalanobis distance). The KL divergence monitor does the analogous thing for strategic behavior: it flags an operator when their observed decisions diverge from the predicted decision distribution beyond a threshold.

Both are tests of the same hypothesis: is the evidence consistent with the current model? If not, something has changed, and the model needs revision.

## Transfer to CFR: implicit opponent modeling

In CFR (Module 5), there is no explicit opponent model. Instead, CFR iteratively updates *both* players' strategies based on accumulated regrets, converging to a Nash equilibrium. How does this relate to opponent modeling?

**Reach probabilities track beliefs about the opponent's behavior.** At information set \\(I\\) belonging to player \\(i\\), the counterfactual reach probability \\(\pi_{-i}(I)\\) is the probability that play reaches \\(I\\) if all players except \\(i\\) play their current strategy. This is an implicit model of the opponent's strategy — not an explicit type distribution, but a probability distribution over what the opponent has been doing.

**Counterfactual values correct for opponent deviation.** When CFR computes the counterfactual regret of action \\(a\\) at \\(I\\), it asks: "how much better would I have done by always playing \\(a\\) at \\(I\\), holding the opponent's strategy fixed?" This is precisely the best-response computation against a fixed opponent model — the opponent model is the current strategy profile maintained by CFR.

**The key difference:** explicit opponent modeling assumes the opponent has a fixed (or slowly-changing) strategy that you estimate and best-respond to. CFR assumes both players are simultaneously adapting, and finds the equilibrium where neither wants to change. Explicit modeling is better against static, predictable opponents; CFR is better against adaptive opponents or when you have no history to train on.

In the SSA context, an operator whose behavior you have 50 historical observations on is a candidate for explicit modeling. A new adversary with no history is best approached with an equilibrium strategy, since you have no data to build a model. As observations accumulate, gradually shift from Nash toward the best-response-to-model, monitoring KL divergence to detect when the model becomes stale.

## Key Takeaways

- The exploit-generalize tradeoff is the central design choice in opponent modeling: best-responding to a model maximizes expected gain against the current opponent but is exploitable if the opponent adapts. A Nash equilibrium is safe but cannot exploit predictable opponents.
- Frequency models are simple and interpretable but assume stationarity. Exponentially-weighted variants partially address strategy drift. Neither captures structured behavioral patterns across multiple timesteps.
- Bayesian opponent models maintain a distribution over discrete operator types and update via Bayes' rule as actions are observed. The resulting posterior can concentrate quickly (10-15 observations) on the true type when types are well-separated, giving actionable exploitation strategies.
- Neural opponent models (LSTM-based) capture longer-range behavioral patterns and context-dependent strategies, at the cost of requiring substantial training data and lacking the interpretability of Bayesian type models.
- Model health monitoring via KL divergence between predicted and observed action distributions provides an early warning of strategy changes. This is the behavioral analog of innovation-based anomaly detection in orbital mechanics, and should trigger model resets or Bayesian prior resets when divergence exceeds a threshold.
- In CFR, opponent modeling is implicit: reach probabilities encode beliefs about the opponent's strategy, and counterfactual values compute best responses to those beliefs. Explicit opponent modeling is more powerful against static, identifiable opponents; CFR equilibrium strategies are safer against unknown or adaptive adversaries.

{{#quiz 04-opponent-modeling.toml}}
