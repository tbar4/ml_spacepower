# Lesson 5: Recurrent Networks — LSTM and GRU

**Module:** Neural Networks as Function Approximators — M02
**Source:** Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"; Cho et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder"; Goodfellow et al. "Deep Learning" Chapter 10; PyTorch documentation `nn.LSTM`, `nn.GRU`

---

## Where this fits

Lessons 1–4 built the complete feedforward neural network toolkit: activation functions, MLP construction, loss functions, and the training loop. Every network in those lessons took a fixed-size input vector and produced an output — the same computation every time, with no memory of past inputs.

Satellite TLE histories, orbital maneuver campaigns, and time-series sensor data are sequences. The relevant information is not in any single observation but in how observations change over time. This lesson introduces the recurrent neural network architectures that process sequences natively: the LSTM (Long Short-Term Memory) and the GRU (Gated Recurrent Unit). These are the architectures used in Module 9's maneuver detection pipeline.

---

## The sequence modeling problem

A feedforward MLP maps a fixed input vector `x ∈ R^n` to an output. To process a sequence `x_1, x_2, ..., x_T`, you could concatenate all steps into a single long vector and feed it to a large MLP. This fails for two reasons:

1. **Variable-length sequences**: A satellite's TLE history may be 20 epochs or 60 epochs, depending on tracking coverage. A fixed-size input cannot handle this without padding and masking, and even with padding, long sequences produce enormous input vectors.
2. **No parameter sharing across time**: The MLP learns separate weights for "what happened at position 3" and "what happened at position 17." But a maneuver at day 3 and a maneuver at day 17 of a 30-day window are the same kind of event — the same weights should recognize them. Parameter sharing enforces this symmetry.

A recurrent network solves both problems by processing the sequence one step at a time, maintaining a hidden state that summarizes what has been seen so far.

---

## Vanilla RNN and why it fails

The basic recurrent neural network applies the same learned transformation at every time step:

```python
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
```

The hidden state `h_t` is updated at each step using the previous hidden state and the current input. After T steps, `h_T` summarizes the entire sequence.

The problem: **vanishing gradients**. When you backpropagate through T steps (backpropagation through time, BPTT), the gradient of the loss with respect to an early hidden state involves a product of T Jacobian matrices `∂h_t/∂h_{t-1}`. If those Jacobians have eigenvalues less than 1 (typical for tanh outputs), the product shrinks exponentially with T. By the time you reach step 1 of a 30-step sequence, the gradient is numerically zero. The network cannot learn from events that happened more than 5–10 steps in the past.

For orbital sequences where the maneuver signature may be spread across 20+ days, this is fatal.

---

## LSTM: explicit memory management

The LSTM, introduced by Hochreiter and Schmidhuber in 1997, replaces the vanilla RNN with a gated architecture that separates short-term memory (the hidden state `h_t`) from long-term memory (the cell state `c_t`). Three learned gates control information flow:

**Forget gate** — decides what fraction of the existing cell state to discard:
```
f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)
```
Output is in (0, 1). A value near 1 means "remember everything from before"; near 0 means "forget everything."

**Input gate** — decides what new information to write into the cell state:
```
i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)    # how much to write
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)  # what to write
```

**Cell state update** — combines forget and input:
```
c_t = f_t * c_{t-1} + i_t * g_t
```

**Output gate** — produces the hidden state from the updated cell state:
```
o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)
```

The cell state `c_t` is the LSTM's long-term memory. Because the cell state path only involves element-wise multiplication and addition (no matrix multiplication), gradients flow through it with much less distortion than through vanilla RNN hidden states. An event at day 1 can still influence `c_T` at day 30 via the cell state highway.

The key intuition: **the gates learn when to remember and when to forget**. A satellite in normal station-keeping has a consistent mean motion for weeks; the forget gate learns to retain this baseline. When a maneuver occurs, the input gate writes the new mean motion value strongly; the forget gate learns to partially reset the baseline. The output gate determines what aspects of the memory to expose to the classifier.

---

## GRU: simpler alternative

The Gated Recurrent Unit (GRU, Cho et al. 2014) achieves similar performance with fewer parameters by merging the cell state and hidden state:

```
z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)     # update gate (like forget+input combined)
r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)     # reset gate
h̃_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # candidate hidden state
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # update
```

The update gate `z_t` controls how much the hidden state changes at each step — near 0 means "keep the old hidden state"; near 1 means "replace it with the new candidate." The reset gate `r_t` controls how much past context the candidate state sees.

GRU and LSTM perform comparably on most tasks. GRU has fewer parameters (3 gate matrices vs. 4) and trains faster. LSTM has more representational flexibility. For orbital sequences with 30–60 epochs, the difference is marginal — start with LSTM if you want the canonical architecture, GRU if training speed is the constraint.

---

## PyTorch implementation

```python
import torch
import torch.nn as nn

class ManeuverLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,     # features per TLE epoch (e.g. 6)
        hidden_size: int,    # LSTM hidden dimension
        num_layers: int,     # stacked LSTM layers
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # input: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        # Use the final layer's last hidden state for classification
        last_hidden = h_n[-1]           # (batch, hidden_size)
        return self.classifier(last_hidden).squeeze(-1)  # (batch,)
```

Critical details:
- `batch_first=True` matches the natural tensor shape `(batch, seq_len, features)`. The default `batch_first=False` expects `(seq_len, batch, features)` — a common source of shape bugs.
- `lstm_out` contains the hidden state at every time step; `h_n` contains only the final hidden state. For classification, you want `h_n[-1]` (the last layer's final state), not `lstm_out[:, -1, :]` (the last time step's output from all layers — same values for a single-layer LSTM, different for multi-layer).
- Dropout between layers is set via the `dropout` parameter, not a separate `nn.Dropout`. But this dropout only applies *between* stacked layers — for dropout on the output, add an explicit `nn.Dropout` before the classifier.

For the GRU version, replace `nn.LSTM` with `nn.GRU` and change `(h_n, c_n)` to just `h_n`:

```python
self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                   num_layers=num_layers, batch_first=True,
                   dropout=dropout if num_layers > 1 else 0.0)

gru_out, h_n = self.gru(x)
last_hidden = h_n[-1]
```

---

## Handling variable-length sequences

When batching TLE windows of different lengths, you need to pad shorter sequences to match the longest sequence in the batch. PyTorch provides `pack_padded_sequence` and `pad_packed_sequence` to avoid computing LSTM steps over padding tokens:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # x: (batch, max_seq_len, input_size), padded
    # lengths: (batch,) — actual length of each sequence
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                   enforce_sorted=False)
    lstm_out_packed, (h_n, c_n) = self.lstm(packed)
    lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
    last_hidden = h_n[-1]
    return self.classifier(last_hidden).squeeze(-1)
```

For Module 9's daily-gridded windows, all sequences in a batch are the same length (30 days), so packing is not strictly necessary. It matters when mixing 30-day and 60-day windows in the same batch.

---

## Sequence classification vs. sequence labeling

There are two distinct tasks you can do with an LSTM:

**Sequence classification** (what Module 9 uses): one label per window — "did a maneuver occur anywhere in this 30-day sequence?" The `h_n[-1]` approach above is correct: the final hidden state summarizes the full sequence, and one prediction is made per window.

**Sequence labeling**: one label per time step — "at each day, was the satellite maneuvering?" This requires `lstm_out` (all hidden states at every step) with a classifier applied at each position: `self.classifier(lstm_out)` produces `(batch, seq_len, 1)`.

For maneuver detection, sequence classification is usually sufficient. Sequence labeling is useful if you want to localize the maneuver epoch precisely rather than just detecting that one occurred — but TLE data does not have fine enough temporal resolution to justify the additional complexity for most applications.

---

## Where LSTMs appear in the rest of the curriculum

- **Module 9, Lesson 1**: An LSTM trained on 30-day TLE windows classifies orbital sequences as maneuver or no-maneuver — the direct application of this lesson.
- **Module 9, Lesson 2**: A transformer encoder is compared against this LSTM baseline; understanding the LSTM is a prerequisite for understanding why the transformer is sometimes better.
- The opponent modeling algorithms in Module 7 can be implemented using recurrent networks when the history of observations must be encoded into a belief state — the LSTM hidden state functions as a learnable belief representation.

---

## Key Takeaways

- **Vanilla RNNs fail on long sequences because of vanishing gradients.** Backpropagation through time multiplies Jacobians across every step — for 30+ steps the gradient reaching early time steps is numerically zero, preventing the network from learning from distant past events.
- **LSTMs solve vanishing gradients by separating long-term memory (cell state) from short-term memory (hidden state).** The cell state pathway involves only element-wise operations, allowing gradients to flow back across many steps without exponential decay.
- **Three gates — forget, input, output — learn when to remember and when to reset.** For orbital sequences, the forget gate learns to maintain the station-keeping baseline; the input gate writes maneuver events strongly; the output gate exposes relevant memory to the classifier.
- **GRU is a simpler alternative with fewer parameters and comparable performance.** Prefer LSTM for canonical compatibility; prefer GRU when training speed is the constraint.
- **Use `batch_first=True` and take `h_n[-1]` for sequence classification.** `batch_first=True` matches the natural `(batch, seq_len, features)` shape. `h_n[-1]` gives the last LSTM layer's final hidden state — the correct input to the classification head.
- **For maneuver detection, sequence classification (one label per window) is appropriate.** Use `lstm_out` and per-step classification only when you need to localize the maneuver epoch precisely.

---

## Quiz

{{#quiz 05-recurrent-networks.toml}}
