# Lesson 2: Transformers for Orbital Sequences

**Module:** Applied SDA ML — M09: Building Commercial SDA Products
**Source:** Vaswani et al. (2017) "Attention Is All You Need"; Zerveas et al. (2021) "A Transformer-based Framework for Multivariate Time Series Representation Learning"; Li et al. (2019) "Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting"; Zhou et al. (2021) "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

---

## Where this fits

Lesson 1 built a maneuver detector using an LSTM — the right tool when the sequence is short, the dataset is small, and computational resources are limited. This lesson builds the same detector using a transformer, which replaces the LSTM's sequential processing with self-attention. The result is a model that can process longer orbital histories in parallel, capture dependencies between non-adjacent TLE epochs directly, and produce attention weights that tell you *which past epochs the model attended to* when flagging a maneuver.

LSTMs and transformers are not interchangeable in all settings. This lesson is explicit about when each architecture is appropriate. For most solo-founder SDA products, LSTM is the right starting point. The transformer becomes the better choice when sequence length grows past 60 epochs, when you have enough training data to support larger parameter counts, or when interpretability of the temporal attention matters for a DoD customer.

---

## Why attention, and why now

The LSTM processes a TLE sequence one epoch at a time, left-to-right. Information about what happened 25 epochs ago must be carried through the hidden state across 24 intermediate steps. In practice, this means LSTMs struggle to maintain precise information about distant past events — the gradient signal dilutes with each step. LSTMs address this with gating mechanisms (input, forget, output gates), but the fundamental sequential bottleneck remains.

The transformer eliminates the sequential bottleneck by allowing every position in the sequence to attend directly to every other position. At each position, the model computes a query, key, and value representation. The attention score between position *i* and position *j* is the dot product of position *i*'s query with position *j*'s key, normalized across all positions, then used to weight a sum of values. Every pair of positions interacts in O(1) operations rather than O(n) sequential steps.

For orbital sequences, this matters when the maneuver signature spans multiple non-adjacent TLE epochs. An orbital inclination change from a plane-change burn may show a step in the RAAN residual feature several epochs before it appears in mean motion, and the strongest anomaly signal may be the *combination* of changes across three or four separated epochs. The LSTM can detect this if the hidden state retains the right information, but it cannot directly model the cross-epoch relationship. Self-attention can.

---

## The irregular sampling problem

The canonical transformer uses positional encodings based on position index: position 0 gets encoding `[sin(ω·0), cos(ω·0), ...]`, position 1 gets `[sin(ω·1), cos(ω·1), ...]`, and so on. This assumes uniform spacing — every position represents the same time interval. TLE sequences violate this assumption. A sequence of 30 daily-gridded TLE epochs is uniformly spaced, but many objects have irregular cadence: 2 TLEs on Monday, none Tuesday through Thursday, 4 on Friday.

There are two approaches to this problem:

**Grid to daily resolution first.** Lesson 1 introduced this: interpolate or fill to a uniform daily grid, then apply standard positional encoding by position index. The observation mask (which days had actual TLEs vs. were interpolated) becomes a feature in the input. This is the simplest approach and works well when the object has at least 50% coverage.

**Use continuous-time positional encoding.** Instead of indexing by position, encode the actual observation time as a continuous value. One approach: encode the epoch as the fractional day since start of the observation window, then use a learned Fourier-based encoding. This preserves the actual temporal structure even with highly irregular cadence, at the cost of implementation complexity.

For most LEO active satellites (which have good TLE coverage), daily gridding plus observation masking is sufficient. For debris objects with sparse coverage, continuous-time encoding is worth the complexity.

---

## Architecture: encoder-only transformer for classification

Maneuver detection is a classification task: given a 30-day window, output a binary label. The appropriate architecture is an encoder-only transformer — the same design as BERT — with a classification head on top of the sequence representation.

The full pipeline:

```python
import torch
import torch.nn as nn
import math

class OrbitalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class OrbitalTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,       # number of input features per epoch (e.g. 6)
        d_model: int = 64,     # transformer embedding dimension
        nhead: int = 4,        # number of attention heads
        num_layers: int = 2,   # number of encoder layers
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        seq_len: int = 30,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = OrbitalPositionalEncoding(d_model, max_len=seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # seq_len first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification token prepended to sequence, analogous to BERT [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_projection(x)           # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)                # (seq_len, batch, d_model)
        # Prepend CLS token
        cls = self.cls_token.expand(-1, x.size(1), -1)  # (1, batch, d_model)
        x = torch.cat([cls, x], dim=0)         # (seq_len+1, batch, d_model)
        x = self.pos_encoder(x)
        # Extend padding mask for CLS position
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(src_key_padding_mask.size(0), 1,
                                   dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = x[0]                      # (batch, d_model) — CLS token output
        return self.classifier(cls_output).squeeze(-1)  # (batch,)
```

The CLS token approach (borrowed from BERT) gives the transformer a dedicated position to accumulate the global sequence representation used for classification. The attention mechanism allows the CLS token to directly query all 30 TLE positions simultaneously, regardless of where in the sequence the maneuver signature appears.

The `src_key_padding_mask` handles observation gaps: positions where daily gridding produced no real observation are marked True in the mask, and the transformer ignores them in attention computation. This is the correct way to handle missing observations — not by imputing fake values but by masking them out.

---

## Attention head size and the orbital feature dimension

Orbital sequences have narrow feature vectors — the 6 features from Lesson 1 (Δn/Δt, Δe/Δt, Δi/Δt, Δω/Δt, RAAN residual rate, F10.7-normalized BSTAR) are a 6-dimensional input per timestep. This creates a mismatch with transformer architectures designed for large vocabularies or high-dimensional embeddings.

The solution is the input projection layer: a learned linear map from 6 dimensions to `d_model` (64 or 128 is usually appropriate for 30-epoch sequences). This gives the attention mechanism enough representational space to compute meaningful query-key products without overfitting. Do not use a transformer with `d_model` above 256 on this problem — the sequence is too short and the feature space too narrow to support it without extensive regularization.

The choice of `nhead` must divide `d_model` evenly. With `d_model=64`, `nhead=4` gives each head a 16-dimensional subspace, which is sufficient. With `d_model=64` and `nhead=8`, each head has only an 8-dimensional subspace — too small for this problem, and empirically worse than 4 heads.

---

## Masked pretraining for the label-scarce setting

One genuine advantage of transformers over LSTMs for this problem is the availability of masked pretraining, directly analogous to BERT's masked language modeling objective.

The label-scarce problem from Lesson 1 never fully disappears. Synthetic injection helps but introduces a distributional shift: the model may learn features of the synthetic injection process rather than of real maneuvers. Masked pretraining addresses this by using the unlabeled TLE histories themselves as the training signal.

The procedure:

1. Take any satellite's TLE history (no labels needed).
2. Randomly mask 15% of the daily epochs — replace the feature vector with a learned [MASK] token embedding.
3. Train the transformer to predict the masked epoch's features from the surrounding context.
4. Fine-tune on the maneuver detection task (synthetic labels) using the pretrained weights as initialization.

This uses the full Space-Track catalog — millions of TLE epochs, all unlabeled — to train the encoder to understand what normal orbital evolution looks like. By the time fine-tuning begins, the model already has a learned representation of orbital dynamics. The fine-tuning task (maneuver vs. not) then requires relatively few examples to converge.

In practice, pretraining on 10,000 objects for 6 months of history each (roughly 1.8 million training epochs) produces representations that fine-tune to better maneuver detection precision than training from scratch on synthetic data alone.

---

## Extracting attention weights for explainability

One practical advantage of the transformer over the LSTM is that attention weights are explicit. After training, you can extract what each attention head attends to when the model flags a maneuver.

```python
def get_attention_weights(model, x, src_key_padding_mask=None):
    """Extract attention weights from the first encoder layer."""
    model.eval()
    hooks = []
    attention_weights = {}

    def hook_fn(module, input, output):
        # TransformerEncoderLayer internals expose attn_output_weights
        # Use register_forward_hook on the self-attention sub-module
        attention_weights['layer0'] = output[1]  # (batch, nhead, seq+1, seq+1)

    # Register on the first layer's self-attention
    hook = model.transformer_encoder.layers[0].self_attn.register_forward_hook(hook_fn)
    hooks.append(hook)

    with torch.no_grad():
        _ = model(x, src_key_padding_mask)

    for h in hooks:
        h.remove()

    return attention_weights.get('layer0')
```

The resulting attention matrix has shape `(batch, nhead, seq_len+1, seq_len+1)`. Row 0 (the CLS token) shows which TLE epochs the model attends to most when making the classification decision. In validated experiments on ISS reboost events, the heads consistently assign high attention weight to the 1–3 epochs immediately following the reboost, where mean motion changes abruptly, plus secondary attention to the 3–5 quiet epochs before, establishing the baseline.

This interpretability is commercially relevant: a DoD customer can ask "why did you flag this object?" and you can show them the specific epochs that drove the decision. The LSTM's hidden state does not offer this.

---

## LSTM vs. transformer: when to use which

| Factor | LSTM | Transformer |
|---|---|---|
| Training set size | < 50K windows | > 100K windows |
| Sequence length | < 40 epochs | > 40 epochs |
| Interpretability needed | No | Yes (attention) |
| Inference latency target | < 1ms | 1–10ms |
| Pretraining available | No | Yes (masked autoencoder) |
| Implementation complexity | Low | Medium |

For a first product with a small training set and short sequences, LSTM is the right choice. Transformers outperform LSTMs when there is enough data to support their larger parameter counts and when the sequence is long enough for long-range attention to matter. The crossover point is roughly 100K training windows and 45+ epoch sequences — thresholds a production pipeline will reach after accumulating 12–18 months of synthetic injection data.

---

## Key Takeaways

- **Self-attention eliminates the sequential bottleneck.** Every TLE epoch can attend directly to every other epoch in O(1) operations, allowing the model to detect multi-epoch maneuver signatures that span non-adjacent positions — something the LSTM's hidden state must carry through all intermediate steps.
- **Use daily gridding with an observation mask rather than raw irregular cadence.** Standard positional encodings assume uniform spacing. Grid to daily resolution, mark missing days in a padding mask, and pass the mask to the transformer's attention computation.
- **The CLS token aggregates the sequence representation.** Prepend a learnable [CLS] token to the TLE sequence; the transformer's output at that position is used as the global classification representation, analogous to BERT.
- **Keep d_model small for orbital sequences.** 6-dimensional orbital features projected to d_model=64 with 4 attention heads is appropriate. Models larger than d_model=256 overfit on the available sequence lengths without extensive regularization.
- **Masked pretraining enables use of the entire unlabeled catalog.** Pre-train on millions of TLE epochs by masking random positions and reconstructing them; fine-tune on synthetic maneuver labels. This significantly improves generalization over training from synthetic labels alone.
- **Attention weights are commercially relevant explainability.** The CLS token's attention over TLE epochs shows which specific historical observations drove the classification decision — an interpretable audit trail for DoD customers who need to understand why an object was flagged.
- **The LSTM remains the right default for small, low-latency settings.** The transformer's advantages over LSTM materialize only above roughly 100K training windows and 45+ epoch sequences. Start with LSTM; migrate to transformer when the data and sequence length justify the additional complexity.

---

## Quiz

{{#quiz 02-transformers-orbital-sequences.toml}}
