# Lesson 6: Regularization and Model Evaluation

**Module:** Neural Networks as Function Approximators — M02
**Source:** Goodfellow et al. "Deep Learning" Chapters 7 and 11; Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"; Ioffe & Szegedy (2015) "Batch Normalization: Accelerating Deep Network Training"; PyTorch documentation `nn.Dropout`, `nn.BatchNorm1d`

---


<!-- toc -->

## Where this fits

Lesson 4 built the training loop: forward pass, compute loss, backward pass, optimizer step, repeat. That loop works — but for the applications in this curriculum, following it naively produces models that appear to work during training but fail in deployment.

The problem is overfitting: a model that memorizes the training data rather than learning the underlying pattern. Overfitting is a constant threat in module 9's maneuver detection setting, where positive training examples are scarce (a few hundred real events, supplemented by synthetic injection), but it is equally relevant everywhere in this curriculum where simulation data is used to train real-world models.

This lesson covers the tools that prevent overfitting and the evaluation practices that detect it. These are not advanced topics — they are the minimum professional practice for any model that will be used on data it has not seen before.

---

## The overfitting problem

A model overfits when it learns the noise and idiosyncrasies of the training set rather than the underlying signal. The signature is a widening gap between training loss and validation loss: training loss keeps decreasing while validation loss flattens or rises.

```
Epoch   Train Loss   Val Loss
  10      0.42        0.44
  20      0.31        0.38
  30      0.22        0.39   ← gap opening
  40      0.15        0.44   ← overfitting
  50      0.10        0.51   ← severe overfitting
```

The correct model to deploy is from epoch 20–25, not epoch 50. Without a validation set, you cannot detect this gap.

**Why it happens**: A neural network has enough parameters to memorize any training set. Given 1,000 training examples and a 100,000-parameter network, the network can assign exactly the right output to every training example by memorizing each one. This does not require learning anything about the underlying relationship — and a model that memorized training examples generalizes poorly to new ones.

The gap between training performance and generalization performance is measured by the **generalization gap** = validation loss − training loss. The goal of regularization is to minimize this gap.

---

## Train, validation, and test splits

The first and most important regularization tool is a proper data split. Every ML project requires three non-overlapping datasets:

**Training set** — the data the model sees during gradient descent. Loss is computed on this set; weights are updated based on this loss.

**Validation set** — the data the model never trains on, used to monitor generalization during training. Use this to select hyperparameters, choose when to stop training, and compare different model architectures. Typically 10–20% of total data.

**Test set** — the data that is touched exactly once, after all training and architecture decisions are finalized, to report final performance. The test set is the honest performance estimate. If you use it to make training decisions (even once), it is no longer honest — it has effectively become another validation set.

Common mistake: performing hyperparameter search, selecting the best model based on test performance, and reporting that as the final result. This is data leakage; the test set should never influence any decision.

For Module 9's maneuver detection problem, the split is explicit in Lesson 1:
- **Training set**: synthetic maneuver injection into debris/quiet-period TLE histories
- **Validation set**: a held-out portion of the synthetic data, stratified by object class
- **Test set**: real labeled maneuver events (ISS reboosts, DISCOS-documented events) — reserved for final evaluation only, never used during training or validation

This split reflects the honest evaluation requirement for a product: train on synthetic, validate on synthetic, evaluate generalization on real.

---

## Dropout

Dropout is the most widely used regularization technique for neural networks. During training, each neuron is randomly set to zero with probability `p` (the dropout rate) at each forward pass. During inference, all neurons are active and their outputs are scaled by `(1 - p)` to maintain the expected output magnitude.

```python
class RegularizedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),   # ← applied after activation
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

**Important**: `nn.Dropout` is automatically disabled during evaluation. You must call `model.eval()` before inference and `model.train()` before resuming training:

```python
model.train()
for batch in train_loader:
    # dropout is active
    ...

model.eval()
with torch.no_grad():
    # dropout is disabled — deterministic predictions
    val_preds = model(val_x)
```

Forgetting `model.eval()` during inference is one of the most common bugs in PyTorch code. Validation metrics measured with dropout active will be lower than deployment performance.

**Why dropout works**: Dropout prevents co-adaptation — a pattern where groups of neurons collectively memorize a training example by each learning one piece of it. By randomly disabling neurons, dropout forces each neuron to be independently useful. The model that emerges is equivalent to an ensemble of thinned networks sharing parameters.

Typical dropout rates: 0.1–0.3 after fully-connected layers, 0.5 for heavily regularized models. Do not apply dropout to the final classification layer.

---

## L2 weight decay

Weight decay adds a penalty proportional to the squared magnitude of all model weights to the loss function:

```
total_loss = task_loss + λ * Σ ||w_i||²
```

This penalizes large weights, which correspond to neurons that have learned to rely heavily on specific input features — a form of memorization. In PyTorch, weight decay is applied through the optimizer:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

The `weight_decay` parameter is λ in the formula above. Values between `1e-5` and `1e-3` are typical. Weight decay and dropout are complementary and can be used together.

---

## Batch normalization

Batch normalization normalizes the activations of each layer to have zero mean and unit variance across the batch dimension, then applies a learned scale and shift:

```python
self.net = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.BatchNorm1d(hidden_size),   # ← after linear, before activation
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, 1),
)
```

Batch norm is most useful for:
- Stabilizing training on datasets with features at very different scales (which orbital element features often are — mean motion in rev/day vs. inclination in degrees vs. eccentricity dimensionless)
- Allowing larger learning rates without instability
- Providing mild regularization

Batch norm has a subtlety analogous to dropout: `model.eval()` switches it from using batch statistics to using running statistics accumulated during training. Always call `model.eval()` before inference.

For small batches (fewer than 16 examples), batch norm statistics are noisy. Use `nn.LayerNorm` instead, which normalizes across features rather than the batch dimension and is stable at any batch size.

---

## Early stopping

Early stopping is the simplest effective regularization technique: stop training when validation loss stops improving.

```python
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_state_dict = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best(self, model: nn.Module):
        """Load the best checkpoint after training ends."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

# Usage
early_stop = EarlyStopping(patience=15)
for epoch in range(max_epochs):
    train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    if early_stop.step(val_loss, model):
        print(f"Early stop at epoch {epoch}")
        break
early_stop.restore_best(model)  # restore weights from best validation epoch
```

The `patience` hyperparameter controls how many epochs of non-improvement to tolerate before stopping. A value of 10–20 is typical — enough to wait out temporary validation loss plateaus from learning rate fluctuations, but not so long that severe overfitting accumulates.

The critical detail: save and restore the model from the best validation epoch, not the final epoch. Without this, early stopping detects overfitting but still deploys the overfit model.

---

## Evaluation metrics for imbalanced classification

Accuracy is the wrong metric for maneuver detection. The public TLE catalog contains ~25,000 objects with coverage going back years — the vast majority of 30-day windows contain no maneuver. Even if only 1% of windows contain maneuvers, a classifier that always predicts "no maneuver" achieves 99% accuracy while being completely useless.

Use instead:

**Precision** = TP / (TP + FP): of the windows flagged as maneuvers, what fraction actually were? A low precision product will flood the operator with false alarms and be ignored.

**Recall** = TP / (TP + FN): of the actual maneuver windows, what fraction were detected? A low recall product misses the events it exists to detect.

**F1 score** = 2 * (Precision * Recall) / (Precision + Recall): harmonic mean, appropriate when you need a single number but both precision and recall matter.

**AUC-ROC**: area under the receiver operating characteristic curve. Measures discrimination ability independent of threshold choice. Useful for comparing models; not sufficient for reporting product performance.

**Operational metrics** (Module 9, Lesson 1 discusses these in detail): detection latency (days after maneuver until detection), miss rate by maneuver size, false alarm rate per object per month. These are the metrics a DoD customer evaluates when deciding whether to pay for the product.

```python
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_binary_classifier(model, loader, threshold=0.5):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    print(classification_report(all_labels, all_preds,
                                  target_names=['no maneuver', 'maneuver']))
    print(f"AUC-ROC: {roc_auc_score(all_labels, all_probs):.4f}")
```

---

## Practical checklist

For any neural network applied to a real classification task:

1. Split data into train / val / test *before* any training decision. Never look at test set until final evaluation.
2. Train with dropout and/or weight decay.
3. Call `model.eval()` before validation/test inference; call `model.train()` before each training epoch.
4. Use early stopping with checkpoint restoration.
5. Report precision, recall, F1, and AUC — not accuracy — for imbalanced classes.
6. For production deployment, characterize the threshold: report the precision-recall tradeoff curve and let the operator choose where to operate on it based on their false alarm tolerance.

---

## Key Takeaways

- **The validation set detects overfitting; the test set measures final generalization.** Use the validation set for all training decisions (early stopping, hyperparameter selection); touch the test set exactly once at the end.
- **Dropout prevents co-adaptation by randomly disabling neurons during training.** Call `model.eval()` to disable dropout during inference — forgetting this is one of the most common PyTorch bugs.
- **L2 weight decay penalizes large weights, reducing memorization of training examples.** Applied via `weight_decay` in the optimizer; use values between 1e-5 and 1e-3.
- **Batch normalization stabilizes training and allows larger learning rates.** Most useful when input features have widely different scales — orbital elements (mean motion, eccentricity, inclination) are a good example. Also requires `model.eval()` to switch to running statistics during inference.
- **Early stopping with checkpoint restoration prevents deploying an overfit model.** Save model weights whenever validation loss improves; restore best checkpoint when training stops.
- **Accuracy is the wrong metric for imbalanced classification.** Use precision, recall, F1, and AUC-ROC. For operational deployment, characterize the precision-recall tradeoff curve and let the operator choose the operating point.

---

## Quiz

{{#quiz 06-regularization-and-evaluation.toml}}
