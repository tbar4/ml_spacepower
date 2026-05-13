# Lesson 2: Conditional Probability and Bayes' Rule

**Module:** ML and Game Theory for Space Power — M01: Foundations
**Source:** *Bayesian Statistics the Fun Way* — Will Kurt, Chapters 3–5 (Conditional Probability, Bayes' Rule, and Sequential Updates); *Math for Deep Learning* — Ronald T. Kneusel, Chapter 2 (Probability and Conditional Probability)

---


<!-- toc -->

## Where this fits

In partially observable settings, which describes almost every real SSA scenario and every game with hidden information, an agent maintains a **belief**: a probability distribution over what it cannot directly observe. Every time a new observation arrives, that belief gets updated. The mechanism for that update is Bayes' rule. When you later read about belief states in POMDPs, reach probabilities in CFR, or opponent modeling in multi-agent RL, you are reading about Bayes' rule with domain-specific packaging. This lesson is the unpackaged version.

## A scenario: classifying an unknown radar contact

Your ground radar just flagged a new contact. You can measure the object's radar cross-section (RCS), which gives you some evidence about what type of object it might be. But RCS is noisy and imperfect: a debris fragment and a small satellite can produce similar returns.

Before the measurement, your catalog tells you that in this particular orbital regime, the objects are distributed like this:

| Object type       | Fraction in catalog |
|-------------------|---------------------|
| Active satellite  | 60%                 |
| Debris            | 30%                 |
| Rocket body       | 10%                 |

This is your starting belief. You believe there is a 60% chance the contact is a satellite, 30% chance it is debris, 10% chance it is a rocket body. You have not seen the RCS measurement yet.

Then the RCS measurement comes in. It shows a medium-small return. Your sensor physics models tell you how likely that specific measurement is for each object type:

| Object type       | P(this RCS reading \| object is this type) |
|-------------------|--------------------------------------------|
| Active satellite  | 0.70                                       |
| Debris            | 0.20                                       |
| Rocket body       | 0.40                                       |

Now you have new evidence. The question is: given that measurement, how should your beliefs change?

That question is Bayes' rule.

## What is conditional probability?

**Conditional probability** is the probability of one thing given that you know another thing has happened.

The notation is \(P(A \mid B)\), read as "probability of A given B." The vertical bar means "given that."

For your RCS example:
- \(P(\text{debris} \mid \text{medium-small RCS})\) means: "what is the probability that this is debris, given that we observed a medium-small RCS return?"
- \(P(\text{medium-small RCS} \mid \text{debris})\) means: "if this were debris, how likely is this particular RCS reading?"

These look similar but they are answering completely different questions. The first is what you want to know. The second is what your sensor model gives you. Bayes' rule connects them.

### Conditioning means restricting your universe

Here is a concrete way to think about conditional probability.

Imagine you have a catalog of 1,000 past contacts in this orbital regime:
- 600 active satellites
- 300 debris
- 100 rocket bodies

Of those 600 active satellites, suppose 420 produced a medium-small RCS reading (that is 70% of 600).
Of those 300 debris, suppose 60 produced a medium-small RCS reading (that is 20% of 300).
Of those 100 rocket bodies, suppose 40 produced a medium-small RCS reading (that is 40% of 100).

Now your sensor reports a medium-small RCS. **How many objects in the catalog showed that reading?**

420 + 60 + 40 = 520 objects produced a medium-small RCS reading.

Of those 520 objects, how many were active satellites? 420. So:

\(P(\text{active sat} \mid \text{medium-small RCS}) = 420 / 520 \approx 0.808\)

How many were debris? 60. So:

\(P(\text{debris} \mid \text{medium-small RCS}) = 60 / 520 \approx 0.115\)

How many were rocket bodies? 40. So:

\(P(\text{rocket body} \mid \text{medium-small RCS}) = 40 / 520 \approx 0.077\)

Your belief shifted. You started at 60% / 30% / 10%. After seeing the medium-small RCS, you are now at 80.8% / 11.5% / 7.7%. The measurement strongly favored active satellites (because satellites produce this return 70% of the time, while debris produce it only 20% of the time), so the satellite probability went up and debris went down.

---

## Independence

**Independence** is the special case where knowing one thing tells you nothing about another thing.

**Formal definition**: events A and B are independent if and only if:

\[ P(A \mid B) = P(A) \]

**Decoding**: "the probability of A, given that B happened, is the same as the probability of A before you knew about B." B is irrelevant to A. You can also write independence as \(P(A \cap B) = P(A) \times P(B)\) — which is where the product rule from lesson 1 comes from. Both formulations say the same thing.

### Conditional dependence

The opposite of independence is **conditional dependence**: knowing B changes your belief about A. In the RCS scenario, object type and RCS reading are dependent — knowing the type changes how likely you think a given RCS reading is, and knowing the reading changes how likely you think a given type is.

Dependence is the normal situation. Independence is a simplifying assumption you make when the dependence is negligible or when you lack the data to model it properly.

### The SSA independence question: two radars vs. one atmosphere

Consider two independent radar sites, Site A (Colorado) and Site B (Alaska), each measuring the same RSO. If their noise processes are independent, you can use the product rule: the probability of both sites producing measurement errors above threshold is \(P(\text{A error}) \times P(\text{B error})\). That product is small, which is why multi-site fusion reduces false alarm rates.

But independence fails when both sites share a common cause. **Correlated noise from the same atmospheric layer** is a real failure mode: if a large ionospheric disturbance affects the entire continental US, both Site A and Site B will experience elevated range errors simultaneously. Their errors are now dependent — \(P(\text{B error} \mid \text{A error}) > P(\text{B error})\) — and treating them as independent will underestimate the probability of simultaneous bad measurements at both sites.

The operational implication is significant: if you design a conjunction assessment protocol that requires "two independent radar confirmations" to flag a high-priority conjunction, and your two radars are correlated by shared atmosphere, you are getting less confirmation than you think. The assumption of independence is a model choice, and it should be tested rather than assumed.

```python
import torch

# Independent radars: P(both error) = product of marginal error rates
p_site_a_error = torch.tensor(0.05)
p_site_b_error = torch.tensor(0.04)

# Independence assumption: product rule
p_both_error_independent = p_site_a_error * p_site_b_error
print(f"P(both error | independent): {p_both_error_independent.item():.4f}")  # 0.0020

# Correlated radars: during ionospheric storm, P(B error | A error) is elevated
# Suppose during an ionospheric event, if A has an error, B has 40% error rate
p_b_given_a_error_correlated = torch.tensor(0.40)
p_both_error_correlated = p_site_a_error * p_b_given_a_error_correlated
print(f"P(both error | correlated):  {p_both_error_correlated.item():.4f}")   # 0.0200

# The correlated scenario produces 10x more simultaneous errors — a meaningful difference
# in a protocol that requires both radars to agree
ratio = p_both_error_correlated / p_both_error_independent
print(f"Ratio (correlated / independent): {ratio.item():.1f}x more simultaneous errors")
```

When independence fails and you do not know it, your probability estimates are wrong in a systematic direction — usually overconfident. This is the failure mode Will Kurt calls "naively combining evidence" in *Bayesian Statistics the Fun Way* Chapter 5.

---

## The total probability formula

Bayes' rule has a denominator — \(P(E)\) — that often looks mysterious. The **law of total probability** makes it concrete.

**The formula**:

\[ P(E) = \sum_{i} P(E \mid H_i) \cdot P(H_i) \]

**Decoding**:

- \(H_1, H_2, \ldots\): a complete, mutually exclusive set of hypotheses. "Complete" means at least one is true. "Mutually exclusive" means at most one is true. Together they partition the space of possibilities.
- \(P(E \mid H_i)\): the likelihood of the evidence under hypothesis \(i\).
- \(P(H_i)\): the prior probability of hypothesis \(i\).
- The sum adds up contributions to \(P(E)\) from each hypothesis, weighted by how probable that hypothesis is.

The law of total probability is the denominator in Bayes' rule because it answers: "what is the probability of seeing this evidence at all, summed over every possible explanation?" When you normalize the unnormalized posteriors, you are dividing by exactly this sum.

### Worked example with a third sensor type

Extend the catalog scenario. Now there are three object types in the catalog, and you have received a specific RCS reading. What is the total probability of that reading?

| Hypothesis \(H_i\) | Prior \(P(H_i)\) | Likelihood \(P(E \mid H_i)\) | Joint \(P(E, H_i)\) |
|---|---|---|---|
| Active satellite | 0.60 | 0.70 | 0.420 |
| Debris | 0.30 | 0.20 | 0.060 |
| Rocket body | 0.10 | 0.40 | 0.040 |
| **Total** | **1.00** | — | **0.520** |

\(P(E) = 0.420 + 0.060 + 0.040 = 0.520\)

This says: if you sampled a random contact from this orbital regime and ran the RCS sensor, you would get a medium-small reading 52% of the time — across all the different object types combined.

Now add a fourth object type, "rocket body fragment," that your updated catalog recently identified with prior probability 0.05. When you add a new hypothesis, you have to renormalize the priors (they must sum to 1) and include the new type's likelihood in the total probability sum.

```python
import torch

# Extended catalog with four object types
# Priors (must sum to 1 — renormalized to add rocket body fragment at 5%)
prior = torch.tensor([0.57, 0.285, 0.095, 0.05])
# Likelihoods P(medium-small RCS | each type)
# Rocket body fragments: small, tumbling — medium-small RCS is fairly common
likelihood = torch.tensor([0.70, 0.20, 0.40, 0.55])

# Joint probabilities P(E, H_i) = P(E|H_i) * P(H_i)
joint = prior * likelihood
print(f"Joint probabilities: {joint.tolist()}")

# Total probability of the evidence (denominator of Bayes' rule)
p_evidence = joint.sum()
print(f"P(evidence) = {p_evidence.item():.4f}")

# Posteriors
posterior = joint / p_evidence
labels = ["Active sat", "Debris", "Rocket body", "RB fragment"]
print("\nPosterior beliefs:")
for label, p in zip(labels, posterior.tolist()):
    print(f"  {label}: {p:.3f}")

# Note: the total probability changes when you add a new hypothesis type.
# If you had computed P(E) with only 3 types and now have 4, your normalization
# constant is different. This is why it matters to have a complete hypothesis set.
```

The practical lesson: the denominator of Bayes' rule is only correct when your hypothesis set is complete. If your catalog is missing a category of objects (say, fractured rocket body debris is not yet cataloged), your posteriors will be systematically wrong — they will distribute probability among the known types even when the true answer is "none of the above."

---

## Bayes' rule: the formula

Bayes' rule is just the efficient way to do the catalog calculation you just did, without needing an actual catalog.

Here it is, first in words:

> **Updated belief = (how well the evidence fits this hypothesis) × (prior belief) / (total probability of this evidence)**

Now with symbols. We call the hypothesis H (e.g., "this is debris") and the evidence E (e.g., "medium-small RCS reading"):

\[ P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)} \]

**Decoding each piece:**

**\(P(H \mid E)\)**: The **posterior**. This is what we want: the probability of hypothesis H after seeing evidence E. "Posterior" means "after." Before the evidence, we had a prior. After the evidence, we have a posterior.

**\(P(H)\)**: The **prior**. This is what we believed about H before seeing the evidence. In the catalog example, this was 0.60 for active satellite, 0.30 for debris, 0.10 for rocket body. The word "prior" means "before."

**\(P(E \mid H)\)**: The **likelihood**. This is how probable the evidence is, assuming H is true. Your sensor model gives you this. "If this contact is debris, how likely is this RCS reading?" That is a likelihood.

**\(P(E)\)**: The **marginal probability of the evidence**, sometimes called the normalizing constant. This is the total probability of seeing this evidence, regardless of which hypothesis is true. In the catalog, it was 520/1000 = 0.52 (the fraction of contacts that produced a medium-small reading at all).

You calculate \(P(E)\) by summing over all hypotheses:
\[ P(E) = P(E \mid H_1) \cdot P(H_1) + P(E \mid H_2) \cdot P(H_2) + P(E \mid H_3) \cdot P(H_3) + \ldots \]

This just says: the total probability of seeing this evidence is the sum over each possible explanation, weighted by how likely each explanation was.

**The shortcut**: you usually do not compute \(P(E)\) directly. Instead, compute the numerator (\(P(E \mid H) \cdot P(H)\)) for every hypothesis, then divide each by the sum. That sum is \(P(E)\), and you get it for free as a byproduct of normalization.

## Applying Bayes' rule step by step

Let us walk through the calculation systematically.

**Step 1: Write down your priors.**

| Hypothesis \(H\) | Prior \(P(H)\) |
|---------------------|------------------|
| Active satellite    | 0.60             |
| Debris              | 0.30             |
| Rocket body         | 0.10             |

**Step 2: Write down the likelihoods.** For each hypothesis, how probable is the evidence (medium-small RCS)?

| Hypothesis \(H\) | Likelihood \(P(E \mid H)\) |
|---------------------|-------------------------------|
| Active satellite    | 0.70                          |
| Debris              | 0.20                          |
| Rocket body         | 0.40                          |

**Step 3: Multiply prior by likelihood for each hypothesis.** This gives you the unnormalized posterior.

| Hypothesis          | Prior × Likelihood    | Unnormalized posterior |
|---------------------|-----------------------|------------------------|
| Active satellite    | 0.60 × 0.70 = 0.420   | 0.420                  |
| Debris              | 0.30 × 0.20 = 0.060   | 0.060                  |
| Rocket body         | 0.10 × 0.40 = 0.040   | 0.040                  |
| **Total**           |                       | **0.520**              |

**Step 4: Divide each unnormalized posterior by the total.** The total (0.520) is \(P(E)\). Dividing by it makes the posteriors sum to 1.

| Hypothesis          | Posterior \(P(H \mid E)\) |
|---------------------|------------------------------|
| Active satellite    | 0.420 / 0.520 ≈ **0.808**   |
| Debris              | 0.060 / 0.520 ≈ **0.115**   |
| Rocket body         | 0.040 / 0.520 ≈ **0.077**   |
| **Total**           | **1.000**                   |

These match the catalog calculation from before. Active satellite went from 60% to 81%. Debris dropped from 30% to 12%. The RCS measurement was informative: it was much more likely under "active satellite" than under "debris," so it pushed probability mass toward the satellite hypothesis.

## Code

```python
import torch

# Step 1: Priors
prior = torch.tensor([0.60, 0.30, 0.10])  # active sat, debris, rocket body

# Step 2: Likelihoods P(medium-small RCS | each hypothesis)
likelihood = torch.tensor([0.70, 0.20, 0.40])

# Step 3: Unnormalized posterior = prior * likelihood
unnormalized = prior * likelihood
print(f"Unnormalized: {unnormalized.tolist()}")  # [0.42, 0.06, 0.04]

# Step 4: Normalize so they sum to 1
posterior = unnormalized / unnormalized.sum()
print(f"Posterior:    {posterior.tolist()}")
# approximately [0.808, 0.115, 0.077]

labels = ["Active sat", "Debris    ", "Rocket body"]
for label, p in zip(labels, posterior.tolist()):
    print(f"  {label}: {p:.3f}")
```

That four-line calculation is the complete Bayes update. Every belief update you will see in POMDPs and multi-agent RL follows this same structure.

## Sequential updates: learning from multiple observations

Bayes' rule can be applied repeatedly. Each posterior becomes the new prior for the next observation.

Suppose after the RCS reading, you also get a photometric brightness measurement. Your sensor model says that this specific brightness reading would be observed with these probabilities:

| Hypothesis       | P(this brightness \| hypothesis) |
|------------------|----------------------------------|
| Active satellite | 0.30 (satellites tend to be brighter) |
| Debris           | 0.50 (debris often has tumbling glints)|
| Rocket body      | 0.40                                  |

Use the previous posterior as the new prior:

```python
# Previous posterior becomes the new prior
new_prior = posterior  # [0.808, 0.115, 0.077]

# New likelihood from brightness measurement
likelihood_2 = torch.tensor([0.30, 0.50, 0.40])

# Bayes update (same four steps as before)
unnormalized_2 = new_prior * likelihood_2
posterior_2 = unnormalized_2 / unnormalized_2.sum()

for label, p in zip(labels, posterior_2.tolist()):
    print(f"  {label}: {p:.3f}")
```

The brightness reading favored debris (50% likely for debris vs 30% for active satellite), so the active satellite probability will drop somewhat and debris will recover. Two observations have nudged our belief, and we could apply a third, fourth, and so on.

---

## Three measurements: order independence of Bayesian updates

Now add a third measurement: the **albedo** reading from a photometric sensor. Albedo measures how reflective the object is. Your sensor model provides:

| Hypothesis       | P(this albedo \| hypothesis) |
|------------------|------------------------------|
| Active satellite | 0.60 (solar panels are highly reflective) |
| Debris           | 0.25 (rough, irregular surfaces are darker) |
| Rocket body      | 0.50 |

A key property of Bayes' rule is that **the order in which you apply measurements does not change the final result**. Whether you apply RCS first, then brightness, then albedo — or albedo first, then RCS, then brightness — you will arrive at the same posterior. This is not obvious but follows directly from the commutativity of multiplication: the joint probability of all measurements and the hypothesis is the same product regardless of the order you write the terms.

```python
import torch

# Starting prior
prior = torch.tensor([0.60, 0.30, 0.10])
labels = ["Active sat", "Debris    ", "Rocket body"]

# Three measurement likelihoods
likelihood_rcs        = torch.tensor([0.70, 0.20, 0.40])
likelihood_brightness = torch.tensor([0.30, 0.50, 0.40])
likelihood_albedo     = torch.tensor([0.60, 0.25, 0.50])

def bayes_update(prior, likelihood):
    """One step of Bayes update: multiply, normalize, return posterior."""
    unnorm = prior * likelihood
    return unnorm / unnorm.sum()

# Order 1: RCS -> brightness -> albedo
p_order1 = prior
p_order1 = bayes_update(p_order1, likelihood_rcs)
p_order1 = bayes_update(p_order1, likelihood_brightness)
p_order1 = bayes_update(p_order1, likelihood_albedo)

# Order 2: albedo -> brightness -> RCS
p_order2 = prior
p_order2 = bayes_update(p_order2, likelihood_albedo)
p_order2 = bayes_update(p_order2, likelihood_brightness)
p_order2 = bayes_update(p_order2, likelihood_rcs)

# Order 3: all three likelihoods multiplied at once, then normalized
# This is mathematically equivalent to doing them in any sequence
unnorm_all = prior * likelihood_rcs * likelihood_brightness * likelihood_albedo
p_order3   = unnorm_all / unnorm_all.sum()

print("Posterior after all three measurements:")
print(f"  Order (RCS, bright, albedo):  {p_order1.tolist()}")
print(f"  Order (albedo, bright, RCS):  {p_order2.tolist()}")
print(f"  All at once:                  {p_order3.tolist()}")
# All three should be identical (up to floating point rounding)

print("\nMax difference between orders:", (p_order1 - p_order2).abs().max().item())
# Should be < 1e-6

print("\nFinal belief state:")
for label, p in zip(labels, p_order1.tolist()):
    print(f"  {label}: {p:.4f}")
```

Order independence has a practical consequence for SSA pipelines: you do not need to wait for all sensors to report before starting to update your belief. Each sensor measurement can be processed as it arrives, and the running posterior reflects all evidence received so far. A tracking filter that receives RCS from Radar A, then albedo from an optical telescope, then a second RCS from Radar B produces the same final belief as one that batches all three.

The only caveat: each measurement must be **conditionally independent** given the hypothesis — meaning the measurement noise of one sensor does not depend on the measurement noise of another, once you know the object type. When sensors share common-mode errors (same atmosphere, same timing reference), this conditional independence breaks down and order independence no longer holds cleanly.

---

## When Bayes is hard: strong priors dominate

Bayes' rule always works correctly. But it can produce results that feel wrong until you understand the mechanism.

**The problem**: when the prior is very strong, a small number of observations cannot overcome it. This is not a bug — it is the mathematically correct behavior. But it has serious operational implications.

### A numerical demonstration

Suppose your catalog assigns a 99% prior probability that an object is an active satellite and only 1% prior probability that it is debris. This is a strong prior. Now you receive an RCS measurement that is much more consistent with debris than with satellites:

| Hypothesis | Prior | Likelihood (this RCS) | Unnormalized |
|---|---|---|---|
| Active satellite | 0.99 | 0.05 | 0.0495 |
| Debris | 0.01 | 0.80 | 0.0080 |
| **Total** | | | **0.0575** |

Posterior: Active satellite = 0.0495 / 0.0575 ≈ **86%**, Debris ≈ **14%**.

Even though the likelihood ratio strongly favors debris (0.80 vs 0.05 — a 16-to-1 ratio), the strong prior keeps the satellite hypothesis in front. The prior was 99-to-1 for satellite; the likelihood is 16-to-1 for debris; the posterior is about 6-to-1 for satellite. The prior dominated.

### How many observations to overcome a strong prior

To overcome a prior of 99:1 (\(P(H) = 0.99\)), you need evidence strong enough that the cumulative likelihood ratio exceeds 99. If each observation has a likelihood ratio of 16:1 in favor of debris, you need approximately:

\[ \text{number of observations} \approx \frac{\log(99)}{\log(16)} \approx \frac{4.6}{2.8} \approx 1.6 \]

So about two observations with that likelihood ratio would flip the belief. But if the likelihood ratio per observation is weaker — say 2:1 — you need:

\[ \frac{\log(99)}{\log(2)} \approx \frac{4.6}{0.69} \approx 6.6 \]

About seven observations with a modest 2:1 likelihood ratio are needed.

### The catalog-is-wrong failure mode

This has an immediate operational consequence: **if your SSA catalog is wrong, sensors cannot easily correct it**.

Suppose a satellite has been mislabeled as an active satellite in the catalog when it actually went dead (stopped maneuvering) two years ago. The catalog entry has been reinforced by thousands of routine observations. The prior on "active satellite" is effectively 0.9999 — extremely strong, because the catalog entry has been confirmed so many times.

Now a new observation comes in that looks inconsistent with an active satellite (no RF emission detected, unexpected brightness change). Bayes' rule will update the belief, but the update will be tiny. It takes many anomalous observations, processed by an analyst who is looking for the anomaly, to overcome a deeply entrenched catalog entry. This is not an algorithmic failure — it is Bayes' rule working correctly given the evidence history. The fix is not a different algorithm; it is a process that actively hunts for inconsistencies rather than waiting for the posterior to shift organically.

```python
import torch

def bayes_update(prior, likelihood):
    unnorm = prior * likelihood
    return unnorm / unnorm.sum()

# Strong prior: catalog says 99% active satellite
prior = torch.tensor([0.99, 0.01])
labels = ["Active sat", "Debris"]

# Each observation: likelihood ratio 2-to-1 in favor of debris
# P(this observation | active sat) = 0.33, P(this observation | debris) = 0.67
likelihood_per_obs = torch.tensor([0.33, 0.67])

print("Belief evolution with strong prior (99% active sat):")
print(f"  Start:           Active={prior[0]:.4f}, Debris={prior[1]:.4f}")

belief = prior.clone()
for obs_count in range(1, 15):
    belief = bayes_update(belief, likelihood_per_obs)
    if obs_count in [1, 2, 3, 5, 7, 10, 14]:
        print(f"  After {obs_count:2d} obs:    Active={belief[0]:.4f}, Debris={belief[1]:.4f}")

# Show how a much stronger likelihood ratio accelerates the update
print("\nSame prior, but stronger likelihood ratio (16:1 for debris):")
belief_strong = prior.clone()
likelihood_strong = torch.tensor([0.05, 0.80])
# Renormalize: we only care about the ratio, not the absolute values
# (the normalization step handles this)
for obs_count in range(1, 5):
    belief_strong = bayes_update(belief_strong, likelihood_strong)
    print(f"  After {obs_count} obs:    Active={belief_strong[0]:.4f}, Debris={belief_strong[1]:.4f}")
```

The output shows slow convergence with weak evidence and fast convergence with strong evidence — both are correct Bayesian behavior. The implication for system design: if you want sensors to correct catalog errors quickly, you need either very informative sensor models (high likelihood ratios) or active outlier detection that flags objects whose posterior is evolving rapidly away from their catalog entry.

---

## The base rate trap: the most common mistake in probabilistic reasoning

There is one error so common and so important that it deserves its own section.

**The mistake**: ignoring the prior and treating the likelihood as if it were the posterior.

Suppose someone tells you "our sensor has a 90% detection rate for rocket bodies" and you detect a signal. A naive interpretation: "90% chance this is a rocket body."

**This is almost always wrong.** The correct interpretation requires the prior.

If rocket bodies represent only 1% of all objects in this orbital regime (prior = 0.01), then even with a 90% detection rate, most signals will not be rocket bodies. The vast majority of the time, the sensor is detecting one of the 99% non-rocket-body objects at whatever rate that detection applies.

Bayes' rule mechanically prevents this error because the prior appears explicitly in the formula. The moment you write down \(P(H) = 0.01\) before doing the calculation, you cannot forget it.

For SSA, this matters practically. If you are looking for adversarial satellite maneuvers and only 1 in 1,000 maneuvers is adversarial (with 999 being routine station-keeping), a detector that is 95% accurate at identifying adversarial maneuvers will still produce many false positives if you ignore the base rate.

## Why this matters going forward

In partially observable games, an agent cannot see the full game state. All it has is its own actions and the observations it has received. At each step it maintains a belief over what the hidden state might be, and it updates that belief using Bayes' rule every time a new observation arrives. This is the belief state in a POMDP.

CFR in extensive-form games uses "reach probabilities" that track, for each decision point, how likely it is that the game reached this point under a particular strategy. These are a form of conditional probability, and updating them follows the same logic you just practiced.

When you eventually read code that says `belief_state.update(observation)` or `reach_prob *= policy[action]`, you will know what is happening inside those lines.

---

## Key Takeaways

- **Conditional probability restricts your universe.** \(P(A \mid B)\) asks: among all worlds where B is true, how many also have A? This mental model — filtering down to the relevant subset — is the right way to reason about any sensor measurement or game observation that updates your beliefs.
- **Independence is a model choice, not a fact.** Two sensors may be independent under normal conditions and highly correlated during atmospheric events. Treating correlated measurements as independent underestimates the probability of simultaneous failures. Always ask what shared causes could violate your independence assumptions.
- **The law of total probability makes the Bayes denominator concrete.** \(P(E) = \sum_i P(E \mid H_i) P(H_i)\) is not a formula to memorize — it is the answer to "what fraction of all contacts would produce this reading?" If your hypothesis set is incomplete, your denominator is wrong, and so are all your posteriors.
- **Sequential Bayesian updates are order-independent** (given conditionally independent measurements). You can apply observations as they arrive without waiting to batch them. This is what makes online tracking filters practical: each new sensor report is a Bayes step, and the running posterior is always the best current estimate.
- **Strong priors resist correction.** A catalog entry reinforced by thousands of observations may take dozens of anomalous readings to overturn. This is mathematically correct but operationally dangerous: catalog errors persist. Active anomaly detection — looking for objects whose posterior is drifting unexpectedly — matters more than assuming the catalog will self-correct through routine observations.
- **The base-rate error is the most common mistake.** Likelihoods are not posteriors. A sensor that is "90% accurate" does not produce 90% correct classifications if the prior for the target class is 1%. Bayes' rule with an explicit prior is the only reliable protection against this mistake.

---

## Quiz

{{#quiz 02-conditional-probability-bayes.toml}}
