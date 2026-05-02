# Lesson 2: Conditional Probability and Bayes' Rule

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

| Object type       | P(this RCS reading | object is this type) |
|-------------------|------------------------------------------------------|
| Active satellite  | 0.70                                                 |
| Debris            | 0.20                                                 |
| Rocket body       | 0.40                                                 |

Now you have new evidence. The question is: given that measurement, how should your beliefs change?

That question is Bayes' rule.

## What is conditional probability?

**Conditional probability** is the probability of one thing given that you know another thing has happened.

The notation is \\(P(A \mid B)\\), read as "probability of A given B." The vertical bar means "given that."

For your RCS example:
- \\(P(\text{debris} \mid \text{medium-small RCS})\\) means: "what is the probability that this is debris, given that we observed a medium-small RCS return?"
- \\(P(\text{medium-small RCS} \mid \text{debris})\\) means: "if this were debris, how likely is this particular RCS reading?"

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

\\(P(\text{active sat} \mid \text{medium-small RCS}) = 420 / 520 \approx 0.808\\)

How many were debris? 60. So:

\\(P(\text{debris} \mid \text{medium-small RCS}) = 60 / 520 \approx 0.115\\)

How many were rocket bodies? 40. So:

\\(P(\text{rocket body} \mid \text{medium-small RCS}) = 40 / 520 \approx 0.077\\)

Your belief shifted. You started at 60% / 30% / 10%. After seeing the medium-small RCS, you are now at 80.8% / 11.5% / 7.7%. The measurement strongly favored active satellites (because satellites produce this return 70% of the time, while debris produce it only 20% of the time), so the satellite probability went up and debris went down.

## Bayes' rule: the formula

Bayes' rule is just the efficient way to do the catalog calculation you just did, without needing an actual catalog.

Here it is, first in words:

> **Updated belief = (how well the evidence fits this hypothesis) × (prior belief) / (total probability of this evidence)**

Now with symbols. We call the hypothesis H (e.g., "this is debris") and the evidence E (e.g., "medium-small RCS reading"):

\\[ P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)} \\]

**Decoding each piece:**

**\\(P(H \mid E)\\)**: The **posterior**. This is what we want: the probability of hypothesis H after seeing evidence E. "Posterior" means "after." Before the evidence, we had a prior. After the evidence, we have a posterior.

**\\(P(H)\\)**: The **prior**. This is what we believed about H before seeing the evidence. In the catalog example, this was 0.60 for active satellite, 0.30 for debris, 0.10 for rocket body. The word "prior" means "before."

**\\(P(E \mid H)\\)**: The **likelihood**. This is how probable the evidence is, assuming H is true. Your sensor model gives you this. "If this contact is debris, how likely is this RCS reading?" That is a likelihood.

**\\(P(E)\\)**: The **marginal probability of the evidence**, sometimes called the normalizing constant. This is the total probability of seeing this evidence, regardless of which hypothesis is true. In the catalog, it was 520/1000 = 0.52 (the fraction of contacts that produced a medium-small reading at all).

You calculate \\(P(E)\\) by summing over all hypotheses:
\\[ P(E) = P(E \mid H_1) \cdot P(H_1) + P(E \mid H_2) \cdot P(H_2) + P(E \mid H_3) \cdot P(H_3) + \ldots \\]

This just says: the total probability of seeing this evidence is the sum over each possible explanation, weighted by how likely each explanation was.

**The shortcut**: you usually do not compute \\(P(E)\\) directly. Instead, compute the numerator (\\(P(E \mid H) \cdot P(H)\\)) for every hypothesis, then divide each by the sum. That sum is \\(P(E)\\), and you get it for free as a byproduct of normalization.

## Applying Bayes' rule step by step

Let us walk through the calculation systematically.

**Step 1: Write down your priors.**

| Hypothesis \\(H\\) | Prior \\(P(H)\\) |
|---------------------|------------------|
| Active satellite    | 0.60             |
| Debris              | 0.30             |
| Rocket body         | 0.10             |

**Step 2: Write down the likelihoods.** For each hypothesis, how probable is the evidence (medium-small RCS)?

| Hypothesis \\(H\\) | Likelihood \\(P(E \mid H)\\) |
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

**Step 4: Divide each unnormalized posterior by the total.** The total (0.520) is \\(P(E)\\). Dividing by it makes the posteriors sum to 1.

| Hypothesis          | Posterior \\(P(H \mid E)\\) |
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

| Hypothesis       | P(this brightness | hypothesis) |
|------------------|--------------------------------------|
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

This is exactly what a tracking filter does in SSA: it takes a stream of sensor measurements and updates a probability distribution over the object's state at each step.

## The base rate trap: the most common mistake in probabilistic reasoning

There is one error so common and so important that it deserves its own section.

**The mistake**: ignoring the prior and treating the likelihood as if it were the posterior.

Suppose someone tells you "our sensor has a 90% detection rate for rocket bodies" and you detect a signal. A naive interpretation: "90% chance this is a rocket body." 

**This is almost always wrong.** The correct interpretation requires the prior.

If rocket bodies represent only 1% of all objects in this orbital regime (prior = 0.01), then even with a 90% detection rate, most signals will not be rocket bodies. The vast majority of the time, the sensor is detecting one of the 99% non-rocket-body objects at whatever rate that detection applies.

Bayes' rule mechanically prevents this error because the prior appears explicitly in the formula. The moment you write down \\(P(H) = 0.01\\) before doing the calculation, you cannot forget it.

For SSA, this matters practically. If you are looking for adversarial satellite maneuvers and only 1 in 1,000 maneuvers is adversarial (with 999 being routine station-keeping), a detector that is 95% accurate at identifying adversarial maneuvers will still produce many false positives if you ignore the base rate.

## Why this matters going forward

In partially observable games, an agent cannot see the full game state. All it has is its own actions and the observations it has received. At each step it maintains a belief over what the hidden state might be, and it updates that belief using Bayes' rule every time a new observation arrives. This is the belief state in a POMDP.

CFR in extensive-form games uses "reach probabilities" that track, for each decision point, how likely it is that the game reached this point under a particular strategy. These are a form of conditional probability, and updating them follows the same logic you just practiced.

When you eventually read code that says `belief_state.update(observation)` or `reach_prob *= policy[action]`, you will know what is happening inside those lines.

## Quiz

{{#quiz 02-conditional-probability-bayes.toml}}
