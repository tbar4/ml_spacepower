# Lesson 2: Conditional probability and Bayes' rule

## Where this fits

Multi-agent and partially observable settings (which is most of what we care about) all involve some form of belief update: you observe something, and you adjust your model of the world based on that observation. Bayes' rule is the formal mechanism for that update. When you read about belief states in POMDPs, opponent modeling in extensive-form games, posterior policies, or sensor fusion in SSA, you are reading about Bayes' rule wearing a costume. Get this lesson and you have a tool that you'll keep reaching for.

## Concept

**Conditional probability** answers a "given that I know X happened, what's the probability of Y?" question. Written \\(P(Y \mid X)\\), and read aloud as "probability of Y given X."

A simple example. Pick a person at random. The probability they're a parent is some number, call it \\(P(\text{parent})\\). The probability they're a parent given that they own a minivan is much higher, call it \\(P(\text{parent} \mid \text{owns minivan})\\). Knowing the minivan information shifts the probability. That shift is the entire point of conditioning.

**Independence** is when knowing one thing doesn't shift the probability of another. \\(P(Y \mid X) = P(Y)\\) means "knowing X tells you nothing about Y." Most things in life are not independent, which is why Bayes is useful.

**Bayes' rule** is what you use when you know \\(P(\text{evidence} \mid \text{hypothesis})\\) and you want \\(P(\text{hypothesis} \mid \text{evidence})\\). These are different probabilities, and confusing them is one of the most common errors in probability reasoning.

Take a radar example. You usually have models of what kinds of returns each object class produces, written \\(P(\text{return} \mid \text{class})\\). What you actually want to know, after seeing a return, is \\(P(\text{class} \mid \text{return})\\). Bayes' rule turns the first into the second.

## The math

The basic rule is short:

\\[ P(Y \mid X) = \frac{P(X, Y)}{P(X)} \\]

Decoding:

- \\(P(Y \mid X)\\) is "probability of Y given X" (the thing we want).
- \\(P(X, Y)\\) is the **joint probability** that both X and Y happen.
- \\(P(X)\\) is the **marginal probability** of X (regardless of what Y does).

The intuition: out of all worlds where X happens (probability \\(P(X)\\)), what fraction also have Y happening (probability \\(P(X, Y)\\))? That's your conditional.

Bayes' rule rearranges this into a more useful form. Starting from \\(P(X, Y) = P(X \mid Y) P(Y) = P(Y \mid X) P(X)\\) (which says you can write the joint two ways), divide through to get:

\\[ P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)} \\]

Decoding (and now using H for "hypothesis" and E for "evidence" because that's how you'll see it written in textbooks):

- \\(P(H \mid E)\\): the **posterior**. What you believe about H after seeing the evidence.
- \\(P(H)\\): the **prior**. What you believed about H before seeing any evidence.
- \\(P(E \mid H)\\): the **likelihood**. How probable the evidence is, assuming the hypothesis is true. This is usually what your sensor model gives you.
- \\(P(E)\\): the **marginal likelihood** or evidence. Total probability of seeing this evidence, summed over every possible hypothesis: \\(P(E) = \sum_h P(E \mid H = h) P(H = h)\\). This term mostly exists to make the posterior sum to 1.

The shape that's worth memorizing:

\\[ \text{posterior} \propto \text{likelihood} \times \text{prior} \\]

The \\(\propto\\) symbol means "proportional to." The marginal \\(P(E)\\) is just whatever number you need to divide by at the end to make probabilities sum to 1, so most of the time you just compute the unnormalized version and normalize at the end.

## A warning: the base rate fallacy

The single most common error in conditional reasoning is to ignore the prior. People hear "this radar return has 90% likelihood given hostile object" and conclude "this object is 90% likely to be hostile." Those are different statements. If hostile objects are 1 in 10,000 in this orbital regime, even a strong likelihood barely moves the posterior.

Bayes' rule mechanically prevents this error if you actually use it. The prior \\(P(H)\\) shows up explicitly in the formula. Drop it and you've stopped doing Bayes.

## Code

Bayes' rule in PyTorch (or numpy) is just elementwise multiplication and a normalization:

```python
import torch

# Suppose we're classifying an observed RSO into one of three categories.
# Prior beliefs about what kind of object this likely is, before any sensor data.
prior = torch.tensor([0.60, 0.30, 0.10])  # active sat, debris, rocket body

# Sensor model: probability of observing the radar return we just got,
# given each possible object class.
likelihood = torch.tensor([0.70, 0.20, 0.40])

# Bayes' rule: posterior is proportional to likelihood * prior.
unnormalized = prior * likelihood
posterior = unnormalized / unnormalized.sum()

print(posterior)
# tensor([0.7568, 0.1081, 0.1351])
```

Notice that the posterior probability of "active satellite" jumped from 60% to about 76%, because the radar return was relatively likely under the active-satellite hypothesis. The posterior probability of "debris" dropped from 30% to about 11%, because the return was relatively unlikely under the debris hypothesis. That's Bayes doing exactly what your intuition would do: shifting probability mass toward the hypotheses that explain the evidence well, away from the ones that don't.

## Worked example: sequential updates

A nice property of Bayes is that you can apply it repeatedly. Each posterior becomes the next prior.

Same RSO, but now we get a second observation (say, a brightness measurement). Suppose the likelihood under each class for this new observation is:

```python
likelihood_2 = torch.tensor([0.30, 0.50, 0.40])  # active sat, debris, rocket body
```

(Brightness slightly favors debris over the alternatives.)

Update again:

```python
posterior_after_2 = posterior * likelihood_2
posterior_after_2 = posterior_after_2 / posterior_after_2.sum()
print(posterior_after_2)
# tensor([0.6293, 0.1499, 0.2208])
```

The active-satellite hypothesis is still in the lead, but its margin shrank because the second observation was less consistent with it. Two observations have nudged us from a 60/30/10 prior to roughly 63/15/22, and a third would nudge us further. This is what tracking algorithms in SSA do under the hood, just dressed up with continuous distributions and motion models.

A subtle and useful fact: this only works cleanly when the observations are conditionally independent given the hypothesis. That is, given that we know the object is (say) a rocket body, the radar return and the brightness measurement don't influence each other. If they're correlated (maybe both are affected by the same atmospheric condition), naive sequential Bayes will overcount the evidence. We'll come back to this when we get to belief states in POMDPs.

## Why this matters going forward

In partially observable games (poker, but also any realistic SSA scenario where you can't see your adversary's full state), an agent's "belief state" is exactly a Bayes posterior over the hidden parts of the world, updated each timestep as new observations arrive. CFR variants and PSRO will lean on Bayes implicitly when they reason about which information set you might actually be in. When you see those algorithms maintaining "beliefs" or "reach probabilities," what's happening underneath is the same `posterior = likelihood * prior / sum(...)` you just wrote.

## Quiz

{{#quiz 02-conditional-probability-bayes.toml}}
