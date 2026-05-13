# Module SP Project: Wargame Design Brief and Deterrence-by-Detection Assessment


<!-- toc -->

## What you are building

This module has no code and no math. The project is analytical: you will produce a structured design brief for a computational wargame addressing a specific adversarial SSA scenario, then evaluate whether the ML deterrence-by-detection thesis actually holds for that scenario. The output is a written document — the kind you could put in front of a government customer, a thesis committee, or a conference reviewer.

The point of writing this down is to force precision. It is easy to say "game-theoretic reasoning applies to orbital conflict." It is harder to say: "for this scenario, the correct game structure is an imperfect-information sequential game, solved with CFR, where the defender's information set encodes these specific observables, and the Nash equilibrium tells us the attacker will mix these specific maneuver intensities." The project asks for the harder version.

---

## The scenario

Choose one of the following. Pick the one most relevant to your thesis direction.

**Option A: Luch co-orbital positioning near nuclear C2.**
A co-orbital inspection platform (modeled on Russia's Luch) maneuvers to within 50 km of a U.S. nuclear early warning satellite in GEO. The maneuvering platform is registered as a communications relay satellite. The approach geometry is consistent with either routine orbital adjustment or deliberate coercive positioning.

**Option B: Chinese dual-use satellite pre-positioning (Taiwan Phase 0).**
A constellation of satellites registered under civilian operators executes a series of orbital adjustments over three weeks, resulting in coverage geometry optimized for the Taiwan Strait theater. Each individual maneuver is within normal station-keeping variance. The pattern across the constellation is not.

**Option C: Conjunction-masking approach to a commercial SDA asset.**
An adversary satellite maneuvers to a position whose conjunction geometry with a piece of tracked debris makes it ambiguous whether a subsequent close approach to a commercial SDA satellite is deliberate RPO or incidental. This is the Module 8 capstone game as a real-world scenario.

The structure of the project is the same regardless of which option you choose. Option C is the most computationally tractable. Options A and B have higher strategic stakes and connect more directly to the deterrence-by-detection thesis at the nuclear and coalition levels.

---

## Part 1: Escalation ladder placement

Place your chosen scenario on the 8-rung space escalation ladder from Lesson 5. For each of the following, write 2-4 sentences:

**1a. Current rung.** Which rung does the scenario currently occupy? Cite specific features of the scenario that locate it on this rung rather than adjacent ones. If the scenario spans multiple rungs depending on how the attacker's intent is interpreted, say so explicitly.

**1b. Rung transition triggers.** What specific action would move the scenario up to the next rung? What would move it back down? Be concrete: not "further escalation" but "if the co-orbital vehicle maneuvers to within 5 km and activates its RF payload, that transitions from Rung 1 to Rung 4 because..."

**1c. Firebreak analysis.** Does your scenario sit near one of the two major firebreaks (Rung 2 to 3, or Rung 5 to 6)? If so, what determines whether the firebreak holds? If the scenario involves nuclear C2 assets, identify the specific compression of the escalation ladder and what decision time that implies.

**1d. Harrison ISR blinding problem.** Does ISR blinding make the scenario more or less stable? Apply Harrison's finding directly: if the defender cannot see the scenario developing, does that increase or decrease the probability of miscalculated response?

---

## Part 2: Wargame design brief

Design the computational wargame for your scenario. Every design choice below has a strategic assumption behind it. State both.

**2a. Players and interests.**
Who are the players? (Not just "attacker" and "defender" — be specific about what each player wants and why those interests conflict.) Is this two-player zero-sum, two-player non-zero-sum, or multi-actor? Justify your answer.

> Example of what is expected: "Two players, zero-sum. The Adversary wants to establish co-orbital proximity without triggering a response; the Defender wants to detect and characterize any approach. Zero-sum is justified because the Adversary's primary goal (undetected positioning) is achieved exactly to the degree the Defender's primary goal (detection) is not. Commercial satellite operators are not modeled as separate players because their decisions are not strategically reactive to the adversary's actions — they are part of the environment, not the game."

**2b. Information structure.**
What does each player observe? What do they not observe? Map this to the game-theoretic vocabulary from Lessons 1 and 6: perfect information, imperfect information, incomplete information. For imperfect-information games, enumerate the information sets for each player.

> The information structure is not cosmetic. "The Adversary observes the defender's sensor allocation" vs. "the Adversary does not observe the defender's sensor allocation" produces qualitatively different equilibria. State which assumption you are making and why it matches the actual operational situation.

**2c. Action space.**
What actions are available to each player? Use the module's vocabulary: not "the attacker can attack" but specific, enumerated options with physical meaning. What actions are you excluding and why?

**2d. Chance nodes and stochastic outcomes.**
What outcomes are determined by noise or Nature rather than player choice? What probability distributions govern them? Where do those probabilities come from (physics, calibrated estimates, assumptions)?

**2e. Payoff structure.**
What is each player's payoff in each terminal outcome? If zero-sum, state the Adversary's payoff; the Defender's is its negation. Map each payoff value to a specific operational consequence.

**2f. Solution concept.**
Which solution concept is appropriate for your game? Choose from: minimax/backward induction (perfect information), Nash equilibrium via CFR (imperfect-information two-player zero-sum), PSRO (multi-actor or iterative best response), alpha-rank (evolutionary stability in a population), Bayesian opponent modeling (type inference). Justify the choice based on the game's structure, not on which tool you know how to build.

---

## Part 3: Computational tool audit

Map your game design to the specific tools built in Modules 4 through 9. For each module below, answer the question:

**Module 4 (Search and Planning):** Does your game require a tree search component? If the game is multi-step rather than single-shot, which variant of MCTS applies — standard MCTS (perfect information) or IS-MCTS (imperfect information)? What depth does the game tree realistically require?

**Module 5 (CFR and equilibrium computation):** Is CFR the right solver for your game? If yes: how many information sets does your game have? Is vanilla CFR tractable or do you need MCCFR or deep CFR? What does the resulting Nash equilibrium actually say about attacker strategy?

**Module 6 (MARL):** If your scenario has more than two strategic actors, or if the "game" is really a sequence of episodes where both sides adapt, does PSRO or alpha-rank apply? What population of strategies would the meta-game include?

**Module 7 (Partial observability):** What is the defender's belief state? What observables update it? Which belief representation is appropriate — discrete Bayesian update, Kalman filter, or particle filter? What is the adversary type distribution the defender maintains?

**Module 9 (Applied SDA ML):** What features from the TLE history would the Module 9 pipeline extract to generate the defender's observables? What is the realistic detection latency given TLE update cadence? Is that latency short enough to matter operationally for your scenario?

---

## Part 4: Deterrence-by-detection assessment

Apply the ML deterrence thesis from Lesson 5 to your specific scenario. Work through each step of the argument:

**4a. Does detection change the equilibrium?**
In your CFR or PSRO game, what happens to the attacker's equilibrium strategy when the defender's detection capability improves? Does better detection cause the attacker to shift toward lower-intensity maneuvers, toward higher-intensity maneuvers (accepting detection risk for higher payoff), or toward a different timing strategy? If you ran the CFR solver from Module 5 (or the capstone from Module 8) against your game's payoff table, what direction would the Nash equilibrium shift?

You do not need to run the actual computation. Reason qualitatively from the payoff structure: if detection probability increases, which of the attacker's actions becomes less attractive and which becomes more attractive?

**4b. Required detection latency.**
At what timescale must detection occur for the deterrent effect to operate? If detection takes 48 hours but the threatening maneuver completes in 6 hours, detection provides forensic value but no deterrent value. State the operational tempo of your scenario and whether the TLE-based pipeline's latency (1-4 day update cadence for most objects) is sufficient, or whether sub-orbital-period cadence data from commercial optical/radar sensors is required.

**4c. Attribution chain.**
Detection is not attribution. Walk through the attribution chain for your scenario: (1) anomaly detected; (2) maneuver characterized; (3) intent inferred; (4) actor attributed; (5) response authorized. Where does the ML pipeline contribute? Where does it hand off to human intelligence analysis? Where does the chain break down?

**4d. Adversary adaptation.**
If the adversary knows the detection pipeline exists and understands its features, how does their optimal strategy change? This is the adversarial ML problem: the attacker designs maneuvers to minimize detection probability while achieving the positioning objective. Does the Nash equilibrium strategy (from your answer to 4a) already account for this? If not, what would the adapted attacker strategy look like?

**4e. Honest limitations for your scenario.**
List at least three specific limitations of the deterrence-by-detection argument as applied to your chosen scenario. Use the Lesson 5 list as a starting point but make them specific: not "adversaries can adapt" but "in the Luch scenario, the adversary can achieve the same coercive positioning with a sequence of small station-keeping maneuvers each of which is individually sub-threshold, defeating the Mahalanobis scoring in the Module 9 pipeline because..."

---

## Part 5: Thesis position statement

Write a 3-4 paragraph position statement making the ML deterrence argument for your specific scenario. This is the kind of text that would appear in a thesis proposal, a conference paper abstract, or a research brief for a government customer.

The statement must:
- Open with the specific strategic problem (not "space is important" but the specific scenario you analyzed)
- State the deterrence mechanism precisely (what changes in adversary behavior when ML detection capability is present)
- Cite at least three specific computational tools from the curriculum as the technical foundation of the claim
- Acknowledge the primary counterargument and explain why the argument holds despite it
- Close with the research gap or next step — what would need to be demonstrated to operationalize this claim

**Length:** 3-4 paragraphs, 300-500 words. No more. Brevity forces precision.

**Audience:** A program manager at AFRL or a Space Force acquisitions office who has technical fluency but is not a researcher. They want to know: what does this capability do, how does it work in principle, and why should they care.

---

## Part 6: Reflect

After completing Parts 1-5, answer these questions briefly (one paragraph each):

**6a.** Which design choice in Part 2 were you most uncertain about? What would you need to know about the actual operational scenario to resolve that uncertainty?

**6b.** Your wargame design encodes strategic assumptions. If one of those assumptions is wrong — if the scenario is not zero-sum, or if the attacker has more information than you assumed, or if the payoffs are miscalibrated — how does the Nash equilibrium you described in Part 4a change? Is the deterrence argument fragile to that assumption, or robust?

**6c.** The overview of this module says: "Every wargame is a theory in disguise." Looking at your design brief, what theory of space conflict is your wargame a disguise for? State it in one sentence.

---

## What you have built

- A precise placement of a real-world SSA scenario on the space escalation ladder with firebreak analysis
- A complete wargame design brief with every assumption stated explicitly
- A mapping from the game's structure to the specific modules' computational tools
- A deterrence-by-detection assessment that distinguishes what the ML pipeline can and cannot contribute
- A thesis position statement suitable for a research proposal or government brief

The discipline of the exercise is the point. The gap between "game-theoretic reasoning is relevant to orbital conflict" and "this specific game structure, solved with CFR, produces this Nash equilibrium strategy, which implies this change in attacker behavior when detection capability improves" is where the thesis argument either holds up or falls apart. Writing it out is how you find out which.
