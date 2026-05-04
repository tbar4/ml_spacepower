# Lesson 3: From Strategic Theory to Wargame Design

---

## The wargame design problem

Every wargame makes modeling choices — what to include, what to abstract away, which actors are present, what actions are available, how outcomes are determined. Those choices are not neutral. They encode assumptions about the strategic problem.

A wargame that models the space domain as two-player zero-sum implicitly assumes that the strategic problem is bilateral and that one side's gain is the other's loss. This is a reasonable model for certain kinetic counterspace scenarios (one state attacks another state's satellite constellation). It is a poor model for deterrence stability analysis (where the interesting question is what happens with three or more actors, each with different risk tolerances), for gray zone operations (where attribution uncertainty makes the game imperfect-information in a fundamental way), or for commercial-military interactions (where a commercial satellite operator's decisions create strategic effects without being a party to the conflict).

This lesson connects the strategic frameworks from Lessons 1 and 2 to the specific game-theoretic tools you will build in Modules 4 through 8. The goal is not to tell you which tool to use for which problem — the goal is to make you able to explain to a government customer why your wargame design encodes the strategic assumptions it does.

---

## Wargaming as analytical tool: the Mahnken/Marshall framework

Thomas Mahnken and Barry Watts, in *Net Assessment and Military Strategy*, describe wargaming as an analytical tool for testing strategic assumptions against adversary behavior — not a prediction engine, but a framework for exploring the implications of different strategic choices.

The net assessment tradition (associated with Andrew Marshall's Office of Net Assessment at the Pentagon) uses wargames to:

1. **Test strategic concepts**: Does the assumption that stealth aircraft negate Soviet air defenses hold when Soviet radar doctrine adapts?
2. **Discover unknown unknowns**: What aspects of the problem did the planning staff not anticipate? Games surface surprises in a low-cost environment.
3. **Stress-test doctrine**: Does the doctrine perform as expected when an adversary plays optimally, rather than playing as planners expect?
4. **Generate data for modeling**: Human wargames produce move sequences that can be analyzed statistically for patterns.

The computational approach adds a fourth capability: **exploring strategy spaces too large for human wargamers**. A human wargame might run for a week and generate 50 game histories. An AlphaZero-style system can generate millions of game histories in the same time. This changes what questions are tractable.

A 2023 academic study (*Exploratory Wargaming with Superhuman Tactician*, drawing on AlphaZero applied to a military air combat game) found that AlphaZero converged to Nash equilibrium strategies that human players did not discover — specifically, the computational agent found mixed strategies (deliberate randomization) that exploited human players' tendencies to follow fixed patterns. The strategic implication: in genuine zero-sum adversarial scenarios, Nash equilibrium is the right solution concept, and humans systematically fail to achieve it.

---

## Mapping strategic questions to game structures

Different strategic questions map to different game structures. Choosing the wrong structure does not just give you the wrong answer — it gives you a non-answer, because the structural assumptions of the game do not match the strategic reality being modeled.

### Zero-sum kinetic conflict → two-player perfect-information or minimax

When the strategic question is "given that both sides know each other's capabilities and are in direct military conflict, what is the optimal force allocation?" the game is approximately two-player zero-sum with near-complete information. Minimax and alpha-beta pruning (Module 4, Lesson 1) are the right tools.

Example: How should an attacker allocate kinetic ASAT strikes against a defender's satellite constellation to maximally degrade ISR capability? This is a combinatorial optimization problem with a clear zero-sum structure and no hidden information beyond noise.

### Orbital gray zone → imperfect-information extensive-form game

When the strategic question involves hidden intent (is that maneuver station-keeping or approach to a target?), incomplete information about adversary capabilities, or attribution uncertainty, the game has private information that belongs in the information set of one player but not the other. Information Set MCTS (Module 4, Lesson 5) and CFR (Module 5) are designed precisely for this structure.

The conjunction-masking capstone game is an imperfect-information game: the attacker knows whether the maneuver is offensive or defensive, but the defender only observes noisy orbital data. CFR finds the Nash equilibrium strategy for both players given this information structure — the attacker learns the optimal concealment pattern, the defender learns the optimal sensor allocation to detect it.

> The strategic insight embedded in CFR: at Nash equilibrium, the attacker does not always choose the most plausible cover story and the defender does not always surveil the most suspicious object. Mixed strategies (probabilistic action selection) prevent the adversary from exploiting a fixed pattern.

### Multi-actor deterrence → PSRO and population-level solution concepts

When the strategic question involves three or more actors with heterogeneous capabilities, doctrines, and objectives — the realistic space competition landscape — standard two-player Nash equilibrium is inadequate. A three-player game rarely has a pure Nash equilibrium; when it does, it is not necessarily the right solution concept (because side-payments and coalition formation become relevant).

Policy Space Response Oracles (Module 6, Lesson 3) addresses this by building a meta-game over a population of strategies and finding the Nash equilibrium of the meta-game. This is more tractable than solving the full three-plus-player game and captures the key dynamics: the right strategy against the United States is different from the right strategy against China, and both are different from the optimal strategy in an environment where both adversaries are present.

Alpha-rank (Module 6, Lesson 4) provides an alternative solution concept based on evolutionary dynamics — which strategies survive in a population of competing agents over time? Alpha-rank is more robust to the mixed equilibrium problem and produces a tractable ranking of strategies by their evolutionary fitness.

### Behavioral attribution under incomplete observation → particle filters and opponent modeling

When the strategic question is "what is the adversary's type (doctrine, objective, risk tolerance) given the sequence of observed actions?" — the opponent modeling problem — the right framework is Bayesian inference. The particle filter (Module 7, Lesson 2) maintains a belief state over adversary type and updates it as new observations arrive.

This maps to the actual intelligence problem in SDA: you observe a sequence of orbital maneuvers (or the absence of expected station-keeping maneuvers) and want to infer whether the satellite is:
- A commercial operator doing routine operations
- A military satellite doing routine station-keeping
- A military satellite conducting an intelligence-gathering approach to a target
- A satellite that has been disabled and is drifting

Each hypothesis implies a different future trajectory. Bayesian opponent modeling (Module 7, Lesson 4) formalizes this as type inference: you maintain a distribution over "adversary types" (where a type encodes a policy — a mapping from states to actions) and update the distribution as observations arrive.

---

## The information asymmetry at the center of orbital conflict

The most important structural feature of space conflict for wargame design is **asymmetric information about intent**.

In terrestrial military conflict, the presence of armed soldiers on your territory has an unambiguous meaning. In space, a maneuvering satellite near your critical satellite might be:
- An inspection satellite gathering intelligence (hostile, non-destructive)
- A satellite on a rendezvous trajectory for on-orbit servicing (benign)
- A kinetic kill vehicle positioning for an attack (hostile, destructive)
- A satellite that has suffered a navigation failure (benign, accidental)

The defender cannot distinguish between these until the action occurs — and some actions are irreversible on short timescales. This is not a failure of intelligence; it is a structural feature of the orbital environment. Orbital mechanics severely limits what you can infer about future intent from observed position and velocity.

This structural feature has a game-theoretic implication: the game is **not** merely imperfect information due to fog of war (which could be resolved with better ISR). It is fundamentally imperfect information because intent is not observable even in principle from kinematics alone. The right mathematical framework is not perfect-information game theory with noise, but imperfect-information game theory where intent belongs to a private information set.

CFR and IS-MCTS are correct here not because they are sophisticated tools but because they model the correct information structure.

---

## What computational wargaming adds over human wargaming

Human wargames have significant limitations as analytical tools:

**Small sample sizes**: Even a week-long wargame generates tens or hundreds of game histories, not thousands. Statistical conclusions from such samples are unreliable.

**Human cognitive biases**: Human players systematically deviate from Nash equilibrium. They overweight recent events, avoid mixed strategies because pure strategies "feel" more decisive, and are influenced by social dynamics within the wargame team.

**Discovery of Nash equilibrium strategies**: As the *Exploratory Wargaming* study showed, AlphaZero-style systems discover mixed equilibrium strategies that human players miss — specifically, the probabilistic patterns that exploit predictable adversary behavior.

**Scale**: A game with a large state space (many satellites, many sensors, long time horizons) is computationally tractable for distributed RL (Module 3, Lesson 8: IMPALA) but not for human wargamers.

**What human wargaming adds over computational**: Scenario validity ("is this game capturing the real strategic problem?"), doctrine elicitation ("what do planners actually believe about adversary capabilities?"), and norm-setting ("what are the unwritten rules about what actions are acceptable?"). The best wargaming programs combine both: human subject-matter experts define scenarios and action spaces, computational methods explore those spaces exhaustively.

Sandra Erwin's reporting on Slingshot Aerospace's wargame training program (SpaceNews, 2024) describes this hybrid approach: human operators define the strategic scenario and initial conditions, AI agents trained on orbital mechanics explore the strategy space, and human analysts interpret the resulting strategy profiles. The AI finds equilibria; the humans decide whether the game modeled the right problem.

---

## The LLM-in-the-loop connection

Module 8, Lesson 7 covers LLM-in-the-loop wargame adjudication: using a locally deployed language model as an umpire to adjudicate the realism and consequences of proposed actions, rather than having a hard-coded rule system or human umpires.

The strategic theory context for that lesson: LLM adjudication is appropriate when the action space is too large or too ambiguous for hard-coded rules (a satellite maneuver could be routine or hostile depending on context), and when the cost of a human umpire is too high for the scale of runs needed (thousands of game episodes for RL training). The LLM serves as an imperfect but tractable model of the "umpire's assessment" — encoding domain knowledge about what is plausible in the space environment.

The limitation: LLMs encode the training distribution, which reflects the documented operational doctrine of the past. They do not model adversarial adaptation (an adversary who learns that a specific action triggers a specific umpire ruling and exploits that pattern). For production wargaming, LLM adjudication is best combined with CFR or RL training so the agent learns to play against the umpire's model, including any exploitable patterns in that model.

---

## The capstone game design, explained

The Module 8 capstone is: a two-player extensive-form game where the attacker tries to mask a maneuver and the defender allocates sensors to detect it.

The strategic theory that justifies each design choice:

**Two-player**: Models the bilateral conflict between a single adversary satellite and a U.S./allied SSA capability. Simplified from the realistic multi-actor environment for tractability — a deliberate scoping choice, not an assumption that the real environment is bilateral.

**Extensive-form (sequential moves)**: Orbital operations are sequential — maneuvers happen in sequence, sensor observations arrive in sequence, the attacker decides when and how to maneuver based on what the defender is observing. Perfect simultaneous action would miss the temporal dynamics.

**Attacker's private information (intent)**: The attacker knows the maneuver is offensive; the defender does not. This is not modeled as noise — it is a fundamental information asymmetry encoded in the information set structure. This is why CFR is the appropriate solver.

**Sensor allocation as defender action**: The defender's resource constraint (limited sensor dwell time) is the binding constraint in real SDA operations. The action space — how to allocate scarce observation time across multiple tracked objects — reflects the real operational problem.

**Conjunction as cover**: The attacker uses a conjunction event (a close approach that is plausible given orbital mechanics) to mask the maneuver. This is not invented — co-orbital inspection satellites have been documented approaching adversary assets under conditions that create ambiguity about whether the approach is intentional or a forced coincidence of orbital geometry.

Every design choice encodes a strategic assumption. The contribution of this module is making those assumptions visible — so you can defend them to a customer, revise them when a new threat scenario emerges, and extend the game to new strategic questions without starting from scratch.

---

## What you need to be able to do

After this lesson, you should be able to:

- Explain why wargame design choices are not neutral and how they encode strategic assumptions
- Map a strategic question to the appropriate game structure: zero-sum kinetic conflict → minimax; gray zone with hidden intent → IS-MCTS or CFR; multi-actor deterrence → PSRO; behavioral attribution → particle filters and opponent modeling
- Explain why CFR produces the right solution concept for the conjunction-masking game specifically (private information about intent, mixed strategy equilibria prevent exploitation of fixed patterns)
- Describe what computational wargaming adds over human wargaming and what human wargaming adds over computational
- Explain each design choice in the Module 8 capstone game in terms of the strategic assumption it encodes

---

{{#quiz 05-strategy-to-wargame-design.toml}}
