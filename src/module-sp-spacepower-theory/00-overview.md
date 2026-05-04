# Module SP: Spacepower Theory and Strategic Context

---

## Why this module exists

Every wargame is a theory in disguise. When you define the action space of your SSA game — what moves are available to an attacker trying to mask a maneuver — you are implicitly taking a position on what coercive options exist in the space domain. When you choose imperfect information over perfect information for your game structure, you are making a claim about the epistemic situation of real orbital operations. When you train your CFR solver against a specific reward function, you are encoding a theory of what actors value.

This module makes those theories explicit.

Spacepower theory is a young field. The first systematic treatments appeared in the late 1990s and early 2000s. The U.S. Space Force only adopted a formal doctrine document (the Space Capstone Publication) in 2020. China's publicly available military space doctrine still requires reading PLA publications in translation. The field is contested, rapidly evolving, and directly relevant to what you are building.

You do not need to agree with every theorist covered here. What you need is enough fluency to:

- Recognize when a wargame design choice encodes a contestable strategic assumption
- Explain to a government customer why your game structure reflects the actual strategic problem they face
- Know which strategic questions CFR answers well, which ones PSRO answers better, and which ones neither answers at all

This module has no code. It has no math. It does have quotes you should recognize, frameworks you should be able to apply, and questions that do not have clean answers.

---

## The core debate this module maps

There are two foundational schools of thought in spacepower theory, and almost every specific debate in the field traces back to them.

**The sanctuary school** holds that space should be treated as a domain separate from military competition — a place for reconnaissance, communications, and scientific cooperation that functions best when all parties implicitly agree not to weaponize it. This position dominated U.S. policy through the Cold War and into the 1990s. It produced the Outer Space Treaty (1967) and the norm against debris-generating ASAT tests that still shapes international discussions.

**The high ground school** holds that space is simply the next domain — no different in principle from sea or air — and that military advantage in space translates directly to advantage in terrestrial conflicts. Everett Dolman is the clearest contemporary voice for this position. His formulation is blunt: "Who controls low-Earth orbit controls near-Earth space. Who controls near-Earth space dominates Terra."

The U.S. Space Force's 2020 Space Capstone Publication effectively ends the sanctuary debate for U.S. government customers: "Space is a warfighting domain." Your government customers operate in a post-sanctuary world. The debate matters for understanding why certain wargame framings resonate with DoD customers and why others do not.

---

## Lessons in this module

**Lesson 1: Foundations of Spacepower Theory**

The theoretical vocabulary you need before any strategic conversation. Covers the spacepower definition (Lutes), the sanctuary vs. high ground debate (Dolman), the USSF Space Capstone Publication's seven spacepower disciplines, Ziarnick's General Theory of Space Power, Chinese spacepower theory (Carlson's geography/legitimacy/economy framework), and the Outer Space Treaty — what it prohibits and what it does not.

**Lesson 2: Counterspace Operations and the New RMA**

The operational level of space competition. Covers the counterspace taxonomy (kinetic/non-kinetic, reversible/irreversible, attributable/non-attributable), deterrence stability and the stability-instability paradox, Krepinevich's domain expansion and MTR/RMA distinction, PLA doctrine (Science of Military Strategy 2013), the current counterspace landscape, and commercial space as military infrastructure — what the Viasat hack and Starlink in Ukraine revealed about the strategic role of commercial satellites.

**Lesson 3: Chinese Spacepower Theory and Gray Zone Competition**

Chinese doctrine in depth. Covers PLA informationized warfare doctrine (information dominance before kinetics), Qiao Liang's Unrestricted Warfare framework (all means, all domains, boundary dissolution), the Three Warfares (legal, psychological, public opinion) with space-specific examples including near-space legal warfare, the civilian-military blur in Chinese space operations, gray zone wargame findings (China's civilian spacecraft positioning strategy that produced no actionable U.S. response), and Hal Brands on coalition dynamics and the structure of the New Cold War in space.

**Lesson 4: Escalation Dynamics, Crisis Stability, and the ML Deterrence Framework**

The thesis core. Covers why space escalation is structurally different from terrestrial escalation (compressed rungs, attribution delay, absence of norms), the Russian concept of calibrated escalation as cost imposition, Brands and Cooper's six deterrence dilemmas, Todd Harrison's counterintuitive finding that ISR blinding increases escalation risk, the crisis communication problem (Campbell), and Kessler Syndrome as a partial structural deterrent. Culminates in the ML deterrence-by-detection thesis: how SDA ML capabilities contribute to strategic stability by reducing the orbital ambiguity on which gray zone operations depend, and the honest limitations of that argument.

**Lesson 5: From Strategic Theory to Wargame Design**

The bridge from theory to implementation. Covers how strategic questions map to specific game structures, why information asymmetry in orbital operations implies imperfect-information game theory (IS-MCTS and CFR), why multi-actor deterrence dynamics require population-level solution concepts (PSRO and alpha-rank), and why behavioral inference maps to opponent modeling and particle filters. Uses the exploratory wargaming literature to show what computational approaches reveal that human wargames miss. Provides explicit design rationale for every choice in the Module 8 capstone game.

---

## How this module connects to the rest of the curriculum

These connections are the point of the module.

**Module 4 (Search and Planning) — IS-MCTS for fog-of-war SSA games**: The reason you use Information Set MCTS rather than standard minimax is that orbital operations involve fundamental epistemic asymmetry — you rarely know whether an adversary's maneuver is station-keeping or repositioning for an approach. Lesson 5 here provides the strategic motivation for that choice.

**Module 5 (Game Theory) — CFR for the conjunction maneuver game**: CFR finds Nash equilibria in extensive-form games. Lesson 1 establishes why Nash equilibrium is the right solution concept for two-player zero-sum adversarial space interactions, and Lesson 5 shows when it is not (multi-actor scenarios requiring PSRO). Lesson 4 poses the deeper question: does the attacker's equilibrium strategy change when the defender has ML-based detection? CFR is the tool that answers it formally.

**Module 6 (MARL) — PSRO for adversarial constellation games**: The strategic rationale for population-based training is that space competition involves multiple actors with heterogeneous capabilities and doctrines (U.S., allied, Russian, Chinese, commercial). Lesson 3 (Chinese gray zone) establishes why coalition dynamics are part of the game. PSRO builds a population of strategies and finds meta-game equilibria — the right tool for a multi-actor strategic landscape.

**Module 7 (Partial Observability) — Particle filters and opponent modeling**: The fundamental epistemic problem in orbital operations is behavioral attribution. Lesson 4 frames this as the binding constraint on deterrence-by-detection: attribution is necessary for response. The opponent modeling lesson in Module 7 is the computational formalization of the attribution problem.

**Module 8 (Capstone) — The SSA conjunction-masking game design**: Every design choice in the capstone game — the attacker's action space, the defender's sensor allocation options, the reward structure — traces back to a strategic assumption that this module makes explicit. Lesson 4's deterrence-by-detection thesis is what the capstone is computationally testing.

**Module 9 (Applied SDA ML) — Maneuver detection as deterrence infrastructure**: The LSTM maneuver detection pipeline is the empirical foundation of the Lesson 4 deterrence argument. It is not just a commercially valuable product — it is the kind of behavioral transparency capability that the deterrence-by-detection framework requires to function.

---

## A note on sources

The highlights and sources underlying this module include:

- Everett Dolman, *Astropolitik: Classical Geopolitics in the Space Age* (2002)
- Charles Lutes et al., *Toward a Theory of Spacepower* (2011)
- U.S. Space Force, *Spacepower: Doctrine for Space Forces* (Space Capstone Publication, 2020)
- Brent Ziarnick, *Developing National Power in Space* (2015)
- Joshua Carlson, *Spacepower Ascendant* (Chinese spacepower analysis)
- Andrew Krepinevich, *The Origins of Victory* (2023)
- Thomas Mahnken and Barry Watts (eds.), *Net Assessment and Military Strategy* (2018)
- Qiao Liang and Wang Xiangsui, *Unrestricted Warfare* (1999, translated)
- Hal Brands, *Lessons From the New Cold War* (2024); *The Eurasian Century* (2023)
- Hal Brands and Zack Cooper, "Dilemmas of Deterrence" (CSIS, 2024)
- Anya Fink, "Russian Strategy for Escalation Management: Evolution of Key Concepts"
- Todd Harrison, "Battle Networks and the Future Force" (CSIS)
- Kurt M. Campbell, "The U.S.-China Crisis Waiting to Happen"
- Alan T. Dugger, "Space as a Gray Zone: The Future of Orbital Warfare" (2024)
- John Jordan Klein, *Fight for the Final Frontier* (2019)
- Todd Pennington and Emmy Kanarowski, "China's 'Near Space' Legal Warfare"
- Christian Brose, *The Kill Chain* (2020)
- Secure World Foundation, *Global Counterspace Capabilities: An Open Source Assessment* (annual)
- PLA Academy of Military Science, *Science of Military Strategy* (2013, translated)
- Clayton Swope, "The Future of Military Power Is Space Power"
- Sandra Erwin, various SPACENEWS articles on USSF doctrine and commercial SDA
- "The Ghost in the Orbit: How Hybrid Surveillance Reshapes Risks"

These sources span U.S., allied, and Chinese perspectives. Where sources conflict, the conflict is noted — a contested strategic landscape is a more accurate picture than a tidy synthesis.
